"""Shared, self-contained Phase-2 span tagger used by the six method modules.

AI-assisted reference implementation: disclose its use under the course rules.
No official test labels or outside pretrained weights are used here.
"""
from __future__ import annotations

import math
import random
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F


def _encode(tokenizer, text):
    result = tokenizer.backend.encode(text, add_special_tokens=False)
    return tuple(result.ids), tuple((int(a), int(b)) for a, b in result.offsets)


@lru_cache(maxsize=4096)
def _cached_encode(tokenizer_key, text, tokenizer):
    return _encode(tokenizer, text)


def encode(tokenizer, text):
    # The backend identity keeps independently trained Phase-1 BPEs separate.
    return _cached_encode(id(tokenizer.backend), text, tokenizer)


def _tag_id(label_index, kind):
    return 1 + 4 * label_index + {'B': 0, 'I': 1, 'L': 2, 'U': 3}[kind]


def _to_tags(offsets, spans, labels):
    tags = [0] * len(offsets)
    for start, end, name in spans:
        if name not in labels:
            continue
        indices = [i for i, (a, b) in enumerate(offsets) if b > start and a < end and a < b]
        if not indices:
            continue
        label_index = labels.index(name)
        if len(indices) == 1:
            tags[indices[0]] = _tag_id(label_index, 'U')
        else:
            tags[indices[0]] = _tag_id(label_index, 'B')
            for i in indices[1:-1]:
                tags[i] = _tag_id(label_index, 'I')
            tags[indices[-1]] = _tag_id(label_index, 'L')
    return tags


def _boundary_targets(offsets, spans, max_shift):
    starts, ends = [-100] * len(offsets), [-100] * len(offsets)
    for start, end, _name in spans:
        indices = [i for i, (a, b) in enumerate(offsets) if b > start and a < end and a < b]
        if not indices:
            continue
        first, last = indices[0], indices[-1]
        start_delta = start - offsets[first][0]
        end_delta = offsets[last][1] - end
        if 0 <= start_delta <= max_shift:
            starts[first] = start_delta
        if 0 <= end_delta <= max_shift:
            ends[last] = end_delta
    return starts, ends


def _windows(length, size, stride):
    if length <= size:
        return [(0, length)]
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:
        starts.append(length - size)
    return [(start, min(start + size, length)) for start in starts]


def _pad(sequences, pad_value, dtype=torch.long):
    width = max(len(seq) for seq in sequences)
    out = torch.full((len(sequences), width), pad_value, dtype=dtype)
    for row, values in enumerate(sequences):
        out[row, :len(values)] = torch.tensor(values, dtype=dtype)
    return out


def collate(rows, tokenizer, labels, config, teacher_tokenizer=None):
    size, stride = int(config['max_tokens']), int(config['window_stride'])
    ids_rows, masks, tag_rows = [], [], []
    start_rows, end_rows = [], []
    teacher_ids_rows, teacher_maps = [], []
    for row in rows:
        text = row['text']
        ids, offsets = encode(tokenizer, text)
        if not ids:
            ids, offsets = (tokenizer.unk_token_id,), ((0, 0),)
        windows = _windows(len(ids), size, stride)
        spans = row.get('spans', [])
        entity_windows = [item for item in windows if any(
            offsets[item[0]][0] <= a and offsets[item[1] - 1][1] >= b
            for a, b, _label in spans)]
        chosen = random.choice(entity_windows if entity_windows and random.random() < .7 else windows)
        lo, hi = chosen
        segment_offsets = offsets[lo:hi]
        # Ignore gold entities cut by a window; overlap makes them available in other windows.
        contained = [(a, b, name) for a, b, name in spans if
                     segment_offsets[0][0] <= a and b <= segment_offsets[-1][1]]
        ids_rows.append(ids[lo:hi])
        masks.append([1] * (hi - lo))
        tag_rows.append(_to_tags(segment_offsets, contained, labels))
        start_targets, end_targets = _boundary_targets(
            segment_offsets, contained, int(config.get('max_char_shift', 16)))
        start_rows.append(start_targets)
        end_rows.append(end_targets)
        if teacher_tokenizer is not None:
            char_lo, char_hi = segment_offsets[0][0], segment_offsets[-1][1]
            teacher_ids, teacher_offsets = encode(teacher_tokenizer, text[char_lo:char_hi])
            if not teacher_ids:
                teacher_ids, teacher_offsets = (teacher_tokenizer.unk_token_id,), ((0, 0),)
            teacher_offsets = [(a + char_lo, b + char_lo) for a, b in teacher_offsets]
            teacher_ids_rows.append(teacher_ids[:int(config['teacher_max_tokens'])])
            teacher_offsets = teacher_offsets[:int(config['teacher_max_tokens'])]
            teacher_maps.append([min(range(len(teacher_offsets)), key=lambda j:
                                 abs(teacher_offsets[j][0] - a)) for a, _b in segment_offsets])
    result = {'input_ids': _pad(ids_rows, tokenizer.pad_token_id),
              'attention_mask': _pad(masks, 0, torch.bool),
              'labels': _pad(tag_rows, -100),
              'start_boundaries': _pad(start_rows, -100),
              'end_boundaries': _pad(end_rows, -100)}
    if teacher_tokenizer is not None:
        result.update(teacher_input_ids=_pad(teacher_ids_rows, teacher_tokenizer.pad_token_id),
                      teacher_attention_mask=_pad([[1] * len(x) for x in teacher_ids_rows], 0, torch.bool),
                      teacher_index=_pad(teacher_maps, 0))
    return result


class AdditiveLoRA(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.b(self.a(x)) * self.scale


class MultiplicativeLoRMA(nn.Module):
    """LoRMA+: y=(I+alpha/r B A) W0 x + frozen original bias."""
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.a = nn.Linear(base.out_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        self.scale = alpha / rank

    def forward(self, x):
        y = self.base(x)
        return y + self.b(self.a(F.linear(x, self.base.weight))) * self.scale


class ResidualAdapter(nn.Module):
    def __init__(self, block, width, bottleneck):
        super().__init__()
        self.block = block
        self.down = nn.Linear(width, bottleneck)
        self.up = nn.Linear(bottleneck, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, *args, **kwargs):
        out = self.block(*args, **kwargs)
        return out + self.up(F.gelu(self.down(out)))


def adapt_base(base, base_id, method, config):
    if method == 'full_finetune':
        return
    for param in base.parameters():
        param.requires_grad_(False)
    if method == 'partial_finetune':
        if base_id != '02':
            raise ValueError('Partial encoder method requires Phase-1 model 02')
        for block in base.encoder_blocks[-int(config['unfrozen_layers']):]:
            for param in block.parameters():
                param.requires_grad_(True)
        for param in base.encoder_final_norm.parameters():
            param.requires_grad_(True)
    elif method in {'lora', 'lorma'}:
        rank, alpha = int(config['rank']), float(config['adapter_alpha'])
        projection_names = {'query', 'value'} if method == 'lora' else {'q_proj', 'v_proj'}
        replaced = 0
        for block in base.blocks:
            attention = block.attention
            for name in projection_names:
                original = getattr(attention, name, None)
                if isinstance(original, nn.Linear):
                    replacement = AdditiveLoRA(original, rank, alpha) if method == 'lora' else MultiplicativeLoRMA(original, rank, alpha)
                    setattr(attention, name, replacement)
                    replaced += 1
        if not replaced:
            raise RuntimeError('No attention projections were adapted')
    elif method == 'adapter':
        for i, block in enumerate(base.blocks):
            base.blocks[i] = ResidualAdapter(block, base.config.hidden_size, int(config['adapter_bottleneck']))
    elif method == 'knowledge_distillation':
        # The smaller Phase-1 student is frozen; its task head learns from gold and teacher.
        pass
    else:
        raise ValueError(f'Unknown adaptation method: {method}')


class EntityTagger(nn.Module):
    def __init__(self, base, base_id, labels, method, config, teacher=None):
        super().__init__()
        self.base, self.base_id = base, base_id
        self.labels = tuple(labels)
        self.method = method
        self.config = dict(config)
        adapt_base(base, base_id, method, config)
        width = base.hidden_size if hasattr(base, 'hidden_size') else (
            base.config.hidden_size if hasattr(base.config, 'hidden_size') else int(base.config['hidden_size']))
        self.head = nn.Linear(width, 1 + 4 * len(labels))
        self.start_boundary_head = nn.Linear(width, int(config.get('max_char_shift', 16)) + 1)
        self.end_boundary_head = nn.Linear(width, int(config.get('max_char_shift', 16)) + 1)
        self.dropout = nn.Dropout(float(config.get('head_dropout', .1)))
        self.teacher = teacher
        if teacher is not None:
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()
        return self

    def features(self, input_ids, attention_mask):
        base = self.base
        length = input_ids.shape[1]
        if self.base_id == '01':
            positions = torch.arange(length, device=input_ids.device)
            x = base.embedding_dropout(base.token_embedding(input_ids) + base.position_embedding(positions)[None])
            for block in base.blocks:
                x = block(x)
            return base.final_norm(x)
        if self.base_id == '02':
            return base.encode(input_ids, attention_mask)[0]
        if self.base_id == '03':
            visibility = base._prefix_hybrid_visibility(attention_mask, attention_mask.sum(1))
            x = base.embedding_dropout(base.token_embedding(input_ids) * math.sqrt(base.hidden_size))
            cosine, sine = base.rotary(length, x.dtype)
            for block in base.blocks:
                x = block(x, visibility, cosine, sine)
            return base.final_norm(x)
        if self.base_id == '04':
            x = base.embedding(input_ids)
            for block in base.blocks:
                x = block(x, attention_mask)
            return base.final_norm(x)
        raise ValueError(self.base_id)

    def forward(self, input_ids, attention_mask, labels=None,
                start_boundaries=None, end_boundaries=None,
                teacher_input_ids=None, teacher_attention_mask=None, teacher_index=None):
        hidden = self.dropout(self.features(input_ids, attention_mask))
        logits = self.head(hidden)
        start_logits = self.start_boundary_head(hidden)
        end_logits = self.end_boundary_head(hidden)
        if labels is None:
            return logits, start_logits, end_logits
        weights = logits.new_ones(logits.shape[-1])
        weights[0] = float(self.config.get('outside_class_weight', .3))
        supervised = F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(),
                                     weight=weights.float(), ignore_index=-100)
        if start_boundaries is not None and (start_boundaries != -100).any():
            supervised = supervised + float(self.config.get('boundary_loss_weight', .2)) * F.cross_entropy(
                start_logits.flatten(0, 1).float(), start_boundaries.flatten(), ignore_index=-100)
        if end_boundaries is not None and (end_boundaries != -100).any():
            supervised = supervised + float(self.config.get('boundary_loss_weight', .2)) * F.cross_entropy(
                end_logits.flatten(0, 1).float(), end_boundaries.flatten(), ignore_index=-100)
        if self.teacher is None:
            return {'loss': supervised}
        if teacher_input_ids is None or teacher_index is None:
            raise ValueError('Distillation batch lacks teacher inputs/alignment')
        with torch.no_grad():
            teacher_logits = self.teacher(teacher_input_ids, teacher_attention_mask)[0].float()
        teacher_logits = teacher_logits.gather(1, teacher_index.unsqueeze(-1).expand(-1, -1, logits.shape[-1]))
        temperature = float(self.config['distill_temperature'])
        per_token = F.kl_div(F.log_softmax(logits.float() / temperature, -1),
                             F.softmax(teacher_logits / temperature, -1), reduction='none').sum(-1)
        valid = labels.ne(-100)
        distill = (per_token * valid).sum() / valid.sum().clamp_min(1) * temperature ** 2
        mix = float(self.config['distill_weight'])
        return {'loss': (1 - mix) * supervised + mix * distill}


def predict(model, text, tokenizer, labels, config):
    ids, offsets = encode(tokenizer, text)
    if not ids:
        return []
    size, stride = int(config['max_tokens']), int(config['window_stride'])
    windows = _windows(len(ids), size, stride)
    device = next(model.parameters()).device
    batch_size = int(config.get('inference_window_batch_size', 4))
    candidates = {}
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        cursor = 0
        while cursor < len(windows):
            group = windows[cursor:cursor + batch_size]
            batch = [ids[a:b] for a, b in group]
            input_ids = _pad(batch, tokenizer.pad_token_id).to(device)
            masks = _pad([[1] * len(row) for row in batch], 0, torch.bool).to(device)
            try:
                logits, start_logits, end_logits = model(input_ids, masks)
            except RuntimeError as exc:
                if 'out of memory' not in str(exc).lower() or batch_size == 1 or device.type != 'cuda':
                    raise
                del input_ids, masks
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                continue
            probabilities = logits.float().softmax(-1)
            kinds = probabilities.argmax(-1).cpu().tolist()
            scores = probabilities.max(-1).values.cpu().tolist()
            start_deltas = start_logits.argmax(-1).cpu().tolist()
            end_deltas = end_logits.argmax(-1).cpu().tolist()
            for row, (start_token, end_token) in enumerate(group):
                predictions = kinds[row][:end_token - start_token]
                confidence = scores[row][:end_token - start_token]
                active_start, active_label, active_scores = None, None, []

                def flush(last):
                    if active_start is None:
                        return
                    raw_a = offsets[start_token + active_start][0]
                    raw_b = offsets[start_token + last][1]
                    a = raw_a + start_deltas[row][active_start]
                    b = raw_b - end_deltas[row][last]
                    if a >= b:
                        a, b = raw_a, raw_b
                    a = max(0, min(a, len(text)))
                    b = max(0, min(b, len(text)))
                    while a < b and text[a].isspace():
                        a += 1
                    while b > a and text[b - 1].isspace():
                        b -= 1
                    if a < b:
                        key = (a, b, labels[active_label])
                        candidates[key] = max(candidates.get(key, 0), sum(active_scores) / len(active_scores))

                for i, tag in enumerate(predictions):
                    if tag == 0:
                        if active_start is not None:
                            flush(i - 1)
                        active_start, active_label, active_scores = None, None, []
                        continue
                    label, kind = divmod(tag - 1, 4)
                    if kind == 3:  # U
                        if active_start is not None:
                            flush(i - 1)
                        active_start, active_label, active_scores = i, label, [confidence[i]]
                        flush(i)
                        active_start, active_label, active_scores = None, None, []
                    elif kind == 0 or active_start is None or label != active_label:
                        if active_start is not None:
                            flush(i - 1)
                        active_start, active_label, active_scores = i, label, [confidence[i]]
                    else:
                        active_scores.append(confidence[i])
                        if kind == 2:  # L
                            flush(i)
                            active_start, active_label, active_scores = None, None, []
                if active_start is not None:
                    flush(len(predictions) - 1)
            cursor += len(group)
    if was_training:
        model.train()
    ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
    if config.get('return_candidate_scores', False):
        return [[a, b, name, float(score)] for (a, b, name), score in ranked]
    accepted = []
    by_label = config.get('confidence_threshold_by_label') or {}
    for (a, b, name), score in ranked:
        if score < float(by_label.get(name, config.get('confidence_threshold', .45))):
            continue
        if any(a < other_b and other_a < b for other_a, other_b, _ in accepted):
            continue
        accepted.append((a, b, name))
    return [list(item) for item in sorted(accepted)]
