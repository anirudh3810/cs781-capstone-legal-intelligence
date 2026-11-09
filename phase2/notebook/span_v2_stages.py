"""Embedded stages for the resumed, stronger Phase-2 span experiment."""

# %% 6 - verified v1 warm start and larger boundary model
V1_CONFIG = read_json(V1_ROOT / 'experiment_config.json')
V1_HASH = hashlib.sha256(json.dumps(V1_CONFIG, sort_keys=True).encode()).hexdigest()
if (V1_CONFIG['base_checkpoint_sha256'] != base_sha or
        V1_CONFIG['model_source_sha256'] != model_sha or
        V1_CONFIG['bpe_source_sha256'] != bpe_sha or
        V1_CONFIG['tokenizer_fingerprint'] != tokenizer_meta['fingerprint'] or
        V1_CONFIG['data_hash'] != data_hash):
    raise RuntimeError('The saved v1 experiment does not match this Phase-1 base or official data')
V1_OOF = read_json(V1_ROOT / 'oof_report.json')
V1_FINAL = read_json(V1_ROOT / 'final' / 'summary.json')
if V1_FINAL.get('status') != 'complete':
    raise RuntimeError('Complete the v1 final model in the same Kaggle session first')
V1_TEST_CANDIDATES = V1_ROOT / 'final' / 'test_candidates.partial.json'
if not V1_TEST_CANDIDATES.is_file():
    raise RuntimeError('The v1 raw test-candidate cache is missing')


def v1_selected(fold_index):
    folder = V1_ROOT / f'fold{fold_index}'
    best = read_json(folder / 'best.json')
    if best.get('config_hash') != V1_HASH or best.get('fold') != fold_index:
        raise RuntimeError(f'Fold {fold_index}: v1 provenance mismatch')
    adapter = folder / best['adapter_file']
    candidate_file = folder / best['candidates_file']
    if (file_sha(adapter) != best['adapter_sha256'] or
            file_sha(candidate_file) != best['candidates_sha256']):
        raise RuntimeError(f'Fold {fold_index}: v1 selected artifact hash changed')
    payload = read_json(candidate_file)
    if payload.get('config_hash') != V1_HASH or set(payload['candidates']) != set(FOLDS[fold_index]):
        raise RuntimeError(f'Fold {fold_index}: v1 OOF candidate IDs changed')
    return adapter, payload['candidates'], best


V1_FOLDS = [v1_selected(fold_index) for fold_index in range(3)]
V1_OOF_CANDIDATES = {}
for _, candidates, _ in V1_FOLDS:
    V1_OOF_CANDIDATES.update(candidates)
if set(V1_OOF_CANDIDATES) != set(texts):
    raise RuntimeError('The v1 OOF candidate map is incomplete')
V1_TUNED_THRESHOLDS = read_json(V1_ROOT / 'thresholds.json')['thresholds']
reproduced = strict_macro(V1_OOF_CANDIDATES, texts, V1_TUNED_THRESHOLDS)['macro_f1']
if abs(reproduced - V1_OOF['calibrated_macro_f1']) > 1e-8:
    raise RuntimeError('The saved v1 calibrated OOF F1 did not reproduce')
print('Verified v1 calibrated OOF F1:', f'{reproduced:.6f}')


class EnhancedSpanExtractor(SpanExtractor):
    def __init__(self, base):
        super().__init__(base)
        width = base.hidden_size
        self.pair_refine = nn.Sequential(
            nn.Linear(5 * width + 16, CONFIG['refine_hidden']),
            nn.GELU(), nn.Dropout(0.18),
            nn.Linear(CONFIG['refine_hidden'], len(LABELS)))
        nn.init.zeros_(self.pair_refine[-1].weight)
        nn.init.zeros_(self.pair_refine[-1].bias)
        for block in base.blocks[-CONFIG['unfreeze_top_blocks']:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)

    def score_pairs(self, hidden, pair_index):
        width = hidden.shape[-1]
        start_index = pair_index[..., 0]
        end_index = pair_index[..., 1]
        def take(index):
            return hidden.gather(1, index.unsqueeze(-1).expand(-1, -1, width))
        start, end = take(start_index), take(end_index)
        left = take((start_index - 1).clamp_min(0))
        right = take((end_index + 1).clamp_max(hidden.shape[1] - 1))
        distance = (end_index - start_index).clamp(min=0, max=CONFIG['max_span_tokens'])
        width_feature = self.width_embedding(distance)
        original = torch.cat((start, end, start * end, width_feature), dim=-1)
        boundary = torch.cat((start, end, left, right, start * end, width_feature), dim=-1)
        return self.pair_head(original) + self.pair_refine(boundary)


def enhanced_model(warm_adapter):
    record = torch.load(BASE_CHECKPOINT, map_location='cpu', weights_only=True)
    if record.get('model_source_sha256') != model_sha:
        raise RuntimeError('Phase-1 base changed before enhanced model construction')
    base = model_module.build_model(record['model_config'])
    base.load_state_dict(record['model_state'], strict=True)
    del record
    model = EnhancedSpanExtractor(base)
    saved = torch.load(warm_adapter, map_location='cpu', weights_only=True)
    if saved.get('config_hash') != V1_HASH:
        raise RuntimeError('Warm adapter is not from the verified v1 experiment')
    current = model.state_dict()
    for name, value in saved['trainable_state'].items():
        if name not in current or current[name].shape != value.shape:
            raise RuntimeError(f'Warm adapter state mismatch: {name}')
        current[name] = value
    model.load_state_dict(current, strict=True)
    print('Enhanced trainable parameters:',
          f'{sum(p.numel() for p in model.parameters() if p.requires_grad):,}',
          'of', f'{sum(p.numel() for p in model.parameters()):,}')
    return model


def v2_best(folder, fold_index):
    path = folder / 'best.json'
    if not path.is_file():
        return None
    best = read_json(path)
    if best.get('config_hash') != CONFIG_HASH or best.get('fold') != fold_index:
        raise RuntimeError(f'Fold {fold_index}: v2 selected checkpoint provenance changed')
    adapter = folder / best['adapter_file']
    candidate_file = folder / best['candidates_file']
    if (file_sha(adapter) != best['adapter_sha256'] or
            file_sha(candidate_file) != best['candidates_sha256']):
        raise RuntimeError(f'Fold {fold_index}: v2 selected files changed')
    payload = read_json(candidate_file)
    if payload.get('config_hash') != CONFIG_HASH or set(payload['candidates']) != set(FOLDS[fold_index]):
        raise RuntimeError(f'Fold {fold_index}: v2 candidate provenance changed')
    actual = strict_macro(payload['candidates'], FOLDS[fold_index])['macro_f1']
    if abs(actual - best['default_macro_f1']) > 1e-8:
        raise RuntimeError(f'Fold {fold_index}: v2 selected score did not reproduce')
    return best


# %% 7 - resumable fold continuation on two GPUs
def parameter_groups(model):
    groups = {'encoder': [], 'lora': [], 'head': [], 'refine': []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith('pair_refine.'):
            key = 'refine'
        elif not name.startswith('base.'):
            key = 'head'
        elif '.a.' in name or '.b.' in name:
            key = 'lora'
        else:
            key = 'encoder'
        groups[key].append(parameter)
    if any(not group for group in groups.values()):
        raise RuntimeError('One of the enhanced model parameter groups is empty')
    lrs = {'encoder': 8e-6, 'lora': 4e-5, 'head': 1e-4, 'refine': 2e-4}
    return [{'params': group, 'lr': lrs[name], 'initial_lr': lrs[name]}
            for name, group in groups.items()]


def train_v2_fold(fold_index):
    folder = ROOT / f'fold{fold_index}'
    folder.mkdir(parents=True, exist_ok=True)
    summary_path = folder / 'summary.json'
    latest_path = folder / 'checkpoint_latest.pt'
    if summary_path.is_file():
        previous = read_json(summary_path)
        if previous.get('config_hash') != CONFIG_HASH:
            raise RuntimeError(f'Fold {fold_index}: incompatible v2 resume summary')
        if previous['status'] == 'complete':
            v2_best(folder, fold_index)
            print(f'Fold {fold_index}: verified complete')
            return previous
        if previous['status'] == 'paused' and not latest_path.is_file():
            raise RuntimeError(f'Fold {fold_index}: paused without a checkpoint')
    warm_path, old_candidates, old_best = V1_FOLDS[fold_index]
    heldout = FOLDS[fold_index]
    train_ids = set(texts) - set(heldout)
    rows = make_windows(train_ids, training=True)
    model = enhanced_model(warm_path).to('cuda:0')
    parallel = nn.DataParallel(model, device_ids=[0, 1], output_device=0)
    optimizer = torch.optim.AdamW(parameter_groups(model), weight_decay=0.01, foreach=False)
    steps_per_epoch = math.ceil(len(rows) / CONFIG['batch_size'])
    total_steps = MAX_EPOCHS * steps_per_epoch
    warmup = max(1, int(total_steps * 0.05))
    def lr_factor(step):
        if step < warmup:
            return max(0.05, (step + 1) / warmup)
        return max(0.20, (total_steps - step) / max(1, total_steps - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    weights = label_weights(train_ids).to('cuda:0')
    best = v2_best(folder, fold_index)
    old_score = float(old_best['default_macro_f1'])
    best_score = best['default_macro_f1'] if best else old_score
    history, epoch_start, stale = [], 0, 0
    if latest_path.is_file():
        saved = torch.load(latest_path, map_location='cpu', weights_only=False)
        if (saved.get('config_hash') != CONFIG_HASH or saved.get('fold') != fold_index or
                saved.get('warm_sha256') != file_sha(warm_path)):
            raise RuntimeError(f'Fold {fold_index}: optimizer checkpoint provenance changed')
        load_trainable_state(model, saved['trainable_state'])
        optimizer.load_state_dict(saved['optimizer_state'])
        scheduler.load_state_dict(saved['scheduler_state'])
        torch.set_rng_state(saved['torch_rng_state'])
        torch.cuda.set_rng_state_all(saved['cuda_rng_states'])
        random.setstate(saved['python_rng_state'])
        history, epoch_start, stale = saved['history'], saved['epoch'], saved['stale']
        print(f'Fold {fold_index}: resuming after continuation epoch {epoch_start}')
    paused = False
    try:
        for epoch in range(epoch_start + 1, MAX_EPOCHS + 1):
            if time.time() - RUN_STARTED > SESSION_GUARD_SECONDS:
                paused = True
                print('Time guard reached. Run All again in this session to resume.')
                break
            order = list(range(len(rows)))
            random.Random(SEED + 20_000 + fold_index * 1000 + epoch).shuffle(order)
            model.train()
            losses = []
            for begin in range(0, len(order), CONFIG['batch_size']):
                group = [rows[i] for i in order[begin:begin + CONFIG['batch_size']]]
                batch = {name: value.to('cuda:0', non_blocking=True)
                         for name, value in collate_windows(group, epoch + 100).items()}
                optimizer.zero_grad(set_to_none=True)
                output = (parallel(batch['input_ids'], batch['attention_mask'], batch['pair_index'])
                          if len(group) > 1 else
                          model(batch['input_ids'], batch['attention_mask'], batch['pair_index']))
                loss = task_loss(output, batch, weights)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Fold {fold_index} epoch {epoch}: nonfinite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                losses.append(float(loss.detach().cpu()))
            candidates = predict_documents(model, parallel, heldout, texts)
            measured = strict_macro(candidates, heldout)
            score = measured['macro_f1']
            entry = {'epoch': epoch, 'train_loss_mean': sum(losses) / len(losses),
                     'heldout_default_macro_f1': score,
                     'raw_candidate_recall_macro': sum(
                         r['exact_candidate_recall'] for r in measured['per_label'].values()) / len(LABELS),
                     'elapsed_seconds': round(time.time() - RUN_STARTED, 1)}
            history.append(entry)
            if score > best_score + 0.002:
                adapter_name = f'selected_epoch{epoch:02d}.pt'
                candidate_name = f'candidates_epoch{epoch:02d}.json'
                atomic_torch(folder / adapter_name,
                             {'config_hash': CONFIG_HASH, 'fold': fold_index, 'epoch': epoch,
                              'trainable_state': trainable_state(model), 'labels': LABELS})
                atomic_json(folder / candidate_name,
                            {'config_hash': CONFIG_HASH, 'fold': fold_index,
                             'epoch': epoch, 'candidates': candidates})
                proposed = {'config_hash': CONFIG_HASH, 'fold': fold_index,
                            'epoch': epoch, 'default_macro_f1': score,
                            'adapter_file': adapter_name,
                            'adapter_sha256': file_sha(folder / adapter_name),
                            'candidates_file': candidate_name,
                            'candidates_sha256': file_sha(folder / candidate_name)}
                old_selected = best
                atomic_json(folder / 'best.json', proposed)
                best = v2_best(folder, fold_index)
                if old_selected is not None:
                    (folder / old_selected['adapter_file']).unlink(missing_ok=True)
                    (folder / old_selected['candidates_file']).unlink(missing_ok=True)
                best_score = score
                stale = 0
            else:
                stale += 1
            atomic_json(folder / 'history.json', history)
            if free_gib() < MIN_FREE_GIB:
                raise RuntimeError('Disk reserve reached before checkpoint write')
            atomic_torch(latest_path, {
                'config_hash': CONFIG_HASH, 'fold': fold_index,
                'warm_sha256': file_sha(warm_path), 'epoch': epoch, 'stale': stale,
                'history': history, 'trainable_state': trainable_state(model),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_states': torch.cuda.get_rng_state_all(),
                'python_rng_state': random.getstate()})
            print(f'Fold {fold_index} continuation {epoch}: loss {entry["train_loss_mean"]:.4f}, '
                  f'F1 {score:.4f}, best {best_score:.4f}, disk {free_gib():.2f} GiB', flush=True)
            if epoch >= MIN_EPOCHS and stale >= PATIENCE:
                print(f'Fold {fold_index}: early stopping after {stale} non-improving epochs')
                break
        if paused and not latest_path.is_file():
            raise RuntimeError('Time guard reached before a resumable fold checkpoint was made')
        best = v2_best(folder, fold_index)
        result = {'status': 'paused' if paused else 'complete', 'fold': fold_index,
                  'config_hash': CONFIG_HASH, 'warm_epoch': old_best['epoch'],
                  'warm_default_macro_f1': old_score,
                  'selected_source': 'v2' if best else 'v1',
                  'selected_epoch': best['epoch'] if best else 0,
                  'selected_default_macro_f1': best_score,
                  'history': history}
        atomic_json(summary_path, result)
        if not paused:
            latest_path.unlink(missing_ok=True)
        return result
    finally:
        del model, parallel, optimizer, scheduler
        torch.cuda.empty_cache()


FOLD_RESULTS = {}
for fold_index in range(3):
    if time.time() - RUN_STARTED > SESSION_GUARD_SECONDS:
        print('Time guard reached before another fold; Run All again to resume.')
        break
    FOLD_RESULTS[fold_index] = train_v2_fold(fold_index)
    if FOLD_RESULTS[fold_index]['status'] != 'complete':
        break


# %% 8 - out-of-fold comparison and document-aware candidate reranker
import pickle
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def selected_v2_candidates(fold_index):
    best = v2_best(ROOT / f'fold{fold_index}', fold_index)
    if best is None:
        return V1_FOLDS[fold_index][1]
    return read_json(ROOT / f'fold{fold_index}' / best['candidates_file'])['candidates']


def combine_candidate_scores(v1_rows, v2_rows):
    merged = {}
    for a, b, label, score in v1_rows:
        merged[(int(a), int(b), label)] = [float(score), 0.0]
    for a, b, label, score in v2_rows:
        key = (int(a), int(b), label)
        merged.setdefault(key, [0.0, 0.0])[1] = float(score)
    return merged


def document_features(doc_id, text, merged):
    n = max(1, len(text))
    strong_frequency = Counter()
    early_label = {}
    start_max, end_max, exact_labels = {}, {}, {}
    for (a, b, label), (s1, s2) in merged.items():
        peak = max(s1, s2)
        key = ' '.join(text[a:b].casefold().split())
        if peak >= 0.35 and key:
            strong_frequency[key] += 1
        if a / n < 0.25 and label in ('APP', 'RESP', 'A.COUNSEL', 'R.COUNSEL'):
            early_label[(key, label)] = max(early_label.get((key, label), 0.0), peak)
        start_max[a] = max(start_max.get(a, 0.0), peak)
        end_max[b] = max(end_max.get(b, 0.0), peak)
        exact_labels.setdefault((a, b), {})[label] = peak
    rows, features = [], []
    for (a, b, label), (s1, s2) in merged.items():
        phrase = text[a:b]
        normalized = ' '.join(phrase.casefold().split())
        before = text[max(0, a-90):a].casefold()
        after = text[b:min(len(text), b+60)].casefold()
        nearby = before + ' ' + after
        position = a / n
        length = b - a
        peak = max(s1, s2)
        other = max((score for lab, score in exact_labels[(a, b)].items()
                     if lab != label), default=0.0)
        role = label in ('APP', 'RESP', 'A.COUNSEL', 'R.COUNSEL')
        opposite = {'APP': 'RESP', 'RESP': 'APP',
                    'A.COUNSEL': 'R.COUNSEL', 'R.COUNSEL': 'A.COUNSEL'}.get(label)
        values = [
            s1, s2, peak, min(s1, s2), abs(s1-s2),
            math.log(max(1e-5, s1) / max(1e-5, 1-s1)),
            math.log(max(1e-5, s2) / max(1e-5, 1-s2)),
            position, float(position < .1), float(position < .25),
            math.log1p(length), min(length, 150) / 150,
            min(phrase.count(' ') + 1, 20) / 20,
            float(phrase[:1].isupper()),
            sum(c.isdigit() for c in phrase) / max(1, length),
            float(any(c in phrase for c in '.,()[]:-/')),
            math.log1p(min(strong_frequency[normalized], 100)),
            early_label.get((normalized, label), 0.0) if role else 0.0,
            early_label.get((normalized, opposite), 0.0) if opposite else 0.0,
            start_max[a] - peak, end_max[b] - peak, other,
            float('appellant' in nearby or 'petitioner' in nearby),
            float('respondent' in nearby or 'defendant' in nearby),
            float('advocate' in nearby or 'counsel' in nearby),
            float('witness' in nearby or 'deposed' in nearby),
            float('judge' in nearby or 'justice' in nearby),
            float(' v. ' in nearby or ' vs ' in nearby or 'versus' in nearby),
            float('section' in nearby or 'article' in nearby),
            float('court' in nearby),
            float('reported' in nearby or 'citation' in nearby),
            float(phrase.isupper()),
        ]
        values.extend(float(label == name) for name in LABELS)
        rows.append((a, b, label, s1, s2))
        features.append(values)
    if not features:
        return rows, np.empty((0, 32 + len(LABELS)), dtype=np.float32)
    return rows, np.asarray(features, dtype=np.float32)


def build_feature_table(doc_ids, v1_map, v2_map, document_texts, with_gold):
    features, rows_by_doc, ranges, targets = [], {}, {}, []
    cursor = 0
    for doc_id in sorted(doc_ids):
        merged = combine_candidate_scores(v1_map[doc_id], v2_map[doc_id])
        rows, matrix = document_features(doc_id, document_texts[doc_id], merged)
        rows_by_doc[doc_id] = rows
        features.append(matrix)
        ranges[doc_id] = (cursor, cursor + len(rows))
        cursor += len(rows)
        if with_gold:
            exact = {(int(row[0]), int(row[1]), row[2]) for row in gold[doc_id]}
            targets.extend(int((a, b, label) in exact) for a, b, label, _, _ in rows)
    if not features or any(matrix.shape[1] != features[0].shape[1] for matrix in features):
        raise RuntimeError('Candidate feature width changed between documents')
    X = np.concatenate(features, axis=0)
    y = np.asarray(targets, dtype=np.uint8) if with_gold else None
    if not np.isfinite(X).all() or (with_gold and len(y) != len(X)):
        raise RuntimeError('Document candidate features are invalid')
    return X, y, rows_by_doc, ranges


def train_reranker(X, y, ranges, training_ids):
    indices = []
    rng = np.random.default_rng(SEED + len(training_ids))
    for doc_id in training_ids:
        lo, hi = ranges[doc_id]
        if lo == hi:
            continue
        selection = np.arange(lo, hi)
        score = np.maximum(X[lo:hi, 0], X[lo:hi, 1])
        keep = (y[lo:hi] == 1) | (score >= 0.22) | (rng.random(hi-lo) < 0.045)
        indices.append(selection[keep])
    indices = np.concatenate(indices)
    if len(indices) == 0 or y[indices].sum() < 100:
        raise RuntimeError('Reranker has too few positive training candidates')
    weights = np.where(y[indices] == 1, 5.0, 1.0).astype(np.float32)
    model = HistGradientBoostingClassifier(
        learning_rate=0.075, max_iter=120, max_leaf_nodes=31,
        min_samples_leaf=35, l2_regularization=2.0,
        early_stopping=False, random_state=SEED)
    model.fit(X[indices], y[indices], sample_weight=weights)
    print('Reranker training:', len(indices), 'candidates,',
          int(y[indices].sum()), 'positive exact spans', flush=True)
    return model


def rerank_documents(model, X, rows_by_doc, ranges, doc_ids):
    output = {}
    for doc_id in doc_ids:
        lo, hi = ranges[doc_id]
        if lo == hi:
            output[doc_id] = []
            continue
        probabilities = model.predict_proba(X[lo:hi])[:, 1]
        rows = [[a, b, label, float(probabilities[i])]
                for i, (a, b, label, _, _) in enumerate(rows_by_doc[doc_id])
                if probabilities[i] >= .025]
        rows.sort(key=lambda row: -row[3])
        output[doc_id] = rows
    return output


def fast_macro(candidate_map, thresholds):
    predicted = decode(candidate_map, thresholds)
    values = []
    for name in LABELS:
        p = {(doc_id, a, b) for doc_id, rows in predicted.items()
             for a, b, label in rows if label == name}
        g = {(doc_id, a, b) for doc_id in candidate_map
             for a, b, label in gold[doc_id] if label == name}
        tp = len(p & g)
        values.append(2 * tp / (len(p) + len(g)) if p or g else 0.0)
    return sum(values) / len(LABELS)


def calibrate(candidate_map, starting_threshold=.45):
    thresholds = {label: starting_threshold for label in LABELS}
    score = fast_macro(candidate_map, thresholds)
    grid = (.05, .10, .15, .20, .30, .40, .50, .60, .70, .80)
    for label in LABELS:
        best_t, best_score = thresholds[label], score
        for trial in grid:
            proposal = dict(thresholds, **{label: trial})
            trial_score = fast_macro(candidate_map, proposal)
            if trial_score > best_score + 1e-8:
                best_t, best_score = trial, trial_score
        thresholds[label], score = best_t, best_score
    return thresholds, score


STAGE8 = None
if len(FOLD_RESULTS) == 3 and all(r['status'] == 'complete' for r in FOLD_RESULTS.values()):
    v2_map = {}
    for fold_index in range(3):
        v2_map.update(selected_v2_candidates(fold_index))
    if set(v2_map) != set(texts):
        raise RuntimeError('The v2 OOF map does not contain all 70 documents')
    v2_default = strict_macro(v2_map, texts)['macro_f1']
    v2_thresholds, v2_calibrated = calibrate(v2_map)
    print('Selected v2 fold OOF: default', f'{v2_default:.6f}',
          'calibrated', f'{v2_calibrated:.6f}', flush=True)
    X, y, rows_by_doc, ranges = build_feature_table(
        texts, V1_OOF_CANDIDATES, v2_map, texts, with_gold=True)
    print('Document-aware candidate table:', X.shape, 'exact positives:', int(y.sum()), flush=True)
    reranked_oof = {}
    for fold_index in range(3):
        validation_ids = FOLDS[fold_index]
        training_ids = sorted(set(texts) - set(validation_ids))
        meta = train_reranker(X, y, ranges, training_ids)
        reranked_oof.update(rerank_documents(meta, X, rows_by_doc, ranges, validation_ids))
        del meta
        print('Cross-fit reranker fold', fold_index, 'complete', flush=True)
    rerank_default = fast_macro(reranked_oof, {})
    rerank_thresholds, rerank_calibrated = calibrate(reranked_oof)
    baseline = float(V1_OOF['calibrated_macro_f1'])
    fold_checks = []
    for fold_index in range(3):
        ids = FOLDS[fold_index]
        old_subset = {doc_id: V1_OOF_CANDIDATES[doc_id] for doc_id in ids}
        new_subset = {doc_id: v2_map[doc_id] for doc_id in ids}
        rerank_subset = {doc_id: reranked_oof[doc_id] for doc_id in ids}
        fold_checks.append({
            'fold': fold_index,
            'v1': fast_macro(old_subset, V1_TUNED_THRESHOLDS),
            'v2': fast_macro(new_subset, v2_thresholds),
            'reranked': fast_macro(rerank_subset, rerank_thresholds)})
    alternatives = {'v1': baseline, 'v2': v2_calibrated, 'reranked': rerank_calibrated}
    method = max(alternatives, key=alternatives.get)
    STAGE8 = {'status': 'complete', 'config_hash': CONFIG_HASH,
              'baseline_v1_calibrated_oof': baseline,
              'v2_default_oof': v2_default, 'v2_calibrated_oof': v2_calibrated,
              'reranker_default_oof': rerank_default,
              'reranker_calibrated_oof': rerank_calibrated,
              'v2_thresholds': v2_thresholds,
              'reranker_thresholds': rerank_thresholds,
              'fold_checks': fold_checks,
              'chosen_method': method, 'chosen_oof': alternatives[method],
              'per_label_v2': strict_macro(v2_map, texts, v2_thresholds)['per_label'],
              'per_label_reranked': strict_macro(reranked_oof, texts, rerank_thresholds)['per_label'],
              'crossfit_note': 'Each meta model excludes its held-out documents; threshold selection still uses all OOF labels.'}
    atomic_json(ROOT / 'model_comparison.json', STAGE8)
    print('V1:', f'{baseline:.6f}', 'v2:', f'{v2_calibrated:.6f}',
          'reranked:', f'{rerank_calibrated:.6f}', 'chosen:', method, flush=True)
    del X, y, rows_by_doc, ranges, reranked_oof, v2_map
else:
    print('Three folds are not yet complete. Run All again in this session to resume.')


# %% 9 - final continuation and test candidate inference
def final_continuation_epochs():
    chosen = [int(FOLD_RESULTS[i]['selected_epoch']) for i in range(3)
              if FOLD_RESULTS[i]['selected_source'] == 'v2']
    if not chosen:
        return 0
    chosen.sort()
    return max(2, min(MAX_EPOCHS, chosen[len(chosen)//2]))


def train_v2_final(epochs):
    final_dir = ROOT / 'final'
    final_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = final_dir / 'selected_adapter.pt'
    latest_path = final_dir / 'checkpoint_latest.pt'
    summary_path = final_dir / 'summary.json'
    if summary_path.is_file():
        previous = read_json(summary_path)
        if previous.get('config_hash') != CONFIG_HASH or previous.get('epochs') != epochs:
            raise RuntimeError('Incompatible final-model resume summary')
        if previous.get('status') == 'complete':
            if file_sha(adapter_path) != previous['adapter_sha256']:
                raise RuntimeError('Final v2 adapter changed')
            print('Final v2 model verified complete')
            return previous
    if file_sha(V1_ROOT / 'final' / 'selected_adapter.pt') != V1_FINAL['adapter_sha256']:
        raise RuntimeError('The v1 final warm-start adapter changed')
    rows = make_windows(texts, training=True)
    model = enhanced_model(V1_ROOT / 'final' / 'selected_adapter.pt').to('cuda:0')
    parallel = nn.DataParallel(model, device_ids=[0, 1], output_device=0)
    optimizer = torch.optim.AdamW(parameter_groups(model), weight_decay=0.01, foreach=False)
    steps_per_epoch = math.ceil(len(rows) / CONFIG['batch_size'])
    total_steps = max(1, epochs * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0.20, (total_steps-step)/total_steps))
    weights = label_weights(texts).to('cuda:0')
    history, epoch_start = [], 0
    if latest_path.is_file():
        saved = torch.load(latest_path, map_location='cpu', weights_only=False)
        if (saved.get('config_hash') != CONFIG_HASH or saved.get('epochs') != epochs or
                saved.get('v1_final_sha256') != V1_FINAL['adapter_sha256']):
            raise RuntimeError('Final optimizer checkpoint provenance mismatch')
        load_trainable_state(model, saved['trainable_state'])
        optimizer.load_state_dict(saved['optimizer_state'])
        scheduler.load_state_dict(saved['scheduler_state'])
        torch.set_rng_state(saved['torch_rng_state'])
        torch.cuda.set_rng_state_all(saved['cuda_rng_states'])
        random.setstate(saved['python_rng_state'])
        history, epoch_start = saved['history'], saved['epoch']
        print('Resuming final v2 model after epoch', epoch_start)
    paused = False
    try:
        for epoch in range(epoch_start + 1, epochs + 1):
            if time.time() - RUN_STARTED > SESSION_GUARD_SECONDS:
                paused = True
                print('Time guard reached during final fit; Run All again to resume.')
                break
            order = list(range(len(rows)))
            random.Random(SEED + 90_000 + epoch).shuffle(order)
            model.train()
            losses = []
            for begin in range(0, len(order), CONFIG['batch_size']):
                group = [rows[i] for i in order[begin:begin + CONFIG['batch_size']]]
                batch = {name: value.to('cuda:0', non_blocking=True)
                         for name, value in collate_windows(group, epoch + 300).items()}
                optimizer.zero_grad(set_to_none=True)
                output = (parallel(batch['input_ids'], batch['attention_mask'], batch['pair_index'])
                          if len(group) > 1 else
                          model(batch['input_ids'], batch['attention_mask'], batch['pair_index']))
                loss = task_loss(output, batch, weights)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Final model epoch {epoch}: nonfinite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                losses.append(float(loss.detach().cpu()))
            history.append({'epoch': epoch, 'loss': sum(losses)/len(losses),
                            'elapsed_seconds': round(time.time()-RUN_STARTED, 1)})
            atomic_json(final_dir / 'history.json', history)
            if free_gib() < MIN_FREE_GIB:
                raise RuntimeError('Disk reserve reached before final optimizer checkpoint')
            atomic_torch(latest_path, {
                'config_hash': CONFIG_HASH, 'epochs': epochs,
                'v1_final_sha256': V1_FINAL['adapter_sha256'], 'epoch': epoch,
                'history': history, 'trainable_state': trainable_state(model),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_states': torch.cuda.get_rng_state_all(),
                'python_rng_state': random.getstate()})
            print(f'Final continuation epoch {epoch}/{epochs}: loss {history[-1]["loss"]:.4f}, '
                  f'disk {free_gib():.2f} GiB', flush=True)
        if not paused:
            atomic_torch(adapter_path, {'config_hash': CONFIG_HASH,
                                        'epochs': epochs,
                                        'trainable_state': trainable_state(model),
                                        'labels': LABELS})
        result = {'status': 'paused' if paused else 'complete',
                  'config_hash': CONFIG_HASH, 'epochs': epochs,
                  'history': history,
                  'adapter_sha256': file_sha(adapter_path) if not paused else None}
        atomic_json(summary_path, result)
        if not paused:
            latest_path.unlink(missing_ok=True)
        return result
    finally:
        del model, parallel, optimizer, scheduler
        torch.cuda.empty_cache()


def v2_test_candidates(final_summary):
    final_dir = ROOT / 'final'
    partial = final_dir / 'test_candidates.partial.json'
    provenance_path = final_dir / 'test_candidate_provenance.json'
    provenance = {'config_hash': CONFIG_HASH,
                  'adapter_sha256': final_summary['adapter_sha256'],
                  'test_sha256': file_sha(SOURCE_DIR / 'fold3-text.json')}
    if provenance_path.is_file() and read_json(provenance_path) != provenance:
        raise RuntimeError('Stale v2 test-candidate cache provenance')
    atomic_json(provenance_path, provenance)
    existing = read_json(partial) if partial.is_file() else {}
    if not set(existing) <= set(test_texts):
        raise RuntimeError('Unknown test IDs in v2 inference cache')
    model = enhanced_model(V1_ROOT / 'final' / 'selected_adapter.pt').to('cuda:0')
    saved = torch.load(final_dir / 'selected_adapter.pt', map_location='cpu', weights_only=True)
    if saved.get('config_hash') != CONFIG_HASH:
        raise RuntimeError('Final v2 adapter configuration mismatch')
    load_trainable_state(model, saved['trainable_state'])
    parallel = nn.DataParallel(model, device_ids=[0, 1], output_device=0)
    try:
        def progress(predictions):
            if len(predictions) % 5 == 0:
                atomic_json(partial, predictions)
                print(f'V2 test inference: {len(predictions)}/{len(test_texts)}', flush=True)
        raw = predict_documents(model, parallel, test_texts, test_texts,
                                existing=existing, on_document=progress)
    finally:
        del model, parallel
        torch.cuda.empty_cache()
    if set(raw) != set(test_texts):
        raise RuntimeError('V2 test inference did not finish all 35 documents')
    atomic_json(partial, raw)
    return raw


FINAL_RESULT = None
RAW_TEST = None
FINAL_GATE = max(MIN_FINAL_CV_F1, V1_OOF['calibrated_macro_f1'] + 0.005)
if STAGE8 is not None:
    improve = STAGE8['chosen_oof'] >= FINAL_GATE
    if not improve:
        print(f'New local score did not pass the {FINAL_GATE:.4f} final-fit gate; '
              'preserving the 0.4584 submission.')
    elif time.time() - RUN_STARTED > SESSION_GUARD_SECONDS:
        print('Time guard reached before final fit. Run All again to continue.')
    else:
        selected_epochs = final_continuation_epochs()
        if selected_epochs:
            FINAL_RESULT = train_v2_final(selected_epochs)
            if FINAL_RESULT['status'] == 'complete':
                RAW_TEST = v2_test_candidates(FINAL_RESULT)
        else:
            FINAL_RESULT = {'status': 'complete', 'epochs': 0,
                            'note': 'No v2 fold beat its v1 checkpoint; using verified v1 final candidates.'}
            RAW_TEST = read_json(V1_TEST_CANDIDATES)
        if RAW_TEST is not None and set(RAW_TEST) != set(test_texts):
            raise RuntimeError('Final raw candidate map does not match the 35 test documents')


# %% 10 - verified submission and compact diagnostic report
PACKAGE = None
if RAW_TEST is not None and STAGE8 is not None:
    method = STAGE8['chosen_method']
    if method == 'v2':
        predictions = decode(RAW_TEST, STAGE8['v2_thresholds'])
    elif method == 'reranked':
        v1_test = read_json(V1_TEST_CANDIDATES)
        if set(v1_test) != set(test_texts):
            raise RuntimeError('The v1 test candidate cache is incomplete')
        selected_map = {}
        for fold_index in range(3):
            selected_map.update(selected_v2_candidates(fold_index))
        X_train, y_train, _, ranges_train = build_feature_table(
            texts, V1_OOF_CANDIDATES, selected_map, texts, with_gold=True)
        meta = train_reranker(X_train, y_train, ranges_train, sorted(texts))
        final_dir = ROOT / 'final'
        meta_path = final_dir / 'document_reranker.pkl'
        with tempfile.NamedTemporaryFile(dir=final_dir, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
        try:
            with temporary.open('wb') as stream:
                pickle.dump(meta, stream, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary, meta_path)
        finally:
            temporary.unlink(missing_ok=True)
        X_test, _, test_rows, test_ranges = build_feature_table(
            test_texts, v1_test, RAW_TEST, test_texts, with_gold=False)
        reranked = rerank_documents(meta, X_test, test_rows, test_ranges, test_texts)
        predictions = decode(reranked, STAGE8['reranker_thresholds'])
        del meta, X_train, X_test
    else:
        raise RuntimeError('The unchanged v1 method should not create a new submission')
    if set(predictions) != set(test_texts):
        raise RuntimeError('Submission is missing an official test document')
    for doc_id, spans in predictions.items():
        for a, b, label in spans:
            if label not in LABEL_ID or not 0 <= a < b <= len(test_texts[doc_id]):
                raise RuntimeError(f'Invalid v2 span in test document {doc_id}')
        if any(spans[i][1] > spans[i+1][0] for i in range(len(spans)-1)):
            raise RuntimeError(f'Overlapping v2 predictions in test document {doc_id}')
    output = ROOT / 'final' / 'prediction.json'
    atomic_json(output, predictions)
    package_path = WORKING / 'CS781_Phase2_Span_V2_Submission.zip'
    temporary = package_path.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        archive.write(output, arcname='prediction.json')
    os.replace(temporary, package_path)
    with zipfile.ZipFile(package_path) as archive:
        if (archive.testzip() is not None or archive.namelist() != ['prediction.json'] or
                json.loads(archive.read('prediction.json')) != predictions):
            raise RuntimeError('The new submission ZIP failed CRC/content verification')
    PACKAGE = {'status': 'READY_FOR_MANUAL_UPLOAD', 'method': method,
               'submission_zip': str(package_path),
               'zip_sha256': file_sha(package_path),
               'test_documents': len(predictions),
               'predicted_spans': sum(map(len, predictions.values()))}
    atomic_json(ROOT / 'final' / 'package.json', PACKAGE)
    print('New verified submission:', package_path)

RECOVERY = None
recovery_members = [(ROOT / 'experiment_config.json', 'experiment_config.json')]
for fold_index in range(3):
    best = v2_best(ROOT / f'fold{fold_index}', fold_index)
    if best is not None:
        recovery_members.append((ROOT / f'fold{fold_index}' / best['adapter_file'],
                                 f'fold{fold_index}/selected_adapter.pt'))
for name in ('selected_adapter.pt', 'document_reranker.pkl'):
    candidate = ROOT / 'final' / name
    if candidate.is_file():
        recovery_members.append((candidate, f'final/{name}'))
if len(recovery_members) > 1:
    recovery_path = WORKING / 'CS781_Phase2_Span_V2_Recovery.zip'
    temporary = recovery_path.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1) as archive:
        archive.writestr('README.txt',
                         'This contains new v2 trainable states only. Reuse the saved Phase-1 '
                         'base-03 checkpoint and v1 notebook artifacts with the v2 notebook source.\n')
        for path, name in recovery_members:
            archive.write(path, arcname=name)
    os.replace(temporary, recovery_path)
    with zipfile.ZipFile(recovery_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('V2 recovery ZIP CRC validation failed')
    RECOVERY = {'zip': str(recovery_path), 'sha256': file_sha(recovery_path),
                'members': [name for _, name in recovery_members]}

status = ('READY_FOR_MANUAL_UPLOAD' if PACKAGE else
          'PAUSED_RUN_ALL_TO_RESUME' if STAGE8 is None or
          (STAGE8['chosen_oof'] >= FINAL_GATE and RAW_TEST is None)
          else 'NO_LOCAL_IMPROVEMENT_KEEP_V1')
REPORT = {'version': VERSION, 'status': status, 'config_hash': CONFIG_HASH,
          'elapsed_seconds': round(time.time()-RUN_STARTED, 1),
          'disk_free_gib': round(free_gib(), 3),
          'v1_baseline_oof': V1_OOF['calibrated_macro_f1'],
          'final_fit_gate': FINAL_GATE,
          'v1_official_score_user_reported': 0.4584,
          'fold_results': FOLD_RESULTS, 'comparison': STAGE8,
          'final_fit': FINAL_RESULT, 'package': PACKAGE, 'recovery': RECOVERY,
          'caveats': [
              'OOF metrics are selected on the same folds and can be optimistic.',
              'The 0.4584 score was reported by the user, not independently verified here.',
              'The report does not establish any private-test F1 for this new submission.',
              'No private labels or outside pretrained weights were used.']}
atomic_json(ROOT / 'report.json', REPORT)
readme = '''CS781 Phase-2 larger span run\n\nThe report status and comparison are in report.json.\nThe v1 source and checkpoints were read only. V2 artifacts remain under phase2_span_v2.\nA new submission ZIP exists separately only when the report status is READY_FOR_MANUAL_UPLOAD.\nThe exact private score is unknown until Codabench evaluates the ZIP.\n'''
report_files = [(ROOT / 'report.json', 'report.json'),
                (ROOT / 'experiment_config.json', 'experiment_config.json')]
if (ROOT / 'model_comparison.json').is_file():
    report_files.append((ROOT / 'model_comparison.json', 'model_comparison.json'))
for fold_index in range(3):
    folder = ROOT / f'fold{fold_index}'
    for name in ('summary.json', 'history.json', 'best.json'):
        path = folder / name
        if path.is_file():
            report_files.append((path, f'fold{fold_index}/{name}'))
    best = v2_best(folder, fold_index)
    if best is not None:
        report_files.append((folder / best['candidates_file'],
                             f'fold{fold_index}/{best["candidates_file"]}'))
if (ROOT / 'final' / 'summary.json').is_file():
    report_files.append((ROOT / 'final' / 'summary.json', 'final/summary.json'))
if (ROOT / 'final' / 'package.json').is_file():
    report_files.append((ROOT / 'final' / 'package.json', 'final/package.json'))
if (ROOT / 'final' / 'test_candidates.partial.json').is_file():
    report_files.append((ROOT / 'final' / 'test_candidates.partial.json',
                         'final/test_candidates.partial.json'))
temporary = REPORT_ZIP.with_suffix('.zip.tmp')
with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED,
                     compresslevel=6) as archive:
    archive.writestr('README.txt', readme)
    for path, name in report_files:
        archive.write(path, arcname=name)
os.replace(temporary, REPORT_ZIP)
with zipfile.ZipFile(REPORT_ZIP) as archive:
    if archive.testzip() is not None:
        raise RuntimeError('V2 report ZIP CRC validation failed')
print('Report:', REPORT_ZIP, 'status:', status)
if PACKAGE:
    print('Submission:', PACKAGE['submission_zip'])
if RECOVERY:
    print('Recovery:', RECOVERY['zip'])
print('Keep the old 0.4584 submission until the new official score is known.')
