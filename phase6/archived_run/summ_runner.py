"""CS781 Phase 6: six resumable Phase-1-based legal summarizers.

The Kaggle notebook embeds this file and utils.py. All data used for fitting
come from the official Phase-6 train split. External weights are used only
inside the course's published BERTScore evaluation procedure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from contextlib import nullcontext
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import cs781_phase6_utils as u

WORK = Path("/kaggle/working")
ROOT = WORK / "CS781_Capstone/phase6_summ"
ARTIFACT = WORK / "CS781_Phase6_All_Artifacts.zip"
u.ROOT, u.ARTIFACT = ROOT, ARTIFACT
REPO = "Exploration-Lab/CS-781-Capstone"
SEED = 781606
SLOTS = {
    "01": {"base": "02", "method": "lora", "lr": 1.7e-4, "rank": 16},
    "02": {"base": "03", "method": "adalora", "lr": 1.4e-4, "rank": 12},
    "03": {"base": "04", "method": "lorma", "lr": 1.6e-4, "rank": 12},
    "04": {"base": "01", "method": "houlsby", "lr": 2.0e-4, "rank": 48},
    "05": {"base": "02", "method": "pfeiffer", "lr": 2.0e-4, "rank": 64},
    "06": {"base": "03", "method": "distill", "lr": 1.4e-4, "rank": 8},
}
SOURCE_LIMIT = {"01": 520, "02": 960, "03": 760, "04": 1000}
TARGET_LIMIT = 480
MICRO_BATCH = 2
ACCUM = 4
MAX_EPOCHS = 8
PATIENCE = 3
EVAL_SIZE = 96
TEACHER_SIZE = 600
SAVE_MINUTES = 10
CUES = re.compile(
    r"\b(?:held|hold|ordered|order|appeal|petition|dismissed|allowed|issue|"
    r"facts|reason|conclusion|therefore|accordingly|court|judgment|"
    r"section|article|act|respondent|petitioner)\b", re.I)
WORD = re.compile(r"[A-Za-z]{3,}")
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def log(*parts):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *parts, flush=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text(value, field):
    if isinstance(value, (list, tuple)):
        pieces = [normalize_text(piece, field) for piece in value
                  if piece is not None and piece != ""]
        value = "\n".join(pieces)
    elif isinstance(value, dict):
        for key in ("text", "sentence", "sentences", "summary", "summaries",
                    "content", "headnote", "abstract", "reference", "paragraphs"):
            if key in value and value[key] is not None and value[key] != "":
                return normalize_text(value[key], field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must contain nonempty text")
    return value.strip()


def reference_text(item):
    """Read released labels without substituting the source as a target."""
    preferred = ("summary", "summaries", "reference_summary", "gold_summary",
                 "reference", "target", "headnote", "headnotes", "abstract",
                 "answer", "label")
    related = sorted(key for key in item if key not in preferred and re.search(
        r"summ|abstract|headnote|reference|target", key, re.I))
    for key in (*preferred, *related):
        if key not in item:
            continue
        try:
            return normalize_text(item[key], key), key
        except ValueError:
            continue
    return None, None


def stage_official():
    dest = ROOT / "official"
    if all((dest / f"{n}.json").is_file() for n in ("train", "test")) and (dest / "provenance.json").is_file():
        record = u.read_json(dest / "provenance.json")
        if all(record["sha256"][n] == u.sha(dest / f"{n}.json") for n in ("train", "test")):
            return
        raise RuntimeError("Official Phase 6 data changed after preparation")
    local = {}
    for name in ("train", "test"):
        hits = sorted(p for p in u.INPUT.rglob(f"{name}.json") if p.is_file()
                      and "phase_6_summ" in str(p).lower().replace("-", "_"))
        if len(hits) != 1:
            hits = sorted(p for p in u.INPUT.rglob(f"{name}.json") if p.is_file())
        local[name] = hits[0] if len(hits) == 1 else None
    if all(local.values()) and local["train"] != local["test"]:
        dest.mkdir(parents=True, exist_ok=True)
        for name, path in local.items():
            shutil.copy2(path, dest / f"{name}.json")
        provenance = {"source": "attached_official_files",
                      "paths": {k: str(v) for k, v in local.items()}}
    else:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
            if not token:
                raise RuntimeError("HF_TOKEN is empty")
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                                   "huggingface_hub>=0.24,<2"])
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
            from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        revision = api.repo_info(REPO, repo_type="dataset", token=token).sha
        files = api.list_repo_files(REPO, repo_type="dataset", revision=revision, token=token)
        chosen = {}
        for name, suffix in (("train", "Train/train.json"), ("test", "Test/test.json")):
            hits = [p for p in files if p.lower().replace("-", "_").endswith(
                ("phase_6_summ/" + suffix).lower())]
            if len(hits) != 1:
                raise RuntimeError(f"Expected one official Phase 6 {suffix}, found {hits}")
            chosen[name] = hits[0]
        dest.mkdir(parents=True, exist_ok=True)
        for name, path in chosen.items():
            cached = Path(hf_hub_download(repo_id=REPO, repo_type="dataset",
                                          filename=path, revision=revision,
                                          token=token, cache_dir=str(ROOT / "hf_cache")))
            with tempfile.NamedTemporaryFile(dir=dest, delete=False) as out:
                temporary = Path(out.name)
                with cached.open("rb") as src:
                    shutil.copyfileobj(src, out, 4 << 20)
            os.replace(temporary, dest / f"{name}.json")
        provenance = {"source": REPO, "revision": revision, "paths": chosen}
        del token
    provenance["sha256"] = {n: u.sha(dest / f"{n}.json") for n in ("train", "test")}
    u.write_json(dest / "provenance.json", provenance)


def parse_split(name):
    raw = u.read_json(ROOT / "official" / f"{name}.json")
    if isinstance(raw, dict):
        for field in ("data", "examples", "documents"):
            if isinstance(raw.get(field), list):
                raw = raw[field]
                break
        else:
            raw = [{"id": k, **(v if isinstance(v, dict) else {"document": v})}
                   for k, v in raw.items()]
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{name}: expected a nonempty list or ID object")
    rows = []
    missing_labels = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"{name} row {i} is not an object")
        source_id = item.get("id", item.get("document_id"))
        if type(source_id) is not str or not source_id:
            raise ValueError(f"{name} row {i} has no string ID")
        document = normalize_text(item.get("document", item.get("text")), "document")
        if name == "test":
            if reference_text(item)[0] is not None:
                raise ValueError("Test contains reference summaries; stop")
            summary = None
        else:
            summary, _ = reference_text(item)
            if summary is None:
                missing_labels.append((i, source_id, sorted(item)))
                continue
        rows.append({"id": f"{name}::{i:08d}", "source_id": source_id,
                     "document": document, "summary": summary})
    if name == "train" and missing_labels:
        if not rows or len(missing_labels) > max(10, int(.02 * len(raw))):
            first = missing_labels[0]
            raise ValueError(f"Training labels missing in {len(missing_labels)}/{len(raw)} "
                             f"rows; first row={first[0]}, id={first[1]!r}, "
                             f"keys={first[2]}")
    if name == "train":
        label_audit = ROOT / "official" / "label_audit.json"
        if not label_audit.is_file():
            u.write_json(label_audit, {
                "source_rows": len(raw), "labeled_rows": len(rows),
                "skipped_without_reference": len(missing_labels),
                "skipped_ids": [entry[1] for entry in missing_labels]})
            if missing_labels:
                log("SKIPPED TRAIN ROWS WITHOUT REFERENCE", len(missing_labels),
                    "of", len(raw), "first IDs", [x[1] for x in missing_labels[:5]])
    if name == "test" and len({r["source_id"] for r in rows}) != len(rows):
        raise ValueError("Official test IDs are repeated")
    return rows


def data():
    return parse_split("train"), parse_split("test")


def text_hash(s):
    return hashlib.sha256(" ".join(s.casefold().split()).encode()).hexdigest()


def split_rows(train):
    parent = list(range(len(train)))
    seen = {}
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, row in enumerate(train):
        for key in (("id", row["source_id"]), ("text", text_hash(row["document"]))):
            if key in seen:
                parent[root(i)] = root(seen[key])
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(train):
        groups[root(i)].append(row)
    items = list(groups.values())
    rng = random.Random(SEED)
    rng.shuffle(items)
    items.sort(key=lambda g: (-len(g), -max(len(x["document"]) for x in g)))
    bins = [[], [], []]
    target = [.90, .05, .05] if len(train) >= 4000 else [.80, .10, .10]
    for group in items:
        cost = [len(bins[j]) / max(1., target[j] * len(train)) for j in range(3)]
        bins[min(range(3), key=lambda j: cost[j])].extend(group)
    if min(map(len, bins)) == 0:
        raise ValueError("Cannot make fit/tune/audit groups")
    return {r["id"]: j for j, bucket in enumerate(bins) for r in bucket}


def prepare(require_cuda=True):
    ROOT.mkdir(parents=True, exist_ok=True)
    u.restore_artifact()
    if require_cuda and torch.cuda.device_count() != 2:
        raise RuntimeError("Select Kaggle's two-GPU accelerator")
    for base in ("01", "02", "03", "04"):
        u.stage_base(base)
    stage_official()
    train, test = data()
    split = split_rows(train)
    sha = {n: u.sha(ROOT / "official" / f"{n}.json") for n in ("train", "test")}
    config = {"version": 1, "seed": SEED, "slots": SLOTS, "source_limit": SOURCE_LIMIT,
              "target_limit": TARGET_LIMIT, "split": split, "official_sha256": sha,
              "base_sha256": {b: u.read_json(ROOT / "bases" / b / "provenance.json")["checkpoint_sha256"]
                              for b in ("01", "02", "03", "04")},
              "runner_sha256": u.sha(__file__)}
    path = ROOT / "run_config.json"
    if path.is_file() and u.read_json(path) != config:
        raise RuntimeError("Recovery code/data/config differs from current notebook")
    u.write_json(path, config)
    shutil.copy2(__file__, ROOT / "summ_runner.py")
    shutil.copy2(Path(u.__file__), ROOT / "utils.py")
    lengths = [len(x["document"]) for x in train]
    summary_words = [len(x["summary"].split()) for x in train]
    audit = {"train": len(train), "test": len(test),
             "split_counts": dict(Counter(split.values())),
             "document_characters": {"median": int(np.median(lengths)),
                                     "p95": int(np.percentile(lengths, 95))},
             "summary_words": {"median": int(np.median(summary_words)),
                               "p95": int(np.percentile(summary_words, 95))},
             "duplicate_documents": len(train) - len({text_hash(x["document"]) for x in train}),
             "private_score": None}
    u.write_json(ROOT / "data_audit.json", audit)
    log("PREPARED", audit)


def segments(document):
    parts = []
    for para in re.split(r"\n\s*\n|\n", document):
        para = para.strip()
        if not para:
            continue
        for sentence in SENTENCE.split(para):
            sentence = sentence.strip()
            if sentence:
                parts.append(sentence)
    return parts or [document]


def select_evidence(document, tokenizer, budget, policy="section"):
    """Deterministic, reference-free source selection with section coverage."""
    if policy == "lead":
        return tokenizer.decode(tokenizer.encode(document)[:budget])
    if policy != "section":
        raise ValueError(policy)
    units = segments(document)
    encoded = [tokenizer.encode(x) for x in units]
    if sum(len(x) for x in encoded) <= budget:
        return document
    n = len(units)
    freqs = Counter(w.lower() for unit in units for w in WORD.findall(unit))
    scores = []
    for i, (unit, ids) in enumerate(zip(units, encoded)):
        words = [w.lower() for w in WORD.findall(unit)]
        rarity = sum(1. / math.sqrt(freqs[w]) for w in set(words)) / max(1, len(set(words)))
        cue = min(3, len(CUES.findall(unit)))
        position = i / max(1, n - 1)
        edge = 1.5 if position < .1 or position > .85 else 0.
        scores.append(rarity + .28 * cue + edge)
    chosen, used = set(), 0
    anchors = [0, n - 1]
    for i in anchors:
        if i not in chosen and len(encoded[i]) <= budget - used:
            chosen.add(i); used += len(encoded[i])
    # Reserve representation from each broad phase of a judgment before
    # salience ranking fills the remaining budget.
    for lo, hi in ((0, n//3), (n//3, 2*n//3), (2*n//3, n)):
        candidates = [i for i in range(lo, hi) if i not in chosen and
                      len(encoded[i]) <= budget-used]
        if candidates:
            i = max(candidates, key=lambda j: scores[j])
            chosen.add(i); used += len(encoded[i])
    order = sorted(range(n), key=lambda i: scores[i] / max(16, len(encoded[i])) ** .25,
                   reverse=True)
    for i in order:
        if i in chosen or len(encoded[i]) > budget - used:
            continue
        similarity = max((len(set(WORD.findall(units[i].lower())) &
                              set(WORD.findall(units[j].lower()))) /
                          max(1, len(set(WORD.findall(units[i].lower()))))
                          for j in chosen), default=0.)
        if similarity > .85:
            continue
        chosen.add(i); used += len(encoded[i])
    result = "\n".join(units[i] for i in sorted(chosen))
    if not result:
        result = document[:2000]
    return result


def token_cache(base_id):
    path = ROOT / "token_cache" / f"base_{base_id}.pt"
    if path.is_file():
        return
    _, tok = u.load_base(base_id)
    train, test = data()
    policy_path = ROOT / f"selection_policy_{base_id}.json"
    if policy_path.is_file():
        policy = u.read_json(policy_path)["policy"]
    else:
        fit_ids = {r["id"] for r in stage_rows("fit")}
        sample = balanced_eval_rows([r for r in train if r["id"] in fit_ids], 128)
        comparison = {"lead": [], "section": []}
        for row in sample:
            reference = set(w.lower() for w in WORD.findall(row["summary"]))
            for candidate in comparison:
                evidence = select_evidence(row["document"], tok,
                                           SOURCE_LIMIT[base_id], candidate)
                observed = set(w.lower() for w in WORD.findall(evidence))
                comparison[candidate].append(len(reference & observed) /
                                             max(1, len(reference)))
        means = {k: float(np.mean(v)) for k, v in comparison.items()}
        policy = "section" if means["section"] >= means["lead"] - .005 else "lead"
        u.write_json(policy_path, {"policy": policy, "fit_only_sample": len(sample),
                                   "reference_word_recall": means})
        log("SELECTOR", base_id, policy, means)
    target_ids = {row["id"]: tok.encode(row["summary"]) for row in train}
    fit_ids = {r["id"] for r in stage_rows("fit")}
    fit = [r for r in train if r["id"] in fit_ids]
    boundaries = [float(np.percentile([len(r["document"]) for r in fit], q))
                  for q in (25, 50, 75)]
    groups = [[] for _ in range(4)]
    for row in fit:
        index = int(np.searchsorted(boundaries, len(row["document"]), side="right"))
        groups[index].append(len(target_ids[row["id"]]) + 1)
    overall = int(np.median([len(target_ids[r["id"]])+1 for r in fit]))
    bin_medians = [int(np.median(g)) if g else overall for g in groups]
    constant_error = np.mean([abs(len(target_ids[r["id"]])+1-overall) for r in fit])
    adaptive_error = np.mean([
        abs(len(target_ids[r["id"]])+1 -
            bin_medians[int(np.searchsorted(boundaries, len(r["document"]), side="right"))])
        for r in fit])
    length_policy = "adaptive" if adaptive_error < .95*constant_error else "constant"
    u.write_json(ROOT / f"length_policy_{base_id}.json",
                 {"policy": length_policy, "boundaries": boundaries,
                  "bin_medians": bin_medians, "overall_median": overall,
                  "fit_constant_mae": float(constant_error),
                  "fit_adaptive_mae": float(adaptive_error)})
    result = {}
    source_lengths, target_lengths = [], []
    for row in train + test:
        source = select_evidence(row["document"], tok, SOURCE_LIMIT[base_id], policy)
        prefix = tok.encode("Legal judgment:\n" + source + "\nSummary:")
        cap = SOURCE_LIMIT[base_id] + 24
        prefix = [tok.bos_token_id] + prefix[:cap - 1]
        target = (target_ids[row["id"]][:TARGET_LIMIT - 1] + [tok.eos_token_id]
                  if row["summary"] is not None else None)
        bin_index = int(np.searchsorted(boundaries, len(row["document"]), side="right"))
        hint = bin_medians[bin_index] if length_policy == "adaptive" else overall
        maximum = 1024 if base_id == "01" else 1280 if base_id == "03" else 1536 if base_id == "04" else 1024
        if base_id != "02" and len(prefix) + TARGET_LIMIT > maximum:
            prefix = prefix[:maximum - TARGET_LIMIT]
        result[row["id"]] = {"source": prefix, "target": target,
                             "length_hint": hint}
        source_lengths.append(len(prefix))
        if target is not None:
            target_lengths.append(len(target))
    path.parent.mkdir(parents=True, exist_ok=True)
    u.save_torch(path, result)
    audit = {"base": base_id, "selection_policy": policy,
             "source_median": int(np.median(source_lengths)),
             "source_p95": int(np.percentile(source_lengths, 95)),
             "target_median": int(np.median(target_lengths)),
             "target_p95": int(np.percentile(target_lengths, 95)),
             "targets_truncated": sum(len(target_ids[r["id"]]) >= TARGET_LIMIT
                                      for r in train)}
    u.write_json(ROOT / f"token_audit_{base_id}.json", audit)
    log("TOKENIZED", base_id, audit)


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha=2.):
        super().__init__()
        self.base = base
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        self.scale = alpha

    def forward(self, x):
        return self.base(x) + self.b(self.a(x)) * self.scale


class AdaLoRALinear(LoRALinear):
    def __init__(self, base, rank):
        super().__init__(base, rank, alpha=2.)
        self.s = nn.Parameter(torch.ones(rank))
        self.register_buffer("mask", torch.ones(rank))
        self.register_buffer("importance", torch.zeros(rank))

    def forward(self, x):
        hidden = self.a(x) * (self.s * self.mask)
        return self.base(x) + self.b(hidden) * self.scale

    @torch.no_grad()
    def record(self):
        if self.s.grad is not None:
            self.importance.mul_(.9).add_((self.s * self.s.grad).abs(), alpha=.1)


class LoRMALinear(nn.Module):
    """Effective weight is (I + B A) W, a multiplicative low-rank update."""
    def __init__(self, base, rank, alpha=1.):
        super().__init__()
        self.base = base
        self.a = nn.Linear(base.out_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        self.scale = alpha

    def forward(self, x):
        y = self.base(x)
        return y + self.b(self.a(y)) * self.scale


class AdapterOutput(nn.Module):
    def __init__(self, base, width, bottleneck):
        super().__init__()
        self.base = base
        self.down = nn.Linear(width, bottleneck)
        self.up = nn.Linear(bottleneck, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        y = self.base(x)
        return y + self.up(F.gelu(self.down(y)))


def replace_linear(model, name, wrapper):
    parent_name, attr = name.rsplit(".", 1)
    parent = model.get_submodule(parent_name)
    current = getattr(parent, attr)
    if not isinstance(current, nn.Linear):
        raise TypeError(f"{name} is not a linear projection")
    setattr(parent, attr, wrapper(current))


def adapt_model(base, slot):
    spec = SLOTS[slot]
    method, family, rank = spec["method"], spec["base"], spec["rank"]
    for p in base.parameters():
        p.requires_grad_(False)
    names = [n for n, m in base.named_modules() if isinstance(m, nn.Linear)]
    selected = []
    if method in ("lora", "distill"):
        if family == "02":
            selected = [n for n in names if "attention." in n and n.rsplit(".", 1)[-1]
                        in ("query", "value", "output")]
        else:
            selected = [n for n in names if ".attention." in n and n.rsplit(".", 1)[-1]
                        in ("query", "value", "output")]
        for n in selected:
            replace_linear(base, n, lambda m: LoRALinear(m, rank))
    elif method == "adalora":
        selected = [n for n in names if ".attention." in n and n.rsplit(".", 1)[-1]
                    in ("query", "value", "output")]
        for n in selected:
            replace_linear(base, n, lambda m: AdaLoRALinear(m, rank))
    elif method == "lorma":
        selected = [n for n in names if n.endswith(("q_proj", "v_proj", "o_proj", "down_proj"))]
        for n in selected:
            replace_linear(base, n, lambda m: LoRMALinear(m, rank))
    elif method == "houlsby":
        selected = [n for n in names if n.endswith(("attention.output", "ffn.down"))]
        for n in selected:
            m = base.get_submodule(n)
            replace_linear(base, n, lambda old: AdapterOutput(old, m.out_features, rank))
    elif method == "pfeiffer":
        selected = [n for n in names if n.endswith("feed_forward.output")]
        for n in selected:
            m = base.get_submodule(n)
            replace_linear(base, n, lambda old: AdapterOutput(old, m.out_features, rank))
    if not selected:
        raise RuntimeError(f"No layers selected for {slot} {method} {family}")
    # Norm adjustment helps the small Phase-1 models learn a new generation
    # format while the original attention and embedding weights remain fixed.
    for name, p in base.named_parameters():
        if any(t in name for t in ("final_norm", "output_bias")):
            p.requires_grad_(True)
    trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
    if trainable < 1000:
        raise RuntimeError(f"Adaptation {slot} has no meaningful trainable state")
    return base, {"wrapped": selected, "trainable_parameters": trainable}


def model_for(slot, device):
    base_id = SLOTS[slot]["base"]
    base, tokenizer = u.load_base(base_id)
    base, record = adapt_model(base, slot)
    base.to(device)
    if hasattr(base, "set_gradient_checkpointing"):
        base.set_gradient_checkpointing(True)
    return base, tokenizer, record


def trainable_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()
            if p.requires_grad}


def load_trainable(model, state):
    current = dict(model.named_parameters())
    expected = {n for n, p in current.items() if p.requires_grad}
    if set(state) != expected:
        raise RuntimeError(f"Trainable checkpoint keys differ: {len(set(state)^expected)}")
    with torch.no_grad():
        for n, value in state.items():
            if current[n].shape != value.shape:
                raise RuntimeError(f"Wrong trainable shape for {n}")
            current[n].copy_(value)


def allocate_ada_ranks(model, epoch):
    modules = [m for m in model.modules() if isinstance(m, AdaLoRALinear)]
    if not modules:
        return
    all_scores = []
    for i, module in enumerate(modules):
        for j, score in enumerate(module.importance.tolist()):
            all_scores.append((score, i, j))
    all_scores.sort(reverse=True)
    fraction = max(.5, 1. - .5 * epoch / max(1, MAX_EPOCHS))
    keep = {(i, j) for _, i, j in all_scores[:max(len(modules) * 2, int(len(all_scores) * fraction))]}
    with torch.no_grad():
        for i, module in enumerate(modules):
            module.mask.copy_(torch.tensor([1. if (i, j) in keep else 0.
                                           for j in range(len(module.mask))],
                                          device=module.mask.device))


def batch_tensors(rows, cache, tokenizer, base_id, device):
    pad, bos = tokenizer.pad_token_id, tokenizer.bos_token_id
    source = [cache[r["id"]]["source"] for r in rows]
    target = [cache[r["id"]]["target"] for r in rows]
    if any(x is None for x in target):
        raise ValueError("Training batch contains test row")
    if base_id == "02":
        src_len, tgt_len = max(map(len, source)), max(map(len, target))
        enc = torch.full((len(rows), src_len), pad, dtype=torch.long)
        enc_mask = torch.zeros((len(rows), src_len), dtype=torch.bool)
        dec = torch.full((len(rows), tgt_len), pad, dtype=torch.long)
        labels = torch.full((len(rows), tgt_len), -100, dtype=torch.long)
        for i, (s, t) in enumerate(zip(source, target)):
            enc[i, :len(s)] = torch.tensor(s)
            enc_mask[i, :len(s)] = True
            dec[i, :len(t)] = torch.tensor([bos] + t[:-1])
            labels[i, :len(t)] = torch.tensor(t)
        return (enc.to(device), dec.to(device), enc_mask.to(device)), labels.to(device)
    sequences = [s + [bos] + t[:-1] for s, t in zip(source, target)]
    length = max(map(len, sequences))
    ids = torch.full((len(rows), length), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), length), dtype=torch.bool)
    labels = torch.full((len(rows), length), -100, dtype=torch.long)
    prefix_lengths = []
    for i, (s, t, seq) in enumerate(zip(source, target, sequences)):
        ids[i, :len(seq)] = torch.tensor(seq)
        mask[i, :len(seq)] = True
        labels[i, len(s):len(s)+len(t)] = torch.tensor(t)
        prefix_lengths.append(len(s))
    args = (ids.to(device), mask.to(device),
            torch.tensor(prefix_lengths, device=device))
    return args, labels.to(device)


def forward_loss(model, args, labels, base_id):
    if base_id == "02":
        logits = model(args[0], args[1], args[2])
    elif base_id == "03":
        logits = model(args[0], attention_mask=args[1], prefix_lengths=args[2])
    elif base_id == "04":
        logits = model(args[0], attention_mask=args[1])
    else:
        logits = model(args[0])
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                           labels.reshape(-1), ignore_index=-100)


def stage_rows(stage):
    train, _ = data()
    split = u.read_json(ROOT / "run_config.json")["split"]
    if stage == "fit":
        return [r for r in train if split[r["id"]] == 0]
    if stage == "tune":
        return [r for r in train if split[r["id"]] == 1]
    if stage == "audit":
        return [r for r in train if split[r["id"]] == 2]
    if stage == "all":
        return train
    raise ValueError(stage)


def balanced_eval_rows(rows, size):
    if len(rows) <= size:
        return rows
    ordered = sorted(rows, key=lambda r: len(r["document"]))
    return [ordered[min(len(ordered)-1, int((i+.5)*len(ordered)/size))]
            for i in range(size)]


def adaptation_state(model):
    params = trainable_state(model)
    ada = {}
    for name, module in model.named_modules():
        if isinstance(module, AdaLoRALinear):
            ada[name] = {"mask": module.mask.detach().cpu().clone(),
                         "importance": module.importance.detach().cpu().clone()}
    return {"params": params, "ada": ada}


def load_adaptation(model, state):
    load_trainable(model, state["params"])
    modules = dict(model.named_modules())
    actual = {n for n, m in modules.items() if isinstance(m, AdaLoRALinear)}
    if set(state.get("ada", {})) != actual:
        raise RuntimeError("AdaLoRA buffer keys differ from checkpoint")
    with torch.no_grad():
        for name, values in state.get("ada", {}).items():
            modules[name].mask.copy_(values["mask"])
            modules[name].importance.copy_(values["importance"])


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(value):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if torch.cuda.is_available() and value.get("cuda") is not None:
        torch.cuda.set_rng_state(value["cuda"])


def epoch_batches(rows, cache, epoch, world, microbatch):
    ordered = sorted(rows, key=lambda r: len(cache[r["id"]]["source"]) +
                     len(cache[r["id"]]["target"]))
    buckets = [ordered[i:i+64] for i in range(0, len(ordered), 64)]
    rng = random.Random(SEED + epoch * 1009)
    for bucket in buckets:
        rng.shuffle(bucket)
    rng.shuffle(buckets)
    flat = [r for bucket in buckets for r in bucket]
    width = world * microbatch
    if len(flat) % width:
        flat += flat[:width - len(flat) % width]
    return [flat[i:i+width] for i in range(0, len(flat), width)]


def microbatch_for(slot):
    path = ROOT / "performance" / f"slot_{slot}.json"
    return int(u.read_json(path)["microbatch"]) if path.is_file() else MICRO_BATCH


def autotune(slot):
    """Measure a representative long batch on GPU 0 before committing a fit."""
    path = ROOT / "performance" / f"slot_{slot}.json"
    if path.is_file():
        return
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        u.write_json(path, {"microbatch": MICRO_BATCH, "reason": "CPU smoke"})
        return
    base_id = SLOTS[slot]["base"]
    cache = torch.load(ROOT / "token_cache" / f"base_{base_id}.pt",
                       map_location="cpu", weights_only=False)
    rows = stage_rows("fit")
    row = max(rows, key=lambda r: len(cache[r["id"]]["source"]) +
              len(cache[r["id"]]["target"]))
    candidates = []
    for batch_size in (1, 2, 3, 4):
        model, tok, _ = model_for(slot, device)
        args = labels = loss = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            args, labels = batch_tensors([row] * batch_size, cache, tok, base_id, device)
            torch.cuda.synchronize(device)
            started = time.monotonic()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16
                                if torch.cuda.get_device_capability(device)[0] >= 8
                                else torch.float16):
                loss = forward_loss(model, args, labels, base_id)
            loss.backward()
            torch.cuda.synchronize(device)
            seconds = time.monotonic() - started
            memory = torch.cuda.max_memory_reserved(device)
            total = torch.cuda.get_device_properties(device).total_memory
            throughput = batch_size * (len(cache[row["id"]]["source"]) +
                                       len(cache[row["id"]]["target"])) / max(.001, seconds)
            if memory < .87 * total:
                candidates.append((throughput, batch_size, memory / 2**30, seconds))
            log("PROBE", slot, "batch", batch_size, "tokens/sec", round(throughput),
                "reserved_GiB", round(memory/2**30,2), "seconds", round(seconds,2))
        except torch.cuda.OutOfMemoryError:
            log("PROBE OOM", slot, "batch", batch_size)
        finally:
            del model, args, labels, loss
            torch.cuda.empty_cache()
    if not candidates:
        raise RuntimeError(f"Even per-GPU batch 1 does not fit for slot {slot}")
    chosen = max(candidates)
    u.write_json(path, {"microbatch": chosen[1], "tokens_per_second": chosen[0],
                        "reserved_gib": chosen[2], "seconds": chosen[3],
                        "candidate_count": len(candidates)})
    log("AUTOTUNED", slot, "microbatch", chosen[1])


def training_device():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        if torch.cuda.device_count() < world:
            raise RuntimeError("Distributed run needs two CUDA GPUs")
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        dist.init_process_group("nccl")
        device = torch.device("cuda", local)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, rank, world


def training_checkpoint(directory, model, optimizer, scaler, progress, rank, world):
    u.save_torch(directory / f"rng_rank{rank}.pt", rng_state())
    if world > 1:
        dist.barrier()
    if rank == 0:
        target = directory / "latest.pt"
        if target.is_file():
            shutil.copy2(target, directory / "previous_known_good.pt")
        u.save_torch(target, {"adaptation": adaptation_state(model),
                              "optimizer": optimizer.state_dict(),
                              "scaler": scaler.state_dict(),
                              "progress": progress})
        u.write_json(directory / "status.json", progress)
    if world > 1:
        dist.barrier()


def train_stage(slot, stage="fit", max_minutes=20.):
    device, rank, world = training_device()
    try:
        seed_all(SEED + rank)
        spec = SLOTS[slot]
        base_id = spec["base"]
        microbatch = microbatch_for(slot)
        directory = ROOT / "runs" / slot / stage
        directory.mkdir(parents=True, exist_ok=True)
        model, tok, record = model_for(slot, device)
        if world > 1:
            wrapped = DDP(model, device_ids=[device.index], output_device=device.index,
                          broadcast_buffers=False)
        else:
            wrapped = model
        train_rows = stage_rows("fit" if stage == "fit" else "all")
        cache = torch.load(ROOT / "token_cache" / f"base_{base_id}.pt",
                           map_location="cpu", weights_only=False)
        teacher = {}
        if slot == "06":
            pseudo = ROOT / "teacher" / "summaries.json"
            if not pseudo.is_file():
                raise RuntimeError("Distillation requires finished train-only teacher summaries")
            teacher = u.read_json(pseudo)
            if not set(teacher) <= {r["id"] for r in stage_rows("fit")}:
                raise RuntimeError("Teacher summaries include non-fitting rows")
            for row_id, summary in teacher.items():
                cache[row_id] = dict(cache[row_id])
                cache[row_id]["teacher_target"] = tok.encode(summary)[:TARGET_LIMIT-1] + [tok.eos_token_id]
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=spec["lr"], weight_decay=.01,
                                      fused=device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8)
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and
                                           torch.cuda.get_device_capability(device)[0] < 8)
        progress = {"epoch": 0, "batch_cursor": 0, "optimizer_steps": 0,
                    "pending_eval": False, "complete": False, "best_score": -1.,
                    "best_epoch": 0, "no_improve": 0,
                    "trainable_parameters": record["trainable_parameters"],
                    "microbatch": microbatch, "world": world}
        latest = directory / "latest.pt"
        if latest.is_file():
            saved = torch.load(latest, map_location="cpu", weights_only=False)
            load_adaptation(model, saved["adaptation"])
            optimizer.load_state_dict(saved["optimizer"])
            scaler.load_state_dict(saved["scaler"])
            progress = saved["progress"]
            rng_file = directory / f"rng_rank{rank}.pt"
            if rng_file.is_file():
                restore_rng(torch.load(rng_file, map_location="cpu", weights_only=False))
        if progress["complete"] or progress["pending_eval"]:
            if rank == 0:
                u.write_json(directory / "status.json", progress)
            return
        if progress["world"] != world or progress["microbatch"] != microbatch:
            raise RuntimeError("Distributed shape changed across recovery")
        model.train()
        batches = epoch_batches(train_rows, cache, progress["epoch"], world, microbatch)
        start = time.monotonic()
        last_save = start
        rolling_loss, seen = 0., 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index in range(progress["batch_cursor"], len(batches)):
            global_rows = batches[batch_index]
            rows = global_rows[rank*microbatch:(rank+1)*microbatch]
            if slot == "06":
                # Sequence-level KD augments gold examples, not the validation
                # or audit split. A deterministic quarter of eligible rows
                # receives a teacher sequence in each epoch.
                selected = []
                for row in rows:
                    value = cache[row["id"]]
                    if "teacher_target" in value and ((batch_index + progress["epoch"]) % 4 == 0):
                        cache[row["id"]] = dict(value, target=value["teacher_target"])
                        selected.append((row["id"], value))
                args, labels = batch_tensors(rows, cache, tok, base_id, device)
                for row_id, original in selected:
                    cache[row_id] = original
            else:
                args, labels = batch_tensors(rows, cache, tok, base_id, device)
            end_step = (batch_index + 1) % ACCUM == 0 or batch_index + 1 == len(batches)
            synchronization = wrapped.no_sync() if world > 1 and not end_step else nullcontext()
            with synchronization:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16
                                    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
                                    else torch.float16, enabled=device.type == "cuda"):
                    loss = forward_loss(wrapped, args, labels, base_id) / ACCUM
                scaler.scale(loss).backward()
            rolling_loss += float(loss.detach()) * ACCUM
            seen += 1
            if not end_step:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.)
            for module in model.modules():
                if isinstance(module, AdaLoRALinear):
                    module.record()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            progress["optimizer_steps"] += 1
            progress["batch_cursor"] = batch_index + 1
            # A measured step budget remains fixed across resumes.
            steps_per_epoch = math.ceil(len(batches) / ACCUM)
            total_steps = MAX_EPOCHS * steps_per_epoch
            step = progress["optimizer_steps"]
            warm = max(20, int(.04 * total_steps))
            factor = step / warm if step < warm else .08 + .92 * .5 * (
                1 + math.cos(math.pi * min(1., (step-warm)/max(1, total_steps-warm))))
            for group in optimizer.param_groups:
                group["lr"] = spec["lr"] * factor
            now = time.monotonic()
            save_due = now - last_save >= SAVE_MINUTES * 60
            stop_due = now - start >= max_minutes * 60
            if world > 1:
                decisions = torch.tensor([int(save_due), int(stop_due)],
                                         device=device, dtype=torch.int32)
                dist.all_reduce(decisions, op=dist.ReduceOp.MAX)
                save_due, stop_due = bool(decisions[0].item()), bool(decisions[1].item())
            if save_due or stop_due:
                training_checkpoint(directory, model, optimizer, scaler, progress, rank, world)
                last_save = now
            if stop_due:
                break
        if progress["batch_cursor"] == len(batches):
            progress["epoch"] += 1
            progress["batch_cursor"] = 0
            progress["pending_eval"] = stage == "fit"
            allocate_ada_ranks(model, progress["epoch"])
            if stage != "fit":
                target_steps = u.read_json(ROOT / "runs" / slot / "fit" / "status.json")["best_epoch"]
                progress["complete"] = progress["epoch"] >= max(1, target_steps)
        training_checkpoint(directory, model, optimizer, scaler, progress, rank, world)
        if rank == 0:
            log("TRAIN", slot, stage, "epoch", progress["epoch"],
                "cursor", progress["batch_cursor"], "steps", progress["optimizer_steps"],
                "loss", round(rolling_loss/max(1, seen), 4),
                "minutes", round((time.monotonic()-start)/60, 2),
                "peak_gpu_gib", round(torch.cuda.max_memory_allocated(device)/2**30, 2)
                if device.type == "cuda" else 0.)
    finally:
        if world > 1:
            dist.destroy_process_group()


def decoding_limits(base_id, length_hint=None):
    audit = u.read_json(ROOT / f"token_audit_{base_id}.json")
    median = audit["target_median"]
    p95 = audit["target_p95"]
    if length_hint is None:
        maximum = min(TARGET_LIMIT, max(100, int(p95 * 1.05)))
        minimum = min(maximum - 1, max(24, int(median * .45)))
    else:
        maximum = min(TARGET_LIMIT, max(64, int(length_hint * 1.35)))
        minimum = min(maximum - 1, max(20, int(length_hint * .5)))
    return minimum, maximum


def forbidden_repeats(generated):
    if len(generated) < 2:
        return []
    pair = tuple(generated[-2:])
    return [generated[i+2] for i in range(len(generated)-2)
            if tuple(generated[i:i+2]) == pair]


@torch.inference_mode()
def generate_one(model, tok, base_id, source, min_new=None, max_new=None,
                 length_hint=None):
    model.eval()
    default_min, default_max = decoding_limits(base_id, length_hint)
    min_new = default_min if min_new is None else min_new
    max_new = default_max if max_new is None else max_new
    device = next(model.parameters()).device
    source_tensor = torch.tensor([source], device=device, dtype=torch.long)
    current = torch.tensor([[tok.bos_token_id]], device=device, dtype=torch.long)
    generated = []
    if base_id == "02":
        state = model.prepare_generation(source_tensor,
                                         torch.ones_like(source_tensor, dtype=torch.bool))
    elif base_id == "03":
        state = model.prefill_prefix(source_tensor)
    elif base_id == "04":
        state = model.prefill_context(source_tensor)
    else:
        logits, state = model(torch.cat((source_tensor, current), dim=1), use_cache=True)
        logits = logits[:, -1, :]
    for step in range(max_new):
        if base_id == "02":
            logits, state = model.decode_step(current, state)
        elif base_id == "03":
            logits, state = model.decode_cached(current, state)
        elif base_id == "04":
            logits, state = model.decode_cached(current, state)
        elif step:
            logits, state = model(current, past_key_values=state, use_cache=True)
            logits = logits[:, -1, :]
        logits = logits[0].float().clone()
        if step < min_new:
            logits[tok.eos_token_id] = -float("inf")
        for token in forbidden_repeats(generated):
            logits[token] = -float("inf")
        next_id = int(torch.argmax(logits).item())
        if next_id == tok.eos_token_id:
            break
        generated.append(next_id)
        current = torch.tensor([[next_id]], device=device, dtype=torch.long)
    summary = tok.decode(generated).strip()
    if not summary:
        summary = tok.decode(generated[:max(1, min(30, len(generated)))], skip_special_tokens=False).strip()
    if not summary:
        summary = "The court considered the dispute and delivered its decision."
    return re.sub(r"\s+", " ", summary).strip()


def load_selected(slot, device, stage="fit"):
    model, tok, record = model_for(slot, device)
    directory = ROOT / "runs" / slot / stage
    path = directory / ("best.pt" if stage == "fit" else "latest.pt")
    if not path.is_file():
        raise RuntimeError(f"No selected checkpoint for {slot} {stage}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    load_adaptation(model, checkpoint["adaptation"])
    model.eval()
    return model, tok, record, checkpoint


def score_summaries(predictions, references):
    if len(predictions) != len(references) or not predictions:
        raise ValueError("Predictions/references do not align")
    try:
        from rouge_score import rouge_scorer
        import bert_score
    except ImportError as exc:
        raise RuntimeError("Install rouge-score and bert-score before evaluation") from exc
    rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    r1, rl = [], []
    for prediction, reference in zip(predictions, references):
        score = rouge.score(reference, prediction)
        r1.append(score["rouge1"].fmeasure)
        rl.append(score["rougeL"].fmeasure)
    # This is exactly the course evaluator's BERTScore call.
    metric_device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        p, r, f = bert_score.score(predictions, references, lang="en", batch_size=8,
                                   verbose=False, device=metric_device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        p, r, f = bert_score.score(predictions, references, lang="en", batch_size=2,
                                   verbose=False, device=metric_device)
    return {"bertscore_f1": float(f.mean().item()),
            "bertscore_precision": float(p.mean().item()),
            "bertscore_recall": float(r.mean().item()),
            "rouge1_f1": float(np.mean(r1)), "rougeL_f1": float(np.mean(rl)),
            "cases": len(predictions),
            "predicted_words_mean": float(np.mean([len(s.split()) for s in predictions])),
            "reference_words_mean": float(np.mean([len(s.split()) for s in references]))}


def evaluate_epoch(slot, max_minutes=20.):
    directory = ROOT / "runs" / slot / "fit"
    latest = directory / "latest.pt"
    saved = torch.load(latest, map_location="cpu", weights_only=False)
    progress = saved["progress"]
    if not progress["pending_eval"]:
        u.write_json(directory / "status.json", progress)
        return
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tok, _ = model_for(slot, device)
    load_adaptation(model, saved["adaptation"])
    cache = torch.load(ROOT / "token_cache" / f"base_{SLOTS[slot]['base']}.pt",
                       map_location="cpu", weights_only=False)
    rows = balanced_eval_rows(stage_rows("tune"), EVAL_SIZE)
    partial_path = directory / f"eval_epoch_{progress['epoch']:02d}_partial.json"
    partial = u.read_json(partial_path) if partial_path.is_file() else {}
    if not set(partial) <= {r["id"] for r in rows}:
        raise RuntimeError("Evaluation partial contains unexpected IDs")
    started = time.monotonic()
    for i, row in enumerate(rows):
        if row["id"] in partial:
            continue
        partial[row["id"]] = generate_one(model, tok, SLOTS[slot]["base"],
                                           cache[row["id"]]["source"],
                                           length_hint=cache[row["id"]].get("length_hint"))
        if (i+1) % 8 == 0 or time.monotonic()-started > max_minutes*60:
            u.write_json(partial_path, partial)
        if (i+1) % 16 == 0:
            log("EVALUATING", slot, "epoch", progress["epoch"], i+1, "/", len(rows))
        if time.monotonic()-started > max_minutes*60:
            log("EVALUATION PAUSED", slot, len(partial), "/", len(rows))
            return
    u.write_json(partial_path, partial)
    generated = [partial[r["id"]] for r in rows]
    metric = score_summaries(generated, [r["summary"] for r in rows])
    history_path = directory / "history.json"
    history = u.read_json(history_path) if history_path.is_file() else []
    history = [item for item in history if item["epoch"] != progress["epoch"]]
    history.append({"epoch": progress["epoch"], "metric": metric,
                    "optimizer_steps": progress["optimizer_steps"]})
    u.write_json(history_path, history)
    u.write_json(directory / f"predictions_epoch_{progress['epoch']:02d}.json",
                 [{"id": row["source_id"], "summary": text}
                  for row, text in zip(rows, generated)])
    score = metric["bertscore_f1"]
    if score > progress["best_score"] + .0003:
        progress["best_score"] = score
        progress["best_epoch"] = progress["epoch"]
        progress["no_improve"] = 0
        u.save_torch(directory / "best.pt",
                     {"adaptation": saved["adaptation"], "progress": dict(progress)})
    else:
        progress["no_improve"] += 1
    progress["pending_eval"] = False
    progress["complete"] = (progress["epoch"] >= MAX_EPOCHS or
                            (progress["epoch"] >= 2 and progress["no_improve"] >= PATIENCE))
    saved["progress"] = progress
    u.save_torch(latest, saved)
    u.write_json(directory / "status.json", progress)
    log("EVALUATED", slot, "epoch", progress["epoch"], metric,
        "best", progress["best_epoch"], "done", progress["complete"])


def score_selected(slot, split="tune", size=0, stage="fit", max_minutes=20.):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tok, _, _ = load_selected(slot, device, stage)
    cache = torch.load(ROOT / "token_cache" / f"base_{SLOTS[slot]['base']}.pt",
                       map_location="cpu", weights_only=False)
    rows = stage_rows(split)
    if size:
        rows = balanced_eval_rows(rows, size)
    output = ROOT / "scores" / f"slot_{slot}_{stage}_{split}_{size or 'all'}.json"
    if output.is_file():
        return
    partial_path = output.with_name(output.stem + "_partial.json")
    partial = u.read_json(partial_path) if partial_path.is_file() else {}
    if not set(partial) <= {r["id"] for r in rows}:
        raise RuntimeError("Selected-score partial contains unexpected IDs")
    started = time.monotonic()
    for i, row in enumerate(rows):
        if row["id"] in partial:
            continue
        partial[row["id"]] = generate_one(model, tok, SLOTS[slot]["base"],
                                           cache[row["id"]]["source"],
                                           length_hint=cache[row["id"]].get("length_hint"))
        if (i+1) % 10 == 0 or time.monotonic()-started > max_minutes*60:
            u.write_json(partial_path, partial)
        if time.monotonic()-started > max_minutes*60:
            log("SELECTED SCORE PAUSED", slot, split, len(partial), "/", len(rows))
            return
    u.write_json(partial_path, partial)
    generated = [partial[r["id"]] for r in rows]
    metric = score_summaries(generated, [r["summary"] for r in rows])
    u.write_json(output, {"slot": slot, "stage": stage, "split": split,
                          "metric": metric, "selected_epoch": 
                          u.read_json(ROOT / "runs" / slot / "fit" / "status.json")["best_epoch"]})
    u.write_json(output.with_name(output.stem + "_predictions.json"),
                 [{"id": r["source_id"], "summary": s}
                  for r, s in zip(rows, generated)])
    log("SCORED", output, metric)


def teacher_summaries(max_minutes=20.):
    target = ROOT / "teacher" / "summaries.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = u.read_json(target) if target.is_file() else {}
    rows = balanced_eval_rows(stage_rows("fit"), min(TEACHER_SIZE, len(stage_rows("fit"))))
    expected = {r["id"] for r in rows}
    if not set(existing) <= expected:
        raise RuntimeError("Teacher file has unexpected IDs")
    if set(existing) == expected:
        return
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tok, _, _ = load_selected("01", device)
    cache = torch.load(ROOT / "token_cache/base_02.pt", map_location="cpu", weights_only=False)
    started = time.monotonic()
    for i, row in enumerate(rows):
        if row["id"] in existing:
            continue
        existing[row["id"]] = generate_one(model, tok, "02",
                                           cache[row["id"]]["source"],
                                           length_hint=cache[row["id"]].get("length_hint"))
        if (i+1) % 10 == 0 or time.monotonic()-started > max_minutes*60:
            u.write_json(target, existing)
        if time.monotonic()-started > max_minutes*60:
            break
    u.write_json(target, existing)
    log("TEACHER", len(existing), "/", len(expected))


def generate_test(slot, max_minutes=20.):
    _, test = data()
    destination = ROOT / "submissions" / slot
    destination.mkdir(parents=True, exist_ok=True)
    stage = "final" if (ROOT / "runs" / slot / "final" / "status.json").is_file() and (
        u.read_json(ROOT / "runs" / slot / "final" / "status.json")["complete"]) else "fit"
    partial = destination / f"partial_{stage}.json"
    existing = u.read_json(partial) if partial.is_file() else {}
    expected = {r["source_id"] for r in test}
    if not set(existing) <= expected:
        raise RuntimeError("Partial test file has unexpected IDs")
    if set(existing) == expected:
        return
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tok, _, _ = load_selected(slot, device, stage)
    cache = torch.load(ROOT / "token_cache" / f"base_{SLOTS[slot]['base']}.pt",
                       map_location="cpu", weights_only=False)
    started = time.monotonic()
    for i, row in enumerate(test):
        if row["source_id"] in existing:
            continue
        existing[row["source_id"]] = generate_one(model, tok, SLOTS[slot]["base"],
                                                   cache[row["id"]]["source"],
                                                   length_hint=cache[row["id"]].get("length_hint"))
        if (i+1) % 10 == 0 or time.monotonic()-started > max_minutes*60:
            u.write_json(partial, existing)
        if time.monotonic()-started > max_minutes*60:
            break
    u.write_json(partial, existing)
    log("GENERATED", slot, len(existing), "/", len(test))


def package_slot(slot):
    _, test = data()
    dest = ROOT / "submissions" / slot
    stage = "final" if (ROOT / "runs" / slot / "final" / "status.json").is_file() and (
        u.read_json(ROOT / "runs" / slot / "final" / "status.json")["complete"]) else "fit"
    partial = u.read_json(dest / f"partial_{stage}.json")
    if set(partial) != {r["source_id"] for r in test}:
        raise RuntimeError(f"Slot {slot} test generation incomplete")
    rows = [{"id": r["source_id"], "summary": partial[r["source_id"]]}
            for r in test]
    if any(type(r["summary"]) is not str or not r["summary"].strip() for r in rows):
        raise RuntimeError("Blank generated summary")
    u.write_json(dest / "prediction.json", rows)
    output = WORK / f"CS781_Phase6_Submission_{slot}.zip"
    temporary = output.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(dest / "prediction.json", "prediction.json")
    os.replace(temporary, output)
    with zipfile.ZipFile(output) as archive:
        if archive.namelist() != ["prediction.json"] or json.loads(
                archive.read("prediction.json")) != rows:
            raise RuntimeError("Submission round trip failed")
    shutil.copy2(output, dest / "submission.zip")
    fit = u.read_json(ROOT / "runs" / slot / "fit" / "status.json")
    report = {"slot": slot, "base": SLOTS[slot]["base"],
              "method": SLOTS[slot]["method"], "status": "READY_FOR_MANUAL_UPLOAD",
              "checkpoint_stage": stage,
              "fit_best_bertscore_sample": fit["best_score"],
              "fit_best_epoch": fit["best_epoch"],
              "trainable_parameters": fit["trainable_parameters"],
              "test_cases": len(test),
              "prediction_sha256": u.sha(dest / "prediction.json"),
              "zip_sha256": u.sha(output), "private_score": None,
              "phase1_checkpoint_sha256": u.read_json(
                  ROOT / "bases" / SLOTS[slot]["base"] / "provenance.json")["checkpoint_sha256"]}
    u.write_json(ROOT / "reports" / f"submission_{slot}.json", report)
    log("PACKAGED", slot, "cases", len(test), "sample BERTScore F1", fit["best_score"])


def final_report():
    reports = [u.read_json(ROOT / "reports" / f"submission_{slot}.json") for slot in SLOTS]
    distinct = len({r["prediction_sha256"] for r in reports}) == len(reports)
    bases = {r["base"] for r in reports}
    methods = {r["method"] for r in reports}
    status = ("SIX_DISTINCT_PACKAGES_READY" if distinct and bases == {"01","02","03","04"}
              and len(methods) == 6 else "REVIEW_REQUIRED")
    scored = []
    for slot in SLOTS:
        path = ROOT / "scores" / f"slot_{slot}_fit_tune_all.json"
        if path.is_file():
            scored.append((u.read_json(path)["metric"]["bertscore_f1"], slot))
    recommended = max(scored)[1] if scored else None
    audit_path = (ROOT / "scores" / f"slot_{recommended}_fit_audit_all.json"
                  if recommended else None)
    u.write_json(ROOT / "report.json",
                 {"status": status, "reports": reports, "private_scores": None,
                  "artifact": str(ARTIFACT),
                  "recommended_slot": recommended,
                  "full_tune_bertscore_f1": max(scored)[0] if scored else None,
                  "untouched_audit_metric": u.read_json(audit_path)["metric"]
                  if audit_path is not None and audit_path.is_file() else None})
    log("FINAL REPORT", status, "recommended", recommended)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare","tokenize","autotune","train","eval",
                         "score","teacher","generate","package","report","snapshot"))
    parser.add_argument("--slot", choices=tuple(SLOTS), default="01")
    parser.add_argument("--base", choices=("01","02","03","04"), default="02")
    parser.add_argument("--stage", choices=("fit","final"), default="fit")
    parser.add_argument("--split", choices=("tune","audit"), default="tune")
    parser.add_argument("--size", type=int, default=0)
    parser.add_argument("--max-minutes", type=float, default=20.)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "tokenize":
        token_cache(args.base)
    elif args.action == "autotune":
        autotune(args.slot)
    elif args.action == "train":
        train_stage(args.slot, args.stage, args.max_minutes)
    elif args.action == "eval":
        evaluate_epoch(args.slot, args.max_minutes)
    elif args.action == "score":
        score_selected(args.slot, args.split, args.size, args.stage, args.max_minutes)
    elif args.action == "teacher":
        teacher_summaries(args.max_minutes)
    elif args.action == "generate":
        generate_test(args.slot, args.max_minutes)
    elif args.action == "package":
        package_slot(args.slot)
    elif args.action == "report":
        final_report()
    elif args.action == "snapshot":
        u.snapshot()


if __name__ == "__main__":
    main()
