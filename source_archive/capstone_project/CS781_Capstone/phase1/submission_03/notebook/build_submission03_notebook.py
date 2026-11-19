from __future__ import annotations

import copy
import textwrap
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "CS781_Phase1_Submission02_Infrastructure_FIXED.ipynb"
OUTPUT = ROOT / "CS781_Phase1_Submission03_Infrastructure_FIXED.ipynb"
PROJECT_COPY = Path(__file__).resolve().parent / OUTPUT.name


def cleaned(text: str) -> str:
    return textwrap.dedent(text).strip() + "\n"


def set_markdown(nb, index: int, text: str) -> None:
    nb.cells[index] = nbformat.v4.new_markdown_cell(cleaned(text))


def set_code(nb, index: int, text: str) -> None:
    nb.cells[index] = nbformat.v4.new_code_cell(cleaned(text))


nb = copy.deepcopy(nbformat.read(SOURCE, as_version=4))

# Mechanical namespace isolation first; submission-specific cells are replaced below.
for cell in nb.cells:
    if "source" not in cell:
        continue
    cell.source = (
        cell.source
        .replace("phase1_submission02", "phase1_submission03")
        .replace("submission_02", "submission_03")
        .replace("Submission 02", "Submission 03")
        .replace("submission02", "submission03")
    )

nb.metadata["cs781"] = {
    "phase": 1,
    "submission": 3,
    "purpose": "Kaggle-safe single-stack prefix-hybrid language-model infrastructure",
    "architecture_generated_by_llm": False,
    "student_owned_core_required": True,
}

set_markdown(
    nb,
    0,
    r"""
    # CS781 Capstone - Phase 1, Submission 03
    ## Student-owned single-stack prefix-hybrid legal language model

    This notebook is an independent Submission 03 workspace. It preserves the reliable official-data,
    checkpoint, recovery, prediction, and Codabench packaging infrastructure while changing the model
    integration and training objective to a **single-stack prefix-hybrid language model**.

    The prompt prefix uses bidirectional self-attention. Continuation positions attend to the complete
    prefix and only earlier continuation positions. Unlike Submission 01, the prompt is not encoded with
    a purely causal mask. Unlike Submission 02, there are no separate encoder and decoder stacks and no
    encoder-decoder cross-attention.

    **Academic-integrity boundary:** the official capstone document permits AI assistance with ancillary
    code and architecture understanding, but forbids AI from writing the whole assessed model,
    hyperparameter tuning, and result analysis/conclusions. Consequently, this notebook supplies the
    infrastructure and an enforced interface, but the submitted model module, numerical architecture,
    training hyperparameters, hypothesis, and conclusions remain student-owned. Full training is blocked
    until those inputs are supplied and truthfully attested.
    """,
)

set_markdown(
    nb,
    1,
    r"""
    ## Safe execution map

    - `prepare`: discover files, inspect schemas, build/reuse bounded caches, register the tokenizer,
      and validate the student module. This is the safe default.
    - `fresh_train`: start only after the representative GPU save/restore smoke test passes. It refuses
      an existing latest checkpoint.
    - `resume_train`: requires a compatible checkpoint and prints the restored optimizer step; it never
      silently falls back to step zero.
    - `inference_only`: restore the frozen tokenizer, student source/configuration, and selected model;
      only the official test prompts are required.

    `Run All` is safe by default. Corpus preprocessing, tokenizer training, GPU smoke testing, full
    training, inference, packaging, and final recovery export remain independently opt-in.
    """,
)

set_code(
    nb,
    4,
    r"""
    def _discover_attached_tokenizer() -> str | None:
        search_root = Path("/kaggle/input") if Path("/kaggle/input").exists() else Path.cwd()
        candidates = []
        for tokenizer_json in search_root.rglob("tokenizer.json"):
            directory = tokenizer_json.parent
            if (directory / "tokenizer_config.json").is_file() and (directory / "tokenizer_metadata.json").is_file():
                candidates.append(directory.resolve())
        preferred = [
            item for item in candidates
            if "submission01" in str(item).lower().replace("_", "").replace("-", "")
        ]
        choices = sorted(set(preferred or candidates))
        return str(choices[0]) if len(choices) == 1 else None

    def _discover_student_model_source() -> str | None:
        filename = "prefix_hybrid_lm.py"
        local = Path.cwd() / filename
        if local.is_file() and not Path("/kaggle/input").exists():
            return str(local.resolve())
        search_root = Path("/kaggle/input")
        if not search_root.exists():
            return None
        candidates = sorted(path.resolve() for path in search_root.rglob(filename) if path.is_file())
        return str(candidates[0]) if len(candidates) == 1 else None

    @dataclass
    class RunConfig:
        experiment_id: str = "phase1_submission03_prefix_hybrid"
        seed: int = 781
        execution_mode: str = "prepare"  # prepare | fresh_train | resume_train | inference_only
        workspace: str = field(default_factory=lambda: str(
            Path("/kaggle/working/CS781_Capstone/phase1/submission_03")
            if Path("/kaggle/working").exists()
            else Path.cwd() / "CS781_Capstone/phase1/submission_03"
        ))
        input_root: str = field(default_factory=lambda: str(
            Path("/kaggle/input") if Path("/kaggle/input").exists() else Path.cwd()
        ))
        train_files: tuple[str, ...] = (
            "train_acts.json", "train_high_courts.json",
            "train_supreme_court.json", "train_tribunals.json",
        )
        test_file: str = "Phase1- Test.json"
        test_filename_aliases: tuple[str, ...] = ("Phase1- Test.json", "test.json")
        hf_repo_id: str = "Exploration-Lab/CS-781-Capstone"
        hf_allow_patterns: tuple[str, ...] = ("*Phase_1_Pretraining/*", "*Phase-1_Pretraining/*")
        official_file_overrides: dict[str, str] = field(default_factory=dict)
        training_schema_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
        test_schema_override: dict[str, Any] | None = None

        # Keep these matched to a declared control unless your own experiment changes them.
        validation_fraction: float = 0.05
        source_mixing_ratios: dict[str, float] = field(default_factory=lambda: {
            "acts": 0.25, "high_courts": 0.25,
            "supreme_court": 0.25, "tribunals": 0.25,
        })
        # STUDENT: choose before preprocessing. The official document permits a compute-bounded subset.
        corpus_document_cap_per_source: int | None = None

        tokenizer_import_path: str | None = field(default_factory=_discover_attached_tokenizer)
        student_model_source_path: str | None = field(default_factory=_discover_student_model_source)
        completed_notebook_path: str | None = None

        architecture_record: dict[str, Any] = field(default_factory=lambda: {
            "family": "single-stack bidirectional-prefix / causal-continuation hybrid",
            "substantive_difference_from_submission01": (
                "the prompt segment is bidirectional rather than globally causal"
            ),
            "substantive_difference_from_submission02": (
                "one shared Transformer stack and one segment-aware self-attention mask; "
                "no separate encoder/decoder and no cross-attention"
            ),
            "attention_semantics": (
                "prefix queries attend to all valid prefix keys; continuation queries attend to all "
                "valid prefix keys and continuation keys at or before the query"
            ),
            "parameter_range": [50_000_000, 300_000_000],
            "student_model_config": "REQUIRED BEFORE TRAINING",
            "student_exact_parameter_derivation": "REQUIRED BEFORE TRAINING",
            "student_design_rationale": "REQUIRED BEFORE TRAINING",
        })
        innovation_hypothesis: str = "STUDENT MUST WRITE BEFORE TRAINING"
        ablation_control: str = (
            "STUDENT MUST NAME A MATCHED Submission 01 and/or Submission 02 control"
        )
        student_owned_architecture_attestation: bool = False
        student_configuration_confirmed: bool = False
        full_training_authorized: bool = False
        trained_from_scratch_attestation: bool = False
        decoding: dict[str, Any] = field(default_factory=lambda: {
            "target_words": 128,
            "max_new_tokens": None,
            "policies": [],  # STUDENT supplies named, ordered policies before inference.
        })

    CFG = RunConfig()
    if CFG.execution_mode not in {"prepare", "fresh_train", "resume_train", "inference_only"}:
        raise ValueError(f"Unsupported execution_mode: {CFG.execution_mode}")

    ROOT = Path(CFG.workspace)
    DIRS = {name: ROOT / name for name in (
        "model", "tokenizer", "data", "checkpoints", "histories", "evaluations",
        "predictions", "plots", "recovery", "manifests", "submission"
    )}
    for folder in DIRS.values():
        folder.mkdir(parents=True, exist_ok=True)

    RUN_CONFIG_PATH = DIRS["manifests"] / "run_config.json"
    def register_run_config(cfg: RunConfig) -> dict[str, Any]:
        current = json.loads(json.dumps(asdict(cfg), ensure_ascii=False))
        mutable_runtime = {
            "execution_mode", "full_training_authorized", "trained_from_scratch_attestation",
            "completed_notebook_path", "tokenizer_import_path", "student_model_source_path",
        }
        mutable_before_checkpoint = {
            "corpus_document_cap_per_source", "architecture_record", "innovation_hypothesis",
            "ablation_control", "student_owned_architecture_attestation",
            "student_configuration_confirmed", "decoding",
        }
        if RUN_CONFIG_PATH.exists():
            existing = json.loads(RUN_CONFIG_PATH.read_text(encoding="utf-8"))
            checkpoints_exist = any(DIRS["checkpoints"].glob("*.pt"))
            conflicts, merged = {}, dict(existing)
            for key, new_value in current.items():
                old_value = existing.get(key)
                if old_value == new_value:
                    continue
                if key in mutable_runtime or (key in mutable_before_checkpoint and not checkpoints_exist):
                    merged[key] = new_value
                else:
                    conflicts[key] = (old_value, new_value)
            if conflicts:
                raise RuntimeError(
                    f"Existing Submission 03 configuration conflict: {conflicts}. "
                    "Use a new experiment workspace instead of mutating a checkpointed run."
                )
            atomic_write_json(RUN_CONFIG_PATH, merged)
            return merged
        atomic_write_json(RUN_CONFIG_PATH, current)
        return current

    random.seed(CFG.seed); np.random.seed(CFG.seed); torch.manual_seed(CFG.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(CFG.seed)
    register_run_config(CFG)
    print({
        "workspace": str(ROOT), "mode": CFG.execution_mode,
        "corpus_cap_per_source": CFG.corpus_document_cap_per_source,
        "tokenizer_import_path": CFG.tokenizer_import_path,
        "student_model_source_path": CFG.student_model_source_path,
    })
    """,
)

nb.cells[6].source = nb.cells[6].source.replace(
    "Submission 03 will use one explicitly selected GPU.",
    "Submission 03 will use one explicitly selected GPU."
)

set_markdown(
    nb,
    17,
    r"""
    ## 6. Student-owned prefix-hybrid model and enforced contract

    Supply a standalone `prefix_hybrid_lm.py` that defines:

    - `make_model_config(vocab_size) -> Mapping[str, Any]`
    - `build_model(config) -> torch.nn.Module`
    - either `model(input_ids=..., attention_mask=..., prefix_lengths=...)`, or a module-level
      `forward_adapter(model, input_ids, attention_mask, prefix_lengths)`.

    For prefix length `p`, valid key `j`, and query `i`, the required visibility is:

    - if `i < p`, allow `j < p` (bidirectional prompt representation);
    - if `i >= p`, allow `j < p` or `p <= j <= i` (causal continuation conditioned on the prompt);
    - padded keys are never visible.

    The module must be implemented from scratch, contain 50M-300M parameters, use no pretrained model
    weights, and reconstruct exactly from its saved numerical configuration. The notebook tests tensor
    shape, finite gradients, target causality, bidirectional-prefix behavior, prompt conditioning, context
    limits, source hashing, and exact state reconstruction.
    """,
)

set_code(
    nb,
    18,
    r"""
    STUDENT_MODEL_CONFIG_OVERRIDE = None
    MODEL = MODEL_CONFIG = RECONSTRUCT_MODEL = MODEL_MODULE = None
    MODEL_SOURCE_PATH = MODEL_SOURCE_SHA256 = None

    def install_student_model_source(source_path: Path) -> Path:
        source_path = Path(source_path).expanduser().resolve()
        if not source_path.is_file(): raise FileNotFoundError(source_path)
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        banned = {"transformers", "fairseq"}
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree) if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported |= {
            (node.module or "").split(".")[0]
            for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        }
        if imported & banned:
            raise RuntimeError(f"Forbidden pretrained-model libraries in model source: {imported & banned}")
        destination = DIRS["model"] / "prefix_hybrid_lm.py"
        payload = source_path.read_bytes()
        if destination.exists() and destination.read_bytes() != payload and any(DIRS["checkpoints"].glob("*.pt")):
            raise RuntimeError("Refusing to change model source after checkpoints exist")
        atomic_write_bytes(destination, payload)
        return destination

    def hybrid_logits(model, input_ids, attention_mask, prefix_lengths):
        if MODEL_MODULE is not None and callable(getattr(MODEL_MODULE, "forward_adapter", None)):
            output = MODEL_MODULE.forward_adapter(model, input_ids, attention_mask, prefix_lengths)
        else:
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prefix_lengths=prefix_lengths,
            )
        return output.logits if hasattr(output, "logits") else output[0] if isinstance(output, (tuple, list)) else output

    def count_parameters(model: torch.nn.Module) -> dict[str, int]:
        return {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        }

    def validate_hybrid_contract(model, vocab_size: int, device: str, sequence_length: int = 13):
        model = model.to(device); model.train(); model.zero_grad(set_to_none=True)
        input_ids = torch.randint(0, vocab_size, (2, sequence_length), dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        prefix_lengths = torch.tensor([5, 7], dtype=torch.long, device=device)
        logits = hybrid_logits(model, input_ids, attention_mask, prefix_lengths)
        expected = (2, sequence_length, vocab_size)
        if tuple(logits.shape) != expected:
            raise ValueError(f"Expected logits shape {expected}, got {tuple(logits.shape)}")
        if not torch.isfinite(logits).all(): raise FloatingPointError("Non-finite logits")
        labels = torch.randint(0, vocab_size, (2, sequence_length), dtype=torch.long, device=device)
        for row, prefix in enumerate(prefix_lengths.tolist()): labels[row, :prefix] = -100
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size), labels.reshape(-1), ignore_index=-100
        )
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
            raise FloatingPointError("Missing or non-finite gradients")
        counts = count_parameters(model)
        if not 50_000_000 <= counts["total"] <= 300_000_000:
            raise ValueError(f"Official parameter bound violated: {counts['total']:,}")
        model.zero_grad(set_to_none=True)
        return {"logits_shape": list(logits.shape), "loss": float(loss.detach().cpu()), **counts, "device": device}

    _restored_model_source = DIRS["model"] / "prefix_hybrid_lm.py"
    _configured_model_source = (
        Path(CFG.student_model_source_path).expanduser()
        if CFG.student_model_source_path
        else _restored_model_source if _restored_model_source.exists() else None
    )
    if _configured_model_source is not None:
        if TOKENIZER_META is None:
            raise RuntimeError("Register/restore the tokenizer before importing the student model")
        MODEL_SOURCE_PATH = install_student_model_source(_configured_model_source)
        MODEL_SOURCE_SHA256 = sha256_file(MODEL_SOURCE_PATH)
        MODEL_MODULE = import_source_module(
            f"cs781_submission03_model_{MODEL_SOURCE_SHA256[:12]}", MODEL_SOURCE_PATH
        )
        if not callable(getattr(MODEL_MODULE, "build_model", None)):
            raise AttributeError("Student module must define build_model(config)")
        if STUDENT_MODEL_CONFIG_OVERRIDE is not None:
            MODEL_CONFIG = dict(STUDENT_MODEL_CONFIG_OVERRIDE)
        elif callable(getattr(MODEL_MODULE, "make_model_config", None)):
            MODEL_CONFIG = dict(MODEL_MODULE.make_model_config(int(TOKENIZER_META["vocab_size"])))
        else:
            candidate = getattr(MODEL_MODULE, "MODEL_CONFIG", None)
            if not isinstance(candidate, Mapping):
                raise RuntimeError("Provide STUDENT_MODEL_CONFIG_OVERRIDE, make_model_config, or MODEL_CONFIG")
            MODEL_CONFIG = dict(candidate)
        if int(MODEL_CONFIG.get("vocab_size", -1)) != int(TOKENIZER_META["vocab_size"]):
            raise ValueError("Model vocabulary does not match the frozen tokenizer")
        if int(MODEL_CONFIG.get("max_sequence_length", 0)) <= 2:
            raise ValueError("MODEL_CONFIG must declare max_sequence_length > 2")
        RECONSTRUCT_MODEL = MODEL_MODULE.build_model
        MODEL = RECONSTRUCT_MODEL(MODEL_CONFIG)
        MODEL_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
        MODEL_REPORT = validate_hybrid_contract(MODEL, TOKENIZER_META["vocab_size"], MODEL_DEVICE)
        MODEL_REPORT.update({
            "architecture_family": "single-stack prefix hybrid",
            "model_source_sha256": MODEL_SOURCE_SHA256,
            "model_config": MODEL_CONFIG,
        })
        atomic_write_json(DIRS["manifests"] / "model_validation.json", MODEL_REPORT)
        MODEL.to("cpu")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print("Student model validation:", json.dumps(MODEL_REPORT, indent=2))
    else:
        MODEL_REPORT = None
        print("STUDENT ACTION REQUIRED: attach and select your independently authored prefix_hybrid_lm.py")
    """,
)

set_markdown(nb, 25, "## 9. Disposable infrastructure and student-integration smoke tests")

set_code(
    nb,
    27,
    r"""
    def validate_continuation_causality(model, vocab_size: int, device: str) -> None:
        model.eval(); prefix = 5
        first = torch.randint(0, vocab_size, (1, 12), dtype=torch.long, device=device)
        second = first.clone(); second[:, 9:] = torch.randint(0, vocab_size, (1, 3), device=device)
        mask = torch.ones_like(first, dtype=torch.bool)
        lengths = torch.tensor([prefix], dtype=torch.long, device=device)
        with torch.no_grad():
            a = hybrid_logits(model, first, mask, lengths)[:, :9]
            b = hybrid_logits(model, second, mask, lengths)[:, :9]
        if not torch.allclose(a, b, atol=1e-5, rtol=1e-5):
            raise AssertionError("Future continuation tokens leaked into earlier logits")

    def validate_bidirectional_prefix(model, vocab_size: int, device: str) -> None:
        model.eval(); prefix = 6
        first = torch.randint(0, vocab_size, (1, 11), dtype=torch.long, device=device)
        second = first.clone(); second[:, prefix - 1] = (second[:, prefix - 1] + 1) % vocab_size
        mask = torch.ones_like(first, dtype=torch.bool)
        lengths = torch.tensor([prefix], dtype=torch.long, device=device)
        with torch.no_grad():
            a = hybrid_logits(model, first, mask, lengths)[:, 0]
            b = hybrid_logits(model, second, mask, lengths)[:, 0]
        if torch.allclose(a, b, atol=1e-7, rtol=1e-7):
            raise AssertionError("Earlier prefix representation is insensitive to a later prefix token")

    def validate_prefix_conditioning(model, vocab_size: int, device: str) -> None:
        model.eval(); prefix = 6
        first = torch.randint(0, vocab_size, (1, 11), dtype=torch.long, device=device)
        second = first.clone(); second[:, :prefix] = torch.randint(0, vocab_size, (1, prefix), device=device)
        mask = torch.ones_like(first, dtype=torch.bool)
        lengths = torch.tensor([prefix], dtype=torch.long, device=device)
        with torch.no_grad():
            a = hybrid_logits(model, first, mask, lengths)[:, prefix:]
            b = hybrid_logits(model, second, mask, lengths)[:, prefix:]
        if torch.allclose(a, b, atol=1e-7, rtol=1e-7):
            raise AssertionError("Continuation logits are insensitive to the prompt prefix")

    def run_student_integration_smoke_test(model, tokenizer, model_config, reconstruct_model, device=None):
        device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        meta = tokenizer_metadata(tokenizer, DIRS["tokenizer"])
        report = validate_hybrid_contract(model, meta["vocab_size"], device)
        validate_continuation_causality(model, meta["vocab_size"], device)
        report["continuation_future_leakage"] = "PASS"
        validate_bidirectional_prefix(model, meta["vocab_size"], device)
        report["bidirectional_prefix"] = "PASS"
        validate_prefix_conditioning(model, meta["vocab_size"], device)
        report["prefix_conditioning"] = "PASS"
        probe = tokenizer_encode(tokenizer, "A constitutional court considered the appeal.")
        if not probe or min(probe) < 0 or max(probe) >= meta["vocab_size"]:
            raise AssertionError("Tokenizer emitted an out-of-vocabulary ID")
        report["token_id_range"] = "PASS"
        limit = int(model_config["max_sequence_length"])
        try:
            too_long = torch.zeros((1, limit + 1), dtype=torch.long, device=device)
            hybrid_logits(model, too_long, torch.ones_like(too_long, dtype=torch.bool),
                          torch.tensor([limit], dtype=torch.long, device=device))
            raise AssertionError("Context limit not enforced")
        except (ValueError, RuntimeError):
            report["context_limit"] = "PASS"
        clone = reconstruct_model(model_config)
        clone.load_state_dict(model.state_dict(), strict=True)
        report["reconstruction"] = "PASS"; del clone
        model.to("cpu")
        if str(device).startswith("cuda"): torch.cuda.empty_cache()
        atomic_write_json(DIRS["manifests"] / "student_integration_smoke_test.json", report)
        return report

    if MODEL is not None and TOKENIZER_META is not None:
        INTEGRATION_SMOKE_RESULTS = run_student_integration_smoke_test(
            MODEL, tokenizer, MODEL_CONFIG, RECONSTRUCT_MODEL
        )
        print(json.dumps(INTEGRATION_SMOKE_RESULTS, indent=2))
    else:
        INTEGRATION_SMOKE_RESULTS = None
        print("Student prefix-hybrid integration smoke test is waiting for model/tokenizer registration.")
    """,
)

set_markdown(nb, 28, "## 10. Resumable prefix-hybrid continuation training")

set_code(
    nb,
    29,
    r"""
    # STUDENT-OWNED: fill every None before the GPU smoke test. Freeze before checkpoint 1.
    TRAINING_CONFIG = {
        "experiment_id": CFG.experiment_id,
        "objective": "target-token cross-entropy under a bidirectional-prefix/causal-continuation mask",
        "optimizer": "AdamW",
        "learning_rate": None,
        "betas": None,
        "epsilon": None,
        "weight_decay": None,
        "schedule": "linear_warmup_cosine_decay",
        "warmup_steps": None,
        "min_lr_ratio": None,
        "batch_size": None,
        "gradient_accumulation_steps": None,
        "source_length": None,
        "target_length": None,
        "max_steps": None,
        "samples_per_epoch": None,
        "gradient_clip_norm": None,
        "gradient_checkpointing": None,
        "validation_interval": None,
        "validation_sequences_per_source": None,
        "checkpoint_interval": 250,
        "recovery_interval": 1000,
        "num_workers": 0,
        "source_mixture": dict(CFG.source_mixing_ratios),
        "source_mixture_unit": "document draws, resampled deterministically per epoch",
        "seed": CFG.seed,
        "precision_policy": "bf16 if supported; otherwise fp16+GradScaler on CUDA; fp32 on CPU",
    }

    def validate_student_training_config(config: Mapping[str, Any]) -> dict[str, Any]:
        required = (
            "learning_rate", "betas", "epsilon", "weight_decay", "warmup_steps", "min_lr_ratio",
            "batch_size", "gradient_accumulation_steps", "source_length", "target_length",
            "max_steps", "samples_per_epoch", "gradient_clip_norm", "gradient_checkpointing",
            "validation_interval", "validation_sequences_per_source", "checkpoint_interval",
            "recovery_interval",
        )
        missing = [key for key in required if config.get(key) is None]
        if missing: raise RuntimeError(f"Student training configuration is incomplete: {missing}")
        if config.get("optimizer") != "AdamW": raise ValueError("This infrastructure validates AdamW only")
        if config.get("schedule") != "linear_warmup_cosine_decay":
            raise ValueError("This infrastructure validates linear_warmup_cosine_decay only")
        positive = (
            "learning_rate", "epsilon", "batch_size", "gradient_accumulation_steps", "source_length",
            "target_length", "max_steps", "samples_per_epoch", "gradient_clip_norm",
            "validation_interval", "validation_sequences_per_source", "checkpoint_interval", "recovery_interval",
        )
        if any(float(config[key]) <= 0 for key in positive): raise ValueError("Positive quantities must be > 0")
        if int(config["num_workers"]) != 0:
            raise ValueError("num_workers must remain 0 for exact sampler restoration")
        if len(config["betas"]) != 2 or not all(0 <= float(value) < 1 for value in config["betas"]):
            raise ValueError("AdamW betas must contain two values in [0, 1)")
        if float(config["weight_decay"]) < 0: raise ValueError("weight_decay must be non-negative")
        if not 0 <= float(config["min_lr_ratio"]) <= 1: raise ValueError("min_lr_ratio must be in [0, 1]")
        if int(config["warmup_steps"]) >= int(config["max_steps"]):
            raise ValueError("warmup_steps must be smaller than max_steps")
        if MODEL_CONFIG is not None:
            total = int(config["source_length"]) + int(config["target_length"])
            if total > int(MODEL_CONFIG["max_sequence_length"]):
                raise ValueError("source_length + target_length exceeds model max_sequence_length")
        return dict(config)

    def save_or_verify_json(path: Path, value: Any) -> None:
        path = Path(path)
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != value:
                raise RuntimeError(f"Existing immutable manifest differs: {path}")
        else:
            atomic_write_json(path, value)

    def append_event(path: Path, event: Mapping[str, Any]) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as handle:
            handle.write((json.dumps(dict(event), ensure_ascii=False) + "\n").encode("utf-8"))
            handle.flush(); os.fsync(handle.fileno())

    class PrefixHybridDataset(Dataset):
        def __init__(self, documents, tokenizer, ratios, samples, source_length, target_length, seed, cache_documents=128):
            self.documents = list(documents); self.tokenizer = tokenizer; self.ratios = dict(ratios)
            self.samples = int(samples); self.source_length = int(source_length); self.target_length = int(target_length)
            self.seed = int(seed); self.epoch = 0; self.cache_documents = int(cache_documents); self.cache = {}
            self.pad_id = getattr(tokenizer, "pad_token_id", None)
            self.bos_id = getattr(tokenizer, "bos_token_id", None)
            self.eos_id = getattr(tokenizer, "eos_token_id", None)
            if None in (self.pad_id, self.bos_id, self.eos_id):
                raise ValueError("Prefix-hybrid training requires pad, bos, and eos IDs")
            if len({int(self.pad_id), int(self.bos_id), int(self.eos_id)}) != 3:
                raise ValueError("pad, bos, and eos IDs must be distinct")
            self.sequence_length = self.source_length + self.target_length
            self._resample()
        def _resample(self):
            self.document_indices = source_sampling_indices(
                self.documents, self.ratios, self.samples, self.seed + self.epoch
            )
        def __len__(self): return self.samples
        def set_epoch(self, epoch: int): self.epoch = int(epoch); self._resample()
        def _tokens(self, document_index: int) -> list[int]:
            if document_index not in self.cache:
                if len(self.cache) >= self.cache_documents: self.cache.pop(next(iter(self.cache)))
                self.cache[document_index] = tokenizer_encode(
                    self.tokenizer, document_text(self.documents[document_index])
                )
            return self.cache[document_index]
        def __getitem__(self, index: int):
            document_index = self.document_indices[int(index)]
            document = self.documents[document_index]
            ids = self._tokens(document_index)
            needed = self.source_length + self.target_length
            start = 0
            if len(ids) >= needed:
                rng = random.Random(self.seed + self.epoch * len(self) + int(index))
                start = rng.randrange(0, len(ids) - needed + 1)
                source = list(ids[start:start + self.source_length])
                target = list(ids[start + self.source_length:start + needed])
            elif len(ids) >= 2:
                split = min(self.source_length, max(1, len(ids) - 1))
                source = list(ids[:split]); target = list(ids[split:split + self.target_length])
            else:
                source = list(ids) or [int(self.eos_id)]; target = []
            if len(target) < self.target_length: target = [*target, int(self.eos_id)]
            target = target[:self.target_length]
            prefix_length = len(source)
            valid_length = prefix_length + len(target)
            input_ids = [*source, int(self.bos_id), *target[:-1]]
            labels = [-100] * prefix_length + target
            input_ids = (input_ids + [int(self.pad_id)] * self.sequence_length)[:self.sequence_length]
            labels = (labels + [-100] * self.sequence_length)[:self.sequence_length]
            attention_mask = [True] * valid_length + [False] * (self.sequence_length - valid_length)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
                "prefix_lengths": torch.tensor(prefix_length, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "document_index": document_index,
                "document_source": document.source,
                "document_uid": document.uid,
                "window_start": start,
                "processed_tokens": prefix_length + len(target),
            }

    def build_fixed_hybrid_validation_rows(documents, tokenizer, config):
        rows, manifest = [], []
        per_source = int(config["validation_sequences_per_source"])
        for source in sorted({document.source for document in documents}):
            source_documents = [document for document in documents if document.source == source]
            ordered = _stable_document_order(source_documents, int(config["seed"]) + 10_000)
            for local_index, document in enumerate(ordered):
                if len([row for row in manifest if row["source"] == source]) >= per_source: break
                dataset = PrefixHybridDataset(
                    [document], tokenizer, {source: 1.0}, 1,
                    int(config["source_length"]), int(config["target_length"]),
                    int(config["seed"]) + 10_000 + local_index,
                )
                row = dataset[0]
                if int((row["labels"] != -100).sum()) == 0: continue
                rows.append(row)
                manifest.append({
                    "source": source,
                    "document_uid": document.uid,
                    "window_start": int(row["window_start"]),
                    "prefix_tokens": int(row["prefix_lengths"]),
                    "target_tokens": int((row["labels"] != -100).sum()),
                })
        expected = per_source * len({document.source for document in documents})
        if len(rows) != expected:
            raise RuntimeError(f"Could create only {len(rows)} fixed validation rows; expected {expected}")
        return rows, manifest

    def choose_precision(device: str):
        if not str(device).startswith("cuda"): return None, "fp32", False
        if torch.cuda.is_bf16_supported(): return torch.bfloat16, "bf16", False
        return torch.float16, "fp16", True

    def make_scaler(enabled: bool):
        try: return torch.amp.GradScaler("cuda", enabled=enabled)
        except (AttributeError, TypeError): return torch.cuda.amp.GradScaler(enabled=enabled)

    def scaler_update_succeeded(before: float, after: float) -> bool:
        return float(after) >= float(before)

    def learning_rate_multiplier(step: int, max_steps: int, warmup_steps: int, min_ratio: float) -> float:
        if step < warmup_steps: return max(step, 1) / max(warmup_steps, 1)
        progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
        return float(min_ratio) + (1.0 - float(min_ratio)) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def atomic_copy(source: Path, destination: Path) -> None:
        source, destination = Path(source), Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=destination.suffix, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            shutil.copyfile(source, temporary)
            if sha256_file(source) != sha256_file(temporary): raise IOError("Atomic copy verification failed")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def save_inference_checkpoint(path, model, model_config, tokenizer_meta, dataset_fp, step, validation_loss):
        state = {
            "format": "phase1_submission03_prefix_hybrid_inference_v1",
            "created_utc": utc_now(), "model_state": model.state_dict(),
            "model_config": dict(model_config), "model_source_sha256": MODEL_SOURCE_SHA256,
            "tokenizer_fingerprint": tokenizer_meta["fingerprint"], "dataset_fingerprint": dataset_fp,
            "step": int(step), "validation_loss": None if validation_loss is None else float(validation_loss),
        }
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            torch.save(state, temporary)
            check = torch.load(temporary, map_location="cpu", weights_only=False)
            if check.get("format") != state["format"] or check.get("model_source_sha256") != MODEL_SOURCE_SHA256:
                raise RuntimeError("Inference checkpoint readback failed")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return {"path": str(path), "sha256": sha256_file(path), "step": int(step)}

    def export_training_recovery_snapshot(root: Path) -> Path:
        root = Path(root).resolve(); output = DIRS["recovery"] / "training_recovery.latest.zip"
        include = []
        for relative in ("model", "tokenizer", "manifests", "histories", "checkpoints"):
            include.extend(path for path in (root / relative).rglob("*") if path.is_file())
        include = sorted({path.resolve() for path in include if path.resolve() != output.resolve()})
        manifest = {
            "created_utc": utc_now(),
            "files": [
                {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                 "sha256": sha256_file(path)}
                for path in include if path.name != "recovery_integrity.json"
            ],
        }
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
                for item in manifest["files"]: archive.write(root / item["path"], item["path"])
                archive.writestr("manifests/recovery_integrity.json", json.dumps(manifest, indent=2) + "\n")
            verify_recovery_archive(temporary)
            os.replace(temporary, output)
            verify_recovery_archive(output)
        finally:
            temporary.unlink(missing_ok=True)
        atomic_write_json(DIRS["recovery"] / "training_recovery.latest.json", {
            "created_utc": utc_now(), "path": str(output),
            "sha256": sha256_file(output), "bytes": output.stat().st_size,
        })
        return output

    @torch.no_grad()
    def evaluate_hybrid_loss(model, loader, device, amp_dtype):
        was_training = model.training; model.eval(); loss_sum = 0.0; token_count = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prefix_lengths = batch["prefix_lengths"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type=torch.device(device).type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = hybrid_logits(model, input_ids, attention_mask, prefix_lengths)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                    ignore_index=-100, reduction="sum",
                )
            loss_sum += float(loss.cpu()); token_count += int((labels != -100).sum())
        model.train(was_training)
        if not token_count: raise ValueError("Validation loader contained no targets")
        return loss_sum / token_count

    def train_submission03(model, tokenizer, train_docs, val_docs, model_config, tokenizer_meta,
                           *, device="cuda:0", resume=False):
        cfg = validate_student_training_config(TRAINING_CONFIG)
        if not CFG.student_configuration_confirmed or not CFG.student_owned_architecture_attestation:
            raise RuntimeError("Confirm and attest the student-owned model/configuration first")
        latest = DIRS["checkpoints"] / "checkpoint.latest.pt"
        if resume and not latest.exists():
            raise FileNotFoundError("resume_train requires checkpoint.latest.pt; refusing step-zero restart")
        if not resume and latest.exists():
            raise RuntimeError("fresh_train found checkpoint.latest.pt; use resume_train or a new workspace")
        save_or_verify_json(DIRS["manifests"] / "training_config.json", cfg)
        split_meta = json.loads((DIRS["manifests"] / "document_split.json").read_text(encoding="utf-8"))
        dataset_fp = split_meta["dataset_fingerprint"]
        ratios = validate_source_mixture(cfg["source_mixture"], (document.source for document in train_docs))
        dataset = PrefixHybridDataset(
            train_docs, tokenizer, ratios, cfg["samples_per_epoch"], cfg["source_length"],
            cfg["target_length"], cfg["seed"],
        )
        sampler = StatefulRandomSampler(dataset, seed=cfg["seed"])
        loader = DataLoader(
            dataset, batch_size=cfg["batch_size"], sampler=sampler, num_workers=0,
            pin_memory=str(device).startswith("cuda"), drop_last=True,
        )
        validation_rows, validation_manifest = build_fixed_hybrid_validation_rows(val_docs, tokenizer, cfg)
        save_or_verify_json(DIRS["manifests"] / "validation_sequence_manifest.json", {
            "protocol": "fixed held-out prefix/continuation windows", "rows": validation_manifest,
        })
        validation_loader = DataLoader(validation_rows, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
        model = model.to(device)
        if hasattr(model, "set_gradient_checkpointing"):
            model.set_gradient_checkpointing(bool(cfg["gradient_checkpointing"]))
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["learning_rate"], betas=tuple(cfg["betas"]),
            eps=cfg["epsilon"], weight_decay=cfg["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: learning_rate_multiplier(
                step, cfg["max_steps"], cfg["warmup_steps"], cfg["min_lr_ratio"]
            ),
        )
        amp_dtype, actual_precision, use_scaler = choose_precision(device)
        scaler = make_scaler(use_scaler); checkpoint_scaler = scaler if use_scaler else None
        history_path = DIRS["histories"] / "training.jsonl"; history = JsonlHistory(history_path)
        progress = ProgressState(cfg["experiment_id"], device, cfg["max_steps"])
        validation_history = []
        if resume:
            preview = torch.load(latest, map_location="cpu", weights_only=False)
            if preview.get("training_config") != cfg: raise RuntimeError("Checkpoint training config differs")
            if preview.get("model_source_sha256") != MODEL_SOURCE_SHA256:
                raise RuntimeError("Checkpoint model source differs")
            reconcile_history_to_checkpoint(latest, history_path, apply=True)
            state = restore_checkpoint(
                latest, model=model, optimizer=optimizer, scheduler=scheduler,
                scaler=checkpoint_scaler, sampler=sampler,
                expected_tokenizer_fingerprint=tokenizer_meta["fingerprint"],
                expected_dataset_fingerprint=dataset_fp,
                expected_experiment_id=cfg["experiment_id"],
                expected_model_config=model_config, history_path=history_path,
                device=device, expected_model_source_sha256=MODEL_SOURCE_SHA256,
            )
            progress = ProgressState(**state["progress"])
            validation_history = list(state.get("validation_history", []))
            dataset.set_epoch(sampler.epoch)
            print("Restored optimizer step:", progress.step)
        elif history.records():
            raise RuntimeError("Training history exists without a matching fresh/resume decision")
        reporter = ProgressReporter(progress); iterator = iter(loader)
        best_loss = min((float(row["loss"]) for row in validation_history), default=math.inf)
        try:
            while reporter.state.step < cfg["max_steps"]:
                model.train(); optimizer.zero_grad(set_to_none=True)
                microbatches, valid_targets, processed_tokens = [], 0, 0
                for _ in range(cfg["gradient_accumulation_steps"]):
                    try: batch = next(iterator)
                    except StopIteration:
                        sampler.next_epoch(); dataset.set_epoch(sampler.epoch)
                        iterator = iter(loader); batch = next(iterator)
                    valid_targets += int((batch["labels"] != -100).sum())
                    processed_tokens += int(batch["processed_tokens"].sum())
                    microbatches.append(batch)
                if valid_targets <= 0: raise ValueError("Accumulation window has no target tokens")
                numerator = 0.0
                for batch in microbatches:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    prefix_lengths = batch["prefix_lengths"].to(device)
                    labels = batch["labels"].to(device)
                    with torch.autocast(
                        device_type=torch.device(device).type, dtype=amp_dtype,
                        enabled=amp_dtype is not None,
                    ):
                        logits = hybrid_logits(model, input_ids, attention_mask, prefix_lengths)
                        loss_sum = torch.nn.functional.cross_entropy(
                            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                            ignore_index=-100, reduction="sum",
                        )
                        backward_loss = loss_sum / valid_targets
                    if not torch.isfinite(loss_sum): raise FloatingPointError("Non-finite training loss")
                    (scaler.scale(backward_loss) if use_scaler else backward_loss).backward()
                    numerator += float(loss_sum.detach().cpu())
                if use_scaler: scaler.unscale_(optimizer)
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["gradient_clip_norm"]
                ).cpu())
                learning_rate = float(optimizer.param_groups[0]["lr"])
                if use_scaler:
                    before = float(scaler.get_scale()); scaler.step(optimizer); scaler.update()
                    if not scaler_update_succeeded(before, scaler.get_scale()):
                        append_event(DIRS["histories"] / "overflow_events.jsonl", {
                            "attempted_step": reporter.state.step + 1, "epoch": sampler.epoch,
                            "scale_before": before, "scale_after": float(scaler.get_scale()),
                            "created_utc": utc_now(),
                        })
                        continue
                else:
                    optimizer.step()
                scheduler.step(); step = reporter.state.step + 1
                total_tokens = reporter.state.tokens_processed + processed_tokens
                mean_loss = numerator / valid_targets
                validation_loss = None; improved = False
                if step % cfg["validation_interval"] == 0 or step == cfg["max_steps"]:
                    validation_loss = evaluate_hybrid_loss(model, validation_loader, device, amp_dtype)
                    validation_history.append({
                        "step": step, "loss": validation_loss,
                        "perplexity": math.exp(validation_loss) if validation_loss < 709 else math.inf,
                    })
                    improved = validation_loss < best_loss; best_loss = min(best_loss, validation_loss)
                reporter.update(
                    step=step, epoch=sampler.epoch, loss=mean_loss, learning_rate=learning_rate,
                    tokens_processed=total_tokens, gradient_norm=gradient_norm,
                )
                history.append({
                    "step": step, "epoch": sampler.epoch, "loss": mean_loss,
                    "validation_loss": validation_loss, "learning_rate": learning_rate,
                    "gradient_norm": gradient_norm, "tokens_processed": total_tokens,
                    "tokens_per_second": total_tokens / max(reporter.state.cumulative_seconds, 1e-9),
                    "optimizer_step_seconds": reporter.state.cumulative_seconds / max(step, 1),
                    "gpu_memory_gib": (
                        torch.cuda.max_memory_allocated(torch.device(device)) / 2**30
                        if str(device).startswith("cuda") else None
                    ),
                    "precision": actual_precision,
                    "checkpoint_step": reporter.state.last_checkpoint_step,
                    "created_utc": utc_now(),
                })
                if step % cfg["checkpoint_interval"] == 0 or improved or step == cfg["max_steps"]:
                    reporter.state.last_checkpoint_step = step
                    save_checkpoint_atomic(
                        latest, model=model, optimizer=optimizer, scheduler=scheduler,
                        scaler=checkpoint_scaler, sampler=sampler, progress=asdict(reporter.state),
                        model_config=model_config, tokenizer_fingerprint=tokenizer_meta["fingerprint"],
                        dataset_fingerprint_value=dataset_fp, training_config=cfg,
                        history_path=history_path, validation_history=validation_history,
                        experiment_id=cfg["experiment_id"], device=device,
                        model_source_sha256=MODEL_SOURCE_SHA256,
                    )
                    if improved:
                        save_inference_checkpoint(
                            DIRS["checkpoints"] / "selected_model.pt", model, model_config,
                            tokenizer_meta, dataset_fp, step, validation_loss,
                        )
                    if step % cfg["recovery_interval"] == 0:
                        export_training_recovery_snapshot(ROOT)
            final_loss = validation_history[-1]["loss"] if validation_history else None
            save_inference_checkpoint(
                DIRS["checkpoints"] / "model_final.pt", model, model_config,
                tokenizer_meta, dataset_fp, reporter.state.step, final_loss,
            )
            if not (DIRS["checkpoints"] / "selected_model.pt").exists():
                atomic_copy(DIRS["checkpoints"] / "model_final.pt", DIRS["checkpoints"] / "selected_model.pt")
            export_training_recovery_snapshot(ROOT)
        finally:
            reporter.close()
        result = {
            "completed_step": reporter.state.step,
            "tokens_processed": reporter.state.tokens_processed,
            "best_validation_loss": best_loss,
            "actual_precision": actual_precision,
            "latest_checkpoint": str(latest),
            "selected_checkpoint": str(DIRS["checkpoints"] / "selected_model.pt"),
        }
        atomic_write_json(DIRS["manifests"] / "training_result.json", result)
        return result

    def run_gpu_training_smoke_test(device="cuda:0"):
        if not torch.cuda.is_available(): raise RuntimeError("Representative smoke test requires CUDA")
        cfg = validate_student_training_config(TRAINING_CONFIG)
        if MODEL is None or tokenizer is None or not TRAIN_DOCS or not VAL_DOCS:
            raise RuntimeError("Model, tokenizer, and bounded corpus cache are required")
        validation_rows, validation_manifest = build_fixed_hybrid_validation_rows(VAL_DOCS, tokenizer, cfg)
        save_or_verify_json(DIRS["manifests"] / "validation_sequence_manifest.json", {
            "protocol": "fixed held-out prefix/continuation windows", "rows": validation_manifest,
        })
        tiny = PrefixHybridDataset(
            TRAIN_DOCS, tokenizer, cfg["source_mixture"], max(2, cfg["batch_size"] * 2),
            cfg["source_length"], cfg["target_length"], cfg["seed"],
        )
        sampler = StatefulRandomSampler(tiny, seed=cfg["seed"])
        batch = next(iter(DataLoader(
            tiny, batch_size=cfg["batch_size"], sampler=sampler, num_workers=0, drop_last=True
        )))
        smoke_model = RECONSTRUCT_MODEL(MODEL_CONFIG).to(device)
        if hasattr(smoke_model, "set_gradient_checkpointing"):
            smoke_model.set_gradient_checkpointing(bool(cfg["gradient_checkpointing"]))
        optimizer = torch.optim.AdamW(
            smoke_model.parameters(), lr=cfg["learning_rate"], betas=tuple(cfg["betas"]),
            eps=cfg["epsilon"], weight_decay=cfg["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: learning_rate_multiplier(step, 10, 1, cfg["min_lr_ratio"])
        )
        amp_dtype, precision, use_scaler = choose_precision(device); scaler = make_scaler(use_scaler)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        prefix_lengths = batch["prefix_lengths"].to(device)
        labels = batch["labels"].to(device)
        torch.cuda.reset_peak_memory_stats(torch.device(device)); optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = hybrid_logits(smoke_model, input_ids, attention_mask, prefix_lengths)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
            )
        (scaler.scale(loss) if use_scaler else loss).backward()
        if use_scaler: scaler.unscale_(optimizer)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            smoke_model.parameters(), cfg["gradient_clip_norm"]
        ).cpu())
        if use_scaler:
            before = float(scaler.get_scale()); scaler.step(optimizer); scaler.update()
            if not scaler_update_succeeded(before, scaler.get_scale()):
                raise RuntimeError("FP16 smoke update overflowed")
        else:
            optimizer.step()
        scheduler.step(); loss_value = float(loss.detach().cpu())
        split_meta = json.loads((DIRS["manifests"] / "document_split.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name); history = JsonlHistory(temporary / "history.jsonl")
            history.append({"step": 1, "loss": loss_value})
            checkpoint = temporary / "smoke.pt"
            progress = ProgressState(
                "submission03_gpu_smoke", device, 1, step=1,
                tokens_processed=int(batch["processed_tokens"].sum()), last_checkpoint_step=1,
            )
            save_checkpoint_atomic(
                checkpoint, model=smoke_model, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler if use_scaler else None, sampler=sampler,
                progress=asdict(progress), model_config=MODEL_CONFIG,
                tokenizer_fingerprint=TOKENIZER_META["fingerprint"],
                dataset_fingerprint_value=split_meta["dataset_fingerprint"],
                training_config={"smoke": True}, history_path=history.path,
                validation_history=[], experiment_id="submission03_gpu_smoke",
                device=device, model_source_sha256=MODEL_SOURCE_SHA256,
            )
            del logits, loss; torch.cuda.empty_cache()
            clone = RECONSTRUCT_MODEL(MODEL_CONFIG)
            clone_optimizer = torch.optim.AdamW(
                clone.parameters(), lr=cfg["learning_rate"], betas=tuple(cfg["betas"]),
                eps=cfg["epsilon"], weight_decay=cfg["weight_decay"],
            )
            clone_scheduler = torch.optim.lr_scheduler.LambdaLR(
                clone_optimizer, lambda step: learning_rate_multiplier(step, 10, 1, cfg["min_lr_ratio"])
            )
            clone_scaler = make_scaler(use_scaler)
            clone_sampler = StatefulRandomSampler(tiny, seed=cfg["seed"])
            restored = restore_checkpoint(
                checkpoint, model=clone, optimizer=clone_optimizer, scheduler=clone_scheduler,
                scaler=clone_scaler if use_scaler else None, sampler=clone_sampler,
                expected_tokenizer_fingerprint=TOKENIZER_META["fingerprint"],
                expected_dataset_fingerprint=split_meta["dataset_fingerprint"],
                expected_experiment_id="submission03_gpu_smoke", expected_model_config=MODEL_CONFIG,
                history_path=history.path, device=None,
                expected_model_source_sha256=MODEL_SOURCE_SHA256,
            )
            if restored["progress"]["step"] != 1 or not clone_optimizer.state:
                raise AssertionError("Checkpoint did not restore model, optimizer, or progress")
        report = {
            "status": "PASS", "loss": loss_value, "gradient_norm": gradient_norm,
            "precision": precision,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(torch.device(device)) / 2**30,
            "source_length": cfg["source_length"], "target_length": cfg["target_length"],
            "batch_size": cfg["batch_size"], "checkpoint_restore": "PASS", "created_utc": utc_now(),
        }
        atomic_write_json(DIRS["manifests"] / "gpu_training_smoke_test.json", report)
        del smoke_model, optimizer, clone, clone_optimizer; torch.cuda.empty_cache()
        return report

    RUN_GPU_TRAINING_SMOKE_TEST = False
    GPU_TRAINING_SMOKE_RESULT = (
        run_gpu_training_smoke_test("cuda:0") if RUN_GPU_TRAINING_SMOKE_TEST else None
    )

    def preflight_full_training():
        problems = []
        if CFG.execution_mode not in {"fresh_train", "resume_train"}:
            problems.append("execution_mode is not a training mode")
        if not CFG.full_training_authorized: problems.append("full_training_authorized is False")
        if not CFG.student_owned_architecture_attestation: problems.append("architecture attestation is False")
        if not CFG.student_configuration_confirmed: problems.append("configuration confirmation is False")
        if not CFG.trained_from_scratch_attestation: problems.append("scratch-training attestation is False")
        architecture_text = json.dumps(CFG.architecture_record, ensure_ascii=False).upper()
        if "REQUIRED BEFORE TRAINING" in architecture_text:
            problems.append("architecture_record still contains required student placeholders")
        if CFG.innovation_hypothesis.startswith("STUDENT MUST"):
            problems.append("innovation_hypothesis is still a placeholder")
        if CFG.ablation_control.startswith("STUDENT MUST"):
            problems.append("ablation_control is still a placeholder")
        try: validate_student_training_config(TRAINING_CONFIG)
        except Exception as exc: problems.append(str(exc))
        if MODEL is None or tokenizer is None: problems.append("model/tokenizer unavailable")
        if not TRAIN_DOCS or not VAL_DOCS: problems.append("bounded train/validation cache unavailable")
        for name in (
            "document_split.json", "model_validation.json", "student_integration_smoke_test.json",
            "gpu_training_smoke_test.json", "validation_sequence_manifest.json",
        ):
            if not (DIRS["manifests"] / name).exists(): problems.append(f"missing manifest {name}")
        smoke_path = DIRS["manifests"] / "gpu_training_smoke_test.json"
        if smoke_path.exists() and json.loads(smoke_path.read_text(encoding="utf-8")).get("status") != "PASS":
            problems.append("representative GPU training smoke test did not pass")
        if problems: raise RuntimeError("Full-training preflight failed:\n- " + "\n- ".join(problems))

    RUN_FULL_TRAINING = False
    if RUN_FULL_TRAINING:
        preflight_full_training()
        TRAINING_RESULT = train_submission03(
            MODEL, tokenizer, TRAIN_DOCS, VAL_DOCS, MODEL_CONFIG, TOKENIZER_META,
            device="cuda:0", resume=CFG.execution_mode == "resume_train",
        )
    else:
        TRAINING_RESULT = None
        print("Training is opt-in and blocked until the student core/config and GPU restore smoke test pass.")
    """,
)

# Both the final and the periodic training archive now use the same authenticated format.
nb.cells[24].source = nb.cells[24].source.replace(
    "return sorted(Path(input_root).rglob('phase1_submission03_recovery.zip')) if Path(input_root).exists() else []",
    "return sorted({*Path(input_root).rglob('phase1_submission03_recovery.zip'), "
    "*Path(input_root).rglob('training_recovery.latest.zip')}) if Path(input_root).exists() else []",
)

set_markdown(nb, 33, "## 12. Resumable prefix-hybrid generation and prediction management")

_generation_common = nb.cells[34].source.split("@torch.inference_mode()", 1)[0]
set_code(
    nb,
    34,
    _generation_common
    + textwrap.dedent(r"""
    @torch.inference_mode()
    def hybrid_continuation(model, tokenizer, prompt, device, *, policy, target_words=128, max_new_tokens=None):
        if max_new_tokens is None: raise ValueError("A declared max_new_tokens is required")
        max_sequence_length = int(MODEL_CONFIG.get("max_sequence_length", 0))
        if max_sequence_length <= 2: raise ValueError("MODEL_CONFIG must declare max_sequence_length")
        prompt_budget = max_sequence_length - int(max_new_tokens) - 1
        if prompt_budget < 1:
            raise ValueError("max_new_tokens leaves no room for a prompt prefix and BOS token")
        prompt_ids = tokenizer_encode(tokenizer, prompt)
        if not prompt_ids: raise ValueError("Prompt encoded to no tokens")
        prompt_ids = prompt_ids[-prompt_budget:]
        bos_id = getattr(tokenizer, "bos_token_id", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if bos_id is None: raise ValueError("Prefix-hybrid inference requires bos_token_id")
        forbidden = {int(value) for value in (bos_id, pad_id) if value is not None}
        generated = []; generator = torch.Generator(device=device).manual_seed(int(policy["seed"]))
        model = model.to(device); model.eval()
        for _ in range(int(max_new_tokens)):
            sequence = [*prompt_ids, int(bos_id), *generated]
            input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            prefix_lengths = torch.tensor([len(prompt_ids)], dtype=torch.long, device=device)
            logits = hybrid_logits(model, input_ids, attention_mask, prefix_lengths)[0, -1].float()
            if forbidden: logits[list(forbidden)] = -torch.inf
            next_id = select_next_token(logits, generated, policy, generator)
            generated.append(next_id)
            continuation = tokenizer_decode(tokenizer, generated)
            if len(words(continuation)) >= target_words + 1:
                result = exact_word_prefix(continuation, target_words)
                report = generation_quality_report(result)
                if report["unique_word_ratio"] < float(_policy_value(policy, "min_unique_word_ratio", 0.0)):
                    raise RuntimeError(f"Declared uniqueness threshold failed: {report}")
                if report["max_word_fraction"] > float(_policy_value(policy, "max_word_fraction", 1.0)):
                    raise RuntimeError(f"Declared repetition threshold failed: {report}")
                return result, report
            if eos_id is not None and next_id == int(eos_id):
                raise RuntimeError(f"Premature EOS after {len(words(continuation))} words")
        raise RuntimeError(f"Generation ended before {target_words} verified words")

    def load_test_records(path: Path, schema: Mapping[str, Any]) -> list[dict[str, str]]:
        id_field, prompt_field = schema["id_field"], schema["text_field"]
        result, seen = [], set()
        for index, row in enumerate(iter_json_records(path, id_field=id_field, text_field=prompt_field)):
            identifier = str(row[id_field]); prompt = normalize_text(row[prompt_field])
            if identifier in seen: raise ValueError(f"Duplicate official ID: {identifier}")
            seen.add(identifier); result.append({"id": identifier, "prompt": prompt})
        return result

    def test_records_fingerprint(records):
        digest = hashlib.sha256()
        for row in records:
            digest.update(f"{row['id']}\t{hashlib.sha256(row['prompt'].encode()).hexdigest()}\n".encode())
        return digest.hexdigest()
    """),
)

set_code(
    nb,
    35,
    r"""
    def load_selected_model_for_inference(checkpoint_path, tokenizer_meta, reconstruct_model, device):
        checkpoint_path = Path(checkpoint_path)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if state.get("tokenizer_fingerprint") != tokenizer_meta["fingerprint"]:
            raise RuntimeError("Selected checkpoint tokenizer mismatch")
        if state.get("model_source_sha256") != MODEL_SOURCE_SHA256:
            raise RuntimeError("Selected checkpoint model-source mismatch")
        restored = reconstruct_model(state["model_config"])
        restored.load_state_dict(state["model_state"], strict=True)
        return restored.to(device).eval(), state

    def generate_official_predictions(model, tokenizer, test_records, device, *, checkpoint_id, tokenizer_meta):
        policies = list(CFG.decoding.get("policies") or [])
        max_new_tokens = CFG.decoding.get("max_new_tokens")
        if not policies or max_new_tokens is None:
            raise RuntimeError("Declare ordered decoding policies and max_new_tokens before inference")
        names = [str(policy.get("name", "")) for policy in policies]
        if not all(names) or len(names) != len(set(names)):
            raise ValueError("Every decoding policy requires a unique nonempty name")
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", checkpoint_id)
        partial_path = DIRS["predictions"] / f"official_predictions.{safe}.partial.json"
        meta_path = partial_path.with_suffix(".meta.json")
        attempts_path = DIRS["predictions"] / f"generation_attempts.{safe}.jsonl"
        expected_meta = {
            "checkpoint_id": checkpoint_id,
            "checkpoint_sha256": sha256_file(DIRS["checkpoints"] / "selected_model.pt"),
            "model_source_sha256": MODEL_SOURCE_SHA256,
            "tokenizer_fingerprint": tokenizer_meta["fingerprint"],
            "test_fingerprint": test_records_fingerprint(test_records),
            "decoding": dict(CFG.decoding),
        }
        predictions, failures = {}, []
        if partial_path.exists():
            if not meta_path.exists() or json.loads(meta_path.read_text(encoding="utf-8")) != expected_meta:
                raise RuntimeError("Partial predictions do not match checkpoint/tokenizer/test/decoding provenance")
            predictions = json.loads(partial_path.read_text(encoding="utf-8"))
            expected_ids = {row["id"] for row in test_records}
            if set(predictions) - expected_ids: raise RuntimeError("Partial predictions contain foreign IDs")
            for identifier, value in predictions.items():
                if not isinstance(value, dict) or set(value) != {"text"}:
                    raise RuntimeError(f"Invalid cached prediction structure: {identifier}")
                if len(words(value["text"])) != int(CFG.decoding["target_words"]):
                    raise RuntimeError(f"Invalid cached word count: {identifier}")
        else:
            atomic_write_json(meta_path, expected_meta)
        for row in tqdm(test_records, desc="Official test generation"):
            if row["id"] in predictions: continue
            errors = []
            for policy_index, policy in enumerate(policies):
                started = time.time()
                try:
                    continuation, quality = hybrid_continuation(
                        model, tokenizer, row["prompt"], device, policy=policy,
                        target_words=int(CFG.decoding["target_words"]),
                        max_new_tokens=max_new_tokens,
                    )
                    append_event(attempts_path, {
                        "id": row["id"], "policy_index": policy_index, "policy": policy["name"],
                        "status": "success", "quality": quality,
                        "seconds": time.time() - started, "created_utc": utc_now(),
                    })
                    predictions[row["id"]] = {"text": continuation}
                    atomic_write_json(partial_path, predictions)
                    break
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"; errors.append(error)
                    append_event(attempts_path, {
                        "id": row["id"], "policy_index": policy_index, "policy": policy["name"],
                        "status": "failure", "error": error,
                        "seconds": time.time() - started, "created_utc": utc_now(),
                    })
            if row["id"] not in predictions: failures.append({"id": row["id"], "errors": errors})
        atomic_write_json(DIRS["predictions"] / "generation_failures.json", failures)
        if failures: raise RuntimeError(f"Generation failed for {len(failures)} examples; no ZIP will be created")
        provenance = {
            **expected_meta, "prediction_count": len(predictions),
            "partial_prediction_sha256": sha256_file(partial_path), "created_utc": utc_now(),
        }
        atomic_write_json(DIRS["predictions"] / "inference_provenance.json", provenance)
        return predictions, failures

    def build_local_generation_examples(documents, examples_per_source=4):
        examples = []
        for source in sorted({document.source for document in documents}):
            collected = 0
            for document in _stable_document_order(
                [item for item in documents if item.source == source], CFG.seed + 20_000
            ):
                units = words(document_text(document))
                if len(units) < 640: continue
                examples.append({
                    "id": f"{source}:{document.uid}",
                    "source": source,
                    "prompt": " ".join(units[:512]),
                    "reference": " ".join(units[512:640]),
                })
                collected += 1
                if collected >= int(examples_per_source): break
            if collected < int(examples_per_source):
                raise RuntimeError(f"Insufficient long held-out documents for local evaluation source {source}")
        return examples

    def evaluate_local_continuations(model, tokenizer, examples, evaluator, device, checkpoint_id):
        if evaluator is None: raise RuntimeError("Load and inspect the professor-supplied evaluator first")
        policies = list(CFG.decoding.get("policies") or [])
        if not policies: raise RuntimeError("Declare decoding policies before local evaluation")
        generated, references = [], []
        for row in tqdm(examples, desc="Held-out continuation evaluation"):
            last_error = None
            for policy in policies:
                try:
                    text, _ = hybrid_continuation(
                        model, tokenizer, row["prompt"], device, policy=policy,
                        target_words=int(CFG.decoding["target_words"]),
                        max_new_tokens=CFG.decoding["max_new_tokens"],
                    )
                    generated.append(text); references.append(row["reference"]); break
                except Exception as exc:
                    last_error = exc
            else:
                raise RuntimeError(f"All declared policies failed for {row['id']}: {last_error}")
        return run_official_evaluator(evaluator, generated, references, checkpoint_id)

    def run_phase1_inference_pipeline(device="cuda:0", run_local_evaluation=False):
        if MODEL_SOURCE_SHA256 is None or RECONSTRUCT_MODEL is None or TOKENIZER_META is None:
            raise RuntimeError("Restore tokenizer and authenticated student model source before inference")
        checkpoint_path = DIRS["checkpoints"] / "selected_model.pt"
        if not checkpoint_path.exists(): raise FileNotFoundError("selected_model.pt is required")
        selected_model, state = load_selected_model_for_inference(
            checkpoint_path, TOKENIZER_META, RECONSTRUCT_MODEL, device
        )
        checkpoint_id = f"selected-{sha256_file(checkpoint_path)[:12]}"
        local_evaluation = None
        if run_local_evaluation:
            if not VAL_DOCS: raise RuntimeError("Optional local evaluation requires held-out validation documents")
            examples = build_local_generation_examples(VAL_DOCS)
            atomic_write_json(DIRS["evaluations"] / "fixed_local_examples.json", examples)
            local_evaluation = evaluate_local_continuations(
                selected_model, tokenizer, examples, OFFICIAL_EVALUATOR, device, checkpoint_id
            )
        test_records = load_test_records(
            require_unique_file(DISCOVERED, CFG.test_file), SCHEMAS["test"]
        )
        predictions, failures = generate_official_predictions(
            selected_model, tokenizer, test_records, device,
            checkpoint_id=checkpoint_id, tokenizer_meta=TOKENIZER_META,
        )
        return selected_model, local_evaluation, test_records, predictions, failures

    RUN_INFERENCE = False
    RUN_OPTIONAL_LOCAL_EVALUATION = False
    if RUN_INFERENCE:
        MODEL, EVAL_RECORD, TEST_RECORDS, PREDICTIONS, FAILURES = run_phase1_inference_pipeline(
            "cuda:0", run_local_evaluation=RUN_OPTIONAL_LOCAL_EVALUATION
        )
    else:
        print("Inference is opt-in. In inference_only mode the training corpus is not loaded.")
    """,
)

set_markdown(
    nb,
    42,
    r"""
    ## Kaggle execution checklist for Submission 03

    1. Keep Submissions 01 and 02 backed up. This notebook writes only under `.../submission_03`.
    2. Attach the official four-source Phase 1 corpus/test data under `/kaggle/input`; confirm the schema
       report and that the official test file is excluded from training.
    3. Select and document a compute-bounded corpus cap. Enable `RUN_CORPUS_PREPROCESSING` to build or
       reopen the deterministic disk-backed cache; the original multi-GB corpus is not copied.
    4. Point `CFG.tokenizer_import_path` to the frozen course/Submission 01 BPE tokenizer for a controlled
       architecture comparison. Never change its mapping after checkpointing begins.
    5. Independently implement `prefix_hybrid_lm.py` using the contract in Section 6. Fill its exact numerical
       configuration and parameter derivation. The runtime count must be between 50M and 300M.
    6. Fill every `None` in `TRAINING_CONFIG`, write your hypothesis/rationale and matched controls, and only
       then set the student attestations truthfully. These decisions must not be delegated to an LLM.
    7. Run all CPU/GPU integration tests. Require bidirectional-prefix sensitivity, no future-continuation
       leakage, prefix conditioning, finite gradients, exact reconstruction, and checkpoint restore PASS.
    8. Use exactly one training mode. `fresh_train` refuses an existing checkpoint; `resume_train` requires
       one and prints the restored optimizer step. Never alter a checkpointed configuration silently.
    9. Download `training_recovery.latest.zip` every milestone. It now uses the same per-member SHA manifest
       as the restore helper and excludes the large bounded corpus cache.
    10. Time a small declared-policy inference sample before all official IDs. Prefix-hybrid generation
        recomputes the segment-aware stack and may be slow without a student-implemented compatible cache.
    11. In `inference_only`, restore only tokenizer, model source/config, selected checkpoint, and official test.
        Local evaluation is optional and independent of official prediction generation.
    12. Package only after every official ID has one valid 128-word continuation. The ZIP must contain only
        root-level `prediction.json`. Upload `submission.zip` to Codabench and record the real acceptance ID.

    Submission 03 is complete only after the trained checkpoint is verified, artifact validation passes,
    every official ID is valid, and Codabench acceptance is truthfully recorded.
    """,
)

# Keep all distributable code cells unexecuted.
for cell in nb.cells:
    if cell.cell_type == "code":
        cell.execution_count = None
        cell.outputs = []

nbformat.validate(nb)
for destination in (OUTPUT, PROJECT_COPY):
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, destination)
print(f"Wrote {OUTPUT} and project copy with {len(nb.cells)} cells")

