"""Build an isolated Submission 04 Kaggle notebook from the reviewed Submission 03 infrastructure.

The source notebook contains two one-off recovery cells from its completed run;
those cells are deliberately excluded. All executable outputs are cleared.
"""

from __future__ import annotations

import copy
import hashlib
import textwrap
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "CS781_Phase1_Submission03_Infrastructure_FIXED.ipynb"
OUTPUT = ROOT / "CS781_Phase1_Submission04_Infrastructure_FIXED.ipynb"
PROJECT_COPY = Path(__file__).resolve().parent / OUTPUT.name


def replace_once(source: str, old: str, new: str, description: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{description}: expected one anchor, found {count}")
    return source.replace(old, new)


def cell_code(text: str):
    return nbformat.v4.new_code_cell(textwrap.dedent(text).strip() + "\n")


def cell_markdown(text: str):
    return nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip() + "\n")


source_bytes = SOURCE.read_bytes()
nb = copy.deepcopy(nbformat.read(SOURCE, as_version=4))
if len(nb.cells) != 45:
    raise RuntimeError(f"Expected 45 reviewed Submission 03 cells, found {len(nb.cells)}")
nb.cells = [cell for index, cell in enumerate(nb.cells) if index not in (36, 37)]
for cell in nb.cells:
    cell.source = (
        cell.source.replace("phase1_submission03", "phase1_submission04")
        .replace("submission_03", "submission_04")
        .replace("Submission 03", "Submission 04")
        .replace("submission03", "submission04")
    )
    if cell.cell_type == "code":
        cell.execution_count = None
        cell.outputs = []

nb.metadata["cs781"] = {
    "phase": 1,
    "submission": 4,
    "purpose": "efficient causal Transformer Kaggle infrastructure",
    "source_notebook_sha256": hashlib.sha256(source_bytes).hexdigest(),
    "student_owned_model_required": True,
    "executed": False,
}

nb.cells[0] = cell_markdown("""
# CS781 Capstone — Phase 1, Submission 04
## Efficient causal legal language model experiment

This independent notebook keeps the official-data, BPE, two GPU training, checkpoint,
recovery, evaluation and packaging infrastructure used in Submission 03. The model
contract changes to a causal **local/global attention Transformer** with grouped-query
attention and rotary position embeddings (RoPE). BPE maps text to token IDs; RoPE
encodes token positions inside attention. It is not a tokenizer.

This architectural experiment must differ in information flow and attention cost from
the dense globally causal Submission 01 and the prefix-hybrid Submission 03. A mere
change of layer count, heads, learning rate or generation policy does not qualify.

The capstone prohibits an LLM from writing the whole assessed model architecture,
tuning its hyperparameters or writing experimental conclusions. The model implementation
and exact numerical configuration are therefore student-owned. All long-running work
is opt-in; Run All in the default `prepare` mode cannot start training or inference.
""")

nb.cells[1] = cell_markdown("""
## Safe execution map

- `prepare` inspects official files, restores or trains an allowed BPE tokenizer,
  and tests the supplied student model.
- `fresh_train` refuses an existing checkpoint.
- `resume_train` requires a compatible checkpoint and prints its restored step.
- `inference_only` loads only the official test prompts, tokenizer, student model
  source and selected checkpoint.

The intended budget is 18,000 optimizer steps across resumable sessions. A 9.25-hour
per-session guard writes a complete checkpoint and recovery archive before stopping.
No step rate or final score is assumed; use the measured Kaggle smoke-test throughput.
""")

config = nb.cells[4].source
config = replace_once(config, 'experiment_id: str = "phase1_submission04_prefix_hybrid"',
                      'experiment_id: str = "phase1_submission04_efficient_causal"', "experiment identity")
config = replace_once(config, 'filename = "prefix_hybrid_lm.py"', 'filename = "efficient_causal_lm.py"', "model source filename")
config = replace_once(
    config,
    'execution_mode: str = "auto_train"  # auto_train | fresh_train | resume_train | inference_only',
    'execution_mode: str = "prepare"  # prepare | fresh_train | resume_train | inference_only',
    "safe execution mode",
)
config = replace_once(
    config,
    'if CFG.execution_mode not in {"prepare", "auto_train", "fresh_train", "resume_train", "inference_only"}:',
    'if CFG.execution_mode not in {"prepare", "fresh_train", "resume_train", "inference_only"}:',
    "execution-mode validation",
)
config = replace_once(config, 'corpus_document_cap_per_source: int | None = 8_000',
                      'corpus_document_cap_per_source: int | None = None', "student corpus cap")
config = config.replace(
    '# Submission 03 intentionally trains a new 25k byte-level BPE tokenizer.\n'
    '    # A restored tokenizer in this workspace is still reused on resume/inference.',
    '# Point to a frozen course or prior-submission BPE tokenizer for a controlled comparison.\n'
    '    # Alternatively enable the existing BPE-training cell from official training data.',
)
start = config.index('    architecture_record: dict[str, Any] = field(default_factory=lambda: {')
end = config.index('    student_owned_architecture_attestation:', start)
config = config[:start] + textwrap.indent(textwrap.dedent('''
    architecture_record: dict[str, Any] = field(default_factory=lambda: {
        "family": "efficient causal local/global attention Transformer",
        "substantive_difference": (
            "bounded local causal attention plus explicitly routed global information, "
            "grouped-query key/value sharing and a measured cache footprint"
        ),
        "position_representation": "RoPE applied to attention queries and keys",
        "tokenizer": "frozen BPE trained only on permitted legal text",
        "parameter_range": [50_000_000, 300_000_000],
        "student_model_config": "REQUIRED BEFORE TRAINING",
        "student_exact_parameter_derivation": "REQUIRED BEFORE TRAINING",
        "student_design_rationale": "REQUIRED BEFORE TRAINING",
    })
    innovation_hypothesis: str = "STUDENT MUST WRITE BEFORE TRAINING"
    ablation_control: str = "STUDENT MUST NAME MATCHED EARLIER SUBMISSIONS"
''').strip("\n") + "\n", "    ") + config[end:]
config = replace_once(config, 'full_training_authorized: bool = True',
                      'full_training_authorized: bool = False', "training authorization")
config = replace_once(config, 'tokenizer_import_path: str | None = None',
                      'tokenizer_import_path: str | None = field(default_factory=_discover_attached_tokenizer)',
                      "BPE source discovery")
nb.cells[4] = cell_code(config)

# Inference must work with only the official test prompts plus recovered artifacts.
schema_cell = nb.cells[10].source
schema_start = schema_cell.index('_schema_names = [*CFG.train_files, CFG.test_file]')
schema_cell = schema_cell[:schema_start] + textwrap.dedent('''
_schema_names = [*CFG.train_files, CFG.test_file]
if CFG.execution_mode == "inference_only":
    test_path = require_unique_file(DISCOVERED, CFG.test_file)
    SCHEMAS = {CFG.test_file: inspect_schema(test_path)}
    inferred_test = infer_record_schema(test_path, role="test")
    schema_path = DIRS["manifests"] / "resolved_json_schemas.json"
    if schema_path.exists():
        RESOLVED_SCHEMA = json.loads(schema_path.read_text(encoding="utf-8"))
        if RESOLVED_SCHEMA.get("test") != inferred_test:
            raise RuntimeError("Recovered official test schema differs from attached test file")
    else:
        RESOLVED_SCHEMA = {"training": {}, "test": inferred_test}
        atomic_write_json(schema_path, RESOLVED_SCHEMA)
    print("Verified official test schema without loading training files:", inferred_test)
elif all(len(DISCOVERED.get(name, [])) == 1 or name in CFG.official_file_overrides for name in _schema_names):
    SCHEMAS = {name: inspect_schema(require_unique_file(DISCOVERED, name)) for name in _schema_names}
    print(json.dumps(SCHEMAS, indent=2, ensure_ascii=False))
    RESOLVED_SCHEMA = resolve_official_schemas(CFG, DISCOVERED)
    atomic_write_json(DIRS["manifests"] / "resolved_json_schemas.json", RESOLVED_SCHEMA)
    print("Verified per-file schemas:", json.dumps(RESOLVED_SCHEMA, indent=2))
else:
    SCHEMAS = {}
    RESOLVED_SCHEMA = None
    unresolved = {
        name: len(DISCOVERED.get(name, [])) for name in _schema_names
        if len(DISCOVERED.get(name, [])) != 1 and name not in CFG.official_file_overrides
    }
    print("Schema inspection deferred until files resolve uniquely:", unresolved)
''').strip() + "\n"
nb.cells[10] = cell_code(schema_cell)

nb.cells[17] = cell_markdown("""
## 6. Student-owned efficient causal model contract

Provide `efficient_causal_lm.py` with `make_model_config(vocab_size)` and
`build_model(config)`. Its full-sequence call accepts `input_ids` and
`attention_mask` and returns `[batch, sequence, vocabulary]` logits.

For generation it must provide `prefill_context(input_ids)` and
`decode_cached(next_input_ids, cache) -> (logits, cache)`. These are verified
against full-sequence logits before official inference.

The architectural record must describe the actual local/global attention rule,
causal information flow, grouped-query sharing, RoPE application, exact parameter
calculation, and how the cache remains bounded. The notebook checks causal
leakage and the 50M–300M count, but scientific authorship stays with the student.
""")

nb.cells[18] = cell_code(r'''
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
        raise RuntimeError(f"Pretrained-model libraries in model source: {imported & banned}")
    destination = DIRS["model"] / "efficient_causal_lm.py"
    payload = source_path.read_bytes()
    if destination.exists() and destination.read_bytes() != payload and any(DIRS["checkpoints"].glob("*.pt")):
        raise RuntimeError("Refusing to change model source after checkpoints exist")
    atomic_write_bytes(destination, payload)
    return destination

def efficient_logits(model, input_ids, attention_mask, prefix_lengths=None):
    # The retained dataset supplies prefix_lengths for target-only loss. Attention
    # itself is globally causal; the student model never receives prefix lengths.
    del prefix_lengths
    if MODEL_MODULE is not None and callable(getattr(MODEL_MODULE, "forward_adapter", None)):
        output = MODEL_MODULE.forward_adapter(model, input_ids, attention_mask)
    else:
        output = model(input_ids=input_ids, attention_mask=attention_mask)
    return output.logits if hasattr(output, "logits") else output[0] if isinstance(output, (tuple, list)) else output

def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }

def validate_efficient_contract(model, vocab_size: int, device: str, sequence_length: int = 13):
    model = model.to(device); model.train(); model.zero_grad(set_to_none=True)
    input_ids = torch.randint(0, vocab_size, (2, sequence_length), dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    logits = efficient_logits(model, input_ids, attention_mask)
    expected = (2, sequence_length, vocab_size)
    if tuple(logits.shape) != expected: raise ValueError(f"Expected {expected}, got {tuple(logits.shape)}")
    if not torch.isfinite(logits).all(): raise FloatingPointError("Non-finite logits")
    labels = torch.randint(0, vocab_size, (2, sequence_length), dtype=torch.long, device=device)
    labels[:, :5] = -100
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab_size), labels.reshape(-1), ignore_index=-100)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
        raise FloatingPointError("Missing or non-finite gradients")
    report = count_parameters(model)
    if not 50_000_000 <= report["total"] <= 300_000_000:
        raise ValueError(f"Official parameter bound violated: {report['total']:,}")
    model.zero_grad(set_to_none=True)
    return {"logits_shape": list(logits.shape), "loss": float(loss.detach().cpu()), **report, "device": device}

_restored_model_source = DIRS["model"] / "efficient_causal_lm.py"
_configured_model_source = (
    Path(CFG.student_model_source_path).expanduser()
    if CFG.student_model_source_path
    else _restored_model_source if _restored_model_source.exists() else None
)
if _configured_model_source is not None:
    if TOKENIZER_META is None: raise RuntimeError("Register/restore BPE before importing the model")
    MODEL_SOURCE_PATH = install_student_model_source(_configured_model_source)
    MODEL_SOURCE_SHA256 = sha256_file(MODEL_SOURCE_PATH)
    MODEL_MODULE = import_source_module(f"cs781_submission04_model_{MODEL_SOURCE_SHA256[:12]}", MODEL_SOURCE_PATH)
    if not callable(getattr(MODEL_MODULE, "build_model", None)):
        raise AttributeError("Student module must define build_model(config)")
    if STUDENT_MODEL_CONFIG_OVERRIDE is not None:
        MODEL_CONFIG = dict(STUDENT_MODEL_CONFIG_OVERRIDE)
    elif callable(getattr(MODEL_MODULE, "make_model_config", None)):
        MODEL_CONFIG = dict(MODEL_MODULE.make_model_config(int(TOKENIZER_META["vocab_size"])))
    else:
        raise RuntimeError("Provide make_model_config(vocab_size) or STUDENT_MODEL_CONFIG_OVERRIDE")
    if int(MODEL_CONFIG.get("vocab_size", -1)) != int(TOKENIZER_META["vocab_size"]):
        raise ValueError("Model vocabulary does not match the frozen BPE tokenizer")
    if int(MODEL_CONFIG.get("max_sequence_length", 0)) <= 2:
        raise ValueError("MODEL_CONFIG must declare max_sequence_length > 2")
    RECONSTRUCT_MODEL = MODEL_MODULE.build_model
    MODEL = RECONSTRUCT_MODEL(MODEL_CONFIG)
    MODEL_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    MODEL_REPORT = validate_efficient_contract(MODEL, TOKENIZER_META["vocab_size"], MODEL_DEVICE)
    MODEL_REPORT.update({"architecture_family": "efficient causal local/global Transformer",
                         "model_source_sha256": MODEL_SOURCE_SHA256, "model_config": MODEL_CONFIG})
    atomic_write_json(DIRS["manifests"] / "model_validation.json", MODEL_REPORT)
    MODEL.to("cpu")
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("Student model validation:", json.dumps(MODEL_REPORT, indent=2))
else:
    MODEL_REPORT = None
    print("STUDENT ACTION REQUIRED: attach independently authored efficient_causal_lm.py")
''')

nb.cells[27] = cell_code(r'''
def validate_strict_causality(model, vocab_size: int, device: str) -> None:
    model.eval()
    first = torch.randint(0, vocab_size, (1, 16), dtype=torch.long, device=device)
    second = first.clone(); second[:, 10:] = torch.randint(0, vocab_size, (1, 6), device=device)
    mask = torch.ones_like(first, dtype=torch.bool)
    with torch.no_grad():
        a = efficient_logits(model, first, mask)[:, :10]
        b = efficient_logits(model, second, mask)[:, :10]
    if not torch.allclose(a, b, atol=1e-5, rtol=1e-5):
        raise AssertionError("Future token information leaked into earlier causal logits")

def validate_prompt_conditioning(model, vocab_size: int, device: str) -> None:
    model.eval()
    first = torch.randint(0, vocab_size, (1, 16), dtype=torch.long, device=device)
    second = first.clone(); second[:, :8] = torch.randint(0, vocab_size, (1, 8), device=device)
    mask = torch.ones_like(first, dtype=torch.bool)
    with torch.no_grad():
        a = efficient_logits(model, first, mask)[:, 8:]
        b = efficient_logits(model, second, mask)[:, 8:]
    if torch.allclose(a, b, atol=1e-7, rtol=1e-7):
        raise AssertionError("Continuation logits are insensitive to the prompt")

def run_student_integration_smoke_test(model, tokenizer, model_config, reconstruct_model, device=None):
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    meta = tokenizer_metadata(tokenizer, DIRS["tokenizer"])
    report = validate_efficient_contract(model, meta["vocab_size"], device)
    validate_strict_causality(model, meta["vocab_size"], device)
    report["strict_causality"] = "PASS"
    validate_prompt_conditioning(model, meta["vocab_size"], device)
    report["prompt_conditioning"] = "PASS"
    probe = tokenizer_encode(tokenizer, "A constitutional court considered the appeal.")
    if not probe or min(probe) < 0 or max(probe) >= meta["vocab_size"]:
        raise AssertionError("BPE emitted an out-of-vocabulary ID")
    report["token_id_range"] = "PASS"
    limit = int(model_config["max_sequence_length"])
    try:
        too_long = torch.zeros((1, limit + 1), dtype=torch.long, device=device)
        efficient_logits(model, too_long, torch.ones_like(too_long, dtype=torch.bool))
        raise AssertionError("Context limit not enforced")
    except ValueError:
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
    print("Efficient-model integration test awaits student source and frozen BPE.")
''')

nb.cells[28] = cell_markdown("""
## 10. Resumable two GPU causal continuation training

The training data keeps the same source/target windows as Submission 03 for a controlled
comparison. The source receives no loss; the continuation is learned with a fully causal
attention rule. The two GPU path calculates loss on each replica before reducing, avoiding
large vocabulary logits on the primary GPU. A one GPU configuration is also supported
when selected before the first checkpoint.
""")

training = nb.cells[29].source
training = training.replace("bidirectional-prefix/causal-continuation mask", "strictly causal local/global attention")
training = replace_once(training, '"max_steps": 10_000,', '"max_steps": 18_000,', "step target")
training = replace_once(training, '"recovery_interval": 1000,',
                        '"recovery_interval": 1000,\n    "max_session_seconds": 33_300,  # 9.25 hours; leaves time to export a recovery ZIP',
                        "wall-time safety budget")
training = training.replace('PrefixHybridDataset', 'CausalContinuationDataset')
training = training.replace('build_fixed_hybrid_validation_rows', 'build_fixed_causal_validation_rows')
training = training.replace('evaluate_hybrid_loss', 'evaluate_causal_loss')
training = training.replace('train_submission03', 'train_submission04')
training = training.replace('hybrid_logits(', 'efficient_logits(')
training = training.replace('fixed held-out prefix/continuation windows', 'fixed held-out causal continuation windows')
training = training.replace('Prefix-hybrid training requires', 'Causal continuation training requires')
training = replace_once(training,
                        '"validation_sequences_per_source", "checkpoint_interval",\n        "recovery_interval",\n    )',
                        '"validation_sequences_per_source", "checkpoint_interval",\n        "recovery_interval", "max_session_seconds",\n    )',
                        "required time budget")
training = replace_once(training, '"validation_interval", "validation_sequences_per_source", "checkpoint_interval", "recovery_interval",',
                        '"validation_interval", "validation_sequences_per_source", "checkpoint_interval", "recovery_interval", "max_session_seconds",',
                        "positive time budget")
training = replace_once(training, 'if gpu_ids != [0, 1]: raise ValueError("Dual-T4 configuration requires gpu_ids [0, 1]")',
                        'if gpu_ids not in ([0], [0, 1]): raise ValueError("Select GPU IDs [0] or [0, 1] before checkpoint 1")',
                        "GPU configuration validator")
training = replace_once(training,
                        'f"Dual-T4 training requires CUDA devices {gpu_ids}; found {torch.cuda.device_count()}"',
                        'f"Requested CUDA devices {gpu_ids}; found {torch.cuda.device_count()}"',
                        "GPU error wording")
training = replace_once(training,
                        'wrapped=torch.nn.DataParallel(training_replica,device_ids=gpu_ids,output_device=gpu_ids[0],dim=0)',
                        'wrapped=(torch.nn.DataParallel(training_replica,device_ids=gpu_ids,output_device=gpu_ids[0],dim=0)\n'
                        '         if len(gpu_ids)>1 else training_replica)', "single GPU fallback")
training = replace_once(training, '"phase1_submission04_prefix_hybrid_inference_v1"',
                        '"phase1_submission04_efficient_causal_inference_v1"', "checkpoint format")
training = replace_once(training, '    best_loss = min((float(row["loss"]) for row in validation_history), default=math.inf)\n    try:',
                        '    best_loss = min((float(row["loss"]) for row in validation_history), default=math.inf)\n'
                        '    session_started = time.monotonic()\n    try:', "session timer")
training = replace_once(training, '        while reporter.state.step < cfg["max_steps"]:\n            model.train();',
                        '        while reporter.state.step < cfg["max_steps"]:\n'
                        '            if time.monotonic() - session_started >= cfg["max_session_seconds"]:\n'
                        '                append_event(DIRS["histories"] / "session_events.jsonl", {\n'
                        '                    "event": "time_budget_reached", "step": reporter.state.step,\n'
                        '                    "elapsed_seconds": time.monotonic() - session_started,\n'
                        '                    "created_utc": utc_now(),\n'
                        '                })\n'
                        '                break\n'
                        '            model.train();', "time-budget break")
training = replace_once(training,
                        '        final_loss = validation_history[-1]["loss"] if validation_history else None\n'
                        '        save_inference_checkpoint(\n'
                        '            DIRS["checkpoints"] / "model_final.pt", model, model_config,\n'
                        '            tokenizer_meta, dataset_fp, reporter.state.step, final_loss,\n'
                        '        )\n'
                        '        if not (DIRS["checkpoints"] / "selected_model.pt").exists():\n'
                        '            atomic_copy(DIRS["checkpoints"] / "model_final.pt", DIRS["checkpoints"] / "selected_model.pt")\n'
                        '        export_training_recovery_snapshot(ROOT)',
                        '        if reporter.state.last_checkpoint_step != reporter.state.step:\n'
                        '            reporter.state.last_checkpoint_step = reporter.state.step\n'
                        '            save_checkpoint_atomic(\n'
                        '                latest, model=model, optimizer=optimizer, scheduler=scheduler,\n'
                        '                scaler=checkpoint_scaler, sampler=sampler, progress=asdict(reporter.state),\n'
                        '                model_config=model_config, tokenizer_fingerprint=tokenizer_meta["fingerprint"],\n'
                        '                dataset_fingerprint_value=dataset_fp, training_config=cfg,\n'
                        '                history_path=history_path, validation_history=validation_history,\n'
                        '                experiment_id=cfg["experiment_id"], device=device,\n'
                        '                model_source_sha256=MODEL_SOURCE_SHA256,\n'
                        '            )\n'
                        '        if reporter.state.step == cfg["max_steps"]:\n'
                        '            final_loss = validation_history[-1]["loss"] if validation_history else None\n'
                        '            save_inference_checkpoint(\n'
                        '                DIRS["checkpoints"] / "model_final.pt", model, model_config,\n'
                        '                tokenizer_meta, dataset_fp, reporter.state.step, final_loss,\n'
                        '            )\n'
                        '            if not (DIRS["checkpoints"] / "selected_model.pt").exists():\n'
                        '                atomic_copy(DIRS["checkpoints"] / "model_final.pt", DIRS["checkpoints"] / "selected_model.pt")\n'
                        '        export_training_recovery_snapshot(ROOT)', "safe end-of-session save")
training = replace_once(training, '        "completed_step": reporter.state.step,',
                        '        "completed_step": reporter.state.step,\n'
                        '        "target_step": cfg["max_steps"],\n'
                        '        "training_complete": reporter.state.step == cfg["max_steps"],\n'
                        '        "session_elapsed_seconds": time.monotonic() - session_started,',
                        "training result status")
training = replace_once(training, 'RUN_FULL_TRAINING = True', 'RUN_FULL_TRAINING = False', "training switch")
training = training.replace('"auto_train",', '')
training = replace_once(training,
                        '    resume_training = (\n        CFG.execution_mode == "resume_train"\n        or (\n            CFG.execution_mode == "auto_train"\n            and (\n                DIRS["checkpoints"]\n                / "checkpoint.latest.pt"\n            ).exists()\n        )\n    )',
                        '    resume_training = CFG.execution_mode == "resume_train"',
                        "explicit resume decision")
training = training.replace('"continuation_future_leakage",\n            "bidirectional_prefix",\n            "prefix_conditioning",',
                            '"strict_causality",\n            "prompt_conditioning",')
training = replace_once(training,
    '    batch = next(iter(DataLoader(\n        tiny,batch_size=cfg["batch_size"],sampler=sampler,num_workers=0,',
    '    batch_prep_started = time.monotonic()\n'
    '    batch = next(iter(DataLoader(\n        tiny,batch_size=cfg["batch_size"],sampler=sampler,num_workers=0,',
    "representative batch timing")
training = replace_once(training,
    '        drop_last=True,pin_memory=True,\n    )))\n    smoke_model,active_gpu_ids=',
    '        drop_last=True,pin_memory=True,\n    )))\n'
    '    batch_preparation_seconds = time.monotonic() - batch_prep_started\n'
    '    smoke_model,active_gpu_ids=',
    "batch preparation measurement")
training = replace_once(training,
    '    optimizer.zero_grad(set_to_none=True)\n    with torch.autocast(device_type="cuda",',
    '    for gpu in active_gpu_ids: torch.cuda.synchronize(gpu)\n'
    '    compute_started = time.monotonic()\n'
    '    optimizer.zero_grad(set_to_none=True)\n    with torch.autocast(device_type="cuda",',
    "synchronized compute timer")
training = replace_once(training,
    '    scheduler.step(); loss_value = float(loss.detach().cpu())',
    '    scheduler.step()\n'
    '    for gpu in active_gpu_ids: torch.cuda.synchronize(gpu)\n'
    '    microbatch_compute_seconds = time.monotonic() - compute_started\n'
    '    loss_value = float(loss.detach().cpu())',
    "compute timing result")
training = replace_once(training,
    '        "batch_size": cfg["batch_size"], "checkpoint_restore": "PASS", "created_utc": utc_now(),',
    '        "batch_size": cfg["batch_size"],\n'
    '        "batch_preparation_seconds": batch_preparation_seconds,\n'
    '        "microbatch_compute_seconds": microbatch_compute_seconds,\n'
    '        "microbatch_tokens_per_second": int(batch["processed_tokens"].sum()) / max(microbatch_compute_seconds, 1e-9),\n'
    '        "timing_note": "one microbatch; full optimizer step uses gradient_accumulation_steps microbatches",\n'
    '        "checkpoint_restore": "PASS", "created_utc": utc_now(),',
    "GPU smoke throughput report")
nb.cells[29] = cell_code(training)

training = nb.cells[29].source
training = replace_once(training,
    '        "student_integration_smoke_test.json",\n        "validation_sequence_manifest.json",',
    '        "student_integration_smoke_test.json",\n'
    '        "gpu_training_smoke_test.json",\n'
    '        "validation_sequence_manifest.json",',
    "required smoke manifest")
training = replace_once(training,
    '    if not CFG.full_training_authorized:\n        problems.append(\n            "full_training_authorized is False"\n        )',
    '    if not CFG.full_training_authorized:\n        problems.append("full_training_authorized is False")\n'
    '    if not CFG.student_owned_architecture_attestation:\n'
    '        problems.append("student-owned architecture attestation is False")\n'
    '    if not CFG.student_configuration_confirmed:\n'
    '        problems.append("student configuration confirmation is False")\n'
    '    if not CFG.trained_from_scratch_attestation:\n'
    '        problems.append("scratch-training attestation is False")',
    "truthful preflight attestations")
training = replace_once(training,
    '    if problems:\n        raise RuntimeError(',
    '    smoke_path = DIRS["manifests"] / "gpu_training_smoke_test.json"\n'
    '    if smoke_path.exists() and json.loads(smoke_path.read_text(encoding="utf-8")).get("status") != "PASS":\n'
    '        problems.append("GPU training smoke test has not passed")\n'
    '    if problems:\n        raise RuntimeError(',
    "GPU smoke status gate")
nb.cells[29] = cell_code(training)

nb.cells[42] = cell_markdown("""
## Kaggle execution checklist for Submission 04

1. Keep previous submissions and their checkpoints separate. Submission 04 writes only
   to `/kaggle/working/CS781_Capstone/phase1/submission_04`.
2. Attach the four official Phase 1 legal training sources and the official test prompts
   under `/kaggle/input`. Confirm the notebook's inspected schemas before preprocessing.
3. Attach a frozen allowed BPE tokenizer. RoPE is supplied by the student model's
   attention implementation, not by the tokenizer.
4. Independently implement `efficient_causal_lm.py` using Section 6's model and cache
   contract. Record your attention rule, RoPE equation, design rationale and exact
   parameter derivation. The runtime model must contain 50M–300M parameters.
5. Review Cell 4's corpus cap, tokenizer/model paths, architecture record, hypothesis,
   control comparison and attestations. Keep the switches false during preparation.
6. Review Cell 29's inherited two GPU settings and proposed 18,000-step budget.
   Select `[0]` and adjust `batch_size` before checkpoint 1 if only one GPU is available.
   Run the representative smoke test and read measured VRAM, precision and step timing.
7. Set `execution_mode="fresh_train"` for the first run or `"resume_train"` after an
   authenticated recovery restore. The 9.25-hour session guard saves recovery before
   exiting; the 18,000-step target may take multiple sessions.
8. Download each milestone `training_recovery.latest.zip` outside Kaggle. The archive
   has the integrity manifest accepted by the restore helper.
9. For inference use `execution_mode="inference_only"`. The notebook requires an
   authenticated `selected_model.pt`, tokenizer and student model source. It checks
   cached logits against full-sequence logits before generating official IDs.
10. Every successful prediction is saved immediately. Failed IDs receive the declared
    retry rounds and seeds. Reopening the same run skips complete IDs and retries only
    missing IDs. An incomplete set prevents `submission.zip` creation.
11. Upload only `submission.zip` after the root-level `prediction.json` passes the
    exact official-ID and 128-word checks. Record the actual Codabench acceptance.

The notebook cannot promise a score or that 18,000 steps will fit in one Kaggle session.
The smoke-test throughput and actual validation results determine what is achievable.
""")

recovery = nb.cells[39].source
recovery = replace_once(recovery,
    "RECOVERY_ZIP,RECOVERY_REPORT=create_recovery_zip(ROOT)\nprint('Complete model/recovery ZIP:',RECOVERY_ZIP)",
    "if SUBMISSION_ZIP is not None and (DIRS['checkpoints']/'selected_model.pt').exists():\n"
    "    RECOVERY_ZIP,RECOVERY_REPORT=create_recovery_zip(ROOT)\n"
    "    print('Complete model/recovery ZIP:',RECOVERY_ZIP)\n"
    "else:\n"
    "    RECOVERY_ZIP=RECOVERY_REPORT=None\n"
    "    print('Complete recovery archive deferred until a validated submission exists.')",
    "do not archive an incomplete preparation run")
nb.cells[39] = cell_code(recovery)

verification = nb.cells[41].source
verification = replace_once(verification,
    'checks["training_completed_10000_steps"] = bool(\n'
    '        training_result.exists() and training_config.exists()\n'
    '        and int(json.loads(training_result.read_text()).get("completed_step", -1)) == 10_000\n'
    '        and int(json.loads(training_config.read_text()).get("max_steps", -1)) == 10_000\n'
    '    )',
    'checks["training_target_completed"] = bool(\n'
    '        training_result.exists() and training_config.exists()\n'
    '        and json.loads(training_result.read_text()).get("training_complete") is True\n'
    '        and int(json.loads(training_result.read_text()).get("completed_step", -1))\n'
    '            == int(json.loads(training_config.read_text()).get("max_steps", -2))\n'
    '    )',
    "dynamic completed-step check")
verification = replace_once(verification,
    "FINAL_REPORT = final_verification(CFG, MODEL, TOKENIZER_META, [row['id'] for row in TEST_RECORDS])",
    "FINAL_REPORT = final_verification(\n"
    "    CFG, MODEL, TOKENIZER_META,\n"
    "    [row['id'] for row in TEST_RECORDS] if 'TEST_RECORDS' in globals() else None,\n"
    ")",
    "safe final verification in prepare mode")
nb.cells[41] = cell_code(verification)

for cell in nb.cells:
    if cell.cell_type == "code":
        cell.execution_count = None
        cell.outputs = []

# The source notebook's preprocessing and tokenizer switches reflected a live
# Submission 03 run; reset them for a safe new experiment.
nb.cells[14] = cell_code(replace_once(nb.cells[14].source,
    'RUN_CORPUS_PREPROCESSING = True', 'RUN_CORPUS_PREPROCESSING = False', "corpus switch"))
nb.cells[16] = cell_code(replace_once(nb.cells[16].source,
    'RUN_TOKENIZER_TRAINING=True', 'RUN_TOKENIZER_TRAINING=False', "BPE training switch"))

nb.cells[33] = cell_markdown("""
## 12. Cached causal decoding and resumable official predictions

The student model must implement a cache that preserves the selected local/global
attention semantics. The notebook compares two cached steps with full-sequence logits
before generating any official prediction. Each valid prediction is saved immediately.
On a later Kaggle run, matching checkpoint/tokenizer/test/decoding fingerprints allow
only missing IDs to be regenerated. Each failed ID receives a declared set of retries.
""")

generation = nb.cells[34].source
generation = generation.replace('hybrid_continuation', 'efficient_continuation')
generation = generation.replace('hybrid_logits(', 'efficient_logits(')
generation = generation.replace('prefill_prefix', 'prefill_context')
generation = generation.replace('Prefix-hybrid inference requires', 'Causal inference requires')
generation = replace_once(generation,
    '    use_cache = callable(getattr(model, "prefill_context", None)) and callable(getattr(model, "decode_cached", None))\n'
    '    if use_cache:',
    '    use_cache = callable(getattr(model, "prefill_context", None)) and callable(getattr(model, "decode_cached", None))\n'
    '    if not use_cache:\n'
    '        raise RuntimeError("Student model must implement cached causal decoding")\n'
    '    if use_cache:', "mandatory cache")
nb.cells[34] = cell_code(generation)

# EOS cannot satisfy a 128-word continuation, so exclude it from all eligible
# sampling distributions rather than burning a retry on premature termination.
generation = nb.cells[34].source
generation = replace_once(generation,
    '    forbidden = {int(value) for value in (bos_id, pad_id) if value is not None}',
    '    forbidden = {int(value) for value in (bos_id, pad_id, eos_id) if value is not None}',
    "premature EOS prevention")
generation = replace_once(generation,
    '        if eos_id is not None and next_id == int(eos_id):\n'
    '            raise RuntimeError(f"Premature EOS after {len(words(continuation))} words")\n',
    '', "remove unreachable early EOS error")
nb.cells[34] = cell_code(generation)

inference = nb.cells[35].source
inference = inference.replace('hybrid_continuation', 'efficient_continuation')
inference = inference.replace('hybrid_logits(', 'efficient_logits(')
inference = inference.replace('prefill_prefix', 'prefill_context')
inference = replace_once(inference,
    'require_unique_file(DISCOVERED, CFG.test_file), SCHEMAS["test"]',
    'require_unique_file(DISCOVERED, CFG.test_file), RESOLVED_SCHEMA["test"]',
    "official test schema reference")
inference = replace_once(inference,
    '    checkpoint_id = f"selected-{sha256_file(checkpoint_path)[:12]}"',
    '    if dict(state["model_config"]) != dict(MODEL_CONFIG):\n'
    '        raise RuntimeError("Selected checkpoint model configuration differs from the current student module")\n'
    '    checkpoint_id = f"selected-{sha256_file(checkpoint_path)[:12]}"',
    "selected checkpoint configuration")
inference = replace_once(inference, 'RUN_INFERENCE = True', 'RUN_INFERENCE = False', "inference switch")
start = inference.index('def generate_official_predictions(')
end = inference.index('def build_local_generation_examples(', start)
inference = inference[:start] + textwrap.dedent(r'''
def generate_official_predictions(model, tokenizer, test_records, device, *, checkpoint_id, tokenizer_meta):
    policies = list(CFG.decoding.get("policies") or [])
    default_budget = int(CFG.decoding.get("max_new_tokens") or 0)
    recovery_rounds = int(CFG.decoding.get("recovery_rounds", 1))
    stride = int(CFG.decoding.get("retry_seed_stride", 0))
    if not policies or default_budget <= 0 or recovery_rounds <= 0 or stride <= 0:
        raise RuntimeError("Declare ordered policies, token budget, recovery rounds, and retry stride")
    names = [str(policy.get("name", "")) for policy in policies]
    if not all(names) or len(names) != len(set(names)):
        raise ValueError("Every decoding policy needs a unique nonempty name")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", checkpoint_id)
    partial_path = DIRS["predictions"] / f"official_predictions.{safe}.partial.json"
    meta_path = partial_path.with_suffix(".meta.json")
    attempts_path = DIRS["predictions"] / f"generation_attempts.{safe}.jsonl"
    selected_path = DIRS["checkpoints"] / "selected_model.pt"
    expected_meta = {
        "checkpoint_id": checkpoint_id,
        "checkpoint_sha256": sha256_file(selected_path),
        "model_source_sha256": MODEL_SOURCE_SHA256,
        "tokenizer_fingerprint": tokenizer_meta["fingerprint"],
        "test_fingerprint": test_records_fingerprint(test_records),
        "decoding": dict(CFG.decoding),
    }
    expected_ids = {row["id"] for row in test_records}
    if len(expected_ids) != len(test_records): raise ValueError("Duplicate official test IDs")
    if attempts_path.exists() and (not meta_path.exists() or
        json.loads(meta_path.read_text(encoding="utf-8")) != expected_meta):
        raise RuntimeError("Generation attempt log has incompatible provenance")
    if partial_path.exists():
        if not meta_path.exists() or json.loads(meta_path.read_text(encoding="utf-8")) != expected_meta:
            raise RuntimeError("Partial predictions do not match current checkpoint/tokenizer/test/decoding")
        predictions = json.loads(partial_path.read_text(encoding="utf-8"))
        if set(predictions) - expected_ids: raise RuntimeError("Partial predictions contain foreign IDs")
        for identifier, value in predictions.items():
            if not isinstance(value, dict) or set(value) != {"text"} or not isinstance(value["text"], str):
                raise RuntimeError(f"Invalid cached prediction structure: {identifier}")
            if len(words(value["text"])) != int(CFG.decoding["target_words"]):
                raise RuntimeError(f"Invalid cached prediction word count: {identifier}")
    else:
        predictions = {}
        atomic_write_json(meta_path, expected_meta)
    prior_rounds = []
    if attempts_path.exists():
        for line in attempts_path.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            event = json.loads(line)
            if event.get("id") not in expected_ids or int(event.get("round", -1)) < 0:
                raise RuntimeError("Invalid prior generation-attempt record")
            prior_rounds.append(int(event["round"]))
    round_base = max(prior_rounds, default=-1) + 1
    latest_errors = {}
    for round_offset in range(recovery_rounds):
        recovery_round = round_base + round_offset
        pending = [row for row in test_records if row["id"] not in predictions]
        if not pending: break
        print(f"Recovery round {round_offset + 1}/{recovery_rounds} (absolute {recovery_round}): {len(pending)} missing IDs")
        for row in tqdm(pending, desc=f"Official generation round {recovery_round + 1}"):
            errors = []
            id_offset = int(hashlib.sha256(row["id"].encode()).hexdigest()[:8], 16)
            for policy_index, policy in enumerate(policies):
                actual_policy = dict(policy)
                actual_policy["seed"] = int(policy["seed"]) + recovery_round * stride + id_offset
                budget = int(policy.get("max_new_tokens", default_budget))
                started = time.monotonic()
                try:
                    continuation, quality = efficient_continuation(
                        model, tokenizer, row["prompt"], device, policy=actual_policy,
                        target_words=int(CFG.decoding["target_words"]), max_new_tokens=budget,
                    )
                    append_event(attempts_path, {
                        "id": row["id"], "round": recovery_round, "policy_index": policy_index,
                        "effective_policy": actual_policy, "max_new_tokens": budget,
                        "status": "success", "quality": quality,
                        "seconds": time.monotonic() - started, "created_utc": utc_now(),
                    })
                    predictions[row["id"]] = {"text": continuation}
                    atomic_write_json(partial_path, predictions)
                    latest_errors.pop(row["id"], None)
                    break
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    errors.append(error)
                    append_event(attempts_path, {
                        "id": row["id"], "round": recovery_round, "policy_index": policy_index,
                        "effective_policy": actual_policy, "max_new_tokens": budget,
                        "status": "failure", "error": error,
                        "seconds": time.monotonic() - started, "created_utc": utc_now(),
                    })
            if row["id"] not in predictions: latest_errors[row["id"]] = errors
        atomic_write_json(DIRS["predictions"] / "generation_failures.json", latest_errors)
    missing = expected_ids - set(predictions)
    if missing:
        raise RuntimeError(f"{len(missing)} official IDs remain invalid; saved partial predictions are resumable")
    provenance = {
        **expected_meta, "prediction_count": len(predictions),
        "partial_prediction_sha256": sha256_file(partial_path), "created_utc": utc_now(),
    }
    atomic_write_json(DIRS["predictions"] / "inference_provenance.json", provenance)
    return predictions, []

''') + inference[end:]
nb.cells[35] = cell_code(inference)

# The retries are declared before generation and become part of provenance.
config = nb.cells[4].source
config = replace_once(config, '"max_new_tokens": 448,',
                      '"max_new_tokens": 448,\n        "recovery_rounds": 3,\n'
                      '        "retry_seed_stride": 100_003,', "declared retry policy")
config = replace_once(config, '"name": "retry_wider_nucleus",',
                      '"name": "retry_wider_nucleus",\n                "max_new_tokens": 512,',
                      "retry token budget")
config = replace_once(config,
    '                "min_unique_word_ratio": 0.50,\n'
    '                "max_word_fraction": 0.10,\n'
    '            },\n'
    '        ],',
    '                "min_unique_word_ratio": 0.50,\n'
    '                "max_word_fraction": 0.10,\n'
    '            },\n'
    '            {\n'
    '                "name": "completion_fallback_length_only",\n'
    '                "strategy": "top_k_top_p",\n'
    '                "seed": 1781,\n'
    '                "temperature": 0.95,\n'
    '                "top_k": 160,\n'
    '                "top_p": 0.96,\n'
    '                "repetition_penalty": 1.12,\n'
    '                "no_repeat_ngram_size": 3,\n'
    '                "max_new_tokens": 512,\n'
    '            },\n'
    '        ],',
    "declared length-only final fallback")
nb.cells[4] = cell_code(config)

training = nb.cells[29].source
training = replace_once(training,
    'if int(CFG.decoding["max_new_tokens"]) + 2 > int(MODEL_CONFIG["max_sequence_length"]):',
    'if max([int(CFG.decoding["max_new_tokens"])] +\n'
    '               [int(policy.get("max_new_tokens", CFG.decoding["max_new_tokens"]))\n'
    '                for policy in CFG.decoding["policies"]]) + 2 > int(MODEL_CONFIG["max_sequence_length"]):',
    "largest declared inference budget")
nb.cells[29] = cell_code(training)

nbformat.validate(nb)
for destination in (OUTPUT, PROJECT_COPY):
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, destination)
print(f"Wrote Submission 04 notebook ({len(nb.cells)} cells) to {OUTPUT}")



