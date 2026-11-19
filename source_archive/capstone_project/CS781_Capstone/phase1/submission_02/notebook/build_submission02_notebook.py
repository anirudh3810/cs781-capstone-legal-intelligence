from __future__ import annotations

import copy
import textwrap
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "CS781_Phase1_Submission01_Infrastructure_FIXED.ipynb"
OUTPUT = ROOT / "CS781_Phase1_Submission02_Infrastructure_FIXED.ipynb"
PROJECT_COPY = Path(__file__).resolve().parent / OUTPUT.name


def cleaned(text: str) -> str:
    return textwrap.dedent(text).strip() + "\n"


def set_markdown(nb, index: int, text: str) -> None:
    nb.cells[index] = nbformat.v4.new_markdown_cell(cleaned(text))


def set_code(nb, index: int, text: str) -> None:
    nb.cells[index] = nbformat.v4.new_code_cell(cleaned(text))


nb = nbformat.read(SOURCE, as_version=4)
nb = copy.deepcopy(nb)
nb.metadata["cs781"] = {
    "phase": 1,
    "submission": 2,
    "purpose": "Kaggle-safe encoder-decoder experiment infrastructure",
    "architecture_generated_by_llm": False,
    "student_owned_core_required": True,
}

set_markdown(
    nb,
    0,
    r"""
    # CS781 Capstone - Phase 1, Submission 02
    ## Encoder-decoder legal language model experiment

    This notebook is a separate Submission 02 workspace. It does not read from or overwrite
    Submission 01. It provides the supporting infrastructure for official-data discovery,
    schema validation, bounded preprocessing, tokenizer reuse, GPU checks, resumable training,
    checkpoint verification, inference-only restoration, resumable prediction generation,
    Codabench packaging, and recovery archives.

    **Binding academic-integrity boundary:** the capstone permits AI help with ancillary code and
    architecture understanding, but prohibits AI from writing the whole assessed architecture,
    tuning hyperparameters, or writing the scientific analysis/conclusions. Therefore the
    encoder-decoder implementation, its numerical architecture configuration, the training
    hyperparameters, hypothesis, ablation, and conclusions remain clearly marked student-owned
    inputs. The notebook will refuse full training until those inputs are supplied and attested.

    The intended substantive difference from Submission 01 is architectural: a bidirectional
    encoder conditions a causal decoder through encoder-decoder cross-attention. Changing only
    width, depth, heads, learning rate, decoding, or duration is not enough.
    """,
)

set_markdown(
    nb,
    1,
    r"""
    ## Safe execution map

    - `prepare`: inspect files, schemas, infrastructure, and saved artifacts. This is the default.
    - `fresh_train`: prepare the bounded corpus cache and start only when both explicit training
      switches are enabled. It refuses an existing compatible latest checkpoint.
    - `resume_train`: reconstruct everything and require a compatible checkpoint; it never falls
      back silently to step zero.
    - `inference_only`: restore the saved tokenizer, student model source, configuration, and
      selected checkpoint; load only the official test prompts. The training corpus and local
      evaluator are not required.

    `Run All` is safe by default: expensive preprocessing, tokenizer training, GPU smoke testing,
    full training, inference, packaging, and recovery export are all opt-in.
    """,
)

# Add imports needed by the hardened Submission 02 cells.
cell3 = nb.cells[3].source
cell3 = cell3.replace("import csv\n", "import ast\nimport csv\n")
cell3 = cell3.replace("import shutil\n", "import shutil\nimport platform\n")
nb.cells[3].source = cell3

set_code(
    nb,
    4,
    r"""
    @dataclass
    class RunConfig:
        experiment_id: str = "phase1_submission02_encoder_decoder"
        seed: int = 781
        execution_mode: str = "prepare"  # prepare | fresh_train | resume_train | inference_only
        workspace: str = field(default_factory=lambda: str(
            Path("/kaggle/working/CS781_Capstone/phase1/submission_02")
            if Path("/kaggle/working").exists()
            else Path.cwd() / "CS781_Capstone/phase1/submission_02"
        ))
        input_root: str = field(default_factory=lambda: str(
            Path("/kaggle/input") if Path("/kaggle/input").exists() else Path.cwd()
        ))
        train_files: tuple[str, ...] = (
            "train_acts.json",
            "train_high_courts.json",
            "train_supreme_court.json",
            "train_tribunals.json",
        )
        test_file: str = "Phase1- Test.json"
        test_filename_aliases: tuple[str, ...] = ("Phase1- Test.json", "test.json")
        hf_repo_id: str = "Exploration-Lab/CS-781-Capstone"
        hf_allow_patterns: tuple[str, ...] = ("*Phase_1_Pretraining/*", "*Phase-1_Pretraining/*")
        official_file_overrides: dict[str, str] = field(default_factory=dict)
        training_schema_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
        test_schema_override: dict[str, Any] | None = None

        # Reused from the Submission 01 control for a fair architecture comparison.
        validation_fraction: float = 0.05
        source_mixing_ratios: dict[str, float] = field(default_factory=lambda: {
            "acts": 0.25,
            "high_courts": 0.25,
            "supreme_court": 0.25,
            "tribunals": 0.25,
        })
        # STUDENT: choose this bounded cache size before preprocessing. None is a hard stop.
        # The cache stores only deterministically selected training documents, not the full corpus.
        corpus_document_cap_per_source: int | None = None

        # Reuse an already trained course/Submission 01 tokenizer when possible.
        tokenizer_import_path: str | None = None
        # STUDENT: path to your independently authored encoder-decoder Python module.
        student_model_source_path: str | None = None
        # Optional path to a downloaded completed notebook for the final recovery archive.
        completed_notebook_path: str | None = None

        architecture_record: dict[str, Any] = field(default_factory=lambda: {
            "family": "encoder-decoder Transformer",
            "substantive_difference_from_submission01": (
                "bidirectional encoder plus causal decoder with encoder-decoder cross-attention"
            ),
            "parameter_range": [50_000_000, 300_000_000],
            "student_model_config": "REQUIRED BEFORE TRAINING",
            "student_design_rationale": "REQUIRED BEFORE TRAINING",
        })
        innovation_hypothesis: str = "STUDENT MUST WRITE BEFORE TRAINING"
        ablation_control: str = "Submission 01 decoder-only baseline under a matched data/tokenizer/evaluation protocol"
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
            conflicts = {}
            merged = dict(existing)
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
                    f"Existing Submission 02 configuration conflict: {conflicts}. "
                    "Use a new experiment workspace instead of mutating a trained run."
                )
            atomic_write_json(RUN_CONFIG_PATH, merged)
            return merged
        atomic_write_json(RUN_CONFIG_PATH, current)
        return current

    random.seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CFG.seed)
    register_run_config(CFG)
    print({"workspace": str(ROOT), "mode": CFG.execution_mode})
    """,
)

set_code(
    nb,
    6,
    r"""
    def gpu_inventory() -> list[dict[str, Any]]:
        rows = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            rows.append({
                "index": index,
                "device": f"cuda:{index}",
                "name": props.name,
                "total_memory_gib": round(total_bytes / 2**30, 2),
                "free_memory_gib": round(free_bytes / 2**30, 2),
                "capability": f"{props.major}.{props.minor}",
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            })
        return rows

    RUNTIME = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
    }
    GPUS = gpu_inventory()
    print(json.dumps({"runtime": RUNTIME, "gpus": GPUS}, indent=2))
    if not GPUS:
        print("WARNING: no CUDA GPU detected. Full training and the representative smoke test are blocked.")
    elif len(GPUS) == 1:
        print("Submission 02 will use one explicitly selected GPU.")
    else:
        print("Multiple GPUs detected. This notebook remains single-process unless the student explicitly implements and validates DDP.")
    """,
)

set_code(
    nb,
    10,
    r"""
    MAX_IN_MEMORY_JSON_BYTES = 256 * 2**20

    def _ijson():
        ensure_runtime_dependency("ijson", "ijson>=3.3,<4")
        import ijson
        return ijson

    def detect_json_layout(path: Path) -> tuple[str, str | None]:
        ijson = _ijson()
        with Path(path).open("rb") as handle:
            parser = ijson.parse(handle)
            try:
                _, event, _ = next(parser)
            except StopIteration:
                raise ValueError(f"Empty JSON file: {path}")
            if event == "start_array":
                return "array", None
            if event != "start_map":
                raise TypeError(f"Unsupported JSON root event {event}: {path}")
            pending_key = None
            for prefix, event, value in parser:
                if prefix == "" and event == "map_key":
                    pending_key = str(value)
                    continue
                if pending_key is not None:
                    if event == "start_array":
                        return "wrapper_array", pending_key
                    if event in ("start_map", "string", "number", "boolean", "null"):
                        return "id_keyed_object", None
            return "empty_object", None

    def iter_json_records(path: Path, *, id_field: str | None = None, text_field: str | None = None):
        ijson = _ijson()
        layout, wrapper = detect_json_layout(path)
        with Path(path).open("rb") as handle:
            if layout == "array":
                for index, row in enumerate(ijson.items(handle, "item")):
                    if not isinstance(row, dict):
                        raise TypeError("Array records must be JSON objects")
                    result = dict(row)
                    result.setdefault("_record_index", index)
                    yield result
            elif layout == "wrapper_array":
                for index, row in enumerate(ijson.items(handle, f"{wrapper}.item")):
                    if not isinstance(row, dict):
                        raise TypeError("Wrapped records must be JSON objects")
                    result = dict(row)
                    result.setdefault("_record_index", index)
                    yield result
            elif layout == "id_keyed_object":
                for index, (key, value) in enumerate(ijson.kvitems(handle, "")):
                    if isinstance(value, dict):
                        row = dict(value)
                        if id_field and id_field not in row:
                            row[id_field] = str(key)
                        row.setdefault("_container_key", str(key))
                        row.setdefault("_record_index", index)
                        yield row
                    elif isinstance(value, str):
                        yield {
                            id_field or "_container_key": str(key),
                            text_field or "_container_value": value,
                            "_container_key": str(key),
                            "_record_index": index,
                        }
                    else:
                        raise TypeError(f"Unsupported top-level value under {key!r}: {type(value).__name__}")
            elif layout != "empty_object":
                raise TypeError(f"Unsupported JSON layout: {layout}")

    def sample_json_records(path: Path, limit: int = 256) -> list[dict[str, Any]]:
        import itertools
        return list(itertools.islice(iter_json_records(path), int(limit)))

    def inspect_schema(path: Path, sample_size: int = 3) -> dict[str, Any]:
        rows = sample_json_records(path, max(256, sample_size))
        keys = Counter(key for row in rows for key in row)
        layout, wrapper = detect_json_layout(path)
        return {
            "path": str(path),
            "bytes": Path(path).stat().st_size,
            "layout": layout,
            "wrapper_key": wrapper,
            "sampled_records": len(rows),
            "sample_field_presence": dict(keys),
            "samples": [
                {key: {"type": type(value).__name__, "preview": str(value)[:140]} for key, value in row.items()}
                for row in rows[:sample_size]
            ],
        }

    def infer_record_schema(path: Path, *, role: str) -> dict[str, Any]:
        rows = sample_json_records(path, 256)
        if not rows:
            raise ValueError(f"No records in {path}")
        text_candidates = (
            "_container_value", "text", "document", "content", "case_text",
            "judgment", "judgement", "prompt",
        )
        common_text = [
            name for name in text_candidates
            if all(isinstance(row.get(name), str) and row[name].strip() for row in rows)
        ]
        if len(common_text) != 1:
            raise RuntimeError(f"{role} text field is ambiguous in {path}: {common_text}")

        id_candidates = ("_container_key", "document_id", "doc_id", "case_id", "uid", "id")
        valid_ids = []
        for name in id_candidates:
            values = [row.get(name) for row in rows]
            if all(value is not None and str(value).strip() for value in values):
                if len({str(value) for value in values}) == len(values):
                    valid_ids.append(name)
        if role == "test":
            if len(valid_ids) != 1:
                raise RuntimeError(f"Official test IDs are ambiguous in {path}: {valid_ids}")
            return {"id_field": valid_ids[0], "id_strategy": "field", "text_field": common_text[0]}
        if len(valid_ids) == 1:
            return {"id_field": valid_ids[0], "id_strategy": "field", "text_field": common_text[0]}
        if not valid_ids:
            return {"id_field": None, "id_strategy": "record_index", "text_field": common_text[0]}
        raise RuntimeError(f"Training IDs are ambiguous in {path}: {valid_ids}")

    def resolve_official_schemas(cfg: RunConfig, discovered: Mapping[str, Sequence[Path]]) -> dict[str, Any]:
        training = {}
        for filename in cfg.train_files:
            inferred = infer_record_schema(require_unique_file(discovered, filename), role="training")
            override = cfg.training_schema_overrides.get(filename)
            if override is not None and dict(override) != inferred:
                raise RuntimeError(f"Schema override for {filename} conflicts with inspected data: {override} != {inferred}")
            training[filename] = inferred
        test = infer_record_schema(require_unique_file(discovered, cfg.test_file), role="test")
        if cfg.test_schema_override is not None and dict(cfg.test_schema_override) != test:
            raise RuntimeError(f"Test schema override conflicts with inspected data: {cfg.test_schema_override} != {test}")
        return {"training": training, "test": test}

    _schema_names = [*CFG.train_files, CFG.test_file]
    if all(len(DISCOVERED.get(name, [])) == 1 or name in CFG.official_file_overrides for name in _schema_names):
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
    """,
)

set_markdown(nb, 11, "## 4. Bounded official-corpus cache, provenance, deduplication, and split")

set_code(
    nb,
    12,
    r"""
    import heapq
    import zlib

    @dataclass(frozen=True)
    class Document:
        uid: str
        text: str | None
        source: str
        raw_id: str
        content_hash: str
        word_count: int
        store_path: str | None = None
        store_offset: int | None = None
        store_length: int | None = None

    def normalize_text(value: Any) -> str:
        if not isinstance(value, str):
            raise TypeError(f"Text must be a string, got {type(value).__name__}")
        result = value.replace("\r\n", "\n").strip()
        if not result:
            raise ValueError("Empty text")
        return result

    def document_text(document: Document) -> str:
        if document.text is not None:
            return document.text
        if document.store_path is None or document.store_offset is None or document.store_length is None:
            raise RuntimeError(f"Document has no text storage: {document.uid}")
        with Path(document.store_path).open("rb") as handle:
            handle.seek(document.store_offset)
            payload = handle.read(document.store_length)
        return zlib.decompress(payload).decode("utf-8")

    def fast_file_identity(path: Path, sample_bytes: int = 4 * 2**20) -> dict[str, Any]:
        path = Path(path)
        size = path.stat().st_size
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            digest.update(handle.read(sample_bytes))
            if size > sample_bytes:
                handle.seek(max(0, size - sample_bytes))
                digest.update(handle.read(sample_bytes))
        return {"name": path.name, "bytes": size, "head_tail_sha256": digest.hexdigest()}

    def stable_raw_id(row: Mapping[str, Any], index: int, schema: Mapping[str, Any]) -> str:
        if schema["id_strategy"] == "record_index":
            return f"row-{index:012d}"
        value = str(row[schema["id_field"]]).strip()
        if not value:
            raise ValueError("Empty identifier")
        return value

    def deterministic_selected_indices(
        path: Path, source: str, schema: Mapping[str, Any], cap: int, seed: int
    ) -> tuple[set[int], int]:
        if cap <= 0:
            raise ValueError("corpus_document_cap_per_source must be positive")
        heap: list[tuple[int, int]] = []
        count = 0
        for index, row in enumerate(iter_json_records(path, id_field=schema.get("id_field"), text_field=schema["text_field"])):
            raw_id = stable_raw_id(row, index, schema)
            score = int(hashlib.sha256(f"{seed}:{source}:{raw_id}".encode()).hexdigest(), 16)
            item = (-score, index)
            if len(heap) < cap:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
            count += 1
        return {index for _, index in heap}, count

    def _load_document_index(index_path: Path, store_path: Path) -> list[Document]:
        documents = []
        with Path(index_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                documents.append(Document(
                    uid=row["uid"], text=None, source=row["source"], raw_id=row["raw_id"],
                    content_hash=row["content_hash"], word_count=int(row["word_count"]),
                    store_path=str(store_path), store_offset=int(row["store_offset"]),
                    store_length=int(row["store_length"]),
                ))
        return documents

    def load_bounded_documents(
        path: Path,
        source: str,
        schema: Mapping[str, Any],
        store_dir: Path,
        cap: int,
        seed: int,
    ) -> tuple[list[Document], list[dict[str, Any]], dict[str, Any]]:
        path = Path(path)
        identity = fast_file_identity(path)
        selection_config = {
            "source_identity": identity,
            "schema": dict(schema),
            "cap": int(cap),
            "seed": int(seed),
            "selection": "smallest SHA-256(seed:source:stable_raw_id)",
            "format_version": 2,
        }
        key = hashlib.sha256(json.dumps(selection_config, sort_keys=True).encode()).hexdigest()[:20]
        store_dir = Path(store_dir)
        store_dir.mkdir(parents=True, exist_ok=True)
        store_path = store_dir / f"{source}.{key}.zbin"
        index_path = store_dir / f"{source}.{key}.index.jsonl"
        meta_path = store_dir / f"{source}.{key}.meta.json"
        if store_path.exists() and index_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("selection_config") != selection_config:
                raise RuntimeError(f"Existing cache metadata mismatch: {meta_path}")
            return _load_document_index(index_path, store_path), list(meta.get("rejections", [])), meta

        selected, total_records = deterministic_selected_indices(path, source, schema, cap, seed)
        with tempfile.NamedTemporaryFile(dir=store_dir, delete=False) as handle:
            temporary_store = Path(handle.name)
        with tempfile.NamedTemporaryFile(dir=store_dir, delete=False) as handle:
            temporary_index = Path(handle.name)
        rejections = []
        accepted = 0
        try:
            with temporary_store.open("wb") as store_handle, temporary_index.open("w", encoding="utf-8") as index_handle:
                for index, row in enumerate(iter_json_records(path, id_field=schema.get("id_field"), text_field=schema["text_field"])):
                    if index not in selected:
                        continue
                    try:
                        raw_id = stable_raw_id(row, index, schema)
                        text = normalize_text(row[schema["text_field"]])
                        encoded = text.encode("utf-8")
                        compressed = zlib.compress(encoded, level=3)
                        offset = store_handle.tell()
                        store_handle.write(compressed)
                        record = {
                            "uid": f"{source}:{raw_id}", "source": source, "raw_id": raw_id,
                            "content_hash": hashlib.sha256(encoded).hexdigest(),
                            "word_count": len(text.split()), "store_offset": offset,
                            "store_length": len(compressed),
                        }
                        index_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        accepted += 1
                    except Exception as exc:
                        rejections.append({"index": index, "error": f"{type(exc).__name__}: {exc}"})
                store_handle.flush(); os.fsync(store_handle.fileno())
                index_handle.flush(); os.fsync(index_handle.fileno())
            os.replace(temporary_store, store_path)
            os.replace(temporary_index, index_path)
        finally:
            temporary_store.unlink(missing_ok=True)
            temporary_index.unlink(missing_ok=True)
        meta = {
            "selection_config": selection_config,
            "total_source_records": total_records,
            "accepted": accepted,
            "rejected": len(rejections),
            "rejections": rejections,
            "created_utc": utc_now(),
        }
        atomic_write_json(meta_path, meta)
        return _load_document_index(index_path, store_path), rejections, meta

    def corpus_audit(docs: Sequence[Document]) -> dict[str, Any]:
        ids = Counter(document.uid for document in docs)
        hashes = Counter(document.content_hash for document in docs)
        return {
            "documents": len(docs),
            "sources": dict(Counter(document.source for document in docs)),
            "whitespace_word_count": sum(document.word_count for document in docs),
            "words_by_source": {
                source: sum(document.word_count for document in docs if document.source == source)
                for source in sorted({document.source for document in docs})
            },
            "duplicate_uids": sorted(key for key, value in ids.items() if value > 1),
            "duplicate_content_hashes": sorted(key for key, value in hashes.items() if value > 1),
        }

    def deduplicate_exact_documents(docs: Sequence[Document]) -> tuple[list[Document], list[dict[str, str]]]:
        kept, dropped, seen_ids, seen_hashes = [], [], {}, {}
        for document in sorted(docs, key=lambda value: (value.uid, value.content_hash)):
            reason = "duplicate_uid" if document.uid in seen_ids else "duplicate_exact_text" if document.content_hash in seen_hashes else None
            if reason:
                retained = seen_ids.get(document.uid) if reason == "duplicate_uid" else seen_hashes[document.content_hash]
                dropped.append({"uid": document.uid, "kept_uid": retained, "reason": reason})
                continue
            kept.append(document)
            seen_ids[document.uid] = document.uid
            seen_hashes[document.content_hash] = document.uid
        return kept, dropped

    def dataset_fingerprint(docs: Sequence[Document]) -> str:
        digest = hashlib.sha256()
        for document in sorted(docs, key=lambda value: value.uid):
            digest.update(f"{document.uid}\t{document.content_hash}\n".encode())
        return digest.hexdigest()
    """,
)

set_code(
    nb,
    14,
    r"""
    def load_official_training_corpus(cfg: RunConfig, discovered: Mapping[str, Sequence[Path]], schemas: Mapping[str, Any]):
        if cfg.corpus_document_cap_per_source is None:
            raise RuntimeError("Set CFG.corpus_document_cap_per_source explicitly before bounded preprocessing")
        ratios = validate_source_mixture(
            cfg.source_mixing_ratios,
            (Path(filename).stem.removeprefix("train_") for filename in cfg.train_files),
        )
        all_docs, rejections, provenance = [], {}, {}
        store_dir = DIRS["data"] / "bounded_document_store"
        for filename in cfg.train_files:
            path = require_unique_file(discovered, filename)
            source = Path(filename).stem.removeprefix("train_")
            docs, bad, meta = load_bounded_documents(
                path, source, schemas["training"][filename], store_dir,
                cfg.corpus_document_cap_per_source, cfg.seed,
            )
            all_docs.extend(docs)
            rejections[source] = bad
            provenance[source] = {
                "file": filename,
                "source_identity": meta["selection_config"]["source_identity"],
                "total_source_records": meta["total_source_records"],
                "selected": len(docs),
                "rejected_selected_records": len(bad),
                "selection": meta["selection_config"],
            }
        clean, dropped = deduplicate_exact_documents(all_docs)
        audit = corpus_audit(clean)
        audit.update({
            "selected_before_deduplication": len(all_docs),
            "removed_exact_duplicates": len(dropped),
            "requested_sampling_ratios": ratios,
            "sampling_unit": "document draws, not token percentages",
            "official_test_excluded": True,
        })
        train, validation, split = deterministic_document_split(clean, cfg.validation_fraction, cfg.seed)
        audit["training"] = corpus_audit(train)
        audit["validation"] = corpus_audit(validation)
        save_or_verify_split(DIRS["manifests"] / "document_split.json", split)
        overlap = sampled_cross_split_overlap_audit(train, validation, cfg.seed)
        artifacts = {
            "dataset_provenance.json": provenance,
            "dataset_audit.json": audit,
            "rejected_records.json": rejections,
            "deduplication_log.json": dropped,
            "source_sampling_config.json": {"ratios": ratios, "unit": "document draws"},
            "cross_split_overlap_audit.json": overlap,
        }
        for name, value in artifacts.items():
            path = DIRS["manifests"] / name
            if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
                raise RuntimeError(f"Persisted manifest changed: {path}")
            if not path.exists():
                atomic_write_json(path, value)
        examples = make_continuation_examples(validation, max_per_source=16, seed=cfg.seed)
        examples_path = DIRS["data"] / "local_continuation_examples.json"
        if examples_path.exists() and json.loads(examples_path.read_text(encoding="utf-8")) != examples:
            raise RuntimeError("Fixed development examples changed")
        if not examples_path.exists():
            atomic_write_json(examples_path, examples)
        atomic_write_json(DIRS["manifests"] / "local_generation_protocol.json", {
            "selection": "up to 16 eligible documents per source in deterministic hash order",
            "seed": cfg.seed, "prompt_words": 512, "reference_words": 128,
            "examples": len(examples), "ids": [row["id"] for row in examples],
        })
        return train, validation, examples, audit

    RUN_CORPUS_PREPROCESSING = False
    if CFG.execution_mode == "inference_only":
        TRAIN_DOCS = VAL_DOCS = LOCAL_DEV = DATA_AUDIT = None
        print("Inference-only mode: skipped every training-corpus load and preprocessing operation.")
    elif RUN_CORPUS_PREPROCESSING:
        if RESOLVED_SCHEMA is None:
            raise RuntimeError("Official file schemas must be resolved before preprocessing")
        TRAIN_DOCS, VAL_DOCS, LOCAL_DEV, DATA_AUDIT = load_official_training_corpus(CFG, DISCOVERED, RESOLVED_SCHEMA)
        print("Loaded bounded official corpus cache:", DATA_AUDIT)
    else:
        TRAIN_DOCS = VAL_DOCS = LOCAL_DEV = DATA_AUDIT = None
        print("Corpus preprocessing is opt-in. Review the cap and set RUN_CORPUS_PREPROCESSING=True when needed.")
    """,
)

# Keep the audited tokenizer implementation but add import-first behavior and no automatic training.
cell16 = nb.cells[16].source
cell16 = cell16.replace(
    "BPE_SOURCE_PATH=DIRS['model']/'course_bpe.py';install_python_source(BPE_SOURCE_PATH,COURSE_BPE_SOURCE)",
    """BPE_SOURCE_PATH=DIRS['model']/'course_bpe.py';install_python_source(BPE_SOURCE_PATH,COURSE_BPE_SOURCE)
course_tokenizer_import = Path(CFG.tokenizer_import_path).expanduser() if CFG.tokenizer_import_path else None
if course_tokenizer_import is not None and not (DIRS['tokenizer']/'tokenizer.json').exists():
    if not course_tokenizer_import.is_dir(): raise FileNotFoundError(course_tokenizer_import)
    if any(DIRS['tokenizer'].iterdir()): raise RuntimeError('Tokenizer destination is nonempty')
    for source_file in course_tokenizer_import.iterdir():
        if source_file.is_file(): shutil.copy2(source_file, DIRS['tokenizer']/source_file.name)
    print('Imported the frozen Submission 01/course tokenizer from',course_tokenizer_import)""",
)
cell16 = cell16.replace(
    "elif TRAIN_DOCS and RUN_TOKENIZER_TRAINING:",
    "elif TRAIN_DOCS is not None and RUN_TOKENIZER_TRAINING:",
)
cell16 = cell16.replace(
    "No saved tokenizer found. Review TOKENIZER_TRAINING_CONFIG, then set RUN_TOKENIZER_TRAINING=True once.",
    "No saved/imported tokenizer found. Set CFG.tokenizer_import_path, or explicitly authorize one tokenizer-training run.",
)
nb.cells[16].source = cell16

set_markdown(
    nb,
    17,
    r"""
    ## 6. Student-owned encoder-decoder model and enforced contract

    For contiguous training tokens split into source `x` and target `y`, the intended model family is:

    - `H = Encoder(x)`, where encoder self-attention may use both left and right source context.
    - The decoder uses causal self-attention over `y_<t` and cross-attention to `H`.
    - `p(y_t | y_<t, x) = softmax(W h_t)`.
    - Teacher-forced loss is the mean negative log-likelihood over non-padding target tokens.

    This description establishes the experiment contract, not its assessed implementation. Your model module must
    independently define `build_model(config)` and either expose a standard forward method accepting
    `encoder_input_ids`, `decoder_input_ids`, and `encoder_attention_mask`, or define
    `forward_adapter(model, encoder_input_ids, decoder_input_ids, encoder_attention_mask)`.
    The exact layer design, mathematical derivation, rationale, and numerical configuration must be yours.
    """,
)

set_code(
    nb,
    18,
    r"""
    STUDENT_MODEL_CONFIG_OVERRIDE = None  # STUDENT may provide a dict after writing the architecture.
    MODEL = MODEL_CONFIG = RECONSTRUCT_MODEL = MODEL_MODULE = None
    MODEL_SOURCE_PATH = MODEL_SOURCE_SHA256 = None

    def install_student_model_source(source_path: Path) -> Path:
        source_path = Path(source_path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
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
            raise RuntimeError(f"Pretrained-model libraries are forbidden in the submitted model source: {imported & banned}")
        destination = DIRS["model"] / "encoder_decoder_lm.py"
        payload = source_path.read_bytes()
        if destination.exists() and destination.read_bytes() != payload and any(DIRS["checkpoints"].glob("*.pt")):
            raise RuntimeError("Refusing to change model source after checkpoints exist")
        atomic_write_bytes(destination, payload)
        return destination

    def seq2seq_logits(model, encoder_input_ids, decoder_input_ids, encoder_attention_mask=None):
        if MODEL_MODULE is not None and callable(getattr(MODEL_MODULE, "forward_adapter", None)):
            output = MODEL_MODULE.forward_adapter(model, encoder_input_ids, decoder_input_ids, encoder_attention_mask)
        else:
            output = model(
                encoder_input_ids=encoder_input_ids,
                decoder_input_ids=decoder_input_ids,
                encoder_attention_mask=encoder_attention_mask,
            )
        return output.logits if hasattr(output, "logits") else output[0] if isinstance(output, (tuple, list)) else output

    def count_parameters(model: torch.nn.Module) -> dict[str, int]:
        return {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        }

    def validate_seq2seq_contract(model, vocab_size: int, device: str, source_length: int = 12, target_length: int = 7):
        model = model.to(device)
        model.train(); model.zero_grad(set_to_none=True)
        encoder_ids = torch.randint(0, vocab_size, (2, source_length), dtype=torch.long, device=device)
        decoder_ids = torch.randint(0, vocab_size, (2, target_length), dtype=torch.long, device=device)
        encoder_mask = torch.ones_like(encoder_ids, dtype=torch.bool)
        logits = seq2seq_logits(model, encoder_ids, decoder_ids, encoder_mask)
        expected = (2, target_length, vocab_size)
        if tuple(logits.shape) != expected:
            raise ValueError(f"Expected logits shape {expected}, got {tuple(logits.shape)}")
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite logits")
        labels = torch.randint(0, vocab_size, (2, target_length), dtype=torch.long, device=device)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab_size), labels.reshape(-1))
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
            raise FloatingPointError("Missing or non-finite gradients")
        counts = count_parameters(model)
        if not 50_000_000 <= counts["total"] <= 300_000_000:
            raise ValueError(f"Official parameter bound violated: {counts['total']:,}")
        model.zero_grad(set_to_none=True)
        return {"logits_shape": list(logits.shape), "loss": float(loss.detach().cpu()), **counts, "device": device}

    _restored_model_source = DIRS["model"] / "encoder_decoder_lm.py"
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
        MODEL_MODULE = import_source_module(f"cs781_submission02_model_{MODEL_SOURCE_SHA256[:12]}", MODEL_SOURCE_PATH)
        if not callable(getattr(MODEL_MODULE, "build_model", None)):
            raise AttributeError("Student module must define build_model(config)")
        if STUDENT_MODEL_CONFIG_OVERRIDE is not None:
            MODEL_CONFIG = dict(STUDENT_MODEL_CONFIG_OVERRIDE)
        elif callable(getattr(MODEL_MODULE, "make_model_config", None)):
            MODEL_CONFIG = dict(MODEL_MODULE.make_model_config(int(TOKENIZER_META["vocab_size"])))
        else:
            candidate = getattr(MODEL_MODULE, "MODEL_CONFIG", None)
            if not isinstance(candidate, Mapping):
                raise RuntimeError("Provide STUDENT_MODEL_CONFIG_OVERRIDE, make_model_config(vocab_size), or module MODEL_CONFIG")
            MODEL_CONFIG = dict(candidate)
        if int(MODEL_CONFIG.get("vocab_size", -1)) != int(TOKENIZER_META["vocab_size"]):
            raise ValueError("Model vocabulary does not match the frozen tokenizer")
        RECONSTRUCT_MODEL = MODEL_MODULE.build_model
        MODEL = RECONSTRUCT_MODEL(MODEL_CONFIG)
        MODEL_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
        MODEL_REPORT = validate_seq2seq_contract(MODEL, TOKENIZER_META["vocab_size"], MODEL_DEVICE)
        MODEL_REPORT.update({
            "architecture_family": "encoder-decoder Transformer",
            "model_source_sha256": MODEL_SOURCE_SHA256,
            "model_config": MODEL_CONFIG,
        })
        atomic_write_json(DIRS["manifests"] / "model_validation.json", MODEL_REPORT)
        MODEL.to("cpu")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print("Student model validation:", json.dumps(MODEL_REPORT, indent=2))
    else:
        MODEL_REPORT = None
        print("STUDENT ACTION REQUIRED: set CFG.student_model_source_path to the independently authored encoder-decoder module.")
    """,
)

# Submission-specific recovery filename.
cell24 = nb.cells[24].source.replace("phase1_submission01_recovery.zip", "phase1_submission02_recovery.zip")
nb.cells[24].source = cell24

set_code(
    nb,
    26,
    r"""
    def run_infrastructure_smoke_tests() -> dict[str, str]:
        results: dict[str, str] = {}
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            fixture_texts = [
                " ".join(["alpha"] * 700),
                " ".join(["beta"] * 700),
                " ".join(["gamma"] * 700),
            ]
            docs = [
                Document(
                    uid=f"fixture:row-{index:012d}", text=text, source="fixture",
                    raw_id=f"row-{index:012d}", content_hash=hashlib.sha256(text.encode()).hexdigest(),
                    word_count=len(text.split()),
                )
                for index, text in enumerate(fixture_texts)
            ]
            assert stable_raw_id({"_record_index": 0}, 0, {"id_strategy": "record_index", "id_field": None}) == "row-000000000000"
            results["idless_training_schema_strategy"] = "PASS"
            train, validation, split = deterministic_document_split(docs, 1 / 3, 781)
            assert not ({doc.uid for doc in train} & {doc.uid for doc in validation})
            save_or_verify_split(temporary / "split.json", split)
            results["document_split_and_leakage"] = "PASS"
            examples = make_continuation_examples(validation)
            assert len(examples) == 1 and len(examples[0]["prompt"].split()) == 512
            assert len(examples[0]["reference"].split()) == 128
            results["local_continuation_examples"] = "PASS"
            sampler = StatefulRandomSampler(range(10), seed=9)
            iterator = iter(sampler); prefix = [next(iterator) for _ in range(4)]
            state = sampler.state_dict(); resumed = StatefulRandomSampler(range(10), seed=9)
            resumed.load_state_dict(state); suffix = list(resumed)
            assert len(prefix + suffix) == 10 and len(set(prefix + suffix)) == 10
            results["sampler_continuation"] = "PASS"
            history = JsonlHistory(temporary / "history.jsonl")
            history.append({"step": 1, "loss": 1.0})
            assert history.records() == [{"step": 1, "loss": 1.0}]
            results["history_continuity"] = "PASS"
            dummy_prediction = {"1": {"text": " ".join(["word"] * 128)}}
            prediction_path = temporary / "prediction.json"; atomic_write_json(prediction_path, dummy_prediction)
            zip_path = temporary / "submission.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.write(prediction_path, "prediction.json")
            assert validate_zip(zip_path)["members"] == ["prediction.json"]
            results["zip_integrity"] = "PASS"
        return results

    SMOKE_RESULTS = run_infrastructure_smoke_tests()
    print(json.dumps(SMOKE_RESULTS, indent=2))
    """,
)

set_code(
    nb,
    27,
    r"""
    def validate_decoder_causality(model, vocab_size: int, device: str) -> None:
        model.eval()
        encoder = torch.randint(0, vocab_size, (1, 11), dtype=torch.long, device=device)
        first = torch.randint(0, vocab_size, (1, 9), dtype=torch.long, device=device)
        second = first.clone(); second[:, 6:] = torch.randint(0, vocab_size, (1, 3), device=device)
        mask = torch.ones_like(encoder, dtype=torch.bool)
        with torch.no_grad():
            first_logits = seq2seq_logits(model, encoder, first, mask)[:, :6]
            second_logits = seq2seq_logits(model, encoder, second, mask)[:, :6]
        if not torch.allclose(first_logits, second_logits, atol=2e-5, rtol=2e-5):
            raise AssertionError("Decoder future-token leakage detected")

    def validate_encoder_conditioning(model, vocab_size: int, device: str) -> None:
        model.eval()
        first_encoder = torch.randint(0, vocab_size, (1, 11), dtype=torch.long, device=device)
        second_encoder = torch.randint(0, vocab_size, (1, 11), dtype=torch.long, device=device)
        decoder = torch.randint(0, vocab_size, (1, 7), dtype=torch.long, device=device)
        mask = torch.ones_like(first_encoder, dtype=torch.bool)
        with torch.no_grad():
            a = seq2seq_logits(model, first_encoder, decoder, mask)
            b = seq2seq_logits(model, second_encoder, decoder, mask)
        if torch.allclose(a, b, atol=1e-7, rtol=1e-7):
            raise AssertionError("Model output is insensitive to encoder input; cross-attention may be disconnected")

    def run_student_integration_smoke_test(model, tokenizer, model_config, reconstruct_model, device=None):
        if MODEL_SOURCE_SHA256 is None:
            raise RuntimeError("Student model source is not registered")
        device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        meta = tokenizer_metadata(tokenizer, DIRS["tokenizer"])
        report = validate_seq2seq_contract(model, meta["vocab_size"], device)
        validate_decoder_causality(model, meta["vocab_size"], device)
        validate_encoder_conditioning(model, meta["vocab_size"], device)
        report["decoder_causality"] = "PASS"
        report["encoder_conditioning"] = "PASS"
        probe_ids = tokenizer_encode(tokenizer, "A constitutional court considered the appeal.")
        if not probe_ids or min(probe_ids) < 0 or max(probe_ids) >= meta["vocab_size"]:
            raise AssertionError("Tokenizer emitted an out-of-vocabulary ID")
        clone = reconstruct_model(model_config)
        clone.load_state_dict(model.state_dict(), strict=True)
        report["reconstruction"] = "PASS"
        report["model_source_sha256"] = MODEL_SOURCE_SHA256
        del clone
        model.to("cpu")
        if str(device).startswith("cuda"): torch.cuda.empty_cache()
        atomic_write_json(DIRS["manifests"] / "student_integration_smoke_test.json", report)
        return report

    if MODEL is not None and tokenizer is not None:
        INTEGRATION_SMOKE_RESULTS = run_student_integration_smoke_test(
            MODEL, tokenizer, MODEL_CONFIG, RECONSTRUCT_MODEL
        )
        print(json.dumps(INTEGRATION_SMOKE_RESULTS, indent=2))
    else:
        INTEGRATION_SMOKE_RESULTS = None
        print("Student encoder-decoder integration smoke test is waiting for model and tokenizer registration.")
    """,
)

set_markdown(nb, 28, "## 10. Resumable encoder-decoder teacher-forcing training")

set_code(
    nb,
    29,
    r"""
    # STUDENT-OWNED numerical choices. None values are deliberate hard stops, not defaults.
    TRAINING_CONFIG = {
        "experiment_id": CFG.experiment_id,
        "objective": "teacher-forced target-token cross-entropy conditioned on a bidirectional source encoding",
        "optimizer": None,
        "learning_rate": None,
        "betas": None,
        "epsilon": None,
        "weight_decay": None,
        "schedule": None,
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
            "optimizer", "learning_rate", "betas", "epsilon", "weight_decay", "schedule",
            "warmup_steps", "min_lr_ratio", "batch_size", "gradient_accumulation_steps",
            "source_length", "target_length", "max_steps", "samples_per_epoch",
            "gradient_clip_norm", "gradient_checkpointing", "validation_interval",
            "validation_sequences_per_source",
        )
        missing = [key for key in required if config.get(key) is None]
        if missing:
            raise RuntimeError(f"Student training configuration is incomplete: {missing}")
        if config["optimizer"] != "AdamW":
            raise ValueError("This infrastructure currently validates AdamW only; document and implement another optimizer explicitly")
        if config["schedule"] != "linear_warmup_cosine_decay":
            raise ValueError("This infrastructure currently validates linear_warmup_cosine_decay only")
        positive = (
            "learning_rate", "epsilon", "batch_size", "gradient_accumulation_steps", "source_length",
            "target_length", "max_steps", "samples_per_epoch", "gradient_clip_norm",
            "validation_interval", "validation_sequences_per_source", "checkpoint_interval", "recovery_interval",
        )
        if any(float(config[key]) <= 0 for key in positive):
            raise ValueError("Configured positive quantities must be > 0")
        if int(config["num_workers"]) != 0:
            raise ValueError("num_workers must remain 0 for exact committed sampler restoration")
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

    class MixedDocumentSeq2SeqDataset(Dataset):
        def __init__(self, documents, tokenizer, ratios, samples, source_length, target_length, seed, cache_documents=128):
            self.documents = list(documents); self.tokenizer = tokenizer; self.ratios = dict(ratios)
            self.samples = int(samples); self.source_length = int(source_length); self.target_length = int(target_length)
            self.seed = int(seed); self.epoch = 0; self.cache_documents = int(cache_documents); self.cache = {}
            self.pad_id = getattr(tokenizer, "pad_token_id", None)
            self.bos_id = getattr(tokenizer, "bos_token_id", None)
            self.eos_id = getattr(tokenizer, "eos_token_id", None)
            if None in (self.pad_id, self.bos_id, self.eos_id):
                raise ValueError("Seq2seq training requires distinct pad, bos, and eos token IDs")
            if len({int(self.pad_id), int(self.bos_id), int(self.eos_id)}) != 3:
                raise ValueError("pad, bos, and eos token IDs must be distinct")
            self._resample()
        def _resample(self):
            self.document_indices = source_sampling_indices(self.documents, self.ratios, self.samples, self.seed + self.epoch)
        def __len__(self): return self.samples
        def set_epoch(self, epoch: int): self.epoch = int(epoch); self._resample()
        def _tokens(self, document_index: int) -> list[int]:
            if document_index not in self.cache:
                if len(self.cache) >= self.cache_documents: self.cache.pop(next(iter(self.cache)))
                self.cache[document_index] = tokenizer_encode(self.tokenizer, document_text(self.documents[document_index]))
            return self.cache[document_index]
        def __getitem__(self, index: int):
            document_index = self.document_indices[int(index)]
            ids = self._tokens(document_index)
            needed = self.source_length + self.target_length
            if len(ids) >= needed:
                rng = random.Random(self.seed + self.epoch * len(self) + int(index))
                start = rng.randrange(0, len(ids) - needed + 1)
                source = list(ids[start:start + self.source_length])
                target = list(ids[start + self.source_length:start + needed])
            elif len(ids) >= 2:
                split = min(self.source_length, max(1, len(ids) - 1))
                source = list(ids[:split])
                target = list(ids[split:split + self.target_length])
            else:
                source = list(ids) or [int(self.eos_id)]
                target = []
            target = [*target, int(self.eos_id)][:self.target_length]
            source_valid = len(source); target_valid = len(target)
            source = (source + [int(self.pad_id)] * self.source_length)[:self.source_length]
            labels = (target + [-100] * self.target_length)[:self.target_length]
            decoder_input = [int(self.bos_id)] + [token for token in target[:-1]]
            decoder_input = (decoder_input + [int(self.pad_id)] * self.target_length)[:self.target_length]
            return {
                "encoder_input_ids": torch.tensor(source, dtype=torch.long),
                "encoder_attention_mask": torch.tensor([True] * source_valid + [False] * (self.source_length - source_valid)),
                "decoder_input_ids": torch.tensor(decoder_input, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "document_index": document_index,
                "processed_tokens": source_valid + target_valid,
            }

    def build_fixed_seq2seq_validation_rows(documents, tokenizer, config):
        rows = []
        per_source = int(config["validation_sequences_per_source"])
        for source in sorted({document.source for document in documents}):
            source_documents = [document for document in documents if document.source == source]
            dataset = MixedDocumentSeq2SeqDataset(
                source_documents, tokenizer, {source: 1.0}, per_source,
                int(config["source_length"]), int(config["target_length"]),
                int(config["seed"]) + 10_000,
            )
            rows.extend(dataset[index] for index in range(len(dataset)))
        manifest = [
            {"document_index": int(row["document_index"]), "processed_tokens": int(row["processed_tokens"])}
            for row in rows
        ]
        return rows, manifest

    def choose_precision(device: str):
        if not str(device).startswith("cuda"):
            return None, "fp32", False
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, "bf16", False
        return torch.float16, "fp16", True

    def make_scaler(enabled: bool):
        try: return torch.amp.GradScaler("cuda", enabled=enabled)
        except (AttributeError, TypeError): return torch.cuda.amp.GradScaler(enabled=enabled)

    def scaler_update_succeeded(scale_before: float, scale_after: float) -> bool:
        return float(scale_after) >= float(scale_before)

    def learning_rate_multiplier(step: int, max_steps: int, warmup_steps: int, min_ratio: float) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
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
            "format": "phase1_submission02_seq2seq_inference_v1", "created_utc": utc_now(),
            "model_state": model.state_dict(), "model_config": dict(model_config),
            "model_source_sha256": MODEL_SOURCE_SHA256,
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
        return {"path": str(path), "sha256": sha256_file(path), "step": step}

    def export_training_recovery_snapshot(root: Path) -> Path:
        root = Path(root); output = DIRS["recovery"] / "training_recovery.latest.zip"
        include = []
        for relative in ("model", "tokenizer", "manifests", "histories"):
            include.extend(path for path in (root / relative).rglob("*") if path.is_file())
        for name in ("checkpoint.latest.pt", "previous_known_good.pt", "selected_model.pt", "model_final.pt"):
            path = DIRS["checkpoints"] / name
            if path.exists(): include.append(path)
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
                for path in sorted(set(include)): archive.write(path, path.relative_to(root).as_posix())
            validate_zip(temporary); os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        atomic_write_json(DIRS["recovery"] / "training_recovery.latest.json", {
            "created_utc": utc_now(), "path": str(output), "sha256": sha256_file(output), "bytes": output.stat().st_size,
        })
        return output

    @torch.no_grad()
    def evaluate_seq2seq_loss(model, loader, device, amp_dtype):
        was_training = model.training; model.eval(); loss_sum = 0.0; token_count = 0
        for batch in loader:
            encoder = batch["encoder_input_ids"].to(device)
            encoder_mask = batch["encoder_attention_mask"].to(device)
            decoder = batch["decoder_input_ids"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type=torch.device(device).type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = seq2seq_logits(model, encoder, decoder, encoder_mask)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction="sum"
                )
            loss_sum += float(loss.cpu()); token_count += int((labels != -100).sum())
        model.train(was_training)
        if not token_count: raise ValueError("Validation loader contained no targets")
        return loss_sum / token_count

    def train_submission02(model, tokenizer, train_docs, val_docs, model_config, tokenizer_meta, *, device="cuda:0", resume=False):
        cfg = validate_student_training_config(TRAINING_CONFIG)
        if not CFG.student_configuration_confirmed or not CFG.student_owned_architecture_attestation:
            raise RuntimeError("Confirm and attest the student-owned architecture/training configuration first")
        latest = DIRS["checkpoints"] / "checkpoint.latest.pt"
        if resume and not latest.exists():
            raise FileNotFoundError("resume_train requires checkpoint.latest.pt; refusing to start from step zero")
        if not resume and latest.exists():
            raise RuntimeError("fresh_train found an existing checkpoint; use resume_train or a new workspace")
        save_or_verify_json(DIRS["manifests"] / "training_config.json", cfg)
        split_meta = json.loads((DIRS["manifests"] / "document_split.json").read_text(encoding="utf-8"))
        dataset_fp = split_meta["dataset_fingerprint"]
        ratios = validate_source_mixture(cfg["source_mixture"], (doc.source for doc in train_docs))
        dataset = MixedDocumentSeq2SeqDataset(
            train_docs, tokenizer, ratios, cfg["samples_per_epoch"], cfg["source_length"],
            cfg["target_length"], cfg["seed"],
        )
        sampler = StatefulRandomSampler(dataset, seed=cfg["seed"])
        loader = DataLoader(dataset, batch_size=cfg["batch_size"], sampler=sampler, num_workers=0,
                            pin_memory=str(device).startswith("cuda"), drop_last=True)
        validation_rows, validation_manifest = build_fixed_seq2seq_validation_rows(val_docs, tokenizer, cfg)
        save_or_verify_json(DIRS["manifests"] / "validation_sequence_manifest.json", {
            "protocol": "fixed source-conditioned target sequences", "rows": validation_manifest,
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
            optimizer, lambda step: learning_rate_multiplier(step, cfg["max_steps"], cfg["warmup_steps"], cfg["min_lr_ratio"])
        )
        amp_dtype, actual_precision, use_scaler = choose_precision(device)
        scaler = make_scaler(use_scaler); checkpoint_scaler = scaler if use_scaler else None
        history_path = DIRS["histories"] / "training.jsonl"; history = JsonlHistory(history_path)
        progress = ProgressState(cfg["experiment_id"], device, cfg["max_steps"])
        validation_history = []
        if resume:
            preview = torch.load(latest, map_location="cpu", weights_only=False)
            if preview.get("training_config") != cfg: raise RuntimeError("Checkpoint training config differs")
            if preview.get("model_source_sha256") != MODEL_SOURCE_SHA256: raise RuntimeError("Checkpoint model source differs")
            reconcile_history_to_checkpoint(latest, history_path, apply=True)
            state = restore_checkpoint(
                latest, model=model, optimizer=optimizer, scheduler=scheduler, scaler=checkpoint_scaler,
                sampler=sampler, expected_tokenizer_fingerprint=tokenizer_meta["fingerprint"],
                expected_dataset_fingerprint=dataset_fp, expected_experiment_id=cfg["experiment_id"],
                expected_model_config=model_config, history_path=history_path, device=device,
                expected_model_source_sha256=MODEL_SOURCE_SHA256,
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
                model.train(); optimizer.zero_grad(set_to_none=True); microbatches = []; valid_targets = 0; processed_tokens = 0
                for _ in range(cfg["gradient_accumulation_steps"]):
                    try: batch = next(iterator)
                    except StopIteration:
                        sampler.next_epoch(); dataset.set_epoch(sampler.epoch); iterator = iter(loader); batch = next(iterator)
                    valid_targets += int((batch["labels"] != -100).sum())
                    processed_tokens += int(batch["processed_tokens"].sum())
                    microbatches.append(batch)
                if valid_targets <= 0: raise ValueError("Accumulation window has no target tokens")
                numerator = 0.0
                for batch in microbatches:
                    encoder = batch["encoder_input_ids"].to(device)
                    encoder_mask = batch["encoder_attention_mask"].to(device)
                    decoder = batch["decoder_input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    with torch.autocast(device_type=torch.device(device).type, dtype=amp_dtype, enabled=amp_dtype is not None):
                        logits = seq2seq_logits(model, encoder, decoder, encoder_mask)
                        loss_sum = torch.nn.functional.cross_entropy(
                            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction="sum"
                        )
                        backward_loss = loss_sum / valid_targets
                    if not torch.isfinite(loss_sum): raise FloatingPointError("Non-finite training loss")
                    (scaler.scale(backward_loss) if use_scaler else backward_loss).backward()
                    numerator += float(loss_sum.detach().cpu())
                if use_scaler: scaler.unscale_(optimizer)
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"]).cpu())
                learning_rate = float(optimizer.param_groups[0]["lr"])
                if use_scaler:
                    before = float(scaler.get_scale()); scaler.step(optimizer); scaler.update()
                    if not scaler_update_succeeded(before, scaler.get_scale()):
                        append_event(DIRS["histories"] / "overflow_events.jsonl", {
                            "attempted_step": reporter.state.step + 1,
                            "epoch": sampler.epoch,
                            "scale_before": before,
                            "scale_after": float(scaler.get_scale()),
                            "created_utc": utc_now(),
                        })
                        continue
                else:
                    optimizer.step()
                scheduler.step()
                step = reporter.state.step + 1
                total_tokens = reporter.state.tokens_processed + processed_tokens
                mean_loss = numerator / valid_targets
                validation_loss = None; improved = False
                if step % cfg["validation_interval"] == 0 or step == cfg["max_steps"]:
                    validation_loss = evaluate_seq2seq_loss(model, validation_loader, device, amp_dtype)
                    validation_history.append({"step": step, "loss": validation_loss})
                    improved = validation_loss < best_loss; best_loss = min(best_loss, validation_loss)
                reporter.update(step=step, epoch=sampler.epoch, loss=mean_loss, learning_rate=learning_rate,
                                tokens_processed=total_tokens, gradient_norm=gradient_norm)
                history.append({
                    "step": step, "epoch": sampler.epoch, "loss": mean_loss,
                    "validation_loss": validation_loss, "learning_rate": learning_rate,
                    "gradient_norm": gradient_norm, "tokens_processed": total_tokens,
                    "tokens_per_second": total_tokens / max(reporter.state.cumulative_seconds, 1e-9),
                    "gpu_memory_gib": torch.cuda.max_memory_allocated(torch.device(device)) / 2**30 if str(device).startswith("cuda") else None,
                    "precision": actual_precision, "checkpoint_step": reporter.state.last_checkpoint_step,
                    "created_utc": utc_now(),
                })
                if step % cfg["checkpoint_interval"] == 0 or improved or step == cfg["max_steps"]:
                    reporter.state.last_checkpoint_step = step
                    save_checkpoint_atomic(
                        latest, model=model, optimizer=optimizer, scheduler=scheduler, scaler=checkpoint_scaler,
                        sampler=sampler, progress=asdict(reporter.state), model_config=model_config,
                        tokenizer_fingerprint=tokenizer_meta["fingerprint"], dataset_fingerprint_value=dataset_fp,
                        training_config=cfg, history_path=history_path, validation_history=validation_history,
                        experiment_id=cfg["experiment_id"], device=device, model_source_sha256=MODEL_SOURCE_SHA256,
                    )
                    if improved:
                        save_inference_checkpoint(DIRS["checkpoints"] / "selected_model.pt", model, model_config,
                                                  tokenizer_meta, dataset_fp, step, validation_loss)
                    if step % cfg["recovery_interval"] == 0:
                        export_training_recovery_snapshot(ROOT)
            save_inference_checkpoint(DIRS["checkpoints"] / "model_final.pt", model, model_config,
                                      tokenizer_meta, dataset_fp, reporter.state.step,
                                      validation_history[-1]["loss"] if validation_history else None)
            if not (DIRS["checkpoints"] / "selected_model.pt").exists():
                atomic_copy(DIRS["checkpoints"] / "model_final.pt", DIRS["checkpoints"] / "selected_model.pt")
            export_training_recovery_snapshot(ROOT)
        finally:
            reporter.close()
        result = {
            "completed_step": reporter.state.step, "tokens_processed": reporter.state.tokens_processed,
            "best_validation_loss": best_loss, "actual_precision": actual_precision,
            "latest_checkpoint": str(latest), "selected_checkpoint": str(DIRS["checkpoints"] / "selected_model.pt"),
        }
        atomic_write_json(DIRS["manifests"] / "training_result.json", result)
        return result

    def run_gpu_training_smoke_test(device="cuda:0"):
        if not torch.cuda.is_available(): raise RuntimeError("Representative smoke test requires CUDA")
        cfg = validate_student_training_config(TRAINING_CONFIG)
        if MODEL is None or tokenizer is None or not TRAIN_DOCS or not VAL_DOCS:
            raise RuntimeError("Model, tokenizer, and bounded corpus cache are required")
        validation_rows, validation_manifest = build_fixed_seq2seq_validation_rows(VAL_DOCS, tokenizer, cfg)
        save_or_verify_json(DIRS["manifests"] / "validation_sequence_manifest.json", {
            "protocol": "fixed source-conditioned target sequences", "rows": validation_manifest,
        })
        tiny = MixedDocumentSeq2SeqDataset(
            TRAIN_DOCS, tokenizer, cfg["source_mixture"], max(2, cfg["batch_size"] * 2),
            cfg["source_length"], cfg["target_length"], cfg["seed"],
        )
        smoke_sampler = StatefulRandomSampler(tiny, seed=cfg["seed"])
        batch = next(iter(DataLoader(tiny, batch_size=cfg["batch_size"], sampler=smoke_sampler, num_workers=0)))
        smoke_model = RECONSTRUCT_MODEL(MODEL_CONFIG).to(device)
        optimizer = torch.optim.AdamW(smoke_model.parameters(), lr=cfg["learning_rate"], betas=tuple(cfg["betas"]),
                                      eps=cfg["epsilon"], weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: learning_rate_multiplier(step, 10, 1, cfg["min_lr_ratio"])
        )
        amp_dtype, precision, use_scaler = choose_precision(device); scaler = make_scaler(use_scaler)
        torch.cuda.reset_peak_memory_stats(torch.device(device)); optimizer.zero_grad(set_to_none=True)
        encoder = batch["encoder_input_ids"].to(device); mask = batch["encoder_attention_mask"].to(device)
        decoder = batch["decoder_input_ids"].to(device); labels = batch["labels"].to(device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = seq2seq_logits(smoke_model, encoder, decoder, mask)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
        (scaler.scale(loss) if use_scaler else loss).backward()
        if use_scaler: scaler.unscale_(optimizer)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(smoke_model.parameters(), cfg["gradient_clip_norm"]).cpu())
        if use_scaler:
            before = float(scaler.get_scale()); scaler.step(optimizer); scaler.update()
            if not scaler_update_succeeded(before, scaler.get_scale()): raise RuntimeError("FP16 smoke update overflowed")
        else: optimizer.step()
        scheduler.step()
        loss_value = float(loss.detach().cpu())
        split_meta = json.loads((DIRS["manifests"] / "document_split.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            history = JsonlHistory(temporary / "history.jsonl")
            history.append({"step": 1, "loss": loss_value})
            checkpoint = temporary / "smoke.pt"
            progress = ProgressState(
                "submission02_gpu_smoke", device, 1, step=1,
                tokens_processed=int(batch["processed_tokens"].sum()), last_checkpoint_step=1,
            )
            save_checkpoint_atomic(
                checkpoint, model=smoke_model, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler if use_scaler else None, sampler=smoke_sampler,
                progress=asdict(progress), model_config=MODEL_CONFIG,
                tokenizer_fingerprint=TOKENIZER_META["fingerprint"],
                dataset_fingerprint_value=split_meta["dataset_fingerprint"],
                training_config={"smoke": True}, history_path=history.path,
                validation_history=[], experiment_id="submission02_gpu_smoke", device=device,
                model_source_sha256=MODEL_SOURCE_SHA256,
            )
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
                expected_experiment_id="submission02_gpu_smoke", expected_model_config=MODEL_CONFIG,
                history_path=history.path, device=None,
                expected_model_source_sha256=MODEL_SOURCE_SHA256,
            )
            if restored["progress"]["step"] != 1 or not clone_optimizer.state:
                raise AssertionError("Checkpoint did not restore model, optimizer, or progress")
        report = {
            "status": "PASS", "loss": loss_value, "gradient_norm": gradient_norm,
            "precision": precision, "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(torch.device(device)) / 2**30,
            "source_length": cfg["source_length"], "target_length": cfg["target_length"],
            "batch_size": cfg["batch_size"], "checkpoint_restore": "PASS", "created_utc": utc_now(),
        }
        atomic_write_json(DIRS["manifests"] / "gpu_training_smoke_test.json", report)
        del smoke_model, optimizer, logits, loss, clone, clone_optimizer; torch.cuda.empty_cache()
        return report

    RUN_GPU_TRAINING_SMOKE_TEST = False
    GPU_TRAINING_SMOKE_RESULT = run_gpu_training_smoke_test("cuda:0") if RUN_GPU_TRAINING_SMOKE_TEST else None

    def preflight_full_training():
        problems = []
        if CFG.execution_mode not in {"fresh_train", "resume_train"}: problems.append("execution_mode is not a training mode")
        if not CFG.full_training_authorized: problems.append("CFG.full_training_authorized is False")
        if not CFG.student_owned_architecture_attestation: problems.append("student architecture attestation is False")
        if not CFG.student_configuration_confirmed: problems.append("student configuration confirmation is False")
        if not CFG.trained_from_scratch_attestation: problems.append("scratch-training attestation is False")
        try: validate_student_training_config(TRAINING_CONFIG)
        except Exception as exc: problems.append(str(exc))
        if MODEL is None or tokenizer is None: problems.append("model/tokenizer unavailable")
        if not TRAIN_DOCS or not VAL_DOCS: problems.append("bounded training/validation cache unavailable")
        for name in ("document_split.json", "model_validation.json", "gpu_training_smoke_test.json", "validation_sequence_manifest.json"):
            if not (DIRS["manifests"] / name).exists(): problems.append(f"missing manifest {name}")
        if problems: raise RuntimeError("Full-training preflight failed:\n- " + "\n- ".join(problems))

    RUN_FULL_TRAINING = False
    if RUN_FULL_TRAINING:
        preflight_full_training()
        TRAINING_RESULT = train_submission02(
            MODEL, tokenizer, TRAIN_DOCS, VAL_DOCS, MODEL_CONFIG, TOKENIZER_META,
            device="cuda:0", resume=CFG.execution_mode == "resume_train",
        )
    else:
        TRAINING_RESULT = None
        print("Training is opt-in and blocked until the student-owned core and representative GPU smoke test pass.")
    """,
)

# Checkpoint source identity is part of compatibility in Submission 02.
cell21 = nb.cells[21].source
cell21 = cell21.replace(
    "experiment_id:str,device:str|None=None)->dict[str,Any]:",
    "experiment_id:str,device:str|None=None,model_source_sha256:str|None=None)->dict[str,Any]:",
)
cell21 = cell21.replace(
    "'validation_history':list(validation_history)}",
    "'validation_history':list(validation_history),'model_source_sha256':model_source_sha256}",
)
nb.cells[21].source = cell21

cell22 = nb.cells[22].source
cell22 = cell22.replace(
    "history_path:Path|None=None,device:str|None=None)->dict[str,Any]:",
    "history_path:Path|None=None,device:str|None=None,expected_model_source_sha256:str|None=None)->dict[str,Any]:",
)
cell22 = cell22.replace(
    "if expected_model_config is not None and dict(expected_model_config)!=state['model_config']: raise ValueError('Architecture config mismatch')",
    "if expected_model_config is not None and dict(expected_model_config)!=state['model_config']: raise ValueError('Architecture config mismatch')\n    if expected_model_source_sha256 is not None and state.get('model_source_sha256')!=expected_model_source_sha256: raise ValueError('Model source fingerprint mismatch')",
)
nb.cells[22].source = cell22

set_markdown(nb, 33, "## 12. Resumable encoder-decoder generation and prediction management")

set_code(
    nb,
    34,
    r"""
    WORD_PATTERN = re.compile(r"\S+")
    def words(text: str) -> list[str]: return WORD_PATTERN.findall(text.strip())
    def exact_word_prefix(text: str, count: int = 128) -> str:
        units = words(text)
        if len(units) < count: raise ValueError(f"Only {len(units)} words, need {count}")
        return " ".join(units[:count])

    def _policy_value(policy: Mapping[str, Any], name: str, default=None):
        value = policy.get(name, default)
        return default if value is None else value

    def select_next_token(logits: torch.Tensor, generated: Sequence[int], policy: Mapping[str, Any], generator):
        scores = logits.float().clone()
        repetition_penalty = float(_policy_value(policy, "repetition_penalty", 1.0))
        if repetition_penalty <= 0: raise ValueError("repetition_penalty must be positive")
        for token in set(int(value) for value in generated):
            scores[token] = scores[token] / repetition_penalty if scores[token] > 0 else scores[token] * repetition_penalty
        ngram = int(_policy_value(policy, "no_repeat_ngram_size", 0))
        if ngram > 0 and len(generated) >= ngram - 1:
            prefix = tuple(generated[-(ngram - 1):]) if ngram > 1 else tuple()
            banned = set()
            for index in range(len(generated) - ngram + 1):
                gram = tuple(generated[index:index + ngram])
                if gram[:-1] == prefix: banned.add(gram[-1])
            if banned: scores[list(banned)] = -torch.inf
        strategy = str(policy.get("strategy"))
        if strategy == "greedy": return int(scores.argmax())
        if strategy not in {"top_k", "top_p", "top_k_top_p"}:
            raise ValueError(f"Unsupported declared decoding strategy: {strategy}")
        temperature = float(_policy_value(policy, "temperature", 1.0))
        if temperature <= 0: raise ValueError("temperature must be positive")
        scores = scores / temperature
        top_k = int(_policy_value(policy, "top_k", 0))
        if top_k > 0:
            threshold = torch.topk(scores, min(top_k, scores.numel())).values[-1]
            scores[scores < threshold] = -torch.inf
        top_p = float(_policy_value(policy, "top_p", 1.0))
        if not 0 < top_p <= 1: raise ValueError("top_p must be in (0, 1]")
        if top_p < 1:
            sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            probabilities = torch.softmax(sorted_scores, dim=-1)
            cumulative = probabilities.cumsum(dim=-1)
            remove = cumulative > top_p; remove[1:] = remove[:-1].clone(); remove[0] = False
            scores[sorted_indices[remove]] = -torch.inf
        probabilities = torch.softmax(scores, dim=-1)
        if not torch.isfinite(probabilities).all() or float(probabilities.sum()) <= 0:
            raise FloatingPointError("Invalid sampling distribution")
        return int(torch.multinomial(probabilities, 1, generator=generator))

    def generation_quality_report(text: str) -> dict[str, Any]:
        units = words(text); lowered = [unit.lower() for unit in units]
        counts = Counter(lowered)
        return {
            "word_count": len(units),
            "unique_word_ratio": len(counts) / max(1, len(units)),
            "max_word_fraction": max(counts.values(), default=0) / max(1, len(units)),
        }

    @torch.inference_mode()
    def seq2seq_continuation(model, tokenizer, prompt, device, *, policy, target_words=128, max_new_tokens=None):
        if max_new_tokens is None: raise ValueError("A declared max_new_tokens is required")
        max_encoder_length = int(MODEL_CONFIG.get("max_encoder_length", MODEL_CONFIG.get("context_length", 0)))
        if max_encoder_length <= 0: raise ValueError("MODEL_CONFIG must declare max_encoder_length")
        max_decoder_length = int(MODEL_CONFIG.get("max_decoder_length", 0))
        if max_decoder_length <= 1: raise ValueError("MODEL_CONFIG must declare max_decoder_length")
        if int(max_new_tokens) + 1 > max_decoder_length:
            raise ValueError("max_new_tokens plus BOS exceeds the configured decoder context")
        prompt_ids = tokenizer_encode(tokenizer, prompt)
        if not prompt_ids: raise ValueError("Prompt encoded to no tokens")
        encoder_ids = prompt_ids[-max_encoder_length:]
        bos_id = getattr(tokenizer, "bos_token_id", None); eos_id = getattr(tokenizer, "eos_token_id", None)
        if bos_id is None: raise ValueError("Encoder-decoder inference requires bos_token_id")
        encoder = torch.tensor([encoder_ids], dtype=torch.long, device=device)
        encoder_mask = torch.ones_like(encoder, dtype=torch.bool)
        generated = []
        seed = int(policy.get("seed"))
        generator = torch.Generator(device=device).manual_seed(seed)
        model = model.to(device); model.eval()
        for _ in range(int(max_new_tokens)):
            decoder = torch.tensor([[int(bos_id), *generated]], dtype=torch.long, device=device)
            logits = seq2seq_logits(model, encoder, decoder, encoder_mask)
            next_id = select_next_token(logits[0, -1], generated, policy, generator)
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
    """,
)

set_code(
    nb,
    35,
    r"""
    def load_selected_model_for_inference(checkpoint_path, tokenizer_meta, reconstruct_model, device):
        state = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
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
            raise RuntimeError("Student must declare ordered decoding policies and max_new_tokens before inference")
        for index, policy in enumerate(policies):
            if not isinstance(policy, Mapping) or not policy.get("name") or policy.get("seed") is None:
                raise ValueError(f"Decoding policy {index} requires a name and deterministic seed")
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", checkpoint_id)
        partial_path = DIRS["predictions"] / f"official_predictions.{safe}.partial.json"
        meta_path = partial_path.with_suffix(".meta.json")
        attempts_path = DIRS["predictions"] / f"generation_attempts.{safe}.jsonl"
        expected_meta = {
            "checkpoint_id": checkpoint_id,
            "tokenizer_fingerprint": tokenizer_meta["fingerprint"],
            "model_source_sha256": MODEL_SOURCE_SHA256,
            "test_fingerprint": test_records_fingerprint(test_records),
            "decoding": dict(CFG.decoding),
        }
        predictions = {}
        if partial_path.exists():
            if not meta_path.exists() or json.loads(meta_path.read_text(encoding="utf-8")) != expected_meta:
                raise RuntimeError("Partial predictions do not authenticate current checkpoint/tokenizer/model/test/decoding")
            predictions = json.loads(partial_path.read_text(encoding="utf-8"))
            expected_ids = {row["id"] for row in test_records}
            if set(predictions) - expected_ids: raise RuntimeError("Partial prediction file contains foreign IDs")
            for key, value in predictions.items():
                if not isinstance(value, dict) or set(value) != {"text"} or len(words(value["text"])) != 128:
                    raise RuntimeError(f"Invalid cached prediction: {key}")
        else:
            atomic_write_json(meta_path, expected_meta)
        failures = []
        for row in tqdm(test_records, desc="Official test generation"):
            if row["id"] in predictions: continue
            completed = False; last_error = None
            for policy_index, policy in enumerate(policies):
                effective_policy = dict(policy)
                effective_policy["seed"] = int(policy["seed"]) + int(hashlib.sha256(row["id"].encode()).hexdigest()[:8], 16)
                started = time.perf_counter()
                try:
                    continuation, quality = seq2seq_continuation(
                        model, tokenizer, row["prompt"], device, policy=effective_policy,
                        target_words=128, max_new_tokens=max_new_tokens,
                    )
                    record = {
                        "id": row["id"], "policy_index": policy_index, "policy_name": policy["name"],
                        "status": "success", "quality": quality, "seconds": time.perf_counter() - started,
                        "created_utc": utc_now(),
                    }
                    with attempts_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    predictions[row["id"]] = {"text": continuation}
                    atomic_write_json(partial_path, predictions)
                    completed = True; break
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    with attempts_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({
                            "id": row["id"], "policy_index": policy_index, "policy_name": policy["name"],
                            "status": "failed", "error": last_error,
                            "seconds": time.perf_counter() - started, "created_utc": utc_now(),
                        }, ensure_ascii=False) + "\n")
            if not completed: failures.append({"id": row["id"], "error": last_error})
        atomic_write_json(DIRS["predictions"] / "generation_failures.json", failures)
        if failures: raise RuntimeError(f"Generation remains incomplete for {len(failures)} IDs; submission packaging is blocked")
        provenance = {
            **expected_meta, "prediction_count": len(predictions),
            "partial_prediction_sha256": sha256_file(partial_path), "attempt_log_sha256": sha256_file(attempts_path),
            "created_utc": utc_now(),
        }
        atomic_write_json(DIRS["predictions"] / "inference_provenance.json", provenance)
        return predictions, failures

    def run_phase1_inference_pipeline(device="cuda:0", run_local_evaluation=False):
        if CFG.execution_mode != "inference_only":
            print("Inference may also be run after training; execution mode is", CFG.execution_mode)
        if MODEL is None or tokenizer is None or RECONSTRUCT_MODEL is None:
            raise RuntimeError("Restore the tokenizer and student model source/config before inference")
        selected = DIRS["checkpoints"] / "selected_model.pt"
        if not selected.exists(): raise FileNotFoundError("selected_model.pt is required")
        selected_model, state = load_selected_model_for_inference(selected, TOKENIZER_META, RECONSTRUCT_MODEL, device)
        checkpoint_id = f"selected-{sha256_file(selected)[:12]}"
        local_evaluation = None
        if run_local_evaluation:
            examples_path = DIRS["data"] / "local_continuation_examples.json"
            if OFFICIAL_EVALUATOR is None or not examples_path.exists():
                raise RuntimeError("Optional local evaluation requires both evaluator and saved development examples")
            local_evaluation = {"status": "STUDENT RUNS WITH THE SAME DECLARED DECODING POLICY"}
        schema_path = DIRS["manifests"] / "resolved_json_schemas.json"
        if RESOLVED_SCHEMA is not None:
            schemas = RESOLVED_SCHEMA
        elif schema_path.exists():
            schemas = json.loads(schema_path.read_text(encoding="utf-8"))
        else:
            raise RuntimeError("Official test schema is unavailable")
        test_records = load_test_records(require_unique_file(DISCOVERED, CFG.test_file), schemas["test"])
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
        print("Inference is opt-in and does not require the training corpus or local evaluator.")
    """,
)

# Submission packaging stays strict; it now also records non-blocking quality diagnostics.
cell37 = nb.cells[37].source
cell37 = cell37.replace(
    "report['prediction_sha256']=sha256_file(prediction_path)",
    "report['prediction_sha256']=sha256_file(prediction_path)\n    report['quality_diagnostics']={key:generation_quality_report(value['text']) for key,value in predictions.items()}",
)
nb.cells[37].source = cell37

cell39 = nb.cells[39].source.replace("phase1_submission01_recovery.zip", "phase1_submission02_recovery.zip")
cell39 = cell39.replace("phase1_submission01_recovery.previous.zip", "phase1_submission02_recovery.previous.zip")
cell39 = cell39.replace(
    "excluded={out,previous,manifest_path,outdir/'recovery_validation.json',root/'data'/'document_store'}",
    "excluded={out,previous,manifest_path,outdir/'recovery_validation.json',root/'data'/'bounded_document_store'}",
)
cell39 = cell39.replace(
    "manifest=build_integrity_manifest(root,excluded)",
    """notebook_backup=Path(CFG.completed_notebook_path).expanduser() if CFG.completed_notebook_path else None
    if notebook_backup is not None:
        if not notebook_backup.is_file(): raise FileNotFoundError(notebook_backup)
        shutil.copy2(notebook_backup,root/notebook_backup.name)
    manifest=build_integrity_manifest(root,excluded)""",
)
nb.cells[39].source = cell39

set_code(
    nb,
    41,
    r"""
    def final_verification(cfg, model=None, tokenizer_meta=None, official_ids=None):
        checks = {
            "student_owned_architecture_attested": bool(cfg.student_owned_architecture_attestation),
            "student_configuration_confirmed": bool(cfg.student_configuration_confirmed),
            "trained_from_scratch_attested": bool(cfg.trained_from_scratch_attestation),
        }
        model_validation = DIRS["manifests"] / "model_validation.json"
        report = json.loads(model_validation.read_text(encoding="utf-8")) if model_validation.exists() else {}
        checks["parameter_count_50m_to_300m"] = 50_000_000 <= int(report.get("total", -1)) <= 300_000_000
        checks["model_source_frozen"] = bool(
            MODEL_SOURCE_PATH and Path(MODEL_SOURCE_PATH).exists()
            and report.get("model_source_sha256") == sha256_file(MODEL_SOURCE_PATH)
        )
        smoke = DIRS["manifests"] / "gpu_training_smoke_test.json"
        checks["gpu_training_smoke_passed"] = smoke.exists() and json.loads(smoke.read_text()).get("status") == "PASS"
        tokenizer_path = DIRS["tokenizer"] / "tokenizer_metadata.json"
        checks["tokenizer_registered"] = tokenizer_path.exists()
        checks["tokenizer_integrity"] = bool(
            tokenizer_path.exists()
            and json.loads(tokenizer_path.read_text()).get("fingerprint") == directory_fingerprint(DIRS["tokenizer"])
        )
        checks["split_manifest_present"] = (DIRS["manifests"] / "document_split.json").exists()
        checks["validation_protocol_present"] = (DIRS["manifests"] / "validation_sequence_manifest.json").exists()
        checks["training_history_present"] = (DIRS["histories"] / "training.jsonl").exists()
        selected = DIRS["checkpoints"] / "selected_model.pt"
        final = DIRS["checkpoints"] / "model_final.pt"
        checks["selected_model_present"] = selected.exists()
        checks["final_model_present"] = final.exists()
        submission = DIRS["submission"] / "submission.zip"
        checks["submission_zip_valid"] = False
        checks["artifact_provenance_fresh"] = False
        if submission.exists() and official_ids is not None:
            try:
                with zipfile.ZipFile(submission) as archive:
                    if archive.namelist() == ["prediction.json"]:
                        predictions = json.loads(archive.read("prediction.json"))
                        checks["submission_zip_valid"] = validate_predictions(predictions, official_ids, 128)["passed"]
                validation = json.loads((DIRS["submission"] / "submission_validation.json").read_text())
                provenance = validation.get("inference_provenance", {})
                checks["artifact_provenance_fresh"] = bool(
                    selected.exists()
                    and provenance.get("checkpoint_id") == f"selected-{sha256_file(selected)[:12]}"
                    and provenance.get("model_source_sha256") == MODEL_SOURCE_SHA256
                    and validation.get("prediction_sha256") == sha256_file(DIRS["submission"] / "prediction.json")
                )
            except Exception:
                pass
        recovery = DIRS["recovery"] / "phase1_submission02_recovery.zip"
        checks["recovery_zip_valid"] = False
        checks["recovery_is_fresh"] = False
        if recovery.exists():
            try:
                verify_recovery_archive(recovery); checks["recovery_zip_valid"] = True
                inputs = [path for path in (selected, submission) if path.exists()]
                checks["recovery_is_fresh"] = bool(inputs and recovery.stat().st_mtime >= max(path.stat().st_mtime for path in inputs))
            except Exception:
                pass
        artifact_status = "PASS" if all(checks.values()) else "FAIL"
        codabench_record = DIRS["manifests"] / "codabench_acceptance.json"
        accepted = False
        if codabench_record.exists():
            record = json.loads(codabench_record.read_text(encoding="utf-8"))
            accepted = bool(record.get("accepted") and record.get("submission_id"))
        outcome = {
            "artifact_status": artifact_status,
            "codabench_accepted": accepted,
            "completion_status": "COMPLETE" if artifact_status == "PASS" and accepted else "INCOMPLETE",
            "checks": checks,
            "optional_local_evaluation_present": any(DIRS["evaluations"].glob("*.json")),
            "created_utc": utc_now(),
        }
        atomic_write_json(DIRS["manifests"] / "final_verification.json", outcome)
        print(json.dumps(outcome, indent=2))
        return outcome

    # After Codabench accepts the upload, record truthful evidence, for example:
    # atomic_write_json(DIRS['manifests']/'codabench_acceptance.json', {
    #     'accepted': True, 'submission_id': 'REAL_ID', 'submitted_utc': utc_now(),
    #     'result': {'bertscore_f1': 'REAL_VALUE', 'rouge_l_f1': 'REAL_VALUE', 'bleu_4': 'REAL_VALUE'}
    # })
    # Rebuild the recovery archive, then run:
    # FINAL_REPORT = final_verification(CFG, MODEL, TOKENIZER_META, [row['id'] for row in TEST_RECORDS])
    """,
)

set_markdown(
    nb,
    42,
    r"""
    ## Kaggle execution checklist for Submission 02

    1. Keep Submission 01 backed up. Submission 02 uses only `.../submission_02` paths.
    2. Attach the official Phase 1 Kaggle dataset under `/kaggle/input`. The schema inspection must show
       each training file separately; training arrays without ID fields receive deterministic source+row IDs.
    3. Set `CFG.corpus_document_cap_per_source`, review the bounded-cache estimate, and enable
       `RUN_CORPUS_PREPROCESSING` once. The full official corpus is never copied into `/kaggle/working`.
    4. Point `CFG.tokenizer_import_path` to the frozen Submission 01/course tokenizer when performing a
       controlled architecture comparison. Never change its token-to-ID mapping after training begins.
    5. Write the encoder-decoder module yourself from the course material, save it as a Python file, set
       `CFG.student_model_source_path`, and fill the exact numerical model configuration. The validation
       report must show 50M-300M parameters, causal decoder behavior, encoder conditioning, finite gradients,
       and exact reconstruction.
    6. Fill every `None` in `TRAINING_CONFIG`, write the hypothesis/rationale, and record the matched
       Submission 01 control. These are student-owned decisions, not auto-tuned values.
    7. Run the representative GPU smoke test. Review actual precision, peak VRAM, loss, and gradient norm.
       If two GPUs are present, this notebook still uses only the explicitly selected GPU.
    8. Choose exactly one mode: `fresh_train` refuses an existing checkpoint; `resume_train` requires one
       and prints the restored optimizer step. Enable full training only after both explicit gates are true.
    9. Download milestone recovery ZIPs outside Kaggle. A file in `/kaggle/working` is not durable by itself.
    10. For inference-only, restore tokenizer, model source/config, and `selected_model.pt`. Do not load the
        training corpus. Declare ordered decoding/retry policies before generation; every attempt is logged.
    11. Generate all official IDs, validate exactly 128 continuation words, create the root-level
        `prediction.json` inside `submission.zip`, and rebuild the final recovery archive.
    12. Upload only `submission.zip` to Codabench. Record the real acceptance ID/results afterward; do not
        fabricate metrics. Submission 02 is complete only when artifact verification passes and Codabench
        acceptance is truthfully recorded.
    """,
)

# Ensure every code cell is clean and unexecuted in the distributable template.
for cell in nb.cells:
    if cell.cell_type == "code":
        cell.execution_count = None
        cell.outputs = []

nbformat.validate(nb)
for destination in (OUTPUT, PROJECT_COPY):
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, destination)
print(f"Wrote {OUTPUT} and project copy with {len(nb.cells)} cells")
