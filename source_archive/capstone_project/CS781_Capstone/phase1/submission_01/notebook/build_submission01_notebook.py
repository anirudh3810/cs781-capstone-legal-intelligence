from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "CS781_Phase1_Submission01_Infrastructure.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str, *, tags: list[str] | None = None):
    cell = nbf.v4.new_code_cell(textwrap.dedent(text).strip())
    if tags:
        cell.metadata["tags"] = tags
    return cell


nb = nbf.v4.new_notebook()
nb.metadata.update(
    {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "cs781": {
            "phase": 1,
            "submission": 1,
            "purpose": "supporting infrastructure only",
            "architecture_generated_by_llm": False,
        },
    }
)

nb.cells = [
    md(
        """
        # CS781 Capstone — Phase 1, Submission 1
        ## Reproducible legal-language-model infrastructure

        This Kaggle notebook implements only the permitted supporting infrastructure: official-data discovery and validation, deterministic document splits, tokenizer compatibility checks, model-interface checks, checkpoint/recovery management, history logging, local evaluation plumbing, generation bookkeeping, and Codabench packaging.

        **Academic-integrity boundary:** the language-model architecture and experiment-specific training algorithm are intentionally absent. Cells marked **STUDENT IMPLEMENTATION REQUIRED** must be written by the student from course material. No pretrained language-model weights, external legal corpus, fabricated result, automatic hyperparameter search, or scientific conclusion is included.

        Submission 1 should document a clear algorithmic hypothesis, not merely a leaderboard attempt. The notebook records the hypothesis and ablation controls, but it does not invent or implement them for you.

        **Recommended framing after the professor's clarification:** keep the conventional decoder-only configuration as the matched control, then make the submitted model a student-designed intervention in one Transformer subsystem. Hold data, tokenizer, parameter budget, training-token budget, and evaluation procedure fixed so that any accuracy/efficiency change is attributable to that intervention. Course-grounded places to investigate include attention structure, positional treatment, normalization/residual placement, feed-forward computation, or the training objective/schedule. Select and justify the intervention yourself; this notebook will only record and validate the comparison.
        """
    ),
    md(
        """
        ## Execution map

        Fresh session: run sections 1–8, supply your tokenizer/model/training implementation in the marked cells, run the smoke tests, then explicitly run full training. After training, run evaluation, generation, packaging, final verification, and recovery export.

        Crash resume: run sections 1–3, recovery discovery, tokenizer/model reconstruction, checkpoint restore, then your training cell with the restored state. Never rerun split creation with changed settings.
        """
    ),
    md("## 1. Environment setup and reproducibility"),
    code(
        """
        from __future__ import annotations

        import csv
        import dataclasses
        import hashlib
        import importlib.util
        import io
        import json
        import math
        import os
        import random
        import re
        import shutil
        import sys
        import tempfile
        import time
        import zipfile
        from collections import Counter
        from dataclasses import asdict, dataclass, field
        from datetime import datetime, timezone
        from pathlib import Path
        from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

        import numpy as np
        import torch
        from torch.utils.data import Sampler

        try:
            from tqdm.auto import tqdm
        except ImportError as exc:
            raise RuntimeError("tqdm is required; install it in the Kaggle environment") from exc

        def utc_now() -> str:
            return datetime.now(timezone.utc).isoformat()

        def atomic_write_bytes(path: Path, payload: bytes) -> None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
                temporary = Path(handle.name)
            os.replace(temporary, path)

        def atomic_write_json(path: Path, value: Any) -> None:
            atomic_write_bytes(Path(path), (json.dumps(value, indent=2, ensure_ascii=False) + "\\n").encode("utf-8"))

        def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
            digest = hashlib.sha256()
            with Path(path).open("rb") as handle:
                for chunk in iter(lambda: handle.read(chunk_size), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        print({"python": sys.version.split()[0], "torch": torch.__version__, "utc": utc_now()})
        """
    ),
    code(
        """
        # User-owned experiment configuration. Values below only record the specification supplied by the student.
        @dataclass
        class RunConfig:
            experiment_id: str = "phase1_submission01"
            seed: int = 781
            workspace: str = field(default_factory=lambda: str(
                Path("/kaggle/working/CS781_Capstone/phase1/submission_01")
                if Path("/kaggle/working").exists()
                else Path.cwd() / "CS781_Capstone/phase1/submission_01"
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
            # Fill these only after inspecting the actual JSON schema.
            document_id_field: str | None = None
            document_text_field: str | None = None
            test_id_field: str | None = None
            test_prompt_field: str | None = None
            validation_fraction: float | None = None
            source_mixing_ratios: dict[str, float] | None = None
            # Recorded proposal; architecture remains student-authored.
            architecture_record: dict[str, Any] = field(default_factory=lambda: {
                "family": "student-implemented causal language model",
                "intended_parameter_range": [50_000_000, 300_000_000],
                "blocks": 8,
                "hidden_dimension": "student-confirmed value in 512-560 range",
                "heads": 8,
                "ffn_dimension": "approximately 4x hidden dimension",
                "normalization": "Pre-LayerNorm",
                "activation": "GELU",
                "attention": "student implementation",
                "position_encoding": "student implementation",
                "context_length": 1024,
                "objective": "causal next-token prediction",
            })
            innovation_hypothesis: str = "STUDENT MUST WRITE BEFORE TRAINING"
            ablation_control: str = "STUDENT MUST WRITE BEFORE TRAINING"
            full_training_authorized: bool = False
            trained_from_scratch_attestation: bool = False
            decoding: dict[str, Any] = field(default_factory=lambda: {
                "strategy": "greedy",
                "temperature": None,
                "top_k": None,
                "top_p": None,
                "target_words": 128,
            })

        CFG = RunConfig()
        ROOT = Path(CFG.workspace)
        DIRS = {name: ROOT / name for name in (
            "model", "tokenizer", "data", "checkpoints", "histories", "evaluations",
            "predictions", "plots", "recovery", "manifests", "submission"
        )}
        for directory in DIRS.values():
            directory.mkdir(parents=True, exist_ok=True)

        random.seed(CFG.seed)
        np.random.seed(CFG.seed)
        torch.manual_seed(CFG.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(CFG.seed)
        atomic_write_json(DIRS["manifests"] / "run_config.json", asdict(CFG))
        ROOT
        """
    ),
    md("## 2. Kaggle GPU detection and optional authentication"),
    code(
        """
        def gpu_inventory() -> list[dict[str, Any]]:
            rows = []
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                rows.append({
                    "index": index,
                    "device": f"cuda:{index}",
                    "name": props.name,
                    "memory_gib": round(props.total_memory / 2**30, 2),
                    "capability": f"{props.major}.{props.minor}",
                })
            return rows

        GPUS = gpu_inventory()
        print(json.dumps(GPUS, indent=2))
        if not GPUS:
            print("WARNING: no CUDA GPU detected. Do not start full training.")
        elif len(GPUS) == 1:
            print("One independent experiment can use cuda:0.")
        else:
            print("Independent jobs may be assigned cuda:0 and cuda:1; this notebook does not use DataParallel.")
        """
    ),
    code(
        """
        # Optional only when an attached artifact actually requires Hugging Face authentication.
        # Do not paste a token into the notebook. Use a Kaggle secret named HF_TOKEN.
        def optional_hf_login(secret_name: str = "HF_TOKEN") -> bool:
            try:
                from kaggle_secrets import UserSecretsClient
                from huggingface_hub import login
                token = UserSecretsClient().get_secret(secret_name)
                if not token:
                    return False
                login(token=token, add_to_git_credential=False)
                return True
            except Exception as exc:
                print(f"HF login skipped: {type(exc).__name__}: {exc}")
                return False

        # optional_hf_login()  # Uncomment only if required.
        """
    ),
    md("## 3. Official file discovery and schema inspection"),
    code(
        """
        REQUIRED_TRAIN = set(CFG.train_files)

        def discover_named_files(root: Path, names: Iterable[str]) -> dict[str, list[Path]]:
            root = Path(root)
            wanted = set(names)
            found = {name: [] for name in wanted}
            if root.exists():
                for path in root.rglob("*"):
                    if path.is_file() and path.name in wanted:
                        found[path.name].append(path)
            return found

        DISCOVERED = discover_named_files(Path(CFG.input_root), [*CFG.train_files, CFG.test_file])
        for name, paths in sorted(DISCOVERED.items()):
            print(name, "->", [str(path) for path in paths])

        def require_unique_file(discovered: Mapping[str, Sequence[Path]], name: str) -> Path:
            matches = list(discovered.get(name, []))
            if len(matches) != 1:
                raise RuntimeError(f"Expected exactly one {name!r}; found {len(matches)}. Select explicitly before continuing: {matches}")
            return matches[0]
        """
    ),
    code(
        """
        def load_json_container(path: Path) -> Any:
            with Path(path).open("r", encoding="utf-8-sig") as handle:
                return json.load(handle)

        def records_from_container(container: Any) -> list[dict[str, Any]]:
            if isinstance(container, list):
                records = container
            elif isinstance(container, dict):
                list_fields = [(key, value) for key, value in container.items() if isinstance(value, list)]
                if len(list_fields) != 1:
                    raise ValueError(f"JSON object must contain exactly one unambiguous record list; candidates={[k for k, _ in list_fields]}")
                records = list_fields[0][1]
            else:
                raise TypeError(f"Unsupported JSON root type: {type(container).__name__}")
            if not all(isinstance(record, dict) for record in records):
                raise TypeError("Every dataset record must be a JSON object")
            return records

        def inspect_schema(path: Path, sample_size: int = 3) -> dict[str, Any]:
            container = load_json_container(path)
            records = records_from_container(container)
            keys = Counter(key for record in records for key in record)
            samples = []
            for record in records[:sample_size]:
                samples.append({key: {"type": type(value).__name__, "preview": str(value)[:160]} for key, value in record.items()})
            return {"path": str(path), "root_type": type(container).__name__, "records": len(records), "field_presence": dict(keys), "samples": samples}

        # Run after the official files are attached. Review output, then explicitly fill CFG field names.
        # SCHEMAS = {name: inspect_schema(require_unique_file(DISCOVERED, name)) for name in [*CFG.train_files, CFG.test_file]}
        # print(json.dumps(SCHEMAS, indent=2, ensure_ascii=False))
        """
    ),
    md("## 4. Validated loading, provenance, deduplication, and document-level split"),
    code(
        """
        @dataclass(frozen=True)
        class Document:
            uid: str
            text: str
            source: str
            raw_id: str
            content_hash: str

        def normalize_text(value: Any) -> str:
            if not isinstance(value, str):
                raise TypeError(f"Text must be a string, got {type(value).__name__}")
            text = value.replace("\\r\\n", "\\n").strip()
            if not text:
                raise ValueError("Text is empty")
            return text

        def load_documents(path: Path, source: str, id_field: str, text_field: str) -> tuple[list[Document], list[dict[str, Any]]]:
            records = records_from_container(load_json_container(path))
            accepted, rejected = [], []
            for index, record in enumerate(records):
                try:
                    if id_field not in record or text_field not in record:
                        raise KeyError(f"required fields missing: {id_field!r}, {text_field!r}")
                    raw_id = str(record[id_field]).strip()
                    if not raw_id:
                        raise ValueError("empty document ID")
                    text = normalize_text(record[text_field])
                    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    accepted.append(Document(f"{source}:{raw_id}", text, source, raw_id, content_hash))
                except Exception as exc:
                    rejected.append({"index": index, "error": f"{type(exc).__name__}: {exc}"})
            return accepted, rejected

        def corpus_audit(documents: Sequence[Document]) -> dict[str, Any]:
            uid_counts = Counter(doc.uid for doc in documents)
            hash_counts = Counter(doc.content_hash for doc in documents)
            return {
                "documents": len(documents),
                "sources": dict(Counter(doc.source for doc in documents)),
                "whitespace_word_count": sum(len(doc.text.split()) for doc in documents),
                "duplicate_uids": sorted(uid for uid, count in uid_counts.items() if count > 1),
                "duplicate_content_hashes": sorted(value for value, count in hash_counts.items() if count > 1),
            }

        def dataset_fingerprint(documents: Sequence[Document]) -> str:
            digest = hashlib.sha256()
            for doc in sorted(documents, key=lambda item: item.uid):
                digest.update(f"{doc.uid}\\t{doc.content_hash}\\n".encode("utf-8"))
            return digest.hexdigest()
        """
    ),
    code(
        """
        def deterministic_document_split(
            documents: Sequence[Document], validation_fraction: float, seed: int
        ) -> tuple[list[Document], list[Document], dict[str, Any]]:
            if not 0 < validation_fraction < 1:
                raise ValueError("validation_fraction must be explicitly set between 0 and 1")
            audit = corpus_audit(documents)
            if audit["duplicate_uids"] or audit["duplicate_content_hashes"]:
                raise ValueError("Resolve duplicate IDs/content before splitting; audit contains details")
            ordered = list(documents)
            random.Random(seed).shuffle(ordered)
            val_count = max(1, round(len(ordered) * validation_fraction))
            validation, training = ordered[:val_count], ordered[val_count:]
            train_ids, val_ids = {doc.uid for doc in training}, {doc.uid for doc in validation}
            if train_ids & val_ids:
                raise AssertionError("Document leakage detected")
            metadata = {
                "created_utc": utc_now(),
                "seed": seed,
                "validation_fraction": validation_fraction,
                "dataset_fingerprint": dataset_fingerprint(documents),
                "training_ids": sorted(train_ids),
                "validation_ids": sorted(val_ids),
            }
            return training, validation, metadata

        def save_or_verify_split(path: Path, metadata: Mapping[str, Any]) -> None:
            path = Path(path)
            if path.exists():
                existing = json.loads(path.read_text(encoding="utf-8"))
                comparable = ("seed", "validation_fraction", "dataset_fingerprint", "training_ids", "validation_ids")
                if any(existing.get(key) != metadata.get(key) for key in comparable):
                    raise RuntimeError("Existing split manifest is incompatible; refusing to overwrite")
                print("Existing split manifest verified:", path)
            else:
                atomic_write_json(path, dict(metadata))

        def make_continuation_examples(documents: Sequence[Document], prompt_words: int = 512, continuation_words: int = 128) -> list[dict[str, str]]:
            examples = []
            needed = prompt_words + continuation_words
            for doc in documents:
                words = doc.text.split()
                if len(words) >= needed:
                    examples.append({"id": doc.uid, "prompt": " ".join(words[:prompt_words]), "reference": " ".join(words[prompt_words:needed])})
            return examples
        """
    ),
    code(
        """
        # OFFICIAL DATA EXECUTION CELL — review schemas first, then set CFG fields above.
        # This cell intentionally refuses guessed schemas and missing source ratios.
        def load_official_training_corpus(cfg: RunConfig, discovered: Mapping[str, Sequence[Path]]):
            if not cfg.document_id_field or not cfg.document_text_field:
                raise RuntimeError("Inspect schemas and explicitly set document_id_field/document_text_field")
            if cfg.validation_fraction is None:
                raise RuntimeError("Student must explicitly set validation_fraction")
            if cfg.source_mixing_ratios is None:
                raise RuntimeError("Student must explicitly record source_mixing_ratios; this notebook does not choose them")
            all_documents, rejection_log, provenance = [], {}, {}
            for filename in cfg.train_files:
                path = require_unique_file(discovered, filename)
                source = Path(filename).stem.removeprefix("train_")
                docs, rejected = load_documents(path, source, cfg.document_id_field, cfg.document_text_field)
                all_documents.extend(docs)
                rejection_log[source] = rejected
                provenance[source] = {"path": str(path), "sha256": sha256_file(path), "accepted": len(docs), "rejected": len(rejected)}
            audit = corpus_audit(all_documents)
            atomic_write_json(DIRS["manifests"] / "dataset_provenance.json", provenance)
            atomic_write_json(DIRS["manifests"] / "dataset_audit.json", audit)
            atomic_write_json(DIRS["manifests"] / "rejected_records.json", rejection_log)
            train_docs, val_docs, split = deterministic_document_split(all_documents, cfg.validation_fraction, cfg.seed)
            save_or_verify_split(DIRS["manifests"] / "document_split.json", split)
            examples = make_continuation_examples(val_docs)
            atomic_write_json(DIRS["data"] / "local_continuation_examples.json", examples)
            return train_docs, val_docs, examples, audit

        # TRAIN_DOCS, VAL_DOCS, LOCAL_DEV, DATA_AUDIT = load_official_training_corpus(CFG, DISCOVERED)
        """,
        tags=["manual-data"],
    ),
    md("## 5. Tokenizer persistence and compatibility"),
    code(
        """
        def directory_fingerprint(path: Path) -> str:
            path = Path(path)
            digest = hashlib.sha256()
            files = sorted(
                item for item in path.rglob("*")
                if item.is_file() and item.name != "tokenizer_metadata.json"
            )
            if not files:
                raise ValueError(f"No tokenizer files found under {path}")
            for item in files:
                digest.update(item.relative_to(path).as_posix().encode("utf-8"))
                digest.update(bytes.fromhex(sha256_file(item)))
            return digest.hexdigest()

        def tokenizer_metadata(tokenizer: Any, tokenizer_dir: Path) -> dict[str, Any]:
            special = {}
            for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id", "mask_token_id"):
                special[name] = getattr(tokenizer, name, None)
            try:
                vocab_size = len(tokenizer)
            except Exception as exc:
                raise TypeError("Tokenizer must implement __len__") from exc
            probe = "The Court considered section 14 and dismissed the appeal."
            token_ids = tokenizer.encode(probe, add_special_tokens=False)
            decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
            if not token_ids or not decoded.strip():
                raise ValueError("Tokenizer round-trip produced empty output")
            return {
                "created_utc": utc_now(),
                "vocab_size": vocab_size,
                "special_token_ids": special,
                "fingerprint": directory_fingerprint(tokenizer_dir),
                "round_trip_probe": {"input": probe, "ids": token_ids, "decoded": decoded},
            }

        def tokenizer_token_audit(documents: Iterable[Document], tokenizer: Any) -> dict[str, Any]:
            by_source: Counter[str] = Counter()
            total = 0
            document_count = 0
            for document in documents:
                count = len(tokenizer.encode(document.text, add_special_tokens=False))
                by_source[document.source] += count
                total += count
                document_count += 1
            return {"documents": document_count, "tokens": total, "tokens_by_source": dict(by_source)}

        def register_tokenizer(tokenizer: Any, destination: Path) -> dict[str, Any]:
            destination = Path(destination)
            metadata_path = destination / "tokenizer_metadata.json"
            if metadata_path.exists():
                existing = json.loads(metadata_path.read_text(encoding="utf-8"))
                current = directory_fingerprint(destination)
                if existing.get("fingerprint") != current:
                    raise RuntimeError("Tokenizer directory changed after registration; refusing silent replacement")
                return existing
            if any(destination.iterdir()):
                raise RuntimeError("Tokenizer destination is nonempty but unregistered; inspect it explicitly")
            tokenizer.save_pretrained(destination)
            metadata = tokenizer_metadata(tokenizer, destination)
            atomic_write_json(metadata_path, metadata)
            return metadata

        # STUDENT TOKENIZER INTEGRATION REQUIRED:
        # tokenizer = ...  # Load or independently train only on the official Phase 1 training corpus.
        # TOKENIZER_META = register_tokenizer(tokenizer, DIRS["tokenizer"])
        # TOKEN_AUDIT = tokenizer_token_audit(TRAIN_DOCS + VAL_DOCS, tokenizer)
        # atomic_write_json(DIRS["manifests"] / "token_count_audit.json", TOKEN_AUDIT)
        """,
        tags=["student-tokenizer"],
    ),
    md("## 6. Model integration contract and validation (no architecture implementation)"),
    code(
        """
        # STUDENT IMPLEMENTATION REQUIRED.
        # Put your independently authored architecture in DIRS["model"] and import it here.
        # Required contract:
        #   model(input_ids: LongTensor[B,T]) -> logits FloatTensor[B,T,V]
        #   model.get_config() -> JSON-serializable mapping (or provide MODEL_CONFIG separately)
        #   reconstruction_function(MODEL_CONFIG) -> a new model instance with identical structure
        # Optional generation contract:
        #   model.generate(input_ids=..., max_new_tokens=..., do_sample=False, ...)

        MODEL = None
        MODEL_CONFIG = None
        RECONSTRUCT_MODEL: Callable[[Mapping[str, Any]], torch.nn.Module] | None = None

        def count_parameters(model: torch.nn.Module) -> dict[str, int]:
            return {
                "total": sum(parameter.numel() for parameter in model.parameters()),
                "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            }

        def validate_model_contract(model: torch.nn.Module, vocab_size: int, device: str, sequence_length: int = 16) -> dict[str, Any]:
            model = model.to(device)
            model.train()
            model.zero_grad(set_to_none=True)
            input_ids = torch.randint(0, vocab_size, (2, sequence_length), dtype=torch.long, device=device)
            output = model(input_ids)
            logits = output.logits if hasattr(output, "logits") else output[0] if isinstance(output, (tuple, list)) else output
            expected = (2, sequence_length, vocab_size)
            if tuple(logits.shape) != expected:
                raise ValueError(f"Expected logits shape {expected}, got {tuple(logits.shape)}")
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite logits")
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), input_ids[:, 1:].reshape(-1))
            loss.backward()
            gradient_tensors = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not gradient_tensors:
                raise RuntimeError("No gradients were produced")
            if not all(torch.isfinite(gradient).all().item() for gradient in gradient_tensors):
                raise FloatingPointError("Non-finite gradients")
            counts = count_parameters(model)
            if not 50_000_000 <= counts["total"] <= 300_000_000:
                raise ValueError(f"Official parameter bound violated: {counts['total']:,}")
            model.zero_grad(set_to_none=True)
            return {"logits_shape": list(logits.shape), "loss": float(loss.detach().cpu()), **counts, "device": device}

        # MODEL_REPORT = validate_model_contract(MODEL, TOKENIZER_META["vocab_size"], "cuda:0")
        # atomic_write_json(DIRS["manifests"] / "model_validation.json", MODEL_REPORT)
        """,
        tags=["student-model"],
    ),
    md("## 7. Stateful sampling, histories, checkpoints, and progress"),
    code(
        """
        class StatefulRandomSampler(Sampler[int]):
            def __init__(self, data_source: Sequence[Any], seed: int, epoch: int = 0, cursor: int = 0):
                self.length = len(data_source)
                self.seed = int(seed)
                self.epoch = int(epoch)
                self.cursor = int(cursor)

            def _order(self) -> list[int]:
                generator = torch.Generator().manual_seed(self.seed + self.epoch)
                return torch.randperm(self.length, generator=generator).tolist()

            def __iter__(self):
                order = self._order()
                for position in range(self.cursor, self.length):
                    self.cursor = position + 1
                    yield order[position]

            def __len__(self):
                return max(0, self.length - self.cursor)

            def next_epoch(self):
                self.epoch += 1
                self.cursor = 0

            def state_dict(self):
                return {"length": self.length, "seed": self.seed, "epoch": self.epoch, "cursor": self.cursor}

            def load_state_dict(self, state: Mapping[str, Any]):
                if int(state["length"]) != self.length or int(state["seed"]) != self.seed:
                    raise ValueError("Sampler state is incompatible with current dataset/seed")
                self.epoch, self.cursor = int(state["epoch"]), int(state["cursor"])

        class JsonlHistory:
            def __init__(self, path: Path):
                self.path = Path(path)
                self.path.parent.mkdir(parents=True, exist_ok=True)

            def records(self) -> list[dict[str, Any]]:
                if not self.path.exists():
                    return []
                rows = []
                with self.path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            rows.append(json.loads(line))
                steps = [row["step"] for row in rows]
                if len(steps) != len(set(steps)):
                    raise ValueError("Duplicate steps found in history")
                return rows

            def append(self, record: Mapping[str, Any]):
                existing = self.records()
                if any(row["step"] == record["step"] for row in existing):
                    raise ValueError(f"Refusing duplicate history step {record['step']}")
                payload = (json.dumps(dict(record), ensure_ascii=False) + "\\n").encode("utf-8")
                with self.path.open("ab") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
        """
    ),
    code(
        """
        CHECKPOINT_VERSION = 1

        def capture_rng_state() -> dict[str, Any]:
            return {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }

        def restore_rng_state(state: Mapping[str, Any]) -> None:
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(state["torch_cpu"])
            if torch.cuda.is_available() and state.get("torch_cuda") is not None:
                torch.cuda.set_rng_state_all(state["torch_cuda"])

        def save_checkpoint_atomic(
            path: Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
            scheduler: Any, scaler: Any, sampler: StatefulRandomSampler,
            progress: Mapping[str, Any], model_config: Mapping[str, Any], tokenizer_fingerprint: str,
            dataset_fingerprint_value: str, training_config: Mapping[str, Any],
            history_path: Path, validation_history: Sequence[Mapping[str, Any]], experiment_id: str,
        ) -> dict[str, Any]:
            path = Path(path)
            history_sha = sha256_file(history_path) if Path(history_path).exists() else None
            state = {
                "checkpoint_version": CHECKPOINT_VERSION,
                "experiment_id": experiment_id,
                "created_utc": utc_now(),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state": scaler.state_dict() if scaler is not None else None,
                "sampler_state": sampler.state_dict(),
                "rng_state": capture_rng_state(),
                "progress": dict(progress),
                "model_config": dict(model_config),
                "tokenizer_fingerprint": tokenizer_fingerprint,
                "dataset_fingerprint": dataset_fingerprint_value,
                "training_config": dict(training_config),
                "history_sha256": history_sha,
                "validation_history": list(validation_history),
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as handle:
                temporary = Path(handle.name)
            try:
                torch.save(state, temporary)
                verified = torch.load(temporary, map_location="cpu", weights_only=False)
                if verified.get("checkpoint_version") != CHECKPOINT_VERSION:
                    raise RuntimeError("Checkpoint verification failed")
                if path.exists():
                    previous = path.with_name("previous_known_good.pt")
                    shutil.copy2(path, previous)
                os.replace(temporary, path)
                return {"path": str(path), "sha256": sha256_file(path), "step": state["progress"].get("step")}
            finally:
                temporary.unlink(missing_ok=True)
        """
    ),
    code(
        """
        def restore_checkpoint(
            path: Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
            sampler: StatefulRandomSampler, expected_tokenizer_fingerprint: str,
            expected_dataset_fingerprint: str, scheduler: Any = None, scaler: Any = None,
        ) -> dict[str, Any]:
            state = torch.load(Path(path), map_location="cpu", weights_only=False)
            if state.get("checkpoint_version") != CHECKPOINT_VERSION:
                raise ValueError("Unsupported checkpoint version")
            if state["tokenizer_fingerprint"] != expected_tokenizer_fingerprint:
                raise ValueError("Tokenizer/checkpoint fingerprint mismatch")
            if state["dataset_fingerprint"] != expected_dataset_fingerprint:
                raise ValueError("Dataset split/checkpoint fingerprint mismatch")
            model.load_state_dict(state["model_state"], strict=True)
            optimizer.load_state_dict(state["optimizer_state"])
            if scheduler is not None and state.get("scheduler_state") is not None:
                scheduler.load_state_dict(state["scheduler_state"])
            if scaler is not None and state.get("scaler_state") is not None:
                scaler.load_state_dict(state["scaler_state"])
            sampler.load_state_dict(state["sampler_state"])
            restore_rng_state(state["rng_state"])
            return state

        @dataclass
        class ProgressState:
            experiment_id: str
            device: str
            total_steps: int
            step: int = 0
            epoch: int = 0
            tokens_processed: int = 0
            cumulative_seconds: float = 0.0
            last_checkpoint_step: int = 0

        class ProgressReporter:
            def __init__(self, state: ProgressState):
                self.state = state
                self.session_start = time.perf_counter()
                self.last_time = self.session_start
                self.last_step = state.step
                self.last_tokens = state.tokens_processed
                self.bar = tqdm(total=state.total_steps, initial=state.step, desc=f"{state.experiment_id} [{state.device}]")

            def update(self, *, step: int, epoch: int, loss: float, learning_rate: float, tokens_processed: int, gradient_norm: float | None):
                now = time.perf_counter()
                dt, ds, dtoken = now - self.last_time, step - self.last_step, tokens_processed - self.last_tokens
                session_elapsed = now - self.session_start
                cumulative = self.state.cumulative_seconds + session_elapsed
                remaining = max(0, self.state.total_steps - step)
                steps_per_second = ds / dt if dt > 0 else 0.0
                eta = remaining / steps_per_second if steps_per_second > 0 else math.inf
                self.bar.update(max(0, ds))
                self.bar.set_postfix({
                    "epoch": epoch, "loss": f"{loss:.4f}", "lr": f"{learning_rate:.3g}",
                    "tok": tokens_processed, "tok/s": f"{dtoken/dt:.0f}" if dt > 0 else "0",
                    "step/s": f"{steps_per_second:.2f}", "grad": "NA" if gradient_norm is None else f"{gradient_norm:.3f}",
                    "elapsed_s": f"{cumulative:.0f}", "eta_s": "inf" if not math.isfinite(eta) else f"{eta:.0f}",
                    "ckpt": self.state.last_checkpoint_step,
                })
                self.state.step, self.state.epoch, self.state.tokens_processed = step, epoch, tokens_processed
                self.last_time, self.last_step, self.last_tokens = now, step, tokens_processed
        """
    ),
    md("## 8. Recovery/bootstrap discovery"),
    code(
        """
        def validate_zip(path: Path) -> dict[str, Any]:
            with zipfile.ZipFile(path, "r") as archive:
                bad = archive.testzip()
                if bad is not None:
                    raise zipfile.BadZipFile(f"CRC failure: {bad}")
                return {"path": str(path), "members": archive.namelist(), "sha256": sha256_file(path)}

        def discover_recovery_archives(input_root: Path) -> list[Path]:
            return sorted(Path(input_root).rglob("phase1_submission01_recovery.zip")) if Path(input_root).exists() else []

        RECOVERY_CANDIDATES = discover_recovery_archives(Path(CFG.input_root))
        print("Recovery candidates:", [str(path) for path in RECOVERY_CANDIDATES])
        if len(RECOVERY_CANDIDATES) > 1:
            print("STOP: select one archive explicitly; automatic conflict resolution is disabled.")

        def restore_recovery_archive(archive_path: Path, destination: Path) -> None:
            validate_zip(archive_path)
            destination = Path(destination).resolve()
            with zipfile.ZipFile(archive_path, "r") as archive:
                for member in archive.infolist():
                    target = (destination / member.filename).resolve()
                    if destination not in target.parents and target != destination:
                        raise ValueError(f"Unsafe ZIP member: {member.filename}")
                archive.extractall(destination)
        """
    ),
    md("## 9. Disposable smoke tests before full training"),
    code(
        """
        def run_infrastructure_smoke_tests() -> dict[str, str]:
            results: dict[str, str] = {}
            with tempfile.TemporaryDirectory() as temporary_name:
                temporary = Path(temporary_name)
                fixture = [
                    {"id": "a", "text": " ".join(["alpha"] * 700)},
                    {"id": "b", "text": " ".join(["beta"] * 700)},
                    {"id": "c", "text": " ".join(["gamma"] * 700)},
                ]
                dataset_path = temporary / "fixture.json"
                atomic_write_json(dataset_path, fixture)
                docs, rejected = load_documents(dataset_path, "fixture", "id", "text")
                assert len(docs) == 3 and not rejected
                results["dataset_loading_and_schema"] = "PASS"
                train, validation, split = deterministic_document_split(docs, 1 / 3, 781)
                assert not ({doc.uid for doc in train} & {doc.uid for doc in validation})
                save_or_verify_split(temporary / "split.json", split)
                results["document_split_and_leakage"] = "PASS"
                examples = make_continuation_examples(validation)
                assert len(examples) == 1 and len(examples[0]["prompt"].split()) == 512 and len(examples[0]["reference"].split()) == 128
                results["local_continuation_examples"] = "PASS"
                sampler = StatefulRandomSampler(range(10), seed=9)
                iterator = iter(sampler)
                prefix = [next(iterator) for _ in range(4)]
                state = sampler.state_dict()
                resumed = StatefulRandomSampler(range(10), seed=9)
                resumed.load_state_dict(state)
                suffix = list(resumed)
                assert len(prefix + suffix) == 10 and len(set(prefix + suffix)) == 10
                results["sampler_continuation"] = "PASS"
                history = JsonlHistory(temporary / "history.jsonl")
                history.append({"step": 1, "loss": 1.0})
                assert history.records() == [{"step": 1, "loss": 1.0}]
                results["history_continuity"] = "PASS"
                dummy_prediction = {"1": {"text": " ".join(["word"] * 128)}}
                prediction_path = temporary / "prediction.json"
                atomic_write_json(prediction_path, dummy_prediction)
                zip_path = temporary / "submission.zip"
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
                    archive.write(prediction_path, "prediction.json")
                assert validate_zip(zip_path)["members"] == ["prediction.json"]
                results["zip_integrity"] = "PASS"
            return results

        SMOKE_RESULTS = run_infrastructure_smoke_tests()
        print(json.dumps(SMOKE_RESULTS, indent=2))
        """
    ),
    code(
        """
        # MODEL/TOKENIZER SMOKE TEST — run only after your implementations are loaded.
        def run_student_integration_smoke_test(model, tokenizer, model_config, reconstruct_model, device="cuda:0"):
            tokenizer_meta = tokenizer_metadata(tokenizer, DIRS["tokenizer"])
            report = validate_model_contract(model, tokenizer_meta["vocab_size"], device)
            clone = reconstruct_model(model_config)
            clone.load_state_dict(model.state_dict(), strict=True)
            report["reconstruction"] = "PASS"
            atomic_write_json(DIRS["manifests"] / "student_integration_smoke_test.json", report)
            return report

        # run_student_integration_smoke_test(MODEL, tokenizer, MODEL_CONFIG, RECONSTRUCT_MODEL)
        """,
        tags=["student-model"],
    ),
    md("## 10. Explicit full-training integration point"),
    code(
        """
        # STUDENT IMPLEMENTATION REQUIRED: training algorithm and experiment-specific choices.
        # Your function should use the provided ProgressReporter, JsonlHistory, StatefulRandomSampler,
        # save_checkpoint_atomic, and restore_checkpoint utilities. It must return actual artifacts/metadata.
        STUDENT_TRAIN_FUNCTION: Callable[..., Mapping[str, Any]] | None = None
        TRAINING_CONFIG: dict[str, Any] | None = None  # Student records optimizer, schedule, batch/accumulation, precision, checkpoint intervals.

        def preflight_full_training() -> None:
            problems = []
            if not CFG.full_training_authorized:
                problems.append("CFG.full_training_authorized is False")
            if CFG.innovation_hypothesis.startswith("STUDENT MUST"):
                problems.append("innovation hypothesis is not recorded")
            if CFG.ablation_control.startswith("STUDENT MUST"):
                problems.append("ablation control is not recorded")
            if MODEL is None or MODEL_CONFIG is None or RECONSTRUCT_MODEL is None:
                problems.append("student model/reconstruction is not integrated")
            if STUDENT_TRAIN_FUNCTION is None or TRAINING_CONFIG is None:
                problems.append("student training function/config is not integrated")
            required = [DIRS["manifests"] / "document_split.json", DIRS["tokenizer"] / "tokenizer_metadata.json", DIRS["manifests"] / "model_validation.json"]
            problems.extend(f"missing {path}" for path in required if not path.exists())
            if problems:
                raise RuntimeError("Full-training preflight failed:\\n- " + "\\n- ".join(problems))

        # EXPENSIVE CELL: uncomment only after every preflight item passes.
        # preflight_full_training()
        # TRAINING_RESULT = STUDENT_TRAIN_FUNCTION(...)
        """,
        tags=["student-training", "expensive"],
    ),
    md("## 11. Local evaluation plumbing"),
    code(
        """
        def perplexity_from_loss(loss: float) -> float:
            return float(math.exp(loss)) if loss < 709 else math.inf

        def run_official_evaluator(
            evaluator: Callable[[Sequence[str], Sequence[str]], Mapping[str, float]],
            predictions: Sequence[str], references: Sequence[str], checkpoint_id: str,
        ) -> dict[str, Any]:
            if len(predictions) != len(references) or not predictions:
                raise ValueError("Predictions/references must be nonempty and aligned")
            metrics = {key: float(value) for key, value in evaluator(predictions, references).items()}
            record = {"checkpoint_id": checkpoint_id, "created_utc": utc_now(), "metrics": metrics, "examples": len(predictions)}
            output = DIRS["evaluations"] / f"{checkpoint_id}.json"
            atomic_write_json(output, record)
            csv_path = DIRS["evaluations"] / "metrics.csv"
            write_header = not csv_path.exists()
            with csv_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["checkpoint_id", "created_utc", "examples", *sorted(metrics)])
                if write_header:
                    writer.writeheader()
                writer.writerow({"checkpoint_id": checkpoint_id, "created_utc": record["created_utc"], "examples": len(predictions), **metrics})
            return record

        # OFFICIAL_EVALUATOR = ...  # Import the supplied script after inspecting its API.
        # EVAL_RECORD = run_official_evaluator(OFFICIAL_EVALUATOR, dev_predictions, dev_references, checkpoint_id)
        """
    ),
    code(
        """
        def plot_saved_history(history_path: Path, output_path: Path) -> None:
            import matplotlib.pyplot as plt
            records = JsonlHistory(history_path).records()
            if not records:
                raise ValueError("No history records to plot")
            steps = [row["step"] for row in records]
            figure, axes = plt.subplots(2, 2, figsize=(12, 8))
            fields = [("loss", "Training loss"), ("learning_rate", "Learning rate"), ("tokens_per_second", "Tokens/s"), ("gpu_memory_gib", "GPU memory GiB")]
            for axis, (field_name, title) in zip(axes.flat, fields):
                x, y = zip(*[(row["step"], row[field_name]) for row in records if row.get(field_name) is not None]) if any(row.get(field_name) is not None for row in records) else ([], [])
                axis.plot(x, y)
                axis.set_title(title)
                axis.set_xlabel("Optimizer step")
                axis.grid(alpha=0.25)
            figure.tight_layout()
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output_path, dpi=160)
            plt.close(figure)

        # plot_saved_history(DIRS["histories"] / "training.jsonl", DIRS["plots"] / "training_history.png")
        """
    ),
    md("## 12. Generation and prediction management"),
    code(
        """
        WORD_PATTERN = re.compile(r"\\S+")

        def words(text: str) -> list[str]:
            return WORD_PATTERN.findall(text.strip())

        def exact_word_prefix(text: str, count: int = 128) -> str:
            units = words(text)
            if len(units) < count:
                raise ValueError(f"Continuation has {len(units)} words; required {count}. Generation must continue—padding is forbidden.")
            return " ".join(units[:count])

        @torch.inference_mode()
        def greedy_continuation(
            model: torch.nn.Module, tokenizer: Any, prompt: str, device: str,
            target_words: int = 128, token_chunk: int = 96, max_total_new_tokens: int = 1024,
        ) -> str:
            model.eval()
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = encoded["input_ids"].to(device)
            prompt_length = input_ids.shape[1]
            generated = input_ids
            produced = 0
            while produced < max_total_new_tokens:
                request = min(token_chunk, max_total_new_tokens - produced)
                generated = model.generate(input_ids=generated, max_new_tokens=request, do_sample=False)
                produced = generated.shape[1] - prompt_length
                continuation_ids = generated[0, prompt_length:]
                text = tokenizer.decode(continuation_ids, skip_special_tokens=True)
                if len(words(text)) >= target_words:
                    return exact_word_prefix(text, target_words)
                if getattr(tokenizer, "eos_token_id", None) is not None and generated[0, -1].item() == tokenizer.eos_token_id:
                    break
            raise RuntimeError(f"Generation ended with fewer than {target_words} words; no fabricated padding was added")

        def load_test_records(path: Path, id_field: str, prompt_field: str) -> list[dict[str, str]]:
            records = records_from_container(load_json_container(path))
            output, seen = [], set()
            for index, record in enumerate(records):
                if id_field not in record or prompt_field not in record:
                    raise KeyError(f"Test record {index} lacks required fields")
                identifier, prompt = str(record[id_field]), normalize_text(record[prompt_field])
                if identifier in seen:
                    raise ValueError(f"Duplicate official test ID: {identifier}")
                seen.add(identifier)
                output.append({"id": identifier, "prompt": prompt})
            return output
        """
    ),
    code(
        """
        def generate_official_predictions(model, tokenizer, test_records, device: str) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
            predictions, failures = {}, []
            for record in tqdm(test_records, desc="Official test generation"):
                try:
                    continuation = greedy_continuation(model, tokenizer, record["prompt"], device, target_words=CFG.decoding["target_words"])
                    predictions[record["id"]] = {"text": continuation}
                except Exception as exc:
                    failures.append({"id": record["id"], "error": f"{type(exc).__name__}: {exc}"})
            atomic_write_json(DIRS["predictions"] / "generation_failures.json", failures)
            if failures:
                raise RuntimeError(f"Generation failed for {len(failures)} IDs; prediction file was not marked ready")
            return predictions, failures

        # TEST_PATH = require_unique_file(DISCOVERED, CFG.test_file)
        # TEST_RECORDS = load_test_records(TEST_PATH, CFG.test_id_field, CFG.test_prompt_field)
        # PREDICTIONS, FAILURES = generate_official_predictions(MODEL, tokenizer, TEST_RECORDS, "cuda:0")
        """,
        tags=["manual-generation"],
    ),
    md("## 13. Codabench validation and packaging"),
    code(
        """
        def validate_predictions(predictions: Any, official_ids: Sequence[str], target_words: int = 128) -> dict[str, Any]:
            errors = []
            if not isinstance(predictions, dict):
                return {"passed": False, "errors": ["prediction root must be an object"]}
            expected, actual = set(map(str, official_ids)), set(map(str, predictions))
            if missing := sorted(expected - actual):
                errors.append(f"missing IDs: {missing[:20]}")
            if unknown := sorted(actual - expected):
                errors.append(f"unknown IDs: {unknown[:20]}")
            for identifier in sorted(expected & actual):
                entry = predictions[identifier]
                if not isinstance(entry, dict) or set(entry) != {"text"}:
                    errors.append(f"{identifier}: entry must contain only a text field")
                    continue
                text = entry["text"]
                if not isinstance(text, str) or not text.strip():
                    errors.append(f"{identifier}: empty/non-string text")
                elif len(words(text)) != target_words:
                    errors.append(f"{identifier}: {len(words(text))} words, expected {target_words}")
            return {"passed": not errors, "errors": errors, "expected_ids": len(expected), "actual_ids": len(actual)}

        def create_submission_zip(predictions: Mapping[str, Any], official_ids: Sequence[str], output_dir: Path) -> tuple[Path, dict[str, Any]]:
            report = validate_predictions(predictions, official_ids, CFG.decoding["target_words"])
            if not report["passed"]:
                raise ValueError("Submission validation failed:\\n" + "\\n".join(report["errors"]))
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            prediction_path = output_dir / "prediction.json"
            atomic_write_json(prediction_path, predictions)
            zip_path = output_dir / "submission.zip"
            with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".zip", delete=False) as handle:
                temporary = Path(handle.name)
            try:
                with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
                    archive.write(prediction_path, "prediction.json")
                zip_report = validate_zip(temporary)
                if zip_report["members"] != ["prediction.json"]:
                    raise ValueError(f"Invalid ZIP root structure: {zip_report['members']}")
                os.replace(temporary, zip_path)
            finally:
                temporary.unlink(missing_ok=True)
            report["zip"] = validate_zip(zip_path)
            atomic_write_json(output_dir / "submission_validation.json", report)
            return zip_path, report

        # SUBMISSION_ZIP, SUBMISSION_REPORT = create_submission_zip(PREDICTIONS, [r["id"] for r in TEST_RECORDS], DIRS["submission"])
        """
    ),
    md("## 14. Recovery archive and integrity manifest"),
    code(
        """
        def build_integrity_manifest(root: Path, excluded: set[Path] | None = None) -> dict[str, Any]:
            root = Path(root).resolve()
            excluded = {Path(path).resolve() for path in (excluded or set())}
            files = []
            for path in sorted(item for item in root.rglob("*") if item.is_file()):
                if path.resolve() in excluded:
                    continue
                files.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
            return {"created_utc": utc_now(), "root": str(root), "files": files}

        def create_recovery_zip(root: Path) -> tuple[Path, dict[str, Any]]:
            root = Path(root).resolve()
            recovery_dir = root / "recovery"
            output = recovery_dir / "phase1_submission01_recovery.zip"
            previous = recovery_dir / "phase1_submission01_recovery.previous.zip"
            recovery_dir.mkdir(parents=True, exist_ok=True)
            excluded = {output, previous}
            manifest = build_integrity_manifest(root, excluded)
            manifest_path = root / "manifests" / "recovery_integrity.json"
            atomic_write_json(manifest_path, manifest)
            with tempfile.NamedTemporaryFile(dir=recovery_dir, suffix=".zip", delete=False) as handle:
                temporary = Path(handle.name)
            try:
                with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
                    for item in sorted(path for path in root.rglob("*") if path.is_file() and path.resolve() not in excluded):
                        archive.write(item, item.relative_to(root).as_posix())
                validate_zip(temporary)
                if output.exists():
                    shutil.copy2(output, previous)
                os.replace(temporary, output)
                report = validate_zip(output)
                atomic_write_json(recovery_dir / "recovery_validation.json", report)
                return output, report
            finally:
                temporary.unlink(missing_ok=True)

        # RECOVERY_ZIP, RECOVERY_REPORT = create_recovery_zip(ROOT)
        """
    ),
    md("## 15. Final PASS/FAIL verification"),
    code(
        """
        def final_verification(cfg: RunConfig, model: Any = None, tokenizer_meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
            checks: dict[str, bool] = {}
            checks["trained_from_scratch_attested"] = bool(cfg.trained_from_scratch_attestation)
            validation_path = DIRS["manifests"] / "model_validation.json"
            model_report = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {}
            checks["parameter_count_50m_to_300m"] = 50_000_000 <= int(model_report.get("total", -1)) <= 300_000_000
            checks["tokenizer_registered"] = (DIRS["tokenizer"] / "tokenizer_metadata.json").exists()
            checks["split_manifest_present"] = (DIRS["manifests"] / "document_split.json").exists()
            checks["training_history_present"] = (DIRS["histories"] / "training.jsonl").exists()
            checks["evaluation_records_present"] = any(DIRS["evaluations"].glob("*.json"))
            checks["final_or_best_model_present"] = any(DIRS["checkpoints"].glob("*final*.pt")) or any(DIRS["checkpoints"].glob("*best*.pt"))
            submission_zip = DIRS["submission"] / "submission.zip"
            checks["submission_zip_valid"] = False
            if submission_zip.exists():
                try:
                    checks["submission_zip_valid"] = validate_zip(submission_zip)["members"] == ["prediction.json"]
                except Exception:
                    pass
            recovery_zip = DIRS["recovery"] / "phase1_submission01_recovery.zip"
            checks["recovery_zip_valid"] = False
            if recovery_zip.exists():
                try:
                    validate_zip(recovery_zip)
                    checks["recovery_zip_valid"] = True
                except Exception:
                    pass
            checks["student_architecture_source_present"] = any(DIRS["model"].glob("*.py"))
            overall = all(checks.values())
            report = {"overall": "PASS" if overall else "FAIL", "checks": checks, "created_utc": utc_now()}
            atomic_write_json(DIRS["manifests"] / "final_verification.json", report)
            print(json.dumps(report, indent=2))
            return report

        # FINAL_REPORT = final_verification(CFG, MODEL, TOKENIZER_META)
        """
    ),
    md(
        """
        ## 16. Student-owned cells and decisions

        You must independently provide and document:

        1. The algorithmic hypothesis and a fair ablation/control. A standard decoder-only model can be the control, but the professor's guidance means the submission should clearly isolate a meaningful innovation.
        2. The model architecture source, including attention/position/normalization choices.
        3. The tokenizer choice or training procedure using only official Phase 1 data.
        4. Verified JSON field mappings after inspecting the official files.
        5. Validation fraction, source-mixing ratios, optimizer, learning-rate schedule, batch/accumulation, precision, training duration/steps, and checkpoint intervals.
        6. The actual training loop and any novel training algorithm.
        7. Import/adaptation of the supplied official BERTScore, ROUGE-L, and BLEU-4 evaluator.
        8. Experimental analysis and conclusions.
        9. The required LLM-use declaration: this notebook's data/checkpoint/validation/packaging infrastructure was AI-assisted; architecture, tuning, and scientific analysis were not generated here.

        Do not mark a submission ready until the final verification report prints `PASS`.
        """
    ),
]

nbf.write(nb, OUTPUT)
print(f"Wrote {OUTPUT} with {len(nb.cells)} cells")
