"""CPU integration smoke test using the four saved Phase 1 checkpoints."""
from __future__ import annotations

import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import torch

import utils as u
sys.modules["cs781_phase6_utils"] = u
import summ_runner as r


def extract(archive, destination):
    with zipfile.ZipFile(archive) as source:
        for name in source.namelist():
            if not (name.startswith(("model/", "tokenizer/")) or
                    name == "checkpoints/selected_model.pt") or name.endswith("/"):
                continue
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(name) as inp, target.open("wb") as out:
                shutil.copyfileobj(inp, out, 4 << 20)
    u.write_json(destination / "provenance.json",
                 {"checkpoint_sha256": u.sha(destination / "checkpoints/selected_model.pt")})


def main():
    torch.set_num_threads(2)
    project = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix="phase6_smoke_", dir=project / "phase6") as path:
        tmp = Path(path)
        r.ROOT = u.ROOT = tmp / "run"
        r.WORK = tmp
        r.ARTIFACT = u.ARTIFACT = tmp / "CS781_Phase6_All_Artifacts.zip"
        official = r.ROOT / "official"
        official.mkdir(parents=True)
        train = [{"id": f"doc_{i}", "document": (
            f"The petitioner filed an appeal in case {i}. The court considered "
            "the facts, the statute and the arguments. Accordingly, the appeal "
            "was allowed and the prior order was set aside."),
                  "summary": f"The court allowed the appeal in case {i} and set aside the prior order."}
                 for i in range(24)]
        train[0]["summary"] = {"sentences": [train[0]["summary"]]}
        train[1]["headnote"] = [train[1].pop("summary")]
        train[2]["summary"] = []
        train[3]["case_summary"] = train[3].pop("summary")
        test = [{"id": f"test_{i}", "document": (
            "The court considered the petition and dismissed the appeal.")}
                for i in range(3)]
        u.write_json(official / "train.json", train)
        u.write_json(official / "test.json", test)
        rows, test_rows = r.data()
        assert len(rows) == 23 and rows[0]["summary"] and rows[1]["summary"]
        assert rows[2]["summary"] and u.read_json(
            official / "label_audit.json")["skipped_without_reference"] == 1
        split = r.split_rows(rows)
        assert len(split) == len(rows) and set(split.values()) == {0, 1, 2}
        u.write_json(r.ROOT / "run_config.json", {"split": split})
        assert [x["source_id"] for x in test_rows] == [x["id"] for x in test]
        # Exercise Kaggle's ZIP staging path, not just pre-extracted bases.
        u.INPUT = project
        u.ROOT = tmp / "zip_stage"
        staged = u.stage_base("01")
        assert (staged / "model/decoder_lm.py").is_file()
        assert (staged / "checkpoints/selected_model.pt").is_file()
        assert (staged / "tokenizer/tokenizer.json").is_file()
        print("PASS PHASE-1 ZIP STAGING", flush=True)
        u.ROOT = r.ROOT
        for base in ("01", "02", "03", "04"):
            extract(project / f"phase1_submission{base}_recovery.zip",
                    r.ROOT / "bases" / base)
            r.token_cache(base)
        for slot in r.SLOTS:
            base = r.SLOTS[slot]["base"]
            cache = torch.load(r.ROOT / "token_cache" / f"base_{base}.pt",
                               map_location="cpu", weights_only=False)
            model, tok, info = r.model_for(slot, torch.device("cpu"))
            args, labels = r.batch_tensors(rows[:1], cache, tok, base, torch.device("cpu"))
            loss = r.forward_loss(model, args, labels, base)
            assert torch.isfinite(loss), slot
            loss.backward()
            missing = [name for name, p in model.named_parameters()
                       if p.requires_grad and p.grad is None]
            assert not missing, (slot, missing[:5])
            state = r.adaptation_state(model)
            r.load_adaptation(model, state)
            generated = r.generate_one(model, tok, base,
                                       cache[test_rows[0]["id"]]["source"],
                                       min_new=1, max_new=3)
            assert isinstance(generated, str) and generated.strip(), slot
            print("PASS", slot, r.SLOTS[slot]["method"],
                  "trainable", info["trainable_parameters"],
                  "loss", round(float(loss.detach()), 3), flush=True)
            del model, loss, state
        r.train_stage("01", "fit", max_minutes=.01)
        progress = u.read_json(r.ROOT / "runs/01/fit/status.json")
        assert progress["optimizer_steps"] >= 1
        first_steps = progress["optimizer_steps"]
        r.train_stage("01", "fit", max_minutes=.01)
        resumed = u.read_json(r.ROOT / "runs/01/fit/status.json")
        assert resumed["optimizer_steps"] > first_steps
        print("PASS RESUMED TRAINING", resumed["optimizer_steps"], flush=True)
        for slot in r.SLOTS:
            dest = r.ROOT / "submissions" / slot
            dest.mkdir(parents=True)
            u.write_json(dest / "partial_fit.json",
                         {row["source_id"]: "The court decided the case." for row in test_rows})
            status = r.ROOT / "runs" / slot / "fit" / "status.json"
            u.write_json(status, {"best_score": .5, "best_epoch": 1,
                                  "trainable_parameters": 1000})
            r.package_slot(slot)
            with zipfile.ZipFile(tmp / f"CS781_Phase6_Submission_{slot}.zip") as archive:
                assert archive.namelist() == ["prediction.json"]
        # Check archive restore with a tiny separate tree to avoid copying the
        # four real base checkpoints again.
        mini = tmp / "mini"
        u.ROOT = mini
        u.write_json(mini / "run_config.json", {"test": 1})
        u.save_torch(mini / "latest.pt", {"step": torch.tensor(3)})
        u.snapshot()
        shutil.rmtree(mini)
        mount = tmp / "mount"
        mount.mkdir()
        shutil.copy2(u.ARTIFACT, mount / u.ARTIFACT.name)
        u.INPUT = mount
        u.restore_artifact()
        assert u.read_json(mini / "run_config.json")["test"] == 1
        print("PASS RECOVERY AND SIX PACKAGES", flush=True)


if __name__ == "__main__":
    main()
