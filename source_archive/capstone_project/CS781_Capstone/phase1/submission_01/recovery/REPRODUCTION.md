# Phase 1 Submission 1 recovery notes

The recovery export created by the notebook includes this file, source/configuration, tokenizer files, compatible checkpoints, split/provenance manifests, histories, evaluation outputs, predictions, plots, checksums, and the separate Codabench ZIP when present.

Restore only through the notebook's recovery section. It validates CRC integrity, blocks unsafe ZIP paths, rejects ambiguous multiple archives, and verifies dataset/tokenizer fingerprints before loading checkpoint state.

Files inside `/kaggle/working` are temporary until saved as a Kaggle output/version or uploaded as a private Kaggle dataset. Export and preserve a validated recovery ZIP before the session expires.
