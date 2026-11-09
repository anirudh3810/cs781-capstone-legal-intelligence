# CS781 Capstone: In-House Legal Intelligence System

Anirudh Bankhede · IIT Kanpur · Roll 240131 · Codabench `a_240131`

This repository contains the training, inference, validation and recovery code for seven legal NLP phases. Four compact foundation architectures were trained from scratch, then reused in task-specific descendants. The downstream experiments compare full and partial tuning, low-rank updates, bottleneck adapters, activation scaling, soft prompts and distillation.

## Start here

Use the self-contained notebooks in the table below on Kaggle. Their cells write the Python runtime and orchestrate the experiment; the `.py` modules alongside them make the implementation reviewable. The original checkpoints and restricted official data are supplied separately as private Kaggle Inputs. No model checkpoint or restricted legal corpus is distributed in this repository.

| Phase | Task | Preferred saved implementation | Official Codabench score | Valid subs | Recorded local evidence |
|---|---|---|:---:|:---:|---|
| 1 | Legal continuation | [Base 04 notebook](phase1/notebooks/CS781_Phase1_Submission04_Infrastructure_FIXED.ipynb) | **0.8257** | 4 | 55.113M parameters; validation loss 2.68014; 304.584M processed tokens |
| 2 | Exact character-span NER | [Span V3 hybrid](phase2/notebooks/CS781_Phase2_Span_V3_RunAll.ipynb) | **0.5276** | 6 | Selected OOF macro F1 0.48583, with reused selection labels |
| 3 | Sentence rhetorical roles | [One-submission ensemble](phase3/notebooks/CS781_Phase3_One_Strong_Submission_RunAll.ipynb) | **0.5058** | 6 | Dev supported-label macro F1 0.54942; all-13 F1 0.46489 |
| 4 | Binary judgment outcome | [Archived run notebook](phase4/archived_run/CS781_Phase4_Six_Submissions_RunAll.ipynb), slot 02 | **0.7553** | 6 | Dev macro F1 0.75510; outer CV absent |
| 5 | Multi-label statute IDs | [Archived run notebook](phase5/archived_run/CS781_Phase5_Six_Submissions_RunAll.ipynb), slot 01 | **0.3777** | 6 | Holdout macro F1 0.37934; slot 06 distillation: 0.37650 |
| 6 | Legal summarization | [Six-method notebook](phase6/notebooks/CS781_Phase6_Six_Submissions_RunAll.ipynb), slot 02 | **0.8234** | 6 | Untouched audit BERTScore F1 0.82624 on 352 cases |
| 7 | Prior-case retrieval | [Six-method notebook](phase7/notebooks/CS781_Phase7_Six_Submissions_RunAll.ipynb), slot 01 | **0.3077** | 7 | Tuning-sample micro F1@7 0.34367 on 83 queries |

The table above reports both **Official Codabench Best Scores** (recorded on 9 November 2026 across 41 valid submissions) and local validation evidence. Selection of a final official winner remains provisional. All available alternatives are included, so the officially winning variant can be selected without omitting its code. The machine-readable selection record is [evidence/selected_models.json](evidence/selected_models.json).

## Repository structure

```text
phase1/               Four original model sources, saved configs, pretraining notebooks
phase2/               Six NER adaptations, span V1/V2/V3, diagnostics, notebook builders
phase3/               Six adaptations and later ensemble, hierarchy and context experiments
phase4/               Hierarchical judgment classifier and exact archived runtime
phase5/               Hierarchical multi-label classifier and exact archived runtime
phase6/               Six summarizers, deterministic decoding and recovery orchestration
phase7/               Hybrid retrieval, six adaptations, official evaluator, recovery tests
evidence/             Aggregate numeric results, curves and source provenance
tools/                Release checks; no training data
requirements.txt      Shared Python dependencies
```

## Environment and authorized inputs

Python 3.10 or newer and a recent CUDA-enabled PyTorch environment are expected. The notebooks were written for Kaggle's two-GPU accelerator. Use a clean Kaggle notebook and select GPU ×2; the exact GPU type and available quota depend on the account. Install shared packages locally with:

```bash
python -m pip install -r requirements.txt
```

Accept course dataset access at [Exploration-Lab/CS-781-Capstone](https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone), enable Kaggle Internet, add a read token as a Kaggle Secret named `HF_TOKEN`, and explicitly enable that secret for the notebook. Never put a token in a source cell or commit it. An offline alternative is to attach the exact official files requested by the chosen phase's notebook.

For Phases 2–7 attach the four original, locally trained Phase 1 recoveries:

```text
phase1_submission01_recovery.zip
phase1_submission02_recovery.zip
phase1_submission03_recovery.zip
phase1_submission04_recovery.zip
```

The later notebook loaders also support selected extracted Kaggle folder layouts. Keep the recovery's `model/`, `tokenizer/` and `checkpoints/selected_model.pt` hierarchy. Do not replace a base with an unrelated pretrained model. Each task stages its own descendants from the original base; tasks are not chained through one another's predictions.

## Architecture and methodology

### Phase 1: four scratch-trained foundations

| Base | Structure | Positions / normalization | Parameters | Saved training |
|---|---|---|---:|---|
| 01 | 8-layer causal decoder, width 768, 12 heads, FFN 2048, SwiGLU | Learned positions, pre-LayerNorm | 82,011,648 | 12,000 steps; 74,410,133 processed tokens |
| 02 | 6-layer encoder + 6-layer decoder, width 512, 8 heads, FFN 2048 | Learned positions, pre-LayerNorm, GELU | 61,342,976 | 12,000 steps; 242,874,217 processed tokens |
| 03 | Shared 24-layer prefix/continuation stack, width 384, 6 heads, FFN 1024 | RoPE, pre-RMSNorm, SwiGLU | 52,206,376 | 10,000 steps; 203,014,261 processed tokens |
| 04 | 14-layer causal decoder, width 512, 8 query / 2 KV heads, FFN 1536 | RoPE, pre-RMSNorm, local window 256 + global anchors | 55,113,128 | 15,000 steps; 304,583,885 processed tokens |

Tokenizers are 32k byte BPE for bases 01/02 and 25k byte BPE for 03/04. Data are official Acts, High Courts, Supreme Court and Tribunal sources. Equal source probabilities are probabilities of **document draws**, not equal token shares. Saved corpus subsets and validation protocols differ, so the losses are not a controlled architecture comparison. Exact numeric configurations are under `phase1/baseXX/saved_config/`.

Run the appropriate Phase 1 notebook in prepare mode first, review its corpus/schema checks, then select the training mode described in its cells. Later sessions may restore the corresponding recovery. The original base-01 architecture and configuration are present, but the original base-01 training notebook was not found in the supplied folder. Bases 02–04 have their original self-contained training notebooks.

### Phase 2: legal named entities

The six initial experiments use BIOUL tagging and character-boundary refinement: full base01, partial base02, LoRA base03, multiplicative LoRMA base04, bottleneck adapter base01, and base03 student distillation. Later span experiments directly score start/end boundaries and pairs in 384-token windows with stride 288 and a 64-token span cap. V2 adds residual token refinement and upper-layer tuning. V3 adds a biaffine-style multiplicative start/end interaction, unfreezes 12 upper blocks, and selects per-label sources from V2/V3 raw or reranked candidates.

Original text is preserved so offsets remain exact Python character indices. Three document folds cover 70 public labeled documents. Checkpoint selection, per-label source selection and calibration reuse OOF labels; the reported selected OOF F1 is optimistic.

For a clean reconstruction, run Span Extractor V1, then V2, then V3 in order; each continuation validates its predecessor's recovered states. To resume V3, attach the V1/V2 recoveries, V3 recovery if present, original base03, and official Phase 2 files. The V3 notebook is not an independent cold start. The original six-method notebook additionally requires `phase2_method_common.py` and the six `phase2_method_XX.py` modules as Kaggle Inputs.

### Phase 3: rhetorical roles

Initial experiments compare LoRA, rank-stabilized LoRA, IA³, LoRMA, prompt tuning and distillation. The one-submission experiment trains full base03/base01 models with three-layer bidirectional GRU document heads and selects a blend with prior models on development data. It has the highest saved development supported-label F1. The separate HighF1 branch uses base03, target-focused windows up to 448 tokens, attention/mean/max pooling, domain and position embeddings, a two-layer bidirectional GRU over 12-sentence windows, pair heads and optional CRF decoding. Five document folds prevent same-document sentence leakage. Raw decoding was selected by OOF evidence; the optional wider branch did not enter the final model.

Use the six-method notebook first. The one-submission notebook restores its six-method artifacts. The HighF1 notebook can train its separate branch from the original foundations. Compare supported-label and fixed all-13 macro F1 explicitly, especially when rare labels are absent from a split.

### Phase 4: judgment outcome

Encode up to eight 640-token overlapping chunks with a Phase 1 descendant, project to width 256, add relative document positions, contextualize chunks, and concatenate learned attention, mean and max summaries. A binary head is trained with a case-level objective. A dev-selected probability threshold produces labels 0/1. All six adaptations are listed below. The saved best dev method is partial tuning of base03 (slot02), threshold 0.55, selected epoch3.

The archive has no completed outer folds: `PROVISIONAL_0_FOLDS` is a real limitation. Use the `archived_run` notebook to preserve the schema handling that produced the saved reports. The accelerated slot06 runtime is also preserved separately; the supplied general workspace runtime differs from it.

### Phase 5: statute identification

The encoder processes four 512-token chunks per training case and up to twelve at inference, including statute-cue windows. A chunk Transformer feeds a global multi-label head and label-specific max evidence. Independent sigmoid scores are decoded with calibrated thresholds. Repeated statute IDs in the released data are deduplicated by the archived parser. The updated recovery contains six completed reports and six distinct prediction packages. Partial base03 (slot01) remains strongest on holdout and selected dev macro F1. Distilled base03 (slot06, the last submission) achieves a close holdout macro F1 of 0.376503 (micro F1 0.478877) with higher subset accuracy (0.045151 vs 0.041427) and 32.0% fewer trainable parameters (15.08M vs 22.18M).

| Slot | Method / base | Holdout macro F1 | Holdout micro F1 | Holdout subset acc | Selected dev macro F1 | Trainable parameters | Selected final epoch |
|---|---|---:|---:|---:|---:|---:|---:|
| 01 | Partial / B03 | 0.379343 | 0.481324 | 0.041427 | 0.385573 | 22,175,306 | 7 |
| 02 | Full / B01 | 0.374302 | 0.482138 | 0.046858 | 0.382241 | 82,994,378 | 6 |
| 03 | LoRMA / B04 | 0.343317 | 0.448398 | 0.033514 | 0.348887 | 1,863,370 | 7 |
| 04 | Adapters / B02 | 0.327873 | 0.432402 | 0.029015 | 0.329778 | 1,313,866 | 6 |
| 05 | Adaptive rank / B03 | 0.343392 | 0.452210 | 0.040807 | 0.343983 | 1,868,426 | 7 |
| 06 | Distillation / B03 | 0.376503 | 0.478877 | 0.045151 | 0.380021 | 15,078,474 | 7 |

These local evaluations remain separate from official Codabench scores. Each method's test package covers 13,019 cases.

Use the `archived_run` notebook for the saved data parser. The current workspace source rejects duplicate IDs and is retained for comparison. The archived notebook embeds the saved parser and helpers.

### Phase 6: summarization

The six adaptations are encoder-decoder LoRA, prefix-model adaptive rank masking, local/global LoRMA, decoder Houlsby adapters, encoder-decoder Pfeiffer adapters and prefix-model sequence distillation. Sentence selection packs beginning, ending, section and legal-cue evidence into architecture-specific source limits; reference summaries are capped at 480 tokens. Length control, deterministic greedy generation and repetition control produce the outputs. BERTScore is an evaluator only; its pretrained scorer is not a generative task backbone.

The saved recommendation is slot02 (base03, rank budget initially12), selected epoch3. Fitting/tuning/audit splits contain 6324/352/352 labeled rows. Two rows lacking references are skipped. The 352-case audit BERTScore F1 is 0.826237, ROUGE-1 F1 0.400379 and ROUGE-L F1 0.211570.

### Phase 7: prior-case retrieval

Full-document and passage BM25 rankings are fused with a shared neural encoder. Documents use up to six 384-token views; 256-dimensional normalized chunk vectors yield an equal-weight combination of document cosine and query-chunk max interaction. Reciprocal rank fusion combines sparse and neural orders. Gold-aware negatives, one hard-negative mining refresh, and train-only teacher scores support the six methods.

Slot01 full base03 has the best saved 83-query tuning score. Slots02–06 are partial base01, LoRMA base04, IA³ base02 encoder, soft prompts base03, and distilled base02 encoder. A matched two-pass gold-only control accompanies the KD student. The supplied recovery stops partway through slot05's fifth pass; a later standalone slot05 package exists, but its updated final checkpoint/report is absent. This is recorded separately from the other five completed archived reports.

## Method matrix

| Phase | 01 | 02 | 03 | 04 | 05 | 06 |
|---|---|---|---|---|---|---|
| 2 | Full / B01 | Partial / B02 | LoRA / B03 | LoRMA / B04 | Adapter / B01 | KD / B03 |
| 3 | LoRA / B03 | rsLoRA / B01 | IA³ / B02 | LoRMA / B04 | Prompt / B03 | KD / B03 |
| 4 | Full / B01 | Partial / B03 | LoRMA / B04 | Adaptive rank / B02 | Adapter / B03 | KD / B03 |
| 5 | Partial / B03 | Full / B01 | LoRMA / B04 | Adapter / B02 | Adaptive rank / B03 | KD / B03 |
| 6 | LoRA / B02 | Adaptive rank / B03 | LoRMA / B04 | Houlsby / B01 | Pfeiffer / B02 | KD / B03 |
| 7 | Full / B03 | Partial / B01 | LoRMA / B04 | IA³ / B02 encoder | Prompt / B03 | KD / B02 encoder |

The adaptive-rank modules are project implementations using importance/masking schedules, not a claim of byte-identical upstream AdaLoRA. Multiplicative updates preserve the identity initialization and original bias. Distillation supervision is generated only from fitting data.

## Execution, resume and outputs

1. Import the selected notebook from this repository into Kaggle. Attach authorized inputs and recoveries, enable the secret and Internet where required, and select the documented accelerator.
2. Run the setup and preparation cells. Confirm IDs, schema, split statistics and original checkpoint provenance before training.
3. Use **Run All** for the task pipeline. It trains method-specific descendants, validates, predicts and creates per-submission ZIPs. A complete six-method suite can require several Kaggle sessions.
4. Download or persist the most recent `CS781_PhaseN_All_Artifacts.zip` after every session. A draft working directory alone is not durable storage.
5. In a new backend, attach that recovery and rerun the same notebook. Hashes, completed stages and saved progress determine what is reused. Keep partial files and optimizer/RNG state.
6. Submit method ZIPs separately to Codabench. Each task's packaging code validates its schema; no notebook automatically uploads to the leaderboard.

Phases 4–6 use two-GPU distributed jobs with synchronized saving. Phase 7 uses two independent workers, each assigned one physical GPU; local `cuda:0` is the first visible GPU **inside that worker**. Phase 7 probes long-query/large-candidate batches, enables checkpointing on memory failure and reduces microbatches while preserving the effective batch and completed-example cursor. It saves approximately every five minutes and archives after both workers exit. A roughly 8.5-hour backend guard preserves time for packaging; rerunning an expired live backend does not renew its budget. Use a fresh backend and saved recovery.

Outputs must match each official schema. In Phase 7 the JSON is an object mapping each query-ID string to exactly seven unique candidate-ID strings from the correct test pool. Leading zeros are preserved. All Codabench prediction ZIPs put `prediction.json` directly at the root. The final report/code ZIP has a different format: `code/` and `report/` directly at its root.

## Validation and evidence

```bash
python tools/check_release.py
# Integration tests below require the original recoveries, PyTorch and task-specific setup:
python phase4/test_cjpe_runner.py
python phase5/test_lsi_runner.py
python phase6/test_summ_runner.py
python phase7/test_pcr_runner.py
python phase7/test_pipeline.py
python phase7/test_recovery.py
```

The release check validates syntax, notebook structure, selected entry points and credential patterns; it does not train models. Existing model/pipeline tests exercise real checkpoint loading with synthetic examples, stage transitions, output packaging, interruption and recovery. Actual GPU training measurements come from the saved Kaggle artifacts. [Source provenance](evidence/source_provenance.json) records exact archive member hashes. Numeric results and learning curves are exported without legal text in [verified_local_results.json](evidence/verified_local_results.json) and [learning_curves.json](evidence/learning_curves.json).

The official Phase 7 evaluator is preserved under `phase7/reference/`, with source and digest in that folder. No test labels are used for fitting or method selection. The isolated audit split is present only where the archived pipeline completed it; a tuning score is not relabeled as an audit or private score.

## References

The project uses the official course specification, starter/evaluation material and lecture slides, together with the following method references: [Transformer](https://arxiv.org/abs/1706.03762), [RoPE](https://arxiv.org/abs/2104.09864), [RMSNorm](https://arxiv.org/abs/1910.07467), [GQA](https://arxiv.org/abs/2305.13245), [LoRA](https://arxiv.org/abs/2106.09685), [AdaLoRA](https://arxiv.org/abs/2303.10512), [LoRMA](https://aclanthology.org/2025.findings-acl.527/), [IA³](https://arxiv.org/abs/2205.05638), [adapters](https://arxiv.org/abs/1902.00751), [prompt tuning](https://arxiv.org/abs/2104.08691), [distillation](https://arxiv.org/abs/1503.02531), [BERTScore](https://arxiv.org/abs/1904.09675), [ROUGE](https://aclanthology.org/W04-1013/), and [reciprocal rank fusion](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/).
