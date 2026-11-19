# Phase 7 Run All notebook

Memory fix (6 November): probes now use long queries and five long candidate views per query, with automatic checkpointing/batch reduction after a training OOM. Existing runs from the original notebook are accepted through a checked configuration migration. Keep the working directory; old batch profiles are re-probed and completed-example positions are converted when reducing the batch.

Upload `CS781_Phase7_Six_Submissions_RunAll.ipynb` to Kaggle. Attach the four **original Phase 1 recovery datasets**, either the recovery ZIPs or Kaggle-extracted folders containing `model`, `tokenizer` and `checkpoints/selected_model.pt`. Attach the six official Phase 7 query/candidate JSON files, or enable Internet and grant this notebook access to a Kaggle Secret named `HF_TOKEN` with access to the gated official course dataset. Select the **two-GPU** accelerator and Run All.

The notebook embeds every source and the official evaluator. No separate source uploads are necessary. It trains full tuning on base 03, partial tuning on 01, LORMA on 04, IA³ on the 02 encoder, prompt tuning on 03, and a distilled 02 encoder. A two-pass gold-only student control shares the student's initialization, batch profile and sampling policy; matching epoch/step comparisons are recorded.

The implemented ranker combines full-document and passage BM25 with learned document vectors and chunk late interaction. It uses all-pool vector scoring, original checkpoint/tokenizer provenance, known-positive masking, hard-negative mining from the fitting split, a train-only retrieval teacher, and selection by the exact official Micro-F1@7. The optional cross-encoder research branch in the strategy is not part of this notebook; all six counted methods use the tested chunk-interaction pipeline.

Two independent worker processes are assigned one GPU each. This uses both GPUs without DDP synchronization. Mixed precision and a per-method batch/memory probe improve throughput. Memory utilization depends on actual case lengths; allocating all GPU memory does not necessarily improve useful training speed.

Workers save model/adaptation, optimizer, scaler, RNG, epoch and batch cursor about every **five minutes** and after each segment. A segment is at most approximately **18 minutes**, or one training pass, before handing control back. Candidate encoding, teacher scores and ranking outputs have resumable partial files. The parent waits until both workers exit, writes a report and atomically rebuilds:

    /kaggle/working/CS781_Phase7_All_Artifacts.zip

The archive includes original model sources/checkpoints, tokenizers, official data and revision hashes, token caches, optimizer states, best/latest/previous checkpoints, lexical ranks, teacher scores, selected vector caches, validation predictions, logs and finished submission ZIPs. The final notebook cell verifies every member's hash. A backend-time marker prevents Run All from resetting its own session budget.

Completed methods produce `CS781_Phase7_Submission_01.zip` through `_06.zip`. Each contains exactly root-level `prediction.json`: a query-ID mapping to seven unique candidate IDs from the test pool. The notebook validates files; it does not upload them to Codabench. If methods produce identical top-seven sets, the final report flags them for review rather than manufacturing arbitrary differences.

If the backend budget ends before completion, download/save the newest all-artifacts ZIP or commit the output while it is available. Attach it as a Kaggle Input in another session, keep the same notebook version, and Run All. Both ZIP and Kaggle-extracted recovery folders are supported. Completed stages are reused. In the same live backend, Run All can reuse existing working files, but it cannot extend Kaggle's actual session lifetime. Persistence alone does not guarantee recovery after a fatal draft-session loss.

Local verification:

    python -m py_compile phase7/pcr_runner.py phase7/utils.py phase7/build_notebook.py
    python phase7/test_pcr_runner.py
    python phase7/test_pipeline.py
    python phase7/test_recovery.py
    python phase7/build_notebook.py

The real-model integration check uses all four original checkpoints with synthetic retrieval pools. Actual Phase 7 files and two CUDA GPUs are not available locally, so real dataset throughput, validation F1 and private Codabench F1 are unverified. The notebook saves those measurements during its Kaggle run. Disclose AI assistance accurately in the course report.
