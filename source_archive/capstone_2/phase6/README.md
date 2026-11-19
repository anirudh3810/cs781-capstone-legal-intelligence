# Phase 6 Run All notebook

Use [CS781_Phase6_Six_Submissions_RunAll.ipynb](../CS781_Phase6_Six_Submissions_RunAll.ipynb) in Kaggle. Attach the four **original Phase 1 recovery ZIPs**, select the two-GPU accelerator, and enable Internet plus the Kaggle Secret HF_TOKEN with access to the official gated course dataset. Alternatively, attach both official Phase 6 train.json and test.json as Kaggle Inputs. Choose **Run All**.

The notebook embeds utils.py and summ_runner.py; separate source uploads are unnecessary. It trains six distinct Phase 1 adaptations, validates by the course's primary BERTScore F1, creates six ZIPs with root-level prediction.json, and writes CS781_Phase6_All_Artifacts.zip. It does **not** upload submissions to Codabench.

The loader accepts string, sentence-list, and structured summary references. It records missing labels in `official/label_audit.json` and skips a small number of unlabeled training rows. It stops with row IDs and field names if labels are missing at scale, to avoid accidentally training on documents as targets.

Training uses two-GPU PyTorch DDP, mixed precision, per-method batch-size probing, token caching, length bucketing, and gradient accumulation. A training command ends after about 20 minutes or at an epoch boundary. It saves a complete atomic training checkpoint around every 10 minutes and after every segment. The parent rebuilds the all-artifacts ZIP only after workers exit. The archive contains official data, original Phase 1 weights and tokenizers, code, training states, RNG states, validation and test predictions, reports, and submission ZIPs.

Kaggle's session limit may require multiple Run All executions of the **same** notebook. Before a session ends, download/save the newest all-artifacts ZIP or commit it as a Kaggle output. Attach it as a Kaggle Input along with the four original recoveries next time. The notebook detects completed stages and resumes unfinished ones. A file left only in a fatal Kaggle draft session is not guaranteed recoverable.

Local checks run on the real Phase 1 recoveries and synthetic Phase 6 examples:

    python -m py_compile phase6\utils.py phase6\summ_runner.py phase6\build_notebook.py
    python phase6\build_notebook.py
    python phase6\test_summ_runner.py

The real Phase 6 files and two CUDA GPUs are not present locally, so actual Kaggle throughput, true validation score, and private Codabench score remain unverified. Review the generated metrics and predictions before uploading. The model code is AI-assisted; disclose that truthfully in the final course report.
