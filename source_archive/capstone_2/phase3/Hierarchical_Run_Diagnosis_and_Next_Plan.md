# Phase 3 hierarchical run: diagnosis and next plan

## What the 0.5058 submission actually contains

The new `report.json` says `selected: prior6`. The 2,197 predicted labels in
`CS781_Phase3_Hierarchical_Submission.zip` are **identical** to those in
`CS781_Phase3_Submission_06.zip`. This was a deliberate fallback, not a
packaging accident or an independently trained model earning the same score.

The fixed selection rule rejected the new models. Its development-set results
were:

| Prediction | Supported-label macro F1 | All-13 macro F1 | CL all-13 | IT all-13 |
|---|---:|---:|---:|---:|
| Original slot 06 | 0.4861 | 0.4113 | 0.3306 | 0.4335 |
| New five-fold plus full ensemble | 0.4323 | 0.3658 | 0.2841 | 0.3482 |
| Fixed 75% new / 25% slot 06 blend | 0.5061 | 0.4282 | 0.3290 | 0.4318 |

The blend cleared the development gain threshold, but the selection rule also
required the **unblended** new model to stay within 0.035 all-13 F1 of slot 06
in each domain. It missed that condition in both CL and IT. The blend's private
score is unknown. A previous ensemble gained on the same ten-document dev set
but lost private score, so the small dev gain alone is weak evidence.

## Why the new architecture did not improve

1. **Target-only input removed useful role context from the Phase 1 encoder.**
   Slot 06 encodes neighboring sentences and the target together. The new
   encoder sees only the target sentence. The later document head sees nearby
   sentence embeddings, but those embeddings are cached and the encoder cannot
   adapt to context-dependent document labels. This is a plausible cause, not
   proved in isolation. The training data contains at least 186 normalized
   sentence strings appearing with conflicting role labels across 405 instances.

2. **The new document head helps, but starts from weak sentence features.**
   Five-fold held-out macro F1 is 0.296 for the sentence classifier and 0.357
   after the document head. That 0.061 gain confirms useful document context,
   while the absolute result remains low. The final model alone reaches only
   0.414 supported-label F1 on official dev; ensemble averaging is not hiding
   a strong new individual model.

3. **The model collapses several important roles.** On official dev,
   `ArgumentRespondent` recall drops from 0.559 (slot 06) to 0.217 (new);
   `PrecedentNotReliedUpon` F1 drops from 0.269 to 0; `RulingByLowerCourt`
   F1 drops from 0.333 to 0.179. The new model predicts
   `RatioOfTheDecision` 722 times for 403 gold sentences. These are exactly
   the distinctions that need party identity and neighboring legal argument.

4. **The run did not use the requested compute budget.** Kaggle execution
   metadata records 36 minutes 56 seconds for the entire Run All cell. Each
   fold's encoder trained for about 3–5 minutes; each document head for about
   20–25 seconds. The original slot 06 alone recorded 6,824 training seconds
   (about 114 minutes). The new notebook had a 9.75-hour *deadline*, not a
   training objective. Logs show about 1.0 GiB allocated on rank 0 during
   encoder training and 0.18 GiB for the head on a 15 GiB T4. Both GPUs were
   launched through DDP, but no in-training utilization trace was collected;
   memory figures alone cannot establish their compute utilization.

5. **Checkpoint selection was noisy.** Each fold fitted 60 documents and
   chose checkpoints on only four inner-validation documents. The five encoder
   best epochs were 3, 9, 3, 8, and 8. Outer-fold document F1 ranged from
   0.366 to 0.476. This spread makes one tiny validation split a poor guide
   for a final 80-document fit.

6. **Rare classes impose a data limit.** `Dissent` has 284 training sentences
   but occurs in only two documents; `PrecedentOverruled` has three sentences
   in one document. Out-of-fold F1 for both is zero. More epochs on the same
   examples cannot create document diversity. Macro F1 is especially sensitive
   to these classes. The official dev set has no examples of either class, so
   it cannot validate their recall. Under the notebook's all-13 dev metric,
   those absent classes make the maximum possible score 11/13 = 0.846; a
   score of 0.8 would require about 0.945 average F1 on the 11 classes that
   are present. The private test class distribution is unknown.

These findings do not prove that a score of 0.8 is impossible, but they do not
support promising it. The present private result is 0.5058 and the next
experiment needs a real held-out gain before a private submission.

## Next experiment: context-aware, joint training

### 1. Establish a fair document-held-out baseline

Use fixed document folds with both CL and IT represented. Refit a slot-06-like
context model from the approved Phase 1 weights on the **same** fitting folds
as the new candidate. Do not compare a new held-out result to slot 06 trained
on all 80 documents; that would leak fold labels into the baseline. Report
all-13 and supported-label macro F1, domain scores, per-label scores and
confusion for every fold. Keep official dev as an audit; avoid tuning many
weights against its ten documents.

### 2. Run one architecture pilot before the expensive refit

Start from Phase 1 model 03 as the course requires. Feed a target sentence
with explicit target markers plus nearby text to the 24-layer encoder, keeping
the whole target when truncating. Pool **target tokens only**, then pass a
sequence of sentence vectors through a learned document model. Backpropagate
the document-label loss through the encoder and document head together.
Use learned document positions and domain information. This restores the
role cues that slot 06 retained and removes the frozen-feature bottleneck.

Begin with a 16–32-sentence document window and overlapping windows; add
longer context only when held-out errors show that it helps. Add a coarse
role auxiliary loss (Fact / Argument / Precedent / Ruling / Other) and a label
boundary loss. Inspect `ArgumentPetitioner` versus `ArgumentRespondent`,
`Ratio` versus `PrecedentReliedUpon`, and lower-court rulings every epoch.
Treat rules for the ultra-rare labels as a separate, validated ablation.

### 3. Spend compute where it changes the model

Use two-GPU DDP, mixed precision, activation checkpointing and token-budget
batches with richer 384–512-token sentence inputs. Log **both GPUs'** sampled
utilization, allocated/reserved memory, tokens per second, optimizer steps,
training loss and held-out F1. Increase workload to a safe memory level on the
T4s; do not equate high memory with high utilization. Plan roughly 2 hours
for the first pilot, then up to 6–7 hours for fold confirmations and the final
fit, with time reserved for inference, integrity checks and the recovery ZIP.
Stop or change strategy if held-out F1 plateaus or degrades; extra epochs alone
already failed to solve the role confusion.

### 4. Use a go/no-go gate

First compare the joint model to the slot-06-like baseline on the **same**
held-out documents. Require a meaningful macro-F1 gain across both domains,
and improvement in the respondent, precedent and ruling confusions, before
running more folds. A practical pilot gate is at least +0.03 to +0.05
supported-label macro F1 with no material domain regression. If it passes,
confirm on several document folds, then fit one final model on all 80 training
documents using a fixed epoch schedule from the folds. Select the final model
or a predeclared blend using fold evidence; use official dev as a sanity check,
not as a broad weight-search target.

### 5. Package and validate

Retain periodic optimizer checkpoints while training and verified recovery
archives. Before submission, verify the single root-level `prediction.json`,
all ten document IDs, all 2,197 sentence counts, the 13 allowed labels, and
that its predictions actually differ from prior slot 06. Record the chosen
checkpoint, fold metrics and archive hashes. Codabench private macro F1 is
still unknown until upload; 0.8 must remain a target rather than a guarantee.
