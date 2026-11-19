# Phase 3 context-joint run: diagnosis and next experiment

## What the identical 0.5058 score means

`CS781_Phase3_Context_Joint_All_Artifacts.zip` records `selected: prior6` and
`labels_changed_vs_slot06: 0`. I compared its embedded `submission.zip` with
`CS781_Phase3_Submission_06.zip`: all 2,197 predicted labels are identical.
The score does **not** measure the newly trained joint model on the private
test set. The notebook stopped after its pilot gate and packaged the old
submission again. This was a poor user-facing outcome even though the gate
correctly exposed the weak held-out result.

| Pilot, same 16 held-out documents | Supported-label macro F1 | All-13 macro F1 |
|---|---:|---:|
| Refit context baseline | 0.4587 | 0.3881 |
| New context-joint model | 0.4660 | 0.3943 |

The gain was only **0.0073**, below the predeclared +0.03 requirement. The
joint model trained for 15 epochs. Its development supported-label F1 peaked
at **0.5191 in epoch 10** and fell to **0.4855 by epoch 15**, while training
loss fell from **2.424 to 0.038**. This is overfitting. The refit baseline
peaked at 0.5223 on dev in epoch 3 and scored 0.4587 on held-out documents,
showing that the ten-document dev set is also a noisy guide to generalization.

This was not caused by idle accelerators. During joint training, sampled GPU
utilization averaged **95% on GPU 0 and 98% on GPU 1**. The model used about
2.3 GiB per GPU, but high utilization shows that adding epochs or batch size
merely to fill memory is not the next fix.

## The actual modeling error

The training set has 80 documents, not 16,845 independent examples. Two
documents alone contain **3,625 sentences (21.5% of training sentences)**.
The first held-out fold has 3,463 sentences, of which one case contributes
1,816. The test set also contains a 942-sentence CL case and a 420-sentence
IT case. Sentence-window training lets long cases dominate the updates while
still limiting the document head to 16 sentences of context.

Rhetorical roles are often long *sections*. In training, 73% of all
`ArgumentRespondent` sentences occur in runs at least 16 sentences long;
one run has 379 sentences. The longest `Fact` run has 570 sentences, and
the longest `PrecedentReliedUpon` run has 226. A local head cannot reliably
carry a speaker or section identity across such distances.

The first held-out fold's most frequent new-model confusions were:

| Gold -> predicted | Sentences |
|---|---:|
| Ratio -> PrecedentReliedUpon | 199 |
| ArgumentRespondent -> ArgumentPetitioner | 171 |
| PrecedentReliedUpon -> Ratio | 164 |
| Ratio -> Statute | 120 |
| RulingByLowerCourt -> Ratio | 83 |
| PrecedentNotReliedUpon -> PrecedentReliedUpon | 79 |

For example, a 185-sentence respondent section begins by naming counsel
appearing for the CCI. Once that cue leaves the 16-sentence window, many later
sentences look like generic legal arguments and are assigned to the petitioner.
Likewise, a quoted precedent can resemble the present court's reasoning until
the surrounding introduction or conclusion identifies whose view it is and
whether the court adopts it.

Two quick checks rule out easy fixes. A domain-aware transition decoder on
the existing held-out probabilities raised supported macro F1 from **0.466
to at most 0.478**; a contextual word-ngram linear classifier reached
**0.394**. They can be useful auxiliary signals but cannot close the gap.
With *gold* section boundaries supplied as an oracle, assigning a section
from its first 20 model predictions reached **0.674** on this one fold.
That is a diagnostic upper opportunity, **not** a deployable or unbiased
score: real boundaries must be predicted, and the same oracle rule reaches
only **0.537** on official dev. A new approach must learn boundaries,
speaker attribution, and span labels rather than merely smooth logits.

The 13-class macro metric also has a genuine data limit: `Dissent` appears
in only two training documents, and `PrecedentOverruled` in one document
with three sentences. They have no official dev examples. Their private-test
behavior cannot be established by more epochs on the current split.

## One next experiment: full-document section and source tracker

This is a different prediction unit and learning problem, not another larger
local classifier. It also addresses why the earlier hierarchical notebook
failed: that run froze **target-only** sentence embeddings before a 128-sentence
head. Here, the sentence encoder first learns with marked target plus nearby
text, and the document model receives out-of-sample local evidence, explicit
source cues, and the complete document sequence.

1. **Rebuild local evidence without overfitting.** Start each fold from the
   user's Phase 1 base-03 checkpoint, as required by the course. Encode the
   marked target and nearby text. Freeze the lower encoder layers, train the
   upper layers and role head with stronger dropout and weight averaging,
   and sample *documents* approximately equally so the two giant cases do
   not consume one-fifth of training updates. Select checkpoints on official
   dev or an inner training split; reserve the outer fold untouched.

2. **Track the whole document.** For each sentence, retain the Phase 1-based
   embedding, local role logits, normalized position, and learned lexical
   indicators for section headings, quoted/cited text, counsel/party mentions,
   lower-court attribution, and acceptance/rejection of a precedent. Run a
   lightweight hierarchical document encoder over the full 1-2k sentence
   sequence: blocks of 64-128 sentences followed by a global recurrent or
   attention layer. This is inexpensive compared with running the 24-layer
   token encoder on all sentences jointly and allows the model to carry a
   respondent/court/precedent source identity across hundreds of sentences.
   Do not train this head on overconfident, in-sample logits and then expect
   it to handle out-of-sample test logits. Within each outer fitting set,
   produce the head's training inputs by two-way document cross-fitting:
   train each local encoder on one half and extract features for the other.
   Train another local encoder on all outer fitting documents for the held-out
   fold. This extra work directly addresses train/test feature leakage.

3. **Predict spans and roles together.** Train a boundary head on gold role
   changes and a segment scorer using the beginning, end, and pooled interior
   of a candidate span. Decode with a duration-aware semi-Markov layer, while
   permitting adjacent same-role spans so 379- and 570-sentence sections are
   representable. Give singleton-prone `Issue`, `None`, and present-court
   rulings short-span options. Add explicit auxiliary distinctions for
   petitioner/respondent source and cited precedent relied/not relied. A
   fixed first-order transition matrix alone was tested and is inadequate.

4. **Validate across all five document folds.** Current fold 0 contains an
   1,816-sentence case; a single-fold gate is too unstable. Each fold holds
   out eight CL and eight IT documents. Compare the global span model against
   its *own* local encoder on the same held-out documents. Report pooled OOF
   all-13 and supported macro F1, CL and IT F1, per-document F1, transition
   F1, and the four major confusion pairs above. Require at least **+0.05
   supported macro F1** across pooled OOF predictions and no meaningful
   domain regression before training a full-data submission. This is an
   experiment gate, not a forecast of the private score.

5. **Use the compute for independent evidence.** Run the Phase 1-based
   sentence encoder on both GPUs with DDP and activation checkpointing.
   The two-way inner cross-fits and five outer fits are the main compute cost;
   target roughly 5-8 hours rather than adding late overfit epochs. Extract
   and cache features on both GPUs. The global head is small; train
   multiple regularized seeds or folds in parallel across the two GPUs rather
   than spending the remaining hours overfitting a single head. Save the
   encoder, optimizer, feature cache, global head, OOF probabilities, logs,
   and a verified recovery ZIP. If the fold evidence passes, train once on
   all 80 training documents using the epoch schedule fixed from the folds,
   then create a *new* validated `prediction.json` ZIP. Report exactly how
   many test labels differ from slot 06.

This direction matches published work on the same class of legal rhetorical
role data: the [13-role corpus and label-shift model](https://aclanthology.org/2022.nllp-1.13/)
and [joint span segmentation with a semi-Markov CRF](https://arxiv.org/abs/2302.06448).
Their architectures are a guide; their pretrained weights and external data
must **not** be imported because the CS781 Phase 3 rules require starting
from the user's Phase 1 model.

**Score expectation:** 0.8 private macro F1 cannot be promised from these
results. The present valid held-out score is 0.466. The next run should first
demonstrate a substantial multi-fold gain; otherwise another packaged ZIP
would repeat the same failure without evidence of improvement.
