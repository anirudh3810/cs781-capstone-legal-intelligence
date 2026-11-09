"""Build a self-contained Kaggle notebook for the third span experiment."""

from __future__ import annotations

import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parents[1] / 'CS781_Phase2_Span_V3_RunAll.ipynb'


def split_sections(path):
    sections = []
    for fragment in path.read_text(encoding='utf-8').split('# %% ')[1:]:
        _, separator, content = fragment.partition('\n')
        if not separator:
            raise RuntimeError(f'Malformed section marker: {path}')
        sections.append(content)
    return sections


old = split_sections(HERE / 'span_runall.py')[:5]
runtime = old[0]
for source, target in [
    ("ROOT = WORKING / 'CS781_Capstone' / 'phase2_span_v1'",
     "ROOT = WORKING / 'CS781_Capstone' / 'phase2_span_v3'\nV1_ROOT = WORKING / 'CS781_Capstone' / 'phase2_span_v1'\nV2_ROOT = WORKING / 'CS781_Capstone' / 'phase2_span_v2'"),
    ("REPORT_ZIP = WORKING / 'CS781_Phase2_Span_Extractor_Report.zip'",
     "REPORT_ZIP = WORKING / 'CS781_Phase2_Span_V3_Report.zip'"),
    ("VERSION = 'span-start-end-pair-v1'", "VERSION = 'span-top12-biaffine-rolehybrid-v3'"),
    ("SESSION_GUARD_SECONDS = 31_200", "SESSION_GUARD_SECONDS = 17_100"),
    ("MAX_EPOCHS = 24", "MAX_EPOCHS = 24"),
    ("MIN_EPOCHS = 8", "MIN_EPOCHS = 12"),
    ("PATIENCE = 5", "PATIENCE = 7"),
    ("VALIDATE_EVERY = 2", "VALIDATE_EVERY = 2"),
    ("MIN_FINAL_CV_F1 = 0.34", "MIN_FINAL_CV_F1 = 0.455"),
]:
    if source not in runtime:
        raise RuntimeError(f'Missing old source fragment: {source}')
    runtime = runtime.replace(source, target)
runtime += "\ntry:\n    from sklearn.ensemble import HistGradientBoostingClassifier\nexcept ImportError as exc:\n    raise RuntimeError('This Kaggle session requires standard scikit-learn') from exc\n"
old[0] = runtime

config = old[1]
config = config.replace("'lora_lr': 1.5e-4, 'head_lr': 4e-4",
                        "'lora_lr': 2e-5, 'head_lr': 5e-5")
config = config.replace("'pair_hidden': 128",
                        "'pair_hidden': 128, 'refine_hidden': 384, 'unfreeze_top_blocks': 4,\n    'v3_unfreeze_top_blocks': 12, 'v3_residual_hidden': 768")
config = config.replace("'min_final_cv_f1': MIN_FINAL_CV_F1,",
                        "'min_final_cv_f1': MIN_FINAL_CV_F1,\n    'warm_start': 'verified phase2_span_v2 fold and final adapters',")
old[1] = config

# Reuse the v2 architecture and candidate-feature functions, embedded directly.
v2_text = (HERE / 'span_v2_stages.py').read_text(encoding='utf-8')
v2_ast = ast.parse(v2_text)
reuse_names = {
    'EnhancedSpanExtractor', 'combine_candidate_scores', 'document_features',
    'build_feature_table', 'train_reranker', 'rerank_documents', 'fast_macro', 'calibrate',
}
reused = []
for node in v2_ast.body:
    if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in reuse_names:
        reused.append(ast.get_source_segment(v2_text, node))
if len(reused) != len(reuse_names):
    raise RuntimeError('One or more v2 components could not be embedded')

fresh = split_sections(HERE / 'span_v3_stages.py')
if len(fresh) != 5:
    raise RuntimeError(f'Expected five v3 sections, got {len(fresh)}')
fresh[0] = '\n\n'.join(reused) + '\n\n' + fresh[0]

names = [
    'Runtime and saved artifacts', 'Official data and Phase-1 base',
    'Token alignment and windows', 'Span model and loss',
    'Two-GPU inference and exact F1', 'V2 provenance and larger v3 model',
    'Resumable 12-layer continuation', 'Cross-validated source selection',
    'Final model and test candidates', 'Verified package and report',
]

intro = """# CS781 Phase 2 — stronger span model, Run All

Run this **in the same Kaggle session** that completed `CS781_Phase2_Span_V2_RunAll.ipynb`. Select the **two-T4 accelerator**, then **Run All**. It reads the verified Phase-1 base-03 checkpoint plus the completed v2 fold/final checkpoints and candidate caches. It creates a new `phase2_span_v3` directory; older experiments and the 0.5069 submission remain untouched. No old notebook variables are required.

The neural stage starts at each selected v2 checkpoint, unfreezes the top **12 of 24** encoder blocks, and adds a zero-initialized bilinear span scorer plus a larger boundary MLP. The initialization reproduces v2 predictions before training. Exact-span hard-negative ranking supplements the previous token and pair losses. Both T4s work on each training batch. The 24-epoch cap, early stopping, epoch checkpoints, and four-hour-45-minute guard make this a long but recoverable run. **Run All again in the same session if it pauses.**

The selection stage compares v2, v3, document-aware reranking, and a label-aware mixture on the same three held-out document folds. The mixture uses each model only where its out-of-fold label F1 supports it, and all candidates go through joint overlap decoding. The report explicitly notes selection bias. If the selected local F1 does not improve enough, no new submission is packaged.

After Run All, download `CS781_Phase2_Span_V3_Report.zip` and send it for inspection. If status is `READY_FOR_MANUAL_UPLOAD`, download `CS781_Phase2_Span_V3_Submission.zip`; also save `CS781_Phase2_Span_V3_Recovery.zip` if produced. This experiment uses only your Phase-1 weights and provided Phase-2 labeled data. AI assisted work should be disclosed according to your course rules. **The private F1, including the 0.6 target, cannot be guaranteed before evaluation.**
"""


def markdown(source):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': source.splitlines(True)}


def code(source):
    ast.parse(source)
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {},
            'outputs': [], 'source': source.splitlines(True)}


cells = [markdown(intro)]
for name, source in zip(names, old + fresh):
    cells += [markdown('## ' + name + '\n'), code(source)]
for i, cell in enumerate(cells):
    cell['id'] = f'cs781-span-v3-{i:03d}'
notebook = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.13'},
        'cs781': {'ai_assisted': True, 'phase': 2,
                  'experiment': 'base03-top12-biaffine-structured-reranker'},
    },
    'nbformat': 4, 'nbformat_minor': 5,
}
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print(OUTPUT)
