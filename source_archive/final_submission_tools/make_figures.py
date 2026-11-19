"""Vector figures derived from saved aggregate evidence, without legal case text."""
from pathlib import Path
import json
import sys

BASE = Path(__file__).resolve().parent
PACKAGE = BASE.parent if BASE.name == 'report' else BASE / 'package'
TOOL_ROOT = BASE.parent.parent if BASE.name == 'report' else BASE
sys.path.insert(0, str(TOOL_ROOT / 'tools/python_lib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = PACKAGE / 'report/figures'
OUT.mkdir(parents=True, exist_ok=True)
data = json.loads((PACKAGE / 'code/evidence/verified_local_results.json').read_text(encoding='utf-8'))
curves = json.loads((PACKAGE / 'code/evidence/learning_curves.json').read_text())
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False,
                    'axes.spines.right': False, 'pdf.fonttype': 42, 'figure.dpi': 170})
colors = ['#173f5f', '#2d8190', '#b66d25', '#79558b', '#66843d', '#b34c50']

fig, ax = plt.subplots(1, 2, figsize=(6.45, 2.45), gridspec_kw={'width_ratios': [1.5, 1]})
counts = data['phase3']['audit']['label_sentences']
items = sorted(counts.items(), key=lambda p: p[1])
short = {'ArgumentPetitioner': 'Petitioner argument', 'ArgumentRespondent': 'Respondent argument',
         'RatioOfTheDecision': 'Ratio', 'RulingByPresentCourt': 'Present-court ruling',
         'RulingByLowerCourt': 'Lower-court ruling', 'PrecedentReliedUpon': 'Relied precedent',
         'PrecedentNotReliedUpon': 'Unrelied precedent', 'PrecedentOverruled': 'Overruled precedent'}
ax[0].barh([short.get(k, k) for k, _ in items], [v for _, v in items], color=colors[0], height=.65)
ax[0].set_xscale('log'); ax[0].set_xlabel('Training sentences (log scale)')
ax[0].set_title('(a) Phase 3: label support', loc='left', fontsize=9)
ax[0].tick_params(axis='y', labelsize=7)
hist = data['phase7']['audit']['train']['gold_counts']
bins = [(0, 0), (1, 2), (3, 7), (8, 15), (16, 1000)]
values = [sum(v for k, v in hist.items() if a <= int(k) <= b) for a, b in bins]
ax[1].bar(['0', '1-2', '3-7', '8-15', '16+'], values, color=colors[1], width=.7)
ax[1].set_xlabel('Relevant cases per query'); ax[1].set_ylabel('Training queries')
ax[1].set_title('(b) Phase 7: relevance counts', loc='left', fontsize=9)
for i, v in enumerate(values): ax[1].text(i, v+4, str(v), ha='center', fontsize=7)
ax[1].set_ylim(0, max(values)*1.18)
fig.tight_layout(w_pad=2.8)
fig.savefig(OUT / 'data_distributions.pdf', bbox_inches='tight')
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(6.45, 2.25))
for i, (slot, rows) in enumerate(curves['phase1_validation'].items()):
    axes[0].plot([r['tokens_processed']/1e6 for r in rows], [r['validation_loss'] for r in rows], color=colors[i], label=f'B{slot}')
axes[0].set(xlabel='Processed tokens (millions)', ylabel='Validation cross-entropy', title='(a) Pretraining')
axes[0].legend(fontsize=6, frameon=False); axes[0].set_ylim(2.4, 5.5)
for i, slot in enumerate(('01', '02', '06')):
    rows = curves['phase6_tune'][slot]
    axes[1].plot([r['epoch'] for r in rows], [r['f1'] for r in rows], '.-', label=slot, color=colors[i])
axes[1].set(xlabel='Epoch', ylabel='Tuning BERTScore F1', title='(b) Phase 6')
axes[1].legend(title='Slot', fontsize=6, title_fontsize=6, frameon=False)
for i, slot in enumerate(('01', '05', '06', 'control')):
    rows = curves['phase7_tune'][slot]
    axes[2].plot([r['epoch'] for r in rows], [r['f1'] for r in rows], '.-', label=slot, color=colors[i])
axes[2].set(xlabel='Evaluated pass', ylabel='Tuning micro F1@7', title='(c) Phase 7')
axes[2].legend(fontsize=6, frameon=False)
for a in axes:
    a.grid(axis='y', alpha=.2); a.tick_params(labelsize=7)
fig.tight_layout(w_pad=1.3)
fig.savefig(OUT / 'learning_curves.pdf', bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.45, 2.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 3.8); ax.axis('off')
def box(x, y, w, h, text, face='#edf3f8', edge='#173f5f', fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.05', facecolor=face, edgecolor=edge, linewidth=.8))
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs)
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=9, linewidth=.8, color='#41606c'))
box(.1, 1.3, 1.6, 1.25, 'Official legal\ncorpus only\n+ byte BPE')
box(2.2, .3, 2.35, 3.2, 'Four scratch-trained bases\n\n01: causal 82.0M\n02: enc-dec 61.3M\n03: prefix hybrid 52.2M\n04: local/global 55.1M', fs=8)
arrow(1.75, 1.9, 2.15, 1.9)
labels = ['P2: exact entity spans', 'P3: sentence roles', 'P4: judgment outcome',
          'P5: statute ID sets', 'P6: summary text', 'P7: seven prior cases']
for i, text in enumerate(labels):
    y = 3.2-i*.57
    box(5.35, y, 2.55, .4, text, '#eff7f4', '#2d8190', 7.8)
    arrow(4.6, 1.9, 5.3, y+.2)
box(8.55, 1.1, 1.3, 1.6, 'Per-task\nvalidation\n+ prediction\npackaging', '#fff6e9', '#b66d25', 8)
arrow(7.95, 1.9, 8.5, 1.9)
fig.tight_layout(pad=.1)
fig.savefig(OUT / 'architecture.pdf', bbox_inches='tight')
plt.close(fig)
print('Created three vector figures from saved evidence.')
