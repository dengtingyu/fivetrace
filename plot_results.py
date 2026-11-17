"""Recreate the V3 figure from public aggregates; no source data required."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path(__file__).parent/'results/v3_external.json')
    parser.add_argument('--output', type=Path, default=Path('benchmark_v3.svg'))
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np
    matplotlib.rcParams['svg.fonttype'] = 'none'
    result = json.loads(args.input.read_text(encoding='utf-8')); order = result['models']; cc = result['primary']
    names = ['FiveTrace V3', 'Desklib v1.01 (2025)', 'Tropa-mini (2026)', 'ConPara (AACL 2026)', 'DetectRL-X XLM-R (2026)', 'MAGE author (ACL 2024)']
    data = np.array([[c['models'][m]['macro_F1'] for c in cc] for m in order])
    fig, ax = plt.subplots(figsize=(18.5, 6.3), dpi=160); fig.patch.set_facecolor('#f7f9fc'); ax.set_facecolor('white')
    ax.imshow(data, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    labels = ['TextSight', 'AI-Detector\nBenchmark', 'ToolBenchmarks\nraw', 'LLaDA\nabstract', 'LLaDA\nrephrase', 'LLaMA\nabstract', 'COLING 2025\ntest slice', 'Intel\npeer review']
    labels = [label+f"\nH/AI {c['models']['v3']['human']}/{c['models']['v3']['ai']}" for label, c in zip(labels, cc)]
    ax.set_xticks(range(len(cc)), labels, fontsize=11); ax.xaxis.tick_top(); ax.set_yticks(range(len(order)), names, fontsize=12)
    ax.get_yticklabels()[0].set_fontweight('bold'); ax.tick_params(axis='both', length=0, pad=10)
    for j in range(len(cc)):
        for i in range(len(order)):
            best = abs(data[i, j]-data[:, j].max()) < 1e-12
            ax.text(j, i, f'{data[i,j]:.4f}', ha='center', va='center', fontsize=13, fontweight='bold' if best or i == 0 else 'normal', color='white' if data[i, j] > .64 else '#17315c')
            if best: ax.add_patch(Rectangle((j-.48, i-.46), .96, .92, fill=False, edgecolor='#ffcc66', linewidth=2.))
    ax.set_xticks(np.arange(-.5, len(cc), 1), minor=True); ax.set_yticks(np.arange(-.5, len(order), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=3); ax.tick_params(which='minor', bottom=False, left=False, top=False)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.suptitle('FiveTrace V3 vs. official open detectors', x=.035, y=.98, ha='left', fontsize=24, fontweight='bold', color='#123154')
    fig.text(.035, .905, 'Macro-F1 ↑ · frozen weights and native thresholds · identical inputs per column · known V3 fitting ancestry excluded', color='#44576e', fontsize=12)
    fig.text(.035, .065, 'Six external sources, eight conditions. Abstract columns share human controls. Gold outlines mark column maxima; paired confidence intervals are in the report.', color='#44576e', fontsize=10)
    fig.text(.035, .03, 'Different original training data and context budgets. COLING FPR uses official negatives, including humanized machine text. Source-familiar; no universal unseen-generator ranking.', color='#44576e', fontsize=10)
    fig.subplots_adjust(left=.205, right=.985, top=.71, bottom=.16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={'Date': '2026-11-17', 'Creator': 'FiveTrace aggregate benchmark renderer'})
    plt.close(fig)


if __name__ == '__main__': main()
