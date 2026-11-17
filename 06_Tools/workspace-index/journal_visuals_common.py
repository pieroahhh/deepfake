"""Journal figure/table export utilities. Numerical inputs must be our own results."""

# Workspace relocation: portable paths, historical metadata stays immutable.
import sys as _fk_sys
from pathlib import Path as _FkBootstrapPath
_fk_workspace = next((p for p in _FkBootstrapPath(__file__).resolve().parents
                      if (p / 'workspace-layout.json').is_file()), None)
if _fk_workspace is None:
    _fk_hint = _FkBootstrapPath(__import__('os').environ.get('FK_WORKSPACE_ROOT', '/inputs'))
    if (_fk_hint / 'workspace-layout.json').is_file():
        _fk_workspace = _fk_hint
if _fk_workspace is None:
    raise RuntimeError('workspace-layout.json is missing')
if str(_fk_workspace / '06_Tools/workspace') not in _fk_sys.path:
    _fk_sys.path.insert(0, str(_fk_workspace / '06_Tools/workspace'))
import workspace_paths as _fk_paths

import csv
import hashlib
import html
import io
import json
import math
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = _fk_paths.workspace_root(__file__)
REPO = _fk_paths.relocate((ROOT / 'Deepfake-Detection'), __file__)
OUT = _fk_paths.relocate((REPO / 'runs/paper-style-visuals-001'), __file__)
NAMES = ['FK', 'AVFakeNet adaptation', 'AVTENet adaptation (0.5)']
CAL = 'AVTENet calibrated (exploratory)'
COLORS = {NAMES[0]: '#1f77b4', NAMES[1]: '#2ca02c', NAMES[2]: '#d62728', CAL: '#9467bd'}
MARKERS = {NAMES[0]: 'o', NAMES[1]: 's', NAMES[2]: '^', CAL: 'D'}
SHORT = {NAMES[0]: 'FK', NAMES[1]: 'AVFakeNet\n(adapt.)', NAMES[2]: 'AVTENet\n(adapt.)', CAL: 'AVTENet\ncalibrated*'}
LEGEND = {NAMES[0]: 'FK', NAMES[1]: 'AVFakeNet adapt.', NAMES[2]: 'AVTENet adapt.', CAL: 'AVTENet calibrated*'}
TYPES = ['real', 'audio_modified', 'visual_modified', 'both_modified']
TYPE_LABELS = ['Real', 'Audio fake', 'Visual fake', 'Both fake']


def sha(path):
    h = hashlib.sha256()
    with _fk_paths.relocate((Path(path)), __file__).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    _fk_paths.relocate((Path(path)), __file__).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def divide(a, b):
    return a / b if b else None


def pct(x):
    return 'N/A' if x is None else '%.2f' % (100 * x)


def decimal(x, digits=4):
    if x is None:
        return 'N/A'
    if 0 < abs(x) < 10 ** -digits:
        return '<' + ('%.*f' % (digits, 10 ** -digits))
    return '%.*f' % (digits, x)


def stats(rows):
    rows = list(rows)
    seen = set()
    tp = tn = fp = fn = 0
    labels, scores = [], []
    continuous = True
    for row in rows:
        if row['video_id'] in seen:
            raise ValueError('Duplicate result ID')
        seen.add(row['video_id'])
        label, pred = row['label'], row['prediction']
        if type(label) is not int or label not in (0, 1) or pred not in (0, 1, None):
            raise ValueError('Invalid binary labels')
        if pred is None:
            continue
        tp += int(label == 1 and pred == 1)
        tn += int(label == 0 and pred == 0)
        fp += int(label == 0 and pred == 1)
        fn += int(label == 1 and pred == 0)
        labels.append(label)
        if row.get('score') is None:
            continuous = False
        else:
            scores.append(row['score'])
    n = tp + tn + fp + fn
    recall, specificity = divide(tp, tp + fn), divide(tn, tn + fp)
    auc = None
    if continuous and len(scores) == n and set(labels) == {0, 1}:
        auc = float(roc_auc_score(labels, scores))
    return {'attempted': len(rows), 'videos': len(rows), 'assessed': n, 'unassessed': len(rows) - n,
            'coverage': divide(n, len(rows)), 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'accuracy': divide(tp + tn, n), 'precision': divide(tp, tp + fp), 'recall': recall,
            'sensitivity': recall, 'specificity': specificity,
            'balanced_accuracy': None if recall is None or specificity is None else (recall + specificity) / 2,
            'f1': divide(2 * tp, 2 * tp + fp + fn), 'roc_auc': auc}


def assessed_ids(rows):
    return {r['video_id'] for r in rows if r['prediction'] is not None}


class Sources:
    def __init__(self):
        self.files = {}
        self.groups = {}
        self.cache = {}

    def register(self, path, group):
        path = _fk_paths.relocate((Path(path)), __file__).resolve()
        allowed = [_fk_paths.relocate((REPO / 'runs'), __file__), _fk_paths.relocate((ROOT / 'avh-a/code/runs'), __file__), _fk_paths.relocate((ROOT / 'Heldout-1000/population'), __file__)]
        if path != _fk_paths.relocate((ROOT / 'fk-heldout-copy.json'), __file__) and not any(p.resolve() in path.parents for p in allowed):
            raise ValueError('Numerical source is outside our retained experiment data: ' + str(path))
        relative = path.relative_to(ROOT).as_posix()
        if relative not in self.files:
            self.files[relative] = {'sha256': sha(path), 'bytes': path.stat().st_size, 'origin': 'own_experiment'}
        self.groups.setdefault(group, set()).add(relative)
        return path, relative

    def json(self, path, group):
        path, key = self.register(path, group)
        if key not in self.cache:
            self.cache[key] = json.loads(path.read_text(encoding='utf-8'))
        return self.cache[key]

    def jsonl(self, path, group):
        path, key = self.register(path, group)
        if key not in self.cache:
            self.cache[key] = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        return self.cache[key]

    def csv(self, path, group):
        path, key = self.register(path, group)
        return list(csv.DictReader(io.StringIO(path.read_text(encoding='utf-8-sig'))))


def setup_style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'legend.fontsize': 7,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': .65,
        'lines.linewidth': 1.1, 'lines.markersize': 3.7, 'grid.linewidth': .35,
        'grid.alpha': .45, 'grid.color': '#bcbcbc', 'figure.facecolor': 'white',
        'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'svg.fonttype': 'none',
        'mathtext.fontset': 'stix', 'axes.unicode_minus': False,
    })


def axes_grid(ax, percent=False):
    ax.grid(axis='y')
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    if percent:
        ax.set_ylim(0, 105)


def metric_bars(ax, labels, values, colors, ylabel, title='', percent=True):
    values = [np.nan if v is None else v * (100 if percent else 1) for v in values]
    bars = ax.bar(np.arange(len(labels)), values, color=colors, edgecolor='black', linewidth=.35, width=.62)
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    axes_grid(ax, percent)
    for b, v in zip(bars, values):
        if np.isfinite(v):
            ax.annotate(('%.1f' if percent else '%.2f') % v, (b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 3), textcoords='offset points', ha='center', fontsize=6.6)
    return bars


def roman(n):
    out = ''
    for value, label in [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), (90, 'XC'),
                         (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]:
        while n >= value:
            out += label
            n -= value
    return out


def latex_escape(value):
    text = str(value)
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
                    '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(c, c) for c in text).replace('\n', ' ')


class Pack:
    def __init__(self, output, sources):
        self.root, self.sources = _fk_paths.relocate((Path(output)), __file__), sources
        for child in ('figures', 'tables', 'data', 'metadata'):
            (self.root / child).mkdir(parents=True, exist_ok=True)
        self.items = []

    def data(self, stem, rows):
        rows = list(rows)
        keys = list(dict.fromkeys(k for row in rows for k in row))
        if not keys:
            raise ValueError('A visualization must expose its own plotted data')
        path = _fk_paths.relocate((self.root / 'data'), __file__) / (stem + '.csv')
        with path.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        return str(path.relative_to(self.root)).replace('\\', '/')

    def figure(self, slug, title, caption, fig, data, groups, section='Main comparisons'):
        number = 1 + sum(x['kind'] == 'figure' for x in self.items)
        stem = 'fig%02d-%s' % (number, slug)
        fig.tight_layout(pad=.8)
        for suffix in ('png', 'svg'):
            fig.savefig(_fk_paths.relocate((self.root / 'figures'), __file__) / (stem + '.' + suffix), dpi=600, bbox_inches='tight', pad_inches=.04)
        plt.close(fig)
        item = {'kind': 'figure', 'number': number, 'label': 'Fig. %d' % number, 'stem': stem,
                'title': title, 'caption': caption, 'section': section, 'source_groups': groups,
                'png': 'figures/' + stem + '.png', 'svg': 'figures/' + stem + '.svg',
                'data': self.data(stem, data), 'data_origin': 'own_experiment_only'}
        write_json(_fk_paths.relocate((self.root / 'metadata'), __file__) / (stem + '.json'), item)
        self.items.append(item)
        print(item['label'] + ' ' + title, flush=True)

    def table(self, slug, title, headers, rows, caption, groups, section='Main comparisons', grid=False, widths=None):
        number = 1 + sum(x['kind'] == 'table' for x in self.items)
        stem = 'table%02d-%s' % (number, slug)
        rows = [[str(x) for x in row] for row in rows]
        widths = np.asarray(widths if widths is not None else [1.] * len(headers), dtype=float)
        widths /= widths.sum()
        width = 7.16 if len(headers) >= 5 else 5.2
        chars = [max(7, int(width * w * 14)) for w in widths]
        wrap = lambda text, i: '\n'.join(textwrap.wrap(str(text), width=chars[i], break_long_words=False, break_on_hyphens=False))
        wrapped_headers = [wrap(x, i) for i, x in enumerate(headers)]
        wrapped = [[wrap(x, i) for i, x in enumerate(row)] for row in rows]
        line_counts = [max(x.count('\n') + 1 for x in wrapped_headers)] + [max(x.count('\n') + 1 for x in row) for row in wrapped]
        height = .62 + .16 * sum(line_counts) + .09 * len(line_counts)
        fig, ax = plt.subplots(figsize=(width, height))
        ax.axis('off')
        fig.text(.5, 1 - .07 / height, 'TABLE ' + roman(number), ha='center', va='top', fontsize=8, color='#006d9c', weight='bold')
        fig.text(.5, 1 - .25 / height, title, ha='center', va='top', fontsize=8.5)
        tbl = ax.table(cellText=wrapped, colLabels=wrapped_headers, colWidths=widths, loc='center', cellLoc='center', bbox=[0, 0, 1, .98])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.4)
        total_lines = sum(line_counts)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_height((line_counts[r] + .45) / (total_lines + .45 * len(line_counts)))
            cell.PAD = .04
            cell.set_facecolor('white')
            cell.set_edgecolor('black')
            cell.set_linewidth(.35 if grid else .45)
            cell.visible_edges = 'closed' if grid else ('TB' if r == 0 else ('B' if r == len(rows) else ''))
            if r == 0:
                cell.get_text().set_weight('bold')
        fig.subplots_adjust(left=.015, right=.985, top=(height - .48) / height, bottom=.015)
        for suffix in ('png', 'svg'):
            fig.savefig(_fk_paths.relocate((self.root / 'tables'), __file__) / (stem + '.' + suffix), dpi=600, bbox_inches='tight', pad_inches=.04)
        plt.close(fig)
        csv_path = _fk_paths.relocate((self.root / 'tables'), __file__) / (stem + '.csv')
        with csv_path.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(rows)
        columns = ''.join(r'>{\raggedright\arraybackslash}p{\dimexpr%.5f\textwidth-2\tabcolsep\relax}' % w for w in widths)
        tex = ['% Own experimental data only. Requires booktabs and array.', r'\begin{table*}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{3pt}',
               r'\caption{' + latex_escape(title) + '}', r'\label{tab:' + slug + '}',
               r'\begin{tabular}{' + columns + '}', r'\toprule',
               ' & '.join(latex_escape(x) for x in headers) + r' \\', r'\midrule']
        tex += [' & '.join(latex_escape(x) for x in row) + r' \\' for row in rows]
        tex += [r'\bottomrule', r'\end{tabular}', r'\par\vspace{2pt}', r'\parbox{\textwidth}{\footnotesize ' + latex_escape(caption) + '}', r'\end{table*}', '']
        (_fk_paths.relocate((self.root / 'tables'), __file__) / (stem + '.tex')).write_text('\n'.join(tex), encoding='utf-8')
        item = {'kind': 'table', 'number': number, 'label': 'Table ' + roman(number), 'stem': stem,
                'title': title, 'caption': caption, 'section': section, 'source_groups': groups,
                'png': 'tables/' + stem + '.png', 'svg': 'tables/' + stem + '.svg',
                'data': 'tables/' + stem + '.csv', 'latex': 'tables/' + stem + '.tex', 'data_origin': 'own_experiment_only'}
        write_json(_fk_paths.relocate((self.root / 'metadata'), __file__) / (stem + '.json'), item)
        self.items.append(item)
        print(item['label'] + ' ' + title, flush=True)
