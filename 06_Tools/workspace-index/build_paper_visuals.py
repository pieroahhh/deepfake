"""Replot only our retained experiments in a compact journal figure/table style."""

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

import argparse
import collections
import html
import json
import math
import os
import shutil
import statistics
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = _fk_paths.workspace_root(__file__)
REPO = _fk_paths.relocate((ROOT / 'Deepfake-Detection'), __file__)
os.environ['MPLCONFIGDIR'] = str(_fk_paths.relocate((REPO / '.cache/matplotlib'), __file__))
sys.path.insert(0, str(_fk_paths.relocate((Path(__file__)), __file__).parent))
from journal_visuals_common import *
from sklearn.metrics import precision_recall_curve, roc_curve


class Registry:
    def __init__(self):
        self.entries, self.groups, self.cache = {}, {}, {}

    def register(self, path, group):
        path = _fk_paths.relocate((Path(path)), __file__).resolve()
        allowed = [_fk_paths.relocate((ROOT / 'Deepfake-Detection/runs'), __file__), _fk_paths.relocate((ROOT / 'avh-a/code/runs'), __file__), _fk_paths.relocate((ROOT / 'Heldout-1000/population'), __file__)]
        if path != _fk_paths.relocate((ROOT / 'fk-heldout-copy.json'), __file__) and not any(path.is_relative_to(p) for p in allowed):
            raise ValueError('Rejected non-experiment numerical input: ' + str(path))
        key = path.relative_to(ROOT).as_posix()
        current = sha(path)
        if key in self.entries and self.entries[key]['sha256'] != current:
            raise ValueError('A retained source changed during the build')
        self.entries[key] = {'sha256': current, 'bytes': path.stat().st_size, 'role': 'own_experimental_data'}
        self.groups.setdefault(group, set()).add(key)
        return path, key

    def json(self, path, group):
        path, key = self.register(path, group)
        if key not in self.cache:
            self.cache[key] = json.loads(path.read_text(encoding='utf-8'))
        return self.cache[key]

    def jsonl(self, path, group):
        path, key = self.register(path, group)
        if key not in self.cache:
            self.cache[key] = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
        return self.cache[key]


class Study:
    def __init__(self):
        self.sources = Registry()
        self.pack = Pack(OUT, self.sources)
        self.manifest = {r['video_id']: r for r in self.sources.json(_fk_paths.relocate((ROOT / 'Heldout-1000/population/manifest.json'), __file__), 'population')}
        self.fk = self.sources.json(_fk_paths.relocate((ROOT / 'fk-heldout-copy.json'), __file__), 'main_predictions')
        self.comparison = self.sources.json(_fk_paths.relocate((REPO / 'runs/three-model-heldout-002/comparison.json'), __file__), 'main_comparison')
        self.models = {
            NAMES[0]: self.fk['videos'],
            NAMES[1]: self.sources.jsonl(_fk_paths.relocate((REPO / 'runs/avfakenet-heldout-002/predictions.jsonl'), __file__), 'main_predictions'),
            NAMES[2]: self.sources.jsonl(_fk_paths.relocate((ROOT / 'avh-a/code/runs/auto-next-001/heldout/evaluate/predictions.jsonl'), __file__), 'main_predictions'),
            CAL: self.sources.jsonl(_fk_paths.relocate((REPO / 'runs/avtenet-analysis-001/heldout-calibrated-predictions.jsonl'), __file__), 'calibrated_supplement'),
        }
        expected_keys = {NAMES[0]: 'FK', NAMES[1]: NAMES[1], NAMES[2]: NAMES[2], CAL: 'AVTENet adaptation (calibrated, exploratory)'}
        for name, rows in self.models.items():
            if len(rows) != 1000 or {r['video_id'] for r in rows} != set(self.manifest):
                raise ValueError('Main prediction population mismatch')
            if any(r['label'] != self.manifest[r['video_id']]['label'] for r in rows):
                raise ValueError('Main prediction label mismatch')
            result = stats(rows)
            expected = self.comparison['own_assessed'][expected_keys[name]]
            for key in ('accuracy', 'balanced_accuracy', 'f1', 'roc_auc', 'coverage'):
                if expected[key] is None:
                    assert result[key] is None
                else:
                    assert abs(result[key] - expected[key]) < 1e-10
        self.lookup = {name: {r['video_id']: r for r in rows} for name, rows in self.models.items()}
        self.common = set.intersection(*[{r['video_id'] for r in self.models[n] if r['prediction'] is not None} for n in NAMES])
        assert len(self.common) == 415
        self.own = {n: stats(v) for n, v in self.models.items()}
        self.paired = {n: stats([r for r in v if r['video_id'] in self.common]) for n, v in self.models.items()}

    def fig(self, *args, **kwargs):
        return self.pack.figure(*args, **kwargs)

    def table(self, *args, **kwargs):
        return self.pack.table(*args, **kwargs)

    def metric_table(self, slug, title, values, caption, groups, section='Main comparisons'):
        rows = []
        for name, m in values.items():
            rows.append([name, '%d/%d' % (m['assessed'], m.get('attempted', m.get('videos', 1000))), pct(m['coverage']),
                         pct(m['accuracy']), pct(m['balanced_accuracy']), pct(m['precision']),
                         pct(m.get('recall', m.get('sensitivity'))), pct(m['specificity']), pct(m['f1']), decimal(m['roc_auc'])])
        self.table(slug, title, ['Method', 'Assessed / attempt', 'Coverage %', 'Accuracy %', 'BA %', 'Precision %', 'Recall %', 'Specificity %', 'F1 %', 'AUC'],
                   rows, caption, groups, section, widths=[1.8, 1.2, 1, 1, 1, 1, 1, 1, 1, .8])

    def main_comparisons(self):
        p = self.pack
        self.table('protocol', 'Local study protocol and retained populations', ['Item', 'Recorded setting', 'Scope'], [
            ['Benchmark', 'AV-Deepfake1M', 'Local held-out subset; not official test'],
            ['Training / validation attempts', '3,000 / 500 clips', 'Frozen shared input manifests'],
            ['FK usable train / validation', '2,912 / 485 clips', 'Original preparation exclusions retained'],
            ['Held-out attempts', '1,000 clips', '250 real; 250 per fake type'],
            ['Three-method common assessed', '415 clips', 'Availability-selected intersection'],
            ['Head/control/ablation training', '5 epochs; seed 7', 'Single retained training seed'],
            ['External adaptations', '1-epoch stages / pilot', 'Unequal budgets; reconstruction limits'],
            ['Profiling', '12 clips; 1 warmup + 3 repeats', '3 metadata-selected clips per type'],
        ], 'All numbers are from our retained local experiments. Official train/validation overlap was retained; held-out exclusion was separately audited. Speaker identity disjointness is not established.', ['population', 'main_comparison'])
        self.metric_table('main-own', 'Main detection results on each method\'s assessed clips', {n: self.own[n] for n in NAMES},
                          'Different assessed denominators. Metrics do not establish a coverage-controlled or author-code superiority claim. AVFakeNet has no continuous fused score; its AUC is N/A.', ['main_predictions', 'main_comparison'])
        self.metric_table('main-common', 'Detection results on 415 common assessed clips', {n: self.paired[n] for n in NAMES},
                          'Identical clip IDs and labels for all methods; selected availability still causes missing-data bias. AVTENet uses its primary 0.5 threshold.', ['main_predictions', 'population'])

        fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.6))
        data = []
        for r, (label, values) in enumerate([('Own assessed', self.own), ('Common n = 415', self.paired)]):
            for c, (key, title) in enumerate([('accuracy', 'Accuracy'), ('balanced_accuracy', 'Balanced accuracy'), ('f1', 'F1 score')]):
                metric_bars(axes[r, c], [SHORT[n] for n in NAMES], [values[n][key] for n in NAMES], [COLORS[n] for n in NAMES], '%', label + '\n' + title)
                data += [{'population': label, 'method': n, 'metric': key, 'value': values[n][key], 'assessed': values[n]['assessed']} for n in NAMES]
        self.fig('detection', 'Detection comparison under explicit assessment populations',
                 'Top: each method\'s assessed clips out of 1,000 attempts (FK 980, AVFakeNet 417, AVTENet 923). Bottom: the same 415 common assessed clips. Baselines are local adaptations with unequal training budgets; AVTENet threshold 0.5. A selected intersection does not remove availability bias.', fig, data, ['main_predictions', 'main_comparison'])

        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        assessed = [self.own[n]['assessed'] for n in NAMES]
        ax.bar(range(3), assessed, color=[COLORS[n] for n in NAMES], label='Assessed', edgecolor='black', linewidth=.35)
        ax.bar(range(3), [1000 - n for n in assessed], bottom=assessed, color='#dddddd', edgecolor='black', linewidth=.35, hatch='///', label='Unassessed')
        ax.set_xticks(range(3), [SHORT[n] for n in NAMES]); ax.set_ylabel('Attempted clips'); ax.set_ylim(0, 1100)
        for i, n in enumerate(assessed): ax.text(i, n / 2, '%d\n(%.1f%%)' % (n, n / 10), ha='center', va='center', color='white', fontsize=8)
        ax.legend(loc='upper center', ncol=2, frameon=False); axes_grid(ax)
        self.fig('coverage', 'Assessment coverage over all 1,000 attempts', 'Unassessed clips retain null predictions; they are not counted as genuine or correct. This coverage figure accompanies the conditional detection metrics.', fig,
                 [{'method': n, 'assessed': self.own[n]['assessed'], 'unassessed': self.own[n]['unassessed'], 'attempted': 1000} for n in NAMES], ['main_predictions'])

        self.confusion(False)
        self.confusion(True)
        self.classification_table()
        self.roc_pr()
        self.calibrated()
        self.score_distributions()
        self.subgroups()
        self.upset(False)
        self.upset(True)
        bootstrap = self.sources.json(_fk_paths.relocate((REPO / 'runs/remaining-offline-001/bootstrap/metrics.json'), __file__), 'external_bootstrap')
        self.forest('external-intervals', 'Paired external-comparison uncertainty', bootstrap,
                    ['AVFakeNet adaptation', 'AVTENet adaptation'], 'FK', '415 common assessed clips; 2,000 paired source-family resamples. Intervals describe evaluation-population sampling, not independent training-seed variability.', ['external_bootstrap'])
        self.branch_diagnostics()

    def confusion(self, full_attempts):
        fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.55), gridspec_kw={'width_ratios': [1, 1, 1, .07]})
        data = []
        for ax, name in zip(axes[:3], NAMES):
            rows = self.models[name] if full_attempts else [r for r in self.models[name] if r['video_id'] in self.common]
            matrix = np.zeros((2, 3 if full_attempts else 2), dtype=int)
            for row in rows:
                col = 2 if row['prediction'] is None else row['prediction']
                matrix[row['label'], col] += 1
            normalized = matrix / matrix.sum(axis=1, keepdims=True)
            im = ax.imshow(normalized, vmin=0, vmax=1, cmap='Blues', aspect='auto')
            for (i, j), value in np.ndenumerate(matrix):
                ax.text(j, i, '%d\n%.1f%%' % (value, normalized[i, j] * 100), ha='center', va='center', fontsize=7, color='white' if normalized[i, j] > .55 else 'black')
                data.append({'method': name, 'true_label': i, 'prediction': 'Unassessed' if j == 2 else j, 'count': int(value), 'row_fraction': float(normalized[i, j])})
            ax.set_xticks(range(matrix.shape[1]), ['Real', 'Fake', 'No\npred.'] if full_attempts else ['Real', 'Fake'])
            ax.set_yticks([0, 1], ['Real', 'Fake']); ax.set_xlabel('Prediction'); ax.set_ylabel('True class'); ax.set_title(LEGEND[name])
        fig.colorbar(im, cax=axes[-1], label='True-class fraction')
        self.fig('confusion-attempts' if full_attempts else 'confusion-common',
                 'Full-attempt confusion with abstentions' if full_attempts else 'Confusion matrices on common assessed clips',
                 ('All 1,000 attempts per method; true-class denominators are 250 genuine and 750 fake. The third column explicitly retains abstentions.' if full_attempts else 'All methods use the same 415 assessed clips. Cell labels show counts and row-normalized percentages; the blue scale is shared.') + ' External labels are 0 = real, 1 = fake.', fig, data, ['main_predictions', 'population'])

    def classification_table(self):
        rows = []
        for name in NAMES:
            m = self.paired[name]
            classes = []
            for label, tp, fp, fn, support in [('Real', m['tn'], m['fn'], m['fp'], m['tn'] + m['fp']), ('Fake', m['tp'], m['fp'], m['fn'], m['tp'] + m['fn'])]:
                pr, rc, f1 = divide(tp, tp + fp), divide(tp, tp + fn), divide(2 * tp, 2 * tp + fp + fn)
                classes.append((pr, rc, f1, support))
                rows.append([name, label, decimal(pr), decimal(rc), decimal(f1), str(support)])
            for avg, weights in [('Macro avg.', [1, 1]), ('Weighted avg.', [c[3] for c in classes])]:
                means = [sum(classes[i][j] * weights[i] for i in range(2)) / sum(weights) for j in range(3)]
                rows.append([name, avg, *[decimal(x) for x in means], '415'])
        self.table('classification-report', 'Binary class-specific classification report (common n = 415)',
                   ['Method', 'Class / average', 'Precision', 'Recall', 'F1', 'Support'], rows,
                   'Binary genuine/fake classification from our predictions only. Manipulation types are evaluation subgroups, not extra classifier classes. Macro and weighted averages are calculated from these two classes.', ['main_predictions', 'population'], grid=True, widths=[1.8, 1.4, .9, .9, .9, .8])

    def roc_pr(self):
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75))
        data = []
        for name in (NAMES[0], NAMES[2]):
            rows = [r for r in self.models[name] if r['video_id'] in self.common]
            labels, scores = [r['label'] for r in rows], [r['score'] for r in rows]
            fpr, tpr, _ = roc_curve(labels, scores)
            precision, recall, _ = precision_recall_curve(labels, scores)
            axes[0].plot(fpr, tpr, color=COLORS[name], marker=MARKERS[name], markevery=max(1, len(fpr) // 9), label=LEGEND[name] + ' (AUC %.3f)' % roc_auc_score(labels, scores))
            axes[1].plot(recall, precision, color=COLORS[name], marker=MARKERS[name], markevery=max(1, len(recall) // 9), label=LEGEND[name])
            data += [{'method': name, 'curve': 'ROC', 'x': float(a), 'y': float(b)} for a, b in zip(fpr, tpr)]
            data += [{'method': name, 'curve': 'PR', 'x': float(a), 'y': float(b)} for a, b in zip(recall, precision)]
        m = self.paired[NAMES[1]]
        axes[0].scatter(1 - m['specificity'], m['recall'], marker='s', color=COLORS[NAMES[1]], s=28, label='AVFakeNet: hard operating point', zorder=5)
        axes[1].scatter(m['recall'], m['precision'], marker='s', color=COLORS[NAMES[1]], s=28, label='AVFakeNet: hard operating point', zorder=5)
        data += [{'method': NAMES[1], 'curve': 'hard ROC point', 'x': 1 - m['specificity'], 'y': m['recall']}, {'method': NAMES[1], 'curve': 'hard PR point', 'x': m['recall'], 'y': m['precision']}]
        axes[0].plot([0, 1], [0, 1], ':', color='gray', linewidth=.7)
        axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='(a) ROC'); axes[1].set(xlabel='Recall', ylabel='Precision', title='(b) Precision-recall')
        for ax in axes:
            ax.set_xlim(-.02, 1.02); ax.set_ylim(-.02, 1.05); ax.grid(); ax.legend(loc='lower right', fontsize=6.1, frameon=False)
        self.fig('roc-pr', 'ROC and precision-recall on common assessed clips',
                 'Curves use retained continuous FK and AVTENet scores on the same 415 clips. AVFakeNet contributes one measured hard-decision operating point only: no continuous fused ROC/PR curve, AUC or average precision is invented.', fig, data, ['main_predictions', 'population'])

    def calibrated(self):
        default = self.lookup[NAMES[2]]; exploratory = self.lookup[CAL]
        assert all(default[k].get('score') == exploratory[k].get('score') for k in default)
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.65))
        keys = [('accuracy', 'Accuracy'), ('balanced_accuracy', 'BA'), ('recall', 'Fake recall'), ('f1', 'F1')]
        x = np.arange(len(keys)); width = .34; data = []
        for j, name in enumerate((NAMES[2], CAL)):
            axes[0].bar(x + (j - .5) * width, [self.own[name][k] * 100 for k, _ in keys], width, color=COLORS[name], hatch='//' if name == CAL else '', edgecolor='black', linewidth=.3, label='Default 0.5' if j == 0 else 'Calibrated, exploratory')
            data += [{'analysis': name, 'metric': k, 'value': self.own[name][k], 'assessed': 923} for k, _ in keys]
        axes[0].set_xticks(x, [v for _, v in keys]); axes[0].set_ylabel('%'); axes[0].set_title('(a) Fixed-score decision results'); axes_grid(axes[0], True); axes[0].legend(frameon=False, fontsize=6.4)
        metric_bars(axes[1], ['Default\n0.5', 'Calibrated*'], [self.own[NAMES[2]]['roc_auc'], self.own[CAL]['roc_auc']], [COLORS[NAMES[2]], COLORS[CAL]], 'AUC (%)', '(b) Ranking remains unchanged')
        self.fig('avtenet-calibration', 'AVTENet threshold analysis (supplementary)',
                 'The calibrated threshold was selected from validation after earlier held-out inspection and is exploratory. The same 923 clips and unchanged raw scores are used. Primary AVTENet remains threshold 0.5; this is not a newly trained model or independent confirmation.', fig, data, ['main_predictions', 'calibrated_supplement'], 'Supplementary analyses')
        values = {'Default, own assessed': self.own[NAMES[2]], 'Calibrated*, own assessed': self.own[CAL],
                  'Default, common': self.paired[NAMES[2]], 'Calibrated*, common': self.paired[CAL]}
        self.metric_table('avtenet-calibrated', 'Supplementary AVTENet threshold comparison', values,
                          'Calibrated analysis is post-inspection exploratory. Scores, weights and assessment population are unchanged; the default row is retained as primary.', ['main_predictions', 'calibrated_supplement'], 'Supplementary analyses')

    def score_distributions(self):
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.7)); data = []
        for ax, name, threshold in zip(axes, (NAMES[0], NAMES[2]), (self.fk['decision']['video_threshold'], .5)):
            for label, color in [(0, '#1f77b4'), (1, '#d62728')]:
                values = [r['score'] for r in self.models[name] if r['prediction'] is not None and r['label'] == label]
                ax.hist(values, bins=np.linspace(0, 1, 41), density=True, histtype='step', color=color, linewidth=1.1, label=('Real' if label == 0 else 'Fake') + ' (n=%d)' % len(values))
                data += [{'method': name, 'video_id': r['video_id'], 'label': label, 'score': r['score']} for r in self.models[name] if r['prediction'] is not None and r['label'] == label]
            ax.axvline(threshold, color='black', linestyle='--', linewidth=.8, label='Frozen threshold')
            ax.set(xlim=(0, 1), xlabel='Retained model score', ylabel='Conditional density', title=LEGEND[name]); ax.legend(frameon=False, fontsize=6.5); axes_grid(ax)
        self.fig('score-distributions', 'Class-conditional retained score distributions', 'Density is normalized within true class on each method\'s own assessed subset; denominators differ. Scores are not assumed to be calibrated probabilities. AVFakeNet hard fusion has no continuous score distribution.', fig, data, ['main_predictions'], 'Supplementary analyses')

    def subgroups(self):
        data, own, paired = [], {}, {}
        for kind in TYPES:
            ids = {k for k, r in self.manifest.items() if r['source_modify_type'] == kind}
            own[kind] = {}; paired[kind] = {}
            for name in NAMES:
                for population, use in [('own', ids), ('common', ids & self.common)]:
                    rows = [self.lookup[name][k] for k in sorted(use)]
                    assessed = [r for r in rows if r['prediction'] is not None]
                    value = divide(sum(r['prediction'] == r['label'] for r in assessed), len(assessed))
                    rec = {'group': kind, 'method': name, 'population': population, 'attempted': len(rows), 'assessed': len(assessed),
                           'coverage': divide(len(assessed), len(rows)), 'correct_rate': value}
                    data.append(rec); (own if population == 'own' else paired)[kind][name] = rec
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8)); x = np.arange(4); width = .24
        for i, name in enumerate(NAMES):
            for ax, key in zip(axes, ('correct_rate', 'coverage')):
                ax.bar(x + (i - 1) * width, [own[k][name][key] * 100 for k in TYPES], width, color=COLORS[name], edgecolor='black', linewidth=.25, label=LEGEND[name])
        for ax, title in zip(axes, ('(a) Real specificity / fake recall', '(b) Assessment coverage')):
            ax.set_xticks(x, TYPE_LABELS, rotation=12); ax.set_ylabel('%'); ax.set_title(title); axes_grid(ax, True)
        axes[1].legend(frameon=False, fontsize=6.3, loc='upper center', bbox_to_anchor=(.5, 1.27), ncol=3)
        self.fig('manipulation-types', 'Manipulation-type results with coverage', '250 attempts per type. Correct rate is specificity for genuine clips and recall for fake-only groups. BA and AUC are undefined within a constant-label subtype. Own-assessed denominators differ; FK visual-only-fake weakness remains visible.', fig, data, ['main_predictions', 'population'])
        for population, title in [('own', 'Manipulation subgroups: own assessed'), ('common', 'Manipulation subgroups: common assessed')]:
            self.table('types-' + population, title, ['Type', 'Method', 'Assessed / attempt', 'Coverage %', 'Specificity / recall %'],
                       [[r['group'], r['method'], '%d/%d' % (r['assessed'], r['attempted']), pct(r['coverage']), pct(r['correct_rate'])] for r in data if r['population'] == population],
                       'Real rows report specificity; fake rows report conditional recall. These are subgroups of a binary detector. The common subset is availability-selected.', ['main_predictions', 'population'], widths=[1.3, 1.7, 1.2, 1, 1.5])
        duration = [('Under 8 s', lambda d: d < 8), ('8 to <16 s', lambda d: 8 <= d < 16), ('16 s or longer', lambda d: d >= 16)]
        data = []
        for title, predicate in duration:
            ids = {k for k, row in self.manifest.items() if predicate(row['duration'])}
            for name in NAMES:
                rows = [self.lookup[name][k] for k in sorted(ids)]; m = stats(rows)
                data.append({'group': title, 'method': name, **m, 'assessed_real': m['tn'] + m['fp'], 'assessed_fake': m['tp'] + m['fn']})
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8)); x = np.arange(3)
        for i, name in enumerate(NAMES):
            values = [r for r in data if r['method'] == name]
            for ax, key in zip(axes, ('accuracy', 'coverage')):
                ax.plot(x, [r[key] * 100 for r in values], color=COLORS[name], marker=MARKERS[name], label=LEGEND[name])
        for ax, title in zip(axes, ('(a) Conditional accuracy', '(b) Assessment coverage')):
            ax.set_xticks(x, [r[0] for r in duration]); ax.set_ylabel('%'); ax.set_title(title); axes_grid(ax, True)
        axes[1].legend(frameon=False, fontsize=6.3)
        self.fig('duration-subgroups', 'Duration-stratified descriptive results', 'Duration bins were defined from metadata, not optimized from outcomes. Class composition and availability differ between bins; accuracy trends do not establish a causal duration effect.', fig, data, ['main_predictions', 'population'], 'Supplementary analyses')
        self.table('duration-subgroups', 'Duration subgroups and class composition', ['Duration', 'Method', 'Assessed / attempt', 'Real / fake assessed', 'Coverage %', 'Accuracy %', 'BA %'],
                   [[r['group'], r['method'], '%d/%d' % (r['assessed'], r['attempted']), '%d / %d' % (r['assessed_real'], r['assessed_fake']), pct(r['coverage']), pct(r['accuracy']), pct(r['balanced_accuracy'])] for r in data],
                   'Own-assessed duration groups have different true-class composition and exclusions. All values derive from existing fixed predictions.', ['main_predictions', 'population'], 'Supplementary analyses', widths=[1.2, 1.7, 1.2, 1.2, 1, 1, 1])

    def upset(self, availability):
        counts = collections.Counter()
        ids = self.manifest if availability else self.common
        for vid in ids:
            pattern = tuple((self.lookup[n][vid]['prediction'] is not None) if availability else (self.lookup[n][vid]['prediction'] != self.manifest[vid]['label']) for n in NAMES)
            counts[pattern] += 1
        patterns = sorted(counts, key=lambda k: (-counts[k], k))
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.16, 3.25), sharex=True, gridspec_kw={'height_ratios': [2.1, 1]})
        values = [counts[k] for k in patterns]; bars = top.bar(range(len(patterns)), values, color='#1f77b4', edgecolor='black', linewidth=.35)
        top.bar_label(bars, fontsize=7, padding=2); top.set_ylabel('Clips'); axes_grid(top); top.set_ylim(0, max(values) * 1.16)
        for x, pattern in enumerate(patterns):
            active = []
            for y, value in enumerate(pattern):
                bottom.scatter(x, y, color='#1f77b4' if value else '#dddddd', s=24, zorder=3)
                if value: active.append(y)
            if len(active) > 1: bottom.plot([x, x], [min(active), max(active)], color='#1f77b4', linewidth=1)
        bottom.set_yticks(range(3), [LEGEND[n] for n in NAMES]); bottom.set_xticks(range(len(patterns)), ['P%d' % (i + 1) for i in range(len(patterns))]); bottom.invert_yaxis(); bottom.set_xlabel('Observed membership pattern'); bottom.spines[['top', 'right', 'bottom']].set_visible(False)
        data = [{'pattern': 'P%d' % (i + 1), 'clips': counts[k], **{n: int(v) for n, v in zip(NAMES, k)}} for i, k in enumerate(patterns)]
        self.fig('assessment-overlap' if availability else 'error-overlap', 'Assessment availability patterns' if availability else 'Error overlap on common assessed clips',
                 ('Filled circles mean assessed; all 1,000 attempts are included. This exposes how the 415-clip common subset is selected.' if availability else 'Filled circles mean an incorrect prediction on the 415 common assessed clips; an empty column means all methods were correct. These descriptive overlaps are not an evaluated ensemble.'), fig, data, ['main_predictions', 'population'], 'Main comparisons' if availability else 'Supplementary analyses')

    def forest(self, slug, title, source, names, reference, caption, groups, section='Supporting experiments'):
        diffs = source['bootstrap']['paired_differences']; fig, axes = plt.subplots(1, 2, figsize=(7.16, max(2.3, .35 * len(names) + 1.2))); data = []
        for ax, metric, metric_title in zip(axes, ('balanced_accuracy', 'f1'), ('Balanced accuracy', 'F1 score')):
            for y, name in enumerate(names):
                d = diffs[name][metric]; point = d['observed_difference'] * 100; lo, hi = [v * 100 for v in d['ci95']]
                ax.errorbar(point, y, xerr=[[point - lo], [hi - point]], fmt='o', color='#1f77b4', capsize=2, linewidth=1)
                data.append({'method': name, 'metric': metric, 'reference': reference, 'difference_pp': point, 'lo95_pp': lo, 'hi95_pp': hi, 'replicates': source['bootstrap']['replicates'], 'common_n': source['bootstrap']['videos']})
            ax.axvline(0, color='black', linestyle='--', linewidth=.65); ax.set_yticks(range(len(names)), [n.replace('_', ' ') for n in names]); ax.invert_yaxis(); ax.set_xlabel('Difference vs ' + reference + ' (percentage points)'); ax.set_title(metric_title); ax.grid(axis='x')
        self.fig(slug, title, caption, fig, data, groups, section)
        table_rows = []
        for name in names:
            fields = []
            for metric in ('balanced_accuracy', 'f1'):
                d = diffs[name][metric]
                fields.append('%+.2f [%+.2f, %+.2f]' % (d['observed_difference'] * 100, d['ci95'][0] * 100, d['ci95'][1] * 100))
            table_rows.append([name, *fields, str(source['bootstrap']['videos']), str(source['bootstrap']['replicates'])])
        self.table(slug, title + ' (percentage points)', ['Variant / method', 'BA difference [95% CI]', 'F1 difference [95% CI]', 'Common n', 'Resamples'], table_rows,
                   caption + ' Reference: ' + reference + '.', groups, section, widths=[1.6, 2, 2, .7, .8])

    def branch_diagnostics(self):
        rows = [r for r in self.models[NAMES[1]] if r['prediction'] is not None and 'audio_probability' in r]
        fig, ax = plt.subplots(figsize=(3.5, 2.9)); data = []
        palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for kind, label, color in zip(TYPES, TYPE_LABELS, palette):
            values = [r for r in rows if self.manifest[r['video_id']]['source_modify_type'] == kind]
            ax.scatter([r['visual_fake_vote_fraction'] for r in values], [r['audio_probability'] for r in values], s=10, alpha=.65, color=color, label=label, edgecolors='none')
            data += [{'video_id': r['video_id'], 'type': kind, 'visual_vote_fraction': r['visual_fake_vote_fraction'], 'audio_probability': r['audio_probability'], 'visual_prediction': r['visual_prediction'], 'audio_prediction': r['audio_prediction']} for r in values]
        ax.axvline(.5, color='gray', linestyle='--', linewidth=.6); ax.axhline(.5, color='gray', linestyle='--', linewidth=.6)
        ax.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02), xlabel='Native visual fake-vote fraction', ylabel='Native audio fake probability'); ax.grid(); ax.legend(frameon=False, fontsize=6.4)
        self.fig('avfakenet-branches', 'Retained AVFakeNet branch diagnostics', 'Exploratory view of the 417 jointly assessed clips. Axes are recorded native branch summaries, not a continuous fused score. The visual majority and hard-OR decisions remain unchanged; subgroup labels come from our manifest.', fig, data, ['main_predictions', 'population'], 'Supplementary analyses')
        branch_rows = []
        for modality, prediction, score in [('visual', 'visual_prediction', 'visual_fake_vote_fraction'), ('audio', 'audio_prediction', 'audio_probability')]:
            values = [{'video_id': r['video_id'], 'label': self.manifest[r['video_id']][modality + '_label'], 'prediction': r[prediction], 'score': r[score]} for r in rows if self.manifest[r['video_id']][modality + '_label'] in (0, 1)]
            m = stats(values)
            branch_rows.append([modality, str(len(values)), str(m['tn'] + m['fp']), str(m['tp'] + m['fn']), pct(m['balanced_accuracy']), pct(m['f1']), decimal(m['roc_auc'])])
        self.table('avfakenet-branches', 'Exploratory native-branch diagnostics', ['Branch', 'Assessed', 'Genuine modality', 'Fake modality', 'BA %', 'F1 %', 'Branch-score AUC'], branch_rows,
                   'Targets are modality-specific labels from our metadata. Visual AUC uses the retained fake-vote fraction; audio AUC uses retained audio probability. Neither is a fused AVFakeNet AUC. This joint-assessed subset is availability-selected.', ['main_predictions', 'population'], 'Supplementary analyses')

    def supporting_models(self):
        controls = self.sources.json(_fk_paths.relocate((REPO / 'runs/shared-controls-001/summary.json'), __file__), 'shared_controls')
        values = {'FK': self.own['FK'], **{k: v['video_metrics'] for k, v in controls['models'].items()}}
        labels = ['FK', 'Visual\nonly', 'Audio\nonly', 'Concate-\nnation', 'Late\nfusion']
        keys = ['FK', 'visual_only', 'audio_only', 'concatenation', 'late_fusion']
        colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
        fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.7)); data = []
        for ax, key, title in zip(axes, ('balanced_accuracy', 'f1', 'roc_auc'), ('Balanced accuracy', 'F1 score', 'ROC AUC')):
            metric_bars(ax, labels, [values[n][key] for n in keys], colors, '%', title)
            data += [{'model': n, 'metric': key, 'value': values[n][key], 'assessed': 980} for n in keys]
        self.fig('shared-controls', 'Shared-encoder controls', 'All controls use the same FK frozen encoder caches and joint assessment population (980/1,000), five epochs and seed 7. Late fusion exceeds FK on these retained metrics; the plot does not assert universal FK superiority.', fig, data, ['shared_controls', 'main_predictions'], 'Supporting experiments')
        self.metric_table('shared-controls', 'Shared-encoder control results', values, 'Same frozen encoder caches, five epochs, seed 7 and 980/1,000 joint-assessed clips. These are classifier controls, not independently prepared unimodal systems.', ['shared_controls', 'main_predictions'], 'Supporting experiments')
        core = self.sources.json(_fk_paths.relocate((REPO / 'runs/remaining-ablation-evaluation-001/comparison/metrics.json'), __file__), 'core_ablations')
        core_names = ['full', 'no_av', 'no_temporal', 'uniform_weights']
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.7)); data = []
        for ax, key, title in zip(axes, ('balanced_accuracy', 'f1'), ('Balanced accuracy', 'F1 score')):
            vals = [core['per_method_assessed'][n][key] for n in core_names]
            metric_bars(ax, ['Full', 'No AV', 'No temporal', 'Uniform'], vals, ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd'], '%', title)
            data += [{'variant': n, 'metric': key, 'value': core['per_method_assessed'][n][key]} for n in core_names]
        self.fig('core-ablations', 'Matched retrained FK core ablations', 'All four variants were retrained for five epochs with seed 7 using the same caches and validation-only calibration. Full is the matched rerun. This is one seed; paired intervals concern test-population sampling.', fig, data, ['core_ablations'], 'Supporting experiments')
        self.metric_table('core-ablations', 'Matched core ablation results', {k: core['per_method_assessed'][k] for k in core_names}, 'Same 980 assessed clips, five epochs and seed 7. Follow-up is exploratory after held-out inspection.', ['core_ablations'], 'Supporting experiments')
        self.forest('core-ablation-intervals', 'Core ablation differences from Full', core, core_names[1:], 'Full',
                    '2,000 paired source-family resamples on 980 common assessed clips. BA intervals for no-AV and no-temporal include zero; these are not independent-seed confidence intervals.', ['core_ablations'])
        agg = self.sources.json(_fk_paths.relocate((REPO / 'runs/remaining-supplement-001/aggregation/comparison/metrics.json'), __file__), 'aggregation')
        names = ['full', 'window_max', 'window_mean', 'video_max', 'video_mean']; labels = ['Full', 'Window\nmax', 'Window\nmean', 'Video\nmax', 'Video\nmean']
        fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.7)); data = []
        for ax, key, title in zip(axes, ('balanced_accuracy', 'f1', 'fp'), ('Balanced accuracy', 'F1 score', 'False positives')):
            metric_bars(ax, labels, [agg['per_method_assessed'][n][key] for n in names], ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd'], '%' if key != 'fp' else 'Clips', title, percent=key != 'fp')
            data += [{'variant': n, 'metric': key, 'value': agg['per_method_assessed'][n][key], 'assessed': 980} for n in names]
        self.fig('aggregation', 'Retrained aggregation variants', 'Five-epoch, seed-7 variants on the same 980 assessed clips. Window-mean increases F1 but also false positives; all BA-difference intervals versus Full include zero. Do not select a new primary variant from these inspected held-out outcomes.', fig, data, ['aggregation'], 'Supporting experiments')
        self.metric_table('aggregation', 'Aggregation variant detection results', {k: agg['per_method_assessed'][k] for k in names}, 'Exploratory retrained variants; identical caches and evaluation population. Full is the matched reference.', ['aggregation'], 'Supporting experiments')
        self.forest('aggregation-intervals', 'Aggregation differences from Full', agg, names[1:], 'Full', '2,000 paired source-family resamples on 980 common assessed clips; single training seed. All retained BA intervals include zero.', ['aggregation'])

    def load_robustness(self):
        maps = {n: {} for n in NAMES}; self.robust_rows = {n: {} for n in NAMES}
        locations = [(NAMES[0], 'robustness-fk'), (NAMES[1], 'robustness-avfakenet'), (NAMES[2], 'avtenet/fusion')]
        base = _fk_paths.relocate((REPO / 'runs/remaining-supplement-002'), __file__)
        for name, folder in locations:
            for path in sorted((_fk_paths.relocate((base / folder), __file__)).glob('*.json')):
                if len(path.stem) != 64:
                    continue
                value = self.sources.json(path, 'robustness_predictions')
                if name == NAMES[2]:
                    if value['item']['purpose'] != 'robustness':
                        continue
                    row, pred = value['item']['row'], value['samples'][0]['prediction']
                else:
                    row, pred = value['row'], value['prediction']
                condition, vid = row['condition'], pred['video_id']
                if (condition, vid) in maps[name]:
                    raise ValueError('Duplicate condition/video robustness result')
                assert pred['label'] == row['label']
                maps[name][condition, vid] = pred
                self.robust_rows[name][condition, vid] = row
        conditions = sorted({k[0] for k in maps[NAMES[0]]})
        for condition in conditions:
            ids = [{vid for cond, vid in maps[name] if cond == condition} for name in NAMES]
            assert all(x == ids[0] for x in ids)
            for vid in ids[0]:
                sources = [self.robust_rows[n][condition, vid]['source_sha256'] for n in NAMES]
                labels = [maps[n][condition, vid]['label'] for n in NAMES]
                assert len(set(sources)) == 1 and len(set(labels)) == 1
        self.robust = maps
        self.robust_conditions = ['clean', 'video_compression-28', 'video_compression-38', 'audio_compression-64', 'audio_compression-32']
        summaries = {NAMES[0]: self.sources.json(_fk_paths.relocate((base / 'robustness-fk/summary.json'), __file__), 'robustness_summaries')['metrics'],
                     NAMES[1]: self.sources.json(_fk_paths.relocate((base / 'robustness-avfakenet/summary.json'), __file__), 'robustness_summaries')['metrics'],
                     NAMES[2]: self.sources.json(_fk_paths.relocate((base / 'avtenet/summary.json'), __file__), 'robustness_summaries')['robustness']}
        for name in NAMES:
            for condition in conditions:
                m = stats(v for (c, _), v in maps[name].items() if c == condition)
                assert m['assessed'] == summaries[name][condition]['assessed']
                if m['balanced_accuracy'] is not None:
                    assert abs(m['balanced_accuracy'] - summaries[name][condition]['balanced_accuracy']) < 1e-12

    def robustness(self):
        self.load_robustness()
        friendly = {'clean': 'Original', 'video_compression-28': 'CRF 28', 'video_compression-38': 'CRF 38', 'audio_compression-64': '64 kbps', 'audio_compression-32': '32 kbps'}
        self.robust_own, self.robust_common = [], []
        for condition in self.robust_conditions:
            all_ids = {vid for c, vid in self.robust[NAMES[0]] if c == condition}
            common = set.intersection(*[{vid for c, vid in self.robust[n] if c == condition and self.robust[n][c, vid]['prediction'] is not None} for n in NAMES])
            for name in NAMES:
                self.robust_own.append({'condition': condition, 'method': name, **stats(self.robust[name][condition, v] for v in sorted(all_ids))})
                self.robust_common.append({'condition': condition, 'method': name, **stats(self.robust[name][condition, v] for v in sorted(common))})
        for slug, selected, title in [('video-compression', self.robust_conditions[:3], 'Video compression'), ('audio-compression', ['clean', 'audio_compression-64', 'audio_compression-32'], 'Audio compression')]:
            fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.7)); data = []
            for name in NAMES:
                vals = [next(r for r in self.robust_own if r['condition'] == c and r['method'] == name) for c in selected]
                for ax, key in zip(axes, ('balanced_accuracy', 'coverage')):
                    ax.plot(range(3), [r[key] * 100 for r in vals], color=COLORS[name], marker=MARKERS[name], label=LEGEND[name])
                data += vals
            for ax, subtitle in zip(axes, ('Conditional balanced accuracy', 'Assessment coverage')):
                ax.set_xticks(range(3), [friendly[c] for c in selected]); ax.set_ylabel('%'); ax.set_title(subtitle); axes_grid(ax, True)
            axes[1].legend(frameon=False, fontsize=6.5)
            self.fig(slug, title + ' robustness with coverage', '100 fixed clips per condition, 25 per type; identical condition-file hashes across methods were checked. Curves use own-assessed clips and changing denominators. Nonmonotonic scores do not prove that stronger compression helps. Points are discrete conditions, not an assumed continuous response.', fig, data, ['robustness_predictions', 'robustness_summaries'], 'Supporting experiments')
        for label, rows, caption in [('own', self.robust_own, 'Own-assessed denominators differ; preserve coverage.'), ('common', self.robust_common, 'The intersection is computed separately per condition; its size can change across conditions.')]:
            self.table('robustness-' + label, 'Compression robustness: ' + label + ' assessed', ['Condition', 'Method', 'Assessed / attempt', 'Coverage %', 'BA %', 'F1 %', 'AUC'],
                       [[friendly[r['condition']], r['method'], '%d/%d' % (r['assessed'], r['attempted']), pct(r['coverage']), pct(r['balanced_accuracy']), pct(r['f1']), decimal(r['roc_auc'])] for r in rows],
                       'Our 100-clip robustness population; ' + caption + ' AVFakeNet fused AUC is N/A. These are exploratory condition results, not full 1,000-clip estimates.', ['robustness_predictions'], 'Supporting experiments', widths=[1.2, 1.8, 1.1, 1, 1, 1, .8])
        fixed = set.intersection(*[{vid for c, vid in self.robust[n] if c == condition and self.robust[n][c, vid]['prediction'] is not None} for n in NAMES for condition in self.robust_conditions])
        fixed_rows = []
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))
        for name in NAMES:
            vals = []
            for c in self.robust_conditions:
                m = stats(self.robust[name][c, vid] for vid in sorted(fixed))
                fixed_rows.append({'condition': c, 'method': name, **m}); vals.append(m)
            for ax, key in zip(axes, ('balanced_accuracy', 'f1')):
                ax.plot(range(5), [np.nan if m[key] is None else m[key] * 100 for m in vals], color=COLORS[name], marker=MARKERS[name], label=LEGEND[name])
        for ax, title in zip(axes, ('Balanced accuracy', 'F1 score')):
            ax.set_xticks(range(5), ['Original', 'CRF28', 'CRF38', '64 kbps', '32 kbps'], rotation=20); ax.set_title(title); ax.set_ylabel('%'); axes_grid(ax, True)
        axes[1].legend(frameon=False, fontsize=6.5)
        self.fig('robustness-fixed-common', 'Robustness on fixed common-assessed IDs', 'The same %d clips are assessed by all three methods in all five compression conditions. This removes denominator drift within this plot but retains availability selection bias and a small sample. No condition-specific threshold fitting.' % len(fixed), fig, fixed_rows, ['robustness_predictions'], 'Supporting experiments')
        self.table('robustness-fixed', 'Compression robustness on a fixed common subset', ['Condition', 'Method', 'Fixed n', 'BA %', 'F1 %', 'AUC'],
                   [[friendly[r['condition']], r['method'], str(r['assessed']), pct(r['balanced_accuracy']), pct(r['f1']), decimal(r['roc_auc'])] for r in fixed_rows],
                   'Exactly %d IDs common to every method and every displayed condition; not an unbiased sample of all 100 attempted clips.' % len(fixed), ['robustness_predictions'], 'Supporting experiments', widths=[1.2, 1.8, .8, 1, 1, 1])
        write_json(_fk_paths.relocate((OUT / 'metadata/robustness-fixed-ids.json'), __file__), {'ids': sorted(fixed), 'conditions': self.robust_conditions, 'scope': 'availability-selected exploratory subset'})
        self.delay()

    def delay(self):
        conditions = ['clean', 'av_offset-0.08', 'av_offset-0.16', 'av_offset-0.32']; values = []
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.7))
        for name in NAMES:
            vals = []
            for condition in conditions:
                rows = [v for (c, vid), v in self.robust[name].items() if c == condition and v['label'] == 0]
                assert len(rows) == 25
                m = stats(rows)
                rec = {'method': name, 'delay_seconds': 0 if condition == 'clean' else float(condition.split('-')[1]), 'attempted': 25,
                       'assessed': m['assessed'], 'false_alarms': m['fp'], 'false_alarm_rate': divide(m['fp'], m['fp'] + m['tn']), 'coverage': m['coverage']}
                values.append(rec); vals.append(rec)
            axes[0].plot([r['delay_seconds'] for r in vals], [r['false_alarm_rate'] * 100 for r in vals], color=COLORS[name], marker=MARKERS[name], label=LEGEND[name])
            axes[1].plot([r['delay_seconds'] for r in vals], [r['coverage'] * 100 for r in vals], color=COLORS[name], marker=MARKERS[name], label=LEGEND[name])
        for ax, title in zip(axes, ('Genuine-clip false alarms', 'Assessment coverage')):
            ax.set(xlabel='Added audio delay (s)', ylabel='%', title=title); ax.set_xticks([0, .08, .16, .32]); axes_grid(ax, True)
        axes[1].legend(frameon=False, fontsize=6.5)
        self.fig('genuine-delay', 'Benign audio-video delay stress', '25 genuine clips per delay; labels remain real. Report false alarms and coverage, not fake recall/F1/BA for this single-class test. FK is sensitive to benign misalignment. AVTENet default low false-alarm rate must be read alongside its low fake recall.', fig, values, ['robustness_predictions'], 'Supporting experiments')
        self.table('genuine-delay', 'False alarms on genuine delayed clips', ['Method', 'Delay s', 'Assessed / 25', 'False alarms', 'False-alarm %', 'Coverage %'],
                   [[r['method'], decimal(r['delay_seconds'], 2), str(r['assessed']) + '/25', str(r['false_alarms']), pct(r['false_alarm_rate']), pct(r['coverage'])] for r in values],
                   'Genuine-only synchronization stress. Delay does not create a fake ground-truth class. Each method retains its frozen primary decision rule.', ['robustness_predictions'], 'Supporting experiments', widths=[1.8, .8, 1.1, 1, 1.2, 1])

    def localization_forensics(self):
        summary = self.sources.json(_fk_paths.relocate((REPO / 'runs/remaining-offline-001/localization/summary.json'), __file__), 'localization')
        timelines = {r['video_id']: r for r in self.fk['timeline_metrics_supplement']}
        keys = ['precision', 'recall', 'f1', 'iou']; labels = ['Precision', 'Recall', 'F1', 'IoU']
        values = [summary['duration_weighted_assessed_region'][k] for k in keys]
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6))
        metric_bars(axes[0], labels, values, ['#1f77b4'] * 4, '%', 'Assessed-region localization')
        coverage = [summary['time_assessment_coverage'], summary['manipulated_time_assessment_coverage'], summary['detected_fraction_of_all_manipulated_time']]
        metric_bars(axes[1], ['All-time\ncoverage', 'Edited-time\ncoverage', 'All edited-time\ndetected'], coverage, ['#2ca02c', '#ff7f0e', '#d62728'], '%', 'Coverage-aware timeline accounting')
        data = [{'metric': k, 'value': v, 'scope': 'assessed_region'} for k, v in zip(keys, values)]
        data += [{'metric': k, 'value': v, 'scope': 'full_timeline'} for k, v in zip(['coverage', 'edited_coverage', 'edited_detected'], coverage)]
        self.fig('localization-summary', 'FK source-time localization and coverage', 'Retained frozen-threshold timeline adapter on 1,000 clips; overlapping bin scores use endpoint-partition means. Duration-weighted metrics. This is supporting FK analysis; native baseline localization is not provided.', fig, data, ['localization', 'main_predictions'], 'Supporting experiments')
        self.table('localization', 'FK duration-weighted localization accounting', ['Quantity', 'Measured value', 'Denominator / definition'],
                   [[k.title(), pct(v) + '%', 'Assessed source-time regions'] for k, v in zip(keys, values)] +
                   [['All-time assessment', pct(coverage[0]) + '%', 'All source duration'], ['Edited-time assessment', pct(coverage[1]) + '%', 'All annotated manipulated duration'], ['Edited-time detected', pct(coverage[2]) + '%', 'All annotated manipulated duration, including unassessed']],
                   'Retained threshold %.8f; no new localization model or threshold fitting. Integrity and localization are separate claims.' % summary['window_threshold'], ['localization'], 'Supporting experiments', widths=[1.4, 1, 2.6])
        # The selected case IDs already occur in the retained six-case forensic result.
        forensic = self.sources.json(_fk_paths.relocate((REPO / 'runs/remaining-supplement-002/forensic/summary.json'), __file__), 'forensic')
        selected = [r['video_id'] for r in forensic['real_source_bound_records'] if r['video_id'] in self.manifest]
        fig, axes = plt.subplots(4, 1, figsize=(7.16, 6.2)); data = []
        for ax, vid in zip(axes, selected):
            row, timeline = self.manifest[vid], timelines[vid]
            xs, ys = [], []
            for segment in timeline['segments']:
                a, b = segment['interval']; score = segment['score']
                xs += [a, b]; ys += [np.nan if score is None else score] * 2
                if score is None: ax.axvspan(a, b, color='#cccccc', alpha=.7)
                data.append({'video_id': vid, 'type': row['source_modify_type'], 'start': a, 'end': b, 'score': score, 'label': segment['label'], 'threshold': timeline['threshold']})
            for a, b in row['manipulated_intervals']: ax.axvspan(a, b, color='#d62728', alpha=.18)
            ax.plot(xs, ys, color='#1f77b4', linewidth=.8); ax.axhline(timeline['threshold'], color='black', linestyle='--', linewidth=.65)
            ax.set(xlim=(0, row['duration']), ylim=(0, 1), ylabel='FK score', title=row['source_modify_type'].replace('_', ' ') + ' | ' + vid.split('/')[1] + '/' + vid.split('/')[3])
            axes_grid(ax)
        axes[-1].set_xlabel('Source time (s)')
        self.fig('localization-examples', 'Source-time localization examples', 'Four preselected metadata/hash cases retained from the existing study, including a missed visual-only manipulation. Red spans: annotated edits; gray: unassessed; dashed: frozen threshold. Examples were not selected for successful predictions.', fig, data, ['main_predictions', 'forensic', 'population'], 'Supporting experiments')
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.7)); data = []
        for kind, label in zip(TYPES, TYPE_LABELS):
            own = [t for vid, t in timelines.items() if self.manifest[vid]['source_modify_type'] == kind]
            sorted_coverage = sorted(t['assessment_coverage'] for t in own)
            axes[0].plot(sorted_coverage, np.arange(1, len(own) + 1) / len(own), label=label)
            data += [{'type': kind, 'video_id': t['video_id'], 'time_coverage': t['assessment_coverage'], 'duration': t['duration']} for t in own]
        axes[0].set(xlabel='Fraction of source time assessed', ylabel='Empirical cumulative fraction', title='(a) Per-video assessment coverage', xlim=(0, 1), ylim=(0, 1)); axes[0].grid(); axes[0].legend(frameon=False, fontsize=6.2)
        totals = summary['seconds']; vals = [totals['tp'], totals['fn'], totals['unassessed_positive']]
        metric_bars(axes[1], ['Detected', 'Missed in\nassessed time', 'Unassessed'], [v / sum(vals) for v in vals], ['#1f77b4', '#ff7f0e', '#bcbcbc'], '% of manipulated time', '(b) All annotated manipulated time')
        self.fig('timeline-coverage', 'Timeline coverage and manipulated-time outcomes', 'Left: empirical coverage distributions for 250 clips per type, including zero-coverage clips. Right: annotated manipulated duration partitioned into detected, assessed-but-missed, and unassessed seconds. Unassessed time is not genuine.', fig, data + [{'type': 'all manipulated seconds', 'category': k, 'seconds': v} for k, v in zip(['detected', 'missed', 'unassessed'], vals)], ['main_predictions', 'localization', 'population'], 'Supporting experiments')
        checks = ['original', 'contribution', 'timestamp', 'threshold', 'coverage', 'source_bytes', 'artifact_bytes']
        records = forensic['real_source_bound_records']; matrix = np.zeros((len(records), len(checks))); labels = []
        data, table = [], []
        for i, row in enumerate(records):
            label = self.manifest[row['video_id']]['source_modify_type'].replace('_', ' ') if row['video_id'] in self.manifest else row['video_id'].split('/')[-1]
            labels.append(label)
            tr = [label, row['record_status']]
            for j, check in enumerate(checks):
                value = row['checks'][check]
                status = 'Verified' if check == 'original' and value['valid'] else ('Rejected' if value.get('status') == 'REJECTED_AS_EXPECTED' else 'N/A')
                matrix[i, j] = 1 if status in ('Verified', 'Rejected') else 0
                data.append({'case': label, 'check': check, 'outcome': status, 'record_status': row['record_status']})
                tr.append(status)
            table.append(tr)
        fig, ax = plt.subplots(figsize=(7.16, 2.9)); ax.imshow(matrix, cmap=matplotlib.colors.ListedColormap(['#eeeeee', '#c6dbef']), vmin=0, vmax=1, aspect='auto')
        for i in range(len(records)):
            for j in range(len(checks)): ax.text(j, i, 'V' if j == 0 else ('R' if matrix[i, j] else 'N/A'), ha='center', va='center', fontsize=7)
        ax.set_xticks(range(len(checks)), ['Original', 'Contrib.', 'Timestamp', 'Threshold', 'Coverage', 'Source\nbytes', 'Artifact\nbytes']); ax.set_yticks(range(len(labels)), labels)
        self.fig('forensic-checks', 'Forensic integrity and mutation checks', 'V: original record verifies; R: tampered copy rejected; N/A: no applicable bins/artifacts. All 6 originals verify and all 32 applicable mutation checks reject; 4 checks are N/A. No-audio and short cases correctly remain UNASSESSED. Integrity/reconstruction does not prove authenticity or classifier correctness.', fig, data, ['forensic'], 'Supporting experiments')
        self.table('forensic-checks', 'Real source-bound forensic verification', ['Case', 'Record status', 'Original', 'Contribution', 'Timestamp', 'Threshold', 'Coverage', 'Source bytes', 'Artifact bytes'], table,
                   'Verified originals: 6/6; applicable mutations rejected: 32/32; nonapplicable: 4. Stored-contribution reconstruction error is zero. This is software integrity validation, not media authenticity or predictive validation.', ['forensic'], 'Supporting experiments', grid=True, widths=[1.6, 1.2, 1, 1.1, 1, 1, 1, 1, 1])
        self.evidence_example()

    def evidence_example(self):
        path = _fk_paths.relocate((REPO / 'runs/table-iii-cost-001/fk/artifacts-00-1/record.json'), __file__)
        record = self.sources.json(path, 'forensic_record_example')
        window = next(w for w in record['windows'] if w['status'] == 'USABLE')
        contrib = np.asarray([b['contributions'] for b in window['bins']]); alpha = np.asarray([b['alpha'] for b in window['bins']]); scores = np.asarray([b['score'] for b in window['bins']])
        reconstructed = 1 / (1 + np.exp(-contrib.sum(axis=1)))
        assert float(np.max(np.abs(scores - reconstructed))) < 1e-12
        fig, axes = plt.subplots(3, 1, figsize=(7.16, 5), gridspec_kw={'height_ratios': [1.4, 1, 1]})
        bound = max(.01, abs(contrib).max()); im = axes[0].imshow(contrib.T, aspect='auto', cmap='RdBu_r', vmin=-bound, vmax=bound)
        axes[0].set_yticks(range(4), ['Visual', 'Audio', 'AV', 'Temporal']); axes[0].set_title('(a) Stored signed contributions'); fig.colorbar(im, ax=axes[0], fraction=.02, pad=.02)
        for i, label in enumerate(['Visual', 'Audio', 'AV', 'Temporal']): axes[1].plot(range(10), alpha[:, i], marker=['o', 's', '^', 'D'][i], label=label)
        axes[1].set_ylabel('Weight'); axes[1].legend(frameon=False, ncol=4, fontsize=6.5); axes[1].set_title('(b) Reliability-derived weights'); axes_grid(axes[1])
        axes[2].plot(range(10), scores, marker='o', label='Recorded score'); axes[2].plot(range(10), reconstructed, linestyle='--', color='#d62728', label='Reconstructed score')
        axes[2].set(xlabel='Bin index within selected usable window', ylabel='Score', ylim=(0, 1), title='(c) Decision reconstruction'); axes[2].legend(frameon=False); axes_grid(axes[2])
        data = [{'bin': i, 'channel': label, 'signed_contribution': float(contrib[i, j]), 'alpha': float(alpha[i, j]), 'recorded_bin_score': float(scores[i]), 'reconstructed_bin_score': float(reconstructed[i])} for i in range(10) for j, label in enumerate(['visual', 'audio', 'av', 'temporal'])]
        self.fig('evidence-record', 'Traceable evidence decomposition from an actual FK record', 'First usable window of the first preselected profiling clip. All values are retained signed contributions, weights and scores. Reconstructing the sigmoid of the contribution sum reproduces the record. Internal weights are not proof of which modality was manipulated.', fig, data, ['forensic_record_example'], 'Supporting experiments')

    def computational_cost(self):
        cost = self.sources.json(_fk_paths.relocate((REPO / 'runs/table-iii-cost-001/table-iii-models.json'), __file__), 'cost_instrumented')
        names = list(cost['models']); display = ['FK\nfull record', 'AVFakeNet\nadapt.', 'AVTENet\nstaged sum*']
        groups = ['decode_io', 'preprocessing', 'feature_extraction', 'fusion_evidence', 'aggregation', 'forensic_record', 'other']
        group_labels = ['Decode / I/O', 'Preprocessing', 'Feature extraction', 'Fusion / evidence', 'Aggregation', 'Forensic record', 'Other / overhead']
        colors = ['#bdbdbd', '#ffbb78', '#1f77b4', '#9467bd', '#2ca02c', '#d62728', '#eeeeee']
        fig, ax = plt.subplots(figsize=(7.16, 3.1)); bottom = np.zeros(3); data = []
        for key, label, color in zip(groups, group_labels, colors):
            values = [statistics.mean(r['mean_' + key] for r in cost['models'][n].values()) for n in names]
            ax.bar(range(3), values, bottom=bottom, color=color, edgecolor='black', linewidth=.25, label=label)
            bottom += values
            data += [{'model': n, 'component': label, 'mean_seconds_per_attempt': value, 'attempted': 12} for n, value in zip(names, values)]
        for i, value in enumerate(bottom): ax.text(i, value + .3, '%.2f' % value, ha='center', fontsize=7)
        ax.set_xticks(range(3), display); ax.set_ylabel('Mean instrumented seconds / attempt'); ax.set_ylim(0, max(bottom) * 1.15); axes_grid(ax); ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False, fontsize=6.5)
        self.fig('cost-components', 'Measured inference components corresponding to Table III', '12 original metadata-selected clips; one warmup and three measured repeats. Additive means use nested self times. FK includes full artifact/record persistence. AVTENet is a sum of separately measured stages, not observed monolithic latency. AVFakeNet succeeds on 5/12 attempts; exclusions remain in the denominator. Instrumentation overhead and sequential decode differ from optimized timings.', fig, data, ['cost_instrumented'], 'Computational cost')
        comp = cost['comparison']; table = []
        for r in comp:
            if r['population'] != 'all_attempts': continue
            table.append([r['model'], '%d/%d' % (r['assessed'], r['attempted']), decimal(r['preprocessing']), decimal(r['feature_extraction']),
                          'In aggregation' if 'AVFakeNet' in r['model'] else decimal(r['fusion_evidence']), decimal(r['aggregation']),
                          decimal(r['forensic_record']) if r['model'] == 'FK' else 'N/A', decimal(r['decode_io']), decimal(r['seconds'])])
        self.table('cost-components', 'Table III-aligned measured inference self times', ['Method', 'Assessed / attempt', 'Preprocess s', 'Feature s', 'Fusion / evidence s', 'Aggregation s', 'Forensic s', 'Decode I/O s', 'Total s'], table,
                   'Median per clip after warmup, including unassessed attempts; medians of components need not add to median total. AVFakeNet classifier/aggregation median is zero because 7/12 preparations fail, not because the model is free. Tiny positive values display <0.0001. AVTENet is a staged sum; FK includes persistence.', ['cost_instrumented'], 'Computational cost', widths=[1.8, 1.2, 1, 1, 1.2, 1.1, 1, 1, 1])
        paired = [r for r in comp if r['population'] == 'common_assessed']
        self.table('cost-common', 'Instrumented cost on five common assessed clips', ['Method', 'Common n', 'Preprocess s', 'Feature s', 'Fusion / evidence s', 'Forensic s', 'Total s', 's / source-second'],
                   [[r['model'], str(r['attempted']), decimal(r['preprocessing']), decimal(r['feature_extraction']), 'In aggregation' if 'AVFakeNet' in r['model'] else decimal(r['fusion_evidence']), decimal(r['forensic_record']) if r['model'] == 'FK' else 'N/A', decimal(r['seconds']), decimal(r['seconds_per_source_second'])] for r in paired],
                   'Same five assessed clip IDs. Small selected sample; unequal method functionality and separate AVTENet runtimes remain. These instrumented values do not replace original optimized profiling.', ['cost_instrumented'], 'Computational cost', widths=[1.8, .8, 1, 1, 1.2, 1, 1, 1.2])
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.65)); data = []
        for ax, key, ylabel in zip(axes, ('seconds', 'seconds_per_source_second'), ('Median seconds / clip', 'Seconds / source-second')):
            metric_bars(ax, display, [r[key] for r in paired], ['#1f77b4', '#2ca02c', '#d62728'], ylabel, 'Same five assessed clips', percent=False)
            data += [{'method': r['model'], 'metric': key, 'value': r[key], 'common_n': r['attempted']} for r in paired]
        self.fig('cost-common', 'Common-assessed instrumented cost', 'Five same successfully assessed clips; medians of three timed repetitions. FK includes persisted forensic artifacts; AVTENet is a staged expert/fusion sum. No inference throughput guarantee or matched-functionality claim.', fig, data, ['cost_instrumented'], 'Computational cost')
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.7)); data = []
        for i, name in enumerate(names):
            rows = list(cost['models'][name].values())
            axes[0].scatter([r['duration'] for r in rows], [r['seconds'] for r in rows], marker=['o', 's', '^'][i], color=['#1f77b4', '#2ca02c', '#d62728'][i], s=17, label=['FK', 'AVFakeNet adapt.', 'AVTENet staged*'][i])
            assessed = [r['seconds'] for r in rows if r['assessed']]; unavailable = [r['seconds'] for r in rows if not r['assessed']]
            axes[1].scatter([i - .07] * len(assessed), assessed, marker='o', color=['#1f77b4', '#2ca02c', '#d62728'][i], s=16)
            axes[1].scatter([i + .07] * len(unavailable), unavailable, marker='x', color='black', s=20)
            data += [{'method': name, 'video_id': vid, 'duration': r['duration'], 'seconds': r['seconds'], 'assessed': r['assessed']} for vid, r in cost['models'][name].items()]
        axes[0].set(xlabel='Source duration (s)', ylabel='Instrumented seconds', title='(a) Per-clip cost'); axes[0].legend(frameon=False, fontsize=6.5)
        axes[1].set_xticks(range(3), display); axes[1].set(ylabel='Instrumented seconds', title='(b) Assessed (circle) / unassessed (cross)')
        for ax in axes: axes_grid(ax)
        self.fig('cost-duration', 'Per-clip cost and assessment outcomes', '12 metadata-selected clips, not selected by prediction success. Scatter is descriptive: duration alone does not explain tracking or preparation failures. AVTENet totals are staged sums; FK includes persistent records.', fig, data, ['cost_instrumented'], 'Computational cost')
        allrows = [r for r in comp if r['population'] == 'all_attempts']
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.65)); data = []
        for ax, key, title in zip(axes, ('sampled_peak_rss_bytes', 'cuda_peak_allocated_bytes'), ('Sampled process RSS', 'Peak allocated CUDA memory')):
            metric_bars(ax, display, [r[key] / 2**20 for r in allrows], ['#1f77b4', '#2ca02c', '#d62728'], 'MiB', title, percent=False)
            data += [{'method': r['model'], 'memory_type': key, 'bytes': r[key]} for r in allrows]
        self.fig('memory', 'Measured process and CUDA memory', 'Maximum across the profiling attempts. RSS is sampled every 20 ms plus snapshots and includes instrument/verification data; it is not an exact instantaneous peak. For AVTENet, the maximum across separately run stages is not the memory of simultaneously loaded experts.', fig, data, ['cost_instrumented'], 'Computational cost')
        self.table('memory', 'Profiling memory with explicit measurement boundaries', ['Method', 'Sampled RSS MiB', 'CUDA allocated MiB', 'CUDA reserved MiB', 'Boundary'],
                   [[r['model'], decimal(r['sampled_peak_rss_bytes'] / 2**20, 1), decimal(r['cuda_peak_allocated_bytes'] / 2**20, 1), decimal(r['cuda_peak_reserved_bytes'] / 2**20, 1), 'Separate-stage maximum' if 'AVTENet' in r['model'] else 'Loaded instrumented process'] for r in allrows],
                   'RSS sampled every 20 ms; OS cache uncontrolled. Model loading excluded. AVTENet is not a simultaneous ensemble peak.', ['cost_instrumented'], 'Computational cost', widths=[1.8, 1, 1, 1, 1.7])
        self.optimized_timing()
        components = list(csv.DictReader((_fk_paths.relocate((REPO / 'runs/table-iii-cost-001/table-iii-components.csv'), __file__)).open(encoding='utf-8-sig')))
        self.sources.register(_fk_paths.relocate((REPO / 'runs/table-iii-cost-001/table-iii-components.csv'), __file__), 'cost_components')
        for stage, title in [('fk', 'FK component microtimings'), ('audiovisual', 'AV-HuBERT component microtimings')]:
            selected = sorted([r for r in components if r['stage'] == stage and float(r['median_self_seconds_per_clip']) > 0], key=lambda r: -float(r['median_self_seconds_per_clip']))[:12]
            fig, ax = plt.subplots(figsize=(7.16, 3.8)); ax.barh(range(len(selected)), [float(r['median_self_seconds_per_clip']) * 1000 for r in selected], color='#1f77b4', edgecolor='black', linewidth=.3)
            ax.set_yticks(range(len(selected)), [r['component'].replace('feature.', '').replace('record.', '').replace('pre.', '').replace('io.', '') for r in selected]); ax.invert_yaxis(); ax.set_xlabel('Median self time per clip (ms)'); ax.grid(axis='x'); ax.set_axisbelow(True)
            self.fig(stage + '-microtimings', title, 'Twelve largest positive median self-time components from the same 12-clip instrumented replay. Named boundaries include shared preparation residuals; timing reflects actual calls, including repeated hashing. These are not analytical operation counts.', fig, selected, ['cost_components'], 'Computational cost')

    def optimized_timing(self):
        original = {}; samples = {}
        base = _fk_paths.relocate((REPO / 'runs/remaining-supplement-002'), __file__)
        for name, folder in [(NAMES[0], 'profiling-fk'), (NAMES[1], 'profiling-avfakenet')]:
            source = self.sources.json(_fk_paths.relocate((base / folder / 'summary.json'), __file__), 'optimized_profiling')
            original[name] = {r['row']['video_id']: r['median_seconds'] for r in source['clips']}
            samples[name] = {r['row']['video_id']: r['samples'][1]['status'] == 'ASSESSED' for r in source['clips']}
        source = self.sources.json(_fk_paths.relocate((base / 'avtenet/summary.json'), __file__), 'optimized_profiling')
        original[NAMES[2]] = {r['video_id']: r['median_staged_seconds'] for r in source['profiling']}
        samples[NAMES[2]] = {r['video_id']: r['samples'][1]['prediction']['status'] == 'ASSESSED' for r in source['profiling']}
        common = set.intersection(*[{k for k, v in samples[n].items() if v} for n in NAMES])
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.65)); data = []
        for ax, use, title in zip(axes, (None, common), ('All 12 attempts', 'Common assessed n = %d' % len(common))):
            vals = [statistics.median(v for k, v in original[n].items() if use is None or k in use) for n in NAMES]
            metric_bars(ax, [SHORT[n] for n in NAMES], vals, [COLORS[n] for n in NAMES], 'Seconds / clip', title, percent=False)
            data += [{'model': n, 'population': title, 'median_seconds': v, 'boundary': 'canonical record only, no persistence' if n == 'FK' else ('separate-stage sum' if n == NAMES[2] else 'raw preparation + native inference')} for n, v in zip(NAMES, vals)]
        self.fig('optimized-timing', 'Original optimized profiling (separate measurement boundary)', 'Retained uninstrumented timing on 12 clips; one warmup plus three passes. FK constructs canonical records but does not persist artifacts. AVTENet is a separately timed expert/fusion sum. All-attempt timing includes exclusions; the common-assessed sample is small. Do not mix these values with full-persistence instrumented costs.', fig, data, ['optimized_profiling'], 'Computational cost')

    def training_history(self):
        names = ['full', 'no_av', 'no_temporal', 'uniform_weights']
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.7)); data = []
        for name, color, marker in zip(names, ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd'], ['o', 's', '^', 'D']):
            hist = self.sources.json(_fk_paths.relocate((REPO / ('runs/remaining-ablations-001/%s/attempt-001/metrics.json' % name)), __file__), 'training_history')['history']
            axes[0].plot([r['epoch'] for r in hist], [r['training']['loss'] for r in hist], color=color, marker=marker, label=name.replace('_', ' '))
            axes[1].plot([r['epoch'] for r in hist], [r['validation']['loss'] for r in hist], color=color, marker=marker, label=name.replace('_', ' '))
            data += [{'variant': name, 'epoch': r['epoch'], 'training_loss': r['training']['loss'], 'validation_loss': r['validation']['loss']} for r in hist]
        for ax, title in zip(axes, ('Training loss', 'Validation loss')): ax.set(xlabel='Completed epoch', ylabel='Recorded multilevel loss', title=title); ax.set_xticks(range(1, 6)); ax.grid()
        axes[1].legend(frameon=False, fontsize=6.5)
        self.fig('training-history', 'Retained FK ablation training histories', 'Five completed epochs, seed 7. Values are the actual logged multilevel losses. Disabled terms and architecture differ, so cross-variant loss magnitudes alone do not establish better detection; use held-out metrics and intervals.', fig, data, ['training_history'], 'Supporting experiments')
        history = self.sources.json(_fk_paths.relocate((REPO / 'runs/table-iii-cost-001/historical-training.json'), __file__), 'historical_training')
        steps = {}
        for path, value in history['sources'].items():
            content = value['contents']
            if 'avtenet' in path and path.endswith('metrics.json'):
                stage = next(s for s in ('audio', 'visual', 'audiovisual', 'fusion') if ('/' + s + '/' in path.replace('\\', '/')))
                steps[stage] = sum(r['train']['steps'] for r in content['history'])
        self.table('training-scope', 'Retained training budgets and unresolved cost measurements', ['Method / stage', 'Completed budget', 'Logged training steps', 'Interpretation'],
                   [['FK heads / controls', '5 epochs; seed 7', 'Video-grouped training', 'Frozen encoders; 2,912/485 usable train/validation'],
                    ['AVFakeNet branches', '1 epoch per branch', 'Native sampler; tiny recovery parents', 'Historical adaptation; no new training'],
                    *[['AVTENet ' + s, '1 epoch', str(steps.get(s, 'N/A')), 'Expert updates' if s != 'fusion' else 'Frozen-expert feature fusion'] for s in ('audio', 'visual', 'audiovisual', 'fusion')],
                    ['Controlled training runtime', 'Not measured', 'N/A', 'Online/cached and cache-I/O benchmark remains future work']],
                   'Actual retained historical budgets only; unequal batch/sample units cannot be treated as equal work. No wall time inferred from file timestamps and no new fitting was performed. The analytical training multiplier 3 is not empirically verified.', ['historical_training'], 'Computational cost', widths=[1.5, 1.2, 1.6, 2.5])

    def finish(self):
        counts = collections.Counter(x['kind'] for x in self.pack.items)
        # Reference PDFs supply visual observations only, through this separate design ledger.
        references = []
        for name, pages, style in [
            ('Paper22.pdf', [13, 14, 15, 16], 'Compact line plots with distinct markers, light grids, serif typography, boxed axes and compact ruled tables.'),
            ('Paper16.pdf', [8, 9, 10, 11], 'Simple comparison bars, blue normalized confusion-matrix heatmaps, class-specific precision/recall/F1/support tables, compact numbered captions.'),
        ]:
            path = _fk_paths.relocate((Path('C:/Users/teeku/Downloads') / name), __file__)
            references.append({'filename': name, 'sha256': sha(path), 'reviewed_style_pages_one_based': pages,
                               'use': 'visual_style_only', 'observed_style': style, 'numerical_import': False,
                               'methods_or_results_imported': False, 'embedded_in_deliverables': False})
        write_json(_fk_paths.relocate((OUT / 'metadata/style-references.json'), __file__), references)
        source_groups = {k: sorted(v) for k, v in self.sources.groups.items()}
        for item in self.pack.items:
            item['source_files'] = sorted(set(path for group in item['source_groups'] for path in source_groups[group]))
            write_json(_fk_paths.relocate((OUT / 'metadata' / (item['stem'] + '.json')), __file__), item)
        changed = [name for name, entry in self.sources.entries.items() if sha(_fk_paths.relocate((ROOT / name), __file__)) != entry['sha256']]
        if changed:
            raise ValueError('An original source changed during plotting: ' + str(changed))
        write_json(_fk_paths.relocate((OUT / 'source-manifest.json'), __file__), {'policy': 'Numerical values exclusively from retained own experiments. Reference papers are isolated in metadata/style-references.json and never used as numerical sources.',
                   'files': self.sources.entries, 'groups': source_groups})
        write_json(_fk_paths.relocate((OUT / 'catalog.json'), __file__), {'created_bangkok': datetime.now(timezone(timedelta(hours=7))).isoformat(timespec='seconds'),
                   'counts': dict(counts), 'data_origin': 'own_experiments_only', 'items': self.pack.items})
        note = ('All figures/tables use our retained local AV-Deepfake1M experiments only. Reference papers inform visual style only. '
                'Baselines are local adaptations with unequal training budgets. The 1,000 clips are unused official-validation samples, not official test. '
                'Own-assessed and common-assessed populations are distinct; exploratory calibrated AVTENet stays supplementary. '
                'No new training, inference, threshold fitting, data download or result replacement was performed to create this pack.')
        sections = ['Main comparisons', 'Supporting experiments', 'Computational cost', 'Supplementary analyses']
        css = '''body{font-family:Georgia,"Times New Roman",serif;margin:0;color:#18212b;background:#f1f3f5}header,main{max-width:1150px;margin:auto;padding:28px}header{background:white;border-top:5px solid #12648c}h1{font-size:30px;margin:5px 0 15px}h2{border-bottom:2px solid #12648c;padding-bottom:9px}nav a,.downloads a{margin-right:14px;color:#075680}article{background:white;margin:22px 0;padding:22px;border:1px solid #d0d7de;break-inside:avoid}article h3{font-size:19px;margin:0 0 15px}img{display:block;max-width:100%;height:auto;margin:auto}p{line-height:1.5}details{font-family:system-ui,sans-serif;font-size:12px}code{overflow-wrap:anywhere}a{color:#075680}.badge{font-family:system-ui,sans-serif;color:#075680;font-size:12px;text-transform:uppercase}footer{margin-top:30px;color:#444}.downloads{font-family:system-ui,sans-serif;font-size:13px}@media print{body{background:white}article{border:0}header,main{padding:10px}details{display:none}}'''
        cards = []
        for section in sections:
            cards.append('<h2 id="' + section.lower().replace(' ', '-') + '">' + html.escape(section) + '</h2>')
            for item in self.pack.items:
                if item['section'] != section:
                    continue
                links = '<a href="%s">600-dpi PNG</a><a href="%s">Vector SVG</a><a href="%s">Data CSV</a>' % (item['png'], item['svg'], item['data'])
                if item.get('latex'): links += '<a href="%s">LaTeX</a>' % item['latex']
                sources = '<br>'.join('<code>' + html.escape(path) + '</code>' for path in item['source_files'])
                cards.append('<article id="%s"><span class="badge">Own experimental data</span><h3>%s. %s</h3><img loading="lazy" src="%s" alt="%s"><p>%s</p><div class="downloads">%s</div><details><summary>Retained data sources</summary>%s</details></article>' % (item['stem'], item['label'], html.escape(item['title']), item['png'], html.escape(item['title']), html.escape(item['caption']), links, sources))
        nav = ''.join('<a href="#%s">%s</a>' % (s.lower().replace(' ', '-'), s) for s in sections)
        page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FK experimental figures and tables</title><style>' + css + '</style><header><h1>FK experimental figures and tables</h1><p><strong>%d figures and %d tables</strong> | journal-style visualization pack</p><p>%s</p><nav>%s</nav><p><a href="FIGURES_AND_TABLES.pdf">PDF review booklet</a> · <a href="README.md">Scope and reproducibility</a> · <a href="source-manifest.json">Source hashes</a></p></header><main>%s<footer>Every plot has an exportable data file. Paper22/Paper16 supplied no numerical values or experiment findings.</footer></main></html>' % (counts['figure'], counts['table'], html.escape(note), nav, '\n'.join(cards))
        (_fk_paths.relocate((OUT / 'index.html'), __file__)).write_text(page, encoding='utf-8')
        lines = ['# Our experimental figure and table pack', '', '**%d figures and %d tables**' % (counts['figure'], counts['table']), '', note, '',
                 'Open [index.html](index.html) for the visual gallery, or [FIGURES_AND_TABLES.pdf](FIGURES_AND_TABLES.pdf) for the review booklet.', '',
                 'Exports: 600-dpi PNG, editable vector SVG, plotted-data CSV, LaTeX tables, per-item captions and provenance. Every number is from retained own results or a descriptive calculation from those fixed results.', '',
                 '## Interpretation requirements', '',
                 '- Default AVTENet threshold 0.5 is primary; its validation-calibrated post-inspection analysis is exploratory and supplementary.',
                 '- AVFakeNet fused decisions are hard majority/OR. Fused AUC and continuous ROC/PR curves are N/A; a measured operating point is allowed.',
                 '- Single manipulation subgroups have constant binary labels: report specificity or fake recall, not subgroup BA/AUC.',
                 '- Paired confidence intervals are existing source-family bootstrap results, not training-seed uncertainty.',
                 '- New subgroup/error-overlap/robustness follow-ups use inspected held-out data and are exploratory.',
                 '- Robustness uses identical derivative files across methods; fixed common IDs are separately disclosed.',
                 '- Profiling on 12 clips is descriptive. AVTENet staged sums are not observed monolithic latency; memory peaks from stages are not simultaneous ensemble peaks.',
                 '- FK full persisted-record cost and earlier optimized canonical-only cost have different boundaries and stay separate.',
                 '- Training runtime remains not measured under a matched online/cached protocol; no times are inferred from file timestamps.',
                 '- Forensic verification establishes integrity and score reconstruction, not media authenticity or classification correctness.', '',
                 '## Reference styling', '',
                 'Paper22: line charts, point markers, light grids, compact serif tables. Paper16: simple model bars, blue confusion heatmaps, class-report tables. No screenshot, logo, paper dataset, paper model performance, author result or copied wording appears in our charts. See metadata/style-references.json.', '', '## Catalog', '']
        for section in sections:
            lines += ['### ' + section, '']
            for item in self.pack.items:
                if item['section'] == section:
                    lines.append('- [%s. %s](%s) — %s' % (item['label'], item['title'], item['png'], item['caption']))
            lines.append('')
        (_fk_paths.relocate((OUT / 'README.md'), __file__)).write_text('\n'.join(lines), encoding='utf-8')
        self.make_pdf(note)
        code = _fk_paths.relocate((OUT / 'reproduce'), __file__); code.mkdir(exist_ok=True)
        for name in ('build_paper_visuals.py', 'journal_visuals_common.py'):
            shutil.copy2(_fk_paths.relocate((Path(__file__)), __file__).with_name(name), _fk_paths.relocate((code / name), __file__))
        audit = {'figure_count': counts['figure'], 'table_count': counts['table'], 'own_source_files_checked': len(self.sources.entries),
                 'all_source_hashes_unchanged': not changed, 'reference_papers_used_as_data': False, 'new_training_or_inference': False,
                 'assessed_counts': {n: self.own[n]['assessed'] for n in NAMES}, 'main_common_assessed': len(self.common),
                 'all_plotted_items_expose_csv': all((_fk_paths.relocate((OUT / x['data']), __file__)).exists() for x in self.pack.items),
                 'all_items_have_provenance': all(x['source_files'] for x in self.pack.items),
                 'export_formats': ['PNG 600 dpi', 'SVG', 'CSV', 'LaTeX tables', 'HTML gallery', 'PDF booklet']}
        write_json(_fk_paths.relocate((OUT / 'validation.json'), __file__), audit)
        write_json(_fk_paths.relocate((OUT / 'complete.json'), __file__), {'files': {p.relative_to(OUT).as_posix(): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'complete.json'}, 'counts': dict(counts), 'source_integrity': True})
        print('COMPLETE', dict(counts), 'own sources', len(self.sources.entries), flush=True)

    def make_pdf(self, note):
        # MuPDF already exists in the workspace cache; no installation needed.
        sys.path.insert(0, str(_fk_paths.relocate((ROOT / '.review-cache/archive-v0/1PiHw63lG2QUPDDZ'), __file__)))
        import pymupdf as fitz
        doc = fitz.open()
        cover = doc.new_page(width=595, height=842)
        cover.insert_text((45, 100), 'FK Experimental Figures', fontname='tibo', fontsize=26, color=(.05, .28, .42))
        cover.insert_text((45, 135), 'and Tables', fontname='tibo', fontsize=26, color=(.05, .28, .42))
        count = collections.Counter(x['kind'] for x in self.pack.items)
        cover.insert_textbox(fitz.Rect(45, 170, 550, 215), '%d figures | %d tables\nOwn retained experimental data only' % (count['figure'], count['table']), fontname='tiro', fontsize=15)
        cover.insert_textbox(fitz.Rect(45, 240, 550, 435), note, fontname='tiro', fontsize=12, lineheight=1.4)
        cover.insert_textbox(fitz.Rect(45, 465, 550, 650), 'Visual reference: Paper22 and Paper16. Their visual conventions were reviewed; their numerical results, methods and datasets were not imported.\n\nThis booklet is a figure/table review pack, not a completed manuscript. Editable SVG, 600-dpi PNG, data CSV and LaTeX tables accompany it.\n\nPrimary: three-model detection and coverage. Supporting: controls, ablations, robustness, localization and forensic checks. Computational costs retain their actual boundaries. Supplementary: calibrated and subgroup analyses.', fontname='tiro', fontsize=11, lineheight=1.35)
        for i, item in enumerate(self.pack.items, 2):
            page = doc.new_page(width=595, height=842)
            page.insert_textbox(fitz.Rect(38, 27, 557, 85), item['label'] + '. ' + item['title'], fontname='tibo', fontsize=13, color=(.03, .24, .38))
            pix = fitz.Pixmap(str(_fk_paths.relocate((OUT / item['png']), __file__)))
            image_height = min(565., 519 * pix.height / pix.width)
            page.insert_image(fitz.Rect(38, 96, 557, 96 + image_height), pixmap=pix, keep_proportion=True)
            y = 112 + image_height
            caption = item['caption'] + '\n\nData: ' + item['data'] + '\nProvenance: metadata/' + item['stem'] + '.json'
            remain = page.insert_textbox(fitz.Rect(38, y, 557, 798), caption, fontname='tiro', fontsize=9.1, lineheight=1.22)
            if remain < 0:
                raise ValueError('PDF caption does not fit: ' + item['stem'])
            page.insert_text((38, 820), 'Own experimental data | style reference only', fontname='tiro', fontsize=8, color=(.3, .3, .3))
            page.insert_text((535, 820), str(i), fontname='tiro', fontsize=8)
        doc.set_metadata({'title': 'FK experimental figures and tables', 'author': 'FK research experiment workspace', 'subject': 'Own data only; reference papers used for visual style'})
        doc.save(_fk_paths.relocate((OUT / 'FIGURES_AND_TABLES.pdf'), __file__), garbage=4, deflate=True)
        doc.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish-only', action='store_true', help='Rebuild the gallery/booklet from checked plots and retained source manifest')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    study = Study()
    if args.publish_only:
        manifest = json.loads((_fk_paths.relocate((OUT / 'source-manifest.json'), __file__)).read_text(encoding='utf-8'))
        study.sources.entries = manifest['files']
        study.sources.groups = {k: set(v) for k, v in manifest['groups'].items()}
        study.pack.items = json.loads((_fk_paths.relocate((OUT / 'catalog.json'), __file__)).read_text(encoding='utf-8'))['items']
        study.finish()
        return
    study.main_comparisons()
    study.supporting_models()
    study.robustness()
    study.localization_forensics()
    study.computational_cost()
    study.training_history()
    study.finish()


if __name__ == '__main__':
    main()
