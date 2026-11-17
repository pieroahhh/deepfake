"""Validation-only selection using recomputed evidence weights and hierarchy."""
from dataclasses import replace
from itertools import product

import torch

from .core import Decision


def select_threshold(scores, labels, default=.5):
    if set(labels) != {0, 1}:
        raise ValueError('threshold selection requires both classes with known labels')
    positives, negatives = labels.count(1), labels.count(0)
    best = None
    for threshold in sorted({0., 1., default, *scores}):
        tpr = sum(s >= threshold and y == 1 for s, y in zip(scores, labels))/positives
        tnr = sum(s < threshold and y == 0 for s, y in zip(scores, labels))/negatives
        key = ((tpr+tnr)/2, -abs(threshold-default), -threshold)
        if best is None or key > best[0]:
            best = key, threshold
    return {'threshold': best[1], 'balanced_accuracy': best[0][0]}


@torch.inference_mode()
def select_decision(model, validation_rows, grid=None):
    from .training import _population_signature, forward_video
    signature, _, _, _ = _population_signature(validation_rows, 'validation')
    original = model.decision
    grid = grid or {'top_k': [1, 3, 5], 'peak_weight': [0., .5, 1.],
                    'top_ratio': [.1, .2, .5], 'video_peak_weight': [0., .5, 1.],
                    'reliability_temperature': [.25, .5, 1.]}
    if set(grid) != {'top_k', 'peak_weight', 'top_ratio', 'video_peak_weight', 'reliability_temperature'}:
        raise ValueError('calibration grid must specify all aggregation parameters')
    model.eval()
    cached = []
    for row in validation_rows:
        out, _, targets = forward_video(model, row)
        if out is not None:
            cached.append((row['label'], targets, {k: v.cpu() for k, v in out.items()}))
    candidates, best = [], None
    keys = tuple(grid)
    for values in product(*(grid[k] for k in keys)):
        decision = replace(original, **dict(zip(keys, values)))
        videos, labels, windows, window_labels = [], [], [], []
        for label, targets, out in cached:
            if getattr(model, 'model_kind', 'fk') == 'fk':
                active = torch.tensor([True, True, 'av' not in model.disabled_evidence,
                                       'temporal' not in model.disabled_evidence])
                weights = (out['reliability']/decision.reliability_temperature).masked_fill(~active, -torch.inf).softmax(-1)
                bins = (weights*out['evidence_logits']).sum(-1).sigmoid()
            else:
                bins = out['bin_scores']
            ws = model._hybrid(bins, decision.top_k, decision.peak_weight)
            import math
            video = model._hybrid(ws, max(1, math.ceil(decision.top_ratio*len(ws))), decision.video_peak_weight)
            videos.append(video.item()); labels.append(label)
            for score, target in zip(ws.tolist(), targets['windows']):
                if target['window'] is not None:
                    windows.append(score); window_labels.append(target['window'])
        selection = select_threshold(videos, labels, original.video_threshold)
        distance = sum(abs(getattr(decision, k)-getattr(original, k)) for k in keys)
        key = (selection['balanced_accuracy'], -distance)
        candidates.append({'parameters': dict(zip(keys, values)), **selection})
        if best is None or key > best[0]:
            best = key, decision, selection, windows, window_labels
    if best is None:
        raise ValueError('empty calibration grid')
    _, decision, selection, windows, labels = best
    window_selection = select_threshold(windows, labels, original.window_threshold)
    decision = replace(decision, video_threshold=selection['threshold'], window_threshold=window_selection['threshold'])
    return {'decision': decision.to_dict(), 'video_threshold': decision.video_threshold,
            'balanced_accuracy': selection['balanced_accuracy'], 'window_selection': window_selection,
            'split': 'validation', 'validation_signature': signature, 'assessed_videos': len(cached),
            'candidates': candidates, 'tie_policy': 'closest_L1_to_original_then_grid_order; threshold_closest_to_original_then_lower'}
