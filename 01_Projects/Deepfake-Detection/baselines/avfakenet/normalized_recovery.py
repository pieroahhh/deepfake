"""Explicit experimental normalization for training-only recovery attempts.

This module does not change the historical DSTNet implementation or protocol.
Statistics must be fitted on training examples and remain fixed during fitting
and inference. Buffers are included in the model checkpoint.
"""
import torch
from torch import nn


class FixedNormalizedHead(nn.Module):
    def __init__(self, classifier, mean, scale):
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32).detach().clone()
        scale = torch.as_tensor(scale, dtype=torch.float32).detach().clone()
        if mean.ndim != 1 or mean.shape != scale.shape or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('Expected finite feature means and strictly positive scales')
        self.register_buffer('mean', mean)
        self.register_buffer('scale', scale)
        self.classifier = classifier

    def forward(self, features):
        return self.classifier((features - self.mean) / self.scale)


def trainable_groups(model):
    return {
        'input_block': list(model.input_block.parameters()),
        'patch_embedding': list(model.patch_embedding.parameters()),
        'stages': list(model.stages.parameters()),
        'merges': list(model.merges.parameters()),
        'classifier': list(model.output_block.parameters()),
    }


def gradient_norms(model):
    result = {}
    for name, parameters in trainable_groups(model).items():
        squared = sum(float(p.grad.detach().double().square().sum()) for p in parameters if p.grad is not None)
        result[name] = squared ** .5
    return result


def set_training_mode(model, feature_dropout):
    """Disabling feature dropout does not freeze its weights or gradients."""
    model.train()
    if feature_dropout == 'off':
        for module in (model.input_block, model.patch_embedding, model.stages, model.merges):
            module.eval()
    elif feature_dropout != 'on':
        raise ValueError('Feature dropout must be on or off')
