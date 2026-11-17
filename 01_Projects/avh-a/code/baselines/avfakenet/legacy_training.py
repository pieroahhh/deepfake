"""Separate-branch training on prepared, modality-labeled image batches.

Data loading, source-family split auditing and augmentation belong upstream.
No native-paper accuracy or dataset readiness is implied by this runner.
"""

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


from pathlib import Path

import torch

from deepfake_detection.avfakenet import classification_loss
from deepfake_detection.cli import save


def _epoch(model, batches, smoothing, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    device = next(model.parameters()).device
    target_key = f"{model.modality}_label"
    with torch.set_grad_enabled(training):
        for batch in batches:
            # Do not silently train an audio branch on a video-level fake label.
            images = batch["images"].to(device)
            labels = batch[target_key].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = classification_loss(logits, labels, smoothing)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training/evaluation loss")
            if training:
                loss.backward()
                optimizer.step()
            size = len(labels)
            total_loss += loss.detach().item() * size
            correct += (logits.detach().argmax(-1) == labels).sum().item()
            count += size
    if count == 0:
        raise ValueError("training and validation populations must be nonempty")
    return {"loss": total_loss / count, "accuracy": correct / count, "images": count}


def fit_branch(model, train_batches, validation_batches, output, *, epochs=None):
    """Train one branch; retain the earliest best validation-accuracy checkpoint.

    Both loaders must be re-iterable and use disjoint, audited training and
    validation groups. Batch keys are images and visual_label or audio_label.
    Unknown labels must be excluded upstream. A new output directory is required.
    Set epochs=1 for integration smoke tests, not a completed experiment.
    """
    settings = model.profile["training"]
    budget = settings["maximum_epochs"] if epochs is None else epochs
    if type(budget) is not int or not 1 <= budget <= settings["maximum_epochs"]:
        raise ValueError("epoch budget must be positive and no greater than the profile maximum")
    output = _fk_paths.relocate((Path(output)), __file__)
    output.mkdir(parents=True, exist_ok=False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"],
        betas=tuple(settings["betas"]), eps=settings["epsilon"],
    )
    history, best_accuracy, stale = [], -1.0, 0
    save(_fk_paths.relocate((output / "run.json"), __file__), {
        "status": "STARTED", "profile": model.profile,
        "profile_sha256": model.profile_fingerprint, "modality": model.modality,
        "epoch_budget": budget, "torch_version": str(torch.__version__),
        "label_convention": "0=genuine; 1=fake; targets are modality-specific",
        "data_audit": "caller_required_not_verified_by_runner",
        "seed": torch.initial_seed(),
    })
    for epoch in range(1, budget + 1):
        training = _epoch(model, train_batches, settings["label_smoothing"], optimizer)
        validation = _epoch(model, validation_batches, settings["label_smoothing"])
        history.append({"epoch": epoch, "training": training, "validation": validation})
        if validation["accuracy"] > best_accuracy:
            best_accuracy, stale = validation["accuracy"], 0
            torch.save({
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "profile": model.profile, "profile_sha256": model.profile_fingerprint,
                "modality": model.modality, "epoch": epoch, "validation": validation,
                "torch_rng_state": torch.get_rng_state(),
            }, _fk_paths.relocate((output / "best.pt"), __file__))
        else:
            stale += 1
            if stale >= settings["early_stopping_patience"]:
                break
    checkpoint = torch.load(_fk_paths.relocate_value((_fk_paths.relocate((output / "best.pt"), __file__)), __file__), map_location=next(model.parameters()).device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    result = {
        "status": "FINISHED", "epochs_completed": len(history),
        "best_epoch": checkpoint["epoch"], "best_validation_accuracy": best_accuracy,
        "profile_sha256": model.profile_fingerprint, "history": history,
    }
    save(_fk_paths.relocate((output / "metrics.json"), __file__), result)
    return result
