"""Native decision adapters only; these do not reconstruct baseline networks."""


def avfakenet_decision(frame_predictions, audio_prediction, *, tie_break):
    """External convention: 1=fake. Tie policy is an explicit reconstruction choice."""
    if tie_break not in ("fake", "real", "unassessed"):
        raise ValueError("explicit tie policy required")
    if not frame_predictions or audio_prediction is None:
        return {"status": "UNASSESSED", "prediction": None, "score": None}
    if any(type(x) is not int or x not in (0, 1) for x in [*frame_predictions, audio_prediction]):
        raise ValueError("predictions must be integer 0 or 1")
    fake, total = sum(frame_predictions), len(frame_predictions)
    if 2*fake == total:
        if tie_break == "unassessed":
            return {"status": "UNASSESSED", "prediction": None, "score": None}
        visual = int(tie_break == "fake")
    else:
        visual = int(2*fake > total)
    return {"status": "ASSESSED", "prediction": int(bool(visual or audio_prediction)),
            "score": None, "visual_prediction": visual, "audio_prediction": audio_prediction}


def avtenet_to_external(native_label):
    """AVTENet manuscript: native 0=fake, 1=real; project convention is reversed."""
    if type(native_label) is not int or native_label not in (0, 1):
        raise ValueError("expected native integer class 0 or 1")
    return 1-native_label
