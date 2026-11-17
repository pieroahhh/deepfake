"""Pretrained-backend parity checks requiring matching inputs and artifact provenance."""

import numpy as np


def compare_backend_exports(reference,actual,*,reference_provenance,actual_provenance,atol=1e-5,rtol=1e-5):
    """Compare aligned intermediate tensors, not only final classifier labels.

    Exports should include tubelets, spatial vectors and temporal vectors. This
    utility cannot certify how reference arrays were generated; retain exact
    upstream/checkpoint identity and a shared input-content hash independently.
    """
    if (not np.isfinite(atol) or not np.isfinite(rtol) or atol<0 or rtol<0
            or not reference_provenance.get('implementation_revision')
            or not actual_provenance.get('implementation_revision')
            or not reference_provenance.get('checkpoint_sha256')
            or not actual_provenance.get('checkpoint_sha256')
            or reference_provenance.get('input_sha256')!=actual_provenance.get('input_sha256')
            or not reference_provenance.get('input_sha256')):
        raise ValueError('explicit compatible input/checkpoint/revision provenance and finite tolerances required')
    if set(reference)!=set(actual) or not reference:
        raise ValueError('parity exports must contain the same named stages')
    results={}
    for name in reference:
        expected=np.asarray(reference[name]);value=np.asarray(actual[name])
        if expected.shape!=value.shape or not np.isfinite(expected).all() or not np.isfinite(value).all():
            raise ValueError('shape/nonfinite mismatch in parity stage '+name)
        delta=np.abs(expected.astype(np.float64)-value.astype(np.float64))
        passed=bool(np.allclose(value,expected,atol=atol,rtol=rtol))
        results[name]={'passed':passed,'shape':list(value.shape),
                       'maximum_absolute_error':float(delta.max(initial=0)),
                       'mean_absolute_error':float(delta.mean()) if delta.size else 0.}
    return {'passed':all(stage['passed'] for stage in results.values()),'atol':atol,'rtol':rtol,
            'stages':results,'reference':reference_provenance,'actual':actual_provenance}
