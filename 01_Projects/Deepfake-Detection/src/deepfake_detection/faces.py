"""Explicit detector interfaces, conservative single-face tracking and lip crops.

YuNet/dlib are optional preparation choices, not the AVFakeNet MTCNN method.
Geometric lip crops are a reconstruction; they are not the official LRS3 pipeline.
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


import math

import cv2
import numpy as np

from .forensics import sha256_file


def _verified(path, expected):
    if sha256_file(path) != expected:
        raise ValueError('detector/landmark model checksum mismatch')


class MTCNNDetector:
    """Wrap an explicitly supplied facenet-pytorch MTCNN with bundled weights."""

    method = 'MTCNN'

    def __init__(self, detector, threshold=.9):
        self.detector, self.threshold = detector, threshold

    def __call__(self, rgb):
        boxes, scores, landmarks = self.detector.detect(rgb, landmarks=True)
        if boxes is None:
            return []
        return [{'box': box.tolist(), 'score': float(score), 'landmarks': points.tolist()}
                for box, score, points in zip(boxes, scores, landmarks) if score >= self.threshold]


class YuNetDetector:
    method = 'YuNet_RECONSTRUCTION_ADAPTATION_NOT_MTCNN'

    def __init__(self, model_path, sha256, threshold=.9):
        _verified(model_path, sha256)
        self.detector = cv2.FaceDetectorYN.create(_fk_paths.relocate_value((str(model_path)), __file__), '', (320, 320), score_threshold=threshold)

    def __call__(self, rgb):
        height, width = rgb.shape[:2]
        self.detector.setInputSize((width, height))
        _, detections = self.detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if detections is None:
            return []
        result = []
        for detection in detections:
            x, y, w, h = detection[:4]
            # YuNet and MTCNN both expose two eyes, nose and two mouth corners;
            # order within each eye/mouth pair is normalized geometrically below.
            result.append({'box': [float(x), float(y), float(x+w), float(y+h)],
                           'score': float(detection[-1]), 'landmarks': detection[4:14].reshape(5, 2).tolist()})
        return result


class Dlib68Detector:
    method = 'dlib68_RECONSTRUCTION_ADAPTATION_NOT_MTCNN'

    def __init__(self, predictor_path, sha256):
        import dlib
        _verified(predictor_path, sha256)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(_fk_paths.relocate_value((str(predictor_path)), __file__))

    def __call__(self, rgb):
        result = []
        for box in self.detector(rgb, 0):
            shape = self.predictor(rgb, box)
            result.append({'box': [box.left(), box.top(), box.right()+1, box.bottom()+1],
                           'score': 1.0, 'landmarks': [[p.x, p.y] for p in shape.parts()]})
        return result


def _iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a, area_b = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
    return intersection / (area_a + area_b - intersection) if area_a + area_b > intersection else 0


def _geometry(detection):
    box = np.asarray(detection['box'], dtype=float)
    points = np.asarray(detection['landmarks'], dtype=float)
    if (box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]
            or points.shape not in ((5, 2), (68, 2)) or not np.isfinite(points).all()):
        raise ValueError('invalid face box or landmarks')
    if len(points) == 68:
        eyes = np.stack((points[36:42].mean(0), points[42:48].mean(0)))
        mouth = points[48:68].mean(0)
    else:
        eyes, mouth = points[:2], points[3:5].mean(0)
    eyes = eyes[np.argsort(eyes[:, 0])]
    if np.linalg.norm(eyes[1] - eyes[0]) < 1:
        raise ValueError('degenerate eye landmarks')
    return box, eyes, mouth


def aligned_crops(rgb, detection, *, face_pixels=224, lip_pixels=96, fit_to_frame=False):
    """Rotate eyes horizontal, then crop a face square and mouth-centered square.

    With fit_to_frame, shrink a boundary-crossing crop around its center while
    requiring the facial landmarks to remain inside the observed pixels.
    """
    box, eyes, mouth = _geometry(detection)
    center = eyes.mean(0)
    angle = math.degrees(math.atan2(*(eyes[1]-eyes[0])[::-1]))
    matrix = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
    height, width = rgb.shape[:2]
    aligned = cv2.warpAffine(rgb, matrix, (width, height), flags=cv2.INTER_LINEAR)
    valid = cv2.warpAffine(np.ones((height, width), dtype=np.float32), matrix, (width, height), flags=cv2.INTER_LINEAR)
    face_center = matrix @ np.array([(box[0]+box[2])/2, (box[1]+box[3])/2, 1])
    mouth_center = matrix @ np.append(mouth, 1)

    def crop(at, side, pixels, landmarks):
        def bounds(size):
            x1, y1 = np.floor(at - size / 2).astype(int)
            x2, y2 = np.ceil(at + size / 2).astype(int)
            return x1, y1, x2, y2

        def observed(box):
            x1, y1, x2, y2 = box
            return (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
                    and (valid[y1:y2, x1:x2] >= 1 - 1e-6).all())

        box = bounds(side)
        if not observed(box) and fit_to_frame:
            lower, upper = 1, math.floor(side)
            while lower <= upper:
                candidate_size = (lower + upper) // 2
                candidate = bounds(candidate_size)
                if observed(candidate):
                    box = candidate
                    lower = candidate_size + 1
                else:
                    upper = candidate_size - 1
            x1, y1, x2, y2 = box
            aligned_landmarks = landmarks @ matrix[:, :2].T + matrix[:, 2]
            if not ((aligned_landmarks >= (x1, y1)) & (aligned_landmarks < (x2, y2))).all():
                raise ValueError('in-frame crop cannot retain facial landmarks')
        if not observed(box):
            raise ValueError('face/lip crop extends beyond observed source pixels')
        x1, y1, x2, y2 = box
        return cv2.resize(aligned[y1:y2, x1:x2], (pixels, pixels), interpolation=cv2.INTER_LINEAR)

    side = max(box[2]-box[0], box[3]-box[1])
    landmarks = np.asarray(detection['landmarks'], dtype=float)
    mouth_landmarks = landmarks[48:68] if len(landmarks) == 68 else landmarks[3:5]
    face = crop(face_center, side, face_pixels, landmarks)
    lips = crop(mouth_center, side * .5, lip_pixels, mouth_landmarks)
    return face, lips


def track_and_crop(frames, detector, *, minimum_iou=.3, face_pixels=224, lip_pixels=96, fit_to_frame=False):
    """Initialize only with one face; reject ambiguous associations and missing frames."""
    previous, records = None, []
    for rgb in frames:
        detections = detector(rgb)
        reason, chosen = None, None
        if not detections:
            reason = 'missing_face'
        elif previous is None:
            if len(detections) == 1:
                chosen = detections[0]
            else:
                reason = 'ambiguous_primary_face'
        else:
            candidates = [item for item in detections if _iou(previous, item['box']) >= minimum_iou]
            if len(candidates) == 1:
                chosen = candidates[0]
            else:
                reason = 'lost_or_ambiguous_face_track'
        if chosen is not None:
            try:
                face, lips = aligned_crops(rgb, chosen, face_pixels=face_pixels, lip_pixels=lip_pixels,
                                          fit_to_frame=fit_to_frame)
                tracking_iou = 1.0 if previous is None else _iou(previous, chosen['box'])
                previous = chosen['box']
                records.append({'status': 'USABLE', 'face': face, 'lips': lips,
                                'detection': chosen, 'tracking_iou': tracking_iou, 'method': detector.method})
                continue
            except ValueError as error:
                reason = str(error)
        # Do not jump to a different subject after any gap within this window.
        records.append({'status': 'UNASSESSED', 'reason': reason, 'method': detector.method})
    return records
