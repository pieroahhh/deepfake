
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

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from deepfake_detection.core import Decision, coverage, hybrid, sigmoid, supervision, video_score, windows
from deepfake_detection.forensics import build_record, commitment, preserve, sha256_file, verify
from deepfake_detection.data import audit, assign_groups
from deepfake_detection.baselines import avfakenet_decision, avtenet_to_external
from deepfake_detection.evaluation import common_assessed, detection_metrics
from deepfake_detection.cli import demo, load_json


def fixture():
    decision = Decision()
    ws = windows(3., [(0.,3.)], face_ok=[True,True,True])
    for w in ws:
        w["quality"] = {"visual":1.,"audio":1.}
        if w["status"] == "USABLE":
            start = w["interval"][0]
            w["bins"] = [{"interval":[start+b/5,start+(b+1)/5], "alpha":[.25]*4,
                          "contributions":[.5,-.2,.1,0.]} for b in range(10)]
    return build_record("0"*64,3.,ws,decision,model_version="synthetic",preprocessing={"mode":"test"})


def manifest_row(vid="a",family="family-a",split="train"):
    return dict(video_id=vid,dataset="synthetic",release="fixture",source_family=family,
                path=f"{vid}.mp4",split=split,source_sha256=None,duration=3.,label=1,
                audio_label=None,visual_label=1,temporal_complete=True,manipulated_intervals=[[.2,.3]])


class TemporalTests(unittest.TestCase):
    def test_generated_trailing_window_is_unassessed(self):
        ws=windows(3.,[(0.,3.)],face_ok=[True]*3)
        self.assertEqual([w["status"] for w in ws],["USABLE","USABLE","UNASSESSED"])
        self.assertIn("incomplete_window",ws[-1]["reasons"])

    def test_short_video_has_no_complete_window(self):
        self.assertEqual(windows(.5,[(0.,.5)],face_ok=[True])[0]["status"],"UNASSESSED")

    def test_union_coverage_not_double_counted(self):
        c=coverage(5.,[(0.,2.),(1.,3.),(4.,8.)])
        self.assertEqual(c["covered_duration"],4.)
        self.assertEqual(c["unassessed"],[(3.,4.)])

    def test_overlapping_speech_not_double_counted(self):
        ws=windows(2.,[(0.,.6),(.2,.8)],face_ok=[True,True])
        self.assertAlmostEqual(ws[0]["speech_coverage"],.4)
        self.assertEqual(ws[0]["status"],"UNASSESSED")

    def test_missing_audio(self):
        ws=windows(3.,[(0.,3.)],has_audio=False,face_ok=[True]*3)
        self.assertTrue(all(w["status"]=="UNASSESSED" for w in ws))

    def test_face_quality_must_be_explicit(self):
        with self.assertRaises(ValueError): windows(3.,[(0.,3.)])

    def test_any_overlap_excludes_touching_endpoints(self):
        s=supervision(1,True,[(1.,1.2)],[(0.,1.),(1.,2.)],bins=5)
        self.assertEqual([x["window"] for x in s["windows"]],[0,1])
        self.assertEqual(s["windows"][0]["bins"],[0]*5)
        self.assertTrue(s["video_eligible"])

    def test_positive_removed_by_quality_has_no_video_loss(self):
        self.assertFalse(supervision(1,True,[(3.,4.)],[(0.,2.)])["video_eligible"])

    def test_incomplete_positive_labels_are_unknown(self):
        s=supervision(1,False,[(.2,.4)],[(0.,2.)])
        self.assertFalse(s["video_eligible"])
        self.assertIsNone(s["windows"][0]["window"])
        self.assertTrue(all(x is None for x in s["windows"][0]["bins"]))

    def test_real_video_with_usable_windows_is_eligible(self):
        labels = supervision(0,False,[],[(0.,2.)])
        self.assertTrue(labels["video_eligible"])
        self.assertEqual(labels["windows"], [{"window": 0, "bins": [0] * 10}])
        self.assertFalse(supervision(0,True,[],[])["video_eligible"])

    def test_hybrid_and_ratio_rounding(self):
        self.assertAlmostEqual(hybrid([.1,.7,.9],2,.5),.85)
        self.assertAlmostEqual(video_score([.1,.2,.8],Decision(top_ratio=.5)),.65)
        self.assertIsNone(video_score([],Decision()))

    def test_invalid_config_and_scores(self):
        for kwargs in ({"top_k":11},{"top_ratio":0},{"reliability_temperature":0},{"peak_weight":float("nan")}):
            with self.assertRaises(ValueError): Decision(**kwargs)
        with self.assertRaises(ValueError): hybrid([float("nan")],1,.5)
        self.assertEqual(sigmoid(-1000),0.)
        self.assertEqual(sigmoid(1000),1.)


class ForensicTests(unittest.TestCase):
    def test_invalid_verification_tolerance(self):
        record = fixture()
        for tolerance in (-1, float("nan"), float("inf")):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                verify(record, commitment(record), tolerance=tolerance)

    def test_roundtrip_reconstruction(self):
        r=json.loads(json.dumps(fixture()))
        result=verify(r,commitment(r))
        self.assertTrue(result["valid"])
        self.assertEqual(result["max_absolute_error"],0.)

    def test_float32_weights_roundtrip_without_loosening_score_checks(self):
        record = fixture()
        # The first weight is one float32 step above 0.25.
        weights = [0.25 + 2.0**-25, 0.25, 0.25, 0.25]
        record["windows"][0]["bins"][0]["alpha"] = weights
        rebuilt = build_record(
            record["source_sha256"], record["duration"], record["windows"],
            Decision(**record["decision"]), model_version="synthetic", preprocessing={},
        )
        rebuilt = json.loads(json.dumps(rebuilt))
        self.assertEqual(rebuilt["windows"][0]["bins"][0]["alpha"], weights)
        self.assertEqual(verify(rebuilt, commitment(rebuilt))["max_absolute_error"], 0.)

        rebuilt["windows"][0]["bins"][0]["score"] += 1e-8
        with self.assertRaisesRegex(ValueError, "decision reconstruction mismatch"):
            verify(rebuilt, commitment(rebuilt))

    def test_trusted_commitment_detects_mutations(self):
        r=fixture(); trusted=commitment(r)
        for change in ("contribution","timestamp","threshold","coverage","source"):
            edited=copy.deepcopy(r)
            if change=="contribution": edited["windows"][0]["bins"][0]["contributions"][0]+=1
            elif change=="timestamp": edited["windows"][0]["interval"][0]+=.1
            elif change=="threshold": edited["decision"]["video_threshold"]+=.1
            elif change=="coverage": edited["coverage"]["covered_duration"]-=.1
            else: edited["source_sha256"]="1"*64
            with self.subTest(change=change), self.assertRaisesRegex(ValueError,"commitment mismatch"):
                verify(edited,trusted)

    def test_recomputed_hash_does_not_hide_inconsistent_scores(self):
        r=fixture(); r["windows"][0]["bins"][0]["contributions"][0]+=1
        with self.assertRaisesRegex(ValueError,"reconstruction mismatch"): verify(r,commitment(r))

    def test_recomputed_hash_does_not_hide_coverage_error(self):
        r=fixture(); r["coverage"]["covered_duration"]-=.1
        with self.assertRaisesRegex(ValueError,"coverage mismatch"): verify(r,commitment(r))

    def test_unassessed_outputs_null(self):
        ws=windows(1.,[],has_audio=False,face_ok=[False])
        ws[0]["quality"]={}
        r=build_record("0"*64,1.,ws,Decision(),model_version="test",preprocessing={})
        self.assertIsNone(r["score"])
        self.assertIsNone(r["prediction"])
        self.assertEqual(r["coverage"]["unassessed"],[(0.,1.)])
        verify(r,commitment(r))

    def test_threshold_equality_uses_greater_or_equal(self):
        r=fixture()
        r["decision"]["video_threshold"]=r["score"]
        r["prediction"]=1
        verify(r,commitment(r))
        r["prediction"]=0
        with self.assertRaisesRegex(ValueError,"verdict"): verify(r,commitment(r))

    def test_exclusive_copy_and_source_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            src,dst=_fk_paths.relocate((Path(tmp)/"src"), __file__),_fk_paths.relocate((Path(tmp)/"copy"), __file__)
            src.write_bytes(b"source")
            digest=preserve(src,dst)
            self.assertEqual(sha256_file(dst),digest)
            with self.assertRaises(FileExistsError): preserve(src,dst)
            r=fixture(); r["source_sha256"]=digest; trusted=commitment(r)
            src.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError,"source hash mismatch"): verify(r,trusted,source=src)

    def test_invalid_weights_and_bin_timestamps(self):
        for field,value in (("alpha",[.2]*4),("interval",[.01,.2])):
            r=fixture(); r["windows"][0]["bins"][0][field]=value
            with self.assertRaises(ValueError): verify(r,commitment(r))

    def test_demo_and_strict_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(demo(tmp)["valid"])
            self.assertTrue(verify(load_json(_fk_paths.relocate((Path(tmp)/"record.json"), __file__)),(_fk_paths.relocate((Path(tmp)/"trusted-commitment.txt"), __file__)).read_text().strip())["valid"])
            bad=_fk_paths.relocate((Path(tmp)/"bad.json"), __file__)
            for text in ('{"a":1,"a":2}', '{"a":NaN}'):
                bad.write_text(text)
                with self.assertRaises(ValueError): load_json(bad)


class ManifestTests(unittest.TestCase):
    def test_audit_reports_unknown_identity(self):
        self.assertEqual(audit([manifest_row()])["identity_disjointness"],"not established")

    def test_family_leakage(self):
        with self.assertRaisesRegex(ValueError,"source-family leakage"):
            audit([manifest_row(),manifest_row("b",split="test")])

    def test_hash_leakage(self):
        a,b=manifest_row(),manifest_row("b","family-b","test")
        a["source_sha256"]=b["source_sha256"]="a"*64
        with self.assertRaisesRegex(ValueError,"content-hash leakage"): audit([a,b])

    def test_unknown_not_real_and_annotation_validation(self):
        for field,value in (("audio_label",True),("manipulated_intervals",[[0,4]]),("temporal_complete",None)):
            row=manifest_row(); row[field]=value
            with self.assertRaises(ValueError): audit([row])

    def test_complete_positive_requires_interval(self):
        row=manifest_row(); row["manipulated_intervals"]=[]
        with self.assertRaises(ValueError): audit([row])

    def test_split_is_order_independent_and_group_safe(self):
        rows=[manifest_row("a"),manifest_row("b"),manifest_row("c","family-c")]
        split=assign_groups(rows)
        reversed_split=assign_groups(list(reversed(rows)))
        self.assertEqual({r["video_id"]:r["split"] for r in split}, {r["video_id"]:r["split"] for r in reversed_split})
        self.assertEqual(split[0]["split"],split[1]["split"])


class EvaluationTests(unittest.TestCase):
    def rows(self):
        return [dict(video_id="a",label=0,prediction=0,score=.1,status="ASSESSED"),
                dict(video_id="b",label=1,prediction=1,score=.9,status="ASSESSED"),
                dict(video_id="c",label=0,prediction=None,score=None,status="UNASSESSED")]

    def test_coverage_and_denominators(self):
        m=detection_metrics(self.rows())
        self.assertEqual(m["balanced_accuracy"],1.)
        self.assertEqual(m["assessed"],2)
        self.assertAlmostEqual(m["coverage"],2/3)

    def test_null_not_encoded_as_real(self):
        rows=self.rows(); rows[-1]["prediction"]=0
        with self.assertRaises(ValueError): detection_metrics(rows)

    def test_common_population(self):
        rows=self.rows(); other=copy.deepcopy(rows)
        other[1].update(status="ERROR",prediction=None,score=None)
        common=common_assessed({"ours":rows,"baseline":other})
        self.assertEqual(common["video_ids"],["a"])
        self.assertIsNone(common["metrics"]["ours"]["balanced_accuracy"])
        with self.assertRaises(ValueError): common_assessed({"a":rows,"b":other[:-1]})

    def test_baseline_vote_reset_and_native_or(self):
        self.assertEqual(avfakenet_decision([1,1,0],0,tie_break="fake")["prediction"],1)
        self.assertEqual(avfakenet_decision([0,0,1],0,tie_break="fake")["prediction"],0)
        self.assertEqual(avfakenet_decision([0,0],1,tie_break="fake")["prediction"],1)
        self.assertIsNone(avfakenet_decision([0,1],0,tie_break="unassessed")["prediction"])
        self.assertIsNone(avfakenet_decision([],1,tie_break="fake")["prediction"])
        self.assertEqual(avtenet_to_external(0),1)
