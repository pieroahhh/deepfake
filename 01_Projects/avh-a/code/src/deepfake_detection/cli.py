"""Project entry points. All artifacts are explicit paths."""

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
import json
from pathlib import Path
from .core import Decision, windows
from .data import audit, assign_groups
from .evaluation import detection_metrics
from .forensics import build_record, canonical, commitment, preserve, sha256_file, verify


def load_json(path):
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON number: {value}")
    def unique_keys(pairs):
        result = {}
        for key,value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(_fk_paths.relocate((Path(path)), __file__).read_text(), parse_constant=reject_constant, object_pairs_hook=unique_keys)


def save(path, value):
    path = _fk_paths.relocate((Path(path)), __file__)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects earlier experiment artifacts from accidental replacement.
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def demo(output):
    output = _fk_paths.relocate((Path(output)), __file__)
    output.mkdir(parents=True, exist_ok=True)
    source = _fk_paths.relocate((output/"synthetic-source.txt"), __file__)
    with source.open("xb") as stream:
        stream.write(b"SYNTHETIC SOFTWARE FIXTURE; NOT MEDIA OR EXPERIMENTAL EVIDENCE\n")
    source_hash = preserve(source, _fk_paths.relocate((output/"synthetic-working-copy.txt"), __file__))
    decision = Decision()
    records = windows(3.0, [(0.0,3.0)], face_ok=[True,True,True])
    for w in records:
        w["quality"] = {"visual": 1.0, "audio": 1.0, "provenance": "synthetic fixture"}
        if w["status"] == "USABLE":
            start, end = w["interval"]
            w["bins"] = [{"interval": [start+b*(end-start)/decision.bins,start+(b+1)*(end-start)/decision.bins],
                          "alpha": [0.25]*4, "contributions": [0.2,-0.1,0.15,0.0]} for b in range(decision.bins)]
    record = build_record(source_hash, 3.0, records, decision,
                          model_version="SYNTHETIC-NO-MODEL", preprocessing={"mode": "synthetic fixture"})
    save(_fk_paths.relocate((output/"record.json"), __file__), record)
    with (_fk_paths.relocate((output/"trusted-commitment.txt"), __file__)).open("x") as stream:
        stream.write(commitment(record)+"\n")
    return verify(record, commitment(record), source=source)


def _main(argv=None):
    parser = argparse.ArgumentParser(description="FK project preparation tools")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("demo", help="synthetic record demo, not raw-media inference")
    p.add_argument("--output", required=True)
    p = commands.add_parser("verify")
    p.add_argument("--record", required=True)
    p.add_argument("--trusted-commitment", required=True, help="independently retained commitment file")
    p.add_argument("--source")
    p.add_argument("--artifact-root", help="verify the hashes of v2 derived evidence files")
    for name in ("audit", "split", "metrics"):
        p = commands.add_parser(name)
        p.add_argument("--input", required=True)
        p.add_argument("--output", required=True)
        if name == "split":
            p.add_argument("--seed", default="fk-provisional-v1")
    p = commands.add_parser("fingerprint")
    p.add_argument("--output", required=True)
    p.add_argument("sources", nargs="+")
    p = commands.add_parser("inspect-media", help="FFprobe stream/timestamp metadata")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("infrastructure", help="check explicit local model artifact readiness")
    p.add_argument("--assets", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("train-fk", help="whole-video training from verified feature caches")
    p.add_argument("--manifest", required=True)
    p.add_argument("--model-config", required=True)
    p.add_argument("--training-config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int)
    p.add_argument("--resume")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p = commands.add_parser("cache-fk", help="build verified feature caches and a training manifest")
    p.add_argument("--manifest", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--assets", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--resume", action="store_true", help="verify completed clips and continue in the same output directory")
    p.add_argument("--quality-normalization", help="train-fitted measured-quality-v2 JSON; omitted means legacy quality")
    p = commands.add_parser('fit-quality', help='fit quality normalization from training media only')
    p.add_argument('--manifest', required=True)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--assets', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--resume', action='store_true', help='continue verified per-window state beside the same output JSON')
    p.add_argument('--progress-seconds', type=float, default=10, help='progress interval in seconds (default: 10)')
    p = commands.add_parser('evaluate-fk', help='evaluate FK or shared baselines with all assessment denominators')
    p.add_argument('--manifest', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split', choices=('validation','test'), default='test')
    p.add_argument('--device', choices=('auto','cuda','cpu'), default='auto')
    p.add_argument('--output', required=True)
    p = commands.add_parser('run-experiments', help='train and calibrate FK variants/shared baselines; no test scoring')
    p.add_argument('--plan', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--training-config', required=True)
    p.add_argument('--epochs', type=int)
    p.add_argument('--device', choices=('auto','cuda','cpu'), default='auto')
    p.add_argument('--output', required=True)
    p = commands.add_parser('prepare-avdeepfake1m', help='prepare the primary AV-Deepfake1M train/val population')
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--train-clips', type=int, help='optional whole-family clip cap; default uses all')
    p.add_argument('--validation-clips', type=int, help='optional whole-family clip cap; default uses all')
    p.add_argument('--test-metadata', help='optional genuinely labeled official test metadata; never test_files.txt')
    p.add_argument('--source-overlap-policy', choices=('error','drop-train','keep-official'), default='error',
                   help='keep-official retains all official split rows and reports overlap; drop-train excludes shared train families')
    p.add_argument('--test-clips', type=int)
    p.add_argument('--seed', type=int, default=7)
    p = commands.add_parser("prepare-lavdf-study", help="legacy LAV-DF adapter; not the primary benchmark")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--train-clips", type=int, default=15000)
    p.add_argument("--validation-clips", type=int, default=2500)
    p.add_argument("--test-clips", type=int, default=2500)
    p.add_argument("--seed", type=int, default=7)
    p = commands.add_parser("prepare-lavdf-pilot", help="audit LAV-DF labels and select train/dev families")
    p.add_argument("--dataset-root", required=True, help="extracted LAV-DF directory with metadata.min.json")
    p.add_argument("--output", required=True, help="new directory for manifest.json and audit.json")
    p.add_argument("--train-families", type=int, default=8)
    p.add_argument("--validation-families", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p = commands.add_parser("prepare-experiments", help="write provisional retraining/ablation plans")
    p.add_argument("--model-config", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("report", help="detection CSV/JSON and optional paired grouped bootstrap")
    p.add_argument("--input", required=True, help="JSON containing methods and provenance")
    p.add_argument("--output", required=True)
    p.add_argument("--reference")
    p.add_argument("--replicates", type=int, default=2000)
    p = commands.add_parser("corrupt-media", help="immutable controlled media derivative")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--kind", required=True, choices=['video_compression','video_blur','video_downsample','audio_compression','av_offset'])
    p.add_argument("--strength", required=True, type=float)
    p = commands.add_parser("infer-fk", help="actual local assets/checkpoint required; no random fallback")
    p.add_argument("--input", required=True)
    p.add_argument("--duration", required=True, type=float)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--assets", required=True)
    p.add_argument("--quality-profile", required=True)
    p.add_argument("--fit-face-crops", action="store_true", default=None, help="fit boundary crops within observed pixels")
    p.add_argument("--output", required=True)
    p = commands.add_parser("profile-smoke", help="measure only FK head on synthetic cached features")
    p.add_argument("--output", required=True)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            result = demo(args.output)
        elif args.command == "verify":
            result = verify(load_json(args.record), _fk_paths.relocate((Path(args.trusted_commitment)), __file__).read_text().strip(), source=args.source, artifact_root=args.artifact_root)
        elif args.command == "fingerprint":
            result = [{"path": p, "sha256": sha256_file(p), "bytes": _fk_paths.relocate((Path(p)), __file__).stat().st_size} for p in args.sources]
            save(args.output,result)
        elif args.command == "inspect-media":
            from .media import probe
            result = probe(args.input)
            save(args.output, result)
        elif args.command == "infrastructure":
            from .infrastructure import readiness
            result = readiness(load_json(args.assets))
            save(args.output, result)
        elif args.command == "train-fk":
            from .infrastructure import train_fk_manifest
            result = train_fk_manifest(
                args.manifest, args.model_config, args.training_config, args.output,
                epochs=args.epochs, resume_from=args.resume, calibrate=args.calibrate, device=args.device,
            )
        elif args.command == "cache-fk":
            from .fk_caching import cache_fk_manifest
            result = cache_fk_manifest(args.manifest, args.dataset_root, args.assets, args.output,
                                       device=args.device, resume=args.resume, quality_normalization=args.quality_normalization)
        elif args.command == 'fit-quality':
            from .quality import fit_quality_manifest
            result = fit_quality_manifest(args.manifest,args.dataset_root,args.assets,args.output,
                                          resume=args.resume,progress_seconds=args.progress_seconds)
        elif args.command == 'evaluate-fk':
            from .study import evaluate_checkpoint
            result = evaluate_checkpoint(args.manifest,args.checkpoint,args.output,split=args.split,device=args.device)
        elif args.command == 'run-experiments':
            from .study import run_experiments
            result = run_experiments(args.plan,args.manifest,args.training_config,args.output,epochs=args.epochs,device=args.device)
        elif args.command == 'prepare-avdeepfake1m':
            from .avdeepfake1m import prepare_avdeepfake1m
            limits = {'train':args.train_clips,'validation':args.validation_clips}
            if args.test_metadata:
                limits['test']=args.test_clips
            elif args.test_clips is not None:
                raise ValueError('--test-clips requires labeled --test-metadata')
            result = prepare_avdeepfake1m(args.dataset_root,args.output,clip_limits=limits,
                                         seed=args.seed,test_metadata=args.test_metadata,
                                         source_overlap_policy=args.source_overlap_policy)
        elif args.command == "prepare-lavdf-study":
            from .lavdf import prepare_lavdf_study
            result = prepare_lavdf_study(
                args.dataset_root, args.output, seed=args.seed,
                clip_limits={"train": args.train_clips, "validation": args.validation_clips, "test": args.test_clips},
            )
        elif args.command == "prepare-lavdf-pilot":
            from .lavdf import prepare_lavdf_pilot
            result = prepare_lavdf_pilot(
                args.dataset_root, args.output, train_families=args.train_families,
                validation_families=args.validation_families, seed=args.seed,
            )
        elif args.command == "prepare-experiments":
            from .experiments import prepare_experiments
            result = prepare_experiments(load_json(args.model_config), args.output)
        elif args.command == "report":
            from .reporting import write_detection_report
            data = load_json(args.input)
            result = write_detection_report(data['methods'], args.output, provenance=data['provenance'],
                                            reference=args.reference, replicates=args.replicates)
        elif args.command == "corrupt-media":
            from .robustness import transcode_media
            strength = int(args.strength) if args.kind in ('video_compression','audio_compression') and args.strength.is_integer() else args.strength
            result = transcode_media(args.input, args.output, args.kind, strength)
        elif args.command == "infer-fk":
            from .commands import infer_fk_from_assets
            result = infer_fk_from_assets(args.input,args.duration,args.checkpoint,args.assets,args.output,
                                          args.quality_profile,fit_to_frame=args.fit_face_crops)
        elif args.command == "profile-smoke":
            from .commands import profile_synthetic_fk
            result = profile_synthetic_fk(args.output, iterations=args.iterations, threads=args.threads)
        else:
            rows = load_json(args.input)
            if args.command == "audit":
                result = audit(rows)
            elif args.command == "split":
                result = assign_groups(rows,seed=args.seed)
            else:
                result = detection_metrics(rows)
            save(args.output,result)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result,allow_nan=False,indent=2))


def main(argv=None):
    import sys
    from .gpu_queue import gpu_slot,explicit_cuda
    values=sys.argv[1:] if argv is None else argv
    with gpu_slot(explicit_cuda(values)):
        return _main(values)


if __name__ == "__main__":
    main()
