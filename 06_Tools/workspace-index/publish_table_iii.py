"""Publish coverage-aware Table III tables and a figure from retained measurements."""

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
import importlib.util
import json
import os
import statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = _fk_paths.workspace_root(__file__)
OUT = _fk_paths.relocate((ROOT / 'Deepfake-Detection/runs/table-iii-cost-001'), __file__)
spec = importlib.util.spec_from_file_location('tableiii_data', _fk_paths.relocate((Path(__file__)), __file__).with_name('table_iii_profiler.py'))
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
read, write, digest = helper.read, helper.write, helper.digest
GROUPS = {'pre': 'preprocessing', 'feature': 'feature_extraction', 'evidence': 'fusion_evidence',
          'aggregation': 'aggregation', 'record': 'forensic_record', 'io': 'decode_io'}


def stage_rows(stage):
    return {x['row']['video_id']: x for p in sorted((_fk_paths.relocate((OUT / stage), __file__)).glob('[0-9][0-9].json')) for x in [read(p)]}


def measured(row):
    samples = row['samples'][1:]
    values = {'seconds': statistics.median(s['seconds'] for s in samples), 'mean_seconds': statistics.mean(s['seconds'] for s in samples)}
    for prefix, label in GROUPS.items():
        values[label] = statistics.median(sum(x['self_seconds'] for k, x in s['timers'].items() if k.startswith(prefix + '.')) for s in samples)
        values['mean_' + label] = statistics.mean(sum(x['self_seconds'] for k, x in s['timers'].items() if k.startswith(prefix + '.')) for s in samples)
    values['other'] = max(0., values['seconds'] - sum(values[k] for k in GROUPS.values()))
    values['mean_other'] = max(0., values['mean_seconds'] - sum(values['mean_' + k] for k in GROUPS.values()))
    values['sampled_peak_rss_bytes'] = max(s['sampled_peak_rss_bytes'] for s in samples)
    values['cuda_peak_allocated_bytes'] = max(s['cuda_peak_allocated_bytes'] for s in samples)
    values['cuda_peak_reserved_bytes'] = max(s['cuda_peak_reserved_bytes'] for s in samples)
    values['assessed'] = samples[0]['result']['status'] == 'ASSESSED'
    values['duration'] = row['row']['duration']
    values['seconds_per_source_second'] = values['seconds'] / values['duration']
    return values


def main():
    def seconds(value):
        if value == 0:
            return '0†'
        return '<0.0001' if value < .0001 else '%.4f' % value
    stages = {k: stage_rows(k) for k in ('fk', 'avfakenet', 'audio', 'visual', 'audiovisual', 'fusion')}
    models = {'FK': {k: measured(v) for k, v in stages['fk'].items()},
              'AVFakeNet adaptation': {k: measured(v) for k, v in stages['avfakenet'].items()}}
    joint = set.intersection(*(set(stages[k]) for k in ('audio', 'visual', 'audiovisual', 'fusion')))
    av = {}
    for video in sorted(joint):
        parts = [measured(stages[k][video]) for k in ('audio', 'visual', 'audiovisual', 'fusion')]
        av[video] = {key: sum(p[key] for p in parts) for key in ['seconds', *GROUPS.values(), 'other']}
        av[video]['seconds'] = statistics.median(sum(stages[k][video]['samples'][rep]['seconds'] for k in ('audio', 'visual', 'audiovisual', 'fusion')) for rep in (1, 2, 3))
        for prefix, label in GROUPS.items():
            av[video][label] = statistics.median(sum(v['self_seconds'] for k in ('audio', 'visual', 'audiovisual', 'fusion')
                for name, v in stages[k][video]['samples'][rep]['timers'].items() if name.startswith(prefix + '.')) for rep in (1, 2, 3))
        for key in ('mean_seconds', *('mean_' + k for k in GROUPS.values()), 'mean_other'):
            av[video][key] = sum(p[key] for p in parts)
        av[video].update({key: max(p[key] for p in parts) for key in ('sampled_peak_rss_bytes', 'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes')})
        av[video].update(assessed=parts[-1]['assessed'], duration=parts[-1]['duration'],
                         seconds_per_source_second=av[video]['seconds'] / parts[-1]['duration'])
    models['AVTENet adaptation (staged sum)'] = av
    common = set.intersection(*(set(k for k, v in rows.items() if v['assessed']) for rows in models.values()))
    table = []
    for name, rows in models.items():
        for population in ('all_attempts', 'common_assessed'):
            selected = list(rows.values()) if population == 'all_attempts' else [rows[k] for k in sorted(common)]
            if not selected:
                continue
            table.append({'model': name, 'population': population, 'attempted': len(selected), 'assessed': sum(x['assessed'] for x in selected),
                          **{key: statistics.median(r[key] for r in selected) for key in ['seconds', *GROUPS.values(), 'seconds_per_source_second']},
                          **{key: max(r[key] for r in selected) for key in ('sampled_peak_rss_bytes', 'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes')}})
    with (_fk_paths.relocate((OUT / 'table-iii-comparison.csv'), __file__)).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]) if table else ['model', 'population'])
        writer.writeheader()
        writer.writerows(table)
    components = []
    for stage, rows in stages.items():
        labels = sorted(set(label for row in rows.values() for sample in row['samples'][1:] for label in sample['timers']))
        for label in labels:
            clip_times, calls = [], []
            for row in rows.values():
                clip_times.append(statistics.median(s['timers'].get(label, {}).get('self_seconds', 0.) for s in row['samples'][1:]))
                calls.append(statistics.median(s['timers'].get(label, {}).get('calls', 0) for s in row['samples'][1:]))
            components.append({'stage': stage, 'component': label, 'clips': len(rows), 'median_self_seconds_per_clip': statistics.median(clip_times), 'median_calls_per_clip': statistics.median(calls)})
    with (_fk_paths.relocate((OUT / 'table-iii-components.csv'), __file__)).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['stage', 'component', 'clips', 'median_self_seconds_per_clip', 'median_calls_per_clip'])
        writer.writeheader()
        writer.writerows(components)
    write(_fk_paths.relocate((OUT / 'table-iii-models.json'), __file__), {'models': models, 'common_assessed_ids': sorted(common), 'comparison': table, 'components': components})
    now = datetime.now(timezone(timedelta(hours=7)))
    protocol = read(_fk_paths.relocate((OUT / 'protocol.json'), __file__))
    all_done = all((_fk_paths.relocate((OUT / k / 'complete.json'), __file__)).exists() for k in stages)
    lines = ['# Table III: ผลวัดต้นทุนเพิ่มเติม', '',
             'จัดทำเมื่อ ' + now.isoformat(timespec='seconds') + ' (Asia/Bangkok)', '',
             '**ไม่มีการเทรนเพิ่ม** ใช้ checkpoint, threshold และ profiling manifest เดิม 12 คลิป หนึ่งรอบ warmup และสามรอบจับเวลาต่อคลิปต่อ stage ผลเดิมถูกเก็บไว้ตามเดิม', '',
             'AVFakeNet เป็น local paper-guided reconstruction พร้อม normalization/recovery ที่บันทึกไว้; AVTENet ใช้ joint-attention Google/Hugging Face ViViT adaptation ร่วมกับ AST และ native AV-HuBERT ตัวเลขนี้จึงไม่ใช่ต้นทุนที่ยืนยันกับ author implementation ต้นฉบับ', '',
             '## สถานะการวัด', '', '| Stage | คลิปที่วัดเสร็จ |', '|---|---:|']
    lines += ['| %s | %d/12 |' % (k, len(v)) for k, v in stages.items()]
    lines += ['', '## ตารางต้นทุนตามคอลัมน์ Table III', '',
              'หน่วยวินาทีต่อคลิป: median ของเวลาแต่ละคลิปหลังตัด warmup รวม attempt ที่ UNASSESSED ด้วย ตัวเลขแต่ละคอลัมน์เป็น median แยกกันจึงไม่จำเป็นต้องบวกได้เท่ากับ median ของ total', '',
              '| วิธี | ประเมินสำเร็จ / attempt | Preprocessing | Feature extraction | Fusion / evidence | Aggregation | Forensic record | Decode / I/O | Total |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in table:
        if row['population'] != 'all_attempts':
            continue
        fusion = 'รวมกับ aggregation' if row['model'].startswith('AVFakeNet') else seconds(row['fusion_evidence'])
        forensic = seconds(row['forensic_record']) if row['model'] == 'FK' else 'N/A'
        lines.append('| %s | %d/%d | %s | %s | %s | %s | %s | %s | %s |' %
                     (row['model'], row['assessed'], row['attempted'], seconds(row['preprocessing']), seconds(row['feature_extraction']), fusion,
                      seconds(row['aggregation']), forensic, seconds(row['decode_io']), seconds(row['seconds'])))
    lines += ['', '† AVFakeNet ไม่ได้รัน classifier/aggregation ใน 7 จาก 12 attempt เนื่องจาก preparation ไม่ผ่าน ดังนั้น median ของเวลา feature/aggregation เมื่อรวมทุก attempt จึงเป็นศูนย์ ไม่ใช่ต้นทุนโมเดลศูนย์วินาที ให้ดูตารางคลิปร่วมด้านล่าง ส่วนค่าบวกที่เล็กกว่า 0.0001 s แสดงเป็น <0.0001 และเก็บความละเอียดเต็มใน CSV', '',
              '**AVTENet เป็นผลบวกของ stage ที่วัดแยกกัน** บน Windows และ Linux ไม่ใช่ wall time ของ pipeline ทั้งชุดที่รันใน process เดียว ส่วน FK total รอบนี้รวมการบันทึก artifact/record จริงแล้ว จึงห้ามแทนที่หรือปะปนกับเวลา FK เดิมที่มีขอบเขตแคบกว่า', '',
              '## เทียบเวลาเฉพาะคลิปที่ประเมินสำเร็จร่วมกัน', '',
              'มี %d คลิป ซึ่งเป็น subset ที่ถูกเลือกโดยความพร้อมของทั้งสามวิธี ไม่ขจัด missing-data bias' % len(common), '',
              '| วิธี | คลิปร่วม | Preprocessing | Feature | Fusion / evidence | Aggregation | Forensic | Decode / I/O | Total (s) | s / source-second |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in table:
        if row['population'] == 'common_assessed':
            fusion = 'รวมกับ aggregation' if row['model'].startswith('AVFakeNet') else seconds(row['fusion_evidence'])
            forensic = seconds(row['forensic_record']) if row['model'] == 'FK' else 'N/A'
            lines.append('| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s |' %
                         (row['model'], row['attempted'], seconds(row['preprocessing']), seconds(row['feature_extraction']), fusion,
                          seconds(row['aggregation']), forensic, seconds(row['decode_io']), seconds(row['seconds']), seconds(row['seconds_per_source_second'])))
    lines += ['', '## Memory', '', '| วิธี | Sampled peak RSS (MiB) | Peak allocated VRAM (MiB) | Peak reserved VRAM (MiB) |', '|---|---:|---:|---:|']
    for row in table:
        if row['population'] == 'all_attempts':
            lines.append('| %s | %.1f | %.1f | %.1f |' % (row['model'], row['sampled_peak_rss_bytes'] / 2**20,
                         row['cuda_peak_allocated_bytes'] / 2**20, row['cuda_peak_reserved_bytes'] / 2**20))
    lines += ['', 'RSS เป็นค่าสูงสุดจากการสุ่มอ่านทุก 20 ms ร่วมกับ snapshot ตอนจบการวัด/ตรวจ parity ไม่ใช่ peak ณ ทุกขณะ และรวม process ของเครื่องมือ/ข้อมูลอ้างอิงด้วย สำหรับ AVTENet แสดงค่าสูงสุดจาก stage ที่รันแยกกัน ไม่ใช่ memory เมื่อโหลด expert ทั้งสามพร้อมกัน', '',
              '## ส่วนที่ครอบคลุมเพิ่มแล้ว', '',
              '- Preprocessing: detector, tracking/bookkeeping, shared alignment, face crop, lip crop, VAD, Mel/filterbank และ input transform ตาม stage ที่ใช้จริง',
              '- Feature extraction: EfficientNet, Wav2Vec2, GRU, pooling/projection, DST visual/audio, AST, ViViT, AV-HuBERT และ MS-TCN พร้อม pooling',
              '- Evidence/fusion: โมดูล projection/discrepancy/reliability/head ของ FK และ fusion head ของ AVTENet; แยก aggregation ของ FK/AVTENet เพิ่ม',
              '- Forensic: source-to-record replay เต็มเส้นทาง รวม source/working copy, artifact persistence/hashes, serialization, commitment และ verification โดยใช้ implementation เดิม',
              '- มีจำนวน call และเวลาย่อยใน table-iii-components.csv พร้อมผลทุกคลิปทุก repeat ใน JSON ของแต่ละ stage', '',
              '## ส่วนที่ยังต้องระบุว่าเป็นขอบเขตรวม', '',
              '- FK quality อยู่ใน quality/window residual ซึ่งมี bookkeeping ร่วมอยู่; attention/tensor operations และ validation บางส่วนอยู่ใน evidence residual',
              '- AVFakeNet majority vote กับ hard OR ยังจับรวมกัน; ไม่รายงานว่า OR ใช้ศูนย์วินาที',
              '- งาน orchestration, การแปลงข้อมูล และ argmax บางส่วนอยู่ใน Other/overhead ของกราฟ ไม่ได้สมมุติว่าเวลาคอลัมน์ต่าง ๆ ครอบคลุมทุกคำสั่งโดยไม่มีส่วนเหลือ',
              '- shared face/lip alignment เป็นงานร่วม ไม่กระจายเวลาซ้ำให้ทั้ง detector และ lip crop',
              '- hash ที่เกิดจริงมีการอ่าน source ซ้ำในแต่ละ window และตรวจ derived evidence หลายครั้ง จึงไม่อ้างว่าจำนวน call เท่ากับสูตร source hash สองครั้งโดยอัตโนมัติ',
              '- synchronized timers และ tracing มี overhead และปิด decode prefetch เฉพาะ profiler; raw inputs/intervals/weights เดิมต้องผ่าน parity ก่อนเก็บผล ใช้ profiling เดิมสำหรับเวลา optimized pipeline เดิม',
              '- AVTENet รวม preprocessing ของ expert ที่รันแยกกัน ซึ่งอาจตรวจใบหน้าซ้ำ ไม่ใช่การพิสูจน์ต้นทุนของ shared preprocessing เพียงรอบเดียวตามแบบสถาปัตยกรรมเชิงวิเคราะห์',
              '- ไม่รวม model loading และไม่ได้ควบคุม OS cold cache', '',
              'การตรวจ source hash ของ baseline ทำก่อนรอบจับเวลา ส่วน FK ยังรวม hash ที่ implementation เรียกจริงระหว่าง source-to-record ไว้ในต้นทุน forensic จึงต้องอ่านขอบเขตนี้คู่กับตัวเลข total', '',
              '## Training: ใช้หลักฐานเดิม ไม่มีการฝึกใหม่', '',
              '| วิธี | งบฝึกเดิม | ขอบเขตหลักฐาน |', '|---|---|---|',
              '| FK | 5 epochs | head บน frozen encoder cache; 2,912 train / 485 validation usable |',
              '| AVFakeNet adaptation | 1 pilot epoch ต่อ branch พร้อม tiny recovery diagnostics เดิม | branch metrics/identity เดิม |',
              '| AVTENet adaptation | 1 epoch ต่อ expert และ fusion | training metrics/step counts ของแต่ละ stage เดิม |', '',
              'ยังไม่มี controlled online-vs-cached training benchmark ที่จับเวลา forward/backward/optimizer/cache construction/cache I/O ภายใต้เงื่อนไขร่วมกัน จึงไม่สรุปว่าคอลัมน์ Training วัดยืนยันครบ และไม่แปลงเวลาฝึกจาก timestamp ของไฟล์ ตัวคูณ 3 ในตารางเป็น analytical approximation', '',
              'ข้อมูลที่อ่านจริงและ hash อยู่ใน historical-training.json; งบต่างกันและ adaptation ยังเป็นข้อจำกัดของการเปรียบเทียบ', '',
              '## การกู้ fusion และ integrity', '',
              'คิวเดิมหยุดที่ Windows ปฏิเสธการแทนที่ progress.json กู้เฉพาะ fusion ผ่าน writer ที่ retry/throttle โดยใช้ runner/checkpoint เดิม ผล fusion ที่มีอยู่แล้วคง hash เดิมทั้งหมด ดู fusion-progress-recovery.json และ fusion-recovery-complete.json ใน remaining-supplement-002', '',
              'ViViT/AV-HuBERT ใช้ execution-overlay.json เพื่อครอบ timer ที่เมธอดของ MTCNNDetector ตัวเดิมโดยคงชนิด stateless detector ตามเงื่อนไข cache ไว้ หลังตัวครอบ timer แบบเปลี่ยนชนิดไม่ผ่าน type check ใน attempt แรกของ ViViT ยังไม่มีคลิป ViViT ที่เผยแพร่ก่อนการแก้นี้ จำนวนคลิป/repeat/เกณฑ์ parity และ source ของตัวคำนวณเดิมไม่เปลี่ยน', '',
              'ผล profiling ทุกคลิปที่เผยแพร่ผ่าน assessment/decision หรือ expert-vector parity ด้วยเกณฑ์เดิม 5e-6 และมีการตรวจ hash ของไฟล์อ้างอิงหลัก ไม่มีการแทนที่ prediction หรือ checkpoint เดิม', '',
              '## ไฟล์ประกอบ', '',
              '- [ตารางเปรียบเทียบ CSV](table-iii-comparison.csv)',
              '- [ตารางองค์ประกอบ CSV](table-iii-components.csv)',
              '- [ผลรายคลิป JSON](table-iii-models.json)',
              '- [รายละเอียดภาษาอังกฤษและข้อจำกัด](REPORT.md)', '',
              'สถานะชุดวัด: ' + ('COMPLETE' if all_done else 'PARTIAL — แสดงเฉพาะผลที่เสร็จและผ่าน parity แล้ว'), '']
    (_fk_paths.relocate((OUT / 'REPORT_TH.md'), __file__)).write_text('\n'.join(lines), encoding='utf-8')
    if all_done:
        import shutil
        checked = {}
        original_protected = read(_fk_paths.relocate((ROOT / 'Deepfake-Detection/runs/remaining-supplement-002/identity.json'), __file__))['protected']
        for filename, expected in original_protected.items():
            checked[filename] = digest(_fk_paths.relocate((Path(filename)), __file__)) == expected
        plan_root = _fk_paths.relocate((ROOT / 'avh-a/code'), __file__)
        plan = read(_fk_paths.relocate((plan_root / 'runs/auto-next-001/test-plan.json'), __file__))
        for filename in plan['checkpoints'].values():
            checked[str(_fk_paths.relocate((plan_root / filename), __file__))] = digest(_fk_paths.relocate((plan_root / filename), __file__)) == plan['files'][filename]
        avfake_root = _fk_paths.relocate((ROOT / 'Deepfake-Detection/runs/avfakenet-one-epoch-001'), __file__)
        avfake_done = read(_fk_paths.relocate((avfake_root / 'complete.json'), __file__))
        for branch in ('visual', 'audio'):
            filename = branch + '/best.pt'
            checked[str(_fk_paths.relocate((avfake_root / filename), __file__))] = digest(_fk_paths.relocate((avfake_root / filename), __file__)) == avfake_done['files'][filename]
        if not all(checked.values()):
            raise RuntimeError('A protected input/checkpoint changed')
        write(_fk_paths.relocate((OUT / 'integrity-final.json'), __file__), {'checks': checked, 'all_passed': True, 'no_training': True})
        measurement_files = [p for stage in stages for p in (_fk_paths.relocate((OUT / stage), __file__)).glob('*.json')]
        write(_fk_paths.relocate((OUT / 'measurement-sha256.json'), __file__), {str(p.relative_to(OUT)): digest(p) for p in sorted(measurement_files)})
        sources = _fk_paths.relocate((OUT / 'sources'), __file__)
        sources.mkdir(exist_ok=True)
        for name in ('table_iii_profiler.py', 'table_iii_detector_compat.py', 'run_table_iii.py', 'finish_table_iii.py', 'recover_fusion_progress.py', 'publish_table_iii.py'):
            shutil.copy2(_fk_paths.relocate((Path(__file__)), __file__).with_name(name), _fk_paths.relocate((sources / name), __file__))
        plot(models)
        with (_fk_paths.relocate((OUT / 'REPORT_TH.md'), __file__)).open('a', encoding='utf-8') as stream:
            stream.write('\n![Component costs](component-costs.png)\n')
        finished = datetime.now(timezone(timedelta(hours=7)))
        write(_fk_paths.relocate((OUT / 'publication-complete.json'), __file__), {'finished_bangkok': finished.isoformat(timespec='seconds'),
              'within_one_hour_deadline': finished.timestamp() <= protocol['deadline_unix'], 'no_training': True,
              'files': {name: digest(_fk_paths.relocate((OUT / name), __file__)) for name in ('REPORT.md', 'REPORT_TH.md', 'results.json', 'table-iii-models.json', 'table-iii-comparison.csv', 'table-iii-components.csv', 'component-costs.png', 'integrity-final.json', 'measurement-sha256.json')},
              'source_files': {p.name: digest(p) for p in sources.glob('*.py')}})
    print(str(_fk_paths.relocate((OUT / 'REPORT_TH.md'), __file__)))


def plot(models):
    os.environ['MPLCONFIGDIR'] = str(_fk_paths.relocate((OUT / 'matplotlib'), __file__))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['FK\nfull persisted record', 'AVFakeNet\nadaptation', 'AVTENet\nstaged sum*']
    keys = ['decode_io', 'preprocessing', 'feature_extraction', 'fusion_evidence', 'aggregation', 'forensic_record', 'other']
    titles = ['Decode / I/O', 'Preparation', 'Features', 'Evidence / fusion', 'Aggregation', 'Forensic', 'Other / overhead']
    colors = ['#94a3b8', '#eab308', '#2563eb', '#7c3aed', '#059669', '#dc2626', '#cbd5e1']
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bottom = [0., 0., 0.]
    for key, title, color in zip(keys, titles, colors):
        heights = [statistics.mean(v['mean_' + key] for v in rows.values()) for rows in models.values()]
        ax.bar(range(3), heights, bottom=bottom, label=title, color=color, width=.58)
        bottom = [a + b for a, b in zip(bottom, heights)]
    ax.set_xticks(range(3), labels)
    ax.set_ylabel('Mean seconds per attempted clip (instrumented)')
    ax.set_title('Table III component accounting: 12 fixed clips, frozen models')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(0, max(bottom) * 1.15)
    for i, value in enumerate(bottom):
        ax.text(i, value + max(bottom) * .02, '%.2f s' % value, ha='center')
    fig.text(.08, .025, '* Sum of separately measured expert/fusion stages; not observed monolithic latency.\nIncludes unassessed attempts; FK includes artifact persistence. Training not benchmarked.', fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, 1))
    fig.savefig(_fk_paths.relocate((OUT / 'component-costs.png'), __file__), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
