# โมเดลเปรียบเทียบที่แยกจาก FK

[English](README.md) · [AVFakeNet](avfakenet/README.th.md) · [AVTENet](avtenet/README.th.md)

สองโฟลเดอร์นี้เป็นโค้ดที่สร้างตามแนวทางของ paper และปรับใช้กับ AV-Deepfake1M
ไม่ใช่โค้ดต้นฉบับของผู้เขียนที่ตรวจยืนยันแล้ว และยังไม่ใช่โมเดลตรวจ deepfake
ที่เทรนเสร็จ AVFakeNet เริ่มเทรน classifier ใหม่ ส่วน AVTENet ต้องใช้ pretrained
backbone จริง แล้วเทรน expert และ fusion เพิ่ม

ใช้ population เดียวกับ FK และรักษา `keep-official` ตามที่เลือกไว้ ไม่ตัดกลุ่ม
ต้นฉบับซ้ำออก แต่แต่ละโมเดลเตรียม input ของตนเอง ใช้ cache ของ FK แทนกันไม่ได้
ค่า label ที่ไม่ทราบจะไม่ถูกตีความว่าเป็นคลิปจริง

## ติดตั้งบนเครื่องที่ pull โค้ดมา

ติดตั้ง Git, Python 3.11 และ uv ก่อน เปิด PowerShell ใน repo แล้วรัน:

```powershell
$Repo = (git rev-parse --show-toplevel).Trim()
Set-Location -LiteralPath $Repo
$DatasetRoot = Read-Host 'โฟลเดอร์ dataset ที่แตกไฟล์แล้ว'
$RunRoot = Read-Host 'โฟลเดอร์เก็บผลทดลอง'
$Population = Read-Host 'path ของ manifest.json ที่ prepare-avdeepfake1m ทำเสร็จแล้ว'
$env:HF_HOME = Join-Path $Repo '.cache/huggingface-comparators'
uv --cache-dir .cache/uv-baselines venv --python 3.11 .venv-baselines
uv --cache-dir .cache/uv-baselines pip install --python .venv-baselines/Scripts/python.exe -r baselines/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
uv --cache-dir .cache/uv-baselines pip install --python .venv-baselines/Scripts/python.exe --no-deps -e .
$Python = Join-Path $Repo '.venv-baselines/Scripts/python.exe'
& $Python -m baselines.avfakenet check
```

ตัวอย่างติดตั้งรุ่น CPU สำหรับทดสอบระบบ เครื่องที่จะเทรน GPU ต้องใช้ Torch,
Torchvision และ Torchaudio รุ่น CUDA ที่เข้ากันได้ใน environment แยก
ไม่ติดตั้งทับ `.venv` ของ FK ที่กำลังเตรียม dataset

คำสั่ง `check` ตรวจ dependency และข้อมูล assets ไม่ได้แปลว่าเทรนเสร็จหรือมี
accuracy ตาม paper แล้ว dataset, pretrained weights และไฟล์ผลทดลองไม่ติดไปกับ Git

## เตรียมดัชนีของแต่ละโมเดล

รอให้ `prepare-avdeepfake1m` สำเร็จและมี `manifest.json` กับ `audit.json` ก่อน
เพื่อนสามารถคัดลอกสองไฟล์นี้ไปใช้ได้ โดยระบุ dataset root ของเครื่องตัวเอง

```powershell
$AVFake = Join-Path $RunRoot 'avfakenet-001'
$AVTE = Join-Path $RunRoot 'avtenet-001'
& $Python -m baselines.avfakenet prepare --manifest $Population --output "$AVFake/index"
& $Python -m baselines.avtenet prepare --manifest $Population --output "$AVTE/index"
```

ขั้นนี้สร้างดัชนี SQLite ยังไม่แตกวิดีโอทั้งหมดเป็นภาพ และไม่คัดลอก dataset
ถ้าไม่ใส่ `--limit-per-split` จะใช้ทุกรายการใน manifest
หากต้องการทดสอบระบบก่อน ให้เพิ่ม `--limit-per-split 8` และใช้ output ใหม่
ชุดทดสอบเล็กจะถูกระบุไว้ในรายงาน ไม่ถือเป็นผลเทรน dataset ทั้งหมด

จากนั้นทำตามคู่มือของแต่ละโมเดล ตรวจ `$LASTEXITCODE` หลังแต่ละคำสั่ง
ก่อนเริ่มขั้นต่อไป output ต้องเป็นโฟลเดอร์ใหม่เพื่อไม่เขียนทับผลเก่า

## หยุดแล้วเทรนต่อ

- `last.pt` เก็บสถานะสำหรับเทรนต่อ รวม optimizer และ random states
- `best.pt` เก็บ checkpoint ที่ดีที่สุดตามเกณฑ์ validation ของ stage
- `run.json` บันทึก config, dataset hash, weights, policy และ seed
- `metrics.json` บันทึกผลแต่ละ epoch
- `preparation-events.jsonl` บันทึกสื่อที่อ่านไม่ได้และ label ที่ไม่ทราบ

ใช้ `--resume <โฟลเดอร์เดิม>/last.pt` แล้วเลือก output ใหม่
ต้องเก็บ `best.pt` ไว้ข้าง `last.pt` และใช้ config, assets, index, seed เดิม
`--epochs` หมายถึงจำนวน epoch รวม ไม่ใช่จำนวนที่จะเพิ่ม
progress แสดงทุก 100 steps และ checkpoint บันทึกหลังจบแต่ละ epoch

ย้าย index และ checkpoint ไปเครื่องอื่นได้ แต่สร้าง environment ใหม่ในเครื่องนั้น
ไม่คัดลอก `.venv` ไปใช้ข้ามเครื่อง Exact resume ทดสอบเฉพาะ CPU/runtime เดียวกัน

## อ่านผลและเปรียบเทียบ

ผลอยู่ใน `predictions.jsonl` และ `report.json` รวมคลิปที่ประเมินไม่ได้ไว้ด้วย
ดู coverage ควบคู่กับ F1, precision, recall และ balanced accuracy
AVFakeNet ใช้ hard OR จึงไม่มีคะแนนต่อเนื่องสำหรับ AUC
AVTENet รายงาน AUC ได้เมื่อมีคะแนนและ label ครบสองคลาส

เพิ่ม `--compare-with <รายงาน evaluate-fk.json>` เพื่อเทียบด้วย video ID เดียวกัน
ระบบรายงาน coverage แยก และผลบนชุดที่ทั้งสองโมเดลประเมินได้
test list ที่ไม่มี label ไม่สามารถใช้คำนวณ accuracy ได้

ขณะพัฒนา ลิงก์ checkpoint ViViT แบบ factorized จากต้นทางตอบ HTTP 403
AVTENet ต้องผ่านการโหลดและตรวจ conversion parity ของตัวนี้ก่อน จึงจะถือว่าพร้อม
ทั้ง ensemble ดูสถานะการทดสอบจริงล่าสุดใน [HANDOFF](../docs/handoffs/HANDOFF.md)
