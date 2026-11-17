# ทดลอง FK และ AVFakeNet ภายในงบเวลา 2 วัน

ขอบเขตที่เลือก: ใช้ AV-Deepfake1M ชุดย่อยเดียวกัน 3,000 train / 500 validation,
seed 7 และ GPU ของเครื่องนี้ ไม่รัน AVTENet เนื่องจากยังไม่มี pretrained
factorized ViViT ที่ผ่านการตรวจสอบ การทดลองนี้เป็นผลพัฒนาบน subset/validation
ไม่ใช่ผล benchmark เต็มชุดหรือผล test อิสระ

## สิ่งที่ปรับ

- Environment ใหม่ `.venv-gpu`: Torch 2.2.2 + CUDA 12.1, ใช้ RTX 3070 8 GB
  ไม่ติดตั้งทับ `.venv` หรือ `.venv-baselines`
- สุ่มแบบแบ่งชั้นตาม official split และชนิดการแก้ไขภาพ/เสียง รักษาสัดส่วนเดิม
  โดยเลือก seeded SHA256 ต่ำสุดต่อกลุ่ม รายชื่อถูกตรึงไว้ใน manifest เดียวกัน
- รักษา `keep-official` ไม่ย้าย split และไม่ตัดกลุ่มต้นฉบับซ้ำ
  subset นี้ยังมี 5 source families ร่วม train/validation ตาม audit
- FK 5 epochs, validation calibration ตาม workflow เดิม
- AVFakeNet 3 epochs ต่อ branch, batch size 8 เพื่อเผื่อ VRAM
- AVFakeNet เก็บภาพ crop uint8 แบบ lossless และ Mel input ไว้ใช้ซ้ำข้าม epochs
  และสอง branches แทนการหาใบหน้าซ้ำ ไม่ลด 25 fps หรือขนาด input
- รายงาน coverage ของแต่ละโมเดล และผลบน video IDs ที่ทั้งสองประเมินได้

จำนวนที่สุ่มจริง: train visual_modified 750, audio_modified 750,
both_modified 749, real 751; validation visual_modified 125,
audio_modified 125, both_modified 126, real 124

## เริ่มและติดตาม

พัก `fit-quality` เต็มชุดเก่าด้วย Ctrl+C ก่อน เพื่อไม่แย่ง CPU/disk
ไม่ต้องลบ `D:\Deepfake-Detection-runs\full-official-001\quality.json.work`
ใช้ output ใหม่สำหรับ subset นี้

จาก repo root หลังติดตั้ง GPU environment แล้ว:

```powershell
& '.\.venv-gpu\Scripts\python.exe' scripts/run_two_day.py `
    --manifest '.\runs\two-day-001\population\manifest.json' `
    --dataset-root 'D:\AV-Deepfake1M' `
    --assets '.\checkpoints\paper-alignment-001\assets.local.json' `
    --output '.\runs\two-day-001\experiment'
```

คำสั่งรันตามลำดับ fit-quality → cache-fk → train/calibrate FK → evaluate FK →
AVFakeNet visual/audio → evaluate/compare พร้อมแยก log และไฟล์สถานะ
รันซ้ำคำสั่งเดิมหลังแก้ error ได้ ระบบใช้ quality/cache ที่บันทึกและ resume
training จาก `last.pt` ไป attempt ใหม่ ขั้นที่ไม่มี epoch สำเร็จจะเริ่ม attempt ใหม่
หาก train FK เสร็จแต่ calibration ล้มเหลว ระบบจะหยุดเพื่อไม่เทรนซ้ำโดยเงียบ

```powershell
Get-Content '.\runs\two-day-001\experiment\status.json'
Get-Content '.\runs\two-day-001\experiment\logs\fk-quality.log' -Tail 20 -Wait
```

ดูชื่อ log ของ stage ปัจจุบันจาก `status.json` ก่อน
ผลสุดท้ายอยู่ใน `comparison.json`, `fk/validation.json` และ
`avfakenet/evaluation/report.json` พร้อม checkpoint ในโฟลเดอร์แต่ละโมเดล
แม้ใช้ GPU แล้ว เวลา 48 ชั่วโมงยังต้องตรวจจาก throughput จริง
ไม่มีการรับประกันว่าเทรนถึงจำนวน epochs ที่ตั้งไว้เท่ากับโมเดล converged

## การติดตั้งเครื่องอื่น

ติดตั้ง Python 3.11 และ uv แล้วรัน:

```powershell
uv --cache-dir .cache/uv-gpu venv --python 3.11 .venv-gpu
uv --cache-dir .cache/uv-gpu pip install --python .venv-gpu/Scripts/python.exe -r baselines/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match
uv --cache-dir .cache/uv-gpu pip install --python .venv-gpu/Scripts/python.exe --no-deps -e .
& '.\.venv-gpu\Scripts\python.exe' -c 'import torch; print(torch.cuda.is_available())'
```

ต้องได้ True และมี CUDA-capable GPU/driver ที่รองรับ ตำแหน่ง assets ต้องเป็นของ
เครื่องนั้นเอง แจก manifest กับ audit ได้ แต่ environment/weights/cache ไม่ไปกับ Git
อย่าเปลี่ยนโค้ดหรือ environment ระหว่าง run เพราะ resume ตรวจ provenance
หากนำ manifest เต็มจากเครื่องอื่นมา ต้องคัดลอก audit ที่ตรวจ hash ตรงกันด้วย
