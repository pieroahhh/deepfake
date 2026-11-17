# แบ่งงาน 3 เครื่อง: หนึ่งโมเดลต่อหนึ่งเครื่อง

แผนนี้ใช้แทนแผน 2 เครื่องเดิม ให้ C ทำ **AVTENet ทั้งหมด** เพื่อเริ่มขนานกับ
FK และ AVFakeNet ได้ทันที ไม่ต้องรอ A ทำ FK เสร็จก่อนเริ่ม AST/ViViT

| เครื่อง | สเปก | งาน |
| --- | --- | --- |
| A เครื่องปัจจุบัน | Ryzen 5600 / RTX 3070 8 GB / RAM 32 GB | FK quality → cache → train 5 epochs → calibrate → evaluate และเปรียบเทียบผลสุดท้าย |
| B โน้ตบุ๊ก | i7-13700HX / RTX 4060 8 GB / RAM 16 GB | AVFakeNet visual 3 epochs → audio 3 epochs, batch 8 → evaluate |
| C เครื่องใหม่ | Ryzen 5600X / RTX 4060 8 GB / RAM 32 GB | AVTENet AST → ViViT → AV-HuBERT → fusion อย่างละ 1 epoch → evaluate |

ทุกเครื่องใช้ **train 3,000 / validation 500 คลิปชุดเดียวกัน, seed 7** ตาม manifest
และ official split เดิม ไม่ได้ใช้ dataset เต็ม แต่ละโมเดลมี preprocessing และ coverage
ต่างกัน จึงรายงานทั้งคะแนนและจำนวนคลิปที่ประเมินได้ AVTENet เป็น Google joint-attention
ViViT adaptation ที่ตกลงไว้ ไม่ใช่การทำซ้ำสถาปัตยกรรมใน paper แบบตรงทุกส่วน

## ไฟล์ที่จะส่งให้แต่ละเครื่อง

- B: ส่งโฟลเดอร์ **Machine-B-bundle** ทั้งก้อน เปิด `START_HERE_SECOND_MACHINE.md`
- C: ส่งโฟลเดอร์ **Machine-C-bundle** ทั้งก้อน เปิด `START_HERE_THIRD_MACHINE.md`
- bundle เดิมชื่อ `Second-machine-bundle` เป็นแผนเก่า ไม่ใช้ launcher เก่ากับแผนใหม่นี้
- ไม่ต้องดาวน์โหลด AV-Deepfake1M เต็มอีกครั้ง ทั้งสอง bundle มี 3,500 คลิปเหมือนกัน
- B ไม่มี AVTENet weights/Docker image ส่วน C มี pretrained ทั้ง 3 expert และ CUDA image
- ต้องติดตั้ง environment ใหม่บนแต่ละเครื่อง ไม่คัดลอก `.venv` ข้ามเครื่อง

B ควรมีพื้นที่ว่างจริงอย่างน้อยประมาณ 80 GiB ส่วน C วางแผนสำรอง **200 GiB**
สำหรับ cache ภาพ/ปาก/เสียง, checkpoint, environment และ Docker เป็นการเผื่อพื้นที่
ไม่ใช่ขนาดสุดท้ายที่วัดได้ ความจุ SSD 416 GB ไม่ได้บอกพื้นที่ว่างจริง

## A: ใช้งานที่กำลังรันต่อ

งาน FK ที่กำลังทำอยู่ยังทำต่อใน directory เดิม ไม่สุ่มข้อมูลหรือเริ่ม quality ใหม่
เอา controller ชั้นนอกที่เคยต่อคิว AST/ViViT ออก โดยคง subprocess FK ไว้
ดูสถานะที่ `runs/two-day-001/job.json`, `three-machine-transition.json`
และ `experiment/status.json` **อย่าเปิด A ซ้ำ**

หากเริ่มใหม่หลัง process เดิมหยุดแล้ว ใช้จาก root repo:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\machine_a.ps1 -Stage FK -DatasetRoot 'D:\AV-Deepfake1M'
```

## B: AVFakeNet

ติดตั้ง NVIDIA driver, Python 3.11 และ uv จากนั้นเปิด PowerShell ใน `Machine-B-bundle\code`:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\machine_b.ps1 -Setup
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\machine_b.ps1 -Run
```

ไม่ต้องติดตั้ง Docker ดู progress ที่ `runs/worker-b/avfakenet/status.json`
ส่งโฟลเดอร์ `return-b-to-machine-a` กลับเมื่อสำเร็จ

## C: AVTENet ครบทั้งโมเดล

ติดตั้ง NVIDIA driver, Python 3.11, uv, Docker Desktop Linux/WSL2
เริ่มตั้ง memory allowance ของ Docker ประมาณ 12 GB บน RAM 32 GB
เปิด PowerShell ใน `Machine-C-bundle\code`:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\machine_c.ps1 -Setup
docker load --input '..\avhubert-cuda117-v1.tar'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\machine_c.ps1 -Run
```

หากไม่มี tar ให้ใช้ `machine_c.ps1 -BuildContainer` แทน docker load
launcher ตรวจ CUDA ทั้ง host และ container แล้วรันทีละ expert ไม่เทรนพร้อมกันบน GPU เดียว
ViViT ใช้ SDPA และ gradient checkpointing ตามที่ทดสอบไว้สำหรับ VRAM 8 GB
ดู progress ที่ `runs/worker-c/avtenet/status.json` แล้วติดตาม log ที่ระบุ
เมื่อสำเร็จส่ง `return-c-to-machine-a` กลับ A โดย C ทำ fusion/evaluate เสร็จเองแล้ว

## ส่งผลกลับและเปรียบเทียบ

คัดลอก return ทั้ง B และ C มาที่ A รอ FK สำเร็จ แล้วปรับ path ให้ตรง:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\machine_a.ps1 -Stage Compare -ReturnedB 'E:\return-b-to-machine-a' -ReturnedC 'E:\return-c-to-machine-a'
```

ตรวจ role, manifest hash และ checksum ทุกไฟล์ก่อน import แล้วตรวจ validation IDs/labels
ของทั้งสามโมเดลให้ตรงกัน ผลที่ `runs/two-day-001/three-model-comparison.json`
มี coverage, pairwise และชุดที่ทั้งสามโมเดลประเมินได้ร่วมกัน

## หากหยุดกลางทาง

เก็บ `runs/worker-b` หรือ `runs/worker-c` ไว้และรันคำสั่งเดิมต่อหลังแก้สาเหตุ
training ใช้ checkpoint ของ epoch ที่เสร็จล่าสุด ส่วน epoch ที่ไม่เสร็จต้องทำซ้ำ
prepared cache ใช้ต่อได้ ไม่เปลี่ยนโค้ด/profile/dependency/path/dataset ระหว่าง resume
หาก feature-cache/merge/evaluation เขียนไม่ครบ ให้เก็บไว้ด้วยชื่อใหม่ก่อน retry
และทำเฉพาะเมื่อ process หยุดแล้ว เก็บทุกไฟล์ไว้จน A รับผลและเปรียบเทียบสำเร็จ

แผน 3 เครื่องช่วยให้ GPU ทำงานขนานกันมากขึ้น แต่ยังไม่มีเวลารัน subset เต็มบน B/C จริง
จึงไม่รับรองว่าจะจบใน 48 ชั่วโมง และงบ epoch ที่จำกัดไม่ได้รับรอง convergence
