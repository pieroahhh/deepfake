# AVFakeNet กับ AV-Deepfake1M

[English](README.md) · [ติดตั้งและเตรียมดัชนี](../README.th.md)

โฟลเดอร์นี้รวม DST-Net, preprocessing และคำสั่งเทรน/ประเมินผล
import แบบเดิมยังใช้ได้ แต่การเทรน dataset ผ่านคำสั่งใหม่ใช้ trainer ที่รองรับ resume

## เทรนภาพและเสียงแยกกัน

ใช้ตัวแปรจากคู่มือส่วนกลาง ก่อนเทรนจริงควรทดสอบใน output ใหม่ โดยเพิ่ม
`--epochs 1 --max-steps 1` ซึ่งจะติดสถานะ integration-only ใน checkpoint และรายงาน

```powershell
& $Python -m baselines.avfakenet train --index "$AVFake/index" --dataset-root $DatasetRoot --stage visual --output "$AVFake/visual" --device cpu --seed 7
& $Python -m baselines.avfakenet train --index "$AVFake/index" --dataset-root $DatasetRoot --stage audio --output "$AVFake/audio" --device cpu --seed 7
```

classifier ทั้งสองเริ่มจาก scratch ใช้ AdamW, learning rate 0.001, batch 16,
label smoothing 0.1, สูงสุด 100 epochs และหยุดเมื่อ validation accuracy
ไม่ดีขึ้น 5 epochs ค่านอกเหนือรายละเอียดที่ยืนยันจาก paper เป็น reconstruction choices

ภาพใช้ MTCNN ติดตามใบหน้าและจัดแนวตา ตัวอย่าง 25 fps ขนาด 128×128
label แต่ละเฟรมดูจากเวลาต้นฉบับที่อยู่ในช่วงปลอม ไม่ใช้ label ปลอมของทั้งคลิป
เหมารวมทุกเฟรม เสียงใช้ทั้งช่วงคลิป 16 kHz แปลง Mel เป็นภาพ 175×175
และใช้ label ของเสียงโดยเฉพาะ ค่า label ที่ไม่ทราบจะถูกข้ามและบันทึกเหตุผล

หากใบหน้าหายหรือเสียงไม่ครอบคลุมช่วงที่ต้องใช้ จะบันทึก preparation failure
ไม่เติมเสียงเงียบเพื่อทำให้ดูเหมือนข้อมูลครบ รอบนี้ไม่เปิด stochastic augmentation
จึงต้องระบุว่าเป็นการปรับใช้กับ AV-Deepfake1M ไม่ใช่ผลทำซ้ำ paper แบบตรงทุกค่า

## ประเมินผล

```powershell
& $Python -m baselines.avfakenet evaluate --index "$AVFake/index" --dataset-root $DatasetRoot --visual-checkpoint "$AVFake/visual/best.pt" --audio-checkpoint "$AVFake/audio/best.pt" --split validation --output "$AVFake/evaluation" --device cpu
```

ภาพตัดสินด้วยเสียงข้างมากของเฟรม แล้ว OR กับผลจากเสียง
ถ้าเสียงข้างมากเสมอหรือ input ไม่ครบ จะเป็น `UNASSESSED`
ไม่มี continuous fused score จึงไม่รายงาน AUC ของ hard OR
เพิ่ม `--compare-with <FK-report.json>` เพื่อเทียบคลิปที่ทั้งสองโมเดลประเมินได้

## เทรนต่อ

```powershell
& $Python -m baselines.avfakenet train --index "$AVFake/index" --dataset-root $DatasetRoot --stage visual --output "$AVFake/visual-resumed" --resume "$AVFake/visual/last.pt" --epochs 100 --device cpu --seed 7
```

ต้องมี `best.pt` คู่กับ `last.pt` ใช้ config/seed/index เดิม และ output ใหม่
ถ้าเดิมใช้ `--max-steps` ต้องใช้ค่าเดิมด้วย สำหรับการทดลองจริงให้เริ่ม run ใหม่
แทนการนำ checkpoint ที่จำกัด steps มาปะปนกับ full run

ทดสอบกับคลิปจริงจาก train/validation แล้ว: MTCNN, เทรนหนึ่ง step ทั้งสอง branch,
บันทึก/โหลด checkpoint และประเมินระดับวิดีโอทำงานได้
ชุดเล็กนี้เป็นการทดสอบโค้ด ไม่ใช้ยืนยัน accuracy ของโมเดล
