# AVTENet แบบ feature fusion

[English และคำสั่งครบทุกขั้น](README.md) · [เตรียม environment/index](../README.th.md)

โค้ดแยก 4 stages ได้แก่ visual (factorized ViViT), audio (AST), audiovisual
(AV-HuBERT และ temporal head) และ fusion โดยใช้ dataset เดียวกับ FK
พร้อม policy `keep-official` ไม่ตัดต้นฉบับซ้ำข้าม split

**สถานะ:** AST และ AV-HuBERT ผ่านเทรน/validation หนึ่ง step บนคลิปจริงแล้ว
รวมถึงโหลด checkpoint และสร้าง feature cache; ตรวจ gradient ของ AV-HuBERT
ทั้ง backbone และ temporal head แล้วใน Linux runtime
ViViT แบบ factorized ยังติดลิงก์ checkpoint ต้นทาง HTTP 403 จึงยังไม่ถือว่า
ensemble ครบทั้งตัวพร้อมใช้งาน ห้ามแทนด้วย ViViT คนละสถาปัตยกรรมหรือ random weights

## โหลดและระบุ pretrained weights

```powershell
& $Python -m baselines.avtenet.acquire --output checkpoints/avtenet-assets-001
$Assets = Join-Path $Repo 'checkpoints/avtenet-assets-001/baseline-assets.local.json'
& $Python -m baselines.avtenet check --assets $Assets
```

คำสั่งเก็บรายการที่โหลดสำเร็จ แม้บางตัวโหลดไม่ได้ และคืน exit code 2 เมื่อมี
การดาวน์โหลดล้มเหลว AST เลือก `MIT/ast-finetuned-audioset-10-10-0.4593`
ส่วน AV-HuBERT ใช้ Base LRS3 clean pretraining iteration 5
ค่าเหล่านี้เป็นตัวเลือกที่ประกาศไว้ ไม่ได้อ้างว่าเป็นไฟล์ตรงกับผู้เขียน AVTENet

ViViT ต้องแปลงด้วย tensor mapping ที่ตรวจสอบแล้วและเทียบผลกับ Scenic ต้นทาง
ก่อนนำไปเทรน ใช้ `python -m baselines.avtenet.convert_vivit --help` ดูคำสั่ง
รายละเอียดไฟล์อ้างอิงและเกณฑ์ parity อยู่ในคู่มืออังกฤษ

## เทรน expert

```powershell
& $Python -m baselines.avtenet train --index "$AVTE/index" --dataset-root $DatasetRoot --assets $Assets --stage visual --output "$AVTE/visual" --device cpu --seed 7
& $Python -m baselines.avtenet train --index "$AVTE/index" --dataset-root $DatasetRoot --assets $Assets --stage audio --output "$AVTE/audio" --device cpu --seed 7
```

คำสั่ง visual รันได้หลังมี ViViT ที่ตรวจสอบครบแล้วเท่านั้น
ใช้หน้าต่าง 2 วินาที เลื่อนครั้งละ 1 วินาที และเพิ่มช่วงที่จบตรงปลายคลิป
ถ้าจำเป็น ViViT รับ 32 เฟรม ส่วน AV-HuBERT รับภาพปาก 50 เฟรมคู่เสียง 16 kHz
label ของช่วงนั้นพิจารณาการทับช่วงปลอมของ modality นั้น ค่า unknown ไม่เข้า loss
คลิปสั้นกว่า 2 วินาทีหรือสื่อไม่ครบจะถูกบันทึกไว้ใน coverage

Adam ใช้ learning rate ภาพ 1e-4, เสียง 1e-5, audiovisual/fusion 0.002
สูงสุด 50 epochs และ patience 5 ตาม validation loss
การ crop ด้วย MTCNN และวิธีรวมคะแนนเป็น adaptation ที่ระบุไว้ ไม่ใช่ pipeline
Kinetics/LRS3 ต้นฉบับทุกขั้น และรอบนี้ไม่มี stochastic augmentation

## AV-HuBERT ใช้ Linux แยก

ต้อง fine-tune backbone และ temporal head ใน process เดียวกัน
เตรียม Docker Desktop ให้ใช้ Linux backend แล้ว build:

```powershell
docker build --build-arg AVHUBERT_COMMIT=258fb50e155134eec2c4b49c2ae8de267075fd18 -t fk-avhubert:cpu-v1 baselines/avtenet/runtime
```

ทำ registry อีกไฟล์สำหรับ Linux ชื่อ `assets.docker.local.json` ไว้ใน `$RunRoot`
ตั้ง `repository` เป็น `/opt/av_hubert` และ path ของ checkpoint เป็น
`/workspace/...` หากอยู่ใน repo หรือ `/runs/...` หากอยู่ใน `$RunRoot`
ไม่เขียนทับ registry ของ Windows

```powershell
& ./baselines/avtenet/runtime/invoke-docker.ps1 -DatasetRoot $DatasetRoot -RunRoot $RunRoot -CommandArguments @('train', '--index', '/runs/avtenet-001/index', '--dataset-root', '/dataset', '--assets', '/runs/assets.docker.local.json', '--stage', 'audiovisual', '--output', '/runs/avtenet-001/audiovisual', '--device', 'cpu', '--seed', '7')
```

launcher อ่าน dataset/repo แบบ read-only และเขียนเฉพาะโฟลเดอร์ผล
image ตัวอย่างเป็น CPU สำหรับทดสอบระบบ การเทรน GPU ต้องเตรียม CUDA environment
ที่เข้ากับ legacy dependencies เพิ่ม หากมี WSL2 environment อยู่แล้วใช้
`runtime/invoke-wsl.ps1` โดยส่ง path ในรูปแบบ Linux

## Cache → fusion → evaluation

หลังเลือก `best.pt` ของทั้งสาม expert ให้ใช้ `cache-features` ทีละตัว
audio/visual รันใน environment หลัก ส่วน audiovisual รันใน Linux
ดูคำสั่งครบในคู่มืออังกฤษ จากนั้น:

```powershell
& $Python -m baselines.avtenet merge-features --visual "$AVTE/visual-features" --audio "$AVTE/audio-features" --audiovisual "$AVTE/audiovisual-features" --output "$AVTE/fusion-features"
& $Python -m baselines.avtenet train --index "$AVTE/index" --dataset-root $DatasetRoot --stage fusion --feature-cache "$AVTE/fusion-features" --output "$AVTE/fusion" --device cpu --seed 7
& $Python -m baselines.avtenet evaluate --index "$AVTE/index" --dataset-root $DatasetRoot --checkpoint "$AVTE/fusion/best.pt" --feature-cache "$AVTE/fusion-features" --split validation --output "$AVTE/evaluation" --device cpu
```

cache ผูกกับ checkpoint ทั้งสามตัว พร้อม checksum และจับคู่ video ID/เวลาของช่วง
fusion ใช้เฉพาะช่วงที่มีผลครบสาม expert แต่คลิปที่ประเมินไม่ได้ยังอยู่ในรายงาน
คะแนนวิดีโอเป็นค่าเฉลี่ย fake probability ของช่วงที่ใช้ได้ และ threshold 0.5
เพิ่ม `--compare-with <FK-report.json>` เพื่อเทียบด้วยคลิปเดียวกัน

ทุก stage เทรนต่อได้ด้วย `last.pt` ตามคู่มือส่วนกลาง แต่การสร้าง/รวม feature cache
ยังไม่รองรับ resume ถ้าขัดจังหวะให้ใช้ output ใหม่และเริ่มขั้นนั้นอีกครั้ง
