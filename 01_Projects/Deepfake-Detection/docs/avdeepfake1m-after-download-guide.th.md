# AV-Deepfake1M: คู่มือหลังดาวน์โหลดสำหรับ Windows

**แนวทางปัจจุบันที่เลือก:** ใช้คลิป train/validation ครบตาม official splits พร้อม `--source-overlap-policy keep-official` เก็บทั้งกลุ่มต้นฉบับและเนื้อหาที่ซ้ำไว้แล้วรายงานจำนวน หลังตั้งค่าให้ทำหัวข้อ 8 โดยไม่ใส่เพดานคลิป หัวข้อ 7 เป็น pilot ทางเลือกเท่านั้น ใช้โฟลเดอร์ใหม่ `full-official-001` เพราะ pilot ที่เคยตัดข้อมูลเป็นคนละ population ส่วน test ที่ไม่มี label เก็บไว้สำหรับทำนาย ไม่สร้าง label ขึ้นมาเพื่อฝึก

คู่มือนี้อธิบายตั้งแต่ตรวจไฟล์ที่ดาวน์โหลด ฝึกโมเดลตรวจจับ FK ประเมินผล ไปจนถึงสร้างและตรวจสอบหลักฐานจากการทำนาย ใช้กับ **AV-Deepfake1M รุ่นปี 2024** และ branch **`feature/avdeepfake1m-dataset`** ของโปรเจกต์ ส่วน AV-Deepfake1M++ เป็นคนละรุ่นกับที่คู่มือนี้รองรับ

ทำตามทีละหัวข้อ โดยเลือกเฉพาะขั้นตอนที่ตรงกับเครื่องและข้อมูลของคุณ ไม่ควรนำทุกบล็อกมารวมแล้วรันครั้งเดียว คำสั่งทั้งหมดใช้ **Windows PowerShell** จากโฟลเดอร์หลักของ repository เว้นแต่ระบุไว้เป็นอย่างอื่น ชื่อคำสั่ง ตัวแปร ชื่อไฟล์ และข้อความในบล็อกโค้ดคงตามต้นฉบับเพื่อให้คัดลอกรันได้ตรงกัน

โค้ดผ่านการทดสอบซอฟต์แวร์และการทำงานด้วยข้อมูลสังเคราะห์แล้ว แต่ความเข้ากันได้กับข้อมูลทั้งรุ่น การฝึกเต็มชุด ประสิทธิภาพ GPU และความแม่นยำยังต้องตรวจด้วยข้อมูลจริง เริ่มจากชุดทดลองขนาดเล็ก (pilot) ก่อนขยายไปทั้ง dataset

[English version](avdeepfake1m-after-download-guide.md)

## สารบัญ

1. [ลำดับงานและความหมายของไฟล์](#step-1)
2. [กำหนดพาธและตรวจพื้นที่](#step-2)
3. [ตรวจและแตกไฟล์บีบอัด](#step-3)
4. [เตรียมโค้ดและสภาพแวดล้อม Python](#step-4)
5. [ตรวจ metadata และตัวอย่างวิดีโอ](#step-5)
6. [เตรียมโมเดลที่ฝึกไว้ล่วงหน้า](#step-6)
7. [รันชุดทดลองขนาดเล็ก](#step-7)
8. [ประเมินทรัพยากรและรันเต็มชุด](#step-8)
9. [ทำงานต่อเมื่อถูกขัดจังหวะ](#step-9)
10. [ประเมินและตีความผล](#step-10)
11. [รันโมเดลเปรียบเทียบและ ablation](#step-11)
12. [ทำนายและตรวจสอบหลักฐาน](#step-12)
13. [แก้ปัญหาที่พบบ่อย](#step-13)
14. [รายการตรวจสอบก่อนจบงาน](#step-14)

<a id="step-1"></a>

## 1. ลำดับงานและความหมายของไฟล์

ภาพรวมขั้นตอนด้านล่างคงชื่อคำสั่งตามโปรแกรม โดยเริ่มจากตรวจและแตกไฟล์ เตรียมข้อมูล สร้าง cache ฝึกและปรับค่าตัดสินด้วย validation แล้วจึงประเมินและตรวจหลักฐาน:

```text
Archive checks and extraction
  -> metadata/media checks + environment/assets checks
  -> prepare-avdeepfake1m: freeze the selected population
  -> fit-quality: learn normalization from training media only
  -> cache-fk: extract frozen visual/audio features
  -> train-fk --calibrate: train the detector and select on validation
  -> evaluate-fk: report predictions, metrics, and assessment coverage
  -> infer-fk + verify: create and verify source-bound evidence
```

| รายการ | ความหมาย |
| --- | --- |
| Dataset media | วิดีโอต้นฉบับของ dataset ให้เก็บไว้โดยไม่แก้ไข |
| Dataset metadata | ข้อมูลทางการที่ระบุชุด train/validation/test แหล่งที่มา ประเภทการปลอม และช่วงเวลาที่ปลอม |
| Population manifest | รายการวิดีโอที่เลือกใช้ แปลงเป็นรูปแบบของโปรเจกต์ พร้อม label กลุ่มแหล่งกำเนิด ความยาว และ hash ของไฟล์ |
| Pretrained assets | EfficientNet-B0, Wav2Vec2 Base, Silero VAD และ YuNet เป็นส่วนประกอบที่ต้องใช้ก่อนฝึกตัวตรวจจับ FK |
| Quality profile | ค่าปรับมาตรฐานคุณภาพภาพและเสียง ซึ่งเรียนรู้จากชุดฝึก |
| Feature cache | ผลลัพธ์จาก encoder ที่ตรึงน้ำหนักไว้ พร้อม label และข้อมูลตรวจสอบที่มา นำกลับมาใช้ฝึกส่วนหัวโมเดลได้ |
| Detector checkpoint | น้ำหนักส่วนหัว FK ที่ฝึกแล้ว พร้อมการตั้งค่า และค่าตัดสินที่เลือกหลัง calibration |

โปรเจกต์ใช้การแบ่งชุดตามผู้จัดทำ dataset การตรวจ source family หรือกลุ่มไฟล์ที่มีต้นฉบับร่วมกันไม่ได้ยืนยันด้วยตัวเองว่าบุคคลในแต่ละชุดไม่ซ้ำกัน ห้ามสุ่มย้ายวิดีโอข้าม train, validation และ test

<a id="step-2"></a>

## 2. กำหนดพาธและตรวจพื้นที่

### เริ่มบน Windows เครื่องใดก็ได้

ติดตั้ง Git แล้วเปิด PowerShell **ภายใน checkout ของคุณ** เช่น เปิดโฟลเดอร์ repository ใน VS Code แล้วเปิด terminal ใหม่ ถ้า clone/pull ไว้แล้วให้ข้ามบล็อก clone ต่อไปนี้ สำหรับเครื่องที่ยังไม่มีโค้ด ให้เลือกโฟลเดอร์แม่ที่มีอยู่จริง โดยปลายทาง `Deepfake-Detection` ต้องยังไม่มี:

```powershell
$CloneParent = Read-Host 'Existing parent folder for the clone (full path, without quotes)'
if (-not (Test-Path -LiteralPath $CloneParent -PathType Container)) { throw 'Parent folder does not exist.' }
$CloneTarget = Join-Path (Resolve-Path -LiteralPath $CloneParent).Path 'Deepfake-Detection'
git clone --branch feature/avdeepfake1m-dataset --single-branch `
    https://github.com/tomatoesws/Deepfake-Detection.git $CloneTarget
if ($LASTEXITCODE -ne 0) { throw 'Clone failed.' }
Set-Location -LiteralPath $CloneTarget
```

คู่มือใช้ Git หาตำแหน่ง repository อัตโนมัติ จึงรองรับชื่อผู้ใช้ ไดรฟ์ และตำแหน่ง checkout ที่ต่างกัน คู่มือนี้ยังใช้กับ Windows/PowerShell พาธ `.exe` และการติดตั้งเครื่องมือไม่ใช่ขั้นตอนสำหรับ Linux/macOS

### สิ่งที่ได้จาก pull และสิ่งที่ต้องเตรียมเพิ่ม

| ได้จาก Git | ต้องเตรียมแยกบนแต่ละเครื่อง |
| --- | --- |
| โค้ด configs, dependency lock และคู่มือ | Python `.venv`, FFmpeg/FFprobe, 7-Zip, วิดีโอ/metadata ของ dataset, ไฟล์โมเดล pretrained และ checkpoint/cache/ผลลัพธ์ที่ฝึกแล้ว |

แต่ละเครื่องต้องสร้าง Python environment ของตนเองและมีพาธ assets ที่ใช้ได้ ห้ามคัดลอก `.venv` จากอีกเครื่องมาใช้ตรง ๆ ผู้ใช้งานยังต้องมีสิทธิ์เข้าถึง dataset ตามเงื่อนไข หากต้องการทำการทดลองเดิมต่อ ต้องย้าย artifact ครบชุดแยกจาก Git และทำตามเงื่อนไข resume ในหัวข้อ 9 การ pull โค้ดอย่างเดียวไม่ทำให้งานเดิม resume ได้

### กำหนดพาธประจำเครื่องหนึ่งครั้งต่อ PowerShell session

บล็อกต่อไปจะถาม **พาธเต็มบนเครื่อง 3 ตำแหน่ง** โดยไม่ต้องครอบด้วยเครื่องหมายคำพูด เช่น `E:\Datasets\AV-Deepfake1M-download`, `E:\Datasets\AV-Deepfake1M` และ `E:\FK-runs` หากเก็บข้อมูลบน E: รองรับพาธที่มีช่องว่าง โฟลเดอร์แตกไฟล์/ผลลัพธ์ยังไม่มีก็ได้ หากแตก dataset แล้วให้ระบุพาธจริงและข้ามหัวข้อ 3 เก็บผลลัพธ์ไว้นอก repository เพื่อไม่ปะปนกับการเปลี่ยนโค้ด ฟังก์ชันในบล็อกจะช่วยหยุดเมื่อ Python หรือ 7-Zip คืนรหัสผิดพลาด:

```powershell
$ErrorActionPreference = 'Stop'
$Repo = git rev-parse --show-toplevel
if ($LASTEXITCODE -ne 0) { throw 'Open PowerShell inside the cloned repository first.' }
$Repo = (Resolve-Path -LiteralPath $Repo).Path
Set-Location -LiteralPath $Repo
if (-not (Test-Path -LiteralPath 'pyproject.toml') -or
    -not (Test-Path -LiteralPath 'src/deepfake_detection/cli.py')) {
    throw 'This is not the Deepfake-Detection repository root.'
}

function Read-LocalPath([string]$Prompt) {
    $value = (Read-Host $Prompt).Trim().Trim('"')
    if ([string]::IsNullOrWhiteSpace($value)) { throw 'A path is required.' }
    if ($value -notmatch '^[A-Za-z]:[\\/]' -and
        $value -notmatch '^\\\\[^\\]+\\[^\\]+') {
        throw 'Use a full drive or UNC path.'
    }
    [System.IO.Path]::GetFullPath($value)
}
$Archives = Read-LocalPath 'Downloaded archive folder (full path)'
$Dataset = Read-LocalPath 'Extracted dataset folder (full path)'
$RunRoot = Read-LocalPath 'Experiment output folder outside the repository (full path)'
$Pilot = Join-Path $RunRoot 'pilot-001'
$Full = Join-Path $RunRoot 'full-official-001'
$SevenZipCommand = Get-Command 7z.exe -ErrorAction SilentlyContinue
$SevenZip = if ($SevenZipCommand) { $SevenZipCommand.Source } else {
    Join-Path $env:ProgramFiles '7-Zip\7z.exe'
}
$fkPython = Join-Path $Repo '.venv\Scripts\python.exe'
$Device = 'auto'

# PowerShell 5 does not automatically stop on native nonzero exit codes.
function Invoke-Python {
    $input | & $fkPython @args
    if ($LASTEXITCODE -ne 0) { throw "Python failed with exit code $LASTEXITCODE" }
}
function Invoke-7Zip {
    & $SevenZip @args
    if ($LASTEXITCODE -ne 0) { throw "7-Zip failed or warned: exit code $LASTEXITCODE" }
}

New-Item -ItemType Directory -Path $RunRoot -Force | Out-Null
Get-PSDrive -PSProvider FileSystem | Select-Object Name, Root,
    @{Name='UsedGiB';Expression={[math]::Round($_.Used / 1GB, 1)}},
    @{Name='FreeGiB';Expression={[math]::Round($_.Free / 1GB, 1)}}
```

ไฟล์บีบอัดของรุ่นที่เคยตรวจมีขนาดรวมประมาณ 404 GB ตัวเลขนี้ใช้วางแผนเท่านั้น ไม่ใช่พื้นที่ว่างปัจจุบันหรือขนาดหลังแตกไฟล์ ต้องเตรียมพื้นที่สำหรับ **ไฟล์บีบอัด + วิดีโอที่แตกแล้ว + feature cache + ผลลัพธ์การรัน** ตรวจรายการใน archive และวัดจาก pilot บนเครื่องจริง พร้อมเผื่อไฟล์ชั่วคราวและการรันใหม่

หากพื้นที่จำกัด ให้ทำทีละ split ลบไฟล์บีบอัดของ split นั้นได้หลังแตกและตรวจผ่านแล้ว และต้องมั่นใจว่าดาวน์โหลดกลับมาได้ คู่มือนี้ไม่มีคำสั่งลบ archive อัตโนมัติ การฝึกที่มี label ต้องมีทั้ง train และ validation ส่วนวิดีโอ test ที่ไม่มี label เลื่อนไปเตรียมภายหลังได้

เก็บไฟล์ขนาดใหญ่บนไดรฟ์ที่คุณเลือก ส่วน Python environment อยู่ใน `.venv` ภายใน repository ของเครื่องนั้น กำหนด cache สำหรับดาวน์โหลดแพ็กเกจ/โมเดลครั้งถัดไปใน PowerShell หน้าต่างนี้ให้ไปอยู่ใต้ RunRoot ได้ดังนี้:

```powershell
$env:UV_CACHE_DIR = Join-Path $RunRoot 'download-caches\uv'
$env:HF_HOME = Join-Path $RunRoot 'download-caches\huggingface'
$env:TORCH_HOME = Join-Path $RunRoot 'download-caches\torch'
```

ตัวแปรเหล่านี้ไม่ย้ายไฟล์ที่ดาวน์โหลดไว้ก่อนหน้า

<a id="step-3"></a>

## 3. ตรวจและแตกไฟล์บีบอัด

ติดตั้ง [7-Zip จากเว็บไซต์ทางการ](https://7-zip.org/) ถ้ายังไม่มีไฟล์ตาม `$SevenZip` โดย [หน้า dataset ทางการ](https://huggingface.co/datasets/ControlNet/AV-Deepfake1M) ระบุให้ใช้ `7z` แตก archive หลายส่วน เริ่มจากไฟล์ `.zip.001`

เก็บทุกส่วนของแต่ละ split ไว้ด้วยกัน รุ่นที่เคยตรวจมีไฟล์สุดท้ายเป็น `train.zip.254`, `val.zip.020` และ `test.zip.112` ให้ยึดรายการไฟล์ของ revision ที่คุณดาวน์โหลดเป็นหลัก จำนวนไฟล์ตรงกันอย่างเดียวไม่ได้ยืนยันว่าไฟล์ครบและไม่เสีย

```powershell
if (-not (Test-Path -LiteralPath $SevenZip)) { throw 'Install 7-Zip first.' }
Get-ChildItem -LiteralPath (Join-Path $Archives 'train') -Filter 'train.zip.*' |
    Select-Object Name, Length

Invoke-7Zip t (Join-Path $Archives 'train\train.zip.001')
Invoke-7Zip t (Join-Path $Archives 'val\val.zip.001')
# Optional: needed only when you will use the blind test media.
Invoke-7Zip t (Join-Path $Archives 'test\test.zip.001')
```

ผลทดสอบต้องไม่มีข้อผิดพลาด CRC, missing volume หรือ unexpected end การตรวจอาจใช้เวลานานเพราะต้องอ่านเนื้อหา archive อย่าต่อไฟล์แต่ละส่วนด้วยตัวเอง และไม่ต้องสั่งแตกทุกหมายเลขแยกกัน บรรทัดตรวจ test ในตัวอย่างเป็นขั้นตอนเสริมเมื่อจะใช้วิดีโอ test

### เลือกโฟลเดอร์ปลายทางให้ถูกต้อง

ตรวจพาธภายใน archive ก่อน:

```powershell
Invoke-7Zip l (Join-Path $Archives 'train\train.zip.001')
Invoke-7Zip l (Join-Path $Archives 'val\val.zip.001')
```

ถ้าพาธภายในขึ้นต้นด้วย `train/` หรือ `val/` อยู่แล้ว ให้แตกลง `$Dataset`:

```powershell
Invoke-7Zip x (Join-Path $Archives 'train\train.zip.001') "-o$Dataset"
Invoke-7Zip x (Join-Path $Archives 'val\val.zip.001') "-o$Dataset"
```

ถ้าพาธภายในเริ่มด้วยรหัสวิดีโอหรือโฟลเดอร์ย่อยทันที และ **ไม่มีชื่อ split นำหน้า** ให้ใช้คำสั่งชุดนี้แทน:

```powershell
Invoke-7Zip x (Join-Path $Archives 'train\train.zip.001') "-o$(Join-Path $Dataset 'train')"
Invoke-7Zip x (Join-Path $Archives 'val\val.zip.001') "-o$(Join-Path $Dataset 'val')"
```

ใช้หลักเดียวกันกับ `test.zip.001` เมื่อต้องการใช้งาน เลือกวิธีแตกเพียงวิธีเดียวต่อ split เพื่อไม่ให้เกิดโฟลเดอร์ซ้อนอย่าง `train\train\...` หากแตกไฟล์ผิดพลาด ให้ตรวจพาธที่มีปัญหาก่อนสั่งเขียนทับไฟล์เดิม

คัดลอกไฟล์ metadata ทางการที่เป็น JSON ปกติจากโฟลเดอร์ดาวน์โหลด ตัวอย่างนี้จะตรวจ hash ก่อนหากปลายทางมีไฟล์ชื่อเดียวกันอยู่แล้ว:

```powershell
foreach ($name in @('train_metadata.json', 'val_metadata.json', 'test_files.txt')) {
    $source = Join-Path $Archives $name
    $target = Join-Path $Dataset $name
    if (Test-Path -LiteralPath $source) {
        if (Test-Path -LiteralPath $target) {
            if ((Get-FileHash -LiteralPath $source).Hash -ne (Get-FileHash -LiteralPath $target).Hash) {
                throw "Existing metadata differs: $target"
            }
        } else {
            Copy-Item -LiteralPath $source -Destination $target
        }
    }
}
```

ถ้าดาวน์โหลดมาเฉพาะ metadata แบบบีบอัด ให้ดูรายการภายในแล้วแตกออกมาเป็น JSON จริง อย่าเปลี่ยนนามสกุล ZIP เป็น `.json` หรือสรุปเนื้อหาภายในจากชื่อ archive เพียงอย่างเดียว

โครงสร้างที่ต้องได้มีดังนี้ โดยตำแหน่งวิดีโอต้องตรงกับ metadata ส่วน `test_files.txt` และ `test/` เป็นทางเลือกสำหรับงานฝึกและ validation:

```text
<your $Dataset folder>\
  train_metadata.json
  val_metadata.json
  train\
    <paths corresponding to train metadata>
  val\
    <paths corresponding to val metadata>
  test_files.txt                  # optional, unlabeled inventory
  test\                          # optional for train/validation work
    <paths corresponding to test_files.txt>
```

<a id="step-4"></a>

## 4. เตรียมโค้ดและสภาพแวดล้อม Python

### กรณีมี repository อยู่แล้ว

```powershell
Set-Location -LiteralPath $Repo
git status --short
if ($LASTEXITCODE -ne 0) { throw 'Git status failed.' }
git branch --show-current
if ($LASTEXITCODE -ne 0) { throw 'Cannot read the current branch.' }
```

branch ที่ใช้คือ `feature/avdeepfake1m-dataset` ก่อนเปลี่ยน branch หรือ pull ให้ commit งานที่ต้องการเก็บ หรือ stash ไว้ก่อน รวมถึงเก็บ stash เดิมไว้ หาก working tree ไม่มีงานค้าง จึงรัน:

```powershell
git fetch origin
if ($LASTEXITCODE -ne 0) { throw 'Fetch failed.' }
git switch feature/avdeepfake1m-dataset
if ($LASTEXITCODE -ne 0) { throw 'Branch switch failed.' }
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw 'Pull failed; inspect the divergence/error.' }
```

### Environment สำหรับเครื่องใหม่

ติดตั้ง [uv](https://docs.astral.sh/uv/getting-started/installation/) ก่อนตรวจเครื่องมือด้านล่าง หากยังไม่มีโค้ดให้ clone ตามหัวข้อ 2 ก่อน คำสั่งต่อจากนี้ถือว่าตรวจพบ `$Repo` แล้ว แต่ละเครื่องต้องสร้าง `.venv` ของตนเอง แม้จะ pull โค้ดของการทดลองเดิมมาแล้วก็ตาม:

```powershell
Set-Location -LiteralPath $Repo
Get-Command git, uv -ErrorAction Stop
```

ติดตั้ง [uv](https://docs.astral.sh/uv/getting-started/installation/) หากยังไม่มี คำสั่งต่อไปนี้ใช้สร้างหรือคืนสภาพแวดล้อม CPU ที่เคยตรวจสอบไว้ อย่าติดตั้ง CPU lock ทับสภาพแวดล้อม GPU ที่ตั้งใจจะใช้งานต่อ

```powershell
if (-not (Test-Path -LiteralPath $fkPython)) {
    uv venv --python 3.11 .venv
    if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed.' }
}
uv pip install --python $fkPython -r requirements-paper-alignment.lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
```

ไฟล์ lock มีการติดตั้ง repository นี้แบบ editable อยู่ด้วย ให้รันจาก `$Repo` ไม่จำเป็นต้อง activate environment เพราะคู่มือเรียก Python ตามพาธโดยตรง

### ตรวจแพ็กเกจและอุปกรณ์ประมวลผล

```powershell
@'
import sys, torch, torchvision, transformers, av, cv2, soundfile, safetensors
print(sys.executable)
print(sys.version)
print('torch:', torch.__version__)
print('CUDA build:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
'@ | Invoke-Python -
Invoke-Python -m deepfake_detection.cli --help
```

สภาพแวดล้อมที่เคยตรวจใช้ PyTorch แบบ CPU คำสั่ง `--device auto` จะเลือก CUDA เมื่อ `torch.cuda.is_available()` เป็นจริงเท่านั้น นอกนั้นจะใช้ CPU ถ้าต้องการบังคับใช้ GPU และให้หยุดเมื่อใช้ไม่ได้ ให้ตั้ง `$Device = 'cuda'` หลังตรวจว่า CUDA พร้อมแล้ว

สำหรับ NVIDIA GPU ใช้ [ตัวเลือกติดตั้ง PyTorch ทางการ](https://pytorch.org/get-started/locally/) เลือก Windows และ CUDA build ที่เข้ากันได้ ติดตั้งแพ็กเกจตามคำแนะนำลง **Python ตัวเดียวกับคู่มือนี้** โดยใช้ `uv pip install --python $fkPython ...` พร้อมแพ็กเกจและ index ที่เลือก ตรวจให้ torch และ torchvision เข้ากันได้ การติดตั้ง CUDA toolkit อย่างเดียวไม่เปลี่ยน torch wheel แบบ CPU ให้รองรับ CUDA หลังเปลี่ยนแพ็กเกจให้ตรวจ import/runtime ใหม่ และบันทึกเวอร์ชันจริงแยกจาก CPU lock

### FFmpeg และ FFprobe

ถ้ามี `.venv\Scripts\ffmpeg.exe` และ `ffprobe.exe` แล้ว ให้ข้ามการติดตั้งไปตรวจเวอร์ชันด้านล่าง ถ้ายังไม่มี สามารถสร้างชุดเครื่องมือแบบแยกโฟลเดอร์ตามที่เคยตรวจได้ดังนี้:

```powershell
uv pip install --python $fkPython --target .cache/ffmpeg-tools 'static-ffmpeg==3.0'
if ($LASTEXITCODE -ne 0) { throw 'FFmpeg helper installation failed.' }
@'
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.cache/ffmpeg-tools').resolve()))
from static_ffmpeg.run import get_or_fetch_platform_executables_else_raise
ffmpeg, ffprobe = get_or_fetch_platform_executables_else_raise()
destination = Path(sys.executable).parent
for source, name in [(ffmpeg, 'ffmpeg.exe'), (ffprobe, 'ffprobe.exe')]:
    target = destination / name
    if target.exists():
        raise FileExistsError(f'Inspect existing executable before replacing it: {target}')
    shutil.copy2(source, target)
    print(target)
'@ | Invoke-Python -
```

ตัวช่วยนี้จะดาวน์โหลดไฟล์ executable หากมีอยู่แล้วเพียงตัวเดียว ให้ตรวจและแก้การติดตั้งที่ไม่ครบก่อนรันบล็อกคัดลอก อีกทางเลือกคือใช้ FFmpeg ที่เชื่อถือได้และอยู่บน PATH ตัวค้นหาเครื่องมือของโปรเจกต์รองรับทั้ง PATH และโฟลเดอร์ Python Scripts

```powershell
@'
import subprocess
from deepfake_detection.media import media_tool
for name in ('ffmpeg', 'ffprobe'):
    subprocess.run([media_tool(name), '-version'], check=True)
'@ | Invoke-Python -
Invoke-Python -m unittest discover -s tests -v
```

HANDOFF ที่บันทึกไว้รายงานว่าผ่าน 97 tests ให้ยึดผลที่รันจริงกับ checkout ของคุณเป็นผลปัจจุบัน การผ่าน tests ยืนยันพฤติกรรมซอฟต์แวร์ ส่วนความแม่นยำต้องประเมินด้วยข้อมูลจริง

บันทึกสภาพแวดล้อมที่ใช้จริงก่อนเริ่มประมวลผลข้อมูล:

```powershell
$Setup = Join-Path $RunRoot ('setup-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $Setup | Out-Null
git rev-parse HEAD | Set-Content -LiteralPath (Join-Path $Setup 'git-commit.txt')
if ($LASTEXITCODE -ne 0) { throw 'Cannot record code revision.' }
git status --short | Set-Content -LiteralPath (Join-Path $Setup 'git-status.txt')
if ($LASTEXITCODE -ne 0) { throw 'Cannot record working-tree state.' }
uv pip freeze --python $fkPython | Set-Content -LiteralPath (Join-Path $Setup 'requirements-actual.txt')
if ($LASTEXITCODE -ne 0) { throw 'Cannot record installed packages.' }
```

<a id="step-5"></a>

## 5. ตรวจ metadata และตัวอย่างวิดีโอ

ตรวจเบื้องต้นก่อนเตรียมข้อมูลจำนวนมาก บล็อกนี้ตรวจชนิดของ metadata และค่า split ใช้ adapter ของโปรเจกต์หาพาธ แล้วเปิดวิดีโอ 3 ไฟล์แรกต่อ split ยังไม่ได้ตรวจทุกวิดีโอหรือทุก annotation

```powershell
@'
import json
import sys
from pathlib import Path
from deepfake_detection.avdeepfake1m import media_path, media_duration
root = Path(sys.argv[1]).resolve(strict=True)
for split in ('train', 'val'):
    path = root / f'{split}_metadata.json'
    records = json.loads(path.read_text(encoding='utf-8'))
    assert isinstance(records, list) and records, f'Empty/invalid metadata: {path}'
    assert all(r.get('split') == split for r in records), f'Wrong split: {path}'
    print(split, 'metadata rows:', len(records))
    for row in records[:3]:
        source = (root / media_path(row['file'], split)).resolve(strict=True)
        source.relative_to(root)
        print(source, 'duration_seconds:', media_duration(source))
'@ | Invoke-Python - $Dataset
```

ถ้าหาไฟล์ไม่พบ ให้เทียบค่า `file` กับโครงสร้างที่แตกไว้ adapter รองรับทั้งพาธที่มีและไม่มีชื่อ split ทางการนำหน้า อย่าแก้พาธ annotation หรือความยาวเพื่อกลบปัญหาการแตกไฟล์

<a id="step-6"></a>

## 6. เตรียมโมเดลที่ฝึกไว้ล่วงหน้า

### ใช้ไฟล์ที่ตรวจแล้วบนเครื่องนี้

หากเครื่องนี้มี assets ที่ตรวจแล้ว ให้ตั้ง `$Assets` ไปที่ registry บนเครื่อง ค่าเริ่มต้นด้านล่างตรงกับโฟลเดอร์ดาวน์โหลดใหม่ของคู่มือนี้ หากได้ `False` ให้ทำขั้นดาวน์โหลดถัดไป:

```powershell
$Assets = Join-Path $RunRoot 'pretrained-001\assets.local.json'
Test-Path -LiteralPath $Assets
```

ไฟล์โมเดลเหล่านี้ไม่ได้รวมใน Git และ registry ใช้พาธเต็ม การคัดลอกเฉพาะ JSON ไปอีกเครื่องไม่เพียงพอ คู่มือนี้ไม่ต้องอาศัย registry ของเครื่องผู้พัฒนา แนะนำให้ดาวน์โหลดโมเดลทั้งสี่บนเครื่องตามขั้นถัดไป หากย้าย assets มาใช้ ให้ย้ายทั้งโฟลเดอร์พร้อม integrity manifest แล้วแก้ `directory`/`manifest` ในรายการ artifact, `silero.path` และ `detector.path` เป็นพาธจริงบนเครื่องใหม่ รักษา hash/revision เดิมและตรวจด้วยคำสั่งด้านล่าง อย่าถือว่าพาธใน registry ใช้ข้ามเครื่องได้ทันที

### ดาวน์โหลดเมื่อยังไม่มีโมเดล

ใช้โฟลเดอร์ใหม่ ตัวดาวน์โหลดจะไม่เขียนทับชุดที่มีอยู่แล้ว:

```powershell
$AssetDir = Join-Path $RunRoot 'pretrained-001'
Invoke-Python scripts/acquire_pretrained.py --output $AssetDir `
    --assets efficientnet_b0 wav2vec2_base silero yunet
$Assets = Join-Path $AssetDir 'assets.local.json'
```

ควรได้ `assets.local.json`, `acquisition.json` และโฟลเดอร์โมเดล 4 ตัว พร้อมไฟล์ตรวจความสมบูรณ์ รายการทั้งสี่ที่ขอใน `acquisition.json` ต้องเป็น `ACQUIRED_AND_CHECKED` หากดาวน์โหลดล้มเหลวจะมีไฟล์วิเคราะห์ปัญหาเหลือไว้ ให้แก้สาเหตุก่อนลองใหม่ในโฟลเดอร์ใหม่

### ตรวจ checksum และการทำงานจริง

```powershell
Invoke-Python -m deepfake_detection.cli infrastructure --assets $Assets `
    --output (Join-Path $Setup 'infrastructure.json')
Invoke-Python scripts/check_fk_runtime.py --assets $Assets `
    --output (Join-Path $Setup 'runtime.json')
```

รายงาน infrastructure รวม backbone เสริมสำหรับวิธีอื่นด้วย การขาด AST, factorized ViViT หรือ AV-HuBERT จึงไม่ขัดขวางเส้นทาง FK ที่ใช้โมเดลสี่ตัว สถานะรวมไม่ได้เปลี่ยนเป็น READY อัตโนมัติ ให้ตรวจรายการ `efficientnet_b0`, `wav2vec2_base`, `silero` และ `detector` แล้วตรวจว่ารายงาน runtime เป็น `PASSED`

สคริปต์ runtime ใช้โมเดล pretrained จริงกับข้อมูลสังเคราะห์และส่วนหัว FK ที่ยังไม่ฝึก โดยคำนวณบน CPU แม้รายงานว่า CUDA ใช้งานได้ จึงไม่ใช่ผล benchmark GPU หรือผลวัดความแม่นยำ ส่วน pilot จะตรวจการประมวลผลข้อมูลจริงบนอุปกรณ์ที่คุณเลือก

<a id="step-7"></a>

## 7. รันชุดทดลองขนาดเล็ก

ใช้ pilot ตรวจปัญหาพาธ annotation คุณภาพ หน่วยความจำ และ calibration ก่อนรันเต็มชุด เพดาน train 15,000 / validation 2,500 เป็นขอบเขตข้อมูลสำหรับพัฒนา ไม่ได้หมายความว่าจะรันเร็วบน CPU ลดเพดานได้ แต่ต้องรองรับ source family ทั้งกลุ่มและมีข้อมูลทั้งสองคลาส

อย่าสร้างโฟลเดอร์ `population`, `cache-v2` หรือ `fk-smoke` ล่วงหน้า เพราะคำสั่งจะสร้างเอง ถ้า `pilot-001` มีงานเก่า ให้ตรวจแล้วเปลี่ยนเป็น `pilot-002` แทนการเขียนทับ

### 7.1 กำหนดชุดข้อมูล pilot ให้แน่นอน

ขอบเขต annotation: ข้อมูลจริงบางรายการระบุเวลาปลอมเกินความยาวไฟล์ adapter จะใช้เฉพาะส่วนที่ทับกับเวลาสื่อจริง เก็บช่วงต้นฉบับและส่วนที่เกินไว้ใน `annotation_bounds_adjustments` และนับจำนวนคลิปที่ปรับใน audit โดยไม่ตัดวิดีโอออก หากช่วงปลอมอยู่นอกไฟล์ทั้งหมดจะคง label ระดับวิดีโอและกำหนดข้อมูลช่วงเวลาเป็นไม่ครบ/ไม่ทราบ ระหว่างเตรียมข้อมูลจะแสดง progress ทุก 1,000 คลิป พร้อมเปอร์เซ็นต์เมื่อใช้เต็มชุด แต่ยังไม่มีตัวเลือก resume

ข้อมูลจริงมี `/` ซ้ำใน original และบางรายการอ้างถึงคลิปจริงที่ไม่ได้แจกมา adapter จะปรับรหัสต้นฉบับให้ตรงกันโดยไม่แก้ metadata ที่ดาวน์โหลด โหมด `keep-official` ที่เลือกไว้จะเก็บทุกคลิปตาม split เดิม รวมกลุ่มต้นฉบับ/เนื้อหาที่ซ้ำ โดยรายงานใน `source-overlaps.json` และ `audit.json` ส่วน `source-overlap-exclusions.json` จะว่าง นโยบายนี้ส่งต่อไปใน manifest, cache, checkpoint การตรวจ resume และรายงานประเมินผลด้วย หาก annotation แยกภาพ/เสียงขัดกันยังเก็บต้นฉบับไว้และกำหนดส่วนนั้นเป็นไม่ทราบ โหมดเก่า `drop-train` ยังเลือกใช้ได้ แต่ไม่ใช่แนวทางปัจจุบัน

```powershell
Invoke-Python -m deepfake_detection.cli prepare-avdeepfake1m `
    --dataset-root $Dataset --output (Join-Path $Pilot 'population') `
    --train-clips 15000 --validation-clips 2500 --seed 7 --source-overlap-policy keep-official
```

คำสั่งนี้อ่าน metadata ทางการ หา source family ของไฟล์จริงและปลอม ตรวจ label และช่วงเวลา อ่านความยาวจาก timestamp ของสื่อ และคำนวณ hash ไฟล์ที่เลือก แม้กำหนดเพดานก็ยังโหลด metadata train/validation ทั้งหมด การเลือกเป็นไปตามลำดับที่กำหนดซ้ำได้จาก seed และเลือกทั้ง family จำนวนจริงจึงอาจต่ำกว่าเพดาน หรืออาจล้มเหลวถ้าเพดานเล็กเกินไป

ผลลัพธ์:

- `population\manifest.json`: รายการวิดีโอ train/validation ที่มี label และถูกเลือกใช้
- `population\audit.json`: hash ของ metadata ผลตรวจชุดข้อมูล ค่าการเลือก และสถานะ label ของ test
- `population\test-inference-files.json`: สร้างเมื่อมี `test_files.txt` ที่ไม่มี label รายการนี้ไม่ถูกนำไปฝึกแบบ supervised

ตรวจจำนวนและรายงาน audit:

```powershell
Get-Content -LiteralPath (Join-Path $Pilot 'population\audit.json')
@'
import json, sys
from collections import Counter
rows = json.load(open(sys.argv[1], encoding='utf-8'))
print('split/label counts:', Counter((r['split'], r['label']) for r in rows))
print('source families:', len({r['source_family'] for r in rows}))
print('total source hours:', sum(r['duration'] for r in rows) / 3600)
'@ | Invoke-Python - (Join-Path $Pilot 'population\manifest.json')
```

ตรวจว่า train และ validation ไม่ว่าง และมีทั้งวิดีโอจริง (`0`) กับปลอม (`1`) ถ้าไม่ทราบช่วงเวลาปลอมแยกภาพ/เสียง ให้คงสถานะไม่ทราบไว้ ห้ามสร้าง annotation เอง ต้องแก้ข้อผิดพลาดขั้นเตรียมข้อมูลก่อน fit quality

### 7.2 เรียนรู้ค่าปรับคุณภาพจากชุดฝึก

```powershell
Invoke-Python -m deepfake_detection.cli fit-quality `
    --manifest (Join-Path $Pilot 'population\manifest.json') `
    --dataset-root $Dataset --assets $Assets `
    --output (Join-Path $Pilot 'quality.json')
```

ควรได้ `quality.json` ที่มี schema `measured-quality-v2` และ `fit_split` เป็น `train` โปรไฟล์ผูกกับชุดฝึกที่เลือกและ assets ที่ใช้ ขอบเขต percentile ของความเบลอและ SNR เรียนรู้จาก train เท่านั้น ค่าที่วัดไม่ได้หรือคงที่จะใช้ค่ากลาง โดยไม่ fit จาก validation หรือ test

ขั้นตอนนี้ถอดรหัสสื่อและวัดคุณภาพใบหน้า/เสียง มี progress และบันทึกค่าที่วัดทุกหน้าต่างลง `<output>.work/state.sqlite` รองรับ `--resume` ตามหัวข้อ 9 แต่ไม่มีตัวเลือก `--device` SNR สามารถไม่มีค่าได้หากช่วงเสียงพูด/ไม่พูดไม่เพียงพอสำหรับประมาณค่า

### 7.3 สร้าง cache จาก encoder ที่ตรึงน้ำหนัก

```powershell
$CacheTimer = [System.Diagnostics.Stopwatch]::StartNew()
Invoke-Python -m deepfake_detection.cli cache-fk `
    --manifest (Join-Path $Pilot 'population\manifest.json') `
    --dataset-root $Dataset --assets $Assets `
    --quality-normalization (Join-Path $Pilot 'quality.json') `
    --output (Join-Path $Pilot 'cache-v2') --device $Device
$CacheTimer.Stop()
$CacheTimer.Elapsed.ToString() | Set-Content -LiteralPath (Join-Path $Pilot 'cache-elapsed.txt')
```

ควรได้ `cache-run.json`, `caches\...` และ `training-manifest.json` ที่สร้างเสร็จสมบูรณ์ ไฟล์หลังใช้ schema `fk-cache-manifest-v2` และเก็บผลตรวจ usability ของคลิป/ช่วงเวลาที่ใช้ไม่ได้ด้วย คลิปที่ไม่มีช่วงเวลาใช้งานได้จะไม่ถูกเปลี่ยนเป็นคำทำนายว่าเป็นวิดีโอจริง

ตรวจจำนวนคลาสที่ใช้งานได้และความครอบคลุมก่อนฝึก:

```powershell
@'
import json, sys
from collections import Counter
m = json.load(open(sys.argv[1], encoding='utf-8'))
assert m['schema'] == 'fk-cache-manifest-v2'
for split in ('train', 'validation', 'test'):
    print(split, 'usable videos:', len(m[split]), 'labels:', Counter(r['label'] for r in m[split]))
    assessments = [a for a in m['assessments'] if a['split'] == split]
    print('all assessed-for-usability videos:', len(assessments))
    print('window statuses:', Counter(w['status'] for a in assessments for w in a['windows']))
    if split in ('train', 'validation'):
        assert {r['label'] for r in m[split]} == {0, 1}, f'Both usable classes required: {split}'
print('usable validation window labels:', Counter(
    w['labels']['combined'] for a in m['assessments'] if a['split'] == 'validation'
    for w in a['windows'] if w['status'] == 'USABLE'))
'@ | Invoke-Python - (Join-Path $Pilot 'cache-v2\training-manifest.json')
```

Calibration ต้องมีทั้งสองคลาสใน label ที่ทราบและใช้งานได้ของวิดีโอ validation และช่วงเวลา validation หากมีข้อมูลใช้งานไม่ได้มากผิดปกติ ให้หาสาเหตุก่อนเพิ่มขนาด dataset

### 7.4 ฝึกสอง epoch และทำ calibration

```powershell
Invoke-Python -m deepfake_detection.cli train-fk `
    --manifest (Join-Path $Pilot 'cache-v2\training-manifest.json') `
    --model-config configs/ours.provisional.json `
    --training-config configs/fk.training-v1.json `
    --output (Join-Path $Pilot 'fk-smoke') --epochs 2 --calibrate --device $Device
```

ขั้นนี้เป็น **การตรวจว่ากระบวนการทำงานครบ** ก่อนการทดลองจริง Encoder ที่ตรึงน้ำหนักถูกใช้ตอนสร้าง cache แล้ว ขั้นนี้ฝึกส่วนหัวตัวตรวจจับโดยใช้วิดีโอทั้งคลิปและทุกช่วงเวลาที่ใช้งานได้ โค้ดไม่มีตัวเลือก CLI ทั่วไปสำหรับ batch size, mixed precision หรือ distributed training

| ผลลัพธ์ | หน้าที่ |
| --- | --- |
| `run.json` | บันทึกการตั้งค่า อุปกรณ์ ลายเซ็นชุดข้อมูล และที่มาตอนเริ่มงาน ค่า `STARTED` เพียงอย่างเดียวบอกไม่ได้ว่างานสุดท้ายสำเร็จหรือล้มเหลว |
| `last.pt` | epoch ล่าสุดที่เสร็จ พร้อม optimizer, RNG และประวัติสำหรับ resume |
| `best.pt` | น้ำหนักที่ validation loss ดีที่สุด เมื่อฝึกเสร็จจะจัดเป็น checkpoint สำหรับ inference |
| `metrics.json` | สรุปการฝึกที่เสร็จและประวัติแต่ละ epoch |
| `timing.json` | เวลาฝึกและ validation ต่อ epoch |
| `selection.json` | ค่าที่ทดลองและค่าตัดสินที่เลือกจาก validation calibration |
| `selected.pt` | น้ำหนักที่ดีที่สุดพร้อมค่ารวมคะแนนและ threshold ที่เลือกจาก validation ใช้ประเมินและทำนายหลัง calibration |

Calibration ทำหลังฝึกและอาจใช้เวลา/หน่วยความจำเพิ่มมาก การฝึกเสร็จไม่ได้ยืนยันว่า calibration สำเร็จ ต้องตรวจว่ามีทั้ง `selected.pt` และ `selection.json`

### 7.5 ประเมิน validation ของ pilot

```powershell
Invoke-Python -m deepfake_detection.cli evaluate-fk `
    --manifest (Join-Path $Pilot 'cache-v2\training-manifest.json') `
    --checkpoint (Join-Path $Pilot 'fk-smoke\selected.pt') `
    --split validation --device $Device `
    --output (Join-Path $Pilot 'validation.json')
```

ตรวจว่ารายงานสร้างได้ ใช้ split ถูกต้อง อ่าน metrics/coverage ได้ และไม่มีการตรวจที่มาล้มเหลว ความแม่นยำต่ำยังเป็นผลทดลองที่ถูกต้องได้ อย่าลบวิดีโอที่ยากเพื่อทำให้คะแนนดีขึ้น ผลนี้มาจาก validation ที่ใช้เลือกโมเดลและค่าตัดสินด้วย จึงยังไม่ใช่ผล held-out test

<a id="step-8"></a>

## 8. ประเมินทรัพยากรและรันเต็มชุด

### วัดก่อนขยายขนาด

```powershell
$PilotCacheBytes = (Get-ChildItem -LiteralPath (Join-Path $Pilot 'cache-v2') -Recurse -File |
    Measure-Object -Property Length -Sum).Sum
'Pilot cache GiB: {0:N2}' -f ($PilotCacheBytes / 1GB)
Get-Content -LiteralPath (Join-Path $Pilot 'cache-elapsed.txt')
Get-Content -LiteralPath (Join-Path $Pilot 'fk-smoke\timing.json')
Get-PSDrive -PSProvider FileSystem
```

ประเมินพื้นที่และเวลาจากความยาวสื่อต้นฉบับหรือจำนวนช่วงเวลาที่ใช้ได้ แทนการดูจำนวนวิดีโออย่างเดียว สูตรด้านล่างคือค่าที่วัดจาก pilot คูณอัตราส่วนของชุดเต็มต่อ pilot:

```text
estimated full cache size ~= pilot cache size * (full windows / pilot windows)
estimated full cache time ~= pilot cache time * (full source duration / pilot source duration)
estimated epoch time ~= pilot epoch time * (full usable windows / pilot usable windows)
```

สูตรนี้เป็นเพียงประมาณการ ปัญหาถอดรหัส การตรวจใบหน้า ความยาววิดีโอ ความเร็วไดรฟ์ หน่วยความจำ GPU และต้นทุน calibration ทำให้อัตราส่วนเปลี่ยนได้ เผื่อทรัพยากรและวัดซ้ำกับ pilot ที่ใหญ่ขึ้น ยังไม่ได้ยืนยันเวลา RAM, VRAM หรือพื้นที่ที่ต้องใช้กับข้อมูลเต็มชุดสำหรับ implementation นี้ การ fit quality และ calibration เก็บชุดข้อมูลบางส่วนใน RAM และการค้นหาค่า calibration อาจเป็นคอขวดเพิ่มเติม

### เตรียม train/validation ที่มี label ทั้งหมด

สร้างชุดเต็มในพาธใหม่ โดยไม่ระบุเพดานคลิป:

```powershell
Invoke-Python -m deepfake_detection.cli prepare-avdeepfake1m `
    --dataset-root $Dataset --output (Join-Path $Full 'population') --seed 7 --source-overlap-policy keep-official
```

ตรวจ audit และจำนวนเหมือน pilot แต่เปลี่ยนเป็น `$Full` ขั้นเตรียมจะอ่านและ hash ทุกไฟล์ที่เลือก จึงอาจใช้เวลานาน กำหนดโค้ด assets การตั้งค่า และ environment ให้แน่นอนก่อนสร้าง cache

### Fit โปรไฟล์สำหรับชุดฝึกเต็ม สร้าง cache และฝึก

```powershell
Invoke-Python -m deepfake_detection.cli fit-quality `
    --manifest (Join-Path $Full 'population\manifest.json') `
    --dataset-root $Dataset --assets $Assets --output (Join-Path $Full 'quality.json')

Invoke-Python -m deepfake_detection.cli cache-fk `
    --manifest (Join-Path $Full 'population\manifest.json') `
    --dataset-root $Dataset --assets $Assets `
    --quality-normalization (Join-Path $Full 'quality.json') `
    --output (Join-Path $Full 'cache-v2') --device $Device

Invoke-Python -m deepfake_detection.cli train-fk `
    --manifest (Join-Path $Full 'cache-v2\training-manifest.json') `
    --model-config configs/ours.provisional.json `
    --training-config configs/fk.training-v1.json `
    --output (Join-Path $Full 'fk-main') --calibrate --device $Device

Invoke-Python -m deepfake_detection.cli evaluate-fk `
    --manifest (Join-Path $Full 'cache-v2\training-manifest.json') `
    --checkpoint (Join-Path $Full 'fk-main\selected.pt') `
    --split validation --device $Device --output (Join-Path $Full 'validation.json')
```

รันทีละคำสั่งและตรวจผลก่อนเริ่มขั้นถัดไป ตรวจ coverage และคลาสของ full cache ตามวิธีใน pilot ด้วย

การตั้งค่าที่ตรวจไว้ใช้ AdamW, learning rate `0.0001`, weight decay `0.0001`, สูงสุด **50 epochs**, early-stopping patience **5** และ seed **7** เลือกน้ำหนักจากค่าเฉลี่ย validation video loss จากนั้น calibration เลือกค่ารวมคะแนน/threshold จาก validation บันทึกการเปลี่ยนค่าทั้งหมดก่อนประเมิน test รอบสุดท้าย

ต้อง fit quality profile และสร้าง cache สำหรับชุดเต็มใหม่ เพราะ fingerprint ของชุดฝึกต่างจาก pilot อย่านำ checkpoint ที่ฝึกด้วย LAV-DF หรือ pilot ไปรายงานว่าเป็นผลฝึก AV-Deepfake1M เต็มชุด

<a id="step-9"></a>

### Progress และ resume ของ fit-quality

```powershell
Invoke-Python -m deepfake_detection.cli fit-quality `
    --manifest (Join-Path $Full 'population\manifest.json') `
    --dataset-root $Dataset --assets $Assets `
    --output (Join-Path $Full 'quality.json') --resume --progress-seconds 10
```

ครั้งแรกไม่ใส่ `--resume` ระบบบันทึกทุกหน้าต่างที่เสร็จและ VAD ของคลิปลง
`quality.json.work/state.sqlite` ต้องเก็บโฟลเดอร์นี้ไว้ เมื่อหยุดจะทำซ้ำเฉพาะ
หน้าต่างที่ยังไม่ commit และใช้ output เดิมพร้อม `--resume` เพื่อทำต่อ
แสดงคลิป/หน้าต่างที่เสร็จ ความเร็ว เวลาที่ใช้ และ ETA ทุก 10 วินาทีเมื่อถึงจุดอัปเดต
การอ่าน hash, VAD หรือ decoding อาจใช้เวลาก่อนรายการถัดไป
ETA จะเริ่มมีประโยชน์หลังประมวลผลหลายหน้าต่างแล้ว

Resume ตรวจ manifest, assets, runtime, โค้ดที่เกี่ยวข้อง และ checksum ข้อมูลที่บันทึก
พร้อมตรวจ hash ของคลิปเมื่ออ่าน ถ้าเปลี่ยนสิ่งเหล่านี้ให้ใช้ output ใหม่
ข้อมูลคุณภาพเก็บบนดิสก์และยังคำนวณ percentile แบบเดิมจากหน้าต่าง train ทั้งหมด
ไม่เปลี่ยนเป็นการสุ่มตัวอย่าง มีการป้องกันหลายโปรเซสเขียน work directory เดียวกัน
โฟลเดอร์ work อาจมีขนาดใหญ่เมื่อใช้ทั้งชุด

โปรเซสที่เริ่มก่อนเพิ่มฟีเจอร์นี้ไม่มี saved state ให้กด Ctrl+C แล้วเริ่มคำสั่งใหม่
โดยไม่ใส่ `--resume` หนึ่งครั้ง งานเดิมใน RAM นำเข้าไม่ได้
ไม่ต้องแตกไฟล์หรือ prepare population ซ้ำ ไฟล์ `quality.json` จะถูกสร้างหลังทำเสร็จ
และตรวจสอบได้เมื่อ resume รอบที่เสร็จแล้ว

## 9. ทำงานต่อเมื่อถูกขัดจังหวะ

เมื่อเปิด PowerShell ใหม่ ให้กำหนดตัวแปร/ฟังก์ชันในหัวข้อ 2 อีกครั้ง เข้า `$Repo` และตั้ง `$Assets` ให้ชี้ registry ที่ใช้จริง ใช้โค้ดและ environment เดิมสำหรับการทำงานต่อ

### สร้าง feature cache ต่อ: ใช้โฟลเดอร์เดิม

```powershell
Invoke-Python -m deepfake_detection.cli cache-fk `
    --manifest (Join-Path $Full 'population\manifest.json') `
    --dataset-root $Dataset --assets $Assets `
    --quality-normalization (Join-Path $Full 'quality.json') `
    --output (Join-Path $Full 'cache-v2') --resume --device $Device
```

คำสั่งจะตรวจคลิปที่เสร็จแล้วและทำส่วนที่ค้างต่อ โดย source manifest, assets, profile, ที่มาของโค้ด, เวอร์ชัน torch และอุปกรณ์ encoder ต้องตรงกับรอบเดิม ถ้า `auto` เปลี่ยนจาก CPU ไป CUDA อาจ resume ไม่ได้ อย่าแก้ provenance JSON เพื่อข้ามการตรวจ หากตั้งใจเปลี่ยนกระบวนการให้สร้าง cache ใหม่

### ฝึกต่อ: ใช้ checkpoint ล่าสุดและโฟลเดอร์ผลลัพธ์ใหม่

```powershell
Invoke-Python -m deepfake_detection.cli train-fk `
    --manifest (Join-Path $Full 'cache-v2\training-manifest.json') `
    --model-config configs/ours.provisional.json `
    --training-config configs/fk.training-v1.json `
    --resume (Join-Path $Full 'fk-main\last.pt') `
    --output (Join-Path $Full 'fk-main-resumed-001') --calibrate --device $Device
```

Resume เริ่มต่อจาก epoch ล่าสุดที่เสร็จ หากหยุดกลาง epoch ต้องทำ epoch นั้นใหม่ เพดาน epoch คือ **หมายเลข epoch เป้าหมายรวม** ไม่ใช่จำนวน epoch เพิ่มเติม โปรแกรมตรวจความเข้ากันได้ของโมเดล การตั้งค่า และข้อมูล ให้คง configuration และอุปกรณ์เดิมเพื่อทำซ้ำได้ แล้วใช้ `selected.pt` จากโฟลเดอร์ resumed ในขั้นประเมินถัดไป

ห้าม resume จาก `best.pt` หรือ `selected.pt` หาก checkpoint ถึงเพดานหรือ early stopping แล้ว โปรแกรมจะปฏิเสธ ปัจจุบัน CLI ไม่มีคำสั่งทำ calibration อย่างเดียว หากฝึกเสร็จแต่ calibration ล้มเหลว ให้เก็บ checkpoint และ error log ไว้ อย่าเปลี่ยนชื่อ `best.pt` เป็น `selected.pt` หรือฝืน resume หลัง early stop การกู้เฉพาะ calibration ต้องเพิ่มโค้ดแยก หรือแก้สาเหตุแล้วเริ่มฝึก/calibrate รอบใหม่

ขั้นเตรียม population ไม่มี resume หากขัดจังหวะให้เริ่มด้วยพาธใหม่ ส่วน fit-quality รองรับ resume รายหน้าต่างตามคำสั่งด้านบน เก็บผล/บันทึกที่ล้มเหลวจนทราบสาเหตุ ตัวรันหลายการทดลองไม่มี resume ทั้งแผน ให้เก็บรอบที่สำเร็จไว้แทนการสั่งซ้ำลงโฟลเดอร์เดิม

<a id="step-10"></a>

## 10. ประเมินและตีความผล

### ผลตรวจสอบจาก validation

คำสั่งด้านบนประเมิน validation ที่ใช้เลือก checkpoint และค่าตัดสินด้วย ให้ระบุผลว่า **validation diagnostics** ใช้พัฒนาได้ แต่ยังไม่ใช่ค่าประเมินที่เป็นกลางจาก held-out test

ตรวจ `video_metrics`, `window_metrics`, `window_coverage_by_label`, `timeline_coverage` และรายการ `videos`/`windows` คะแนนจำแนกคำนวณภายใต้เงื่อนไขว่าประเมินตัวอย่างนั้นได้ จึงต้องรายงาน coverage ควบคู่กัน ROC AUC, average precision และค่าอื่นอาจเป็น `null` เมื่อข้อมูลไม่พอคำนวณ คลิปที่ประเมินไม่ได้ไม่เท่ากับคำทำนายว่าเป็นคลิปจริง

Label ของช่วงเวลาใช้การซ้อนทับที่มีระยะเวลามากกว่าศูนย์กับช่วงปลอมที่ระบุไว้ ส่วน timeline metrics เป็นผลเสริมที่รวมคำทำนายช่วงซ้อนกัน ค่า average precision สำหรับการจำแนกและผล timeline นี้ไม่ใช่ benchmark AP/AR สำหรับ temporal localization ทางการ

### ทางเลือก: มี metadata test ทางการพร้อม label จริง

ใช้วิธีนี้เฉพาะเมื่อมี metadata test ที่มี label จริงและเข้ากับ adapter เท่านั้น ใช้ `test_files.txt` แทนไม่ได้ เพิ่ม metadata test ตอนเตรียม population ใหม่ ก่อน fit/cache/train:

```powershell
$Labeled = Join-Path $RunRoot 'full-labeled-test-001'
$TestMetadata = Join-Path $Dataset 'test_metadata.json'
if (-not (Test-Path -LiteralPath $TestMetadata)) { throw 'Actual labeled test metadata is required.' }
Invoke-Python -m deepfake_detection.cli prepare-avdeepfake1m `
    --dataset-root $Dataset --output (Join-Path $Labeled 'population') `
    --test-metadata $TestMetadata --seed 7 --source-overlap-policy keep-official
```

สำหรับทางเลือกนี้ ตั้ง `$Full = $Labeled` แล้วทำ quality/cache/train ตามหัวข้อ 8 ด้วย population ใหม่ ตัว fitter ยังใช้เฉพาะ train ส่วน training/calibration ใช้ train/validation เท่านั้น อย่าผสม manifest ใหม่นี้กับ profile ของรอบเก่า หรือเพิ่มรายการ test เข้า cache manifest เอง

เมื่อกำหนดวิธีทดลองแน่นอนและได้ `selected.pt` แล้ว จึงประเมิน held-out test สำหรับรายงานรอบสุดท้าย:

```powershell
Invoke-Python -m deepfake_detection.cli evaluate-fk `
    --manifest (Join-Path $Full 'cache-v2\training-manifest.json') `
    --checkpoint (Join-Path $Full 'fk-main\selected.pt') `
    --split test --device $Device --output (Join-Path $Full 'test.json')
```

ห้ามใช้ผล test เลือกขอบเขต quality, threshold, epoch หรือปรับ architecture หากแก้บั๊กการประเมิน ให้บันทึกแยกจากการปรับโมเดลตามผล test อย่างชัดเจน

### วิดีโอ test ทางการที่ไม่มี label

`test_files.txt` มีพาธแต่ไม่มี label ตัว adapter จึงสร้างรายการสำหรับ inference แยกไว้ ให้ฝึกและ validate ตามปกติ แล้วทำนายไฟล์ test ที่เลือกตามหัวข้อ 12 หากไม่มี ground truth จะคำนวณ accuracy/F1/AUC บนเครื่องไม่ได้ เส้นทางนี้ยังไม่มีตัวส่งออก submission ทางการแบบทั้งชุด ให้ดู [เอกสาร dataset/การแข่งขันทางการ](https://github.com/ControlNet/AV-Deepfake1M) สำหรับวิธีประเมินที่เกี่ยวข้อง อย่าใช้ label ของ validation แทน หรือกำหนด test ทั้งหมดว่าเป็นคลิปจริง

<a id="step-11"></a>

## 11. รันโมเดลเปรียบเทียบและ ablation

ทำหลังจาก FK รอบเดียวทำงานครบแล้วและกำหนดงบการทดลองได้:

```powershell
Invoke-Python -m deepfake_detection.cli prepare-experiments `
    --model-config configs/ours.provisional.json --output (Join-Path $Full 'experiment-plan')

Get-Content -LiteralPath (Join-Path $Full 'experiment-plan\plan.json')

Invoke-Python -m deepfake_detection.cli run-experiments `
    --plan (Join-Path $Full 'experiment-plan\plan.json') `
    --manifest (Join-Path $Full 'cache-v2\training-manifest.json') `
    --training-config configs/fk.training-v1.json `
    --output (Join-Path $Full 'experiments') --device $Device
```

แผนที่สร้างใช้ seeds 7, 17 และ 27 มีโมเดลเปรียบเทียบที่ใช้ encoder ร่วมกัน 4 แบบ ได้แก่ visual-only, audio-only, concatenation และ late fusion พร้อม FK 8 รูปแบบและรายการวิธีภายนอก ตัวรันนี้ฝึก/calibrate **36 รอบ** และข้ามรายการ AVFakeNet/AVTENet ภายนอก 6 รายการซึ่งต้องใช้ข้อมูลและตัวรันของตนเอง ทำงานทีละรอบตามลำดับ และไม่ประเมิน test อัตโนมัติ เมื่อจบแผน `completed.json` จะบันทึกรอบที่เสร็จและวิธีที่ข้าม

หากต้องการตรวจตัวรันเบื้องต้น ใช้ `--epochs 2` กับ pilot cache และโฟลเดอร์ผลลัพธ์ใหม่ อย่ารายงานผล smoke check เป็นผลหลาย seed รอบสุดท้าย

ตรวจค่าตัดสินที่ถูกเลือกก่อนตีความ ablation การค้นหา calibration ปัจจุบันอาจเปลี่ยนค่าการรวมคะแนนที่ตั้งต้นใน profile `window_max`, `window_mean`, `video_max` และ `video_mean` จึงต้องดู `selection.json` ก่อนสรุปว่ายังคงใช้ max/mean แบบตายตัว หากต้องการตรึง pooling ตลอด calibration ต้องปรับ implementation/วิธีทดลองก่อน

ประเมิน selected checkpoint แต่ละตัวด้วย `evaluate-fk` โดยใช้ split, population และกฎรายงานเดียวกัน การประเมินรองรับ checkpoint ของ shared baseline แต่คำสั่ง inference ที่แยกหลักฐานของ FK ไม่รองรับ checkpoint เหล่านั้น

<a id="step-12"></a>

## 12. ทำนายและตรวจสอบหลักฐาน

ใช้ไฟล์วิดีโอจริงกับ `selected.pt` ของ **FK** ที่ฝึกแล้ว ตัวอย่างนี้อ่านความยาวจากไฟล์จริง ให้แก้ `$InputVideo` เป็นไฟล์ที่มีอยู่บนเครื่อง:

```powershell
$InputVideo = Read-LocalPath 'Input video to analyze (full path)'
if (-not (Test-Path -LiteralPath $InputVideo -PathType Leaf)) { throw 'Input video does not exist.' }
$Checkpoint = Join-Path $Full 'fk-main\selected.pt'
$Evidence = Join-Path $Full 'evidence-example-001'
$Duration = @'
import sys
from deepfake_detection.avdeepfake1m import media_duration
print(media_duration(sys.argv[1]))
'@ | Invoke-Python - $InputVideo

Invoke-Python -m deepfake_detection.cli infer-fk `
    --input $InputVideo --duration $Duration --checkpoint $Checkpoint `
    --assets $Assets --quality-profile measured-quality-v2 --output $Evidence

Invoke-Python -m deepfake_detection.cli verify `
    --record (Join-Path $Evidence 'record.json') `
    --trusted-commitment (Join-Path $Evidence 'trusted-commitment.txt') `
    --source $InputVideo --artifact-root $Evidence
```

หากสาธิตด้วย pilot ให้ตั้ง `$Checkpoint` ไปที่ `$Pilot\fk-smoke\selected.pt` และระบุว่าเป็นโมเดล pilot ใช้ assets เวอร์ชันเดียวกับตอนฝึก อย่าสลับ quality profile เก่าเข้ามาเอง ปัจจุบัน `infer-fk` ไม่มีตัวเลือกอุปกรณ์ จึงไม่ต้องเติม `--device`

ผลลัพธ์มี record และ commitment พร้อมข้อมูล NPZ ที่เตรียมแล้ว เสียง PCM16, metadata การเตรียม และภาพ PNG ตัวแทนต่อช่วงเวลาที่ใช้งานได้ NPZ เก็บพิกเซล/เสียง/timestamp ที่เตรียมแล้ว ไม่ใช่ neural embedding ใน cache ขั้นนี้อ่าน metadata ต้นฉบับผ่าน PyAV/libavformat ส่วน caption/transcript และ perturbation analysis ยังเป็น `NOT_RUN`

ต้องตรวจทั้ง source, record และไฟล์หลักฐานที่สร้างขึ้น หากไม่ส่ง `--artifact-root` การตรวจ artifact จะเป็น `NOT_REQUESTED` อ่านผลตรวจที่คืนมาด้วย การมี JSON หรือ exit code เป็นศูนย์อย่างเดียวไม่เพียงพอ คลิปที่ไม่มีช่วงใช้งานได้อาจไม่มี artifact รายช่วงและมีสถานะประเมินไม่ได้

เมื่อสร้าง record ครั้งแรก ให้เก็บสำเนา trusted commitment ในอีกตำแหน่งที่เชื่อถือได้ หาก record และ commitment สำเนาเดียวถูกเปลี่ยนพร้อมกัน การเทียบสองไฟล์นั้นอย่างเดียวไม่ยืนยันความสมบูรณ์ได้ การตรวจ hash ผ่านยืนยันว่าไฟล์ตรงกับหลักฐานที่เก็บไว้ ส่วนความถูกต้องทางวิทยาศาสตร์ของคำทำนายต้องประเมินแยก

<a id="step-13"></a>

## 13. แก้ปัญหาที่พบบ่อย

| อาการ | วิธีตรวจหรือแก้ |
| --- | --- |
| Missing volume, CRC error หรือ unexpected end | ตรวจว่า archive ทุกส่วนของ split ครบ ดาวน์โหลดส่วนที่มีปัญหาใหม่ แล้วตรวจ `7z t` ซ้ำ |
| มี metadata แต่ไม่พบวิดีโอ | เทียบพาธใน archive กับโครงสร้างที่แตก ตรวจโฟลเดอร์ซ้อน `train\train` และใช้ metadata ทางการเดิม |
| Annotation เกินความยาวหรือ label ขัดกัน | ตรวจรุ่น/schema และความสมบูรณ์ของไฟล์ เก็บรายการที่ผิดไว้ตรวจ ตรวจ bounds adjustments และอย่าแก้ metadata ที่ดาวน์โหลดหรือเปลี่ยน label เพื่อให้ผ่าน |
| ไม่มี family ทั้งกลุ่มที่ใส่ในเพดานได้ | เพิ่มเพดานและเตรียมลงโฟลเดอร์ใหม่ โปรแกรมไม่แบ่ง family เพื่อให้ได้จำนวนตรงเพดาน |
| Calibration ต้องการทั้งสองคลาส | ตรวจ label วิดีโอ/ช่วงเวลา validation ที่ใช้ได้และเหตุผลที่ถูกตัดออก เพิ่มขนาด pilot เมื่อจำเป็น แล้ว fit profile/cache ใหม่ |
| คลิปหรือช่วงเวลาส่วนใหญ่ใช้ไม่ได้ | ตรวจการถอดรหัส ใบหน้า crop policy และช่วงเสียงพูด รายงานข้อมูลที่ถูกตัดออก และอย่าผ่อนเงื่อนไขเงียบ ๆ ในรอบสุดท้าย |
| ไม่พบโมเดลหรือ checksum ไม่ตรง | ตรวจพาธเต็มใน registry และ manifest ดาวน์โหลดใหม่หรือคืน assets เดิม แทนการแก้ค่า hash |
| สถานะ infrastructure รวมไม่เป็น READY | ตรวจโมเดล FK ทั้งสี่รายการและรัน `check_fk_runtime.py` โมเดลเสริมของวิธีอื่นแสดงแยกกัน |
| ขอ CUDA แต่ใช้งานไม่ได้ | ตรวจ Python ที่เรียก torch wheel, driver และ `torch.cuda.is_available()` หากตั้งใจใช้ CPU ให้ระบุให้ชัด |
| GPU out of memory | ปิดงานอื่น ตรวจจำนวนช่วงเวลาของวิดีโอที่ยาวที่สุด การฝึกทั้งวิดีโออาจใช้หน่วยความจำมาก CPU ช่วยวิเคราะห์ได้แต่ยังไม่ยืนยันว่าจะรันเต็มชุดไหว ไม่มีตัวเลือก batch size ให้ลดใน CLI นี้ |
| RAM เพิ่มมากตอน fit/calibration | ลด pilot เพื่อหาสาเหตุ คู่มือนี้ยังไม่ได้เพิ่มการแบ่งข้อมูลหรือปรับหน่วยความจำสำหรับเต็มชุด เก็บ log และวัดการใช้สูงสุดก่อนลองอีกครั้ง |
| พื้นที่เต็ม | ตรวจไดรฟ์ที่เลือก ขนาด archive/ไฟล์ที่แตกและ cache ใหม่ เก็บต้นฉบับกับงานที่เสร็จให้กู้ได้ อย่าลบโฟลเดอร์ผลลัพธ์แบบเหมารวม |
| Output มีอยู่แล้ว | ใช้ชื่อใหม่ ยกเว้น cache ที่รองรับ `--resume` ไฟล์ส่วนใหญ่ตั้งใจป้องกันการเขียนทับ |
| Quality profile ไม่ตรงกับ population | ใช้ profile ที่ fit จาก training manifest และ assets ชุดนั้น สร้างใหม่เมื่อเปลี่ยนข้อมูลโดยตั้งใจ |
| Resume cache แล้ว provenance ไม่ตรง | คืนโค้ด/environment/อุปกรณ์ให้ตรง หรือสร้าง cache ใหม่ ห้ามแก้ hash ใน manifest |
| Resume training ถูกปฏิเสธ | ใช้ `last.pt` พร้อมข้อมูล/การตั้งค่าเดิม โฟลเดอร์ใหม่ และเป้าหมาย epoch รวมที่ยังทำไม่ครบ |
| ฝึกแล้วไม่มี `selected.pt` | Calibration อาจล้มเหลว ตรวจ log และคลาสทั้งสอง ปัจจุบันยังไม่มี CLI กู้เฉพาะ calibration |
| Test ว่างหรือไม่มี label | รายการ test ปกติไม่มี label ใช้ validation diagnostics หรือ metadata test ที่มี label จริง ห้ามสร้าง label ขึ้นเอง |
| GitHub Desktop แจ้ง lock file | ปิดงาน Git ที่กำลังทำ ตรวจพาธ lock ที่แน่นอน ลบเฉพาะ lock ค้างหลังยืนยันว่าไม่มี Git process ใช้ ห้ามลบ lock ทั้งหมดหรือ `.git` |

เมื่อต้องขอความช่วยเหลือ ให้เก็บคำสั่งที่รัน error/traceback ทั้งหมด commit ของโค้ด อุปกรณ์ เวอร์ชันแพ็กเกจ และ audit/run/selection JSON ที่เกี่ยวข้อง ระวังไม่แนบวิดีโอส่วนตัวขนาดใหญ่หรือข้อมูลรับรองใน log

<a id="step-14"></a>

## 14. รายการตรวจสอบก่อนจบงาน

- [ ] ตรวจ archive ผ่าน และพาธหลังแตกตรงกับ metadata ทางการ
- [ ] ตรวจวิดีโอ train/validation ผ่าน และไม่สร้าง label test เอง
- [ ] บันทึก commit, working-tree state, เวอร์ชันแพ็กเกจ อุปกรณ์ และ configuration
- [ ] โมเดล FK ทั้งสี่ผ่านการตรวจไฟล์/runtime และเก็บ manifest กับ registry ไว้
- [ ] Pilot ผ่านการเตรียมข้อมูล fit quality, cache, ฝึก/calibrate สอง epoch และประเมิน validation
- [ ] ประเมินพื้นที่ เวลา และ RAM สำหรับเต็มชุดจากค่าที่วัดจริง
- [ ] เก็บ audit ของชุดเต็ม hash ของ source/metadata และ normalized manifest
- [ ] Fit quality profile ของชุดเต็มจาก train เต็มชุดเท่านั้น
- [ ] เก็บ cache manifest, feature files, จำนวนข้อมูลทั้งหมดที่ใช้คำนวณ coverage และเหตุผลที่ใช้ไม่ได้ไว้ด้วยกัน
- [ ] เก็บ configuration, `last.pt`, `best.pt`, `selected.pt`, metrics, timing และ selection report
- [ ] ระบุผล validation diagnostics และ held-out test แยกชัดเจน
- [ ] ข้อสรุป baseline/ablation ตรงกับรอบที่รันจริงและค่าตัดสินที่เลือก
- [ ] ตรวจ forensic record/artifact และเก็บ trusted commitment แยกในที่เชื่อถือได้
- [ ] รายงาน population, coverage, seeds และข้อจำกัดที่ใช้จริง

เก็บ dataset และไฟล์ที่สร้างจากการรันไว้นอก Git แชร์โค้ด/configuration/เอกสารผ่าน Git และย้ายไฟล์ใหญ่แยกพร้อม hash สำรองโฟลเดอร์การรันทั้งหมด เพราะ manifest อย่างเดียวสร้าง tensor cache หรือวิดีโอต้นฉบับที่หายไปกลับมาไม่ได้

รายละเอียดวิธีทดลองอยู่ใน [paper alignment protocol](avdeepfake1m-paper-alignment.md) สถานะ implementation และข้อจำกัดที่เคยตรวจอยู่ใน [HANDOFF.md](handoffs/HANDOFF.md) และสามารถเทียบคำสั่งกับ [ฉบับภาษาอังกฤษ](avdeepfake1m-after-download-guide.md) ได้
