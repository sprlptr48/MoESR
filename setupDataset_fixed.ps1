# setupDataset.ps1
# Run from: C:\Users\egece\Projects\sr\supersr\codex
# Usage: .\setupDataset.ps1

$ErrorActionPreference = "Stop"
$root     = Split-Path -Parent $MyInvocation.MyCommand.Path
$dataDir  = Join-Path $root "dataset"
$zipDir   = Join-Path $dataDir "_zips"
$extDir   = Join-Path $dataDir "_extracted"

# Create folder structure
foreach ($p in @(
    "$dataDir\train\LR", "$dataDir\train\HR",
    "$dataDir\val\LR",   "$dataDir\val\HR",
    $zipDir, $extDir
)) {
    New-Item -ItemType Directory -Force -Path $p | Out-Null
}

Write-Host "`n[1/4] Downloading DIV2K..." -ForegroundColor Cyan

$div2k = @(
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip",
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip"
)

foreach ($url in $div2k) {
    Write-Host "  -> $url"
    aria2c `
        --dir="$zipDir" `
        --max-connection-per-server=16 `
        --split=16 `
        --min-split-size=5M `
        --continue=true `
        --file-allocation=none `
        "$url"
}

# Extract helper: tries 7z first, falls back to Expand-Archive
function Expand-Best {
    param(
        [Parameter(Mandatory = $true)][string]$zip,
        [Parameter(Mandatory = $true)][string]$dest
    )

    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    $z = Get-Command "7z" -ErrorAction SilentlyContinue
    if ($z) {
        Write-Host "  Extracting (7z): $zip"
        & 7z x "$zip" "-o$dest" -y | Out-Null
    }
    else {
        Write-Host "  Extracting (Expand-Archive - may be slow): $zip"
        Expand-Archive -Path "$zip" -DestinationPath "$dest" -Force
    }
}

Write-Host "`n[2/4] Extracting..." -ForegroundColor Cyan

Expand-Best "$zipDir\DIV2K_train_HR.zip"            "$extDir\train_hr"
Expand-Best "$zipDir\DIV2K_valid_HR.zip"            "$extDir\valid_hr"
Expand-Best "$zipDir\DIV2K_train_LR_bicubic_X4.zip" "$extDir\train_lr"
Expand-Best "$zipDir\DIV2K_valid_LR_bicubic_X4.zip" "$extDir\valid_lr"

Write-Host "`n[3/4] Organising into dataset/..." -ForegroundColor Cyan

# HR: flat PNGs directly inside the extracted folder
Get-ChildItem "$extDir\train_hr\DIV2K_train_HR\*.png" |
    Move-Item -Destination "$dataDir\train\HR"

Get-ChildItem "$extDir\valid_hr\DIV2K_valid_HR\*.png" |
    Move-Item -Destination "$dataDir\val\HR"

# LR: nested one level deeper - DIV2K_train_LR_bicubic\X4\
Get-ChildItem "$extDir\train_lr\DIV2K_train_LR_bicubic\X4\*.png" |
    Move-Item -Destination "$dataDir\train\LR"

Get-ChildItem "$extDir\valid_lr\DIV2K_valid_LR_bicubic\X4\*.png" |
    Move-Item -Destination "$dataDir\val\LR"

# Optional: generate LR pairs from game screenshots
Write-Host "`n[4/4] Game screenshots..." -ForegroundColor Cyan

$screenshotSrc = Join-Path $root "my_screenshots"
if (Test-Path $screenshotSrc) {
    Write-Host "  Found my_screenshots\\ - generating LR pairs with Python..."
    @"
import pathlib
from PIL import Image

src = pathlib.Path(r'$screenshotSrc')
hr  = pathlib.Path(r'$dataDir\\train\\HR')
lr  = pathlib.Path(r'$dataDir\\train\\LR')

imgs = list(src.glob('*.png')) + list(src.glob('*.jpg'))
print(f'  Found {len(imgs)} screenshots')
for f in imgs:
    img = Image.open(f).convert('RGB')
    w, h = img.size
    if w < 512 or h < 512:
        print(f'  Skipping {f.name} (too small: {w}x{h})')
        continue

    w = (w // 4) * 4
    h = (h // 4) * 4
    img = img.crop((0, 0, w, h))
    name = f'screenshot_{f.stem}.png'
    img.save(hr / name)
    img.resize((w // 4, h // 4), Image.BICUBIC).save(lr / name)
    print(f'  Processed {f.name} -> HR {w}x{h} / LR {w//4}x{h//4}')

print('  Done.')
"@ | python -
}
else {
    Write-Host "  No my_screenshots\\ folder found - skipping."
    Write-Host "  (Drop PNGs into codex\\my_screenshots\\ and re-run to add them)"
}

# Cleanup zips/extracted staging
Write-Host "`nCleaning up staging folders..."
Remove-Item -Recurse -Force $extDir

# Final count
$trainHR = (Get-ChildItem "$dataDir\train\HR").Count
$trainLR = (Get-ChildItem "$dataDir\train\LR").Count
$valHR   = (Get-ChildItem "$dataDir\val\HR").Count
$valLR   = (Get-ChildItem "$dataDir\val\LR").Count

Write-Host "`nDone." -ForegroundColor Green
Write-Host "  train/HR : $trainHR images"
Write-Host "  train/LR : $trainLR images"
Write-Host "  val/HR   : $valHR images"
Write-Host "  val/LR   : $valLR images"

if ($trainHR -ne $trainLR) {
    Write-Host "`nWARNING: HR/LR counts don't match - check for missing files." -ForegroundColor Yellow
}

Write-Host "`nNext step:"
Write-Host "  python -m moesr.train --config debug_small --train-lr dataset/train/LR --train-hr dataset/train/HR --val-lr dataset/val/LR --val-hr dataset/val/HR --output-dir runs/smoke --batch-size 2 --steps 200 --device cuda" -ForegroundColor DarkGray
