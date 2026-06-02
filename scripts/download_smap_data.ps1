# Download NASA SMAP/MSL telemetry (.npy) + labeled_anomalies.csv into data\raw\SMAP_MSL\
#
# The legacy S3 bucket (telemanom/data.zip) returns AccessDenied/403 - use Hugging Face or Kaggle.
#
# Usage (from repo root):
#   .\scripts\download_smap_data.ps1
#   .\scripts\download_smap_data.ps1 -Source kaggle   # requires Kaggle API credentials

param(
    [string]$Dest = "data\raw\SMAP_MSL",
    [ValidateSet("huggingface", "kaggle")]
    [string]$Source = "huggingface"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
$DestPath = Join-Path $RepoRoot $Dest
$TrainDir = Join-Path $DestPath "train"
$TestDir = Join-Path $DestPath "test"
$LabelsFile = Join-Path $DestPath "labeled_anomalies.csv"
$LabelsUrl = "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"
$HfBase = "https://huggingface.co/datasets/appleparan/telemanom/resolve/main/data/data"

function Ensure-Dir([string]$Path) {
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Force -Path $Path | Out-Null
    }
}

function Download-File([string]$Url, [string]$OutPath) {
    if (Test-Path $OutPath) {
        Write-Host "  skip (exists): $OutPath"
        return
    }
    Write-Host "  get: $Url"
    Invoke-WebRequest -Uri $Url -OutFile $OutPath -UseBasicParsing
}

function Get-ChannelIds([string]$CsvPath) {
    if (-not (Test-Path $CsvPath)) {
        throw "Missing $CsvPath - download labels first."
    }
    Import-Csv $CsvPath | ForEach-Object { $_.chan_id } | Select-Object -Unique
}

function Install-HuggingFaceLayout {
    Ensure-Dir $TrainDir
    Ensure-Dir $TestDir

    Write-Host "Downloading labeled_anomalies.csv ..."
    Download-File $LabelsUrl $LabelsFile

    $channels = @(Get-ChannelIds $LabelsFile)
    Write-Host "Downloading $($channels.Count) channels from Hugging Face (appleparan/telemanom) ..."

    $i = 0
    foreach ($ch in $channels) {
        $i++
        Write-Host "[$i/$($channels.Count)] $ch"
        Download-File "$HfBase/train/$ch.npy" (Join-Path $TrainDir "$ch.npy")
        Download-File "$HfBase/test/$ch.npy" (Join-Path $TestDir "$ch.npy")
    }
}

function Install-KaggleLayout {
    if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
        Write-Error @"
Kaggle CLI not found. Install with: pip install kaggle
Then place your API token at: $env:USERPROFILE\.kaggle\kaggle.json
See: https://www.kaggle.com/docs/api#authentication
"@
    }

    Ensure-Dir $DestPath
    $zipName = "nasa-anomaly-detection-dataset-smap-msl.zip"
    $zipPath = Join-Path $DestPath $zipName

    Push-Location $DestPath
    try {
        if (-not (Test-Path $zipPath)) {
            Write-Host "Downloading Kaggle dataset patrickfleith/nasa-anomaly-detection-dataset-smap-msl ..."
            kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl
        }
        Write-Host "Extracting $zipName ..."
        Expand-Archive -Force $zipPath -DestinationPath .

        $nestedTrain = Join-Path $DestPath "data\data\train"
        $nestedTest = Join-Path $DestPath "data\data\test"
        if (-not (Test-Path $nestedTrain)) {
            throw "Unexpected zip layout (expected data\data\train). Check the Kaggle dataset structure."
        }

        if (Test-Path $TrainDir) { Remove-Item -Recurse -Force $TrainDir }
        if (Test-Path $TestDir) { Remove-Item -Recurse -Force $TestDir }
        Move-Item -Force $nestedTrain $TrainDir
        Move-Item -Force $nestedTest $TestDir
        Remove-Item -Recurse -Force (Join-Path $DestPath "data") -ErrorAction SilentlyContinue
        Remove-Item -Force $zipPath -ErrorAction SilentlyContinue

        Write-Host "Downloading labeled_anomalies.csv ..."
        Download-File $LabelsUrl $LabelsFile
    }
    finally {
        Pop-Location
    }
}

Write-Host "Target: $DestPath"
switch ($Source) {
    "huggingface" { Install-HuggingFaceLayout }
    "kaggle" { Install-KaggleLayout }
}

$trainCount = @(Get-ChildItem $TrainDir -Filter "*.npy" -ErrorAction SilentlyContinue).Count
$testCount = @(Get-ChildItem $TestDir -Filter "*.npy" -ErrorAction SilentlyContinue).Count
Write-Host ""
Write-Host "Done. train/*.npy: $trainCount  test/*.npy: $testCount  labels: $(Test-Path $LabelsFile)"
if ($trainCount -lt 1 -or $testCount -lt 1) {
    Write-Error "Download incomplete - check network or try -Source kaggle"
}
