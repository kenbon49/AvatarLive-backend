[CmdletBinding()]
param(
    [string]$Destination
)

$ErrorActionPreference = 'Stop'

# $PSScriptRoot is not initialized while parameter defaults are being bound.
if ([string]::IsNullOrWhiteSpace($Destination)) {
    $Destination = Join-Path (Split-Path -Parent $PSScriptRoot) 'CosyVoice\wheels'
}

$WheelName = 'tensorrt_cu12_libs-10.13.3.9-py2.py3-none-manylinux_2_28_x86_64.whl'
$WheelUrl = "https://pypi.nvidia.com/tensorrt-cu12-libs/$WheelName"
$WheelPath = Join-Path $Destination $WheelName

function Test-WheelArchive {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }

    try {
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $archive = [System.IO.Compression.ZipFile]::OpenRead($Path)
        try {
            return $archive.Entries.Count -gt 0
        }
        finally {
            $archive.Dispose()
        }
    }
    catch {
        return $false
    }
}

New-Item -ItemType Directory -Force -Path $Destination | Out-Null

if (Test-WheelArchive -Path $WheelPath) {
    Write-Output "TensorRT wheel is already complete: $WheelPath"
    exit 0
}

if (Test-Path -LiteralPath $WheelPath -PathType Leaf) {
    Write-Output "Resuming partial TensorRT wheel download: $WheelPath"
}
else {
    Write-Output "Downloading TensorRT wheel to: $WheelPath"
}

# curl resumes the same file after an interrupted download when the NVIDIA CDN supports ranges.
& curl.exe -fL --retry 12 --retry-all-errors --retry-delay 5 --connect-timeout 30 `
    --continue-at - --output $WheelPath $WheelUrl
if ($LASTEXITCODE -ne 0) {
    throw "TensorRT wheel download failed with curl exit code $LASTEXITCODE. Re-run this script to resume."
}

if (-not (Test-WheelArchive -Path $WheelPath)) {
    throw "Downloaded file is not a valid wheel archive. Re-run this script to resume or download again."
}

Write-Output "TensorRT wheel download complete: $WheelPath"
