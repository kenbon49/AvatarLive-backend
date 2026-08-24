[CmdletBinding()]
param(
    [ValidateSet("All", "OpenVoice", "MuseTalk", "ServerTotal")]
    [string]$Service = "All",
    [string]$OpenVoiceEnv = "openvoice",
    [string]$MuseTalkEnv = "musetalk",
    [int]$OpenVoicePort = 8084,
    [int]$MuseTalkPort = 8083,
    [int]$ServerTotalPort = 8080,
    [int]$StartupTimeoutSeconds = 600,
    [switch]$NoReload,
    [Parameter(DontShow = $true)]
    [ValidateSet("", "OpenVoice", "MuseTalk", "ServerTotal")]
    [string]$ChildService = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LogDirectory = Join-Path $ProjectRoot ".local-test\logs"

# Some launchers inject both PATH and Path. Windows PowerShell 5.1 cannot pass
# that duplicate pair through Start-Process even though Windows treats them alike.
$processEnvironment = [Environment]::GetEnvironmentVariables("Process")
if ($processEnvironment.Contains("PATH") -and $processEnvironment.Contains("Path")) {
    [Environment]::SetEnvironmentVariable("PATH", $null, "Process")
}

function Import-DotEnv {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return
    }

    foreach ($line in Get-Content -LiteralPath $Path -Encoding UTF8) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#") -or -not $trimmed.Contains("=")) {
            continue
        }

        $name, $value = $trimmed.Split("=", 2)
        $name = $name.Trim()
        $value = $value.Trim()
        if ($value.Length -ge 2 -and (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'")))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($name, "Process"))) {
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

function Set-DefaultEnvironmentVariable {
    param([string]$Name, [string]$Value)

    if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($Name, "Process"))) {
        [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
    }
}

function Get-CondaEnvironmentPython {
    param([string]$EnvironmentName)

    if ([System.IO.Path]::IsPathRooted($EnvironmentName) -and (Test-Path -LiteralPath $EnvironmentName -PathType Container)) {
        $environmentRoot = (Resolve-Path -LiteralPath $EnvironmentName).Path
    }
    else {
        $condaBase = (& conda info --base).Trim()
        if ($LASTEXITCODE -ne 0 -or -not $condaBase) {
            throw "Unable to resolve the Conda base directory."
        }
        $environmentRoot = if ($EnvironmentName -eq "base") {
            $condaBase
        }
        else {
            Join-Path $condaBase "envs\$EnvironmentName"
        }
    }

    $pythonExecutable = Join-Path $environmentRoot "python.exe"
    if (-not (Test-Path -LiteralPath $pythonExecutable -PathType Leaf)) {
        throw "Conda environment '$EnvironmentName' was not found at $environmentRoot."
    }

    # Direct invocation avoids a conda-run temp-file race when GPU services start in parallel.
    $condaPathEntries = @(
        $environmentRoot,
        (Join-Path $environmentRoot "Library\mingw-w64\bin"),
        (Join-Path $environmentRoot "Library\usr\bin"),
        (Join-Path $environmentRoot "Library\bin"),
        (Join-Path $environmentRoot "Scripts"),
        (Join-Path $environmentRoot "bin")
    ) | Where-Object { Test-Path -LiteralPath $_ -PathType Container }
    $env:Path = (($condaPathEntries + $env:Path) -join ";")
    $env:CONDA_PREFIX = $environmentRoot
    $env:CONDA_DEFAULT_ENV = $EnvironmentName
    return $pythonExecutable
}

function Invoke-ChildService {
    param([string]$Name)

    Import-DotEnv (Join-Path $ProjectRoot ".env")
    Set-DefaultEnvironmentVariable "PYTHONUNBUFFERED" "1"
    Set-DefaultEnvironmentVariable "CUDA_VISIBLE_DEVICES" "0"

    switch ($Name) {
        "OpenVoice" {
            $pythonExecutable = Get-CondaEnvironmentPython $OpenVoiceEnv
            Set-DefaultEnvironmentVariable "HF_HUB_OFFLINE" "1"
            Set-DefaultEnvironmentVariable "TRANSFORMERS_OFFLINE" "1"
            Set-DefaultEnvironmentVariable "OPENVOICE_DEVICE" "cuda:0"
            Set-DefaultEnvironmentVariable "OPENVOICE_DEFAULT_VOICE" "default"
            Set-DefaultEnvironmentVariable "OPENVOICE_DEFAULT_REFERENCE" (Join-Path $ProjectRoot "data\input\audio\yongen.wav")
            Set-Location -LiteralPath (Join-Path $ProjectRoot "OpenVoice")
            & $pythonExecutable -u server.py --host 0.0.0.0 --port $OpenVoicePort
        }
        "MuseTalk" {
            $pythonExecutable = Get-CondaEnvironmentPython $MuseTalkEnv
            Set-DefaultEnvironmentVariable "MUSETALK_FPS" "10"
            Set-DefaultEnvironmentVariable "MUSETALK_BATCH_SIZE" "10"
            Set-DefaultEnvironmentVariable "MUSETALK_ALLOW_TF32" "1"
            Set-Location -LiteralPath $ProjectRoot
            & $pythonExecutable -u -m accelerated.server --host 0.0.0.0 --port $MuseTalkPort
        }
        "ServerTotal" {
            $pythonExecutable = Get-CondaEnvironmentPython $MuseTalkEnv
            Import-DotEnv (Join-Path $ProjectRoot "llm_inference\.env")
            Set-DefaultEnvironmentVariable "OPENVOICE_URL" "http://127.0.0.1:$OpenVoicePort"
            Set-DefaultEnvironmentVariable "OPENVOICE_DEFAULT_VOICE" "default"
            Set-DefaultEnvironmentVariable "MUSETALK_WS_URL" "ws://127.0.0.1:$MuseTalkPort/v1/stream"
            Set-DefaultEnvironmentVariable "PIPELINE_REQUEST_TIMEOUT" "300"
            Set-DefaultEnvironmentVariable "PIPELINE_FIRST_UNIT_MIN_CHARS" "1"
            Set-DefaultEnvironmentVariable "PIPELINE_TARGET_UNIT_CHARS" "1"
            Set-DefaultEnvironmentVariable "PIPELINE_COALESCE_HARD_DELIMITERS" "0"
            Set-DefaultEnvironmentVariable "PIPELINE_AUDIO_QUEUE_SIZE" "12"
            Set-DefaultEnvironmentVariable "PIPELINE_TTS_CHUNK_SECONDS" "0.5"
            Set-DefaultEnvironmentVariable "PIPELINE_TTS_STEADY_CHUNK_SECONDS" "1.0"
            Set-DefaultEnvironmentVariable "PIPELINE_TTS_MIN_TAIL_SECONDS" "0.25"
            Set-DefaultEnvironmentVariable "PIPELINE_PLAYBACK_BUFFER_SECONDS" "1.5"
            Set-Location -LiteralPath $ProjectRoot
            $uvicornArgs = @("-u", "-m", "uvicorn", "server_total.app:app", "--host", "0.0.0.0", "--port", $ServerTotalPort)
            if (-not $NoReload) {
                $uvicornArgs += "--reload"
            }
            & $pythonExecutable @uvicornArgs
        }
    }

    exit $LASTEXITCODE
}

if ($ChildService) {
    Invoke-ChildService $ChildService
}

function Test-TcpPort {
    param([int]$Port)

    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $task = $client.ConnectAsync("127.0.0.1", $Port)
        return $task.Wait(300) -and $client.Connected
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

function Assert-PortAvailable {
    param([string]$Name, [int]$Port)

    if (Test-TcpPort $Port) {
        throw "$Name cannot start because port $Port is already in use."
    }
}

function Start-ManagedService {
    param([string]$Name, [int]$Port)

    Assert-PortAvailable $Name $Port
    New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
    $stdoutPath = Join-Path $LogDirectory "$($Name.ToLowerInvariant()).out.log"
    $stderrPath = Join-Path $LogDirectory "$($Name.ToLowerInvariant()).err.log"
    $powerShellExecutable = Join-Path $PSHOME "powershell.exe"
    if (-not (Test-Path -LiteralPath $powerShellExecutable)) {
        $powerShellExecutable = Join-Path $PSHOME "pwsh.exe"
    }

    $arguments = @(
        "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", ('"{0}"' -f $PSCommandPath),
        "-ChildService", $Name,
        "-OpenVoiceEnv", $OpenVoiceEnv,
        "-MuseTalkEnv", $MuseTalkEnv,
        "-OpenVoicePort", $OpenVoicePort,
        "-MuseTalkPort", $MuseTalkPort,
        "-ServerTotalPort", $ServerTotalPort
    )
    if ($NoReload) {
        $arguments += "-NoReload"
    }

    $process = Start-Process -FilePath $powerShellExecutable -ArgumentList $arguments -WorkingDirectory $ProjectRoot -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
    Write-Host ("[{0}] started (PID {1}, port {2})" -f $Name, $process.Id, $Port) -ForegroundColor Cyan
    return [PSCustomObject]@{
        Name = $Name
        Port = $Port
        Process = $process
        Stdout = $stdoutPath
        Stderr = $stderrPath
    }
}

function Wait-ServiceReady {
    param([PSCustomObject]$ManagedService, [string]$HealthUrl, [string]$ExpectedText = "")

    $deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
    Write-Host "[$($ManagedService.Name)] waiting for $HealthUrl ..."
    while ((Get-Date) -lt $deadline) {
        if ($ManagedService.Process.HasExited) {
            $stderrTail = if (Test-Path $ManagedService.Stderr) { (Get-Content $ManagedService.Stderr -Tail 30) -join [Environment]::NewLine } else { "" }
            throw "$($ManagedService.Name) exited with code $($ManagedService.Process.ExitCode).`n$stderrTail"
        }
        try {
            $response = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 3
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300 -and (-not $ExpectedText -or $response.Content.Contains($ExpectedText))) {
                Write-Host "[$($ManagedService.Name)] ready" -ForegroundColor Green
                return
            }
        }
        catch {
            # Model initialization can take several minutes; retry until the shared timeout.
        }
        Start-Sleep -Seconds 2
        $ManagedService.Process.Refresh()
    }
    throw "$($ManagedService.Name) did not become ready within $StartupTimeoutSeconds seconds. Logs: $($ManagedService.Stdout), $($ManagedService.Stderr)"
}

function Stop-ManagedServices {
    param([System.Collections.Generic.List[object]]$Services)

    $servicesInStopOrder = $Services.ToArray()
    [array]::Reverse($servicesInStopOrder)
    foreach ($managed in $servicesInStopOrder) {
        $managed.Process.Refresh()
        if (-not $managed.Process.HasExited) {
            Write-Host "[$($managed.Name)] stopping PID $($managed.Process.Id) ..."
            & taskkill.exe /PID $managed.Process.Id /T /F 2>&1 | Out-Null
        }
    }
}

$managedServices = [System.Collections.Generic.List[object]]::new()
try {
    if ($Service -in @("All", "OpenVoice")) {
        $managedServices.Add((Start-ManagedService "OpenVoice" $OpenVoicePort))
    }
    if ($Service -in @("All", "MuseTalk")) {
        $managedServices.Add((Start-ManagedService "MuseTalk" $MuseTalkPort))
    }

    foreach ($managed in @($managedServices)) {
        if ($managed.Name -eq "OpenVoice") {
            Wait-ServiceReady $managed "http://127.0.0.1:$OpenVoicePort/health" '"status":"ok"'
        }
        elseif ($managed.Name -eq "MuseTalk") {
            Wait-ServiceReady $managed "http://127.0.0.1:$MuseTalkPort/readiness" '"status":"ready"'
        }
    }

    if ($Service -in @("All", "ServerTotal")) {
        $serverTotal = Start-ManagedService "ServerTotal" $ServerTotalPort
        $managedServices.Add($serverTotal)
        Wait-ServiceReady $serverTotal "http://127.0.0.1:$ServerTotalPort/health"
    }

    Write-Host ""
    Write-Host "Local test services are running. Press Ctrl+C to stop all managed processes." -ForegroundColor Green
    Write-Host "Gateway:   http://127.0.0.1:$ServerTotalPort"
    Write-Host "OpenVoice: http://127.0.0.1:$OpenVoicePort"
    Write-Host "MuseTalk:  http://127.0.0.1:$MuseTalkPort"
    Write-Host "Logs:      $LogDirectory"

    while ($true) {
        Start-Sleep -Seconds 2
        foreach ($managed in $managedServices) {
            $managed.Process.Refresh()
            if ($managed.Process.HasExited) {
                throw "$($managed.Name) exited unexpectedly with code $($managed.Process.ExitCode). Check $($managed.Stderr)."
            }
        }
    }
}
finally {
    Stop-ManagedServices $managedServices
}
