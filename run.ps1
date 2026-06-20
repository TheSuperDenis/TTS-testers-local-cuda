[CmdletBinding()]
param(
    [ValidateSet("menu", "xtts", "qwen", "miotts", "luxtts")]
    [string]$Model = "menu",
    [string]$Speaker = "",
    [string]$ReferenceText = "",
    [string]$Language = "en",
    [string]$QwenLanguage = "English",
    [string]$Gpu = "",
    [switch]$Cuda128,
    [switch]$NoBuild,
    [switch]$Rebuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$script:SelectedModelExitCode = 0

function Get-Rtx5070Uuid {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) {
        throw "nvidia-smi was not found. Install or repair the NVIDIA driver first."
    }

    $rows = & nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader
    foreach ($row in $rows) {
        $columns = $row -split ",\s*"
        if ($columns.Count -ge 3 -and $columns[1] -like "*RTX 5070*") {
            return $columns[2]
        }
    }

    throw "No RTX 5070 was found in nvidia-smi output."
}

function Find-FirstExistingPath([string[]]$Candidates) {
    foreach ($candidate in $Candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    return ""
}

function Find-FirstMatchingFile([string]$Directory, [string[]]$Patterns) {
    if (-not (Test-Path -LiteralPath $Directory)) {
        return ""
    }

    foreach ($pattern in $Patterns) {
        $match = Get-ChildItem -LiteralPath $Directory -Filter $pattern -File -ErrorAction SilentlyContinue |
            Sort-Object Name |
            Select-Object -First 1
        if ($match) {
            return $match.FullName
        }
    }
    return ""
}

function Resolve-SpeakerForContainer([string]$SpeakerValue) {
    $voicesDir = Join-Path $Root "voices"
    $speakerPath = ""

    if ($SpeakerValue) {
        $speakerPath = if ([System.IO.Path]::IsPathRooted($SpeakerValue)) {
            $SpeakerValue
        } else {
            $rootCandidate = Join-Path $Root $SpeakerValue
            if (Test-Path -LiteralPath $rootCandidate) {
                $rootCandidate
            } else {
                Join-Path $voicesDir $SpeakerValue
            }
        }
    } else {
        $speakerPath = Find-FirstExistingPath @(
            (Join-Path $Root "reference.wav"),
            (Join-Path $Root "voice.wav"),
            (Join-Path $Root "speaker.wav"),
            (Join-Path $voicesDir "reference.wav"),
            (Join-Path $voicesDir "voice.wav"),
            (Join-Path $voicesDir "default.wav")
        )
        if (-not $speakerPath) {
            $speakerPath = Find-FirstMatchingFile $Root @("*reference*.wav", "*voice*.wav", "*.wav")
        }
        if (-not $speakerPath) {
            $speakerPath = Find-FirstMatchingFile $voicesDir @("*reference*.wav", "*voice*.wav", "*.wav")
        }
    }

    if (-not $speakerPath -or -not (Test-Path -LiteralPath $speakerPath)) {
        Write-Warning "Reference voice '$speakerPath' does not exist yet."
        return ""
    }

    $resolvedSpeaker = (Resolve-Path -LiteralPath $speakerPath).Path
    $resolvedVoices = (Resolve-Path -LiteralPath $voicesDir).Path
    if ($resolvedSpeaker.StartsWith($resolvedVoices, [System.StringComparison]::OrdinalIgnoreCase)) {
        $fileName = Split-Path -Leaf $resolvedSpeaker
        return "/app/voices/$fileName"
    }

    $resolvedRoot = (Resolve-Path -LiteralPath $Root).Path
    if ($resolvedSpeaker.StartsWith($resolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        $fileName = Split-Path -Leaf $resolvedSpeaker
        return "/workspace/$fileName"
    }

    throw "Speaker files must live under '$Root' or '$voicesDir' because Docker mounts only those paths."
}

function Resolve-ReferenceTextForContainer([string]$ReferenceTextValue) {
    $referencePath = ""

    if ($ReferenceTextValue) {
        $referencePath = if ([System.IO.Path]::IsPathRooted($ReferenceTextValue)) {
            $ReferenceTextValue
        } else {
            Join-Path $Root $ReferenceTextValue
        }
    } else {
        $referencePath = Find-FirstExistingPath @(
            (Join-Path $Root "reference.txt"),
            (Join-Path $Root "voice-reference.txt"),
            (Join-Path $Root "transcript.txt")
        )
        if (-not $referencePath) {
            $referencePath = Find-FirstMatchingFile $Root @("*reference*.txt", "*transcript*.txt", "voice*.txt")
        }
    }

    if (-not $referencePath -or -not (Test-Path -LiteralPath $referencePath)) {
        return ""
    }

    $resolvedReference = (Resolve-Path -LiteralPath $referencePath).Path
    $resolvedRoot = (Resolve-Path -LiteralPath $Root).Path
    if (-not $resolvedReference.StartsWith($resolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Reference text must live under '$Root' because Docker mounts that folder read-only."
    }

    $fileName = Split-Path -Leaf $resolvedReference
    return "/workspace/$fileName"
}

function Initialize-Environment {
    New-Item -ItemType Directory -Force -Path `
        (Join-Path $Root "output"), `
        (Join-Path $Root "voices"), `
        (Join-Path $Root ".cache\tts"), `
        (Join-Path $Root ".cache\huggingface"), `
        (Join-Path $Root ".cache\torch") | Out-Null

    if (-not $script:Gpu) {
        $script:Gpu = Get-Rtx5070Uuid
    }

    $speakerPath = Resolve-SpeakerForContainer $Speaker
    $referencePath = Resolve-ReferenceTextForContainer $ReferenceText

    $env:TTS_GPU = $script:Gpu
    $env:XTTS_GPU = $script:Gpu
    $env:QWEN_GPU = $script:Gpu
    $env:MIOTTS_GPU = $script:Gpu
    $env:LUXTTS_GPU = $script:Gpu
    $env:XTTS_LANGUAGE = $Language
    $env:QWEN_LANGUAGE = $QwenLanguage
    $env:XTTS_SPEAKER_WAV = $speakerPath
    $env:XTTS_SPEAKER_TEXT = $referencePath
    $env:QWEN_REF_AUDIO = $speakerPath
    $env:QWEN_REF_TEXT_FILE = $referencePath
    $env:MIOTTS_REF_AUDIO = $speakerPath
    $env:MIOTTS_REF_TEXT_FILE = $referencePath
    $env:LUXTTS_REF_AUDIO = $speakerPath
    $env:LUXTTS_REF_TEXT_FILE = $referencePath

    if ($Cuda128) {
        $env:PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
        $env:TORCH_VERSION = "2.11.0"
        $env:TORCHAUDIO_VERSION = "2.11.0"
    } else {
        $env:PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu130"
        $env:TORCH_VERSION = "2.11.0"
        $env:TORCHAUDIO_VERSION = "2.11.0"
    }

    Write-Host "GPU mask: $env:TTS_GPU"
    Write-Host "Torch wheel index: $env:PYTORCH_INDEX_URL"
    if ($speakerPath) {
        Write-Host "Reference voice: $speakerPath"
    }
    if ($referencePath) {
        Write-Host "Reference text: $referencePath"
    }
}

function Get-ServiceImageName([string]$ServiceName) {
    switch ($ServiceName) {
        "xtts" { return "xttsv2-local:cu13" }
        "qwen" { return "qwen3tts-local:cu13" }
        "miotts" { return "miotts-local:cu13" }
        "luxtts" { return "luxtts-local:cu13" }
        default { throw "Unknown service '$ServiceName'." }
    }
}

function Test-DockerImageExists([string]$ImageName) {
    docker image inspect $ImageName *> $null
    return ($LASTEXITCODE -eq 0)
}

function Remove-SelectedRunContainers([string]$ServiceName) {
    $containerIds = @(
        docker container ls --all --quiet `
            --filter "label=com.docker.compose.project=xttsv2" `
            --filter "label=com.docker.compose.service=$ServiceName" `
            --filter "label=com.docker.compose.oneoff=True"
    )

    if ($containerIds.Count -gt 0) {
        docker container rm --force $containerIds *> $null
    }
}

function Invoke-TtsContainer([string]$ServiceName) {
    $imageName = Get-ServiceImageName $ServiceName
    $imageExists = Test-DockerImageExists $imageName

    if ($NoBuild) {
        Write-Host "Skipping build for selected image: $ServiceName"
    } elseif ($Rebuild -or -not $imageExists) {
        if ($Rebuild) {
            Write-Host "Rebuilding selected image only: $ServiceName"
        } else {
        Write-Host "Selected image missing; building only: $ServiceName"
        }
        docker compose build --progress plain $ServiceName
        if ($LASTEXITCODE -ne 0) {
            $script:SelectedModelExitCode = $LASTEXITCODE
            return
        }
    } else {
        Write-Host "Selected image already exists: $imageName"
        Write-Host "Skipping build. Use -Rebuild if you want to rebuild it."
    }

    Write-Host "Loading $ServiceName now. Do not type synthesis text until the container prints READY and its prompt."
    $exitCode = 1
    try {
        docker compose run --rm --no-deps $ServiceName
        $exitCode = $LASTEXITCODE
    } finally {
        Remove-SelectedRunContainers $ServiceName
        Write-Host "Unloaded selected container: $ServiceName"
    }
    $script:SelectedModelExitCode = $exitCode
}

function Show-Menu {
    Write-Host ""
    Write-Host "TTS model launcher"
    Write-Host "1. XTTS-v2 - reference voice clone"
    Write-Host "2. Qwen3-TTS 0.6B Base - reference voice clone"
    Write-Host "3. MioTTS 0.1B - reference voice clone"
    Write-Host "4. LuxTTS 100M - reference voice clone"
    Write-Host "Q. Quit"
}

function Read-ModelChoice {
    while ($true) {
        Show-Menu
        try {
            $rawChoice = Read-Host "Choose a model"
        } catch {
            return ""
        }

        if ($null -eq $rawChoice) {
            return ""
        }

        $choice = $rawChoice.Trim().ToLowerInvariant()
        switch ($choice) {
            "1" { return "xtts" }
            "xtts" { return "xtts" }
            "2" { return "qwen" }
            "qwen" { return "qwen" }
            "3" { return "miotts" }
            "mio" { return "miotts" }
            "miotts" { return "miotts" }
            "4" { return "luxtts" }
            "lux" { return "luxtts" }
            "luxtts" { return "luxtts" }
            "q" { return "" }
            "quit" { return "" }
            default { Write-Host "Choose 1, 2, 3, 4, or Q." }
        }
    }
}

function Invoke-SelectedModel([string]$SelectedModel) {
    switch ($SelectedModel) {
        "xtts" {
            Write-Host ""
            Write-Host "Starting XTTS-v2. Wait for xttsv2> before typing."
            Invoke-TtsContainer "xtts"
            return
        }
        "qwen" {
            if (-not $env:QWEN_REF_AUDIO -or -not $env:QWEN_REF_TEXT_FILE) {
                Write-Error "Qwen voice cloning requires both a reference WAV and a reference transcript."
                $script:SelectedModelExitCode = 1
                return
            }
            Write-Host ""
            Write-Host "Starting Qwen3-TTS 0.6B. Wait for qwen0.6btts> before typing."
            Invoke-TtsContainer "qwen"
            return
        }
        "miotts" {
            if (-not $env:MIOTTS_REF_AUDIO) {
                Write-Error "MioTTS voice cloning requires a reference WAV."
                $script:SelectedModelExitCode = 1
                return
            }
            Write-Host ""
            Write-Host "Starting MioTTS 0.1B. Wait for miotts0.1b> before typing."
            Invoke-TtsContainer "miotts"
            return
        }
        "luxtts" {
            if (-not $env:LUXTTS_REF_AUDIO -or -not $env:LUXTTS_REF_TEXT_FILE) {
                Write-Error "LuxTTS voice cloning requires both a reference WAV and a reference transcript."
                $script:SelectedModelExitCode = 1
                return
            }
            Write-Host ""
            Write-Host "Starting LuxTTS 100M. Wait for luxtts100m> before typing."
            Invoke-TtsContainer "luxtts"
            return
        }
        default {
            Write-Error "Unknown model '$SelectedModel'."
            $script:SelectedModelExitCode = 1
            return
        }
    }
}

Initialize-Environment

if ($Model -ne "menu") {
    Invoke-SelectedModel $Model
    exit $script:SelectedModelExitCode
}

while ($true) {
    $selected = Read-ModelChoice
    if (-not $selected) {
        exit 0
    }

    Invoke-SelectedModel $selected
    $exitCode = $script:SelectedModelExitCode
    Write-Host ""
    Write-Host "Container exited with code $exitCode."
}
