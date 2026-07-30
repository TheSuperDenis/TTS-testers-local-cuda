[CmdletBinding()]
param(
    [ValidateSet("menu", "xtts", "qwen", "miotts", "luxtts", "piper", "kokoro", "kitten", "chatterbox", "sopro", "pockettts", "moss", "f5tts", "supertonic")]
    [string]$Model = "menu",
    [string]$Speaker = "",
    [string]$ReferenceText = "",
    [string]$Language = "en",
    [string]$QwenLanguage = "English",
    [string]$XttsVoice = "",
    [string]$QwenVoice = "",
    [string]$MioVoice = "",
    [string]$LuxVoice = "",
    [string]$PiperVoice = "",
    [string]$KokoroVoice = "",
    [string]$KittenSize = "",
    [string]$KittenVoice = "",
    [string]$SupertonicVoice = "",
    [string]$SupertonicLanguage = "en",
    [string]$ChatterboxVoice = "",
    [string]$SoproVoice = "",
    [string]$PocketVoice = "",
    [string]$MossVoice = "",
    [string]$F5Voice = "",
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
        (Join-Path $Root ".cache\kokoro"), `
        (Join-Path $Root ".cache\kitten"), `
        (Join-Path $Root ".cache\supertonic"), `
        (Join-Path $Root ".cache\chatterbox"), `
        (Join-Path $Root ".cache\sopro"), `
        (Join-Path $Root ".cache\pockettts"), `
        (Join-Path $Root ".cache\moss"), `
        (Join-Path $Root ".cache\f5tts"), `
        (Join-Path $Root ".cache\miotts"), `
        (Join-Path $Root ".cache\piper"), `
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
    $env:PIPER_GPU = $script:Gpu
    $env:KOKORO_GPU = $script:Gpu
    $env:KITTEN_GPU = $script:Gpu
    $env:SUPERTONIC_GPU = $script:Gpu
    $env:CHATTERBOX_GPU = $script:Gpu
    $env:SOPRO_GPU = $script:Gpu
    $env:MOSS_GPU = $script:Gpu
    $env:F5TTS_GPU = $script:Gpu
    $env:XTTS_LANGUAGE = $Language
    $env:QWEN_LANGUAGE = $QwenLanguage
    $env:SUPERTONIC_LANGUAGE = $SupertonicLanguage
    $env:XTTS_SPEAKER_WAV = $speakerPath
    $env:XTTS_SPEAKER_TEXT = $referencePath
    $env:QWEN_REF_AUDIO = $speakerPath
    $env:QWEN_REF_TEXT_FILE = $referencePath
    $env:MIOTTS_REF_AUDIO = $speakerPath
    $env:MIOTTS_REF_TEXT_FILE = $referencePath
    $env:LUXTTS_REF_AUDIO = $speakerPath
    $env:LUXTTS_REF_TEXT_FILE = $referencePath
    $env:CHATTERBOX_REF_AUDIO = $speakerPath
    $env:SOPRO_REF_AUDIO = $speakerPath
    $env:POCKETTTS_REF_AUDIO = $speakerPath
    $env:MOSS_REF_AUDIO = $speakerPath
    $env:F5TTS_REF_AUDIO = $speakerPath
    $env:F5TTS_REF_TEXT_FILE = $referencePath

    if ($Cuda128) {
        $env:PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
        $env:TORCH_VERSION = "2.11.0"
        $env:TORCHAUDIO_VERSION = "2.11.0"
        $env:ORT_INDEX_URL = "https://pypi.org/simple"
        $env:ORT_PRE = "0"
    } else {
        $env:PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu130"
        $env:TORCH_VERSION = "2.11.0"
        $env:TORCHAUDIO_VERSION = "2.11.0"
        $env:ORT_INDEX_URL = "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/"
        $env:ORT_PRE = "1"
    }

    Write-Host "GPU mask: $env:TTS_GPU"
    Write-Host "Torch wheel index: $env:PYTORCH_INDEX_URL"
}

function Get-ServiceImageName([string]$ServiceName) {
    switch ($ServiceName) {
        "xtts" { return "xttsv2-local:cu13" }
        "qwen" { return "qwen3tts-local:cu13" }
        "miotts" { return "miotts-local:cu13" }
        "luxtts" { return "luxtts-local:cu13" }
        "piper" { return "piper-local:cu13" }
        "kokoro" { return "kokoro-local:cu13" }
        "kitten" { return "kitten-local:cu13" }
        "supertonic" { return "supertonic3-local:cu13" }
        "chatterbox" { return "chatterbox-local:cu13" }
        "sopro" { return "sopro-local:cu13" }
        "pockettts" { return "pockettts-local:cpu" }
        "moss" { return "moss-tts-nano-local:cu13" }
        "f5tts" { return "f5tts-local:cu13" }
        default { throw "Unknown service '$ServiceName'." }
    }
}

function ConvertTo-PiperVoiceKey([string]$Value) {
    return (($Value.Trim().ToLowerInvariant() -replace "[^a-z0-9]+", "-").Trim("-"))
}

function Get-PiperVoiceCatalog {
    $catalogPath = Join-Path $Root "piper_service\voice_catalog.json"
    if (-not (Test-Path -LiteralPath $catalogPath)) {
        throw "Piper voice catalog was not found at '$catalogPath'."
    }

    return @(Get-Content -Raw -LiteralPath $catalogPath | ConvertFrom-Json)
}

function Find-PiperVoice([string]$VoiceValue, [object[]]$Catalog = @()) {
    if (-not $VoiceValue) {
        return $null
    }
    if ($Catalog.Count -eq 0) {
        $Catalog = Get-PiperVoiceCatalog
    }

    $target = ConvertTo-PiperVoiceKey $VoiceValue
    foreach ($voice in $Catalog) {
        $modelStem = [System.IO.Path]::GetFileNameWithoutExtension([string]$voice.modelFile)
        $aliases = @(
            [string]$voice.id,
            [string]$voice.displayName,
            [string]$voice.voiceName,
            $modelStem
        )
        foreach ($alias in $aliases) {
            if ((ConvertTo-PiperVoiceKey $alias) -eq $target) {
                return $voice
            }
        }
    }

    return $null
}

function Show-PiperVoiceGroups {
    Write-Host ""
    Write-Host "Piper voice picker"
    Write-Host "1. American female voices"
    Write-Host "2. British female voices"
    Write-Host "3. All high-quality voices"
    Write-Host "B. Back"
    Write-Host "Q. Quit"
}

function Show-PiperVoices([object[]]$Voices, [string]$GroupLabel) {
    Write-Host ""
    Write-Host "$GroupLabel Piper voices"
    for ($index = 0; $index -lt $Voices.Count; $index++) {
        $voice = $Voices[$index]
        $number = $index + 1
        Write-Host "$number. $($voice.displayName) - $($voice.quality)"
    }
    Write-Host "B. Back"
    Write-Host "Q. Quit"
}

function Read-PiperVoiceChoice([string]$PreferredVoice) {
    $catalog = Get-PiperVoiceCatalog
    if ($PreferredVoice) {
        $voice = Find-PiperVoice $PreferredVoice $catalog
        if (-not $voice) {
            throw "Unknown Piper voice '$PreferredVoice'. Use one of the catalog ids such as gb-cori or us-ljspeech."
        }
        return $voice
    }

    while ($true) {
        Show-PiperVoiceGroups
        try {
            $rawGroup = Read-Host "Choose a voice group"
        } catch {
            return $null
        }
        if ($null -eq $rawGroup) {
            return $null
        }

        $groupChoice = $rawGroup.Trim().ToLowerInvariant()
        $region = ""
        switch ($groupChoice) {
            "1" { $region = "american" }
            "american" { $region = "american" }
            "us" { $region = "american" }
            "usa" { $region = "american" }
            "2" { $region = "british" }
            "british" { $region = "british" }
            "uk" { $region = "british" }
            "gb" { $region = "british" }
            "3" { $region = "all-high" }
            "high" { $region = "all-high" }
            "all-high" { $region = "all-high" }
            "all high" { $region = "all-high" }
            "b" { return $null }
            "back" { return $null }
            "q" { return $null }
            "quit" { return $null }
            default {
                $directVoice = Find-PiperVoice $groupChoice $catalog
                if ($directVoice) {
                    return $directVoice
                }
                Write-Host "Choose 1, 2, 3, B, or Q."
                continue
            }
        }

        $voices = if ($region -eq "all-high") {
            @($catalog | Where-Object { $_.quality -eq "high" })
        } else {
            @($catalog | Where-Object { $_.region -eq $region })
        }
        $groupLabel = switch ($region) {
            "american" { "American female" }
            "british" { "British female" }
            "all-high" { "All high-quality" }
            default { "Piper" }
        }
        while ($true) {
            Show-PiperVoices $voices $groupLabel
            try {
                $rawVoice = Read-Host "Choose a Piper voice"
            } catch {
                return $null
            }
            if ($null -eq $rawVoice) {
                return $null
            }

            $voiceChoice = $rawVoice.Trim()
            $voiceKey = $voiceChoice.ToLowerInvariant()
            if ($voiceKey -in @("b", "back")) {
                break
            }
            if ($voiceKey -in @("q", "quit")) {
                return $null
            }
            if ($voiceChoice -match "^\d+$") {
                $voiceIndex = [int]$voiceChoice
                if ($voiceIndex -ge 1 -and $voiceIndex -le $voices.Count) {
                    return $voices[$voiceIndex - 1]
                }
            }

            $voice = Find-PiperVoice $voiceChoice $voices
            if ($voice) {
                return $voice
            }
            Write-Host "Choose one of the listed voice numbers, B, or Q."
        }
    }
}

function Set-PiperVoiceEnvironment([object]$Voice) {
    $env:PIPER_MODEL_FILE = [string]$Voice.modelFile
    $env:PIPER_CONFIG_FILE = [string]$Voice.configFile
    $env:PIPER_VOICE_NAME = [string]$Voice.voiceName
    Write-Host "Piper voice: $($Voice.displayName) ($($Voice.regionLabel), $($Voice.quality))"
}

function Write-ReferenceInputs {
    if ($env:XTTS_SPEAKER_WAV) {
        Write-Host "Reference voice: $env:XTTS_SPEAKER_WAV"
    }
    if ($env:XTTS_SPEAKER_TEXT) {
        Write-Host "Reference text: $env:XTTS_SPEAKER_TEXT"
    }
}

function Get-KokoroVoiceCatalog {
    $catalogPath = Join-Path $Root "kokoro_service\voice_catalog.json"
    if (-not (Test-Path -LiteralPath $catalogPath)) {
        throw "Kokoro voice catalog was not found at '$catalogPath'."
    }

    return @(Get-Content -Raw -LiteralPath $catalogPath | ConvertFrom-Json)
}

function Find-KokoroVoice([string]$VoiceValue, [object[]]$Catalog = @()) {
    if (-not $VoiceValue) {
        return $null
    }
    if ($Catalog.Count -eq 0) {
        $Catalog = Get-KokoroVoiceCatalog
    }

    $target = ConvertTo-PiperVoiceKey $VoiceValue
    foreach ($voice in $Catalog) {
        $aliases = @(
            [string]$voice.id,
            [string]$voice.displayName
        )
        foreach ($alias in $aliases) {
            if ((ConvertTo-PiperVoiceKey $alias) -eq $target) {
                return $voice
            }
        }
    }

    return $null
}

function Show-KokoroVoiceGroups {
    Write-Host ""
    Write-Host "Kokoro-82M voice picker"
    Write-Host "1. American female voices"
    Write-Host "2. British female voices"
    Write-Host "B. Back"
    Write-Host "Q. Quit"
}

function Show-KokoroVoices([object[]]$Voices, [string]$GroupLabel) {
    Write-Host ""
    Write-Host "$GroupLabel Kokoro voices"
    for ($index = 0; $index -lt $Voices.Count; $index++) {
        $voice = $Voices[$index]
        $number = $index + 1
        Write-Host "$number. $($voice.displayName) - $($voice.id) - grade $($voice.grade)"
    }
    Write-Host "B. Back"
    Write-Host "Q. Quit"
}

function Read-KokoroVoiceChoice([string]$PreferredVoice) {
    $catalog = Get-KokoroVoiceCatalog
    if ($PreferredVoice) {
        $voice = Find-KokoroVoice $PreferredVoice $catalog
        if (-not $voice) {
            throw "Unknown Kokoro voice '$PreferredVoice'. Use one of the catalog ids such as af_heart or bf_emma."
        }
        return $voice
    }

    while ($true) {
        Show-KokoroVoiceGroups
        try {
            $rawGroup = Read-Host "Choose a voice group"
        } catch {
            return $null
        }
        if ($null -eq $rawGroup) {
            return $null
        }

        $groupChoice = $rawGroup.Trim().ToLowerInvariant()
        $region = ""
        switch ($groupChoice) {
            "1" { $region = "american" }
            "american" { $region = "american" }
            "us" { $region = "american" }
            "usa" { $region = "american" }
            "2" { $region = "british" }
            "british" { $region = "british" }
            "uk" { $region = "british" }
            "gb" { $region = "british" }
            "b" { return $null }
            "back" { return $null }
            "q" { return $null }
            "quit" { return $null }
            default {
                $directVoice = Find-KokoroVoice $groupChoice $catalog
                if ($directVoice) {
                    return $directVoice
                }
                Write-Host "Choose 1, 2, B, or Q."
                continue
            }
        }

        $voices = @($catalog | Where-Object { $_.region -eq $region })
        $groupLabel = if ($region -eq "american") { "American female" } else { "British female" }
        while ($true) {
            Show-KokoroVoices $voices $groupLabel
            try {
                $rawVoice = Read-Host "Choose a Kokoro voice"
            } catch {
                return $null
            }
            if ($null -eq $rawVoice) {
                return $null
            }

            $voiceChoice = $rawVoice.Trim()
            $voiceKey = $voiceChoice.ToLowerInvariant()
            if ($voiceKey -in @("b", "back")) {
                break
            }
            if ($voiceKey -in @("q", "quit")) {
                return $null
            }
            if ($voiceChoice -match "^\d+$") {
                $voiceIndex = [int]$voiceChoice
                if ($voiceIndex -ge 1 -and $voiceIndex -le $voices.Count) {
                    return $voices[$voiceIndex - 1]
                }
            }

            $voice = Find-KokoroVoice $voiceChoice $voices
            if ($voice) {
                return $voice
            }
            Write-Host "Choose one of the listed voice numbers, B, or Q."
        }
    }
}

function Set-KokoroVoiceEnvironment([object]$Voice) {
    $env:KOKORO_VOICE_ID = [string]$Voice.id
    $env:KOKORO_VOICE_NAME = "$($Voice.regionLabel) $($Voice.displayName)"
    $env:KOKORO_LANG_CODE = [string]$Voice.langCode
    Write-Host "Kokoro voice: $($Voice.displayName) ($($Voice.regionLabel), grade $($Voice.grade))"
}

function Get-KittenVoiceCatalog {
    $catalogPath = Join-Path $Root "kitten_service\voice_catalog.json"
    if (-not (Test-Path -LiteralPath $catalogPath)) {
        throw "KittenTTS voice catalog was not found at '$catalogPath'."
    }

    return @(Get-Content -Raw -LiteralPath $catalogPath | ConvertFrom-Json)
}

function Get-KittenModelCatalog {
    $catalogPath = Join-Path $Root "kitten_service\model_catalog.json"
    if (-not (Test-Path -LiteralPath $catalogPath)) {
        throw "KittenTTS model catalog was not found at '$catalogPath'."
    }

    return @(Get-Content -Raw -LiteralPath $catalogPath | ConvertFrom-Json)
}

function Find-KittenModel([string]$ModelValue, [object[]]$Catalog = @()) {
    if (-not $ModelValue) {
        return $null
    }
    if ($Catalog.Count -eq 0) {
        $Catalog = Get-KittenModelCatalog
    }

    $target = ConvertTo-PiperVoiceKey $ModelValue
    foreach ($modelOption in $Catalog) {
        $aliases = @(
            [string]$modelOption.id,
            [string]$modelOption.displayName,
            [string]$modelOption.modelId,
            [string]$modelOption.parameters
        )
        foreach ($alias in $aliases) {
            if ((ConvertTo-PiperVoiceKey $alias) -eq $target) {
                return $modelOption
            }
        }
    }

    return $null
}

function Show-KittenModels([object[]]$Models) {
    Write-Host ""
    Write-Host "KittenTTS model sizes"
    for ($index = 0; $index -lt $Models.Count; $index++) {
        $modelOption = $Models[$index]
        $number = $index + 1
        Write-Host "$number. $($modelOption.displayName) - approximately $($modelOption.downloadSizeMb) MB download"
    }
    Write-Host "Q. Quit"
}

function Read-KittenModelChoice([string]$PreferredModel) {
    $catalog = Get-KittenModelCatalog
    if ($PreferredModel) {
        $modelOption = Find-KittenModel $PreferredModel $catalog
        if (-not $modelOption) {
            throw "Unknown KittenTTS size '$PreferredModel'. Use nano, micro, or mini."
        }
        return $modelOption
    }

    while ($true) {
        Show-KittenModels $catalog
        try {
            $rawModel = Read-Host "Choose a KittenTTS model size"
        } catch {
            return $null
        }
        if ($null -eq $rawModel) {
            return $null
        }

        $modelChoice = $rawModel.Trim()
        $modelKey = $modelChoice.ToLowerInvariant()
        if ($modelKey -in @("q", "quit")) {
            return $null
        }
        if ($modelChoice -match "^\d+$") {
            $modelIndex = [int]$modelChoice
            if ($modelIndex -ge 1 -and $modelIndex -le $catalog.Count) {
                return $catalog[$modelIndex - 1]
            }
        }

        $modelOption = Find-KittenModel $modelChoice $catalog
        if ($modelOption) {
            return $modelOption
        }
        Write-Host "Choose one of the listed model numbers or Q."
    }
}

function Set-KittenModelEnvironment([object]$ModelOption) {
    $env:KITTEN_MODEL_ID = [string]$ModelOption.modelId
    $env:KITTEN_MODEL_NAME = [string]$ModelOption.displayName
    Write-Host "KittenTTS model: $($ModelOption.displayName) ($($ModelOption.modelId))"
}

function Find-KittenVoice([string]$VoiceValue, [object[]]$Catalog = @()) {
    if (-not $VoiceValue) {
        return $null
    }
    if ($Catalog.Count -eq 0) {
        $Catalog = Get-KittenVoiceCatalog
    }

    $target = ConvertTo-PiperVoiceKey $VoiceValue
    foreach ($voice in $Catalog) {
        $aliases = @(
            [string]$voice.id,
            [string]$voice.displayName,
            [string]$voice.internalId
        )
        foreach ($alias in $aliases) {
            if ((ConvertTo-PiperVoiceKey $alias) -eq $target) {
                return $voice
            }
        }
    }

    return $null
}

function Show-KittenVoices([object[]]$Voices) {
    Write-Host ""
    Write-Host "KittenTTS female voices"
    for ($index = 0; $index -lt $Voices.Count; $index++) {
        $voice = $Voices[$index]
        $number = $index + 1
        Write-Host "$number. $($voice.displayName) - $($voice.internalId)"
    }
    Write-Host "Q. Quit"
}

function Read-KittenVoiceChoice([string]$PreferredVoice) {
    $catalog = Get-KittenVoiceCatalog
    if ($PreferredVoice) {
        $voice = Find-KittenVoice $PreferredVoice $catalog
        if (-not $voice) {
            throw "Unknown KittenTTS voice '$PreferredVoice'. Use one of the catalog ids such as bella, luna, rosie, or kiki."
        }
        return $voice
    }

    while ($true) {
        Show-KittenVoices $catalog
        try {
            $rawVoice = Read-Host "Choose a KittenTTS voice"
        } catch {
            return $null
        }
        if ($null -eq $rawVoice) {
            return $null
        }

        $voiceChoice = $rawVoice.Trim()
        $voiceKey = $voiceChoice.ToLowerInvariant()
        if ($voiceKey -in @("q", "quit")) {
            return $null
        }
        if ($voiceChoice -match "^\d+$") {
            $voiceIndex = [int]$voiceChoice
            if ($voiceIndex -ge 1 -and $voiceIndex -le $catalog.Count) {
                return $catalog[$voiceIndex - 1]
            }
        }

        $voice = Find-KittenVoice $voiceChoice $catalog
        if ($voice) {
            return $voice
        }
        Write-Host "Choose one of the listed voice numbers or Q."
    }
}

function Set-KittenVoiceEnvironment([object]$Voice) {
    $env:KITTEN_VOICE_ID = [string]$Voice.internalId
    $env:KITTEN_VOICE_NAME = [string]$Voice.displayName
    Write-Host "KittenTTS voice: $($Voice.displayName) ($($Voice.internalId))"
}

function Get-SupertonicVoiceCatalog {
    $catalogPath = Join-Path $Root "supertonic_service\voice_catalog.json"
    if (-not (Test-Path -LiteralPath $catalogPath)) {
        throw "Supertonic 3 voice catalog was not found at '$catalogPath'."
    }

    return @(Get-Content -Raw -LiteralPath $catalogPath | ConvertFrom-Json)
}

function Find-SupertonicVoice([string]$VoiceValue, [object[]]$Catalog = @()) {
    if (-not $VoiceValue) {
        return $null
    }
    if ($Catalog.Count -eq 0) {
        $Catalog = Get-SupertonicVoiceCatalog
    }

    $target = ConvertTo-PiperVoiceKey $VoiceValue
    foreach ($voice in $Catalog) {
        $aliases = @(
            [string]$voice.id,
            [string]$voice.displayName,
            [string]$voice.internalId
        )
        foreach ($alias in $aliases) {
            if ((ConvertTo-PiperVoiceKey $alias) -eq $target) {
                return $voice
            }
        }
    }

    return $null
}

function Show-SupertonicVoices([object[]]$Voices) {
    Write-Host ""
    Write-Host "Supertonic 3 female preset voices"
    for ($index = 0; $index -lt $Voices.Count; $index++) {
        $voice = $Voices[$index]
        $number = $index + 1
        Write-Host "$number. $($voice.displayName) - $($voice.internalId)"
    }
    Write-Host "Q. Quit"
}

function Read-SupertonicVoiceChoice([string]$PreferredVoice) {
    $catalog = Get-SupertonicVoiceCatalog
    if ($PreferredVoice) {
        $voice = Find-SupertonicVoice $PreferredVoice $catalog
        if (-not $voice) {
            throw "Unknown Supertonic 3 voice '$PreferredVoice'. Use f1, f2, f3, f4, or f5."
        }
        return $voice
    }

    while ($true) {
        Show-SupertonicVoices $catalog
        try {
            $rawVoice = Read-Host "Choose a Supertonic 3 voice"
        } catch {
            return $null
        }
        if ($null -eq $rawVoice) {
            return $null
        }

        $voiceChoice = $rawVoice.Trim()
        $voiceKey = $voiceChoice.ToLowerInvariant()
        if ($voiceKey -in @("q", "quit")) {
            return $null
        }
        if ($voiceChoice -match "^\d+$") {
            $voiceIndex = [int]$voiceChoice
            if ($voiceIndex -ge 1 -and $voiceIndex -le $catalog.Count) {
                return $catalog[$voiceIndex - 1]
            }
        }

        $voice = Find-SupertonicVoice $voiceChoice $catalog
        if ($voice) {
            return $voice
        }
        Write-Host "Choose one of the listed voice numbers or Q."
    }
}

function Set-SupertonicVoiceEnvironment([object]$Voice) {
    $env:SUPERTONIC_VOICE_ID = [string]$Voice.internalId
    $env:SUPERTONIC_VOICE_NAME = [string]$Voice.displayName
    Write-Host "Supertonic 3 voice: $($Voice.displayName) ($($Voice.internalId))"
}

function Get-ChatterboxVoiceCatalog {
    $catalogPath = Join-Path $Root "chatterbox_service\voice_catalog.json"
    if (-not (Test-Path -LiteralPath $catalogPath)) {
        throw "Chatterbox voice catalog was not found at '$catalogPath'."
    }

    return @(Get-Content -Raw -LiteralPath $catalogPath | ConvertFrom-Json)
}

function Find-ChatterboxVoice([string]$VoiceValue, [object[]]$Catalog = @()) {
    if (-not $VoiceValue) {
        return $null
    }
    if ($Catalog.Count -eq 0) {
        $Catalog = Get-ChatterboxVoiceCatalog
    }

    $target = ConvertTo-PiperVoiceKey $VoiceValue
    foreach ($voice in $Catalog) {
        $aliases = @(
            (Get-ObjectStringProperty $voice "id"),
            (Get-ObjectStringProperty $voice "displayName"),
            (Get-ObjectStringProperty $voice "mode")
        )
        foreach ($alias in $aliases) {
            if ($alias -and (ConvertTo-PiperVoiceKey $alias) -eq $target) {
                return $voice
            }
        }
    }

    return $null
}

function Show-ChatterboxVoices([object[]]$Voices) {
    Write-Host ""
    Write-Host "Chatterbox voice options"
    for ($index = 0; $index -lt $Voices.Count; $index++) {
        $voice = $Voices[$index]
        $number = $index + 1
        if ($voice.mode -eq "clone") {
            Write-Host "$number. Reference voice clone - local WAV"
        } else {
            Write-Host "$number. $($voice.displayName) - $($voice.description)"
        }
    }
    Write-Host "Q. Quit"
}

function Read-ChatterboxVoiceChoice([string]$PreferredVoice) {
    $catalog = Get-ChatterboxVoiceCatalog
    if ($PreferredVoice) {
        $voice = Find-ChatterboxVoice $PreferredVoice $catalog
        if (-not $voice) {
            throw "Unknown Chatterbox voice '$PreferredVoice'. Use reference-clone or builtin-default."
        }
        return $voice
    }

    while ($true) {
        Show-ChatterboxVoices $catalog
        try {
            $rawVoice = Read-Host "Choose a Chatterbox voice option"
        } catch {
            return $null
        }
        if ($null -eq $rawVoice) {
            return $null
        }

        $voiceChoice = $rawVoice.Trim()
        $voiceKey = $voiceChoice.ToLowerInvariant()
        if ($voiceKey -in @("q", "quit")) {
            return $null
        }
        if ($voiceChoice -match "^\d+$") {
            $voiceIndex = [int]$voiceChoice
            if ($voiceIndex -ge 1 -and $voiceIndex -le $catalog.Count) {
                return $catalog[$voiceIndex - 1]
            }
        }

        $voice = Find-ChatterboxVoice $voiceChoice $catalog
        if ($voice) {
            return $voice
        }
        Write-Host "Choose one of the listed voice numbers or Q."
    }
}

function Set-ChatterboxVoiceEnvironment([object]$Voice) {
    if ($Voice.mode -eq "builtin") {
        $env:CHATTERBOX_VOICE_MODE = "builtin"
        $env:CHATTERBOX_VOICE_NAME = [string]$Voice.displayName
        Write-Host "Chatterbox voice: $($Voice.displayName)"
        return
    }

    $env:CHATTERBOX_VOICE_MODE = "clone"
    $env:CHATTERBOX_VOICE_NAME = [string]$Voice.displayName
    Write-Host "Chatterbox voice: reference voice clone"
}

function Get-SoproVoiceCatalog {
    $catalogPath = Join-Path $Root "sopro_service\voice_catalog.json"
    if (-not (Test-Path -LiteralPath $catalogPath)) {
        throw "SoproTTS voice catalog was not found at '$catalogPath'."
    }

    return @(Get-Content -Raw -LiteralPath $catalogPath | ConvertFrom-Json)
}

function Find-SoproVoice([string]$VoiceValue, [object[]]$Catalog = @()) {
    if (-not $VoiceValue) {
        return $null
    }
    if ($Catalog.Count -eq 0) {
        $Catalog = Get-SoproVoiceCatalog
    }

    $target = ConvertTo-PiperVoiceKey $VoiceValue
    foreach ($voice in $Catalog) {
        $aliases = @(
            (Get-ObjectStringProperty $voice "id"),
            (Get-ObjectStringProperty $voice "displayName"),
            (Get-ObjectStringProperty $voice "mode")
        )
        foreach ($alias in $aliases) {
            if ($alias -and (ConvertTo-PiperVoiceKey $alias) -eq $target) {
                return $voice
            }
        }
    }

    return $null
}

function Show-SoproVoices([object[]]$Voices) {
    Write-Host ""
    Write-Host "SoproTTS voice options"
    for ($index = 0; $index -lt $Voices.Count; $index++) {
        $voice = $Voices[$index]
        $number = $index + 1
        Write-Host "$number. Reference voice clone - local WAV"
    }
    Write-Host "Q. Quit"
}

function Read-SoproVoiceChoice([string]$PreferredVoice) {
    $catalog = Get-SoproVoiceCatalog
    if ($PreferredVoice) {
        $voice = Find-SoproVoice $PreferredVoice $catalog
        if (-not $voice) {
            throw "Unknown SoproTTS voice '$PreferredVoice'. Use reference-clone."
        }
        return $voice
    }

    while ($true) {
        Show-SoproVoices $catalog
        try {
            $rawVoice = Read-Host "Choose a SoproTTS voice option"
        } catch {
            return $null
        }
        if ($null -eq $rawVoice) {
            return $null
        }

        $voiceChoice = $rawVoice.Trim()
        $voiceKey = $voiceChoice.ToLowerInvariant()
        if ($voiceKey -in @("q", "quit")) {
            return $null
        }
        if ($voiceChoice -match "^\d+$") {
            $voiceIndex = [int]$voiceChoice
            if ($voiceIndex -ge 1 -and $voiceIndex -le $catalog.Count) {
                return $catalog[$voiceIndex - 1]
            }
        }

        $voice = Find-SoproVoice $voiceChoice $catalog
        if ($voice) {
            return $voice
        }
        Write-Host "Choose 1 or Q."
    }
}

function Set-SoproVoiceEnvironment([object]$Voice) {
    $env:SOPRO_VOICE_MODE = "clone"
    $env:SOPRO_VOICE_NAME = [string]$Voice.displayName
    Write-Host "SoproTTS voice: reference voice clone"
}

function Get-PocketVoiceCatalog {
    $catalogPath = Join-Path $Root "pockettts_service\voice_catalog.json"
    if (-not (Test-Path -LiteralPath $catalogPath)) {
        throw "PocketTTS voice catalog was not found at '$catalogPath'."
    }

    return @(Get-Content -Raw -LiteralPath $catalogPath | ConvertFrom-Json)
}

function Find-PocketVoice([string]$VoiceValue, [object[]]$Catalog = @()) {
    if (-not $VoiceValue) {
        return $null
    }
    if ($Catalog.Count -eq 0) {
        $Catalog = Get-PocketVoiceCatalog
    }

    $target = ConvertTo-PiperVoiceKey $VoiceValue
    foreach ($voice in $Catalog) {
        $aliases = @(
            (Get-ObjectStringProperty $voice "id"),
            (Get-ObjectStringProperty $voice "displayName"),
            (Get-ObjectStringProperty $voice "voiceId"),
            (Get-ObjectStringProperty $voice "mode")
        )
        foreach ($alias in $aliases) {
            if ($alias -and (ConvertTo-PiperVoiceKey $alias) -eq $target) {
                return $voice
            }
        }
    }

    return $null
}

function Show-PocketVoices([object[]]$Voices) {
    Write-Host ""
    Write-Host "PocketTTS voice options"
    for ($index = 0; $index -lt $Voices.Count; $index++) {
        $voice = $Voices[$index]
        $number = $index + 1
        if ($voice.mode -eq "clone") {
            Write-Host "$number. Reference voice clone - local WAV"
        } else {
            Write-Host "$number. $($voice.displayName) - built-in English voice"
        }
    }
    Write-Host "Q. Quit"
}

function Read-PocketVoiceChoice([string]$PreferredVoice) {
    $catalog = Get-PocketVoiceCatalog
    if ($PreferredVoice) {
        $voice = Find-PocketVoice $PreferredVoice $catalog
        if (-not $voice) {
            throw "Unknown PocketTTS voice '$PreferredVoice'. Use reference-clone or a catalog id such as alba, anna, cosette, or vera."
        }
        return $voice
    }

    while ($true) {
        Show-PocketVoices $catalog
        try {
            $rawVoice = Read-Host "Choose a PocketTTS voice option"
        } catch {
            return $null
        }
        if ($null -eq $rawVoice) {
            return $null
        }

        $voiceChoice = $rawVoice.Trim()
        $voiceKey = $voiceChoice.ToLowerInvariant()
        if ($voiceKey -in @("q", "quit")) {
            return $null
        }
        if ($voiceChoice -match "^\d+$") {
            $voiceIndex = [int]$voiceChoice
            if ($voiceIndex -ge 1 -and $voiceIndex -le $catalog.Count) {
                return $catalog[$voiceIndex - 1]
            }
        }

        $voice = Find-PocketVoice $voiceChoice $catalog
        if ($voice) {
            return $voice
        }
        Write-Host "Choose one of the listed voice numbers or Q."
    }
}

function Set-PocketVoiceEnvironment([object]$Voice) {
    if ($Voice.mode -eq "preset") {
        $env:POCKETTTS_VOICE_MODE = "preset"
        $env:POCKETTTS_VOICE_ID = [string]$Voice.voiceId
        $env:POCKETTTS_VOICE_NAME = [string]$Voice.displayName
        Write-Host "PocketTTS voice: $($Voice.displayName) ($($Voice.voiceId))"
        return
    }

    $env:POCKETTTS_VOICE_MODE = "clone"
    $env:POCKETTTS_VOICE_ID = "reference-clone"
    $env:POCKETTTS_VOICE_NAME = [string]$Voice.displayName
    Write-Host "PocketTTS voice: reference voice clone"
}

function Get-HuggingFaceTokenForDocker {
    $tokenCandidates = @(
        $env:HF_TOKEN,
        $env:HUGGING_FACE_HUB_TOKEN
    )

    foreach ($token in $tokenCandidates) {
        if ($token -and $token.Trim()) {
            return $token.Trim()
        }
    }

    $pathCandidates = @()
    if ($env:HF_TOKEN_PATH) {
        $pathCandidates += $env:HF_TOKEN_PATH
    }
    if ($env:HF_HOME) {
        $pathCandidates += (Join-Path $env:HF_HOME "token")
    }
    if ($env:USERPROFILE) {
        $pathCandidates += (Join-Path $env:USERPROFILE ".cache\huggingface\token")
    }
    if ($env:APPDATA) {
        $pathCandidates += (Join-Path $env:APPDATA "huggingface\token")
    }

    foreach ($path in ($pathCandidates | Where-Object { $_ } | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            continue
        }
        try {
            $token = (Get-Content -Raw -LiteralPath $path).Trim()
            if ($token) {
                return $token
            }
        } catch {
        }
    }

    return ""
}

function Read-HiddenToken([string]$Prompt) {
    try {
        $secure = Read-Host $Prompt -AsSecureString
    } catch {
        return ""
    }
    if ($null -eq $secure -or $secure.Length -eq 0) {
        return ""
    }

    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try {
        $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
        if ($plain) {
            return $plain.Trim()
        }
    } finally {
        if ($bstr -ne [IntPtr]::Zero) {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
        }
    }
    return ""
}

function Test-PocketCloneWeightsCache {
    $cacheCandidates = @(
        (Join-Path $Root ".cache\huggingface\models--kyutai--pocket-tts"),
        (Join-Path $Root ".cache\huggingface\hub\models--kyutai--pocket-tts")
    )

    foreach ($path in $cacheCandidates) {
        if (Test-Path -LiteralPath $path -PathType Container) {
            return $true
        }
    }
    return $false
}

function Ensure-PocketCloneAuth {
    if (Test-PocketCloneWeightsCache) {
        Write-Host "PocketTTS clone auth: gated weights already exist in the local Hugging Face cache."
        return $true
    }

    $token = Get-HuggingFaceTokenForDocker
    if ($token) {
        $env:HF_TOKEN = $token
        $env:HUGGING_FACE_HUB_TOKEN = $token
        Write-Host "PocketTTS clone auth: Hugging Face token found locally and passed to Docker."
        return $true
    }

    Write-Host ""
    Write-Host "PocketTTS clone mode needs access to Kyutai's gated voice-cloning weights."
    Write-Host "Accept the terms at https://huggingface.co/kyutai/pocket-tts, then do one of these locally:"
    Write-Host "  huggingface-cli login"
    Write-Host "  uvx hf auth login"
    Write-Host "  `$env:HF_TOKEN = `"your-local-hugging-face-token`""
    Write-Host ""
    Write-Host "Or paste a Hugging Face token now. It will be hidden while typing and will not be written to repo files."
    $typedToken = Read-HiddenToken "HF token for this launcher session, or Enter to return to menu"
    if ($typedToken) {
        $env:HF_TOKEN = $typedToken
        $env:HUGGING_FACE_HUB_TOKEN = $typedToken
        Write-Host "PocketTTS clone auth: token accepted for this launcher session and passed to Docker."
        return $true
    }

    Write-Host "No token entered. Returning to the launcher menu."
    return $false
}

function Get-MossVoiceCatalog {
    $catalogPath = Join-Path $Root "moss_service\voice_catalog.json"
    if (-not (Test-Path -LiteralPath $catalogPath)) {
        throw "MOSS-TTS-Nano voice catalog was not found at '$catalogPath'."
    }

    return @(Get-Content -Raw -LiteralPath $catalogPath | ConvertFrom-Json)
}

function Find-MossVoice([string]$VoiceValue, [object[]]$Catalog = @()) {
    if (-not $VoiceValue) {
        return $null
    }
    if ($Catalog.Count -eq 0) {
        $Catalog = Get-MossVoiceCatalog
    }

    $target = ConvertTo-PiperVoiceKey $VoiceValue
    foreach ($voice in $Catalog) {
        $aliases = @(
            (Get-ObjectStringProperty $voice "id"),
            (Get-ObjectStringProperty $voice "displayName"),
            (Get-ObjectStringProperty $voice "voiceId"),
            (Get-ObjectStringProperty $voice "group"),
            (Get-ObjectStringProperty $voice "mode")
        )
        foreach ($alias in $aliases) {
            if ($alias -and (ConvertTo-PiperVoiceKey $alias) -eq $target) {
                return $voice
            }
        }
    }

    return $null
}

function Show-MossVoices([object[]]$Voices) {
    Write-Host ""
    Write-Host "MOSS-TTS-Nano voice options"
    for ($index = 0; $index -lt $Voices.Count; $index++) {
        $voice = $Voices[$index]
        $number = $index + 1
        if ($voice.mode -eq "clone") {
            Write-Host "$number. Reference voice clone - local WAV"
        } else {
            $sampleName = Get-ObjectStringProperty $voice "sampleName"
            $sampleSuffix = if ($sampleName) { " - $sampleName" } else { "" }
            Write-Host "$number. $($voice.displayName) - $($voice.group)$sampleSuffix"
        }
    }
    Write-Host "Q. Quit"
}

function Read-MossVoiceChoice([string]$PreferredVoice) {
    $catalog = Get-MossVoiceCatalog
    if ($PreferredVoice) {
        $voice = Find-MossVoice $PreferredVoice $catalog
        if (-not $voice) {
            throw "Unknown MOSS-TTS-Nano voice '$PreferredVoice'. Use reference-clone, ava, or bella."
        }
        return $voice
    }

    while ($true) {
        Show-MossVoices $catalog
        try {
            $rawVoice = Read-Host "Choose a MOSS-TTS-Nano voice option"
        } catch {
            return $null
        }
        if ($null -eq $rawVoice) {
            return $null
        }

        $voiceChoice = $rawVoice.Trim()
        $voiceKey = $voiceChoice.ToLowerInvariant()
        if ($voiceKey -in @("q", "quit")) {
            return $null
        }
        if ($voiceChoice -match "^\d+$") {
            $voiceIndex = [int]$voiceChoice
            if ($voiceIndex -ge 1 -and $voiceIndex -le $catalog.Count) {
                return $catalog[$voiceIndex - 1]
            }
        }

        $voice = Find-MossVoice $voiceChoice $catalog
        if ($voice) {
            return $voice
        }
        Write-Host "Choose one of the listed voice numbers or Q."
    }
}

function Set-MossVoiceEnvironment([object]$Voice) {
    if ($Voice.mode -eq "preset") {
        $env:MOSS_VOICE_MODE = "preset"
        $env:MOSS_VOICE_ID = [string]$Voice.voiceId
        $env:MOSS_VOICE_NAME = [string]$Voice.displayName
        Write-Host "MOSS-TTS-Nano voice: $($Voice.displayName) ($($Voice.group))"
        return
    }

    $env:MOSS_VOICE_MODE = "clone"
    $env:MOSS_VOICE_ID = "reference-clone"
    $env:MOSS_VOICE_NAME = [string]$Voice.displayName
    Write-Host "MOSS-TTS-Nano voice: reference voice clone"
}

function Get-ObjectStringProperty([object]$Object, [string]$Name) {
    if ($null -eq $Object) {
        return ""
    }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property -or $null -eq $property.Value) {
        return ""
    }
    return [string]$property.Value
}

function New-ReferenceCloneVoiceOption {
    param(
        [string]$ModelLabel,
        [string]$Description = "Use your local reference WAV and transcript."
    )

    return [pscustomobject]@{
        id = "reference-clone"
        mode = "clone"
        displayName = "Reference voice clone"
        modelLabel = $ModelLabel
        description = $Description
    }
}

function Read-ReferenceOnlyVoiceChoice([string]$ModelLabel, [string]$PreferredVoice) {
    $voice = New-ReferenceCloneVoiceOption $ModelLabel
    if ($PreferredVoice) {
        $key = ConvertTo-PiperVoiceKey $PreferredVoice
        if ($key -in @("reference", "reference-clone", "clone", "voice-clone")) {
            return $voice
        }
        throw "$ModelLabel does not publish American/British female preset voices in this launcher. Use reference-clone."
    }

    while ($true) {
        Write-Host ""
        Write-Host "$ModelLabel voice options"
        Write-Host "1. Reference voice clone - local WAV/transcript"
        Write-Host "Q. Quit"
        try {
            $rawChoice = Read-Host "Choose a voice option"
        } catch {
            return $null
        }
        if ($null -eq $rawChoice) {
            return $null
        }

        $choice = $rawChoice.Trim().ToLowerInvariant()
        switch ($choice) {
            "1" { return $voice }
            "reference" { return $voice }
            "reference-clone" { return $voice }
            "clone" { return $voice }
            "q" { return $null }
            "quit" { return $null }
            default { Write-Host "Choose 1 or Q." }
        }
    }
}

function Get-QwenVoiceCatalog {
    $catalogPath = Join-Path $Root "qwen_service\voice_catalog.json"
    if (-not (Test-Path -LiteralPath $catalogPath)) {
        throw "Qwen voice catalog was not found at '$catalogPath'."
    }

    return @(Get-Content -Raw -LiteralPath $catalogPath | ConvertFrom-Json)
}

function Find-QwenVoice([string]$VoiceValue, [object[]]$Catalog = @()) {
    if (-not $VoiceValue) {
        return $null
    }
    if ($Catalog.Count -eq 0) {
        $Catalog = Get-QwenVoiceCatalog
    }

    $target = ConvertTo-PiperVoiceKey $VoiceValue
    foreach ($voice in $Catalog) {
        $aliases = @(
            (Get-ObjectStringProperty $voice "id"),
            (Get-ObjectStringProperty $voice "displayName"),
            (Get-ObjectStringProperty $voice "speaker")
        )
        foreach ($alias in $aliases) {
            if ($alias -and (ConvertTo-PiperVoiceKey $alias) -eq $target) {
                return $voice
            }
        }
    }

    return $null
}

function Show-QwenVoices([object[]]$Voices) {
    Write-Host ""
    Write-Host "Qwen3-TTS voice options"
    for ($index = 0; $index -lt $Voices.Count; $index++) {
        $voice = $Voices[$index]
        $number = $index + 1
        if ($voice.mode -eq "clone") {
            Write-Host "$number. Reference voice clone - local WAV/transcript"
        } else {
            Write-Host "$number. $($voice.displayName) - $($voice.description) Native: $($voice.nativeLanguage)"
        }
    }
    Write-Host "Q. Quit"
}

function Read-QwenVoiceChoice([string]$PreferredVoice) {
    $catalog = Get-QwenVoiceCatalog
    if ($PreferredVoice) {
        $voice = Find-QwenVoice $PreferredVoice $catalog
        if (-not $voice) {
            throw "Unknown Qwen voice '$PreferredVoice'. Use reference-clone, serena, vivian, sohee, or ono-anna."
        }
        return $voice
    }

    while ($true) {
        Show-QwenVoices $catalog
        try {
            $rawVoice = Read-Host "Choose a Qwen voice option"
        } catch {
            return $null
        }
        if ($null -eq $rawVoice) {
            return $null
        }

        $voiceChoice = $rawVoice.Trim()
        $voiceKey = $voiceChoice.ToLowerInvariant()
        if ($voiceKey -in @("q", "quit")) {
            return $null
        }
        if ($voiceChoice -match "^\d+$") {
            $voiceIndex = [int]$voiceChoice
            if ($voiceIndex -ge 1 -and $voiceIndex -le $catalog.Count) {
                return $catalog[$voiceIndex - 1]
            }
        }

        $voice = Find-QwenVoice $voiceChoice $catalog
        if ($voice) {
            return $voice
        }
        Write-Host "Choose one of the listed voice numbers or Q."
    }
}

function Set-QwenVoiceEnvironment([object]$Voice) {
    if ($Voice.mode -eq "custom_voice") {
        $env:QWEN_VOICE_MODE = "custom_voice"
        $env:QWEN_MODEL_ID = [string]$Voice.modelId
        $env:QWEN_CUSTOM_SPEAKER = [string]$Voice.speaker
        $env:QWEN_CUSTOM_VOICE_NAME = [string]$Voice.displayName
        $env:QWEN_CUSTOM_INSTRUCT = [string]$Voice.instruct
        Write-Host "Qwen voice: $($Voice.displayName) ($($Voice.description))"
        return
    }

    $env:QWEN_VOICE_MODE = "clone"
    $env:QWEN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    $env:QWEN_CUSTOM_SPEAKER = ""
    $env:QWEN_CUSTOM_VOICE_NAME = ""
    $env:QWEN_CUSTOM_INSTRUCT = ""
    Write-Host "Qwen voice: reference voice clone"
}

function Get-MioVoiceCatalog {
    $catalogPath = Join-Path $Root "miotts_service\voice_catalog.json"
    if (-not (Test-Path -LiteralPath $catalogPath)) {
        throw "MioTTS voice catalog was not found at '$catalogPath'."
    }

    return @(Get-Content -Raw -LiteralPath $catalogPath | ConvertFrom-Json)
}

function Find-MioVoice([string]$VoiceValue, [object[]]$Catalog = @()) {
    if (-not $VoiceValue) {
        return $null
    }
    if ($Catalog.Count -eq 0) {
        $Catalog = Get-MioVoiceCatalog
    }

    $target = ConvertTo-PiperVoiceKey $VoiceValue
    foreach ($voice in $Catalog) {
        $aliases = @(
            (Get-ObjectStringProperty $voice "id"),
            (Get-ObjectStringProperty $voice "displayName"),
            (Get-ObjectStringProperty $voice "presetId")
        )
        foreach ($alias in $aliases) {
            if ($alias -and (ConvertTo-PiperVoiceKey $alias) -eq $target) {
                return $voice
            }
        }
    }

    return $null
}

function Show-MioVoices([object[]]$Voices) {
    Write-Host ""
    Write-Host "MioTTS voice options"
    for ($index = 0; $index -lt $Voices.Count; $index++) {
        $voice = $Voices[$index]
        $number = $index + 1
        if ($voice.mode -eq "clone") {
            Write-Host "$number. Reference voice clone - local WAV"
        } else {
            Write-Host "$number. $($voice.displayName) - $($voice.description)"
        }
    }
    Write-Host "Q. Quit"
}

function Read-MioVoiceChoice([string]$PreferredVoice) {
    $catalog = Get-MioVoiceCatalog
    if ($PreferredVoice) {
        $voice = Find-MioVoice $PreferredVoice $catalog
        if (-not $voice) {
            throw "Unknown MioTTS voice '$PreferredVoice'. Use reference-clone or en-female."
        }
        return $voice
    }

    while ($true) {
        Show-MioVoices $catalog
        try {
            $rawVoice = Read-Host "Choose a MioTTS voice option"
        } catch {
            return $null
        }
        if ($null -eq $rawVoice) {
            return $null
        }

        $voiceChoice = $rawVoice.Trim()
        $voiceKey = $voiceChoice.ToLowerInvariant()
        if ($voiceKey -in @("q", "quit")) {
            return $null
        }
        if ($voiceChoice -match "^\d+$") {
            $voiceIndex = [int]$voiceChoice
            if ($voiceIndex -ge 1 -and $voiceIndex -le $catalog.Count) {
                return $catalog[$voiceIndex - 1]
            }
        }

        $voice = Find-MioVoice $voiceChoice $catalog
        if ($voice) {
            return $voice
        }
        Write-Host "Choose one of the listed voice numbers or Q."
    }
}

function Set-MioVoiceEnvironment([object]$Voice) {
    if ($Voice.mode -eq "preset") {
        $env:MIOTTS_VOICE_MODE = "preset"
        $env:MIOTTS_PRESET_ID = [string]$Voice.presetId
        $env:MIOTTS_PRESET_NAME = [string]$Voice.displayName
        $env:MIOTTS_PRESET_URL = [string]$Voice.sourceUrl
        Write-Host "MioTTS voice: $($Voice.displayName) ($($Voice.presetId))"
        return
    }

    $env:MIOTTS_VOICE_MODE = "clone"
    $env:MIOTTS_PRESET_ID = ""
    $env:MIOTTS_PRESET_NAME = ""
    $env:MIOTTS_PRESET_URL = ""
    Write-Host "MioTTS voice: reference voice clone"
}

function Test-DockerImageExists([string]$ImageName) {
    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = "docker"
    $escapedImageName = $ImageName.Replace('"', '\"')
    $startInfo.Arguments = "image inspect `"$escapedImageName`""
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true

    $process = [System.Diagnostics.Process]::Start($startInfo)
    $process.WaitForExit()
    return ($process.ExitCode -eq 0)
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
    Write-Host "2. Qwen3-TTS 0.6B - clone + CustomVoice presets"
    Write-Host "3. MioTTS 0.1B - clone + official preset"
    Write-Host "4. LuxTTS 100M - reference voice clone"
    Write-Host "5. Piper TTS - preset voice picker (no cloning)"
    Write-Host "6. Kokoro-82M - preset voice picker (no cloning)"
    Write-Host "7. KittenTTS 15M/40M/80M - female preset voice picker (no cloning)"
    Write-Host "8. Chatterbox 500M - reference clone + built-in fallback"
    Write-Host "9. SoproTTS 135M - reference voice clone"
    Write-Host "10. PocketTTS 100M - reference clone + English built-in voices"
    Write-Host "11. MOSS-TTS-Nano 100M - reference clone + English female built-in voices"
    Write-Host "12. F5-TTS - reference voice clone"
    Write-Host "13. Supertonic 3 99M - female preset voice picker (no cloning)"
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
            "5" { return "piper" }
            "piper" { return "piper" }
            "6" { return "kokoro" }
            "kokoro" { return "kokoro" }
            "7" { return "kitten" }
            "kitten" { return "kitten" }
            "kittentts" { return "kitten" }
            "8" { return "chatterbox" }
            "chatterbox" { return "chatterbox" }
            "9" { return "sopro" }
            "sopro" { return "sopro" }
            "soprotts" { return "sopro" }
            "10" { return "pockettts" }
            "pocket" { return "pockettts" }
            "pockettts" { return "pockettts" }
            "11" { return "moss" }
            "moss" { return "moss" }
            "mosstts" { return "moss" }
            "moss-tts" { return "moss" }
            "mossnano" { return "moss" }
            "moss-tts-nano" { return "moss" }
            "12" { return "f5tts" }
            "f5" { return "f5tts" }
            "f5tts" { return "f5tts" }
            "f5-tts" { return "f5tts" }
            "13" { return "supertonic" }
            "supertonic" { return "supertonic" }
            "supertonic3" { return "supertonic" }
            "supertonic-3" { return "supertonic" }
            "q" { return "" }
            "quit" { return "" }
            default { Write-Host "Choose 1 through 13, or Q." }
        }
    }
}

function Invoke-SelectedModel([string]$SelectedModel) {
    switch ($SelectedModel) {
        "xtts" {
            $selectedVoice = Read-ReferenceOnlyVoiceChoice "XTTS-v2" $XttsVoice
            if (-not $selectedVoice) {
                $script:SelectedModelExitCode = 0
                return
            }
            Write-Host ""
            Write-ReferenceInputs
            Write-Host "Starting XTTS-v2. Wait for xttsv2> before typing."
            Invoke-TtsContainer "xtts"
            return
        }
        "qwen" {
            $selectedVoice = Read-QwenVoiceChoice $QwenVoice
            if (-not $selectedVoice) {
                $script:SelectedModelExitCode = 0
                return
            }
            Set-QwenVoiceEnvironment $selectedVoice
            if ($selectedVoice.mode -eq "clone" -and (-not $env:QWEN_REF_AUDIO -or -not $env:QWEN_REF_TEXT_FILE)) {
                Write-Error "Qwen voice cloning requires both a reference WAV and a reference transcript."
                $script:SelectedModelExitCode = 1
                return
            }
            Write-Host ""
            if ($selectedVoice.mode -eq "clone") {
                Write-ReferenceInputs
            }
            Write-Host "Starting Qwen3-TTS 0.6B. Wait for qwen0.6btts> before typing."
            Invoke-TtsContainer "qwen"
            return
        }
        "miotts" {
            $selectedVoice = Read-MioVoiceChoice $MioVoice
            if (-not $selectedVoice) {
                $script:SelectedModelExitCode = 0
                return
            }
            Set-MioVoiceEnvironment $selectedVoice
            if ($selectedVoice.mode -eq "clone" -and -not $env:MIOTTS_REF_AUDIO) {
                Write-Error "MioTTS voice cloning requires a reference WAV."
                $script:SelectedModelExitCode = 1
                return
            }
            Write-Host ""
            if ($selectedVoice.mode -eq "clone") {
                Write-ReferenceInputs
            }
            Write-Host "Starting MioTTS 0.1B. Wait for miotts0.1b> before typing."
            Invoke-TtsContainer "miotts"
            return
        }
        "luxtts" {
            $selectedVoice = Read-ReferenceOnlyVoiceChoice "LuxTTS 100M" $LuxVoice
            if (-not $selectedVoice) {
                $script:SelectedModelExitCode = 0
                return
            }
            if (-not $env:LUXTTS_REF_AUDIO -or -not $env:LUXTTS_REF_TEXT_FILE) {
                Write-Error "LuxTTS voice cloning requires both a reference WAV and a reference transcript."
                $script:SelectedModelExitCode = 1
                return
            }
            Write-Host ""
            Write-ReferenceInputs
            Write-Host "Starting LuxTTS 100M. Wait for luxtts100m> before typing."
            Invoke-TtsContainer "luxtts"
            return
        }
        "piper" {
            $selectedVoice = Read-PiperVoiceChoice $PiperVoice
            if (-not $selectedVoice) {
                $script:SelectedModelExitCode = 0
                return
            }
            Set-PiperVoiceEnvironment $selectedVoice
            Write-Host ""
            Write-Host "Starting Piper TTS with $env:PIPER_VOICE_NAME. Wait for piper> before typing."
            Invoke-TtsContainer "piper"
            return
        }
        "kokoro" {
            $selectedVoice = Read-KokoroVoiceChoice $KokoroVoice
            if (-not $selectedVoice) {
                $script:SelectedModelExitCode = 0
                return
            }
            Set-KokoroVoiceEnvironment $selectedVoice
            Write-Host ""
            Write-Host "Starting Kokoro-82M with $env:KOKORO_VOICE_ID. Wait for kokoro82m> before typing."
            Invoke-TtsContainer "kokoro"
            return
        }
        "kitten" {
            $selectedKittenModel = Read-KittenModelChoice $KittenSize
            if (-not $selectedKittenModel) {
                $script:SelectedModelExitCode = 0
                return
            }
            $selectedVoice = Read-KittenVoiceChoice $KittenVoice
            if (-not $selectedVoice) {
                $script:SelectedModelExitCode = 0
                return
            }
            Set-KittenModelEnvironment $selectedKittenModel
            Set-KittenVoiceEnvironment $selectedVoice
            Write-Host ""
            Write-Host "Starting KittenTTS $env:KITTEN_MODEL_NAME with $env:KITTEN_VOICE_NAME. Wait for kittentts> before typing."
            Invoke-TtsContainer "kitten"
            return
        }
        "supertonic" {
            $selectedVoice = Read-SupertonicVoiceChoice $SupertonicVoice
            if (-not $selectedVoice) {
                $script:SelectedModelExitCode = 0
                return
            }
            Set-SupertonicVoiceEnvironment $selectedVoice
            Write-Host ""
            Write-Host "Supertonic 3 preset voice does not use the local reference WAV or transcript."
            Write-Host "Starting Supertonic 3 with $env:SUPERTONIC_VOICE_NAME. Wait for supertonic3> before typing."
            Invoke-TtsContainer "supertonic"
            return
        }
        "chatterbox" {
            $selectedVoice = Read-ChatterboxVoiceChoice $ChatterboxVoice
            if (-not $selectedVoice) {
                $script:SelectedModelExitCode = 0
                return
            }
            Set-ChatterboxVoiceEnvironment $selectedVoice
            if ($selectedVoice.mode -eq "clone" -and -not $env:CHATTERBOX_REF_AUDIO) {
                Write-Error "Chatterbox voice cloning requires a reference WAV."
                $script:SelectedModelExitCode = 1
                return
            }
            Write-Host ""
            if ($selectedVoice.mode -eq "clone") {
                if ($env:CHATTERBOX_REF_AUDIO) {
                    Write-Host "Reference voice: $env:CHATTERBOX_REF_AUDIO"
                }
                Write-Host "Reference text is not used by Chatterbox."
            }
            Write-Host "Starting Chatterbox 500M with $env:CHATTERBOX_VOICE_NAME. Wait for chatterbox> before typing."
            Invoke-TtsContainer "chatterbox"
            return
        }
        "sopro" {
            $selectedVoice = Read-SoproVoiceChoice $SoproVoice
            if (-not $selectedVoice) {
                $script:SelectedModelExitCode = 0
                return
            }
            Set-SoproVoiceEnvironment $selectedVoice
            if (-not $env:SOPRO_REF_AUDIO) {
                Write-Error "SoproTTS voice cloning requires a reference WAV."
                $script:SelectedModelExitCode = 1
                return
            }
            Write-Host ""
            Write-Host "Reference voice: $env:SOPRO_REF_AUDIO"
            Write-Host "Reference text is not used by SoproTTS."
            Write-Host "Starting SoproTTS 135M with $env:SOPRO_VOICE_NAME. Wait for soprotts> before typing."
            Invoke-TtsContainer "sopro"
            return
        }
        "pockettts" {
            $selectedVoice = Read-PocketVoiceChoice $PocketVoice
            if (-not $selectedVoice) {
                $script:SelectedModelExitCode = 0
                return
            }
            Set-PocketVoiceEnvironment $selectedVoice
            if ($selectedVoice.mode -eq "clone" -and -not $env:POCKETTTS_REF_AUDIO) {
                Write-Error "PocketTTS voice cloning requires a reference WAV."
                $script:SelectedModelExitCode = 1
                return
            }
            if ($selectedVoice.mode -eq "clone" -and -not (Ensure-PocketCloneAuth)) {
                $script:SelectedModelExitCode = 1
                return
            }
            Write-Host ""
            if ($selectedVoice.mode -eq "clone") {
                Write-Host "Reference voice: $env:POCKETTTS_REF_AUDIO"
                Write-Host "Reference text is not used by PocketTTS."
            } else {
                Write-Host "PocketTTS preset voice does not use the local reference WAV or transcript."
            }
            Write-Host "Starting PocketTTS 100M with $env:POCKETTTS_VOICE_NAME. Wait for pockettts> before typing."
            Invoke-TtsContainer "pockettts"
            return
        }
        "moss" {
            $selectedVoice = Read-MossVoiceChoice $MossVoice
            if (-not $selectedVoice) {
                $script:SelectedModelExitCode = 0
                return
            }
            Set-MossVoiceEnvironment $selectedVoice
            if ($selectedVoice.mode -eq "clone" -and -not $env:MOSS_REF_AUDIO) {
                Write-Error "MOSS-TTS-Nano voice cloning requires a reference WAV."
                $script:SelectedModelExitCode = 1
                return
            }
            Write-Host ""
            if ($selectedVoice.mode -eq "clone") {
                Write-Host "Reference voice: $env:MOSS_REF_AUDIO"
                Write-Host "Reference text is not used by MOSS-TTS-Nano."
            } else {
                Write-Host "MOSS-TTS-Nano preset voice does not use the local reference WAV or transcript."
            }
            Write-Host "Starting MOSS-TTS-Nano 100M with $env:MOSS_VOICE_NAME. Wait for mossttsnano> before typing."
            Invoke-TtsContainer "moss"
            return
        }
        "f5tts" {
            $selectedVoice = Read-ReferenceOnlyVoiceChoice "F5-TTS" $F5Voice
            if (-not $selectedVoice) {
                $script:SelectedModelExitCode = 0
                return
            }
            if (-not $env:F5TTS_REF_AUDIO -or -not $env:F5TTS_REF_TEXT_FILE) {
                Write-Error "F5-TTS voice cloning requires both a reference WAV and a reference transcript."
                $script:SelectedModelExitCode = 1
                return
            }
            Write-Host ""
            Write-ReferenceInputs
            Write-Host "Starting F5-TTS with reference voice clone. Wait for f5tts> before typing."
            Invoke-TtsContainer "f5tts"
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
