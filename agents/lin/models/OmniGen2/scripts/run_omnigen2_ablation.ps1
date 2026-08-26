[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,

    [Parameter(Mandatory = $true)]
    [string]$HrPath,

    [double]$Scale = 4.0,
    [int]$Seed = 0,
    [int]$Steps = 50,
    [int]$Gpu = 1,
    [string]$SshTarget = "zeus5"
)

$ErrorActionPreference = "Stop"
if ($Scale -le 0) { throw "Scale must be positive." }
if ($Steps -le 0) { throw "Steps must be positive." }

$resolvedInput = (Resolve-Path -LiteralPath $InputPath).Path
$resolvedHr = (Resolve-Path -LiteralPath $HrPath).Path
$inputItem = Get-Item -LiteralPath $resolvedInput
$safeStem = [regex]::Replace($inputItem.BaseName, "[^A-Za-z0-9_-]", "_")
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..\..\..\..")).Path
$semanticPrompt = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\prompts\omnigen2_sr_text_faithful.txt")).Path
$conservativePrompt = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\prompts\omnigen2_sr_text_conservative.txt")).Path
$localOutputRoot = Join-Path $projectRoot "workspace\SR\COLLECT\selected_LR\OmniGen2"
$localImageDir = Join-Path $localOutputRoot $safeStem

if (Test-Path -LiteralPath $localImageDir) {
    throw "Refusing to overwrite existing image result folder: $localImageDir"
}
New-Item -ItemType Directory -Path $localImageDir | Out-Null
Copy-Item -LiteralPath $resolvedInput -Destination (Join-Path $localImageDir "input.png")
Copy-Item -LiteralPath $resolvedHr -Destination (Join-Path $localImageDir "HR.png")
Copy-Item -LiteralPath $semanticPrompt -Destination (Join-Path $localImageDir "prompt_semantic.txt")
Copy-Item -LiteralPath $conservativePrompt -Destination (Join-Path $localImageDir "prompt_conservative.txt")

$deployRoot = "/share/linmingheng-local/code/OmniGen2"
$repoDir = "$deployRoot/repo"
$venvDir = "$deployRoot/.venv"
$modelDir = "$deployRoot/models/OmniGen2"
$runnerPath = "$deployRoot/bootstrap/omnigen2_scaled_inference.py"

$configs = @(
    [ordered]@{ label = "A_direct_semantic"; mode = "direct_scale"; prompt = $semanticPrompt; prompt_name = "semantic" },
    [ordered]@{ label = "B_direct_conservative"; mode = "direct_scale"; prompt = $conservativePrompt; prompt_name = "conservative" },
    [ordered]@{ label = "C_preup_semantic"; mode = "preup_align"; prompt = $semanticPrompt; prompt_name = "semantic" },
    [ordered]@{ label = "D_preup_conservative"; mode = "preup_align"; prompt = $conservativePrompt; prompt_name = "conservative" }
)

$results = @()
foreach ($config in $configs) {
    $remoteRunName = "${safeStem}_ablation_$($config.label)_seed${Seed}_${timestamp}"
    $remoteRun = "$deployRoot/runs/$remoteRunName"
    $remoteInput = "$remoteRun/input$($inputItem.Extension.ToLowerInvariant())"
    $remotePrompt = "$remoteRun/prompt.txt"
    $remoteOutput = "$remoteRun/output.png"
    $localOutput = Join-Path $localImageDir "output_$($config.label).png"

    Write-Host "[$safeStem] Starting $($config.label)..."
    & ssh $SshTarget "mkdir '$remoteRun'"
    if ($LASTEXITCODE -ne 0) { throw "Could not create $remoteRun" }

    & scp $resolvedInput "${SshTarget}:$remoteInput"
    if ($LASTEXITCODE -ne 0) { throw "Input upload failed for $($config.label)" }
    & scp $config.prompt "${SshTarget}:$remotePrompt"
    if ($LASTEXITCODE -ne 0) { throw "Prompt upload failed for $($config.label)" }

    $remoteCommand = @"
env CPATH='$deployRoot/sysroot/usr/include:$deployRoot/sysroot/usr/include/python3.10' PYTHONPATH='$repoDir' CUDA_VISIBLE_DEVICES='$Gpu' '$venvDir/bin/python' '$runnerPath' --repo-dir '$repoDir' --model-path '$modelDir' --input-image '$remoteInput' --prompt-file '$remotePrompt' --output-image '$remoteOutput' --mode '$($config.mode)' --scale '$Scale' --seed '$Seed' --steps '$Steps' > '$remoteRun/stdout.log' 2> '$remoteRun/stderr.log'
"@

    & ssh -o ServerAliveInterval=30 $SshTarget $remoteCommand
    $runStatus = $LASTEXITCODE
    & ssh $SshTarget "printf '%s\n' '$runStatus' > '$remoteRun/exit_code.txt'"
    if ($runStatus -ne 0) {
        & ssh $SshTarget "tail -n 120 '$remoteRun/stderr.log'"
        throw "$($config.label) failed with exit code $runStatus"
    }

    & scp "${SshTarget}:$remoteOutput" $localOutput
    if ($LASTEXITCODE -ne 0) { throw "Output download failed for $($config.label)" }

    $metadataText = (& ssh $SshTarget "cat '$remoteRun/metadata.json'") -join "`n"
    if ($LASTEXITCODE -ne 0) { throw "Metadata read failed for $($config.label)" }
    $metadata = $metadataText | ConvertFrom-Json
    $results += [ordered]@{
        label = $config.label
        mode = $config.mode
        prompt = $config.prompt_name
        output_file = [IO.Path]::GetFileName($localOutput)
        output_sha256 = $metadata.output_sha256
        output_size = $metadata.output_size
        remote_run = $remoteRun
        status = $metadata.status
    }
    Write-Host "[$safeStem] Completed $($config.label): $($metadata.output_size[0])x$($metadata.output_size[1])"
}

Add-Type -AssemblyName System.Drawing
$inputImage = [System.Drawing.Image]::FromFile($resolvedInput)
$hrImage = [System.Drawing.Image]::FromFile($resolvedHr)
try {
    $manifest = [ordered]@{
        image = $safeStem
        created_at = (Get-Date).ToString("o")
        input_source = $resolvedInput
        input_file = "input.png"
        input_size = @($inputImage.Width, $inputImage.Height)
        input_sha256 = (Get-FileHash -LiteralPath $resolvedInput -Algorithm SHA256).Hash.ToLowerInvariant()
        hr_source = $resolvedHr
        hr_file = "HR.png"
        hr_size = @($hrImage.Width, $hrImage.Height)
        hr_sha256 = (Get-FileHash -LiteralPath $resolvedHr -Algorithm SHA256).Hash.ToLowerInvariant()
        model = "OmniGen2/OmniGen2"
        scale = $Scale
        seed = $Seed
        steps = $Steps
        scheduler = "euler"
        dtype = "bf16"
        text_guidance_scale = 5.0
        image_guidance_scale = 2.0
        cfg_range = @(0.0, 1.0)
        results = $results
    }
}
finally {
    $inputImage.Dispose()
    $hrImage.Dispose()
}

$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $localImageDir "manifest.json") -Encoding UTF8
Write-Host "Ablation complete: $localImageDir"
