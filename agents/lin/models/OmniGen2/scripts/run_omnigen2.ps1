[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,

    [Parameter(Mandatory = $true)]
    [string]$PromptFile,

    [double]$Scale = 4.0,
    [int]$Seed = 0,
    [int]$Steps = 50,
    [int]$Gpu = 1,
    [string]$SshTarget = "zeus5",
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"

if ($Scale -le 0) {
    throw "Scale must be positive."
}
if ($Steps -le 0) {
    throw "Steps must be positive."
}

$resolvedInput = (Resolve-Path -LiteralPath $InputPath).Path
$resolvedPrompt = (Resolve-Path -LiteralPath $PromptFile).Path
$inputItem = Get-Item -LiteralPath $resolvedInput
$safeStem = [regex]::Replace($inputItem.BaseName, "[^A-Za-z0-9_-]", "_")
$scaleToken = ([string]$Scale).Replace(".", "p")
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runName = "${safeStem}_x${scaleToken}_seed${Seed}_${timestamp}"

$deployRoot = "/share/linmingheng-local/code/OmniGen2"
$repoDir = "$deployRoot/repo"
$venvDir = "$deployRoot/.venv"
$modelDir = "$deployRoot/models/OmniGen2"
$runnerPath = "$deployRoot/bootstrap/omnigen2_scaled_inference.py"
$remoteRun = "$deployRoot/runs/$runName"
$remoteInput = "$remoteRun/input$($inputItem.Extension.ToLowerInvariant())"
$remotePrompt = "$remoteRun/prompt.txt"
$remoteOutput = "$remoteRun/output.png"

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..\..\..\..")).Path
$localOutputRoot = Join-Path $projectRoot "workspace\SR\COLLECT\selected_LR\OmniGen2"
New-Item -ItemType Directory -Force -Path $localOutputRoot | Out-Null

Write-Host "Run: $runName"
Write-Host "Uploading input and prompt..."

& ssh $SshTarget "mkdir '$remoteRun'"
if ($LASTEXITCODE -ne 0) {
    throw "Could not create unique remote run directory: $remoteRun"
}

& scp $resolvedInput "${SshTarget}:$remoteInput"
if ($LASTEXITCODE -ne 0) {
    throw "Input upload failed."
}

& scp $resolvedPrompt "${SshTarget}:$remotePrompt"
if ($LASTEXITCODE -ne 0) {
    throw "Prompt upload failed."
}

$remoteCommand = @"
env CPATH='$deployRoot/sysroot/usr/include:$deployRoot/sysroot/usr/include/python3.10' PYTHONPATH='$repoDir' CUDA_VISIBLE_DEVICES='$Gpu' '$venvDir/bin/python' '$runnerPath' --repo-dir '$repoDir' --model-path '$modelDir' --input-image '$remoteInput' --prompt-file '$remotePrompt' --output-image '$remoteOutput' --scale '$Scale' --seed '$Seed' --steps '$Steps' > '$remoteRun/stdout.log' 2> '$remoteRun/stderr.log'
"@

Write-Host "Running OmniGen2 on GPU $Gpu..."
& ssh -o ServerAliveInterval=30 $SshTarget $remoteCommand
$runStatus = $LASTEXITCODE

& ssh $SshTarget "printf '%s\n' '$runStatus' > '$remoteRun/exit_code.txt'"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Could not write remote exit_code.txt."
}

Write-Host "Downloading run artifacts..."
& scp -r "${SshTarget}:$remoteRun" "$localOutputRoot\"
if ($LASTEXITCODE -ne 0) {
    throw "Run artifact download failed. Remote artifacts remain at $remoteRun"
}

$localRun = Join-Path $localOutputRoot $runName
$localOutput = Join-Path $localRun "output.png"
$localMetadata = Join-Path $localRun "metadata.json"

if ($runStatus -ne 0) {
    $stderrPath = Join-Path $localRun "stderr.log"
    Write-Error "OmniGen2 failed with exit code $runStatus. See $stderrPath"
}
if (-not (Test-Path -LiteralPath $localOutput)) {
    throw "Inference reported success but output is missing: $localOutput"
}

Add-Type -AssemblyName System.Drawing
$image = [System.Drawing.Image]::FromFile($localOutput)
try {
    Write-Host "Output: $localOutput"
    Write-Host "Output size: $($image.Width)x$($image.Height)"
    Write-Host "Metadata: $localMetadata"
}
finally {
    $image.Dispose()
}

if (-not $NoOpen) {
    Invoke-Item -LiteralPath $localOutput
}
