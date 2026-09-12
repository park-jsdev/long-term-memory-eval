# Build and push the experiment image; create/update Cloud Run Jobs.
# Requires: Docker Desktop, gcloud.cmd, PROJECT_ID.
# Usage (PowerShell, repo root):
#   $env:PROJECT_ID = "agent-platform-508416"
#   $env:REGION = "us-central1"
#   .\scripts\deploy_gcp.ps1

$ErrorActionPreference = "Stop"

if (-not $env:PROJECT_ID) {
    throw "set `$env:PROJECT_ID first"
}

$REGION = if ($env:REGION) { $env:REGION } else { "us-central1" }
$AR_REPO = if ($env:AR_REPO) { $env:AR_REPO } else { "memorybench" }
$SA = if ($env:SA) { $env:SA } else { "memorybench-runner" }
$BUCKET = if ($env:BUCKET) { $env:BUCKET } else { "$($env:PROJECT_ID)-memorybench" }
$TAG = if ($env:TAG) { $env:TAG } else { (git rev-parse --short HEAD).Trim() }
$IMAGE = "${REGION}-docker.pkg.dev/$($env:PROJECT_ID)/${AR_REPO}/memorybench:${TAG}"
$SA_EMAIL = "$SA@$($env:PROJECT_ID).iam.gserviceaccount.com"
$SECRETS = "OPENAI_API_KEY=openai-api-key:latest,ANTHROPIC_API_KEY=anthropic-api-key:latest,DEEPSEEK_API_KEY=deepseek-api-key:latest"
$ENV_VARS = "PYTHONUNBUFFERED=1,MEMORYBENCH_BUCKET=$BUCKET"

function Invoke-Gcloud {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$GcloudArgs)
    & gcloud.cmd @GcloudArgs
    if ($LASTEXITCODE -ne 0) {
        throw "gcloud.cmd $($GcloudArgs -join ' ') failed with exit $LASTEXITCODE"
    }
}

Invoke-Gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet
docker build -t $IMAGE .
if ($LASTEXITCODE -ne 0) { throw "docker build failed" }
docker push $IMAGE
if ($LASTEXITCODE -ne 0) { throw "docker push failed" }

$common = @(
    "--image=$IMAGE",
    "--region=$REGION",
    "--service-account=$SA_EMAIL",
    "--tasks=1",
    "--parallelism=1",
    "--task-timeout=12h",
    "--max-retries=2",
    "--cpu=1",
    "--memory=2Gi",
    "--set-secrets=$SECRETS",
    "--set-env-vars=$ENV_VARS"
)

function Set-MemorybenchJob {
    param([string]$Name, [string]$ArgsCsv)
    # First deploy: job does not exist. gcloud writes ERROR to stderr; PowerShell
    # Stop would treat that as fatal. Swallow it and create instead.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    & gcloud.cmd run jobs describe $Name --region=$REGION *> $null
    $exists = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = $prev
    if ($exists) {
        Invoke-Gcloud run jobs update $Name @common "--args=$ArgsCsv"
    }
    else {
        Invoke-Gcloud run jobs create $Name @common "--args=$ArgsCsv"
    }
}

Set-MemorybenchJob -Name "memorybench-qa" -ArgsCsv "execute-qa,configs/experiments/poc_gcs.yaml"
Set-MemorybenchJob -Name "memorybench-autorater" -ArgsCsv "execute-autorater,configs/experiments/poc_gcs.yaml"

Write-Host "Image $IMAGE"
Write-Host "Jobs: memorybench-qa, memorybench-autorater"
