# Build and push the experiment image; create/update Cloud Run Jobs.
# Requires: Docker Desktop, gcloud.cmd, PROJECT_ID.
# Usage (PowerShell, repo root):
#   $env:PROJECT_ID = "your-gcp-project-id"
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
$JOB_MEMORY = if ($env:JOB_MEMORY) { $env:JOB_MEMORY } else { "4Gi" }
$JOB_CPU = if ($env:JOB_CPU) { $env:JOB_CPU } else { "1" }
# Aggregate does not need the QA worker's high-memory allocation. Keep its
# default valid for 1 vCPU; allow an explicit paired override when necessary.
$AGGREGATE_MEMORY = if ($env:AGGREGATE_MEMORY) { $env:AGGREGATE_MEMORY } else { "4Gi" }
$AGGREGATE_CPU = if ($env:AGGREGATE_CPU) { $env:AGGREGATE_CPU } else { "1" }
# collect-full is 1 task. Do not multiply this RAM/CPU by QA parallelism —
# us-central1 default quota is 20 vCPU / 40Gi, and 8×32Gi is 256Gi.
# 32Gi requires 8 vCPU. Override with $env:COLLECT_FULL_MEMORY / CPU.
$COLLECT_FULL_MEMORY = if ($env:COLLECT_FULL_MEMORY) { $env:COLLECT_FULL_MEMORY } else { "32Gi" }
$COLLECT_FULL_CPU = if ($env:COLLECT_FULL_CPU) { $env:COLLECT_FULL_CPU } else { "8" }
# 8 saturates current 4/8-cell QA/autorater waves. Aggregate and collect-full stay 1.
$JOB_PARALLELISM = if ($env:JOB_PARALLELISM) { $env:JOB_PARALLELISM } else { "8" }
$EXP_YAML = if ($env:EXPERIMENT_YAML) { $env:EXPERIMENT_YAML } else { "configs/experiments/poc_gcs.yaml" }
$IMAGE = "${REGION}-docker.pkg.dev/$($env:PROJECT_ID)/${AR_REPO}/memorybench:${TAG}"
$SA_EMAIL = "$SA@$($env:PROJECT_ID).iam.gserviceaccount.com"
$SECRETS = "OPENAI_API_KEY=openai-api-key:latest,CODEX_API_KEY=codex-api-key:latest,ANTHROPIC_API_KEY=anthropic-api-key:latest,DEEPSEEK_API_KEY=deepseek-api-key:latest"
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
    "--task-timeout=12h",
    "--max-retries=2",
    "--set-secrets=$SECRETS",
    "--set-env-vars=$ENV_VARS"
)

function Set-MemorybenchJob {
    param(
        [string]$Name,
        [string]$ArgsCsv,
        [string]$Memory = $JOB_MEMORY,
        [string]$Cpu = $JOB_CPU,
        [string]$Parallelism = $JOB_PARALLELISM
    )
    # First deploy: job does not exist. gcloud writes ERROR to stderr; PowerShell
    # Stop would treat that as fatal. Swallow it and create instead.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    & gcloud.cmd run jobs describe $Name --region=$REGION *> $null
    $exists = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = $prev
    $jobArgs = $common + @(
        "--cpu=$Cpu",
        "--memory=$Memory",
        "--parallelism=$Parallelism",
        "--args=$ArgsCsv"
    )
    if ($exists) {
        Invoke-Gcloud run jobs update $Name @jobArgs
    }
    else {
        Invoke-Gcloud run jobs create $Name @jobArgs
    }
}

Set-MemorybenchJob -Name "memorybench-qa" -ArgsCsv "execute-qa,$EXP_YAML"
Set-MemorybenchJob -Name "memorybench-autorater" -ArgsCsv "execute-autorater,$EXP_YAML"
Set-MemorybenchJob -Name "memorybench-aggregate" -ArgsCsv "aggregate,$EXP_YAML" -Memory $AGGREGATE_MEMORY -Cpu $AGGREGATE_CPU -Parallelism "1"
Set-MemorybenchJob -Name "memorybench-collect-full" -ArgsCsv "collect-full,$EXP_YAML" -Memory $COLLECT_FULL_MEMORY -Cpu $COLLECT_FULL_CPU -Parallelism "1"

Write-Host "Image $IMAGE"
Write-Host "Experiment YAML $EXP_YAML"
Write-Host "Jobs: memorybench-qa, memorybench-autorater, memorybench-aggregate, memorybench-collect-full"
