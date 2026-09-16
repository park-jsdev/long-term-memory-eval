# Generate the manifest, then execute Cloud Run QA with N tasks.
# Usage: .\scripts\execute_gcp.ps1 configs/experiments/poc.yaml --parallelism 1

$ErrorActionPreference = "Stop"

if ($args.Count -lt 1) {
    throw "Usage: .\scripts\execute_gcp.ps1 <config.yaml> [--parallelism N]"
}

$Config = $args[0]
$Parallelism = "1"
for ($i = 1; $i -lt $args.Count; $i++) {
    if ($args[$i] -eq "--parallelism") {
        $Parallelism = $args[$i + 1]
        $i++
    }
    else {
        throw "unknown arg $($args[$i])"
    }
}

$REGION = if ($env:REGION) { $env:REGION } else { "us-central1" }

python -m src.memorybench write-manifest $Config
if ($LASTEXITCODE -ne 0) { throw "write-manifest failed" }

$N = python -c @"
from src.memorybench.expand_run_matrix import expand_run_matrix
from src.memorybench.load_experiment_yaml import load_experiment_yaml
print(len(expand_run_matrix(load_experiment_yaml(r'$Config'))))
"@
if ($LASTEXITCODE -ne 0) { throw "matrix expand failed" }
$N = $N.Trim()

Write-Host "Executing $N QA tasks (job parallelism=$Parallelism)"
& gcloud.cmd run jobs update memorybench-qa --region=$REGION --parallelism=$Parallelism
if ($LASTEXITCODE -ne 0) { throw "gcloud run jobs update parallelism failed" }
& gcloud.cmd run jobs execute memorybench-qa `
    --region=$REGION `
    --tasks=$N `
    --wait
if ($LASTEXITCODE -ne 0) { throw "gcloud run jobs execute failed" }
