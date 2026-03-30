# Cloudexe launch script for PLoRA Step 5
# Usage (PowerShell):
#   ./run_step5_cloudexe.ps1 -RankJson "/content/plora_step4_rank_budgets.json" -OutputDir "/content/outputs/plora_step5"

param(
    [string]$RankJson = "plora_step4_rank_budgets.json",
    [string]$OutputDir = "outputs/plora_step5",
    [string]$ModelId = "",
    [string]$BudgetKey = "fair_budget",
    [string]$TargetLanguages = "",
    [int]$MaxSteps = 2000,
    [int]$MaxTrainSamples = 10000,
    [int]$MaxEvalSamples = 1000,
    [int]$TrainBatchSize = 2,
    [int]$GradAccumSteps = 8,
    [double]$LearningRate = 2e-4,
    [switch]$UseBf16
)

python -m pip install -q "transformers>=4.46" "peft>=0.13" "accelerate>=1.1" "datasets>=3.0" "sentencepiece" "safetensors"

$cmd = @(
    "python", "plora_step5_language_routed_training.py",
    "--rank-json", $RankJson,
    "--output-dir", $OutputDir,
    "--budget-key", $BudgetKey,
    "--max-steps", "$MaxSteps",
    "--max-train-samples", "$MaxTrainSamples",
    "--max-eval-samples", "$MaxEvalSamples",
    "--train-batch-size", "$TrainBatchSize",
    "--grad-accum-steps", "$GradAccumSteps",
    "--learning-rate", "$LearningRate"
)

if ($ModelId -ne "") {
    $cmd += @("--model-id", $ModelId)
}
if ($TargetLanguages -ne "") {
    $cmd += @("--target-languages", $TargetLanguages)
}
if ($UseBf16) {
    $cmd += "--use-bf16"
}

Write-Host "Running:" ($cmd -join " ")
& $cmd[0] $cmd[1..($cmd.Length-1)]
