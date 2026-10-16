param(
    [ValidateSet("prepare", "experiment", "analysis", "finalize", "all")]
    [string]$Phase = "all"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python environment not found: $PythonExe"
}

Push-Location $RepoRoot
try {
    function Invoke-Python([string[]]$Arguments) {
        & $PythonExe @Arguments
        if ($LASTEXITCODE -ne 0) { throw "Command failed: python $($Arguments -join ' ')" }
    }

    function Invoke-Prepare {
        Invoke-Python @("scripts/run_eda.py")
        Invoke-Python @("scripts/prepare_data.py")
        Invoke-Python @("scripts/build_indices.py")
        Invoke-Python @("scripts/prepare_paraphrases.py")
    }

    function Assert-SemanticCaches {
        $Required = @(
            "models/embeddings/chunks.npy",
            "models/embeddings/questions.npy",
            "models/embeddings/answers.npy",
            "models/embeddings/paraphrases.npy",
            "models/reranker/stage6_scores.json",
            "models/reranker/stage6_manifest.json"
        )
        $Missing = @($Required | Where-Object { -not (Test-Path -LiteralPath (Join-Path $RepoRoot $_)) })
        if ($Missing.Count -gt 0) {
            throw "GPU caches are missing: $($Missing -join ', '). Follow the private Kaggle instructions in README.md, then rerun -Phase experiment."
        }
    }

    function Invoke-Experiment {
        Assert-SemanticCaches
        Invoke-Python @("scripts/verify_embeddings.py")
        Invoke-Python @("scripts/run_stage5.py")
        Invoke-Python @("scripts/run_stage6_selection.py")
        Invoke-Python @("scripts/select_stage6_reranker.py")
        Invoke-Python @("scripts/run_stage7.py", "--resume")
        Invoke-Python @("scripts/verify_stage7.py")
    }

    function Invoke-Analysis {
        Invoke-Python @("scripts/run_stage8.py")
        Invoke-Python @("scripts/verify_stage8.py")
        Invoke-Python @("scripts/run_stage9.py")
        Invoke-Python @("scripts/verify_stage9.py")
        Invoke-Python @("scripts/run_stage10.py")
        Invoke-Python @("scripts/verify_stage10.py")
        Invoke-Python @("scripts/run_stage12.py")
        Invoke-Python @("scripts/verify_stage12.py")
    }

    function Invoke-Finalize {
        Invoke-Python @("scripts/build_stage13_artifacts.py")
        Invoke-Python @("scripts/verify_stage13.py")
        Invoke-Python @("-m", "pytest", "-q")
    }

    switch ($Phase) {
        "prepare" { Invoke-Prepare }
        "experiment" { Invoke-Experiment }
        "analysis" { Invoke-Analysis }
        "finalize" { Invoke-Finalize }
        "all" { Invoke-Prepare; Invoke-Experiment; Invoke-Analysis; Invoke-Finalize }
    }
}
finally {
    Pop-Location
}

