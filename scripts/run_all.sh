#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:-all}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXE="$REPO_ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON_EXE" ]]; then
  echo "Python environment not found: $PYTHON_EXE" >&2
  exit 1
fi

cd "$REPO_ROOT"

prepare() {
  "$PYTHON_EXE" scripts/run_eda.py
  "$PYTHON_EXE" scripts/prepare_data.py
  "$PYTHON_EXE" scripts/build_indices.py
  "$PYTHON_EXE" scripts/prepare_paraphrases.py
}

assert_semantic_caches() {
  local required=(models/embeddings/chunks.npy models/embeddings/questions.npy models/embeddings/answers.npy models/embeddings/paraphrases.npy models/reranker/stage6_scores.json models/reranker/stage6_manifest.json)
  local missing=()
  for path in "${required[@]}"; do [[ -f "$path" ]] || missing+=("$path"); done
  if ((${#missing[@]})); then
    printf 'GPU caches are missing: %s\n' "${missing[*]}" >&2
    echo "Follow the private Kaggle instructions in README.md, then rerun experiment." >&2
    exit 1
  fi
}

experiment() {
  assert_semantic_caches
  "$PYTHON_EXE" scripts/verify_embeddings.py
  "$PYTHON_EXE" scripts/run_stage5.py
  "$PYTHON_EXE" scripts/run_stage6_selection.py
  "$PYTHON_EXE" scripts/select_stage6_reranker.py
  "$PYTHON_EXE" scripts/run_stage7.py --resume
  "$PYTHON_EXE" scripts/verify_stage7.py
}

analysis() {
  "$PYTHON_EXE" scripts/run_stage8.py
  "$PYTHON_EXE" scripts/verify_stage8.py
  "$PYTHON_EXE" scripts/run_stage9.py
  "$PYTHON_EXE" scripts/verify_stage9.py
  "$PYTHON_EXE" scripts/run_stage10.py
  "$PYTHON_EXE" scripts/verify_stage10.py
  "$PYTHON_EXE" scripts/run_stage12.py
  "$PYTHON_EXE" scripts/verify_stage12.py
}

finalize() {
  "$PYTHON_EXE" scripts/build_stage13_artifacts.py
  "$PYTHON_EXE" scripts/verify_stage13.py
  "$PYTHON_EXE" -m pytest -q
}

case "$PHASE" in
  prepare) prepare ;;
  experiment) experiment ;;
  analysis) analysis ;;
  finalize) finalize ;;
  all) prepare; experiment; analysis; finalize ;;
  *) echo "Usage: $0 {prepare|experiment|analysis|finalize|all}" >&2; exit 2 ;;
esac

