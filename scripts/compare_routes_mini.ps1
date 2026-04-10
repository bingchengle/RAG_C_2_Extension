# Compare economy vs quality on the bundled mini benchmark (same retrieval, different LLM route).
# Prereqs: .env with OPENAI_API_KEY (and BGE_API_KEY only if you use bge_api below).
# Run from repo root:  .\scripts\compare_routes_mini.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Mini = Join-Path $RepoRoot "data\datasets\benchmark_round1_mini"
$MainPy = Join-Path $RepoRoot "main.py"

Set-Location $Mini

# --- Pick ONE retrieval line (must match your .faiss files on disk) ---

# A) Repo default indexes are OpenAI-style (*.faiss, no _bge). Use with enhanced + OpenAI embed + LLM rerank:
$CommonArgs = @(
    "--config", "base",
    "--profile", "enhanced",
    "--embedding-provider", "openai",
    "--reranker-type", "llm"
)

# B) If you rebuilt with:  python $MainPy rebuild-vector-dbs --embedding-provider bge_api
#    then use BGE rerank (and optional BGE embed) instead:
# $CommonArgs = @("--config", "base", "--profile", "enhanced", "--embedding-provider", "bge_api", "--reranker-type", "bge")

Write-Host "=== route=economy ===" -ForegroundColor Cyan
python $MainPy process-questions @CommonArgs --route economy

Write-Host "=== route=quality ===" -ForegroundColor Cyan
python $MainPy process-questions @CommonArgs --route quality

Write-Host "Done. New answers: answers_base*.json ; usage: answers_base*_route_usage.json" -ForegroundColor Green
Set-Location $RepoRoot
