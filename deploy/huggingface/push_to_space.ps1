<#
Push Ledger to a Hugging Face Docker Space.

  powershell -ExecutionPolicy Bypass -File deploy\huggingface\push_to_space.ps1 -HfUser <your-hf-username>

Before running: create the Space on huggingface.co (SDK: Docker, Blank) and add the secrets GROQ_API_KEY and
DEMO_ACCESS_CODE (optionally GEMINI_API_KEY) under Settings -> Variables and secrets.
Git asks for your Hugging Face username and an access token with write permission (not your password).

Only files committed to git are sent (never .env). The Space gets deploy/huggingface/README.md as its README
(it holds the Space settings), and binary files the app doesn't need (PDFs, video) are left out, because
Hugging Face rejects binary files pushed with plain git.
#>
param(
    [Parameter(Mandatory = $true)][string]$HfUser,
    [string]$Space = "ledger-pdf-agent",
    [switch]$DryRun  # build the Space folder and list its files, but don't push
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$sha = git -C $repo rev-parse --short HEAD
$work = Join-Path $env:TEMP "ledger-space-$sha"
if (Test-Path $work) { Remove-Item -Recurse -Force $work }
New-Item -ItemType Directory -Path $work | Out-Null

git -C $repo archive --format=zip -o "$work\src.zip" HEAD  # committed files only
Expand-Archive "$work\src.zip" -DestinationPath "$work\space"
Remove-Item "$work\src.zip"
Set-Location "$work\space"
Copy-Item "deploy\huggingface\README.md" "README.md" -Force
Remove-Item -Recurse -Force "docs", "samples", "deploy", "MEMO.pdf"

Write-Host "Space files (from commit $sha):"
Get-ChildItem -Recurse -File | ForEach-Object { "  " + $_.FullName.Substring($PWD.Path.Length + 1) }
if ($DryRun) { Write-Host "Dry run: nothing pushed. Folder: $work\space"; return }

git init -q -b main
git add -A
git commit -q -m "Deploy Ledger from GitHub commit $sha"
git push --force "https://huggingface.co/spaces/$HfUser/$Space" main
Write-Host "Pushed. Build log and app: https://huggingface.co/spaces/$HfUser/$Space"
