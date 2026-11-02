# Run from the extracted ReleaseLens repository root in PowerShell.
# Requires Git and authenticated GitHub CLI (gh auth login).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "Install Git for Windows first: https://git-scm.com/download/win" }
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) { throw "Install GitHub CLI first: https://cli.github.com/" }
gh auth status
if ($LASTEXITCODE -ne 0) { throw "Run 'gh auth login' then run this script again." }
$repo = "yusufthecoder474/releaselens"
$exists = $false
gh repo view $repo --json name *> $null
if ($LASTEXITCODE -eq 0) { $exists = $true }
if (-not $exists) {
  gh repo create $repo --public --description "ReleaseLens: explainable release-readiness from JUnit XML"
  if ($LASTEXITCODE -ne 0) { throw "Repository creation failed." }
}
git remote get-url origin *> $null
if ($LASTEXITCODE -eq 0) { git remote set-url origin "https://github.com/$repo.git" }
else { git remote add origin "https://github.com/$repo.git" }
git branch -M main
git push -u origin main
if ($LASTEXITCODE -ne 0) { throw "Git push failed; check permissions and authentication." }
Write-Host "Published: https://github.com/$repo"
