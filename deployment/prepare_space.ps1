param([Parameter(Mandatory = $true)][string]$SpaceDir)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

if (-not (Test-Path $SpaceDir)) { throw "Space folder not found: $SpaceDir (clone your Space first)" }
if (-not (Test-Path "$SpaceDir\.git")) { throw "$SpaceDir is not a git clone of your Space" }
if (-not (Test-Path "models\crop_disease_cnn_ft_final.keras")) { throw "models\crop_disease_cnn_ft_final.keras is missing" }

$files = git ls-files
foreach ($f in $files) {
    $dest = Join-Path $SpaceDir $f
    New-Item -ItemType Directory -Force (Split-Path $dest -Parent) | Out-Null
    Copy-Item $f $dest -Force
}
Copy-Item "models\crop_disease_cnn_ft_final.keras" (Join-Path $SpaceDir "models\crop_disease_cnn_ft_final.keras") -Force
Copy-Item "deployment\Dockerfile" (Join-Path $SpaceDir "Dockerfile") -Force

$header = @"
---
title: Crop Doctor
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

"@
$readme = Get-Content "README.md" -Raw -Encoding UTF8
[System.IO.File]::WriteAllText((Join-Path $SpaceDir "README.md"), $header + $readme, (New-Object System.Text.UTF8Encoding $false))

$ignore = Get-Content ".gitignore" | Where-Object { $_ -notmatch '^\*\.keras$' }
[System.IO.File]::WriteAllLines((Join-Path $SpaceDir ".gitignore"), $ignore, (New-Object System.Text.UTF8Encoding $false))

$attrs = @(Get-Content ".gitattributes") + "Dockerfile text eol=lf" + "*.keras filter=lfs diff=lfs merge=lfs -text" + "*.png filter=lfs diff=lfs merge=lfs -text"
[System.IO.File]::WriteAllLines((Join-Path $SpaceDir ".gitattributes"), ($attrs | Select-Object -Unique), (New-Object System.Text.UTF8Encoding $false))

if (Test-Path (Join-Path $SpaceDir ".env")) { Remove-Item (Join-Path $SpaceDir ".env") -Force }

$count = (Get-ChildItem $SpaceDir -Recurse -File | Where-Object { $_.FullName -notmatch '\\\.git\\' }).Count
Write-Host "Space folder ready: $count files in $SpaceDir" -ForegroundColor Green
Write-Host "Next, in that folder: git lfs install; git add .; git commit -m `"Deploy Crop Doctor`"; git push"
