# Dot-source this file from PowerShell before configuring a native Windows build.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:UV_CACHE_DIR = Join-Path $projectRoot '.cache\uv'
$env:HF_HOME = Join-Path $projectRoot '.cache\huggingface'
# Hugging Face supports copied cache files without Windows symlink privileges.
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:TIKTOKEN_CACHE_DIR = Join-Path $projectRoot '.cache\tiktoken'
$env:TEMP = Join-Path $projectRoot '.tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw 'Install Visual Studio 2022 C++ Build Tools, Windows SDK, CMake and AddressSanitizer.'
}
$installation = & $vswhere -latest -version '[17.0,18.0)' -products '*' `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $installation) {
    throw 'No complete Visual Studio 2022 C++ Build Tools installation was found.'
}
$developerCommand = Join-Path $installation 'Common7\Tools\VsDevCmd.bat'
$environmentLines = & $env:ComSpec /d /s /c "`"$developerCommand`" -no_logo -arch=x64 -host_arch=x64 >nul && set"
if ($LASTEXITCODE -ne 0) { throw 'VsDevCmd failed.' }
foreach ($line in $environmentLines) {
    if ($line -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}
$env:Path = "$(Join-Path $projectRoot '.venv\Scripts');$env:Path"
$env:CC = 'cl'
$env:CXX = 'cl'
Write-Host "Native x64 toolchain: $installation"
