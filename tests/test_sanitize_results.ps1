$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$originalResults = Join-Path $projectRoot 'results'
$hashes = @{}
Get-ChildItem -LiteralPath $originalResults -Filter '*.txt' | ForEach-Object {
    $hashes[$_.FullName] = (Get-FileHash -LiteralPath $_.FullName).Hash
}
$buildRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot 'build'))
$temporaryRoot = Join-Path $buildRoot ("sanitizer-test-" + [guid]::NewGuid().ToString('N'))
$temporaryScripts = Join-Path $temporaryRoot 'scripts'
$temporaryResults = Join-Path $temporaryRoot 'results'
New-Item -ItemType Directory -Path $temporaryScripts, $temporaryResults -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $projectRoot 'scripts/sanitize-results.ps1') -Destination $temporaryScripts
Get-ChildItem -LiteralPath $originalResults -Filter '*.txt' | Copy-Item -Destination $temporaryResults
$sanitizer = Join-Path $temporaryScripts 'sanitize-results.ps1'
$fixture = Join-Path $temporaryResults 'sanitize-regression-fixture.txt'
try {
    [IO.File]::WriteAllText($fixture, "https://example.com/run`nhttp://example.com/run`nC:\temp\file`nD:/temp/file`n")
    & $sanitizer
    $actual = [IO.File]::ReadAllText($fixture)
    $expected = "https://example.com/run`nhttp://example.com/run`n<drive>/temp\file`n<drive>/temp/file`n"
    if ($actual -cne $expected) { throw "URL/path sanitization regression: $actual" }
    & $sanitizer
    if ([IO.File]::ReadAllText($fixture) -cne $expected) { throw 'Sanitization is not idempotent.' }
    foreach ($path in $hashes.Keys) {
        if ((Get-FileHash -LiteralPath $path).Hash -ne $hashes[$path]) {
            throw "Sanitizer test modified original result: $path"
        }
    }
    Write-Output 'PASS sanitization preserves URLs, redacts paths, is idempotent and leaves original results unchanged'
}
finally {
    $resolved = [IO.Path]::GetFullPath($temporaryRoot)
    if (-not $resolved.StartsWith($buildRoot + [IO.Path]::DirectorySeparatorChar,
        [StringComparison]::OrdinalIgnoreCase)) { throw 'Temporary directory escaped build root' }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
