# Replace machine-specific path prefixes without removing test output.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$replacements = [ordered]@{
    $projectRoot = '<project>'
    $env:USERPROFILE = '<user>'
    ${env:ProgramFiles(x86)} = '<program-files-x86>'
    $env:ProgramFiles = '<program-files>'
    '/home/runner/work/nanoinfer/nanoinfer' = '<project>'
    '/opt/hostedtoolcache' = '<hosted-toolcache>'
    'C:\hostedtoolcache' = '<hosted-toolcache>'
    '/home/runner' = '<runner>'
    'D:\a' = '<runner>'
}
Get-ChildItem -LiteralPath (Join-Path $projectRoot 'results') -Filter '*.txt' |
    ForEach-Object {
        $text = [IO.File]::ReadAllText($_.FullName)
        foreach ($prefix in $replacements.Keys) {
            if ($prefix) {
                $text = $text.Replace($prefix, $replacements[$prefix])
                $text = $text.Replace($prefix.Replace('\', '/'), $replacements[$prefix])
            }
        }
        $text = $text -replace '(?<![\w])[A-Za-z]:[\\/]', '<drive>/'
        $text = $text.Replace("`r`n", "`n") -replace '(?m)[ \t]+$', ''
        $text = $text -replace '\n{2,}\z', "`n"
        [IO.File]::WriteAllText($_.FullName, $text)
    }
