function Get-DiagnosticProcessPath {
    param($NativeProcess, [string]$BuildDirectory)
    $path = $NativeProcess.Path
    if (-not $path -or
        -not $path.StartsWith($BuildDirectory + [IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase)) { return $null }
    return $path
}
