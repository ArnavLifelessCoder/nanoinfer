function Get-DiagnosticProcessPath {
    param($NativeProcess, [string]$BuildDirectory)
    $path = $NativeProcess.Path
    if (-not $path -or
        -not $path.StartsWith($BuildDirectory + [IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase)) { return $null }
    return $path
}

function Update-DiagnosticObservation {
    param([hashtable]$Seen, [int]$ProcessId, [long]$StartTicks, [double]$CpuSeconds, [datetime]$Now)
    if (-not $Seen.ContainsKey($ProcessId) -or $Seen[$ProcessId].StartTicks -ne $StartTicks) {
        $Seen[$ProcessId] = @{ CPU = $CpuSeconds; Changed = $Now; Captured = $false; StartTicks = $StartTicks }
    }
    $state = $Seen[$ProcessId]
    if ($CpuSeconds -ne $state.CPU) {
        $state.CPU = $CpuSeconds
        $state.Changed = $Now
    }
    return $state
}
