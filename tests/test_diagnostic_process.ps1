$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '../scripts/diagnostic-process.ps1')
$build = Join-Path (Split-Path -Parent $PSScriptRoot) 'build/diagnostic-test'
$native = [pscustomobject]@{ Reads = 0; FirstPath = (Join-Path $build 'gpt2inspect.exe') }
$native | Add-Member -MemberType ScriptProperty -Name Path -Value {
    $this.Reads++
    if ($this.Reads -eq 1) { return $this.FirstPath }
    return $null
}
$result = Get-DiagnosticProcessPath -NativeProcess $native -BuildDirectory $build
if ($result -ne $native.FirstPath -or $native.Reads -ne 1) {
    throw 'The monitor must snapshot Path once; the process can exit between property reads.'
}
if ($null -ne (Get-DiagnosticProcessPath -NativeProcess ([pscustomobject]@{Path=$null}) -BuildDirectory $build)) {
    throw 'An already-exited process must be skipped.'
}
if ($null -ne (Get-DiagnosticProcessPath -NativeProcess ([pscustomobject]@{Path="$build-other/gpt2inspect.exe"}) -BuildDirectory $build)) {
    throw 'A sibling build must be skipped.'
}
Write-Output 'PASS process-path exit race, already-exited process and sibling-build exclusion'
$seen = @{}
$now = [datetime]::UtcNow
$first = Update-DiagnosticObservation $seen 17 100 0.015625 $now
$same = Update-DiagnosticObservation $seen 17 100 0.015625 $now.AddSeconds(20)
if ($same.Changed -ne $first.Changed) { throw 'Idle time must accumulate for the same process.' }
$replacement = Update-DiagnosticObservation $seen 17 200 0.015625 $now.AddSeconds(30)
if ($replacement.Changed -ne $now.AddSeconds(30) -or $replacement.StartTicks -ne 200) {
    throw 'Reused PID and equal CPU time must not inherit the previous process idle timer.'
}
Write-Output 'PASS stable identity and reused-PID idle tracking'
