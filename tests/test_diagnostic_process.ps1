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
