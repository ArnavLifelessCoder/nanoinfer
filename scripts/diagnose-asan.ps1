param(
    [Parameter(Mandatory)][string]$BuildDirectory,
    [Parameter(Mandatory)][string]$Label,
    [int]$IdleSeconds = 12
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'dev-shell.ps1')
. (Join-Path $PSScriptRoot 'diagnostic-process.ps1')
$build = [IO.Path]::GetFullPath((Join-Path $projectRoot $BuildDirectory))
$output = Join-Path $projectRoot 'build/diagnostics'
$procdump = Join-Path $output 'tools/procdump/procdump64.exe'
if (-not (Test-Path -LiteralPath $procdump)) { throw 'Install ProcDump under build/diagnostics/tools/procdump first.' }
$env:NANOINFER_PROCDUMP = $procdump
if ($Label -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Label must contain only letters, digits, underscores or hyphens.' }
$stdout = Join-Path $output "$Label-ctest.txt"
$stderr = Join-Path $output "$Label-ctest-stderr.txt"
if (Test-Path -LiteralPath $stdout) { throw 'Choose a new label; existing diagnostic evidence will not be overwritten.' }
$command = "--test-dir `"$build`" -j 1 --timeout 180 --verbose --no-tests=error"
$test = Start-Process -FilePath (Get-Command ctest).Source -ArgumentList $command `
    -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$seen = @{}
$captures = [System.Collections.Generic.List[object]]::new()
while (-not $test.HasExited) {
    foreach ($native in @(Get-Process -Name 'gpt2*' -ErrorAction SilentlyContinue)) {
        if (-not (Get-DiagnosticProcessPath -NativeProcess $native -BuildDirectory $build)) {
            continue
        }
        $key = $native.Id
        $started = $native.StartTime
        $cpu = $native.CPU
        if ($null -eq $started -or $null -eq $cpu) { continue }
        $state = Update-DiagnosticObservation $seen $key $started.Ticks $cpu ([DateTime]::UtcNow)
        if (-not $state.Captured -and
            ([DateTime]::UtcNow - $state.Changed).TotalSeconds -ge $IdleSeconds) {
            $state.Captured = $true
            $name = "$Label-$($native.ProcessName)-$key"
            $dump = Join-Path $output "dumps/$name.dmp"
            $log = Join-Path $output "$name-capture.txt"
            $record = [ordered]@{
                pid = $key
                start_time = $started.ToString('o')
                idle_since_utc = $state.Changed.ToString('o')
                process = $native.ProcessName
                cpu_seconds = $native.CPU
                thread_count = $native.Threads.Count
                has_exited = $native.HasExited
                dump = "build/diagnostics/dumps/$name.dmp"
            }
            & $procdump -accepteula -ma $key $dump *> $log
            $record.capture_exit_code = $LASTEXITCODE
            $record.dump_exists = Test-Path -LiteralPath $dump
            $captures.Add($record)
            Write-Output "Idle-process capture attempt $name; ProcDump exit=$LASTEXITCODE; dump=$($record.dump_exists)"
        }
    }
    Start-Sleep -Seconds 1
    $test.Refresh()
}
$test.WaitForExit()
$report = [ordered]@{
    label = $Label
    build_directory = $BuildDirectory
    git_head = (git -C $projectRoot rev-parse HEAD)
    asan_options = $env:ASAN_OPTIONS
    ctest_exit_code = $test.ExitCode
    idle_seconds = $IdleSeconds
    captures = @($captures.ToArray())
}
$report | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 (Join-Path $output "$Label-report.json")
Write-Output "CTest exit=$($test.ExitCode); captures=$($captures.Count)"
exit $test.ExitCode
