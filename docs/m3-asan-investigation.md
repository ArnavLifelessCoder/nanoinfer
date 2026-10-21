# M3 local ASan investigation

M3 remains blocked on local ASan reliability. The owner stopped further
runtime-option trials after the compatibility control failed. No workaround
has been adopted. A passing retry is not acceptance evidence.
The five consecutive full-suite runs use a fresh build and stop at the first
failure.

## Captured hangs and M2 baseline

ProcDump 12.02, downloaded from Microsoft's Sysinternals distribution and
signature-checked, captured full dumps before test timeouts. Dumps stay in the
ignored `build/diagnostics/dumps` directory; the complete thread stacks and
module lists are committed:

| Process | Origin | Threads | Stack |
|---|---|---:|---|
| gpt2inspect, PID 51896 | Earlier local ASan run | 1 | [stack](../results/m3-stack-stale-inspector.txt) |
| gpt2ref, PID 18484 | Earlier local ASan run | 1 | [stack](../results/m3-stack-stale-reference.txt) |
| gpt2inspect, PID 46896 | M2 loader integration | 1 | [stack](../results/m3-stack-m2-inspector.txt) |
| gpt2inspect, PID 44420 | M2 schema validation | 1 | [stack](../results/m3-stack-m2-schema.txt) |

The SDK debugger installer was blocked by the execution policy. ProcDump was
installed successfully. `src/platform/dump_stacks_windows.cpp` uses the already
installed Windows DbgEng engine and executes `.reload; ~*kb; lm` against each
dump. The ASan frames use the matching PDB shipped with MSVC 14.44.35207.
Some Windows/CRT frames have only export symbols and offsets; do not interpret
their nearest exported names as fully resolved internal function names.

All four stacks have the following wait chain:

```
Windows process/DLL shutdown
  RtlFreeHeap interceptor
    __asan_win_moveable::IsOwned
      FixedMemoryMap / MoveableMemoryMap
        __sanitizer::AddrHashMap<...>::acquire
          __sanitizer::Mutex::Lock
            Semaphore::Wait -> FutexWait
              WaitOnAddress -> NtWaitForAlertByThreadId
```

Only the exiting thread remains. The blocked mutex is inside the MSVC ASan
allocation-ownership implementation, outside nanoinfer. The dumps do not
establish which former thread owned the lock or why it remained locked.
The older processes can still be dumped even though the process API reports
`HasExited=true`.

Approved M2 `e349784a87bc3d699146f618977fd43c2bbef625` was exported with
`git archive`, built in a fresh directory using the same MSVC ASan settings,
then run once with sequential verbose CTest and a 180-second test timeout.
Result: **6/8 passed**, with `loader_integration` and `model_schema` timing out.
The diagnostic runner captured both idle children before timeout. The failure
therefore predates M3. See the [baseline report](../results/m3-asan-m2-baseline.json).

The loader-integration capture was from the inspector's no-argument usage
path, before any model is opened. The standalone probe below reproduced
output loss and the later symbolizer wait; it has not independently reproduced
the ownership-map shutdown wait. The M2 no-argument capture is the smallest
observed instance of that wait.

## Small reproducer and controls

The [probe](../tests/diagnostics/asan_exit/probe.cpp) contains `puts`, an optional
explicit flush and a return statement. It has no model, loader, framework,
threads or nanoinfer code.
The Python driver checks stdout and the exit code on every launch, captures a
dump on timeout, and stops at the first failure without retrying.

| Experiment | Result | Evidence |
|---|---|---|
| MSVC ASan probe under CTest | Attempt 16 returned the expected exit 2 with empty stdout/stderr | [report](../results/m3-asan-exit-probe.json) |
| Same probe, ASan disabled, under CTest | Attempt 2 returned the expected exit 2 with empty stdout/stderr | [report](../results/m3-exit-probe-control.json) |
| Same non-ASan probe, direct Python launch | Attempt 13 returned exit 0 with empty stdout/stderr | [report](../results/m3-exit-probe-standalone.json) |

The empty-output symptom is independently reproducible without ASan. It must
not be attributed to the captured ASan mutex deadlock without further evidence.
These are diagnostic failures, not successful validation batches.

## Owner-approved local trial

The captured hangs implicate the legacy Windows allocation ownership maps.
Microsoft documents
[`ASAN_OPTIONS=windows_hook_legacy_allocators=false`](https://learn.microsoft.com/en-us/cpp/sanitizers/asan-runtime)
as disabling interception of `GlobalAlloc` and `LocalAlloc`.

Proposed scope: test that option only in the local MSVC ASan validation
process and its children. Keep `windows_hook_rtl_allocators` enabled, keep
compiler instrumentation and fatal error handling, and leave Linux and CI
settings unchanged. No machine-wide environment or security settings change.
The tradeoff is losing ASan checks for those legacy allocator APIs in that
process; nanoinfer does not use them. This is a hypothesis to test, not a
confirmed fix for either symptom.

The owner approved the scoped trial and conditional adoption. First test the
reproducer and deliberate heap-overflow/use-after-free
probes, then run five consecutive complete ASan CTest suites from one fresh
build directory. Stop at the first failure and capture it; no silent retries.
Adopt the workaround only if those checks succeed. Otherwise keep M3 blocked
and investigate the new evidence.

The first trial failed in the heap-overflow probe: ASan reached its error
reporter, then waited in `StartSymbolizerSubprocess` / `CreateProcessA`.
The [captured stack](../results/m3-stack-symbolizer.txt) records all accessible
threads; one thread could not be read by ProcDump (NTSTATUS `0xC0000022`).
This is a different wait from the shutdown ownership-map deadlock.

The owner then approved extending the local trial with `symbolize=false`.
It disables online symbolization, not error detection or fatal exits. Raw
addresses remain in reports for offline symbolization with the matching PDBs.
Both deliberate memory errors exited 1 with their expected ASan diagnostics
under the combined options. Neither option is set in CI or in machine/user
environment settings.

The unflushed probe still lost output on attempt 4 under the combined options.
The non-ASan explicit-flush control passed 200 launches. A separate repeated
CLI-output regression reproduced empty inspector stdout in the unchanged
Debug build. The CLI success paths now explicitly flush stdout before returning;
stderr already uses the standard unit-buffered stream. This change does not
alter the forward pass, output format or acceptance tolerances.

The explicit-flush ASan probe then passed all 200 launches under the extended
options. The project CLI regression passed 200 Debug launches after the change.
The [trial summary](../results/m3-asan-trial-probes.json) retains every failed
record and both complete memory-error diagnostics. Successful repeated-launch
records are summarized by their counts; originals remain under the recorded
`build/diagnostics` paths. The
[failing CLI regression](../results/m3-cli-output-before.txt) and
[passing output](../results/m3-cli-output-after.txt) bracket the fix.

## Reproduce the diagnosis

From the project root, after placing ProcDump in
`build/diagnostics/tools/procdump`:

```powershell
. .\scripts\dev-shell.ps1
git archive --format=tar --output=build/diagnostics/m2-source.tar main
New-Item -ItemType Directory -Force build/diagnostics/m2-source
tar -xf build/diagnostics/m2-source.tar -C build/diagnostics/m2-source
cmake -S build/diagnostics/m2-source -B build/diagnostics/m2-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug -DNANOINFER_ASAN=ON "-DPython3_EXECUTABLE=$PWD/.venv/Scripts/python.exe" "-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=$PWD/build/debug/_deps/googletest-src"
cmake --build build/diagnostics/m2-asan
.\scripts\diagnose-asan.ps1 -BuildDirectory build/diagnostics/m2-asan -Label m2-new-capture

cmake -S tests/diagnostics/asan_exit -B build/diagnostics/exit-probe -G Ninja -DCMAKE_BUILD_TYPE=Debug "-DPython3_EXECUTABLE=$PWD/.venv/Scripts/python.exe" "-DPROCDUMP_EXECUTABLE=$PWD/build/diagnostics/tools/procdump/procdump64.exe"
cmake --build build/diagnostics/exit-probe
ctest --test-dir build/diagnostics/exit-probe --verbose --timeout 180
```

Use another build directory and `-DPROBE_ASAN=OFF` for the non-ASan control.
The runtime and numerical acceptance bounds are unchanged.

For the approved trial, set the options only in the invoking PowerShell
process, use `-DPROBE_FLUSH=ON`, and run the error probe separately:

```powershell
$env:ASAN_OPTIONS = 'windows_hook_legacy_allocators=false:symbolize=false'
python tests/diagnostics/asan_exit/repro.py --executable build/diagnostics/approved-probe/asan_error_probe.exe --directory build/diagnostics/new-error-check --procdump build/diagnostics/tools/procdump/procdump64.exe --check-errors
```

After the probes pass, `scripts/test-asan-reliability.ps1` requires a new build
directory and runs five full sequential CTest suites, stopping at any failure
or captured stall. It restores the caller's previous `ASAN_OPTIONS` afterwards.
It does not retry failed tests. Run it from a fresh PowerShell session so the
default development environment remains unchanged:

```powershell
.\scripts\test-asan-reliability.ps1 -BuildDirectory build/asan-m3-reliability -Label m3-reliability
```

To build the offline stack reader after loading `scripts/dev-shell.ps1`:

```powershell
cl /nologo /std:c++20 /EHsc /W4 /WX /MD /Zi src/platform/dump_stacks_windows.cpp /Fobuild/diagnostics/dump_stacks.obj /Fdbuild/diagnostics/dump-stacks-compile.pdb /Febuild/diagnostics/dump_stacks.exe /link dbgeng.lib /PDB:build/diagnostics/dump_stacks.pdb
$symbols = "$env:VCToolsInstallDir/bin/Hostx64/x64;$PWD/build/asan-m3-reliability"
.\build\diagnostics\dump_stacks.exe build/diagnostics/dumps/CAPTURE.dmp $symbols
```

The tool prints all available thread stacks and a module list. Preserve the
matching executable and PDB for every captured dump.

## Diagnostic monitor regression

The first full-suite trial's CTest log ended with 12/12 passing, but its
monitor aborted with a null-reference error and did not save a complete
diagnostic report. That batch does not count toward the five-suite gate.
The monitor read `Process.Path` twice while the native process could exit
between the reads. A deterministic script-property test reproduces that
transition and the same error:
[before](../results/m3-monitor-before.txt),
[after](../results/m3-monitor-after.txt).
The monitor now snapshots the path once. The regression also checks null paths
and sibling-build exclusion, and runs in the Windows Debug CI job.

The replacement batch in `build/asan-m3-reliability-validated` completed two
12/12 passing CTest suites, but stopped on an idle alert in suite 2. ProcDump
reported no matching PID (exit -2) and produced no dump. This does not prove
a runtime stall. The monitor retained observations by PID alone, so a later
process reusing that PID and the same CPU total could inherit an old idle timer.
A deterministic regression reproduces that false alert:
[before](../results/m3-monitor-identity-before.txt),
[after](../results/m3-monitor-identity-after.txt).
Observations now use PID plus start time, and alerts record start time, idle
timestamp and whether a dump exists.

The original alert lacks start-time evidence, so PID reuse is a supported
monitor failure mode, not a proven explanation for that particular alert.
The [aborted-batch report](../results/m3-monitor-aborted-batch.json) is retained.
Neither interrupted batch is used to satisfy the five-suite gate.
The next batch used another fresh build, `build/asan-m3-final`, label
`m3-final`, with the same approved options and stop-on-failure rule.
It completed three clean 12/12 suites, then failed in suite 4's Python
`export_tiny` fixture after the laptop slept. Windows Power-Troubleshooter
records sleep at `2026-10-20T18:40:02.7143096Z` and wake at
`2026-10-21T05:10:15.3670215Z`. CTest reported 37,814.73 seconds elapsed for
the 180-second timeout and skipped six dependent tests.
The [power-event evidence](../results/m3-power-interruption.json) and
[interrupted suite reports](../results/m3-interrupted-suites.json) are retained.
That interrupted batch does not satisfy the acceptance gate.

After the recorded wake, a new batch uses `build/asan-m3-awake`, label
`m3-awake`, with no power-setting or timeout changes.

That batch stopped after suite 1: `cli_output` failed because `gpt2rt --help`
exceeded its 60-second native-process deadline. The other 11 CTest cases
passed. The idle monitor did not capture this process, so no stack-based cause
can yet be assigned to this timeout. See the
[batch report](../results/m3-awake-failure.json) and
[timeout traceback](../results/m3-help-timeout.txt).
**The five-suite gate is still unmet, and the workaround is not adopted.**

The native-process harness now supports an optional local
`NANOINFER_PROCDUMP` path. On timeout it captures the child before terminating
it, irrespective of CPU activity or whether the process path can still be
queried. The original timeout still fails the test; no retry occurs. A
[failing regression](../results/m3-timeout-capture-before.txt) and
[passing harness suite](../results/m3-timeout-capture-after.txt) verify capture
ordering while retaining exit-code, output and inherited-handle behavior.
An isolated repeated CLI run obtained the missing stack; it is diagnostic
evidence, not an acceptance retry.

## Captured timeout under the approved options

The isolated CLI reproduction stopped on another `gpt2rt --help` timeout and
captured PID 54836 before termination. The
[complete stack](../results/m3-stack-help-timeout.txt) has one remaining thread:

```
Windows DLL/process shutdown
  debug CRT cleanup
    ASan __RuntimeFunctions<Ucrtbased>::FreeDbg
      asan_mz_size -> Allocator::AllocationSize
        LargeMmapAllocator::GetBlockBegin
          StaticSpinMutex::LockSlow
            internal_sched_yield -> SleepEx
```

This is a second ASan allocator lock during shutdown, reached through the debug
CRT interceptor even with legacy allocator hooks disabled. The lock owner is
not established. A CPU-idle heuristic can miss a spinning waiter; the new
deadline capture obtained the stack regardless. `--help` opens no model and
performs no inference, providing a model-free reproduction command.

The owner approved the next local trial: build ASan with
`-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL` (`/MD`) instead of `/MDd`.
Keep `/O1`, `/Zi`, all ASan instrumentation and fatal errors, initially with the
same approved environment options. This tests whether avoiding the debug CRT
interceptor removes the shutdown wait. It is a hypothesis, not a confirmed fix.
[Microsoft supports ASan with both CRT variants](https://learn.microsoft.com/en-us/cpp/sanitizers/asan-building?view=msvc-170);
[CMake documents the runtime selection](https://cmake.org/cmake/help/latest/variable/CMAKE_MSVC_RUNTIME_LIBRARY.html).

Tradeoff: that local ASan build would omit debug CRT and debug STL iterator
checks. The ordinary Debug build retains `/MDd` and those checks. CI settings
remain unchanged. Adoption still requires deliberate memory-error probes
and five consecutive full suites from a fresh build.
The trial is opt-in through `scripts/test-asan-reliability.ps1 -ReleaseCrt`;
neither the project presets nor CI select `/MD` for ASan by default.

The `/MD` error probes detected both deliberate memory errors, and the flushed
standalone probe passed 200 launches ([probe evidence](../results/m3-md-probes.json)).
The fresh `build/asan-m3-md` full-suite batch then failed in its first suite:
11/12 tests passed; `cli_output` timed out in `gpt2rt --help`.
The batch stopped without retrying or starting suite 2.
The [failure report](../results/m3-md-failure.json) records the tested revision
and deadline capture, which is separate from the CPU-idle monitor's captures.

The [release-CRT stack](../results/m3-stack-md-timeout.txt) again has one thread
waiting in `StaticSpinMutex::LockSlow`, reached through
`LargeMmapAllocator::GetBlockBegin`, `asan_mz_size`, and
`__RuntimeFunctions<Ucrtbase>::FreeBase` during DLL shutdown.
Changing the CRT changed the interceptor from `FreeDbg` to `FreeBase` but did
not eliminate the allocator wait. **The `/MD` trial failed and is not adopted.**
The other local runtime options also remain experimental because the five-suite
gate has not passed.

The current ordinary `/MDd` Debug validation passed all 12 CTest cases and all
43 Python tests. Full outputs:
[build](../results/m3-current-debug-build.txt),
[CTest](../results/m3-current-debug-tests.txt),
[pytest](../results/m3-current-python.txt).

## Owner-approved compatibility-environment control

The invoking process inherits `__COMPAT_LAYER=DetectorsAppHealth`, and the
captured process loads `apphelp.dll`. Neither observation proves that Windows
compatibility processing caused the ASan wait. The captured lock owner remains
unknown; changing more allocator hooks would not directly test this hypothesis.

The owner approved this control: omit `__COMPAT_LAYER` only from
the local diagnostic child-process environment. Return to the default `/MDd`
build and retain the two already-approved ASan options initially. Do not alter
registry settings, system/user environment variables, installed software, or
security tools. The behavioral tradeoff is that these test children no longer
request the inherited compatibility layer; ASan instrumentation, allocation
checks, fatal exits, and debug CRT checks remain enabled.

Run the error probes and repeated model-free CLI check first, then use a fresh
build for five complete suites with stop-on-first-failure. Record the inherited
and effective compatibility values. Any failure still requires capture and
investigation. The opt-in `-ClearCompatibilityLayer` switch records both values
and restores the caller's process environment in `finally`.

Both memory-error probes passed and all 200 standalone exit checks passed.
The first CLI invocation used an incorrect fixture path and immediately returned
the expected file-not-found error. That invocation error and the corrected path
are recorded in the [trial report](../results/m3-compat-probes.json).
With the correct fixture, the CLI check timed out in `gpt2rt --help` between
pairs 176 and 200 of the requested 1,000 pairs. The driver stopped at that first
failure. No full-suite acceptance batch was started.

The [captured stack](../results/m3-stack-compat-timeout.txt), PID 28224, contains
one thread waiting in `AddrHashMap::acquire` through
`__asan_win_moveable::IsOwned` and `__asan_wrap_RtlFreeHeap` during shell DLL
cleanup. `apphelp.dll` is absent from the module list. Removing the inherited
compatibility setting therefore did not prevent the failure. The lock owner
remains unknown. **The compatibility control failed and is not adopted.**

## Unexecuted RTL allocator proposal

The latest stack shows that disabling legacy allocator hooks does not prevent
the RTL free interceptor from entering the ownership map. The next proposed
local control is to add `windows_hook_rtl_allocators=false`, retaining `/MDd`,
`windows_hook_legacy_allocators=false`, and `symbolize=false`, and restoring the
ordinary inherited compatibility environment. The owner declined further
runtime-option trials. This control has not been executed and will not be
adopted under the current authorization.

[Microsoft documents the RTL allocator option](https://learn.microsoft.com/en-us/cpp/sanitizers/asan-runtime?view=msvc-170).
The tradeoff is loss of ASan allocation tracking for Windows heap APIs such as
`HeapAlloc`/`HeapFree` and their RTL counterparts. The source tree has no direct
calls to those allocation APIs. Standard C/C++ allocation instrumentation must
remain effective, verified by the overflow and use-after-free probes before
any full-suite batch. Neither suppression of reported errors nor successful
retries are acceptable. CI, registry, persistent environment, and security
settings would remain unchanged. The five consecutive fresh-build suite gate
still applies; the new control is a hypothesis, not a confirmed solution.
