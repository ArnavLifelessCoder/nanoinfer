$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'dev-shell.ps1')
$projectRoot = Split-Path -Parent $PSScriptRoot
$processor = Get-CimInstance Win32_Processor | Select-Object -First 1
$system = Get-CimInstance Win32_ComputerSystem
$os = Get-CimInstance Win32_OperatingSystem
$compiler = (& cl 2>&1 | Out-String).Trim()
$pythonVersions = (& python -c 'import json,sys,numpy,torch,transformers,tiktoken; print(json.dumps(dict(python=sys.version,numpy=numpy.__version__,torch=torch.__version__,transformers=transformers.__version__,tiktoken=tiktoken.__version__)))') |
    ConvertFrom-Json
$record = [ordered]@{
    purpose = 'M1 correctness validation, not a performance benchmark'
    recorded_at = (Get-Date).ToString('o')
    cpu = $processor.Name.Trim()
    physical_cores = $processor.NumberOfCores
    logical_processors = $processor.NumberOfLogicalProcessors
    ram_bytes = $system.TotalPhysicalMemory
    os = $os.Caption
    os_version = $os.Version
    kernel_build = $os.BuildNumber
    compiler = $compiler
    cmake = (& cmake --version | Select-Object -First 1)
    ninja = (& ninja --version)
    python = $pythonVersions
    debug_flags = '/Od /Zi /RTC1 /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /std:c++20'
    asan_flags = '/O1 /Zi /fsanitize=address /W4 /WX /permissive- /Zc:__cplusplus /utf-8 /std:c++20 /INCREMENTAL:NO'
}
$record | ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath (Join-Path $projectRoot 'results\m1-environment.json') -Encoding utf8
