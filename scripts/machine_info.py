"""Collect the machine facts recorded with every benchmark result."""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def _run(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip()


def _windows():
    script = (
        "$p = Get-CimInstance Win32_Processor | Select-Object -First 1;"
        "$s = Get-CimInstance Win32_ComputerSystem;"
        "$o = Get-CimInstance Win32_OperatingSystem;"
        "$m = @(Get-CimInstance Win32_PhysicalMemory | ForEach-Object {"
        "  [ordered]@{capacity_bytes=$_.Capacity; rated_mt_s=$_.Speed;"
        "  configured_mt_s=$_.ConfiguredClockSpeed; part=$_.PartNumber.Trim()} });"
        "[ordered]@{cpu=$p.Name.Trim(); physical_cores=$p.NumberOfCores;"
        "logical_processors=$p.NumberOfLogicalProcessors; reported_base_mhz=$p.MaxClockSpeed;"
        "l2_kib=$p.L2CacheSize; l3_kib=$p.L3CacheSize; ram_bytes=$s.TotalPhysicalMemory;"
        "memory_modules=$m; os=$o.Caption; os_version=$o.Version; kernel_build=$o.BuildNumber}"
        " | ConvertTo-Json -Depth 4"
    )
    info = json.loads(_run(["powershell", "-NoProfile", "-Command", script]) or "{}")
    info["power_scheme"] = _run(["powercfg", "/getactivescheme"])
    return info


def _linux():
    info = {"lscpu": _run(["lscpu"]), "uname": _run(["uname", "-a"])}
    try:
        info["meminfo_total"] = Path("/proc/meminfo").read_text().splitlines()[0]
    except OSError:
        pass
    governor = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if governor.exists():
        info["scaling_governor"] = governor.read_text().strip()
    return info


def _compiler(build):
    """Compiler identity and the exact flags of the GEMM translation units."""
    info = {}
    cache = build / "CMakeCache.txt"
    if cache.exists():
        for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
            key, _, value = line.partition("=")
            name = key.split(":")[0]
            if name in ("CMAKE_CXX_COMPILER", "CMAKE_BUILD_TYPE", "CMAKE_CXX_FLAGS_RELEASE"):
                info[name] = value
    commands = build / "compile_commands.json"
    if commands.exists():
        for entry in json.loads(commands.read_text(encoding="utf-8")):
            source = entry["file"].replace("\\", "/")
            for unit in ("src/kernels/gemm.cpp", "src/kernels/gemm_avx2.cpp"):
                if source.endswith(unit):
                    info[f"command {unit}"] = entry.get("command", "")
    compiler = info.get("CMAKE_CXX_COMPILER", "")
    if compiler:
        if compiler.lower().endswith("cl.exe"):
            banner = subprocess.run([compiler], capture_output=True, text=True, check=False)
            info["version"] = banner.stderr.strip().splitlines()[0] if banner.stderr else ""
        else:
            info["version"] = _run([compiler, "--version"]).splitlines()[0]
    return info


def collect(build):
    info = dict(
        platform=platform.platform(),
        python=sys.version.split()[0],
        machine=platform.machine(),
        system=_windows() if os.name == "nt" else _linux(),
        compiler=_compiler(Path(build)),
    )
    return info


if __name__ == "__main__":
    print(json.dumps(collect(sys.argv[1] if len(sys.argv) > 1 else "build/release"), indent=2))
