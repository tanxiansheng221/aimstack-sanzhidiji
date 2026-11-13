# aimstack engine build + smoke (ASCII only)
# 自动探测 MSVC / Windows SDK 最新版本, 产物: build\aimp_probe.exe
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot

$msvcRoot = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC'
if (-not (Test-Path $msvcRoot)) { Write-Output 'FATAL: MSVC not found (install VS2022 BuildTools, C++ workload)'; exit 9 }
$msvc = Get-ChildItem $msvcRoot -Directory | Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName

$sdk = 'C:\Program Files (x86)\Windows Kits\10'
$sdkv = Get-ChildItem "$sdk\Lib" -Directory | Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty Name

if (-not (Test-Path "$msvc\bin\Hostx64\x64\cl.exe")) { Write-Output 'FATAL: cl.exe not found'; exit 9 }
Write-Output ("MSVC " + (Split-Path $msvc -Leaf) + "  SDK " + $sdkv)

$env:LIB = "$msvc\lib\x64;$sdk\Lib\$sdkv\ucrt\x64;$sdk\Lib\$sdkv\um\x64"
$env:INCLUDE = "$msvc\include;$sdk\Include\$sdkv\ucrt;$sdk\Include\$sdkv\um;$sdk\Include\$sdkv\shared;$sdk\Include\$sdkv\winrt"

New-Item -ItemType Directory -Force -Path build | Out-Null
& "$msvc\bin\Hostx64\x64\cl.exe" /nologo /O2 /W4 /GS /DYNAMICBASE /fp:precise /arch:AVX2 /utf-8 *.c Synchronization.lib /Fe:build\aimp_probe.exe /Fo:build\
if ($LASTEXITCODE -ne 0) { Write-Output "BUILD_FAIL exit=$LASTEXITCODE"; exit 1 }
Write-Output 'BUILD_OK'

# 冒烟: 仅当小测试权重存在时跑 (可用 tools/make_mini_llama.py 生成)
$mini = Join-Path $PSScriptRoot 'build\mini_llama.gguf'
if (Test-Path $mini) {
  & .\build\aimp_probe.exe $mini --gen 3,7,11,5 3
  Write-Output "GEN_EXIT=$LASTEXITCODE"
}
