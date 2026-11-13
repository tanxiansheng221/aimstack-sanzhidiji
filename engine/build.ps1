# aimp M2.6 build + regression (ASCII only)
$ErrorActionPreference = 'Continue'
Set-Location H:\AimCode\aimp
$msvc = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207'
$sdk  = 'C:\Program Files (x86)\Windows Kits\10'
if (-not (Test-Path "$msvc\bin\Hostx64\x64\cl.exe")) { Write-Output 'FATAL: cl.exe not found'; exit 9 }
$env:LIB = "$msvc\lib\x64;$sdk\Lib\10.0.26100.0\ucrt\x64;$sdk\Lib\10.0.26100.0\um\x64"
$env:INCLUDE = "$msvc\include;$sdk\Include\10.0.26100.0\ucrt;$sdk\Include\10.0.26100.0\um;$sdk\Include\10.0.26100.0\shared;$sdk\Include\10.0.26100.0\winrt"
& "$msvc\bin\Hostx64\x64\cl.exe" /nologo /O2 /W4 /GS /DYNAMICBASE /fp:precise /arch:AVX2 /utf-8 src\*.c Synchronization.lib /Fe:build\aimp_probe.exe /Fo:build\
if ($LASTEXITCODE -ne 0) { Write-Output "BUILD_FAIL exit=$LASTEXITCODE"; exit 1 }
Write-Output 'BUILD_OK'
& .\build\aimp_probe.exe .\build\mini_llama.gguf --gen 3,7,11,5 3
Write-Output "GEN_EXIT=$LASTEXITCODE"
