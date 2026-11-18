# Build Monado's openxr-simple-example for Windows.
#
# The demo is the quick-start application described in the README: it is
# Boost-licensed, small enough to read, renders stereo cubes, and exercises
# tracked controllers and haptics. It is built in CI so users do not need a
# compiler or vcpkg installation.

param(
    [string]$Revision = "bf920800ff156424e2710e3ca63f84be121e62cb",
    [string]$OutputDir = "out/demo-windows"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$source = Join-Path $root "out/openxr-src/openxr-simple-example"
$build = Join-Path $root "out/demo-windows-build"
$output = [IO.Path]::GetFullPath((Join-Path $root $OutputDir))

if (-not (Test-Path (Join-Path $source ".git"))) {
    New-Item -ItemType Directory -Force (Split-Path -Parent $source) | Out-Null
    git clone https://gitlab.freedesktop.org/monado/demos/openxr-simple-example.git $source
    if ($LASTEXITCODE -ne 0) { throw "could not clone openxr-simple-example" }
}
git -C $source fetch --depth 1 origin $Revision
if ($LASTEXITCODE -ne 0) { throw "could not fetch openxr-simple-example revision" }
git -C $source checkout --detach $Revision
if ($LASTEXITCODE -ne 0) { throw "could not check out openxr-simple-example" }

# The upstream CMake project treats MSVC warnings as errors (/WX). Newer
# Visual Studio toolchains can introduce warnings in this small C program, and
# a warning should not block the demo archive. Remove only that switch; keep
# every other upstream compile option.
$cmakeLists = Join-Path $source "CMakeLists.txt"
$text = [IO.File]::ReadAllText($cmakeLists)
$text = $text -replace '\s/WX(?=\s|\))', ''

# vcpkg's SDL2 package redefines main() to SDL_main() and documents linking
# SDL2::SDL2main explicitly, but upstream only links ${SDL2_LIBRARIES}. Add the
# target conditionally so non-vcpkg toolchains that fold SDL2main into
# ${SDL2_LIBRARIES} keep working.
if ($text -notmatch 'SDL2::SDL2main') {
    $text = $text.TrimEnd() + "`ntarget_link_libraries(openxr-example PRIVATE `$<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>)`n"
}
[IO.File]::WriteAllText($cmakeLists, $text)

$vcpkg = $env:VCPKG_INSTALLATION_ROOT
if (-not $vcpkg) { $vcpkg = "C:\vcpkg" }
if (-not (Test-Path $vcpkg)) {
    throw "vcpkg is required to build the Windows OpenXR demo"
}
$vcpkgExe = Join-Path $vcpkg "vcpkg.exe"
if (-not (Test-Path $vcpkgExe)) {
    & (Join-Path $vcpkg "bootstrap-vcpkg.bat") -disableMetrics
    if ($LASTEXITCODE -ne 0) { throw "could not bootstrap vcpkg" }
}

& $vcpkgExe install sdl2:x64-windows openxr-loader:x64-windows --clean-after-build
if ($LASTEXITCODE -ne 0) { throw "vcpkg could not install the demo dependencies" }

# The CMake project finds SDL2 and OpenXR through vcpkg's config packages.
# Use CMake's default generator so the build follows the Visual Studio version
# installed on the runner (GitHub's windows-latest moved from VS 2022 to
# VS 2026); both generators are multi-config and support --config Release.
$toolchain = Join-Path $vcpkg "scripts/buildsystems/vcpkg.cmake"
cmake -S $source -B $build -A x64 `
    -DCMAKE_TOOLCHAIN_FILE="$toolchain" `
    -DVCPKG_TARGET_TRIPLET=x64-windows
if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed" }
cmake --build $build --config Release --parallel
if ($LASTEXITCODE -ne 0) { throw "the demo build failed" }

$executable = Get-ChildItem -Path $build -Recurse -Filter "openxr-example.exe" |
    Select-Object -First 1
if (-not $executable) { throw "openxr-example.exe was not produced" }

if (Test-Path $output) { Remove-Item -Recurse -Force $output }
New-Item -ItemType Directory -Force $output | Out-Null
Copy-Item $executable.FullName (Join-Path $output "openxr-example.exe")

# The loader, SDL2 and the app-local VC runtime must sit next to the
# executable on Windows. vcpkg puts every release dependency and the VC
# redistributable DLLs in installed/x64-windows/bin.
$installed = Join-Path $vcpkg "installed/x64-windows"
$bin = Join-Path $installed "bin"
$dlls = @(Get-ChildItem -Path $bin -Filter "*.dll" -ErrorAction SilentlyContinue)
if ($dlls.Count -eq 0) {
    throw "no runtime DLLs were found in $bin"
}
foreach ($dll in $dlls) {
    Copy-Item $dll.FullName (Join-Path $output $dll.Name)
}

# The demo links the MSVC runtime dynamically. Bundle the redistributable DLLs
# for machines without Visual Studio; older Windows installs may only have an
# incompatible baseline version. vswhere ships with the Build Tools.
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    $vsPath = & $vswhere -latest -products * -property installationPath 2>$null
    $redist = if ($vsPath) { Join-Path $vsPath "VC/Redist/MSVC" }
    if ($redist -and (Test-Path $redist)) {
        $crt = Get-ChildItem $redist -Directory |
            Sort-Object Name -Descending |
            ForEach-Object {
                Get-ChildItem (Join-Path $_.FullName "x64") -Directory -Filter "Microsoft.VC*.CRT" `
                    -ErrorAction SilentlyContinue
            } |
            Select-Object -First 1
        if ($crt) {
            Get-ChildItem $crt.FullName -Filter "*.dll" |
                ForEach-Object { Copy-Item $_.FullName (Join-Path $output $_.Name) }
        }
    }
}

foreach ($document in @("LICENSE", "Readme.md")) {
    $path = Join-Path $source $document
    if (Test-Path $path) { Copy-Item $path (Join-Path $output $document) }
}

$archive = Join-Path $root "out/openxr-simple-example-windows-x64.zip"
if (Test-Path $archive) { Remove-Item -Force $archive }
Compress-Archive -Path (Join-Path $output "*") -DestinationPath $archive
Write-Host "Demo: $archive"
