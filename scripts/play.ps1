# Starts the title with unified settings tuned for NVIDIA RTX 4090 (Ada Lovelace,
# 24 GB VRAM, Resizable BAR) and 32 GB system RAM hosts.
#   scripts\play.ps1 -Exe <path of your own copy's executable>
#
# Defaults: release build, Series S profile (20+ FPS target), VICTORY_WAIT_RESIDENT=0,
# VICTORY_LEVELS_ON_DEMAND=1, VICTORY_CARD_AWAKE=1, VICTORY_LOG=warn, ReBAR 2 GiB.
#
# Common switches:
# -SeriesX          switch from Series S to Series X profile (higher resolution/detail)
# -RayTracing       enable the title's ray-traced indirect light (lit interiors)
# -Diag             use target\diag build with INFO logging for diagnostics
# -NoLog            console only (skip writing log file)
# -Safe             disable newer optimizations (first thing to try if something breaks)
# -NoDxr            disable DXR 1.1 RT Core bridge (VICTORY_DXR=1 is on by default)
# -LowRam           force the 32 GB RAM profile (auto-detected on <= 36 GB RAM hosts)
# -SmallCardBuffers experimental: place small buffers (>= 64 KiB) in VRAM
# -Console          explicit 'series-s' (default) or 'series-x'
# -Build            explicit 'release' (default) or 'diag'
# -More             extra env overrides: -More @{VICTORY_PANEL='1'}
# -Log              custom log file path (default: ..\captures\victory-YYYYMMDD-HHMMSS.log)
# Run it from a normal window, not as administrator: the driver then keeps its shader cache elsewhere.
param(
    [Parameter(Mandatory)][string]$Exe,
    [switch]$SeriesX,
    [switch]$RayTracing,
    [switch]$Diag,
    [switch]$Safe,
    [switch]$NoDxr,
    [switch]$LowRam,
    [switch]$SmallCardBuffers,
    [switch]$Pareto,
    [ValidateSet('series-s', 'series-x')][string]$Console = 'series-s',
    [ValidateSet('release', 'diag')][string]$Build = 'release',
    [hashtable]$More = @{},
    [string]$Log = '',
    [switch]$NoLog
)
if ($SeriesX -and -not $PSBoundParameters.ContainsKey('Console')) {
    $Console = 'series-x'
}
if ($Diag -and -not $PSBoundParameters.ContainsKey('Build')) {
    $Build = 'diag'
}
$settings = @{
    # Core async & GPU-side optimizations (merged VIctory-7900XTX + cool_fox + Victory-4090)
    VICTORY_TEXTURE_NO_WAIT     = '1'
    VICTORY_RELAY               = '2'
    VICTORY_WIDE_BACK           = '1'
    VICTORY_CARD_REHASH_RUNS    = '1'
    VICTORY_JOIN_COPIES         = '1'
    VICTORY_INDIRECT_WINDOW     = '1'
    VICTORY_RELAY_COPY          = '1'
    VICTORY_CARD_STEPS          = '2'
    VICTORY_INDIRECT_BATCH      = '2'
    VICTORY_INDIRECT_REST       = '0'
    VICTORY_MAP_THREADS         = '1'
    VICTORY_SHOW_AFTER          = '1'
    VICTORY_AMBIENT             = '0.8'
    VICTORY_LEVELS_ON_DEMAND    = '1'
    VICTORY_CARD_AWAKE          = '1'
    VICTORY_SKIP_PIPELINES      = 'none'
    # NVIDIA Ada / RTX 4090 specific tuning:
    # - 2 GiB read-only windows offloaded to VRAM via ReBAR GPU_UPLOAD (keeps free RAM > 3 GiB so textures stay sharp!)
    # - Disable AMD-specific synchronous MakeResident + Aside-queue fence wait (eliminates 60-85s stalls)
    # - Keep VICTORY_GROUP_SYNC at default (2) so looped compute shaders keep LDS GroupSync
    # - Enable DXR 1.1 RT Core bridge on DXR Tier 1.1 hosts
    # - Keep VICTORY_FIRSTLANE_OWN=1 so all 53 XboxBvhBuilder shaders compile cleanly on NVIDIA Wave32
    VICTORY_WINDOWS_BAR_MB      = '2048'
    VICTORY_WAIT_RESIDENT       = '0'
    VICTORY_DXR                 = '1'
    VICTORY_FIRSTLANE_OWN       = '1'
}
if ($Diag) {
    $settings.VICTORY_D3D12_DRED = 'light'
} else {
    # Keep stdout lean to prevent PowerShell pipe back-pressure
    $settings.VICTORY_LOG = 'warn'
}
if (-not $RayTracing) {
    $settings.VICTORY_AMBIENT_AIR = '1.5'
}
if ($SmallCardBuffers) {
    $settings.VICTORY_CARD_BUFFERS_MIN_KB = '64'
}
if ($LowRam) {
    $settings.VICTORY_LOW_RAM          = '1'
    $settings.VICTORY_WINDOW_KB        = '1024'
    $settings.VICTORY_WINDOW_TAILS     = '1'
    $settings.VICTORY_LIST_POOL_MAX    = '128'
    $settings.VICTORY_PIPELINE_WORKERS = '8'
}
if ($Safe) {
    $settings.VICTORY_UNDER_WIDE     = '0'
    $settings.VICTORY_LAND_ECHO      = '0'
    $settings.VICTORY_WINDOWS_BAR_MB = '0'
    $settings.VICTORY_WAIT_RESIDENT  = '1'
    $settings.VICTORY_DXR            = '0'
} else {
    $settings.VICTORY_THIN         = '1'
    $settings.VICTORY_JOIN_PARTS   = '8'
    $settings.VICTORY_UPLOAD_AHEAD = '1'
}
if ($NoDxr) { $settings.VICTORY_DXR = '0' }
if ($RayTracing) { $settings.VICTORY_BVH_BUILDER = '1' }
foreach ($k in $More.Keys) { $settings[$k] = $More[$k] }

$victory = Join-Path $PSScriptRoot "..\target\$Build\victory.exe"
if (-not (Test-Path $victory)) { Write-Host "No $victory. Build it first (README.md)."; return }
if (-not (Test-Path $Exe)) { Write-Host "No $Exe."; return }
# (Other settings left over in this window go; where your files are stays.)
$paths = 'VICTORY_CONFIG', 'VICTORY_GAME_DIR', 'VICTORY_CACHE_DIR', 'VICTORY_DATA_DIR'
Get-ChildItem Env:VICTORY_* | Where-Object { $_.Name -notin $paths } | ForEach-Object { Remove-Item "Env:$($_.Name)" }
foreach ($k in $settings.Keys) { Set-Item "Env:$k" $settings[$k] }
$Exe = (Resolve-Path $Exe).Path
Push-Location (Join-Path $PSScriptRoot '..')   # victory.toml is looked for here
if (-not $NoLog) {
    $logPath = if ($Log) {
        $Log
    } else {
        Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path ("captures\victory-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $logPath) | Out-Null
    Write-Host "Log: $logPath" -ForegroundColor Cyan
    $enc = New-Object System.Text.UTF8Encoding($false)
    $writer = New-Object System.IO.StreamWriter($logPath, $false, $enc)
    $writer.AutoFlush = $true
    $prevEap = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $victory run $Exe --console $Console --execute 2>&1 | ForEach-Object {
            $line = "$_"
            Write-Host $line
            $writer.WriteLine($line)
        }
    } finally {
        $ErrorActionPreference = $prevEap
        if ($writer) { $writer.Dispose() }
        Pop-Location
    }
} else {
    try { & $victory run $Exe --console $Console --execute } finally { Pop-Location }
}
