# Starts the title with settings tuned for NVIDIA RTX 4090 (Ada Lovelace, 24 GB VRAM,
# Resizable BAR) and 32 GB system RAM hosts (README.md, "How to run it").
#   scripts\play.ps1 -Exe <path of your own copy's executable>
# -Safe             without the newer, faster ways (the first thing to try when something is wrong)
# -RayTracing       the title's own ray-traced indirect light: lit interiors, heavier on the GPU
# -NoDxr            disable the Phase 1 DXR 1.1 RT Core bridge observer (VICTORY_DXR=1 is on by default)
# -LowRam           force the 32 GB RAM profile (auto-detected on <= 36 GB physical RAM hosts)
# -SmallCardBuffers experimental: place small buffers (>= 64 KiB) in VRAM (VICTORY_CARD_BUFFERS_MIN_KB=64)
# -Awake            enable async-compute clock keep-awake (off by default on NVIDIA)
# -Console          console profile: 'series-x' (default) or 'series-s' (lighter, matches cool_fox 25+ FPS target)
# -Pareto           high-FPS preset: Series S + LevelsOnDemand + SmallCardBuffers + Awake + release build
# -Build            target\diag (the default, with diagnostics) or target\release
# -More             other settings: -More @{VICTORY_PANEL='1'}
# -Log              custom log file path (outside repo; default: ..\captures\victory-YYYYMMDD-HHMMSS.log)
# -NoLog            disable writing a log file (console only)
# Run it from a normal window, not as administrator: the driver then keeps its shader cache elsewhere.
param(
    [Parameter(Mandatory)][string]$Exe,
    [switch]$Safe,
    [switch]$RayTracing,
    [switch]$NoDxr,
    [switch]$LowRam,
    [switch]$SmallCardBuffers,
    [switch]$Awake,
    [switch]$Pareto,
    [ValidateSet('series-x', 'series-s')][string]$Console = 'series-x',
    [ValidateSet('diag', 'release')][string]$Build = 'diag',
    [hashtable]$More = @{},
    [string]$Log = '',
    [switch]$NoLog
)
if ($Pareto) {
    if (-not $PSBoundParameters.ContainsKey('Console')) { $Console = 'series-s' }
    if (-not $PSBoundParameters.ContainsKey('Build'))   { $Build = 'release' }
    $Awake = $true
}
$settings = @{
    # Core async & GPU-side optimizations (ported from VIctory-7900XTX and verified on RTX 4090 D)
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
    VICTORY_TITLE_PRIORITIES    = '0'
    VICTORY_SHOW_AFTER          = '1'
    VICTORY_AMBIENT             = '0.8'
    VICTORY_D3D12_DRED          = 'light'
    VICTORY_SKIP_PIPELINES      = 'none'
    # NVIDIA Ada / RTX 4090 specific tuning:
    # - 2 GiB read-only windows offloaded to VRAM via ReBAR GPU_UPLOAD (4.32x faster GPU reads, -2 GiB SysRAM)
    # - Disable AMD-specific synchronous MakeResident + Aside-queue fence wait (saves 60-85s of stalls & 200-2000ms spikes)
    # - Keep VICTORY_GROUP_SYNC at default (2) so looped compute shaders keep LDS GroupSync
    #   and 100% match `victory.exe shaders` cache keys
    # - Enable Phase 1 DXR 1.1 RT Core bridge observer on DXR Tier 1.1 hosts
    # - Keep VICTORY_FIRSTLANE_OWN=1 so all 53 XboxBvhBuilder shaders compile cleanly on NVIDIA Wave32
    VICTORY_WINDOWS_BAR_MB      = '2048'
    VICTORY_WAIT_RESIDENT       = '0'
    VICTORY_DXR                 = '1'
    VICTORY_FIRSTLANE_OWN       = '1'
}
if (-not $RayTracing) {
    $settings.VICTORY_AMBIENT_AIR = '1.5'
}
if ($SmallCardBuffers) {
    $settings.VICTORY_CARD_BUFFERS_MIN_KB = '64'
}
if ($Awake) {
    $settings.VICTORY_CARD_AWAKE = '1'
}
if ($Pareto) {
    $settings.VICTORY_LEVELS_ON_DEMAND = '1'
    # Avoid PowerShell stdout pipe back-pressure from 18,000+ INFO lines (matches cool_fox Play-VIctory.cmd)
    $settings.VICTORY_LOG              = 'warn'
}
if ($LowRam) {
    $settings.VICTORY_LOW_RAM         = '1'
    $settings.VICTORY_WINDOW_KB       = '1024'
    $settings.VICTORY_WINDOW_TAILS    = '1'
    $settings.VICTORY_LIST_POOL_MAX   = '128'
    $settings.VICTORY_PIPELINE_WORKERS = '8'
}
if ($Safe) {
    $settings.VICTORY_UNDER_WIDE = '0'
    $settings.VICTORY_LAND_ECHO  = '0'
    $settings.VICTORY_WINDOWS_BAR_MB = '0'
    $settings.VICTORY_WAIT_RESIDENT  = '1'
    $settings.VICTORY_DXR        = '0'
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
