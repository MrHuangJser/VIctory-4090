# Starts the title with unified settings tuned for NVIDIA RTX 4090 (Ada Lovelace,
# 24 GB VRAM, Resizable BAR) and 32 GB system RAM hosts.
#   scripts\play.ps1 -Exe <path of your own copy's executable>
#
# Defaults: release build, Xbox Series S (XSS) + 1080p (VICTORY_DISPLAY_MODE=1920x1080),
# VICTORY_CARD_BUFFERS_MIN_KB=64 (matches cool_fox; saves ~5 GiB RAM / ~2.6 GiB shared),
# VICTORY_WAIT_RESIDENT=0, VICTORY_LEVELS_ON_DEMAND=1, VICTORY_CARD_AWAKE=1, ReBAR off.
#
# Common switches:
# -SeriesX          switch from Series S to Series X profile (still defaults to 1080p unless overridden)
# -RayTracing       enable the title's ray-traced indirect light + DXR 1.1 RT Core bridge
# -Diag             use target\diag build with INFO logging for diagnostics
# -NoLog            console only (skip writing log file)
# -Safe             disable newer optimizations (first thing to try if something breaks)
# -NoDxr            disable DXR 1.1 RT Core bridge when -RayTracing is enabled
# -LowRam           force the 32 GB RAM profile (auto-detected on <= 36 GB RAM hosts)
# -SmallCardBuffers (on by default: places >= 64 KiB buffers in VRAM)
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
    # Default display binding: Xbox Series S (XSS) + 1080p (override via -SeriesX / -More @{VICTORY_DISPLAY_MODE='...'})
    VICTORY_DISPLAY_MODE        = '1920x1080'
    # NVIDIA Ada / RTX 4090 specific tuning:
    # - Place >= 64 KiB title buffers directly in VRAM (matches cool_fox pareto-residency: -5 GiB SysRAM, -2.6 GiB shared, -8 ms copies)
    # - Keep ReBAR loose-window offload OFF by default (0) so 24 GiB VRAM stays reserved for textures & card buffers without spilling to shared SysRAM
    # - Disable AMD-specific synchronous MakeResident + Aside-queue fence wait (eliminates 60-85s stalls)
    # - Keep VICTORY_GROUP_SYNC at default (2) so looped compute shaders keep LDS GroupSync
    # - Keep VICTORY_FIRSTLANE_OWN=1 so all 53 XboxBvhBuilder shaders compile cleanly on NVIDIA Wave32 without busting shader cache across RT toggles
    # - Enable DXR 1.1 RT Core bridge only when -RayTracing is enabled (zero overhead when RT is off)
    VICTORY_CARD_BUFFERS_MIN_KB = '64'
    VICTORY_WINDOWS_BAR_MB      = '0'
    VICTORY_WAIT_RESIDENT       = '0'
    VICTORY_DXR                 = '0'
    VICTORY_FIRSTLANE_OWN       = '1'
    # Keep wide workers non-spinning by default so 12 thin-list + game threads never lose cores
    VICTORY_WIDE_SPIN           = '0'
}
if ($Diag) {
    $settings.VICTORY_D3D12_DRED = 'light'
} else {
    # Keep stdout/log lean (avoid per-second land/behind/draw I/O back-pressure) while keeping 5s FPS + CPU/GPU/RAM/VRAM telemetry
    $settings.VICTORY_LOG = if ($RayTracing) {
        'warn,victory_gpu::host=info,victory_gpu::telemetry=info,victory_gpu::dxr=info'
    } else {
        'warn,victory_gpu::host=info,victory_gpu::telemetry=info'
    }
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
    $settings.VICTORY_WIDE_SPIN      = '0'
    $settings.Remove('VICTORY_CARD_BUFFERS_MIN_KB')
} else {
    $settings.VICTORY_THIN         = '1'
    $settings.VICTORY_JOIN_PARTS   = '8'
    $settings.VICTORY_UPLOAD_AHEAD = '1'
}
if ($RayTracing) {
    $settings.VICTORY_BVH_BUILDER = '1'
    if (-not $NoDxr -and -not $Safe) {
        $settings.VICTORY_DXR = '1'
    }
}
if ($NoDxr) { $settings.VICTORY_DXR = '0' }
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
