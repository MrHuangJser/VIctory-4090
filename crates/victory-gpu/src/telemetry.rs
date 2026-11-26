//! Periodic host hardware resource utilization telemetry (CPU, RAM, VRAM, and
//! NVIDIA GPU core/memory utilization, clock, power, and temperature via
//! dynamically loaded `nvml.dll`). Logged every 5 seconds alongside the frame
//! rate summary with zero extra crate dependencies.

use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use windows_sys::Win32::Foundation::FILETIME;
use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};
use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes, GetSystemTimes};

const GIB: f64 = (1u64 << 30) as f64;
const NVML_SUCCESS: i32 = 0;
const NVML_CLOCK_GRAPHICS: u32 = 0;
const NVML_TEMPERATURE_GPU: u32 = 0;

#[repr(C)]
#[derive(Default)]
struct NvmlUtilization {
    gpu: u32,
    memory: u32,
}

#[repr(C)]
#[derive(Default)]
struct NvmlMemory {
    total: u64,
    free: u64,
    used: u64,
}

type NvmlInitFn = unsafe extern "C" fn() -> i32;
type NvmlDeviceGetHandleByIndexFn = unsafe extern "C" fn(u32, *mut *mut c_void) -> i32;
type NvmlDeviceGetUtilizationRatesFn = unsafe extern "C" fn(*mut c_void, *mut NvmlUtilization) -> i32;
type NvmlDeviceGetClockInfoFn = unsafe extern "C" fn(*mut c_void, u32, *mut u32) -> i32;
type NvmlDeviceGetPowerUsageFn = unsafe extern "C" fn(*mut c_void, *mut u32) -> i32;
type NvmlDeviceGetTemperatureFn = unsafe extern "C" fn(*mut c_void, u32, *mut u32) -> i32;
type NvmlDeviceGetMemoryInfoFn = unsafe extern "C" fn(*mut c_void, *mut NvmlMemory) -> i32;

struct Nvml {
    device: *mut c_void,
    get_util: NvmlDeviceGetUtilizationRatesFn,
    get_clock: Option<NvmlDeviceGetClockInfoFn>,
    get_power: Option<NvmlDeviceGetPowerUsageFn>,
    get_temp: Option<NvmlDeviceGetTemperatureFn>,
    get_mem: Option<NvmlDeviceGetMemoryInfoFn>,
}

// SAFETY: NVML device handles and function pointers are process-wide and thread-safe.
unsafe impl Send for Nvml {}
unsafe impl Sync for Nvml {}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GpuSample {
    pub(crate) gpu_pct: u32,
    pub(crate) mem_ctrl_pct: u32,
    pub(crate) clock_mhz: Option<u32>,
    pub(crate) power_w: Option<f32>,
    pub(crate) temp_c: Option<u32>,
    pub(crate) board_vram_used_gib: Option<f64>,
    pub(crate) board_vram_total_gib: Option<f64>,
}

fn nvml() -> Option<&'static Nvml> {
    static NVML: OnceLock<Option<Nvml>> = OnceLock::new();
    NVML.get_or_init(|| {
        // SAFETY: dynamically probing standard NVIDIA management library if present.
        unsafe {
            let mut lib = LoadLibraryA(c"nvml.dll".as_ptr().cast());
            if lib.is_null() {
                lib = LoadLibraryA(c"C:\\Windows\\System32\\nvml.dll".as_ptr().cast());
            }
            if lib.is_null() {
                return None;
            }
            macro_rules! sym {
                ($name:literal, $ty:ty) => {
                    GetProcAddress(lib, $name.as_ptr().cast())
                        .map(|p| std::mem::transmute::<_, $ty>(p))
                };
            }
            let init = sym!(c"nvmlInit_v2", NvmlInitFn)?;
            if init() != NVML_SUCCESS {
                return None;
            }
            let get_handle = sym!(c"nvmlDeviceGetHandleByIndex_v2", NvmlDeviceGetHandleByIndexFn)?;
            let mut device: *mut c_void = std::ptr::null_mut();
            if get_handle(0, &mut device) != NVML_SUCCESS || device.is_null() {
                return None;
            }
            let get_util = sym!(c"nvmlDeviceGetUtilizationRates", NvmlDeviceGetUtilizationRatesFn)?;
            Some(Nvml {
                device,
                get_util,
                get_clock: sym!(c"nvmlDeviceGetClockInfo", NvmlDeviceGetClockInfoFn),
                get_power: sym!(c"nvmlDeviceGetPowerUsage", NvmlDeviceGetPowerUsageFn),
                get_temp: sym!(c"nvmlDeviceGetTemperature", NvmlDeviceGetTemperatureFn),
                get_mem: sym!(c"nvmlDeviceGetMemoryInfo", NvmlDeviceGetMemoryInfoFn),
            })
        }
    })
    .as_ref()
}

fn sample_nvml() -> Option<GpuSample> {
    let n = nvml()?;
    // SAFETY: `n.device` is a valid NVML device handle initialized once.
    unsafe {
        let mut util = NvmlUtilization::default();
        if (n.get_util)(n.device, &mut util) != NVML_SUCCESS {
            return None;
        }
        let clock_mhz = n.get_clock.and_then(|f| {
            let mut v = 0u32;
            (f(n.device, NVML_CLOCK_GRAPHICS, &mut v) == NVML_SUCCESS).then_some(v)
        });
        let power_w = n.get_power.and_then(|f| {
            let mut mw = 0u32;
            (f(n.device, &mut mw) == NVML_SUCCESS).then_some(mw as f32 / 1000.0)
        });
        let temp_c = n.get_temp.and_then(|f| {
            let mut t = 0u32;
            (f(n.device, NVML_TEMPERATURE_GPU, &mut t) == NVML_SUCCESS).then_some(t)
        });
        let (board_vram_used_gib, board_vram_total_gib) = n
            .get_mem
            .and_then(|f| {
                let mut m = NvmlMemory::default();
                (f(n.device, &mut m) == NVML_SUCCESS)
                    .then_some((Some(m.used as f64 / GIB), Some(m.total as f64 / GIB)))
            })
            .unwrap_or((None, None));
        Some(GpuSample {
            gpu_pct: util.gpu,
            mem_ctrl_pct: util.memory,
            clock_mhz,
            power_w,
            temp_c,
            board_vram_used_gib,
            board_vram_total_gib,
        })
    }
}

fn ft_u64(ft: FILETIME) -> u64 {
    (u64::from(ft.dwHighDateTime) << 32) | u64::from(ft.dwLowDateTime)
}

#[derive(Clone, Copy)]
struct CpuPrev {
    at: Instant,
    sys_idle: u64,
    sys_kernel: u64,
    sys_user: u64,
    proc_time: u64,
}

fn read_cpu_raw() -> Option<CpuPrev> {
    let mut idle = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let mut kernel = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let mut user = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let mut c = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let mut e = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let mut pk = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let mut pu = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    // SAFETY: plain Win32 query calls into stack-allocated FILETIME structs.
    unsafe {
        if GetSystemTimes(&mut idle, &mut kernel, &mut user) == 0 {
            return None;
        }
        if GetProcessTimes(GetCurrentProcess(), &mut c, &mut e, &mut pk, &mut pu) == 0 {
            return None;
        }
    }
    Some(CpuPrev {
        at: Instant::now(),
        sys_idle: ft_u64(idle),
        sys_kernel: ft_u64(kernel),
        sys_user: ft_u64(user),
        proc_time: ft_u64(pk).saturating_add(ft_u64(pu)),
    })
}

fn sample_cpu() -> Option<(f64, f64, f64, usize)> {
    static PREV: Mutex<Option<CpuPrev>> = Mutex::new(None);
    let now = read_cpu_raw()?;
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut guard = PREV.lock().unwrap_or_else(|e| e.into_inner());
    let Some(prev) = guard.replace(now) else {
        return None;
    };
    let d_idle = now.sys_idle.saturating_sub(prev.sys_idle);
    let d_kernel = now.sys_kernel.saturating_sub(prev.sys_kernel);
    let d_user = now.sys_user.saturating_sub(prev.sys_user);
    // Win32 GetSystemTimes kernel time includes idle time.
    let sys_total = d_kernel.saturating_add(d_user);
    let sys_busy = sys_total.saturating_sub(d_idle);
    let sys_pct = if sys_total > 0 {
        (sys_busy as f64 / sys_total as f64) * 100.0
    } else {
        0.0
    };
    let wall_100ns = now.at.duration_since(prev.at).as_secs_f64() * 1e7;
    let d_proc = now.proc_time.saturating_sub(prev.proc_time) as f64;
    let busy_threads = if wall_100ns > 0.0 {
        (d_proc / wall_100ns).clamp(0.0, cores as f64)
    } else {
        0.0
    };
    let proc_pct = (busy_threads / cores as f64) * 100.0;
    Some((sys_pct, proc_pct, busy_threads, cores))
}

fn sample_ram() -> Option<(f64, f64, f64, f64, f64)> {
    // SAFETY: zero-initialized Win32 structs with exact byte size headers.
    let mut mem: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    mem.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
    if unsafe { GlobalMemoryStatusEx(&mut mem) } == 0 {
        return None;
    }
    let mut pmc: PROCESS_MEMORY_COUNTERS_EX = unsafe { std::mem::zeroed() };
    pmc.cb = size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
    let proc_ws = if unsafe {
        K32GetProcessMemoryInfo(
            GetCurrentProcess(),
            (&raw mut pmc).cast(),
            size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32,
        )
    } != 0
    {
        pmc.WorkingSetSize as f64 / GIB
    } else {
        0.0
    };
    let total = mem.ullTotalPhys as f64 / GIB;
    let avail = mem.ullAvailPhys as f64 / GIB;
    let used = (total - avail).max(0.0);
    let commit_avail = mem.ullAvailPageFile as f64 / GIB;
    Some((used, total, avail, proc_ws, commit_avail))
}

/// Formats a compact one-line hardware resource utilization summary.
pub(crate) fn format_line(vram_groups: Option<[[u64; 4]; 2]>) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(4);
    if let Some((sys_pct, proc_pct, busy_t, cores)) = sample_cpu() {
        parts.push(format!(
            "CPU sys {sys_pct:.1}% proc {proc_pct:.1}% ({busy_t:.1}/{cores}T)"
        ));
    } else {
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        parts.push(format!("CPU priming ({cores}T)"));
    }
    if let Some((used, total, avail, proc_ws, commit_avail)) = sample_ram() {
        parts.push(format!(
            "RAM {used:.1}/{total:.1} GiB (free {avail:.1}, proc {proc_ws:.1}, commit_free {commit_avail:.1})"
        ));
    }
    if let Some(g) = sample_nvml() {
        let mut s = format!("GPU {}% mem {}%", g.gpu_pct, g.mem_ctrl_pct);
        if let Some(clk) = g.clock_mhz {
            s.push_str(&format!(" {clk}MHz"));
        }
        if let Some(w) = g.power_w {
            s.push_str(&format!(" {w:.0}W"));
        }
        if let Some(t) = g.temp_c {
            s.push_str(&format!(" {t}C"));
        }
        if let (Some(used), Some(total)) = (g.board_vram_used_gib, g.board_vram_total_gib) {
            s.push_str(&format!(" boardVRAM {used:.1}/{total:.1}GiB"));
        }
        parts.push(s);
    }
    if let Some([own, shared]) = vram_groups {
        parts.push(format!(
            "VRAM proc {:.1}/{:.1} GiB (shared {:.1}/{:.1})",
            own[1] as f64 / GIB,
            own[0] as f64 / GIB,
            shared[1] as f64 / GIB,
            shared[0] as f64 / GIB,
        ));
    }
    parts.join(" | ")
}

/// Logs the periodic hardware utilization line (CPU, RAM, GPU, VRAM).
pub(crate) fn log_resources(vram_groups: Option<[[u64; 4]; 2]>) {
    let line = format_line(vram_groups);
    tracing::info!("resources: {line}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_system_and_gpu_resources() {
        // Prime CPU delta baseline, then sample after a short slice.
        let _ = format_line(None);
        std::thread::sleep(std::time::Duration::from_millis(60));
        let line = format_line(None);
        eprintln!("telemetry sample: {line}");
        assert!(line.contains("CPU sys "), "missing CPU sample: {line}");
        assert!(line.contains("RAM "), "missing RAM sample: {line}");
    }
}
