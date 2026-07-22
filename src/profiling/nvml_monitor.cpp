// NVML monitor implementation.

#include "nvml_monitor.hpp"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <nvml.h>

namespace roofline {
namespace {

// NVML failures here are never fatal. Losing the power trace costs a diagnostic,
// not a result, so the monitor degrades to silence and the benchmark continues.
bool nvml_ok(nvmlReturn_t status, const char* what) {
    if (status != NVML_SUCCESS) {
        std::fprintf(stderr, "NVML: %s failed (%s); continuing without it\n",
                     what, nvmlErrorString(status));
        return false;
    }
    return true;
}

}  // namespace

std::string utc_timestamp_now() {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto ms =
        duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    const std::time_t t = system_clock::to_time_t(now);

    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif

    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S") << '.' << std::setfill('0')
        << std::setw(3) << ms.count() << 'Z';
    return out.str();
}

NvmlMonitor::NvmlMonitor(std::string csv_path, int sample_interval_ms,
                         int device_index)
    : csv_path_(std::move(csv_path)),
      sample_interval_ms_(sample_interval_ms),
      device_index_(device_index) {
    available_ = nvml_ok(nvmlInit_v2(), "nvmlInit_v2");
}

NvmlMonitor::~NvmlMonitor() {
    stop();
    if (available_) {
        nvmlShutdown();
    }
}

void NvmlMonitor::start() {
    if (!available_ || running_.load()) {
        return;
    }
    running_.store(true);
    thread_ = std::thread(&NvmlMonitor::run, this);
}

void NvmlMonitor::stop() {
    if (!running_.exchange(false)) {
        return;
    }
    if (thread_.joinable()) {
        thread_.join();
    }
}

void NvmlMonitor::run() {
    nvmlDevice_t device{};
    if (!nvml_ok(nvmlDeviceGetHandleByIndex_v2(device_index_, &device),
                 "nvmlDeviceGetHandleByIndex_v2")) {
        return;
    }

    std::ofstream out(csv_path_, std::ios::out | std::ios::trunc);
    if (!out) {
        std::fprintf(stderr, "NVML: cannot open %s for writing\n",
                     csv_path_.c_str());
        return;
    }
    out << "timestamp,power_w,temperature_c,sm_clock_mhz,memory_clock_mhz,"
           "gpu_util_pct\n";

    while (running_.load()) {
        unsigned int milliwatts = 0;
        unsigned int temperature = 0;
        unsigned int sm_clock = 0;
        unsigned int mem_clock = 0;
        nvmlUtilization_t utilization{};

        // Each field is optional: a card or driver that will not report one
        // should not cost the whole sample.
        const bool have_power =
            nvmlDeviceGetPowerUsage(device, &milliwatts) == NVML_SUCCESS;
        const bool have_temp =
            nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU,
                                     &temperature) == NVML_SUCCESS;
        const bool have_sm =
            nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &sm_clock) ==
            NVML_SUCCESS;
        const bool have_mem =
            nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &mem_clock) ==
            NVML_SUCCESS;
        const bool have_util =
            nvmlDeviceGetUtilizationRates(device, &utilization) == NVML_SUCCESS;

        out << utc_timestamp_now() << ','
            << (have_power ? milliwatts / 1000.0 : 0.0) << ','
            << (have_temp ? static_cast<double>(temperature) : 0.0) << ','
            << (have_sm ? sm_clock : 0u) << ',' << (have_mem ? mem_clock : 0u)
            << ',' << (have_util ? utilization.gpu : 0u) << '\n';
        out.flush();
        samples_written_.fetch_add(1);

        std::this_thread::sleep_for(
            std::chrono::milliseconds(sample_interval_ms_));
    }
}

}  // namespace roofline
