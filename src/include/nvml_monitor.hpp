// Background NVML sampler.
//
// Runs for the whole duration of a benchmark run on its own host thread, writing
// timestamped power, temperature, clock, and utilization samples to a CSV. The
// analysis joins those samples to kernel timings by timestamp.
//
// This is what tells me whether a result was shaped by thermals or clock drift
// rather than by kernel design, which matters a great deal on a boost clocked
// consumer card that is also driving a display. Without it, an unexplained slow
// run is a mystery; with it, the clock trace usually answers the question.
//
// Timestamps are ISO 8601 in UTC, which is what the Python loader expects.

#ifndef ROOFLINE_NVML_MONITOR_HPP
#define ROOFLINE_NVML_MONITOR_HPP

#include <atomic>
#include <string>
#include <thread>

namespace roofline {

class NvmlMonitor {
   public:
    // sample_interval_ms is the polling period. A short interval gives a usable
    // trace without measurably loading the host.
    NvmlMonitor(std::string csv_path, int sample_interval_ms = 100,
                int device_index = 0);
    ~NvmlMonitor();

    NvmlMonitor(const NvmlMonitor&) = delete;
    NvmlMonitor& operator=(const NvmlMonitor&) = delete;

    // Both are safe to call more than once.
    void start();
    void stop();

    // False when NVML could not be initialized. The benchmark still runs; it
    // just runs without a power and clock trace, and says so.
    bool available() const { return available_; }

    int samples_written() const { return samples_written_.load(); }

   private:
    void run();

    std::string csv_path_;
    int sample_interval_ms_;
    int device_index_;
    bool available_ = false;
    std::atomic<bool> running_{false};
    std::atomic<int> samples_written_{0};
    std::thread thread_;
};

// UTC ISO 8601 with milliseconds, matching what the Python loader parses.
std::string utc_timestamp_now();

}  // namespace roofline

#endif  // ROOFLINE_NVML_MONITOR_HPP
