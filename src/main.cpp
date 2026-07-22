// Benchmark driver.
//
// Reads the sweep from YAML, resolves it into a list of (kernel, config) cells,
// times each one through the CUDA event harness, and writes a timing CSV plus a
// manifest describing exactly how the numbers were produced. An NVML monitor
// samples power and clocks for the whole run.
//
// Two habits worth calling out. Every allocation is preceded by a free memory
// check, because this GPU is also driving the display and an over-large cell
// should be skipped with a clear line rather than take the desktop down. And
// every output file is written to a temporary path and renamed, so an
// interrupted run can never leave a half written CSV that later looks valid.

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "config.hpp"
#include "cuda_check.hpp"
#include "device_peaks.hpp"
#include "kernels.hpp"
#include "nvml_monitor.hpp"
#include "nvtx_range.hpp"
#include "progress.hpp"
#include "timing.hpp"

namespace fs = std::filesystem;

namespace {

struct Row {
    std::string kernel;
    std::string variant;
    long long problem_size = 0;
    int m = 0, n = 0, k = 0;
    int block_size = 0;
    int tile_dim = 0;
    double mean_ms = 0.0;
    double median_ms = 0.0;
    double stddev_ms = 0.0;
    double achieved_gflops = 0.0;
    double achieved_gbps = 0.0;
    double flops = 0.0;
    double theoretical_bytes = 0.0;
    std::string timestamp;
};

struct Options {
    std::string config_path = "configs/sweep.yaml";
    std::string out_dir;
    bool force = false;
};

Options parse_args(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            opts.config_path = argv[++i];
        } else if (arg == "--out" && i + 1 < argc) {
            opts.out_dir = argv[++i];
        } else if (arg == "--force") {
            opts.force = true;
        } else if (arg == "--help") {
            std::printf(
                "usage: roofline_profiler [--config PATH] [--out DIR] "
                "[--force]\n");
            std::exit(0);
        }
    }
    return opts;
}

// True when the requested bytes fit while leaving the configured headroom for
// the display.
bool fits_in_vram(size_t bytes, int headroom_mib) {
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t headroom = static_cast<size_t>(headroom_mib) * 1024u * 1024u;
    return bytes + headroom < free_bytes;
}

void write_csv_atomic(const fs::path& path, const std::vector<Row>& rows) {
    const fs::path tmp = path.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::out | std::ios::trunc);
        out << "kernel,variant,problem_size,m,n,k,block_size,tile_dim,mean_ms,"
               "median_ms,stddev_ms,achieved_gflops,achieved_gbps,flops,"
               "theoretical_bytes,timestamp\n";
        out.setf(std::ios::fixed);
        for (const auto& r : rows) {
            out.precision(6);
            out << r.kernel << ',' << r.variant << ',' << r.problem_size << ','
                << r.m << ',' << r.n << ',' << r.k << ',' << r.block_size << ','
                << r.tile_dim << ',' << r.mean_ms << ',' << r.median_ms << ','
                << r.stddev_ms << ',' << r.achieved_gflops << ','
                << r.achieved_gbps << ',';
            out.precision(1);
            out << r.flops << ',' << r.theoretical_bytes << ',' << r.timestamp
                << '\n';
        }
    }
    fs::rename(tmp, path);
}

void write_peaks_csv(const fs::path& path, const roofline::Peaks& peaks) {
    const fs::path tmp = path.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::out | std::ios::trunc);
        out << "label,peak_flops,peak_bandwidth\n";
        out.setf(std::ios::fixed);
        out.precision(1);
        out << "theoretical," << peaks.theoretical_flops << ','
            << peaks.theoretical_bandwidth << '\n';
        out << "measured," << peaks.measured_flops << ','
            << peaks.measured_bandwidth << '\n';
    }
    fs::rename(tmp, path);
}

// Escape a string for embedding in JSON. Windows paths are the reason this
// exists: "configs\sweep.yaml" written raw contains \s, which is not a legal
// JSON escape, and the whole manifest fails to parse. My rule that
// results without a manifest do not exist makes an unparseable manifest as bad
// as a missing one, so this is not a cosmetic fix.
std::string json_escape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (const char c : value) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

void write_manifest(const fs::path& path, const roofline::DeviceInfo& info,
                    const roofline::Peaks& peaks, const Options& opts,
                    int total_cells) {
    int driver_version = 0, runtime_version = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));

    const fs::path tmp = path.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::out | std::ios::trunc);
        out << "{\n";
        out << "  \"timestamp_utc\": \"" << roofline::utc_timestamp_now()
            << "\",\n";
        out << "  \"config_path\": \"" << json_escape(opts.config_path)
            << "\",\n";
        out << "  \"total_cells\": " << total_cells << ",\n";
        out << "  \"device\": {\n";
        out << "    \"name\": \"" << json_escape(info.name) << "\",\n";
        out << "    \"compute_capability\": \"" << info.compute_major << '.'
            << info.compute_minor << "\",\n";
        out << "    \"sm_count\": " << info.sm_count << ",\n";
        out << "    \"fp32_lanes_per_sm\": " << info.fp32_lanes_per_sm << ",\n";
        out << "    \"sm_clock_hz\": " << info.sm_clock_hz << ",\n";
        out << "    \"memory_clock_hz\": " << info.memory_clock_hz << ",\n";
        out << "    \"memory_bus_width_bits\": " << info.memory_bus_width_bits
            << ",\n";
        out << "    \"total_global_mem\": " << info.total_global_mem << ",\n";
        out << "    \"l2_cache_bytes\": " << info.l2_cache_bytes << "\n";
        out << "  },\n";
        out << "  \"peaks\": {\n";
        out << "    \"theoretical_flops\": " << peaks.theoretical_flops << ",\n";
        out << "    \"theoretical_bandwidth\": " << peaks.theoretical_bandwidth
            << ",\n";
        out << "    \"measured_flops\": " << peaks.measured_flops << ",\n";
        out << "    \"measured_bandwidth\": " << peaks.measured_bandwidth
            << "\n";
        out << "  },\n";
        out << "  \"cuda\": {\n";
        out << "    \"driver_version\": " << driver_version << ",\n";
        out << "    \"runtime_version\": " << runtime_version << "\n";
        out << "  }\n";
        out << "}\n";
    }
    fs::rename(tmp, path);
}

// Count the cells up front so the progress bar can show a real total and a
// projected wall clock rather than counting up blindly.
int count_cells(const roofline::SweepConfig& cfg) {
    int total = 0;
    total += static_cast<int>(cfg.saxpy_sizes.size() *
                              cfg.saxpy_block_sizes.size());
    total += static_cast<int>(cfg.reduction_sizes.size() *
                              cfg.reduction_block_sizes.size());
    total += static_cast<int>(cfg.transpose_sizes.size() *
                              cfg.transpose_tile_dims.size() * 2);
    total += static_cast<int>(cfg.gemv_sizes.size() *
                              cfg.gemv_block_sizes.size());
    total += static_cast<int>(cfg.gemm_sizes.size() * cfg.gemm_variants.size());
    return total;
}

}  // namespace

int main(int argc, char** argv) {
    const Options opts = parse_args(argc, argv);

    const roofline::SweepConfig cfg = roofline::load_sweep_config(opts.config_path);
    const roofline::DeviceInfo info = roofline::query_device_info();

    std::printf("device: %s (sm_%d%d), %d SMs, %.3f GHz\n", info.name.c_str(),
                info.compute_major, info.compute_minor, info.sm_count,
                info.sm_clock_hz / 1.0e9);

    fs::path out_dir = opts.out_dir.empty()
                           ? fs::path("results") / "raw" / "latest"
                           : fs::path(opts.out_dir);
    fs::create_directories(out_dir);

    std::printf("measuring ceilings ...\n");
    const roofline::Peaks peaks = roofline::measure_all_peaks(info);
    std::printf("  theoretical: %.2f TFLOP/s, %.1f GB/s\n",
                peaks.theoretical_flops / 1e12,
                peaks.theoretical_bandwidth / 1e9);
    std::printf("  measured   : %.2f TFLOP/s, %.1f GB/s\n",
                peaks.measured_flops / 1e12, peaks.measured_bandwidth / 1e9);
    write_peaks_csv(out_dir / "peaks.csv", peaks);

    roofline::NvmlMonitor monitor((out_dir / "nvml.csv").string(), 100);
    monitor.start();
    if (!monitor.available()) {
        std::printf("NVML unavailable; running without a power trace\n");
    }

    const int total_cells = count_cells(cfg);
    roofline::Progress progress(total_cells);
    progress.announce();

    std::vector<Row> rows;
    rows.reserve(total_cells);

    const auto& h = cfg.harness;
    const int warmup = h.warmup_iterations;
    const int batches = h.timed_batches;
    const int per_batch = h.launches_per_batch;

    // ---------------------------------------------------------------- SAXPY
    for (int n : cfg.saxpy_sizes) {
        for (int block : cfg.saxpy_block_sizes) {
            const size_t bytes = static_cast<size_t>(n) * sizeof(float) * 2;
            if (!fits_in_vram(bytes, h.vram_headroom_mib)) {
                std::printf("\nskip saxpy n=%d: does not fit in free VRAM\n", n);
                progress.update("saxpy skipped");
                continue;
            }
            float *x = nullptr, *y = nullptr;
            CUDA_CHECK(cudaMalloc(&x, static_cast<size_t>(n) * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&y, static_cast<size_t>(n) * sizeof(float)));
            CUDA_CHECK(cudaMemset(x, 0, static_cast<size_t>(n) * sizeof(float)));
            CUDA_CHECK(cudaMemset(y, 0, static_cast<size_t>(n) * sizeof(float)));

            const roofline::NvtxRange nvtx_cell(
                "saxpy n=" + std::to_string(n) + " block=" +
                std::to_string(block));

            const auto t = roofline::time_kernel_auto(
                [&] { roofline::launch_saxpy(2.0f, x, y, n, block); }, warmup,
                batches, per_batch);

            Row r;
            r.kernel = "saxpy";
            r.variant = "base";
            r.problem_size = n;
            r.block_size = block;
            r.flops = 2.0 * n;
            r.theoretical_bytes = 3.0 * n * sizeof(float);
            r.mean_ms = t.mean_ms;
            r.median_ms = t.median_ms;
            r.stddev_ms = t.stddev_ms;
            r.achieved_gflops = roofline::achieved_flops(r.flops, t.mean_ms) / 1e9;
            r.achieved_gbps =
                roofline::achieved_bandwidth(r.theoretical_bytes, t.mean_ms) / 1e9;
            r.timestamp = roofline::utc_timestamp_now();
            rows.push_back(r);

            CUDA_CHECK(cudaFree(x));
            CUDA_CHECK(cudaFree(y));
            progress.update("saxpy " + std::to_string(n));
        }
    }

    // ------------------------------------------------------------ Reduction
    for (int n : cfg.reduction_sizes) {
        for (int block : cfg.reduction_block_sizes) {
            const size_t bytes = static_cast<size_t>(n) * sizeof(float);
            if (!fits_in_vram(bytes, h.vram_headroom_mib)) {
                std::printf("\nskip reduction n=%d: does not fit\n", n);
                progress.update("reduction skipped");
                continue;
            }
            float *in = nullptr, *out = nullptr, *scratch = nullptr;
            CUDA_CHECK(cudaMalloc(&in, bytes));
            CUDA_CHECK(cudaMalloc(&out, sizeof(float)));
            CUDA_CHECK(cudaMalloc(
                &scratch,
                static_cast<size_t>(roofline::reduction_scratch_elements()) *
                    sizeof(float)));
            CUDA_CHECK(cudaMemset(in, 0, bytes));

            const roofline::NvtxRange nvtx_cell(
                "reduction n=" + std::to_string(n) + " block=" +
                std::to_string(block));

            const auto t = roofline::time_kernel_auto(
                [&] { roofline::launch_reduction(in, out, scratch, n, block); },
                warmup, batches, per_batch);

            Row r;
            r.kernel = "reduction";
            r.variant = "shared_tree";
            r.problem_size = n;
            r.block_size = block;
            r.flops = static_cast<double>(n);
            r.theoretical_bytes = static_cast<double>(n) * sizeof(float);
            r.mean_ms = t.mean_ms;
            r.median_ms = t.median_ms;
            r.stddev_ms = t.stddev_ms;
            r.achieved_gflops = roofline::achieved_flops(r.flops, t.mean_ms) / 1e9;
            r.achieved_gbps =
                roofline::achieved_bandwidth(r.theoretical_bytes, t.mean_ms) / 1e9;
            r.timestamp = roofline::utc_timestamp_now();
            rows.push_back(r);

            CUDA_CHECK(cudaFree(in));
            CUDA_CHECK(cudaFree(out));
            CUDA_CHECK(cudaFree(scratch));
            progress.update("reduction " + std::to_string(n));
        }
    }

    // ------------------------------------------------------------ Transpose
    for (int size : cfg.transpose_sizes) {
        for (int tile : cfg.transpose_tile_dims) {
            const size_t count = static_cast<size_t>(size) * size;
            const size_t bytes = count * sizeof(float) * 2;
            if (!fits_in_vram(bytes, h.vram_headroom_mib)) {
                std::printf("\nskip transpose %d: does not fit\n", size);
                progress.update("transpose skipped");
                progress.update("transpose skipped");
                continue;
            }
            float *in = nullptr, *out = nullptr;
            CUDA_CHECK(cudaMalloc(&in, count * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&out, count * sizeof(float)));
            CUDA_CHECK(cudaMemset(in, 0, count * sizeof(float)));

            struct Variant {
                const char* name;
                bool tiled;
            };
            for (const Variant v : {Variant{"naive", false},
                                    Variant{"tiled", true}}) {
                const roofline::NvtxRange nvtx_cell(
                    std::string("transpose ") + v.name + " size=" +
                    std::to_string(size) + " tile=" + std::to_string(tile));
                const auto t = roofline::time_kernel_auto(
                    [&] {
                        if (v.tiled) {
                            roofline::launch_transpose_tiled(in, out, size, size,
                                                             tile);
                        } else {
                            roofline::launch_transpose_naive(in, out, size, size,
                                                             tile);
                        }
                    },
                    warmup, batches, per_batch);

                Row r;
                r.kernel = "transpose";
                r.variant = v.name;
                r.problem_size = size;
                r.m = size;
                r.n = size;
                r.tile_dim = tile;
                // A transpose does no floating point work at all; its story is
                // entirely traffic, so only the bandwidth column is meaningful.
                r.flops = 0.0;
                r.theoretical_bytes =
                    2.0 * static_cast<double>(count) * sizeof(float);
                r.mean_ms = t.mean_ms;
                r.median_ms = t.median_ms;
                r.stddev_ms = t.stddev_ms;
                r.achieved_gflops = 0.0;
                r.achieved_gbps =
                    roofline::achieved_bandwidth(r.theoretical_bytes, t.mean_ms) /
                    1e9;
                r.timestamp = roofline::utc_timestamp_now();
                rows.push_back(r);
                progress.update(std::string("transpose ") + v.name + " " +
                                std::to_string(size));
            }

            CUDA_CHECK(cudaFree(in));
            CUDA_CHECK(cudaFree(out));
        }
    }

    // ----------------------------------------------------------------- GEMV
    for (const auto& shape : cfg.gemv_sizes) {
        for (int block : cfg.gemv_block_sizes) {
            const size_t a_count =
                static_cast<size_t>(shape.m) * shape.n;
            const size_t bytes =
                (a_count + shape.m + shape.n) * sizeof(float);
            if (!fits_in_vram(bytes, h.vram_headroom_mib)) {
                std::printf("\nskip gemv %dx%d: does not fit\n", shape.m,
                            shape.n);
                progress.update("gemv skipped");
                continue;
            }
            float *a = nullptr, *x = nullptr, *y = nullptr;
            CUDA_CHECK(cudaMalloc(&a, a_count * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&x, static_cast<size_t>(shape.n) * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&y, static_cast<size_t>(shape.m) * sizeof(float)));
            CUDA_CHECK(cudaMemset(a, 0, a_count * sizeof(float)));
            CUDA_CHECK(cudaMemset(x, 0, static_cast<size_t>(shape.n) * sizeof(float)));

            const roofline::NvtxRange nvtx_cell(
                "gemv " + std::to_string(shape.m) + "x" +
                std::to_string(shape.n) + " block=" + std::to_string(block));

            const auto t = roofline::time_kernel_auto(
                [&] {
                    roofline::launch_gemv(a, x, y, shape.m, shape.n, block);
                },
                warmup, batches, per_batch);

            Row r;
            r.kernel = "gemv";
            r.variant = "base";
            r.problem_size = static_cast<long long>(shape.m) * shape.n;
            r.m = shape.m;
            r.n = shape.n;
            r.block_size = block;
            r.flops = 2.0 * shape.m * shape.n;
            r.theoretical_bytes =
                static_cast<double>(a_count + shape.m + shape.n) * sizeof(float);
            r.mean_ms = t.mean_ms;
            r.median_ms = t.median_ms;
            r.stddev_ms = t.stddev_ms;
            r.achieved_gflops = roofline::achieved_flops(r.flops, t.mean_ms) / 1e9;
            r.achieved_gbps =
                roofline::achieved_bandwidth(r.theoretical_bytes, t.mean_ms) / 1e9;
            r.timestamp = roofline::utc_timestamp_now();
            rows.push_back(r);

            CUDA_CHECK(cudaFree(a));
            CUDA_CHECK(cudaFree(x));
            CUDA_CHECK(cudaFree(y));
            progress.update("gemv " + std::to_string(shape.m));
        }
    }

    // ----------------------------------------------------------------- GEMM
    roofline::gemm_cublas_init();
    for (int size : cfg.gemm_sizes) {
        const size_t count = static_cast<size_t>(size) * size;
        const size_t bytes = count * sizeof(float) * 3;
        if (!fits_in_vram(bytes, h.vram_headroom_mib)) {
            std::printf("\nskip gemm %d: does not fit\n", size);
            for (size_t i = 0; i < cfg.gemm_variants.size(); ++i) {
                progress.update("gemm skipped");
            }
            continue;
        }
        float *a = nullptr, *b = nullptr, *c = nullptr;
        CUDA_CHECK(cudaMalloc(&a, count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&b, count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c, count * sizeof(float)));
        CUDA_CHECK(cudaMemset(a, 0, count * sizeof(float)));
        CUDA_CHECK(cudaMemset(b, 0, count * sizeof(float)));

        for (const std::string& variant : cfg.gemm_variants) {
            const int tile = cfg.gemm_tile_dim;
            bool ran = true;
            const roofline::NvtxRange nvtx_cell("gemm " + variant + " size=" +
                                                std::to_string(size));
            const auto t = roofline::time_kernel_auto(
                [&] {
                    if (variant == "naive") {
                        roofline::launch_gemm_naive(a, b, c, size, size, size,
                                                    tile);
                    } else if (variant == "tiled") {
                        roofline::launch_gemm_tiled(a, b, c, size, size, size,
                                                    tile);
                    } else if (variant == "register_blocked") {
                        roofline::launch_gemm_register_blocked(a, b, c, size,
                                                               size, size);
                    } else if (variant == "vectorized") {
                        ran = roofline::launch_gemm_vectorized(a, b, c, size,
                                                               size, size);
                    } else if (variant == "cublas") {
                        roofline::launch_gemm_cublas(a, b, c, size, size, size);
                    }
                },
                warmup, batches, per_batch);

            if (!ran) {
                std::printf("\nskip gemm vectorized %d: shape not aligned\n",
                            size);
                progress.update("gemm skipped");
                continue;
            }

            Row r;
            r.kernel = "gemm";
            r.variant = variant;
            r.problem_size = size;
            r.m = r.n = r.k = size;
            r.tile_dim = tile;
            r.flops = 2.0 * static_cast<double>(size) * size * size;
            r.theoretical_bytes = 3.0 * static_cast<double>(count) * sizeof(float);
            r.mean_ms = t.mean_ms;
            r.median_ms = t.median_ms;
            r.stddev_ms = t.stddev_ms;
            r.achieved_gflops = roofline::achieved_flops(r.flops, t.mean_ms) / 1e9;
            r.achieved_gbps =
                roofline::achieved_bandwidth(r.theoretical_bytes, t.mean_ms) / 1e9;
            r.timestamp = roofline::utc_timestamp_now();
            rows.push_back(r);
            progress.update("gemm " + variant + " " + std::to_string(size));
        }

        CUDA_CHECK(cudaFree(a));
        CUDA_CHECK(cudaFree(b));
        CUDA_CHECK(cudaFree(c));
    }
    roofline::gemm_cublas_destroy();

    progress.finish();
    monitor.stop();

    write_csv_atomic(out_dir / "timing.csv", rows);
    write_manifest(out_dir / "manifest.json", info, peaks, opts, total_cells);

    std::printf("wrote %zu rows to %s\n", rows.size(),
                (out_dir / "timing.csv").string().c_str());
    std::printf("NVML samples: %d\n", monitor.samples_written());
    return 0;
}
