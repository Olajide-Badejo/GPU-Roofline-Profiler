// Sweep configuration loaded from configs/sweep.yaml.
//
// Every sweep parameter lives in the YAML file, never in source, so changing
// what runs never means editing and rebuilding a kernel. The resolved
// configuration is copied verbatim into each run's manifest, so a result can
// always be traced back to the exact sweep that produced it.

#ifndef ROOFLINE_CONFIG_HPP
#define ROOFLINE_CONFIG_HPP

#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace roofline {

struct HarnessConfig {
    int warmup_iterations = 10;
    int timed_batches = 20;
    int launches_per_batch = 50;
    int vram_headroom_mib = 1536;
};

struct MatrixShape {
    int m = 0;
    int n = 0;
};

struct SweepConfig {
    HarnessConfig harness;

    std::vector<int> saxpy_sizes;
    std::vector<int> saxpy_block_sizes;

    std::vector<int> reduction_sizes;
    std::vector<int> reduction_block_sizes;

    std::vector<int> transpose_sizes;
    std::vector<int> transpose_tile_dims;

    std::vector<MatrixShape> gemv_sizes;
    std::vector<int> gemv_block_sizes;

    std::vector<int> gemm_sizes;
    std::vector<std::string> gemm_variants;
    int gemm_tile_dim = 32;
};

namespace detail {

inline std::vector<int> int_list(const YAML::Node& node) {
    std::vector<int> out;
    if (node && node.IsSequence()) {
        for (const auto& item : node) {
            out.push_back(item.as<int>());
        }
    }
    return out;
}

inline std::vector<std::string> string_list(const YAML::Node& node) {
    std::vector<std::string> out;
    if (node && node.IsSequence()) {
        for (const auto& item : node) {
            out.push_back(item.as<std::string>());
        }
    }
    return out;
}

}  // namespace detail

inline SweepConfig load_sweep_config(const std::string& path) {
    const YAML::Node root = YAML::LoadFile(path);
    SweepConfig cfg;

    if (const auto h = root["harness"]) {
        cfg.harness.warmup_iterations =
            h["warmup_iterations"].as<int>(cfg.harness.warmup_iterations);
        cfg.harness.timed_batches =
            h["timed_batches"].as<int>(cfg.harness.timed_batches);
        cfg.harness.launches_per_batch =
            h["launches_per_batch"].as<int>(cfg.harness.launches_per_batch);
        cfg.harness.vram_headroom_mib =
            h["vram_headroom_mib"].as<int>(cfg.harness.vram_headroom_mib);
    }

    cfg.saxpy_sizes = detail::int_list(root["saxpy"]["sizes"]);
    cfg.saxpy_block_sizes = detail::int_list(root["saxpy"]["block_sizes"]);

    cfg.reduction_sizes = detail::int_list(root["reduction"]["sizes"]);
    cfg.reduction_block_sizes =
        detail::int_list(root["reduction"]["block_sizes"]);

    cfg.transpose_sizes = detail::int_list(root["transpose"]["sizes"]);
    cfg.transpose_tile_dims = detail::int_list(root["transpose"]["tile_dims"]);

    if (const auto g = root["gemv"]["sizes"]) {
        for (const auto& item : g) {
            MatrixShape shape;
            shape.m = item["m"].as<int>();
            shape.n = item["n"].as<int>();
            cfg.gemv_sizes.push_back(shape);
        }
    }
    cfg.gemv_block_sizes = detail::int_list(root["gemv"]["block_sizes"]);

    cfg.gemm_sizes = detail::int_list(root["gemm"]["sizes"]);
    cfg.gemm_variants = detail::string_list(root["gemm"]["variants"]);
    if (const auto t = root["gemm"]["tile_dim"]) {
        cfg.gemm_tile_dim = t.as<int>();
    }

    return cfg;
}

}  // namespace roofline

#endif  // ROOFLINE_CONFIG_HPP
