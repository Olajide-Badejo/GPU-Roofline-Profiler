// A single-line, rank-style progress bar for the benchmark driver (spec
// Section 9). It overwrites one line with a carriage return when stderr is a
// TTY, and falls back to plain periodic lines when output is redirected to a
// file or a CI log, so the log stays readable in both places.
//
// It renders like:
//   [#######...] 68% | gemm_tiled 4096 | elapsed 00:07:42 | eta 00:03:20
//
// The driver constructs one Progress over the total number of (kernel, config)
// cells, calls update() as each cell finishes with a short label, and calls
// finish() at the end. ETA is a straight-line projection from mean time per
// completed cell, which is honest enough for a rank display and avoids
// pretending to model per-kernel cost differences.

#ifndef ROOFLINE_PROGRESS_HPP
#define ROOFLINE_PROGRESS_HPP

#include <chrono>
#include <cstdio>
#include <string>

#ifdef _WIN32
#include <io.h>
#define ROOFLINE_ISATTY _isatty
#define ROOFLINE_FILENO _fileno
#else
#include <unistd.h>
#define ROOFLINE_ISATTY isatty
#define ROOFLINE_FILENO fileno
#endif

namespace roofline {

class Progress {
   public:
    explicit Progress(int total_cells)
        : total_(total_cells > 0 ? total_cells : 1),
          done_(0),
          start_(std::chrono::steady_clock::now()),
          is_tty_(ROOFLINE_ISATTY(ROOFLINE_FILENO(stderr)) != 0) {}

    // Announce the plan up front so a reader knows the scale before the first
    // cell finishes.
    void announce() const {
        std::fprintf(stderr, "planned cells: %d\n", total_);
    }

    // Call once per completed cell. label is a short description of the cell
    // just finished, for example "gemm_tiled 4096".
    void update(const std::string& label) {
        if (done_ < total_) {
            ++done_;
        }
        const double fraction = static_cast<double>(done_) / total_;
        const double elapsed = elapsed_seconds();
        const double eta =
            done_ > 0 ? elapsed * (total_ - done_) / done_ : 0.0;

        char bar[kBarWidth + 1];
        const int filled = static_cast<int>(fraction * kBarWidth);
        for (int i = 0; i < kBarWidth; ++i) {
            bar[i] = i < filled ? '#' : '.';
        }
        bar[kBarWidth] = '\0';

        const char terminator = is_tty_ ? '\r' : '\n';
        std::fprintf(stderr,
                     "[%s] %3d%% | %-24s | elapsed %s | eta %s%c",
                     bar, static_cast<int>(fraction * 100.0), label.c_str(),
                     format_hms(elapsed).c_str(), format_hms(eta).c_str(),
                     terminator);
        std::fflush(stderr);
    }

    void finish() const {
        std::fprintf(stderr, "\ndone: %d cells in %s\n", done_,
                     format_hms(elapsed_seconds()).c_str());
    }

   private:
    static constexpr int kBarWidth = 20;

    double elapsed_seconds() const {
        const auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - start_).count();
    }

    static std::string format_hms(double seconds) {
        if (seconds < 0.0) {
            seconds = 0.0;
        }
        const int total = static_cast<int>(seconds + 0.5);
        const int h = total / 3600;
        const int m = (total % 3600) / 60;
        const int s = total % 60;
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%02d:%02d:%02d", h, m, s);
        return std::string(buf);
    }

    int total_;
    int done_;
    std::chrono::steady_clock::time_point start_;
    bool is_tty_;
};

}  // namespace roofline

#endif  // ROOFLINE_PROGRESS_HPP
