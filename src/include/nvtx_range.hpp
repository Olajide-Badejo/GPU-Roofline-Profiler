// NVTX range helper, so an Nsight Systems timeline reads as labeled work rather
// than as a wall of anonymous kernel launches.
//
// The driver pushes a range per (kernel, configuration) cell. Without these the
// timeline shows thousands of identical launches and the only way to tell which
// configuration is which is by counting, which is not a way to read a profile.
//
// NVTX3 is header only, so this costs nothing to link. When NVTX is unavailable
// the class degrades to doing nothing rather than failing the build, because a
// missing profiler annotation should never stop a benchmark from running.

#ifndef ROOFLINE_NVTX_RANGE_HPP
#define ROOFLINE_NVTX_RANGE_HPP

#include <string>

#if defined(ROOFLINE_HAS_NVTX)
// On Windows nvToolsExt.h pulls in windows.h, which defines min and max as
// macros and then quietly breaks every std::min and std::max downstream. NOMINMAX
// stops that; WIN32_LEAN_AND_MEAN keeps the rest of the Windows surface out.
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif
#include <nvtx3/nvToolsExt.h>
#endif

namespace roofline {

class NvtxRange {
   public:
    explicit NvtxRange(const std::string& name) {
#if defined(ROOFLINE_HAS_NVTX)
        nvtxRangePushA(name.c_str());
        active_ = true;
#else
        (void)name;
#endif
    }

    ~NvtxRange() { pop(); }

    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;

    void pop() {
#if defined(ROOFLINE_HAS_NVTX)
        if (active_) {
            nvtxRangePop();
            active_ = false;
        }
#endif
    }

   private:
#if defined(ROOFLINE_HAS_NVTX)
    bool active_ = false;
#endif
};

}  // namespace roofline

#endif  // ROOFLINE_NVTX_RANGE_HPP
