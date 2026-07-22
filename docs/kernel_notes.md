# Kernel notes

One short entry per kernel: what it improves over the previous rung, and what
that improvement costs. Measured numbers land here once the sweep runs; this file
is the qualitative map.

## SAXPY

- **Improves**: nothing, it is the baseline for the memory bound edge.
- **Costs**: nothing to optimize; two FLOPs per three memory accesses fixes its
  intensity near the far left of the roofline. If it does not approach the copy
  bandwidth ceiling, the harness is suspect.

## Reduction

- **Improves**: exercises shared memory and synchronization rather than pure
  streaming.
- **Costs**: tree reduction needs correct tail handling for non power of two
  sizes, which is where these kernels usually break, so it is tested there first.

## Transpose, naive

- **Improves**: baseline for the transpose pair.
- **Costs**: the transposed write has a large stride and is uncoalesced, which
  shows up as low global store efficiency.

## Transpose, tiled

- **Improves**: stages a tile in shared memory so both the global read and write
  are coalesced; the padded tile (`[TILE_DIM][TILE_DIM+1]`) removes shared memory
  bank conflicts on the transposed access.
- **Costs**: one extra pad column of shared memory per tile, and a `__syncthreads`.
  Cheap for what it buys; this is the cleanest optimization in the suite.

## GEMV

- **Improves**: introduces matrix reuse over the vector, sitting mid axis.
- **Costs**: each matrix element is still read once with low reuse, so it stays
  well below the compute ceiling.

## GEMM, naive

- **Improves**: correct baseline for the GEMM ladder.
- **Costs**: every operand is reloaded from global memory many times; memory bound
  with low intensity.

## GEMM, shared memory tiled

- **Improves**: stages tiles of A and B in shared memory, cutting DRAM traffic and
  raising intensity toward the theoretical value.
- **Costs**: shared memory capacity and a synchronization per tile step.

## GEMM, register blocked

- **Improves**: each thread computes a small output tile held in registers,
  raising reuse per shared memory load. Usually the largest single jump in
  achieved performance.
- **Costs**: register pressure, which can lower occupancy. Whether the reuse gain
  beats the occupancy cost is a counter backed question, not an assumption.

## GEMM, vectorized

- **Improves**: `float4` loads and stores, so each memory instruction moves four
  values and issue pressure drops.
- **Costs**: alignment requirements on the leading dimension.

## cuBLAS SGEMM

- **Improves**: the vendor reference ceiling for a custom kernel to be measured
  against.
- **Costs**: it is a library, not hand written code, and is labeled as such so the
  comparison stays honest.
