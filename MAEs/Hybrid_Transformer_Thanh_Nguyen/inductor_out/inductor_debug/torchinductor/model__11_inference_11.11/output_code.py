# AOT ID: ['11_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /traces/inductor_cache/vl/cvlrgkydfr65j5tq6ztja6o2z4f4ioenvk6mrxyrq67hg3gdmr4g.py
# Topologically Sorted Source Nodes: [multivector, setitem, x, _generalized_scatter], Original ATen: [aten.zeros, aten.slice, aten.view, aten.copy]
# Source node to ATen node mapping:
#   _generalized_scatter => slice_scatter_default
#   multivector => full
#   setitem => copy, slice_1
#   x => view
# Graph fragment:
#   %arg0_1 : Tensor "f32[500, 128, 4][512, 4, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %full : Tensor "f32[500, 128, 1, 16][2048, 16, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([500, 128, 1, 16], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_1 : Tensor "f32[500, 128, 1, 4][2048, 16, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%full, 3, 1, 5), kwargs = {})
#   %view : Tensor "f32[500, 128, 1, 4][512, 4, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg0_1, [500, 128, 1, 4]), kwargs = {})
#   %copy : Tensor "f32[500, 128, 1, 4][2048, 16, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_1, %view), kwargs = {})
#   %slice_scatter_default : Tensor "f32[500, 128, 1, 16][2048, 16, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full, %copy, 3, 1, 5), kwargs = {})
#   return %slice_scatter_default
triton_poi_fused_copy_slice_view_zeros_0 = async_compile.triton('triton_poi_fused_copy_slice_view_zeros_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_slice_view_zeros_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 9216000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_slice_view_zeros_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 5, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-1) + x0 + 4*x1), tmp5, other=0.0)
    tmp7 = 0.0
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, = args
        args.clear()
        assert_size_stride(arg0_1, (500, 128, 4), (512, 4, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((500, 128, 1, 16), (2048, 16, 16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multivector, setitem, x, _generalized_scatter], Original ATen: [aten.zeros, aten.slice, aten.view, aten.copy]
            # [Provenance debug handles] triton_poi_fused_copy_slice_view_zeros_0:1
            stream0 = get_raw_stream(0)
            triton_poi_fused_copy_slice_view_zeros_0.run(arg0_1, buf0, 1024000, stream=stream0)
            del arg0_1
        return (reinterpret_tensor(buf0, (500, 128, 16), (2048, 16, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((500, 128, 4), (512, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
