# AOT ID: ['18_inference']
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


# kernel path: /traces/inductor_cache/rp/crpxo4m2k4chxlloshwt7dwkbj4ew2h27kgqgr7xyqubmefpmgru.py
# Topologically Sorted Source Nodes: [idx, setitem_1], Original ATen: [aten.arange, aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   idx => iota
#   setitem_1 => full_default, index_put_1
# Graph fragment:
#   %buf0 : Tensor  = PlaceHolder[target=buf0]
#   %iota : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %full_default : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put_1 : Tensor "f32[500, 128, 128, 4][65536, 512, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%index_put, [None, %iota, %iota], %full_default), kwargs = {})
#   return %buf1
triton_poi_fused_arange_index_put_lift_fresh_0 = async_compile.triton('triton_poi_fused_arange_index_put_lift_fresh_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_index_put_lift_fresh_0', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2048000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_index_put_lift_fresh_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 128)
    x2 = xindex // 512
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + 516*x1 + 65536*x2), tmp0, xmask)
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
        arg0_1, arg1_1, arg2_1, arg3_1 = args
        args.clear()
        s38 = arg2_1
        assert_size_stride(arg0_1, (500, 128, 128, 4), (65536, 512, 4, 1))
        assert_size_stride(arg1_1, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(arg3_1, (s38, 4), (4, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            aten.index_put_(arg0_1, [arg1_1], arg3_1, False)
            del arg1_1
            del arg3_1
            # Topologically Sorted Source Nodes: [idx, setitem_1], Original ATen: [aten.arange, aten.lift_fresh, aten.index_put]
            # [Provenance debug handles] triton_poi_fused_arange_index_put_lift_fresh_0:1
            stream0 = get_raw_stream(0)
            triton_poi_fused_arange_index_put_lift_fresh_0.run(arg0_1, 256000, stream=stream0)
        return (arg0_1, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((500, 128, 128, 4), (65536, 512, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bool)
    arg2_1 = 893092
    arg3_1 = rand_strided((893092, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
