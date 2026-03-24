# AOT ID: ['13_forward']
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


# kernel path: /traces/inductor_cache/ha/chaqcwrk3gz44jpbwj47gjgryjdgu55xkmai73lveyd3jwc7hzrq.py
# Topologically Sorted Source Nodes: [non_scalar_components, embedding, outputs_mv], Original ATen: [aten.zeros, aten.cat, aten.add]
# Source node to ATen node mapping:
#   embedding => cat
#   non_scalar_components => full_default
#   outputs_mv => add
# Graph fragment:
#   %primals_2 : Tensor "f32[500, 128, 1, 16][2048, 16, 16, 1]cuda:0" = PlaceHolder[target=primals_2]
#   %primals_1 : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=primals_1]
#   %full_default : Tensor "f32[1, 15][15, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 15], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %cat : Tensor "f32[1, 16][16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%primals_1, %full_default], -1), kwargs = {})
#   %add : Tensor "f32[500, 128, 1, 16][2048, 16, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_2, %cat), kwargs = {})
#   return %add
triton_poi_fused_add_cat_zeros_0 = async_compile.triton('triton_poi_fused_add_cat_zeros_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_zeros_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 12288000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_zeros_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = x0
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp8 = tl.where(tmp5, tmp7, 0.0)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1], 16, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = 0.0
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tl.where(tmp5, tmp8, tmp14)
    tmp16 = tmp0 + tmp15
    tl.store(out_ptr0 + (x2), tmp16, None)
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
        primals_1, primals_2 = args
        args.clear()
        assert_size_stride(primals_1, (1, 1), (1, 1))
        assert_size_stride(primals_2, (500, 128, 1, 16), (2048, 16, 16, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((500, 128, 1, 16), (2048, 16, 16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [non_scalar_components, embedding, outputs_mv], Original ATen: [aten.zeros, aten.cat, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_cat_zeros_0:1
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_cat_zeros_0.run(primals_2, primals_1, buf0, 1024000, stream=stream0)
            del primals_1
            del primals_2
        return (buf0, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((500, 128, 1, 16), (2048, 16, 16, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
