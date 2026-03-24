# AOT ID: ['9_inference']
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


# kernel path: /traces/inductor_cache/je/cjeg74tb2rmijq43qu6upg2z5etjnfnr4n2zrsa2efomfeyafryl.py
# Topologically Sorted Source Nodes: [U_vals], Original ATen: [aten.stack]
# Source node to ATen node mapping:
#   U_vals => cat, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3
# Graph fragment:
#   %arg0_1 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %unsqueeze : Tensor "f32[500, 128, 128, 1][16384, 128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg0_1, 3), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[500, 128, 128, 1][16384, 128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg1_1, 3), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[500, 128, 128, 1][16384, 128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg2_1, 3), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[500, 128, 128, 1][16384, 128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg3_1, 3), kwargs = {})
#   %cat : Tensor "f32[500, 128, 128, 4][65536, 512, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze, %unsqueeze_1, %unsqueeze_2, %unsqueeze_3], -1), kwargs = {})
#   return %cat
triton_poi_fused_stack_0 = async_compile.triton('triton_poi_fused_stack_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 262144000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_stack_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 2, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x1), tmp9, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 3, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x1), tmp14, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 4, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr3 + (x1), tmp16, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.where(tmp14, tmp15, tmp19)
    tmp21 = tl.where(tmp9, tmp10, tmp20)
    tmp22 = tl.where(tmp4, tmp5, tmp21)
    tl.store(out_ptr0 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/kh/ckhgjix5gejuryfi5sszrietieeb3pmikglud37dposryqb3omgu.py
# Topologically Sorted Source Nodes: [unsqueeze, unsqueeze_1, valid_pairs], Original ATen: [aten.unsqueeze, aten.bitwise_and]
# Source node to ATen node mapping:
#   unsqueeze => unsqueeze_4
#   unsqueeze_1 => unsqueeze_5
#   valid_pairs => bitwise_and
# Graph fragment:
#   %arg4_1 : Tensor "b8[500, 128][128, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %unsqueeze_4 : Tensor "b8[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, 2), kwargs = {})
#   %unsqueeze_5 : Tensor "b8[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, 1), kwargs = {})
#   %bitwise_and : Tensor "b8[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%unsqueeze_4, %unsqueeze_5), kwargs = {})
#   return %bitwise_and
triton_poi_fused_bitwise_and_unsqueeze_1 = async_compile.triton('triton_poi_fused_bitwise_and_unsqueeze_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'out_ptr0': '*i1', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_unsqueeze_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 16448000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bitwise_and_unsqueeze_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex // 128
    x0 = (xindex % 128)
    x2 = xindex // 16384
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr0 + (x0 + 128*x2), None, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tmp0 & tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/nj/cnje5iriw4atxukcv5kittfnuddiiyytb6em5xvcwacgyeasyam3.py
# Topologically Sorted Source Nodes: [U], Original ATen: [aten.full_like]
# Source node to ATen node mapping:
#   U => full_default
# Graph fragment:
#   %full_default : Tensor "f32[500, 128, 128, 4][65536, 512, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([500, 128, 128, 4], -1000000000.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   return %full_default
triton_poi_fused_full_like_2 = async_compile.triton('triton_poi_fused_full_like_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_full_like_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 262144000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_full_like_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = -1000000000.0
    tl.store(out_ptr0 + (x0), tmp0, None)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
        args.clear()
        assert_size_stride(arg0_1, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(arg1_1, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(arg2_1, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(arg3_1, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(arg4_1, (500, 128), (128, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((500, 128, 128, 4), (65536, 512, 4, 1), torch.float32)
            # Topologically Sorted Source Nodes: [U_vals], Original ATen: [aten.stack]
            # [Provenance debug handles] triton_poi_fused_stack_0:1
            stream0 = get_raw_stream(0)
            triton_poi_fused_stack_0.run(arg0_1, arg1_1, arg2_1, arg3_1, buf0, 32768000, stream=stream0)
            del arg0_1
            del arg1_1
            del arg2_1
            del arg3_1
            buf1 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.bool)
            # Topologically Sorted Source Nodes: [unsqueeze, unsqueeze_1, valid_pairs], Original ATen: [aten.unsqueeze, aten.bitwise_and]
            # [Provenance debug handles] triton_poi_fused_bitwise_and_unsqueeze_1:2
            stream0 = get_raw_stream(0)
            triton_poi_fused_bitwise_and_unsqueeze_1.run(arg4_1, buf1, 8192000, stream=stream0)
            del arg4_1
            buf2 = empty_strided_cuda((500, 128, 128, 4), (65536, 512, 4, 1), torch.float32)
            # Topologically Sorted Source Nodes: [U], Original ATen: [aten.full_like]
            # [Provenance debug handles] triton_poi_fused_full_like_2:3
            stream0 = get_raw_stream(0)
            triton_poi_fused_full_like_2.run(buf2, 32768000, stream=stream0)
        return (buf0, buf1, buf2, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((500, 128), (128, 1), device='cuda:0', dtype=torch.bool)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
