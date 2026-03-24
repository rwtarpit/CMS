# AOT ID: ['17_backward']
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


# kernel path: /traces/inductor_cache/lt/cltk4bnr4qbw4w2snwa7hroajsxajus7ppuut6dkeuzfqdrxmitk.py
# Topologically Sorted Source Nodes: [sum_2], Original ATen: [aten.sum]
# Source node to ATen node mapping:
#   sum_2 => sum_2
# Graph fragment:
#   %tangents_1 : Tensor "f32[500, 10][10, 1]cuda:0" = PlaceHolder[target=tangents_1]
#   %sum_2 : Tensor "f32[1, 10][10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%tangents_1, [0], True), kwargs = {})
#   return %buf3
triton_red_fused_sum_0 = async_compile.triton('triton_red_fused_sum_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 20320, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_sum_0(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 40
    r0_numel = 125
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 10)
    x1 = xindex // 10
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 10*r0_2 + 1250*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/qe/cqeado2j7e3334biw2wv2mmsbluucnpepdcq76yifggr3cezmgj5.py
# Topologically Sorted Source Nodes: [sum_2], Original ATen: [aten.sum]
# Source node to ATen node mapping:
#   sum_2 => sum_2
# Graph fragment:
#   %buf3 : Tensor "f32[1, 10, 4][40, 1, 10]cuda:0" = PlaceHolder[target=buf3]
#   %sum_2 : Tensor "f32[1, 10][10, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%tangents_1, [0], True), kwargs = {})
#   return %sum_2
triton_per_fused_sum_1 = async_compile.triton('triton_per_fused_sum_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r0_': 4},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 240, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_sum_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 10
    r0_numel = 4
    R0_BLOCK: tl.constexpr = 4
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 10*r0_1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/5z/c5zwugeithlmz3j5lepeokkrllomkupk5hggesabzwgz5mmfq5h2.py
# Topologically Sorted Source Nodes: [permute_19, ], Original ATen: [aten.t, aten.mm]
# Source node to ATen node mapping:
#    => constant_pad_nd_default_1
#   permute_19 => permute_19
# Graph fragment:
#   %tangents_1 : Tensor "f32[500, 10][10, 1]cuda:0" = PlaceHolder[target=tangents_1]
#   %permute_19 : Tensor "f32[10, 500][1, 10]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%tangents_1, [1, 0]), kwargs = {})
#   %constant_pad_nd_default_1 : Tensor "f32[12, 500][500, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%permute_19, [0, 0, 0, 2]), kwargs = {})
#   return %constant_pad_nd_default_1
triton_poi_fused_mm_t_2 = async_compile.triton('triton_poi_fused_mm_t_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_t_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 68000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_t_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 12)
    x1 = xindex // 12
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 10, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + 10*x1), tmp2 & xmask, other=0.0)
    tl.store(out_ptr0 + (x2), tmp3, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/su/csu5yb6ykh7m7exzvakqbrgx5qgfwd3b2g4r3uoymvdy33kj5s6x.py
# Topologically Sorted Source Nodes: [unsqueeze_1, mul_20, sum_5, sum_6], Original ATen: [aten.unsqueeze, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   mul_20 => mul_20
#   sum_5 => sum_5
#   sum_6 => sum_6
#   unsqueeze_1 => unsqueeze_1
# Graph fragment:
#   %mm_1 : Tensor "f32[500, 128][128, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %mul_12 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0" = PlaceHolder[target=mul_12]
#   %unsqueeze_1 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_1, 1), kwargs = {})
#   %mul_20 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, %mul_12), kwargs = {})
#   %sum_5 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_20, [0, 1]), kwargs = {})
#   %sum_6 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%unsqueeze_1, [0, 1]), kwargs = {})
#   return %buf8,%buf10
triton_red_fused_native_layer_norm_backward_unsqueeze_3 = async_compile.triton('triton_red_fused_native_layer_norm_backward_unsqueeze_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_unsqueeze_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 2, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 520192, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_native_layer_norm_backward_unsqueeze_3(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 125
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 128)
    x1 = xindex // 128
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 16000*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 128*r0_2 + 16000*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
        tmp6 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tl.store(out_ptr1 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/op/copdwuf3mhzg77r6gdykqjlwzodiog3my63663k33z5qbkfcr3gq.py
# Topologically Sorted Source Nodes: [unsqueeze_1, mul_20, sum_5], Original ATen: [aten.unsqueeze, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   mul_20 => mul_20
#   sum_5 => sum_5
#   unsqueeze_1 => unsqueeze_1
# Graph fragment:
#   %buf8 : Tensor "f32[128, 4][1, 128]cuda:0" = PlaceHolder[target=buf8]
#   %unsqueeze_1 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_1, 1), kwargs = {})
#   %mul_20 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, %mul_12), kwargs = {})
#   %sum_5 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_20, [0, 1]), kwargs = {})
#   return %sum_5
triton_per_fused_native_layer_norm_backward_unsqueeze_4 = async_compile.triton('triton_per_fused_native_layer_norm_backward_unsqueeze_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 4},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_unsqueeze_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 3072, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_layer_norm_backward_unsqueeze_4(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 4
    R0_BLOCK: tl.constexpr = 4
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/fq/cfqs42pnygslvcmp53sdudz7taayalzoshyh5ufidvrmijo6pjn2.py
# Topologically Sorted Source Nodes: [unsqueeze_1, mul_15, mul_16, sum_3, mul_17, sum_4, mul_18, sub_7, sub_8, mul_19], Original ATen: [aten.unsqueeze, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   mul_15 => mul_15
#   mul_16 => mul_16
#   mul_17 => mul_17
#   mul_18 => mul_18
#   mul_19 => mul_19
#   sub_7 => sub_7
#   sub_8 => sub_8
#   sum_3 => sum_3
#   sum_4 => sum_4
#   unsqueeze_1 => unsqueeze_1
# Graph fragment:
#   %mm_1 : Tensor "f32[500, 128][128, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %primals_23 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_23]
#   %mul_12 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0" = PlaceHolder[target=mul_12]
#   %div_1 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=div_1]
#   %sum_3 : Tensor "f32[500, 1, 1][1, 500, 500]cuda:0" = PlaceHolder[target=sum_3]
#   %sum_4 : Tensor "f32[500, 1, 1][1, 500, 500]cuda:0" = PlaceHolder[target=sum_4]
#   %unsqueeze_1 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_1, 1), kwargs = {})
#   %mul_15 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, %primals_23), kwargs = {})
#   %mul_16 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_15, 128), kwargs = {})
#   %sum_3 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_15, [2], True), kwargs = {})
#   %mul_17 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_15, %mul_12), kwargs = {})
#   %sum_4 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_17, [2], True), kwargs = {})
#   %mul_18 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_12, %sum_4), kwargs = {})
#   %sub_7 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_16, %sum_3), kwargs = {})
#   %sub_8 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_7, %mul_18), kwargs = {})
#   %mul_19 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %sub_8), kwargs = {})
#   return %sum_3,%sum_4,%mul_19
triton_per_fused_native_layer_norm_backward_unsqueeze_5 = async_compile.triton('triton_per_fused_native_layer_norm_backward_unsqueeze_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_unsqueeze_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 2, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2000, 'r0_': 1024512}}
)
@triton.jit
def triton_per_fused_native_layer_norm_backward_unsqueeze_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 500
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 128*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r0_1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (r0_1 + 128*x0), xmask, other=0.0)
    tmp13 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
    tmp14 = 128.0
    tmp15 = tmp2 * tmp14
    tmp16 = tmp15 - tmp6
    tmp17 = tmp7 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tl.store(in_out_ptr0 + (r0_1 + 128*x0), tmp19, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/du/cdummcpusvowgf5sofq5r3yu2yksern2qfahidnubjbua5si7i5o.py
# Topologically Sorted Source Nodes: [view_21, sum_7], Original ATen: [aten.view, aten.sum]
# Source node to ATen node mapping:
#   sum_7 => sum_7
#   view_21 => view_21
# Graph fragment:
#   %mul_19 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0" = PlaceHolder[target=mul_19]
#   %view_21 : Tensor "f32[500, 128][128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_19, [500, 128]), kwargs = {})
#   %sum_7 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_21, [0], True), kwargs = {})
#   return %buf14
triton_red_fused_sum_view_6 = async_compile.triton('triton_red_fused_sum_view_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_view_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 260096, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_sum_view_6(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 125
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 128)
    x1 = xindex // 128
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 16000*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/ts/ctsc323w7xyntizbzfumtn7ipkuklxvogzoroqk4dczd52572c4t.py
# Topologically Sorted Source Nodes: [view_23, linear_1, gelu, layer_norm_3, mul_27, sum_10, sum_11], Original ATen: [aten.view, aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   gelu => add_9, erf, mul_7, mul_8, mul_9
#   layer_norm_3 => mul_10, sub_4
#   linear_1 => view_17
#   mul_27 => mul_27
#   sum_10 => sum_10
#   sum_11 => sum_11
#   view_23 => view_23
# Graph fragment:
#   %mm_3 : Tensor "f32[500, 512][512, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %addmm_3 : Tensor "f32[500, 512][512, 1]cuda:0" = PlaceHolder[target=addmm_3]
#   %getitem_11 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=getitem_11]
#   %rsqrt_3 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_3]
#   %view_23 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [500, 1, 512]), kwargs = {})
#   %view_17 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_3, [500, 1, 512]), kwargs = {})
#   %mul_7 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_17, 0.5), kwargs = {})
#   %mul_8 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_17, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_8,), kwargs = {})
#   %add_9 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_9 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %add_9), kwargs = {})
#   %sub_4 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_9, %getitem_11), kwargs = {})
#   %mul_10 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_3), kwargs = {})
#   %mul_27 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_23, %mul_10), kwargs = {})
#   %sum_10 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_27, [0, 1]), kwargs = {})
#   %sum_11 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_23, [0, 1]), kwargs = {})
#   return %buf18,%buf20
triton_red_fused_gelu_native_layer_norm_native_layer_norm_backward_view_7 = async_compile.triton('triton_red_fused_gelu_native_layer_norm_native_layer_norm_backward_view_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_gelu_native_layer_norm_native_layer_norm_backward_view_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 2, 'num_reduction': 2, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2080768, 'r0_': 4000}}
)
@triton.jit
def triton_red_fused_gelu_native_layer_norm_native_layer_norm_backward_view_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 125
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 512)
    x1 = xindex // 512
    _tmp16 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp19 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_2 + 64000*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 512*r0_2 + 64000*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr2 + (r0_2 + 125*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr3 + (r0_2 + 125*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.5
        tmp3 = tmp1 * tmp2
        tmp4 = 0.7071067811865476
        tmp5 = tmp1 * tmp4
        tmp6 = libdevice.erf(tmp5)
        tmp7 = 1.0
        tmp8 = tmp6 + tmp7
        tmp9 = tmp3 * tmp8
        tmp11 = tmp9 - tmp10
        tmp13 = tmp11 * tmp12
        tmp14 = tmp0 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(r0_mask & xmask, tmp17, _tmp16)
        tmp18 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(r0_mask & xmask, tmp20, _tmp19)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp16, xmask)
    tl.store(out_ptr1 + (x3), tmp19, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/54/c54mjphfntavqtdk7bmp7glczvhdil43u754mroebgf3y2gde5k5.py
# Topologically Sorted Source Nodes: [view_23, linear_1, gelu, layer_norm_3, mul_27, sum_10], Original ATen: [aten.view, aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   gelu => add_9, erf, mul_7, mul_8, mul_9
#   layer_norm_3 => mul_10, sub_4
#   linear_1 => view_17
#   mul_27 => mul_27
#   sum_10 => sum_10
#   view_23 => view_23
# Graph fragment:
#   %buf18 : Tensor "f32[512, 4][1, 512]cuda:0" = PlaceHolder[target=buf18]
#   %view_23 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [500, 1, 512]), kwargs = {})
#   %view_17 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_3, [500, 1, 512]), kwargs = {})
#   %mul_7 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_17, 0.5), kwargs = {})
#   %mul_8 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_17, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_8,), kwargs = {})
#   %add_9 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_9 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %add_9), kwargs = {})
#   %sub_4 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_9, %getitem_11), kwargs = {})
#   %mul_10 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_3), kwargs = {})
#   %mul_27 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_23, %mul_10), kwargs = {})
#   %sum_10 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_27, [0, 1]), kwargs = {})
#   return %sum_10
triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_8 = async_compile.triton('triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 4},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 12288, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_8(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 4
    R0_BLOCK: tl.constexpr = 4
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/zh/czhrf3nblpfzpqdf5d2x7kehwwlnqusploecydp72hccespkuzke.py
# Topologically Sorted Source Nodes: [view_23, mul_22, mul_23, sum_8, linear_1, gelu, layer_norm_3, mul_24, sum_9, mul_25, sub_10, sub_11, div_2, mul_26, mul_29, mul_30, mul_31, exp_1, mul_32, mul_33, add_16, mul_34], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.gelu, aten.native_layer_norm, aten.gelu_backward]
# Source node to ATen node mapping:
#   add_16 => add_16
#   div_2 => div_2
#   exp_1 => exp_1
#   gelu => add_9, erf, mul_7, mul_8, mul_9
#   layer_norm_3 => mul_10, sub_4
#   linear_1 => view_17
#   mul_22 => mul_22
#   mul_23 => mul_23
#   mul_24 => mul_24
#   mul_25 => mul_25
#   mul_26 => mul_26
#   mul_29 => mul_29
#   mul_30 => mul_30
#   mul_31 => mul_31
#   mul_32 => mul_32
#   mul_33 => mul_33
#   mul_34 => mul_34
#   sub_10 => sub_10
#   sub_11 => sub_11
#   sum_8 => sum_8
#   sum_9 => sum_9
#   view_23 => view_23
# Graph fragment:
#   %mm_3 : Tensor "f32[500, 512][512, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %primals_19 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_19]
#   %addmm_3 : Tensor "f32[500, 512][512, 1]cuda:0" = PlaceHolder[target=addmm_3]
#   %getitem_11 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=getitem_11]
#   %rsqrt_3 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_3]
#   %sum_8 : Tensor "f32[500, 1, 1][1, 500, 500]cuda:0" = PlaceHolder[target=sum_8]
#   %sum_9 : Tensor "f32[500, 1, 1][1, 500, 500]cuda:0" = PlaceHolder[target=sum_9]
#   %view_23 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [500, 1, 512]), kwargs = {})
#   %mul_22 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_23, %primals_19), kwargs = {})
#   %mul_23 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, 512), kwargs = {})
#   %sum_8 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_22, [2], True), kwargs = {})
#   %view_17 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_3, [500, 1, 512]), kwargs = {})
#   %mul_7 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_17, 0.5), kwargs = {})
#   %mul_8 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_17, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_8,), kwargs = {})
#   %add_9 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_9 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %add_9), kwargs = {})
#   %sub_4 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_9, %getitem_11), kwargs = {})
#   %mul_10 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_3), kwargs = {})
#   %mul_24 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %mul_10), kwargs = {})
#   %sum_9 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_24, [2], True), kwargs = {})
#   %mul_25 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %sum_9), kwargs = {})
#   %sub_10 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_23, %sum_8), kwargs = {})
#   %sub_11 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_10, %mul_25), kwargs = {})
#   %div_2 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_3, 512), kwargs = {})
#   %mul_26 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %sub_11), kwargs = {})
#   %mul_29 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_9, 0.5), kwargs = {})
#   %mul_30 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_17, %view_17), kwargs = {})
#   %mul_31 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_30, -0.5), kwargs = {})
#   %exp_1 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_31,), kwargs = {})
#   %mul_32 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_1, 0.3989422804014327), kwargs = {})
#   %mul_33 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_17, %mul_32), kwargs = {})
#   %add_16 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, %mul_33), kwargs = {})
#   %mul_34 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_26, %add_16), kwargs = {})
#   return %sum_8,%sum_9,%mul_34
triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_view_9 = async_compile.triton('triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_view_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_view_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 2, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4000, 'r0_': 4098048}}
)
@triton.jit
def triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_view_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 500
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r0_1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (r0_1 + 512*x0), xmask, other=0.0)
    tmp16 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tmp8 = 0.5
    tmp9 = tmp7 * tmp8
    tmp10 = 0.7071067811865476
    tmp11 = tmp7 * tmp10
    tmp12 = libdevice.erf(tmp11)
    tmp13 = 1.0
    tmp14 = tmp12 + tmp13
    tmp15 = tmp9 * tmp14
    tmp17 = tmp15 - tmp16
    tmp19 = tmp17 * tmp18
    tmp20 = tmp2 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
    tmp23 = tl.where(xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None].to(tl.float32)
    tmp25 = 0.001953125
    tmp26 = tmp18 * tmp25
    tmp27 = 512.0
    tmp28 = tmp2 * tmp27
    tmp29 = tmp28 - tmp6
    tmp30 = tmp19 * tmp24
    tmp31 = tmp29 - tmp30
    tmp32 = tmp26 * tmp31
    tmp33 = tmp14 * tmp8
    tmp34 = tmp7 * tmp7
    tmp35 = -0.5
    tmp36 = tmp34 * tmp35
    tmp37 = libdevice.exp(tmp36)
    tmp38 = 0.3989422804014327
    tmp39 = tmp37 * tmp38
    tmp40 = tmp7 * tmp39
    tmp41 = tmp33 + tmp40
    tmp42 = tmp32 * tmp41
    tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp42, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/na/cnaqwwqfjyfonm557stkrq7yn6hebl2tmfy7flfgapw6acpkgu2j.py
# Topologically Sorted Source Nodes: [view_24, sum_12], Original ATen: [aten.view, aten.sum]
# Source node to ATen node mapping:
#   sum_12 => sum_12
#   view_24 => view_24
# Graph fragment:
#   %mul_34 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0" = PlaceHolder[target=mul_34]
#   %view_24 : Tensor "f32[500, 512][512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_34, [500, 512]), kwargs = {})
#   %sum_12 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_24, [0], True), kwargs = {})
#   return %buf25
triton_red_fused_sum_view_10 = async_compile.triton('triton_red_fused_sum_view_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_view_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1040384, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_sum_view_10(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 125
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 512)
    x1 = xindex // 512
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_2 + 64000*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/4j/c4j6g2opyrnmpl7m7lvvzprv5urcwicrjm7zp73dcrcd4y6x47bn.py
# Topologically Sorted Source Nodes: [view_26, mul_36, mul_37, sum_13, mul_38, sum_14, mul_39, sub_13, sub_14, mul_40, add_17, mul_43, mul_44, sum_17, multi_head_attention_forward, transpose_2, layer_norm_1, mul_45, sum_18, mul_46, sub_16, sub_17, div_4, mul_47], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add, aten.transpose, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_17 => add_17
#   div_4 => div_4
#   layer_norm_1 => mul_3, sub_2
#   mul_36 => mul_36
#   mul_37 => mul_37
#   mul_38 => mul_38
#   mul_39 => mul_39
#   mul_40 => mul_40
#   mul_43 => mul_43
#   mul_44 => mul_44
#   mul_45 => mul_45
#   mul_46 => mul_46
#   mul_47 => mul_47
#   multi_head_attention_forward => view_14
#   sub_13 => sub_13
#   sub_14 => sub_14
#   sub_16 => sub_16
#   sub_17 => sub_17
#   sum_13 => sum_13
#   sum_14 => sum_14
#   sum_17 => sum_17
#   sum_18 => sum_18
#   transpose_2 => permute_14
#   view_26 => view_26
# Graph fragment:
#   %mm_5 : Tensor "f32[500, 128][128, 1]cuda:0" = PlaceHolder[target=mm_5]
#   %primals_15 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_15]
#   %mul_5 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0" = PlaceHolder[target=mul_5]
#   %mul_19 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0" = PlaceHolder[target=mul_19]
#   %div_3 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=div_3]
#   %sum_13 : Tensor "f32[500, 1, 1][1, 500, 500]cuda:0" = PlaceHolder[target=sum_13]
#   %sum_14 : Tensor "f32[500, 1, 1][1, 500, 500]cuda:0" = PlaceHolder[target=sum_14]
#   %add_17 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0" = PlaceHolder[target=add_17]
#   %primals_13 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_13]
#   %addmm_2 : Tensor "f32[500, 128][128, 1]cuda:0" = PlaceHolder[target=addmm_2]
#   %getitem_7 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=getitem_7]
#   %rsqrt_1 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_1]
#   %sum_17 : Tensor "f32[500, 1, 1][1, 500, 500]cuda:0" = PlaceHolder[target=sum_17]
#   %sum_18 : Tensor "f32[500, 1, 1][1, 500, 500]cuda:0" = PlaceHolder[target=sum_18]
#   %view_26 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_5, [500, 1, 128]), kwargs = {})
#   %mul_36 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_26, %primals_15), kwargs = {})
#   %mul_37 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_36, 128), kwargs = {})
#   %sum_13 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_36, [2], True), kwargs = {})
#   %mul_38 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_36, %mul_5), kwargs = {})
#   %sum_14 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_38, [2], True), kwargs = {})
#   %mul_39 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %sum_14), kwargs = {})
#   %sub_13 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_37, %sum_13), kwargs = {})
#   %sub_14 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_13, %mul_39), kwargs = {})
#   %mul_40 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, %sub_14), kwargs = {})
#   %add_17 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %mul_40), kwargs = {})
#   %mul_43 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %primals_13), kwargs = {})
#   %mul_44 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, 128), kwargs = {})
#   %sum_17 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_43, [2], True), kwargs = {})
#   %view_14 : Tensor "f32[1, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_2, [1, 500, 128]), kwargs = {})
#   %permute_14 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_14, [1, 0, 2]), kwargs = {})
#   %sub_2 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_14, %getitem_7), kwargs = {})
#   %mul_3 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_45 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %mul_3), kwargs = {})
#   %sum_18 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_45, [2], True), kwargs = {})
#   %mul_46 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %sum_18), kwargs = {})
#   %sub_16 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_44, %sum_17), kwargs = {})
#   %sub_17 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_16, %mul_46), kwargs = {})
#   %div_4 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_1, 128), kwargs = {})
#   %mul_47 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_4, %sub_17), kwargs = {})
#   return %sum_13,%sum_14,%add_17,%sum_17,%sum_18,%mul_47
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_11 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 9, 'num_store': 2, 'num_reduction': 4, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 6000, 'r0_': 2049024}}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 500
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (r0_1 + 128*x0), xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r0_1 + 128*x0), xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr5 + (r0_1 + 128*x0), xmask, other=0.0)
    tmp29 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
    tmp15 = 128.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tmp23 = tmp21 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
    tmp26 = tl.where(xmask, tmp24, 0)
    tmp27 = tl.sum(tmp26, 1)[:, None].to(tl.float32)
    tmp30 = tmp28 - tmp29
    tmp32 = tmp30 * tmp31
    tmp33 = tmp23 * tmp32
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, R0_BLOCK])
    tmp36 = tl.where(xmask, tmp34, 0)
    tmp37 = tl.sum(tmp36, 1)[:, None].to(tl.float32)
    tmp38 = 0.0078125
    tmp39 = tmp31 * tmp38
    tmp40 = tmp23 * tmp15
    tmp41 = tmp40 - tmp27
    tmp42 = tmp32 * tmp37
    tmp43 = tmp41 - tmp42
    tmp44 = tmp39 * tmp43
    tl.store(in_out_ptr0 + (r0_1 + 128*x0), tmp21, xmask)
    tl.store(out_ptr4 + (r0_1 + 128*x0), tmp44, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/hy/chywugpxcjhblvaqxjsbssy63iyjoq4ylcpj2f75nrpt3vbw7fue.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward, transpose_2, layer_norm_1, mul_48, sum_19, sum_20], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_1 => mul_3, sub_2
#   mul_48 => mul_48
#   multi_head_attention_forward => view_14
#   sum_19 => sum_19
#   sum_20 => sum_20
#   transpose_2 => permute_14
# Graph fragment:
#   %add_17 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0" = PlaceHolder[target=add_17]
#   %addmm_2 : Tensor "f32[500, 128][128, 1]cuda:0" = PlaceHolder[target=addmm_2]
#   %getitem_7 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=getitem_7]
#   %rsqrt_1 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_1]
#   %view_14 : Tensor "f32[1, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_2, [1, 500, 128]), kwargs = {})
#   %permute_14 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_14, [1, 0, 2]), kwargs = {})
#   %sub_2 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_14, %getitem_7), kwargs = {})
#   %mul_3 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_48 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %mul_3), kwargs = {})
#   %sum_19 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_48, [0, 1]), kwargs = {})
#   %sum_20 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%add_17, [0, 1]), kwargs = {})
#   return %buf36,%buf38
triton_red_fused_native_layer_norm_native_layer_norm_backward_transpose_view_12 = async_compile.triton('triton_red_fused_native_layer_norm_native_layer_norm_backward_transpose_view_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_transpose_view_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 2, 'num_reduction': 2, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 520192, 'r0_': 4000}}
)
@triton.jit
def triton_red_fused_native_layer_norm_native_layer_norm_backward_transpose_view_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 125
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 128)
    x1 = xindex // 128
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 16000*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 128*r0_2 + 16000*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r0_2 + 125*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr3 + (r0_2 + 125*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp5 = tmp3 * tmp4
        tmp6 = tmp0 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
        tmp10 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
    tl.store(out_ptr1 + (x3), tmp11, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/ny/cnypzvsij44whj7phcf76btm4z36mtkzkaft7q5y4bckryt55zcw.py
# Topologically Sorted Source Nodes: [mul_49, sum_22, neg, fma], Original ATen: [aten._softmax_backward_data]
# Source node to ATen node mapping:
#   fma => fma
#   mul_49 => mul_49
#   neg => neg
#   sum_22 => sum_22
# Graph fragment:
#   %bmm_2 : Tensor "f32[4000, 1, 129][129, 129, 1]cuda:0" = PlaceHolder[target=bmm_2]
#   %div : Tensor "f32[4000, 1, 129][129, 129, 1]cuda:0" = PlaceHolder[target=div]
#   %sum_22 : Tensor "f32[4000, 1, 1][1, 4000, 4000]cuda:0" = PlaceHolder[target=sum_22]
#   %mul_49 : Tensor "f32[4000, 1, 129][129, 129, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_2, %div), kwargs = {})
#   %sum_22 : Tensor "f32[4000, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_49, [-1], True), kwargs = {})
#   %neg : Tensor "f32[4000, 1, 129][129, 129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%div,), kwargs = {})
#   %fma : Tensor "f32[4000, 1, 129][129, 129, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.fma.default](args = (%neg, %sum_22, %mul_49), kwargs = {})
#   return %sum_22,%fma
triton_per_fused__softmax_backward_data_13 = async_compile.triton('triton_per_fused__softmax_backward_data_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 8256000}}
)
@triton.jit
def triton_per_fused__softmax_backward_data_13(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4000
    r0_numel = 129
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 129*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r0_1 + 129*x0), r0_mask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(r0_mask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tmp7 = -tmp1
    tmp8 = libdevice.fma(tmp7, tmp6, tmp2)
    tl.store(in_out_ptr0 + (r0_1 + 129*x0), tmp8, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/7o/c7ovh6dazgydic3st6i4rhk7nbr7teel6q6idhayb62vhdt4742g.py
# Topologically Sorted Source Nodes: [mul_50, permute_43, view_32, view_37], Original ATen: [aten.mul, aten.transpose, aten.view]
# Source node to ATen node mapping:
#   mul_50 => mul_50
#   permute_43 => permute_43
#   view_32 => view_32
#   view_37 => view_37
# Graph fragment:
#   %bmm_3 : Tensor "f32[4000, 1, 16][16, 16, 1]cuda:0" = PlaceHolder[target=bmm_3]
#   %mul_50 : Tensor "f32[4000, 1, 16][16, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_3, 0.25), kwargs = {})
#   %permute_43 : Tensor "f32[1, 4000, 16][16, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_50, [1, 0, 2]), kwargs = {})
#   %view_32 : Tensor "f32[1, 500, 128][16, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_43, [1, 500, 128]), kwargs = {})
#   %view_37 : Tensor "f32[500, 128][128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%view_32, [500, 128]), kwargs = {})
#   return %view_37
triton_poi_fused_mul_transpose_view_14 = async_compile.triton('triton_poi_fused_mul_transpose_view_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_transpose_view_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 768000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_transpose_view_14(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.25
    tmp2 = tmp0 * tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/sr/csrljdx4va4l2snjqwle7k5ndhsggk2bkp4gvqzdcoprha7lpyad.py
# Topologically Sorted Source Nodes: [view_34, sum_24, view_38, cat_2], Original ATen: [aten.view, aten.sum, aten.cat]
# Source node to ATen node mapping:
#   cat_2 => cat_2
#   sum_24 => sum_24
#   view_34 => view_34
#   view_38 => view_38
# Graph fragment:
#   %buf60 : Tensor "f32[1, 128, 4][512, 1, 128]cuda:0" = PlaceHolder[target=buf60]
#   %sum_24 : Tensor "f32[1, 128][128, 1]cuda:0" = PlaceHolder[target=sum_24]
#   %view_34 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sum_23, [256]), kwargs = {})
#   %sum_24 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_37, [0], True), kwargs = {})
#   %view_38 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sum_24, [128]), kwargs = {})
#   %cat_2 : Tensor "f32[384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_38, %view_34],), kwargs = {})
#   return %sum_24,%buf62
triton_per_fused_cat_sum_view_15 = async_compile.triton('triton_per_fused_cat_sum_view_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 4},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_sum_view_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 3072, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_cat_sum_view_15(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 4
    R0_BLOCK: tl.constexpr = 4
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/qu/cquzlunrnspe2lpaw4bf3y6zttf6ggarlx2lus6456kxwlcw3wz4.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
# Source node to ATen node mapping:
#    => constant_pad_nd_default
# Graph fragment:
#   %fma : Tensor "f32[4000, 1, 129][129, 516000, 1]cuda:0" = PlaceHolder[target=fma]
#   %constant_pad_nd_default : Tensor "f32[4000, 1, 132][132, 132, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%fma, [0, 3, 0, 0, 0, 0]), kwargs = {})
#   return %constant_pad_nd_default
triton_poi_fused_bmm_16 = async_compile.triton('triton_poi_fused_bmm_16', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 6288000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bmm_16(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 528000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 132)
    x1 = xindex // 132
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 129, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + 129*x1), tmp2 & xmask, other=0.0)
    tl.store(out_ptr0 + (x2), tmp3, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/b2/cb25mmqepwxtjdjviijwmjumksf2gehoak5xfpucqglkymgqjwlw.py
# Topologically Sorted Source Nodes: [, permute_40, permute_41, clone_7, view_30, permute_42, view_31, full_1, _generalized_scatter, _generalized_scatter_1, add_18, unsqueeze_2, permute_44, squeeze_2, clone_8, view_33, sum_23], Original ATen: [aten.bmm, aten.transpose, aten.clone, aten._unsafe_view, aten.view, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.sum]
# Source node to ATen node mapping:
#    => slice_tensor
#   _generalized_scatter => select_scatter_default
#   _generalized_scatter_1 => select_scatter_default_1
#   add_18 => add_18
#   clone_7 => clone_7
#   clone_8 => clone_8
#   full_1 => full_default_1
#   permute_40 => permute_40
#   permute_41 => permute_41
#   permute_42 => permute_42
#   permute_44 => permute_44
#   squeeze_2 => squeeze_2
#   sum_23 => sum_23
#   unsqueeze_2 => unsqueeze_2
#   view_30 => view_30
#   view_31 => view_31
#   view_33 => view_33
# Graph fragment:
#   %bmm_1 : Tensor "f32[4000, 129, 16][2064, 16, 1]cuda:0" = PlaceHolder[target=bmm_1]
#   %bmm_default : Tensor "f32[4000, 16, 132][2112, 132, 1]cuda:0" = PlaceHolder[target=bmm_default]
#   %slice_tensor : Tensor "f32[4000, 16, 129][2112, 132, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%bmm_default, 2, 0, -3), kwargs = {})
#   %permute_40 : Tensor "f32[4000, 129, 16][2112, 1, 132]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%slice_tensor, [0, 2, 1]), kwargs = {})
#   %permute_41 : Tensor "f32[129, 4000, 16][16, 2064, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_1, [1, 0, 2]), kwargs = {})
#   %clone_7 : Tensor "f32[129, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_41,), kwargs = {memory_format: torch.contiguous_format})
#   %view_30 : Tensor "f32[129, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_7, [129, 500, 128]), kwargs = {})
#   %permute_42 : Tensor "f32[129, 4000, 16][1, 2112, 132]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_40, [1, 0, 2]), kwargs = {})
#   %view_31 : Tensor "f32[129, 500, 128][1, 16896, 132]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_42, [129, 500, 128]), kwargs = {})
#   %full_default_1 : Tensor "f32[2, 129, 500, 128][8256000, 64000, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([2, 129, 500, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : Tensor "f32[2, 129, 500, 128][8256000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_1, %view_30, 0, 1), kwargs = {})
#   %select_scatter_default_1 : Tensor "f32[2, 129, 500, 128][8256000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_1, %view_31, 0, 0), kwargs = {})
#   %add_18 : Tensor "f32[2, 129, 500, 128][8256000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default, %select_scatter_default_1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[2, 129, 500, 1, 128][8256000, 64000, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_18, 3), kwargs = {})
#   %permute_44 : Tensor "f32[1, 129, 500, 2, 128][128, 64000, 128, 8256000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_2, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze_2 : Tensor "f32[129, 500, 2, 128][64000, 128, 8256000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_44, 0), kwargs = {})
#   %clone_8 : Tensor "f32[129, 500, 2, 128][128000, 256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_2,), kwargs = {memory_format: torch.contiguous_format})
#   %view_33 : Tensor "f32[129, 500, 256][128000, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_8, [129, 500, 256]), kwargs = {})
#   %sum_23 : Tensor "f32[1, 1, 256][256, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_33, [0, 1], True), kwargs = {})
#   return %buf52
triton_red_fused__unsafe_view_add_bmm_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_17 = async_compile.triton('triton_red_fused__unsafe_view_add_bmm_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 256},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_view_add_bmm_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 33792000, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__unsafe_view_add_bmm_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_17(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 96000
    r0_numel = 172
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp3 = tl.load(in_ptr0 + (16*((r0_2 + 172*x1) // 500) + 16*((128*(((r0_2 + 172*x1) % 500)) + ((x0 % 128))) // 64000) + 2064*(((x0 % 128)) // 16) + 16512*(((r0_2 + 172*x1) % 500)) + ((((x0 % 128)) % 16))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr1 + (132*((x0 % 128)) + 16896*(((r0_2 + 172*x1) % 500)) + ((r0_2 + 172*x1) // 500) + ((128*(((r0_2 + 172*x1) % 500)) + ((x0 % 128))) // 64000)), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x0 // 128
        tmp1 = tl.full([1, 1], 1, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp4 = 0.0
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.full([1, 1], 0, tl.int32)
        tmp7 = tmp0 == tmp6
        tmp9 = tl.where(tmp7, tmp8, tmp4)
        tmp10 = tmp5 + tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/k7/ck7ryfsekf7nfcma7jeze6zlxaoywhvtl4jmtozkw4eycppt4mdm.py
# Topologically Sorted Source Nodes: [, permute_40, permute_41, clone_7, view_30, permute_42, view_31, full_1, _generalized_scatter, _generalized_scatter_1, add_18, unsqueeze_2, permute_44, squeeze_2, clone_8, view_33, sum_23, view_34, view_38, cat_2], Original ATen: [aten.bmm, aten.transpose, aten.clone, aten._unsafe_view, aten.view, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.sum, aten.cat]
# Source node to ATen node mapping:
#    => slice_tensor
#   _generalized_scatter => select_scatter_default
#   _generalized_scatter_1 => select_scatter_default_1
#   add_18 => add_18
#   cat_2 => cat_2
#   clone_7 => clone_7
#   clone_8 => clone_8
#   full_1 => full_default_1
#   permute_40 => permute_40
#   permute_41 => permute_41
#   permute_42 => permute_42
#   permute_44 => permute_44
#   squeeze_2 => squeeze_2
#   sum_23 => sum_23
#   unsqueeze_2 => unsqueeze_2
#   view_30 => view_30
#   view_31 => view_31
#   view_33 => view_33
#   view_34 => view_34
#   view_38 => view_38
# Graph fragment:
#   %buf52 : Tensor "f32[1, 1, 256, 375][96000, 96000, 1, 256]cuda:0" = PlaceHolder[target=buf52]
#   %sum_23 : Tensor "f32[1, 1, 256][256, 256, 1]cuda:0" = PlaceHolder[target=sum_23]
#   %slice_tensor : Tensor "f32[4000, 16, 129][2112, 132, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%bmm_default, 2, 0, -3), kwargs = {})
#   %permute_40 : Tensor "f32[4000, 129, 16][2112, 1, 132]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%slice_tensor, [0, 2, 1]), kwargs = {})
#   %permute_41 : Tensor "f32[129, 4000, 16][16, 2064, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_1, [1, 0, 2]), kwargs = {})
#   %clone_7 : Tensor "f32[129, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_41,), kwargs = {memory_format: torch.contiguous_format})
#   %view_30 : Tensor "f32[129, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_7, [129, 500, 128]), kwargs = {})
#   %permute_42 : Tensor "f32[129, 4000, 16][1, 2112, 132]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_40, [1, 0, 2]), kwargs = {})
#   %view_31 : Tensor "f32[129, 500, 128][1, 16896, 132]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_42, [129, 500, 128]), kwargs = {})
#   %full_default_1 : Tensor "f32[2, 129, 500, 128][8256000, 64000, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([2, 129, 500, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : Tensor "f32[2, 129, 500, 128][8256000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_1, %view_30, 0, 1), kwargs = {})
#   %select_scatter_default_1 : Tensor "f32[2, 129, 500, 128][8256000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_1, %view_31, 0, 0), kwargs = {})
#   %add_18 : Tensor "f32[2, 129, 500, 128][8256000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default, %select_scatter_default_1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[2, 129, 500, 1, 128][8256000, 64000, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_18, 3), kwargs = {})
#   %permute_44 : Tensor "f32[1, 129, 500, 2, 128][128, 64000, 128, 8256000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_2, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze_2 : Tensor "f32[129, 500, 2, 128][64000, 128, 8256000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_44, 0), kwargs = {})
#   %clone_8 : Tensor "f32[129, 500, 2, 128][128000, 256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_2,), kwargs = {memory_format: torch.contiguous_format})
#   %view_33 : Tensor "f32[129, 500, 256][128000, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_8, [129, 500, 256]), kwargs = {})
#   %sum_23 : Tensor "f32[1, 1, 256][256, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_33, [0, 1], True), kwargs = {})
#   %view_34 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sum_23, [256]), kwargs = {})
#   %view_38 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sum_24, [128]), kwargs = {})
#   %cat_2 : Tensor "f32[384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_38, %view_34],), kwargs = {})
#   return %sum_23,%buf63
triton_red_fused__unsafe_view_add_bmm_cat_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_18 = async_compile.triton('triton_red_fused__unsafe_view_add_bmm_cat_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 512},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_view_add_bmm_cat_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 386048, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__unsafe_view_add_bmm_cat_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_18(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 375
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/mf/cmf32yejazyarsso34p3jq5aauyyllwvhp25csh3dtuheydflplj.py
# Topologically Sorted Source Nodes: [, permute_40, permute_41, clone_7, view_30, permute_42, view_31, full_1, _generalized_scatter, _generalized_scatter_1, add_18, unsqueeze_2, permute_44, squeeze_2, clone_8], Original ATen: [aten.bmm, aten.transpose, aten.clone, aten._unsafe_view, aten.view, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze]
# Source node to ATen node mapping:
#    => slice_tensor
#   _generalized_scatter => select_scatter_default
#   _generalized_scatter_1 => select_scatter_default_1
#   add_18 => add_18
#   clone_7 => clone_7
#   clone_8 => clone_8
#   full_1 => full_default_1
#   permute_40 => permute_40
#   permute_41 => permute_41
#   permute_42 => permute_42
#   permute_44 => permute_44
#   squeeze_2 => squeeze_2
#   unsqueeze_2 => unsqueeze_2
#   view_30 => view_30
#   view_31 => view_31
# Graph fragment:
#   %bmm_1 : Tensor "f32[4000, 129, 16][2064, 16, 1]cuda:0" = PlaceHolder[target=bmm_1]
#   %bmm_default : Tensor "f32[4000, 16, 132][2112, 132, 1]cuda:0" = PlaceHolder[target=bmm_default]
#   %slice_tensor : Tensor "f32[4000, 16, 129][2112, 132, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%bmm_default, 2, 0, -3), kwargs = {})
#   %permute_40 : Tensor "f32[4000, 129, 16][2112, 1, 132]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%slice_tensor, [0, 2, 1]), kwargs = {})
#   %permute_41 : Tensor "f32[129, 4000, 16][16, 2064, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_1, [1, 0, 2]), kwargs = {})
#   %clone_7 : Tensor "f32[129, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_41,), kwargs = {memory_format: torch.contiguous_format})
#   %view_30 : Tensor "f32[129, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_7, [129, 500, 128]), kwargs = {})
#   %permute_42 : Tensor "f32[129, 4000, 16][1, 2112, 132]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_40, [1, 0, 2]), kwargs = {})
#   %view_31 : Tensor "f32[129, 500, 128][1, 16896, 132]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_42, [129, 500, 128]), kwargs = {})
#   %full_default_1 : Tensor "f32[2, 129, 500, 128][8256000, 64000, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([2, 129, 500, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : Tensor "f32[2, 129, 500, 128][8256000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_1, %view_30, 0, 1), kwargs = {})
#   %select_scatter_default_1 : Tensor "f32[2, 129, 500, 128][8256000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_1, %view_31, 0, 0), kwargs = {})
#   %add_18 : Tensor "f32[2, 129, 500, 128][8256000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default, %select_scatter_default_1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[2, 129, 500, 1, 128][8256000, 64000, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_18, 3), kwargs = {})
#   %permute_44 : Tensor "f32[1, 129, 500, 2, 128][128, 64000, 128, 8256000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_2, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze_2 : Tensor "f32[129, 500, 2, 128][64000, 128, 8256000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_44, 0), kwargs = {})
#   %clone_8 : Tensor "f32[129, 500, 2, 128][128000, 256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_2,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_8
triton_poi_fused__unsafe_view_add_bmm_clone_select_backward_squeeze_transpose_unsqueeze_view_19 = async_compile.triton('triton_poi_fused__unsafe_view_add_bmm_clone_select_backward_squeeze_transpose_unsqueeze_view_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_bmm_clone_select_backward_squeeze_transpose_unsqueeze_view_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 33024000, 'x': 165120000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_bmm_clone_select_backward_squeeze_transpose_unsqueeze_view_19(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64500
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex // 128
    x2 = (xindex % 128)
    y0 = (yindex % 129)
    y1 = yindex // 129
    x4 = xindex
    tmp3 = tl.load(in_ptr0 + (16*y0 + 16*((x2 + 128*y1) // 64000) + 2064*(x2 // 16) + 16512*y1 + ((x2 % 16))), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (y0 + 132*x2 + 16896*y1 + ((x2 + 128*y1) // 64000)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = x3
    tmp1 = tl.full([1, 1], 1, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = tl.full([1, 1], 0, tl.int32)
    tmp7 = tmp0 == tmp6
    tmp9 = tl.where(tmp7, tmp8, tmp4)
    tmp10 = tmp5 + tmp9
    tl.store(out_ptr0 + (x4 + 256*y1 + 128000*y0), tmp10, xmask & ymask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/cr/ccrekot5vr74lpvoxwahf7s74d6whqhckwnt2f5rftu5qbft5emp.py
# Topologically Sorted Source Nodes: [view_36, permute_53, mul_52, sum_25], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   mul_52 => mul_52
#   permute_53 => permute_53
#   sum_25 => sum_25
#   view_36 => view_36
# Graph fragment:
#   %mm_10 : Tensor "f32[64500, 128][128, 1]cuda:0" = PlaceHolder[target=mm_10]
#   %primals_7 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_7]
#   %view_36 : Tensor "f32[129, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_10, [129, 500, 128]), kwargs = {})
#   %permute_53 : Tensor "f32[500, 129, 128][128, 64000, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_36, [1, 0, 2]), kwargs = {})
#   %mul_52 : Tensor "f32[500, 129, 128][128, 64000, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_53, %primals_7), kwargs = {})
#   %sum_25 : Tensor "f32[500, 129, 1][129, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_52, [2], True), kwargs = {})
#   return %sum_25
triton_per_fused_native_layer_norm_backward_transpose_view_20 = async_compile.triton('triton_per_fused_native_layer_norm_backward_transpose_view_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 65536, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_transpose_view_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 516000, 'r0_': 33024512}}
)
@triton.jit
def triton_per_fused_native_layer_norm_backward_transpose_view_20(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 64500
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/tg/ctgp6qzj5yruapc7l6a7236e6zjo746ja55tpaz5r72lqk4vbrbv.py
# Topologically Sorted Source Nodes: [view_36, permute_53, mul_52, mul_53, layer_norm, mul_54, sum_26, mul_55, sub_19, sub_20, div_5, mul_56, mul_57, sum_27, sum_28], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward, aten.native_layer_norm]
# Source node to ATen node mapping:
#   div_5 => div_5
#   layer_norm => mul, sub
#   mul_52 => mul_52
#   mul_53 => mul_53
#   mul_54 => mul_54
#   mul_55 => mul_55
#   mul_56 => mul_56
#   mul_57 => mul_57
#   permute_53 => permute_53
#   sub_19 => sub_19
#   sub_20 => sub_20
#   sum_26 => sum_26
#   sum_27 => sum_27
#   sum_28 => sum_28
#   view_36 => view_36
# Graph fragment:
#   %mm_10 : Tensor "f32[64500, 128][128, 1]cuda:0" = PlaceHolder[target=mm_10]
#   %primals_7 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_7]
#   %cat_1 : Tensor "f32[500, 129, 128][16512, 128, 1]cuda:0" = PlaceHolder[target=cat_1]
#   %getitem_1 : Tensor "f32[500, 129, 1][129, 1, 1]cuda:0" = PlaceHolder[target=getitem_1]
#   %rsqrt : Tensor "f32[500, 129, 1][129, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %sum_25 : Tensor "f32[500, 129, 1][1, 500, 64512]cuda:0" = PlaceHolder[target=sum_25]
#   %sum_26 : Tensor "f32[500, 129, 1][129, 1, 64512]cuda:0" = PlaceHolder[target=sum_26]
#   %view_36 : Tensor "f32[129, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_10, [129, 500, 128]), kwargs = {})
#   %permute_53 : Tensor "f32[500, 129, 128][128, 64000, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_36, [1, 0, 2]), kwargs = {})
#   %mul_52 : Tensor "f32[500, 129, 128][128, 64000, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_53, %primals_7), kwargs = {})
#   %mul_53 : Tensor "f32[500, 129, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, 128), kwargs = {})
#   %sub : Tensor "f32[500, 129, 128][16512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_1, %getitem_1), kwargs = {})
#   %mul : Tensor "f32[500, 129, 128][16512, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_54 : Tensor "f32[500, 129, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, %mul), kwargs = {})
#   %sum_26 : Tensor "f32[500, 129, 1][129, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_54, [2], True), kwargs = {})
#   %mul_55 : Tensor "f32[500, 129, 128][16512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %sum_26), kwargs = {})
#   %sub_19 : Tensor "f32[500, 129, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_53, %sum_25), kwargs = {})
#   %sub_20 : Tensor "f32[500, 129, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_19, %mul_55), kwargs = {})
#   %div_5 : Tensor "f32[500, 129, 1][129, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt, 128), kwargs = {})
#   %mul_56 : Tensor "f32[500, 129, 128][16512, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_5, %sub_20), kwargs = {})
#   %mul_57 : Tensor "f32[500, 129, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_53, %mul), kwargs = {})
#   %sum_27 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_57, [0, 1]), kwargs = {})
#   %sum_28 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%permute_53, [0, 1]), kwargs = {})
#   return %sum_26,%mul_56
triton_per_fused_native_layer_norm_native_layer_norm_backward_transpose_view_21 = async_compile.triton('triton_per_fused_native_layer_norm_native_layer_norm_backward_transpose_view_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 65536, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'ws_ptr': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'RSPLIT_SIZE': 'constexpr', 'NUM_STAGES': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'MixOrderReductionGrid', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_transpose_view_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 7, 'num_store': 0, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'RSPLIT_SIZE': 64}
)
@triton.jit
def triton_per_fused_native_layer_norm_native_layer_norm_backward_transpose_view_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, RSPLIT_SIZE : tl.constexpr, NUM_STAGES : tl.constexpr):
    xnumel = 64500
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * RSPLIT_SIZE
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    accum0 = tl.full([R0_BLOCK], 0, tl.float32)[None, :]
    accum1 = tl.full([R0_BLOCK], 0, tl.float32)[None, :]
    split_size = min(RSPLIT_SIZE, xnumel - xoffset)
    for _ in tl.range(0, split_size, XBLOCK, num_stages=NUM_STAGES):
        xmask = xindex < xnumel
        x0 = (xindex % 129)
        x1 = xindex // 129
        x5 = xindex
        xindex += XBLOCK
        tmp0 = tl.load(in_ptr0 + (r0_2 + 128*x1 + 64000*x0), xmask, other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_2), None, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr2 + (r0_2 + 128*x5), xmask, other=0.0)
        tmp4 = tl.load(in_ptr3 + (x5), xmask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr4 + (x5), xmask, eviction_policy='evict_last')
        tmp17 = tl.load(in_ptr5 + (x1 + 500*x0), xmask, eviction_policy='evict_last')
        tmp23 = tl.load(in_ptr0 + (r0_2 + 128*x5), xmask, other=0.0)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp3 - tmp4
        tmp7 = tmp5 * tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = tl.where(xmask, tmp9, 0)
        tmp12 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
        tmp13 = 0.0078125
        tmp14 = tmp6 * tmp13
        tmp15 = 128.0
        tmp16 = tmp2 * tmp15
        tmp18 = tmp16 - tmp17
        tmp19 = tmp7 * tmp12
        tmp20 = tmp18 - tmp19
        tmp21 = tmp14 * tmp20
        tmp22 = tmp0 * tmp7
        tl.store(out_ptr1 + (r0_2 + 128*x5), tmp21, xmask)
        tmp24 = tl.sum(tmp22, 0)
        tmp25 = accum0 + tmp24
        accum0 = tmp25
        tmp26 = tl.sum(tmp23, 0)
        tmp27 = accum1 + tmp26
        accum1 = tmp27
    tl.store(ws_ptr + (tl.program_id(0) + 0 * tl.num_programs(0)) * r0_numel + r0_index, accum0, r0_mask)
    tl.store(ws_ptr + (tl.program_id(0) + 1 * tl.num_programs(0)) * r0_numel + r0_index, accum1, r0_mask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/j6/cj6cwm5z4u56dflo7nfolpmcqln7rcpnx7yxv7baxtxnebvsl44i.py
# Topologically Sorted Source Nodes: [view_39, permute_54, add_19, slice_2, add_20], Original ATen: [aten.view, aten.transpose, aten.add, aten.slice]
# Source node to ATen node mapping:
#   add_19 => add_19
#   add_20 => add_20
#   permute_54 => permute_54
#   slice_2 => slice_2
#   view_39 => view_39
# Graph fragment:
#   %add_17 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0" = PlaceHolder[target=add_17]
#   %mm_11 : Tensor "f32[500, 128][128, 1]cuda:0" = PlaceHolder[target=mm_11]
#   %mul_56 : Tensor "f32[500, 129, 128][16512, 128, 1]cuda:0" = PlaceHolder[target=mul_56]
#   %view_39 : Tensor "f32[1, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_11, [1, 500, 128]), kwargs = {})
#   %permute_54 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_39, [1, 0, 2]), kwargs = {})
#   %add_19 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %permute_54), kwargs = {})
#   %slice_2 : Tensor "f32[500, 1, 128][16512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_56, 1, 0, 1), kwargs = {})
#   %add_20 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_19, %slice_2), kwargs = {})
#   return %add_20
triton_poi_fused_add_slice_transpose_view_22 = async_compile.triton('triton_poi_fused_add_slice_transpose_view_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_slice_transpose_view_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1280000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_slice_transpose_view_22(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    x1 = xindex // 128
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_ptr1 + (x0 + 16512*x1), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
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
        primals_2, primals_7, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, view, cat_1, getitem_1, rsqrt, view_3, view_5, div, view_13, addmm_2, getitem_7, rsqrt_1, mul_5, view_16, addmm_3, getitem_11, rsqrt_3, view_18, mul_12, squeeze_1, div_1, div_3, permute_37, permute_38, permute_39, permute_47, permute_49, tangents_1 = args
        args.clear()
        assert_size_stride(primals_2, (128, 512), (512, 1))
        assert_size_stride(primals_7, (128, ), (1, ))
        assert_size_stride(primals_11, (128, 128), (128, 1))
        assert_size_stride(primals_13, (128, ), (1, ))
        assert_size_stride(primals_15, (128, ), (1, ))
        assert_size_stride(primals_17, (512, 128), (128, 1))
        assert_size_stride(primals_19, (512, ), (1, ))
        assert_size_stride(primals_21, (128, 512), (512, 1))
        assert_size_stride(primals_23, (128, ), (1, ))
        assert_size_stride(primals_25, (10, 128), (128, 1))
        assert_size_stride(view, (500, 512), (512, 1))
        assert_size_stride(cat_1, (500, 129, 128), (16512, 128, 1))
        assert_size_stride(getitem_1, (500, 129, 1), (129, 1, 1))
        assert_size_stride(rsqrt, (500, 129, 1), (129, 1, 1))
        assert_size_stride(view_3, (500, 128), (128, 1))
        assert_size_stride(view_5, (64500, 128), (128, 1))
        assert_size_stride(div, (4000, 1, 129), (129, 129, 1))
        assert_size_stride(view_13, (500, 128), (128, 1))
        assert_size_stride(addmm_2, (500, 128), (128, 1))
        assert_size_stride(getitem_7, (500, 1, 1), (1, 1, 1))
        assert_size_stride(rsqrt_1, (500, 1, 1), (1, 1, 1))
        assert_size_stride(mul_5, (500, 1, 128), (128, 128, 1))
        assert_size_stride(view_16, (500, 128), (128, 1))
        assert_size_stride(addmm_3, (500, 512), (512, 1))
        assert_size_stride(getitem_11, (500, 1, 1), (1, 1, 1))
        assert_size_stride(rsqrt_3, (500, 1, 1), (1, 1, 1))
        assert_size_stride(view_18, (500, 512), (512, 1))
        assert_size_stride(mul_12, (500, 1, 128), (128, 128, 1))
        assert_size_stride(squeeze_1, (500, 128), (128, 1))
        assert_size_stride(div_1, (500, 1, 1), (1, 1, 1))
        assert_size_stride(div_3, (500, 1, 1), (1, 1, 1))
        assert_size_stride(permute_37, (4000, 16, 129), (16, 1, 64000))
        assert_size_stride(permute_38, (4000, 129, 16), (16, 64000, 1))
        assert_size_stride(permute_39, (4000, 16, 1), (16, 1, 16))
        assert_size_stride(permute_47, (256, 128), (128, 1))
        assert_size_stride(permute_49, (128, 128), (128, 1))
        assert_size_stride(tangents_1, (500, 10), (10, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf3 = empty_strided_cuda((1, 10, 4), (40, 1, 10), torch.float32)
            # Topologically Sorted Source Nodes: [sum_2], Original ATen: [aten.sum]
            # [Provenance debug handles] triton_red_fused_sum_0:1
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_0.run(tangents_1, buf3, 40, 125, stream=stream0)
            buf4 = empty_strided_cuda((1, 10), (10, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sum_2], Original ATen: [aten.sum]
            # [Provenance debug handles] triton_per_fused_sum_1:2
            stream0 = get_raw_stream(0)
            triton_per_fused_sum_1.run(buf3, buf4, 10, 4, stream=stream0)
            del buf3
            buf1 = empty_strided_cuda((12, 500), (1, 12), torch.float32)
            # Topologically Sorted Source Nodes: [permute_19, ], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] triton_poi_fused_mm_t_2:3
            stream0 = get_raw_stream(0)
            triton_poi_fused_mm_t_2.run(tangents_1, buf1, 6000, stream=stream0)
            buf2 = empty_strided_cuda((12, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_19, ], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:4
            extern_kernels.mm(buf1, squeeze_1, out=buf2)
            del buf1
            del squeeze_1
            buf0 = empty_strided_cuda((500, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_3, permute_18, mm_1], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:5
            extern_kernels.mm(tangents_1, primals_25, out=buf0)
            del primals_25
            del tangents_1
            buf8 = empty_strided_cuda((128, 4), (1, 128), torch.float32)
            buf10 = empty_strided_cuda((128, 4), (1, 128), torch.float32)
            # Topologically Sorted Source Nodes: [unsqueeze_1, mul_20, sum_5, sum_6], Original ATen: [aten.unsqueeze, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_red_fused_native_layer_norm_backward_unsqueeze_3:6
            stream0 = get_raw_stream(0)
            triton_red_fused_native_layer_norm_backward_unsqueeze_3.run(buf0, mul_12, buf8, buf10, 512, 125, stream=stream0)
            buf9 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [unsqueeze_1, mul_20, sum_5], Original ATen: [aten.unsqueeze, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_backward_unsqueeze_4:7
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_backward_unsqueeze_4.run(buf8, buf9, 128, 4, stream=stream0)
            buf11 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [unsqueeze_1, sum_6], Original ATen: [aten.unsqueeze, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_backward_unsqueeze_4:8
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_backward_unsqueeze_4.run(buf10, buf11, 128, 4, stream=stream0)
            buf7 = reinterpret_tensor(buf0, (500, 1, 128), (128, 128, 1), 0); del buf0  # reuse
            # Topologically Sorted Source Nodes: [unsqueeze_1, mul_15, mul_16, sum_3, mul_17, sum_4, mul_18, sub_7, sub_8, mul_19], Original ATen: [aten.unsqueeze, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_backward_unsqueeze_5:9
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_backward_unsqueeze_5.run(buf7, primals_23, mul_12, div_1, 500, 128, stream=stream0)
            del div_1
            del mul_12
            del primals_23
            buf13 = empty_strided_cuda((128, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_21, permute_23, permute_25], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:10
            extern_kernels.mm(reinterpret_tensor(buf7, (128, 500), (1, 128), 0), view_18, out=buf13)
            del view_18
            buf14 = reinterpret_tensor(buf10, (1, 128, 4), (512, 1, 128), 0); del buf10  # reuse
            # Topologically Sorted Source Nodes: [view_21, sum_7], Original ATen: [aten.view, aten.sum]
            # [Provenance debug handles] triton_red_fused_sum_view_6:11
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_view_6.run(buf7, buf14, 512, 125, stream=stream0)
            buf15 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_21, sum_7], Original ATen: [aten.view, aten.sum]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_backward_unsqueeze_4:12
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_backward_unsqueeze_4.run(buf14, buf15, 128, 4, stream=stream0)
            buf12 = empty_strided_cuda((500, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_21, linear_2, permute_22, mm_3], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:13
            extern_kernels.mm(reinterpret_tensor(buf7, (500, 128), (128, 1), 0), primals_21, out=buf12)
            del primals_21
            buf18 = empty_strided_cuda((512, 4), (1, 512), torch.float32)
            buf20 = empty_strided_cuda((512, 4), (1, 512), torch.float32)
            # Topologically Sorted Source Nodes: [view_23, linear_1, gelu, layer_norm_3, mul_27, sum_10, sum_11], Original ATen: [aten.view, aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_red_fused_gelu_native_layer_norm_native_layer_norm_backward_view_7:14
            stream0 = get_raw_stream(0)
            triton_red_fused_gelu_native_layer_norm_native_layer_norm_backward_view_7.run(buf12, addmm_3, getitem_11, rsqrt_3, buf18, buf20, 2048, 125, stream=stream0)
            buf19 = reinterpret_tensor(buf14, (512, ), (1, ), 0); del buf14  # reuse
            # Topologically Sorted Source Nodes: [view_23, linear_1, gelu, layer_norm_3, mul_27, sum_10], Original ATen: [aten.view, aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_8:15
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_8.run(buf18, buf19, 512, 4, stream=stream0)
            del buf18
            buf21 = reinterpret_tensor(buf8, (512, ), (1, ), 0); del buf8  # reuse
            # Topologically Sorted Source Nodes: [view_23, sum_11], Original ATen: [aten.view, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_8:16
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_8.run(buf20, buf21, 512, 4, stream=stream0)
            buf22 = reinterpret_tensor(buf12, (500, 1, 512), (512, 512, 1), 0); del buf12  # reuse
            # Topologically Sorted Source Nodes: [view_23, mul_22, mul_23, sum_8, linear_1, gelu, layer_norm_3, mul_24, sum_9, mul_25, sub_10, sub_11, div_2, mul_26, mul_29, mul_30, mul_31, exp_1, mul_32, mul_33, add_16, mul_34], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.gelu, aten.native_layer_norm, aten.gelu_backward]
            # [Provenance debug handles] triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_view_9:17
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_view_9.run(buf22, primals_19, addmm_3, getitem_11, rsqrt_3, 500, 512, stream=stream0)
            del addmm_3
            del getitem_11
            del primals_19
            del rsqrt_3
            buf24 = empty_strided_cuda((512, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_24, permute_27, permute_29], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:18
            extern_kernels.mm(reinterpret_tensor(buf22, (512, 500), (1, 512), 0), view_16, out=buf24)
            del view_16
            buf25 = reinterpret_tensor(buf20, (1, 512, 4), (2048, 1, 512), 0); del buf20  # reuse
            # Topologically Sorted Source Nodes: [view_24, sum_12], Original ATen: [aten.view, aten.sum]
            # [Provenance debug handles] triton_red_fused_sum_view_10:19
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_view_10.run(buf22, buf25, 2048, 125, stream=stream0)
            buf23 = empty_strided_cuda((500, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_24, linear_1, permute_26, mm_5], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:20
            extern_kernels.mm(reinterpret_tensor(buf22, (500, 512), (512, 1), 0), primals_17, out=buf23)
            del buf22
            del primals_17
            buf26 = empty_strided_cuda((1, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_24, sum_12], Original ATen: [aten.view, aten.sum]
            # [Provenance debug handles] triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_8:21
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_8.run(buf25, buf26, 512, 4, stream=stream0)
            del buf25
            buf29 = empty_strided_cuda((128, 4), (1, 128), torch.float32)
            buf31 = empty_strided_cuda((128, 4), (1, 128), torch.float32)
            # Topologically Sorted Source Nodes: [view_26, mul_41, sum_15, sum_16], Original ATen: [aten.view, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_red_fused_native_layer_norm_backward_unsqueeze_3:22
            stream0 = get_raw_stream(0)
            triton_red_fused_native_layer_norm_backward_unsqueeze_3.run(buf23, mul_5, buf29, buf31, 512, 125, stream=stream0)
            buf33 = reinterpret_tensor(buf7, (500, 1, 128), (128, 64000, 1), 0); del buf7  # reuse
            buf40 = empty_strided_cuda((500, 1, 128), (128, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_26, mul_36, mul_37, sum_13, mul_38, sum_14, mul_39, sub_13, sub_14, mul_40, add_17, mul_43, mul_44, sum_17, multi_head_attention_forward, transpose_2, layer_norm_1, mul_45, sum_18, mul_46, sub_16, sub_17, div_4, mul_47], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add, aten.transpose, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_11:23
            stream0 = get_raw_stream(0)
            triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_11.run(buf33, buf23, primals_15, mul_5, div_3, primals_13, addmm_2, getitem_7, rsqrt_1, buf40, 500, 128, stream=stream0)
            del div_3
            del mul_5
            del primals_13
            del primals_15
            buf36 = empty_strided_cuda((128, 4), (1, 128), torch.float32)
            buf38 = empty_strided_cuda((128, 4), (1, 128), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward, transpose_2, layer_norm_1, mul_48, sum_19, sum_20], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_red_fused_native_layer_norm_native_layer_norm_backward_transpose_view_12:24
            stream0 = get_raw_stream(0)
            triton_red_fused_native_layer_norm_native_layer_norm_backward_transpose_view_12.run(buf33, addmm_2, getitem_7, rsqrt_1, buf36, buf38, 512, 125, stream=stream0)
            del addmm_2
            del getitem_7
            del rsqrt_1
            buf42 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mul_43, mul_44, multi_head_attention_forward, transpose_2, layer_norm_1, mul_46, sub_16, sub_17, div_4, mul_47, permute_30, view_27, permute_32, permute_34], Original ATen: [aten.native_layer_norm_backward, aten.view, aten.transpose, aten.native_layer_norm, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:25
            extern_kernels.mm(reinterpret_tensor(buf40, (128, 500), (1, 128), 0), view_13, out=buf42)
            del view_13
            buf30 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_26, mul_41, sum_15], Original ATen: [aten.view, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_backward_unsqueeze_4:26
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_backward_unsqueeze_4.run(buf29, buf30, 128, 4, stream=stream0)
            del buf29
            buf32 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_26, sum_16], Original ATen: [aten.view, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_backward_unsqueeze_4:27
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_backward_unsqueeze_4.run(buf31, buf32, 128, 4, stream=stream0)
            del buf31
            buf37 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward, transpose_2, layer_norm_1, mul_48, sum_19], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_backward_unsqueeze_4:28
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_backward_unsqueeze_4.run(buf36, buf37, 128, 4, stream=stream0)
            del buf36
            buf39 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [sum_20], Original ATen: [aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_backward_unsqueeze_4:29
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_backward_unsqueeze_4.run(buf38, buf39, 128, 4, stream=stream0)
            buf43 = reinterpret_tensor(buf38, (1, 128, 4), (512, 1, 128), 0); del buf38  # reuse
            # Topologically Sorted Source Nodes: [mul_43, mul_44, multi_head_attention_forward, transpose_2, layer_norm_1, mul_46, sub_16, sub_17, div_4, mul_47, permute_30, view_27, sum_21], Original ATen: [aten.native_layer_norm_backward, aten.view, aten.transpose, aten.native_layer_norm, aten.sum]
            # [Provenance debug handles] triton_red_fused_sum_view_6:30
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_view_6.run(buf40, buf43, 512, 125, stream=stream0)
            buf44 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mul_43, mul_44, multi_head_attention_forward, transpose_2, layer_norm_1, mul_46, sub_16, sub_17, div_4, mul_47, permute_30, view_27, sum_21], Original ATen: [aten.native_layer_norm_backward, aten.view, aten.transpose, aten.native_layer_norm, aten.sum]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_backward_unsqueeze_4:31
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_backward_unsqueeze_4.run(buf43, buf44, 128, 4, stream=stream0)
            buf41 = buf23; del buf23  # reuse
            # Topologically Sorted Source Nodes: [mul_43, mul_44, multi_head_attention_forward, transpose_2, layer_norm_1, mul_46, sub_16, sub_17, div_4, mul_47, permute_30, view_27, permute_31, mm_7], Original ATen: [aten.native_layer_norm_backward, aten.view, aten.transpose, aten.native_layer_norm, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:32
            extern_kernels.mm(reinterpret_tensor(buf40, (500, 128), (128, 1), 0), primals_11, out=buf41)
            del primals_11
            buf46 = empty_strided_cuda((4000, 1, 129), (129, 129, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_29, permute_35, bmm_2], Original ATen: [aten.view, aten.transpose, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:33
            extern_kernels.bmm(reinterpret_tensor(buf41, (4000, 1, 16), (16, 64000, 1), 0), permute_37, out=buf46)
            del permute_37
            buf48 = reinterpret_tensor(buf46, (4000, 1, 129), (129, 516000, 1), 0); del buf46  # reuse
            # Topologically Sorted Source Nodes: [mul_49, sum_22, neg, fma], Original ATen: [aten._softmax_backward_data]
            # [Provenance debug handles] triton_per_fused__softmax_backward_data_13:34
            stream0 = get_raw_stream(0)
            triton_per_fused__softmax_backward_data_13.run(buf48, div, 4000, 129, stream=stream0)
            buf49 = reinterpret_tensor(buf40, (4000, 1, 16), (16, 16, 1), 0); del buf40  # reuse
            # Topologically Sorted Source Nodes: [bmm_3], Original ATen: [aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:35
            extern_kernels.bmm(buf48, permute_38, out=buf49)
            del permute_38
            buf57 = reinterpret_tensor(buf49, (500, 128), (128, 1), 0); del buf49  # reuse
            # Topologically Sorted Source Nodes: [mul_50, permute_43, view_32, view_37], Original ATen: [aten.mul, aten.transpose, aten.view]
            # [Provenance debug handles] triton_poi_fused_mul_transpose_view_14:36
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_transpose_view_14.run(buf57, 64000, stream=stream0)
            buf65 = empty_strided_cuda((384, 128), (128, 1), torch.float32)
            buf59 = reinterpret_tensor(buf65, (128, 128), (128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [permute_50, permute_52], Original ATen: [aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:37
            extern_kernels.mm(reinterpret_tensor(buf57, (128, 500), (1, 128), 0), view_3, out=buf59)
            del view_3
            buf60 = buf43; del buf43  # reuse
            # Topologically Sorted Source Nodes: [sum_24], Original ATen: [aten.sum]
            # [Provenance debug handles] triton_red_fused_sum_view_6:38
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_view_6.run(buf57, buf60, 512, 125, stream=stream0)
            buf58 = empty_strided_cuda((500, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mm_11], Original ATen: [aten.mm]
            # [Provenance debug handles] extern_kernels.mm:39
            extern_kernels.mm(buf57, permute_49, out=buf58)
            del buf57
            del permute_49
            buf64 = empty_strided_cuda((384, ), (1, ), torch.float32)
            buf62 = reinterpret_tensor(buf64, (128, ), (1, ), 0)  # alias
            # Topologically Sorted Source Nodes: [view_34, sum_24, view_38, cat_2], Original ATen: [aten.view, aten.sum, aten.cat]
            # [Provenance debug handles] triton_per_fused_cat_sum_view_15:40
            stream0 = get_raw_stream(0)
            triton_per_fused_cat_sum_view_15.run(buf60, buf62, 128, 4, stream=stream0)
            del buf60
            buf50 = empty_strided_cuda((4000, 1, 132), (132, 528000, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            # [Provenance debug handles] triton_poi_fused_bmm_16:41
            stream0 = get_raw_stream(0)
            triton_poi_fused_bmm_16.run(buf48, buf50, 528000, stream=stream0)
            del buf48
            buf45 = empty_strided_cuda((4000, 129, 16), (2064, 16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_29, permute_35, permute_36, bmm_1], Original ATen: [aten.view, aten.transpose, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:42
            extern_kernels.bmm(reinterpret_tensor(div, (4000, 129, 1), (129, 1, 129), 0), reinterpret_tensor(buf41, (4000, 1, 16), (16, 64000, 1), 0), out=buf45)
            del buf41
            del div
            buf51 = empty_strided_cuda((4000, 16, 132), (2112, 132, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:43
            extern_kernels.bmm(permute_39, buf50, out=buf51)
            del buf50
            del permute_39
            buf52 = empty_strided_cuda((1, 1, 256, 375), (96000, 96000, 1, 256), torch.float32)
            # Topologically Sorted Source Nodes: [, permute_40, permute_41, clone_7, view_30, permute_42, view_31, full_1, _generalized_scatter, _generalized_scatter_1, add_18, unsqueeze_2, permute_44, squeeze_2, clone_8, view_33, sum_23], Original ATen: [aten.bmm, aten.transpose, aten.clone, aten._unsafe_view, aten.view, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_add_bmm_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_17:44
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_add_bmm_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_17.run(buf45, buf51, buf52, 96000, 172, stream=stream0)
            buf63 = reinterpret_tensor(buf64, (256, ), (1, ), 128)  # alias
            # Topologically Sorted Source Nodes: [, permute_40, permute_41, clone_7, view_30, permute_42, view_31, full_1, _generalized_scatter, _generalized_scatter_1, add_18, unsqueeze_2, permute_44, squeeze_2, clone_8, view_33, sum_23, view_34, view_38, cat_2], Original ATen: [aten.bmm, aten.transpose, aten.clone, aten._unsafe_view, aten.view, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.sum, aten.cat]
            # [Provenance debug handles] triton_red_fused__unsafe_view_add_bmm_cat_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_18:45
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_add_bmm_cat_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_18.run(buf52, buf63, 256, 375, stream=stream0)
            del buf52
            buf54 = empty_strided_cuda((129, 500, 2, 128), (128000, 256, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, permute_40, permute_41, clone_7, view_30, permute_42, view_31, full_1, _generalized_scatter, _generalized_scatter_1, add_18, unsqueeze_2, permute_44, squeeze_2, clone_8], Original ATen: [aten.bmm, aten.transpose, aten.clone, aten._unsafe_view, aten.view, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_bmm_clone_select_backward_squeeze_transpose_unsqueeze_view_19:46
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_bmm_clone_select_backward_squeeze_transpose_unsqueeze_view_19.run(buf45, buf51, buf54, 64500, 256, stream=stream0)
            del buf51
            buf55 = reinterpret_tensor(buf65, (256, 128), (128, 1), 16384)  # alias
            # Topologically Sorted Source Nodes: [, permute_40, permute_41, clone_7, view_30, permute_42, view_31, full_1, _generalized_scatter, _generalized_scatter_1, add_18, unsqueeze_2, permute_44, squeeze_2, clone_8, view_33, view_35, permute_45, permute_48], Original ATen: [aten.bmm, aten.transpose, aten.clone, aten._unsafe_view, aten.view, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:47
            extern_kernels.mm(reinterpret_tensor(buf54, (256, 64500), (1, 256), 0), view_5, out=buf55)
            del view_5
            buf56 = reinterpret_tensor(buf45, (64500, 128), (128, 1), 0); del buf45  # reuse
            # Topologically Sorted Source Nodes: [, permute_40, permute_41, clone_7, view_30, permute_42, view_31, full_1, _generalized_scatter, _generalized_scatter_1, add_18, unsqueeze_2, permute_44, squeeze_2, clone_8, view_33, view_35, mm_10], Original ATen: [aten.bmm, aten.transpose, aten.clone, aten._unsafe_view, aten.view, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:48
            extern_kernels.mm(reinterpret_tensor(buf54, (64500, 256), (256, 1), 0), permute_47, out=buf56)
            del buf54
            del permute_47
            buf66 = empty_strided_cuda((500, 129, 1), (1, 500, 64512), torch.float32)
            # Topologically Sorted Source Nodes: [view_36, permute_53, mul_52, sum_25], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_backward_transpose_view_20:49
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_backward_transpose_view_20.run(buf56, primals_7, buf66, 64500, 128, stream=stream0)
            buf68 = empty_strided_cuda((500, 129, 128), (16512, 128, 1), torch.float32)
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [view_36, permute_53, mul_52, mul_53, layer_norm, mul_54, sum_26, mul_55, sub_19, sub_20, div_5, mul_56, mul_57, sum_27, sum_28], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward, aten.native_layer_norm]
            workspace_0 = empty_strided_cuda((258048, ), (1, ), torch.float32)
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_native_layer_norm_backward_transpose_view_21.run(buf56, primals_7, cat_1, getitem_1, rsqrt, buf66, buf68, workspace_0, 64500, 128, stream=stream0)
            buf70 = workspace_0[0 * 1008 * 128 : (0 + 1) * 1008 * 128].view(1008, 128).sum(dim=0)
            buf72 = workspace_0[1 * 1008 * 128 : (1 + 1) * 1008 * 128].view(1008, 128).sum(dim=0)
            del workspace_0
            del buf56
            del buf66
            del cat_1
            del getitem_1
            del primals_7
            del rsqrt
            buf73 = reinterpret_tensor(buf33, (500, 1, 128), (128, 128, 1), 0); del buf33  # reuse
            # Topologically Sorted Source Nodes: [view_39, permute_54, add_19, slice_2, add_20], Original ATen: [aten.view, aten.transpose, aten.add, aten.slice]
            # [Provenance debug handles] triton_poi_fused_add_slice_transpose_view_22:50
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_slice_transpose_view_22.run(buf73, buf58, buf68, 64000, stream=stream0)
            del buf58
            buf75 = empty_strided_cuda((128, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_40, permute_56, permute_58], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:51
            extern_kernels.mm(reinterpret_tensor(buf73, (128, 500), (1, 128), 0), view, out=buf75)
            del view
            buf76 = empty_strided_cuda((1, 128, 4), (512, 1, 128), torch.float32)
            # Topologically Sorted Source Nodes: [view_40, sum_29], Original ATen: [aten.view, aten.sum]
            # [Provenance debug handles] triton_red_fused_sum_view_6:52
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_view_6.run(buf73, buf76, 512, 125, stream=stream0)
            buf77 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_40, sum_29], Original ATen: [aten.view, aten.sum]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_backward_unsqueeze_4:53
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_backward_unsqueeze_4.run(buf76, buf77, 128, 4, stream=stream0)
            del buf76
            buf74 = empty_strided_cuda((500, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_40, linear, permute_55, mm_13], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:54
            extern_kernels.mm(reinterpret_tensor(buf73, (500, 128), (128, 1), 0), primals_2, out=buf74)
            del primals_2
        return (reinterpret_tensor(buf74, (500, 1, 512), (512, 512, 1), 0), buf75, reinterpret_tensor(buf77, (128, ), (1, ), 0), buf73, None, reinterpret_tensor(buf68, (500, 128, 128), (16512, 128, 1), 128), buf70, buf72, buf65, buf64, buf42, reinterpret_tensor(buf44, (128, ), (1, ), 0), buf37, buf39, buf30, buf32, buf24, reinterpret_tensor(buf26, (512, ), (1, ), 0), buf19, buf21, buf13, reinterpret_tensor(buf15, (128, ), (1, ), 0), buf9, buf11, reinterpret_tensor(buf2, (10, 128), (128, 1), 0), reinterpret_tensor(buf4, (10, ), (1, ), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_2 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((10, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view = rand_strided((500, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    cat_1 = rand_strided((500, 129, 128), (16512, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((500, 129, 1), (129, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt = rand_strided((500, 129, 1), (129, 1, 1), device='cuda:0', dtype=torch.float32)
    view_3 = rand_strided((500, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_5 = rand_strided((64500, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div = rand_strided((4000, 1, 129), (129, 129, 1), device='cuda:0', dtype=torch.float32)
    view_13 = rand_strided((500, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_2 = rand_strided((500, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((500, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_1 = rand_strided((500, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_5 = rand_strided((500, 1, 128), (128, 128, 1), device='cuda:0', dtype=torch.float32)
    view_16 = rand_strided((500, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_3 = rand_strided((500, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_11 = rand_strided((500, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_3 = rand_strided((500, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    view_18 = rand_strided((500, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mul_12 = rand_strided((500, 1, 128), (128, 128, 1), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((500, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_1 = rand_strided((500, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    div_3 = rand_strided((500, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_37 = rand_strided((4000, 16, 129), (16, 1, 64000), device='cuda:0', dtype=torch.float32)
    permute_38 = rand_strided((4000, 129, 16), (16, 64000, 1), device='cuda:0', dtype=torch.float32)
    permute_39 = rand_strided((4000, 16, 1), (16, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_47 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_49 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((500, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_2, primals_7, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, view, cat_1, getitem_1, rsqrt, view_3, view_5, div, view_13, addmm_2, getitem_7, rsqrt_1, mul_5, view_16, addmm_3, getitem_11, rsqrt_3, view_18, mul_12, squeeze_1, div_1, div_3, permute_37, permute_38, permute_39, permute_47, permute_49, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
