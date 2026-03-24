# AOT ID: ['15_backward']
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


# kernel path: /traces/inductor_cache/vr/cvr7l5s3skvtu65fhn6z5phjp3hxtt6q2w4fy4ec4tjfqwjsyouv.py
# Topologically Sorted Source Nodes: [convert_element_type, mul_22, clone_5, view_17, sum_2], Original ATen: [aten.native_dropout_backward, aten.view, aten.sum]
# Source node to ATen node mapping:
#   clone_5 => mul_23
#   convert_element_type => convert_element_type
#   mul_22 => mul_22
#   sum_2 => sum_2
#   view_17 => view_17
# Graph fragment:
#   %tangents_1 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=tangents_1]
#   %gt_4 : Tensor "b8[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=gt_4]
#   %mul_23 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=mul_23]
#   %convert_element_type : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%gt_4, torch.float32), kwargs = {})
#   %mul_22 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, 1.1111111111111112), kwargs = {})
#   %mul_23 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %mul_22), kwargs = {})
#   %view_17 : Tensor "f32[64000, 128][128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_23, [64000, 128]), kwargs = {})
#   %sum_2 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_17, [0], True), kwargs = {})
#   return %mul_23,%buf3
triton_red_fused_native_dropout_backward_sum_view_0 = async_compile.triton('triton_red_fused_native_dropout_backward_sum_view_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_dropout_backward_sum_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 107008000, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_native_dropout_backward_sum_view_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 64000
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 128)
    x1 = xindex // 128
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 16384*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 128*r0_2 + 16384*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.int1)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = 1.1111111111111112
        tmp4 = tmp2 * tmp3
        tmp5 = tmp0 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
        tl.store(out_ptr0 + (x0 + 128*r0_2 + 16384*x1), tmp5, r0_mask & xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/bj/cbjeyrjogpccnllh6in26ddik5me3tsp7dakyzumsdvjvp6xro4s.py
# Topologically Sorted Source Nodes: [convert_element_type, mul_22, clone_5, view_17, sum_2], Original ATen: [aten.native_dropout_backward, aten.view, aten.sum]
# Source node to ATen node mapping:
#   clone_5 => mul_23
#   convert_element_type => convert_element_type
#   mul_22 => mul_22
#   sum_2 => sum_2
#   view_17 => view_17
# Graph fragment:
#   %buf3 : Tensor "f32[1, 128, 500][64000, 1, 128]cuda:0" = PlaceHolder[target=buf3]
#   %convert_element_type : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%gt_4, torch.float32), kwargs = {})
#   %mul_22 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, 1.1111111111111112), kwargs = {})
#   %mul_23 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %mul_22), kwargs = {})
#   %view_17 : Tensor "f32[64000, 128][128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_23, [64000, 128]), kwargs = {})
#   %sum_2 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_17, [0], True), kwargs = {})
#   return %sum_2
triton_red_fused_native_dropout_backward_sum_view_1 = async_compile.triton('triton_red_fused_native_dropout_backward_sum_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 512},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_dropout_backward_sum_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 257024, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_native_dropout_backward_sum_view_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 500
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
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/ng/cngstq64kpdqubue3edeglacllfpzfuxmemc237wnwyr4r6piysh.py
# Topologically Sorted Source Nodes: [view_19, mul_25, mul_26, sum_3, linear_1, gelu, dropout_2, layer_norm_3, mul_27, sum_4, mul_28, sub_6, sub_7, div_1, mul_29, mul_30, sum_5, sum_6, convert_element_type_1, mul_31, clone_6, mul_34, mul_35, mul_36, exp_1, mul_37, mul_38, add_15, mul_39], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.gelu, aten.native_dropout, aten.native_layer_norm, aten.native_dropout_backward, aten.gelu_backward]
# Source node to ATen node mapping:
#   add_15 => add_15
#   clone_6 => mul_32
#   convert_element_type_1 => convert_element_type_1
#   div_1 => div_1
#   dropout_2 => mul_16, mul_17
#   exp_1 => exp_1
#   gelu => add_10, erf, mul_13, mul_14, mul_15
#   layer_norm_3 => mul_18, sub_4
#   linear_1 => view_14
#   mul_25 => mul_25
#   mul_26 => mul_26
#   mul_27 => mul_27
#   mul_28 => mul_28
#   mul_29 => mul_29
#   mul_30 => mul_30
#   mul_31 => mul_31
#   mul_34 => mul_34
#   mul_35 => mul_35
#   mul_36 => mul_36
#   mul_37 => mul_37
#   mul_38 => mul_38
#   mul_39 => mul_39
#   sub_6 => sub_6
#   sub_7 => sub_7
#   sum_3 => sum_3
#   sum_4 => sum_4
#   sum_5 => sum_5
#   sum_6 => sum_6
#   view_19 => view_19
# Graph fragment:
#   %mm_1 : Tensor "f32[64000, 512][512, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %primals_19 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_19]
#   %gt_3 : Tensor "b8[500, 128, 512][65536, 512, 1]cuda:0" = PlaceHolder[target=gt_3]
#   %addmm_2 : Tensor "f32[64000, 512][512, 1]cuda:0" = PlaceHolder[target=addmm_2]
#   %getitem_7 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=getitem_7]
#   %rsqrt_3 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_3]
#   %sum_3 : Tensor "f32[500, 128, 1][128, 1, 64000]cuda:0" = PlaceHolder[target=sum_3]
#   %sum_4 : Tensor "f32[500, 128, 1][128, 1, 64000]cuda:0" = PlaceHolder[target=sum_4]
#   %mul_29 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0" = PlaceHolder[target=mul_29]
#   %view_19 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [500, 128, 512]), kwargs = {})
#   %mul_25 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_19, %primals_19), kwargs = {})
#   %mul_26 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, 512), kwargs = {})
#   %sum_3 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_25, [2], True), kwargs = {})
#   %view_14 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_2, [500, 128, 512]), kwargs = {})
#   %mul_13 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, 0.5), kwargs = {})
#   %mul_14 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_14,), kwargs = {})
#   %add_10 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_15 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %add_10), kwargs = {})
#   %mul_16 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt_3, %mul_15), kwargs = {})
#   %mul_17 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, 1.1111111111111112), kwargs = {})
#   %sub_4 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_17, %getitem_7), kwargs = {})
#   %mul_18 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_3), kwargs = {})
#   %mul_27 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %mul_18), kwargs = {})
#   %sum_4 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_27, [2], True), kwargs = {})
#   %mul_28 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, %sum_4), kwargs = {})
#   %sub_6 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_26, %sum_3), kwargs = {})
#   %sub_7 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_6, %mul_28), kwargs = {})
#   %div_1 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_3, 512), kwargs = {})
#   %mul_29 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %sub_7), kwargs = {})
#   %mul_30 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_19, %mul_18), kwargs = {})
#   %sum_5 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_30, [0, 1]), kwargs = {})
#   %sum_6 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_19, [0, 1]), kwargs = {})
#   %convert_element_type_1 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%gt_3, torch.float32), kwargs = {})
#   %mul_31 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1, 1.1111111111111112), kwargs = {})
#   %mul_32 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_29, %mul_31), kwargs = {})
#   %mul_34 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, 0.5), kwargs = {})
#   %mul_35 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, %view_14), kwargs = {})
#   %mul_36 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_35, -0.5), kwargs = {})
#   %exp_1 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_36,), kwargs = {})
#   %mul_37 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_1, 0.3989422804014327), kwargs = {})
#   %mul_38 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, %mul_37), kwargs = {})
#   %add_15 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_34, %mul_38), kwargs = {})
#   %mul_39 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_32, %add_15), kwargs = {})
#   return %sum_3,%sum_4,%mul_29,%mul_39
triton_per_fused_gelu_gelu_backward_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_view_2 = async_compile.triton('triton_per_fused_gelu_gelu_backward_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 65536, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*i1', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'ws_ptr': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'RSPLIT_SIZE': 'constexpr', 'NUM_STAGES': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'MixOrderReductionGrid', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_gelu_backward_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_view_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 6, 'num_store': -2, 'num_reduction': 2, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'RSPLIT_SIZE': 64}
)
@triton.jit
def triton_per_fused_gelu_gelu_backward_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_view_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, RSPLIT_SIZE : tl.constexpr, NUM_STAGES : tl.constexpr):
    xnumel = 64000
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * RSPLIT_SIZE
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    accum0 = tl.full([R0_BLOCK], 0, tl.float32)[None, :]
    accum1 = tl.full([R0_BLOCK], 0, tl.float32)[None, :]
    split_size = min(RSPLIT_SIZE, xnumel - xoffset)
    for _ in tl.range(0, split_size, XBLOCK, num_stages=NUM_STAGES):
        xmask = xindex < xnumel
        x0 = xindex
        xindex += XBLOCK
        tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), xmask, other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
        tmp7 = tl.load(in_ptr2 + (r0_1 + 512*x0), xmask, other=0.0).to(tl.int1)
        tmp9 = tl.load(in_ptr3 + (r0_1 + 512*x0), xmask, other=0.0)
        tmp21 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
        tmp23 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = tl.where(xmask, tmp3, 0)
        tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp10 = 0.5
        tmp11 = tmp9 * tmp10
        tmp12 = 0.7071067811865476
        tmp13 = tmp9 * tmp12
        tmp14 = libdevice.erf(tmp13)
        tmp15 = 1.0
        tmp16 = tmp14 + tmp15
        tmp17 = tmp11 * tmp16
        tmp18 = tmp8 * tmp17
        tmp19 = 1.1111111111111112
        tmp20 = tmp18 * tmp19
        tmp22 = tmp20 - tmp21
        tmp24 = tmp22 * tmp23
        tmp25 = tmp2 * tmp24
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
        tmp28 = tl.where(xmask, tmp26, 0)
        tmp29 = tl.sum(tmp28, 1)[:, None].to(tl.float32)
        tmp30 = 0.001953125
        tmp31 = tmp23 * tmp30
        tmp32 = 512.0
        tmp33 = tmp2 * tmp32
        tmp34 = tmp33 - tmp6
        tmp35 = tmp24 * tmp29
        tmp36 = tmp34 - tmp35
        tmp37 = tmp31 * tmp36
        tmp38 = tmp8 * tmp19
        tmp39 = tmp37 * tmp38
        tmp40 = tmp16 * tmp10
        tmp41 = tmp9 * tmp9
        tmp42 = -0.5
        tmp43 = tmp41 * tmp42
        tmp44 = libdevice.exp(tmp43)
        tmp45 = 0.3989422804014327
        tmp46 = tmp44 * tmp45
        tmp47 = tmp9 * tmp46
        tmp48 = tmp40 + tmp47
        tmp49 = tmp39 * tmp48
        tmp50 = tmp0 * tmp24
        tl.store(in_out_ptr0 + (r0_1 + 512*x0), tmp49, xmask)
        tmp51 = tl.sum(tmp50, 0)
        tmp52 = accum0 + tmp51
        accum0 = tmp52
        tmp53 = tl.sum(tmp0, 0)
        tmp54 = accum1 + tmp53
        accum1 = tmp54
    tl.store(ws_ptr + (tl.program_id(0) + 0 * tl.num_programs(0)) * r0_numel + r0_index, accum0, r0_mask)
    tl.store(ws_ptr + (tl.program_id(0) + 1 * tl.num_programs(0)) * r0_numel + r0_index, accum1, r0_mask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/cy/ccyte42cixz3wl7lg2kby2hqcmxofsezebegjmdq52dgbmqvlvgd.py
# Topologically Sorted Source Nodes: [linear_1, gelu, convert_element_type_1, mul_31, clone_6, mul_34, mul_35, mul_36, exp_1, mul_37, mul_38, add_15, mul_39, view_20, sum_7], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.sum]
# Source node to ATen node mapping:
#   add_15 => add_15
#   clone_6 => mul_32
#   convert_element_type_1 => convert_element_type_1
#   exp_1 => exp_1
#   gelu => add_10, erf, mul_14
#   linear_1 => view_14
#   mul_31 => mul_31
#   mul_34 => mul_34
#   mul_35 => mul_35
#   mul_36 => mul_36
#   mul_37 => mul_37
#   mul_38 => mul_38
#   mul_39 => mul_39
#   sum_7 => sum_7
#   view_20 => view_20
# Graph fragment:
#   %mul_39 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0" = PlaceHolder[target=mul_39]
#   %view_14 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_2, [500, 128, 512]), kwargs = {})
#   %mul_14 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_14,), kwargs = {})
#   %add_10 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %convert_element_type_1 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%gt_3, torch.float32), kwargs = {})
#   %mul_31 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1, 1.1111111111111112), kwargs = {})
#   %mul_32 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_29, %mul_31), kwargs = {})
#   %mul_34 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, 0.5), kwargs = {})
#   %mul_35 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, %view_14), kwargs = {})
#   %mul_36 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_35, -0.5), kwargs = {})
#   %exp_1 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_36,), kwargs = {})
#   %mul_37 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_1, 0.3989422804014327), kwargs = {})
#   %mul_38 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, %mul_37), kwargs = {})
#   %add_15 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_34, %mul_38), kwargs = {})
#   %mul_39 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_32, %add_15), kwargs = {})
#   %view_20 : Tensor "f32[64000, 512][512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_39, [64000, 512]), kwargs = {})
#   %sum_7 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_20, [0], True), kwargs = {})
#   return %buf15
triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_3 = async_compile.triton('triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 512},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 131891200, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_3(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 102400
    r0_numel = 320
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
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
        tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_2 + 163840*x1), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/a7/ca7al35j5n6mgw2bhws6sccvlds63hizcoqc5as6krvmcsjhcpsv.py
# Topologically Sorted Source Nodes: [linear_1, gelu, convert_element_type_1, mul_31, clone_6, mul_34, mul_35, mul_36, exp_1, mul_37, mul_38, add_15, mul_39, view_20, sum_7], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.sum]
# Source node to ATen node mapping:
#   add_15 => add_15
#   clone_6 => mul_32
#   convert_element_type_1 => convert_element_type_1
#   exp_1 => exp_1
#   gelu => add_10, erf, mul_14
#   linear_1 => view_14
#   mul_31 => mul_31
#   mul_34 => mul_34
#   mul_35 => mul_35
#   mul_36 => mul_36
#   mul_37 => mul_37
#   mul_38 => mul_38
#   mul_39 => mul_39
#   sum_7 => sum_7
#   view_20 => view_20
# Graph fragment:
#   %buf15 : Tensor "f32[1, 512, 200][102400, 1, 512]cuda:0" = PlaceHolder[target=buf15]
#   %view_14 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_2, [500, 128, 512]), kwargs = {})
#   %mul_14 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_14,), kwargs = {})
#   %add_10 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %convert_element_type_1 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%gt_3, torch.float32), kwargs = {})
#   %mul_31 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1, 1.1111111111111112), kwargs = {})
#   %mul_32 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_29, %mul_31), kwargs = {})
#   %mul_34 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, 0.5), kwargs = {})
#   %mul_35 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, %view_14), kwargs = {})
#   %mul_36 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_35, -0.5), kwargs = {})
#   %exp_1 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_36,), kwargs = {})
#   %mul_37 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_1, 0.3989422804014327), kwargs = {})
#   %mul_38 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, %mul_37), kwargs = {})
#   %add_15 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_34, %mul_38), kwargs = {})
#   %mul_39 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_32, %add_15), kwargs = {})
#   %view_20 : Tensor "f32[64000, 512][512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_39, [64000, 512]), kwargs = {})
#   %sum_7 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_20, [0], True), kwargs = {})
#   return %sum_7
triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_4 = async_compile.triton('triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 256},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 413696, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_4(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 200
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
        tmp0 = tl.load(in_ptr0 + (x0 + 512*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/al/caliqwy3zmv6avkjjypbhklkyhohmmpkrryozagn4itafpppan3f.py
# Topologically Sorted Source Nodes: [view_22, mul_41, mul_42, sum_8, mul_43, sum_9, mul_44, sub_9, sub_10, mul_45, mul_46, sum_10, sum_11, add_16], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
# Source node to ATen node mapping:
#   add_16 => add_16
#   mul_41 => mul_41
#   mul_42 => mul_42
#   mul_43 => mul_43
#   mul_44 => mul_44
#   mul_45 => mul_45
#   mul_46 => mul_46
#   sub_10 => sub_10
#   sub_9 => sub_9
#   sum_10 => sum_10
#   sum_11 => sum_11
#   sum_8 => sum_8
#   sum_9 => sum_9
#   view_22 => view_22
# Graph fragment:
#   %mm_3 : Tensor "f32[64000, 128][128, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %primals_15 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_15]
#   %mul_11 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=mul_11]
#   %tangents_1 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=tangents_1]
#   %div_2 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=div_2]
#   %sum_8 : Tensor "f32[500, 128, 1][128, 1, 64000]cuda:0" = PlaceHolder[target=sum_8]
#   %sum_9 : Tensor "f32[500, 128, 1][128, 1, 64000]cuda:0" = PlaceHolder[target=sum_9]
#   %view_22 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [500, 128, 128]), kwargs = {})
#   %mul_41 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_22, %primals_15), kwargs = {})
#   %mul_42 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_41, 128), kwargs = {})
#   %sum_8 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_41, [2], True), kwargs = {})
#   %mul_43 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_41, %mul_11), kwargs = {})
#   %sum_9 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_43, [2], True), kwargs = {})
#   %mul_44 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_11, %sum_9), kwargs = {})
#   %sub_9 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_42, %sum_8), kwargs = {})
#   %sub_10 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_9, %mul_44), kwargs = {})
#   %mul_45 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %sub_10), kwargs = {})
#   %mul_46 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_22, %mul_11), kwargs = {})
#   %sum_10 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_46, [0, 1]), kwargs = {})
#   %sum_11 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_22, [0, 1]), kwargs = {})
#   %add_16 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_1, %mul_45), kwargs = {})
#   return %sum_8,%sum_9,%add_16
triton_per_fused_add_native_layer_norm_backward_view_5 = async_compile.triton('triton_per_fused_add_native_layer_norm_backward_view_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'ws_ptr': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'RSPLIT_SIZE': 'constexpr', 'NUM_STAGES': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'MixOrderReductionGrid', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_view_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': -1, 'num_reduction': 2, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'RSPLIT_SIZE': 64}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_backward_view_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, RSPLIT_SIZE : tl.constexpr, NUM_STAGES : tl.constexpr):
    xnumel = 64000
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
    r0_1 = r0_index
    accum0 = tl.full([R0_BLOCK], 0, tl.float32)[None, :]
    accum1 = tl.full([R0_BLOCK], 0, tl.float32)[None, :]
    split_size = min(RSPLIT_SIZE, xnumel - xoffset)
    for _ in tl.range(0, split_size, XBLOCK, num_stages=NUM_STAGES):
        xmask = xindex < xnumel
        x0 = xindex
        xindex += XBLOCK
        tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), xmask, other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
        tmp7 = tl.load(in_ptr2 + (r0_1 + 128*x0), xmask, other=0.0)
        tmp13 = tl.load(in_ptr3 + (r0_1 + 128*x0), xmask, other=0.0)
        tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
        tmp22 = tmp0 * tmp7
        tl.store(out_ptr2 + (r0_1 + 128*x0), tmp21, xmask)
        tmp23 = tl.sum(tmp22, 0)
        tmp24 = accum0 + tmp23
        accum0 = tmp24
        tmp25 = tl.sum(tmp0, 0)
        tmp26 = accum1 + tmp25
        accum1 = tmp26
    tl.store(ws_ptr + (tl.program_id(0) + 0 * tl.num_programs(0)) * r0_numel + r0_index, accum0, r0_mask)
    tl.store(ws_ptr + (tl.program_id(0) + 1 * tl.num_programs(0)) * r0_numel + r0_index, accum1, r0_mask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/an/canxv7kqm6l4eu5y2vhidtblzj7q6nqjwowvv62coebvz5kewuw4.py
# Topologically Sorted Source Nodes: [convert_element_type_2, mul_47, clone_7, multi_head_attention_forward, transpose_1, sub_11, mul_49, mul_50, mul_51, sum_12, mul_52, sum_13, mul_53, sub_12, sub_13, div_3, mul_54, mul_55, sum_14, sum_15, permute_21, clone_8], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone]
# Source node to ATen node mapping:
#   clone_7 => mul_48
#   clone_8 => clone_8
#   convert_element_type_2 => convert_element_type_2
#   div_3 => div_3
#   mul_47 => mul_47
#   mul_49 => mul_49
#   mul_50 => mul_50
#   mul_51 => mul_51
#   mul_52 => mul_52
#   mul_53 => mul_53
#   mul_54 => mul_54
#   mul_55 => mul_55
#   multi_head_attention_forward => view_11
#   permute_21 => permute_21
#   sub_11 => sub_11
#   sub_12 => sub_12
#   sub_13 => sub_13
#   sum_12 => sum_12
#   sum_13 => sum_13
#   sum_14 => sum_14
#   sum_15 => sum_15
#   transpose_1 => permute_10
# Graph fragment:
#   %add_16 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=add_16]
#   %gt_2 : Tensor "b8[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=gt_2]
#   %primals_13 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_13]
#   %addmm_1 : Tensor "f32[64000, 128][128, 1]cuda:0" = PlaceHolder[target=addmm_1]
#   %getitem_3 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=getitem_3]
#   %rsqrt_1 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_1]
#   %sum_12 : Tensor "f32[500, 128, 1][128, 1, 64000]cuda:0" = PlaceHolder[target=sum_12]
#   %sum_13 : Tensor "f32[500, 128, 1][128, 1, 64000]cuda:0" = PlaceHolder[target=sum_13]
#   %convert_element_type_2 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%gt_2, torch.float32), kwargs = {})
#   %mul_47 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, 1.1111111111111112), kwargs = {})
#   %mul_48 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_16, %mul_47), kwargs = {})
#   %view_11 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_1, [128, 500, 128]), kwargs = {})
#   %permute_10 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_11, [1, 0, 2]), kwargs = {})
#   %sub_11 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_10, %getitem_3), kwargs = {})
#   %mul_49 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %rsqrt_1), kwargs = {})
#   %mul_50 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_48, %primals_13), kwargs = {})
#   %mul_51 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_50, 128), kwargs = {})
#   %sum_12 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_50, [2], True), kwargs = {})
#   %mul_52 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_50, %mul_49), kwargs = {})
#   %sum_13 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_52, [2], True), kwargs = {})
#   %mul_53 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_49, %sum_13), kwargs = {})
#   %sub_12 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_51, %sum_12), kwargs = {})
#   %sub_13 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_12, %mul_53), kwargs = {})
#   %div_3 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_1, 128), kwargs = {})
#   %mul_54 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, %sub_13), kwargs = {})
#   %mul_55 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_48, %mul_49), kwargs = {})
#   %sum_14 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_55, [0, 1]), kwargs = {})
#   %sum_15 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_48, [0, 1]), kwargs = {})
#   %permute_21 : Tensor "f32[128, 500, 128][128, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_54, [1, 0, 2]), kwargs = {})
#   %clone_8 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_21,), kwargs = {memory_format: torch.contiguous_format})
#   return %sum_12,%sum_13,%clone_8
triton_per_fused_clone_native_dropout_backward_native_layer_norm_backward_transpose_view_6 = async_compile.triton('triton_per_fused_clone_native_dropout_backward_native_layer_norm_backward_transpose_view_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'ws_ptr': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'RSPLIT_SIZE': 'constexpr', 'NUM_STAGES': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'MixOrderReductionGrid', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clone_native_dropout_backward_native_layer_norm_backward_transpose_view_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 6, 'num_store': -1, 'num_reduction': 2, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'RSPLIT_SIZE': 64}
)
@triton.jit
def triton_per_fused_clone_native_dropout_backward_native_layer_norm_backward_transpose_view_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, RSPLIT_SIZE : tl.constexpr, NUM_STAGES : tl.constexpr):
    xnumel = 64000
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
    r0_1 = r0_index
    accum0 = tl.full([R0_BLOCK], 0, tl.float32)[None, :]
    accum1 = tl.full([R0_BLOCK], 0, tl.float32)[None, :]
    split_size = min(RSPLIT_SIZE, xnumel - xoffset)
    for _ in tl.range(0, split_size, XBLOCK, num_stages=NUM_STAGES):
        xmask = xindex < xnumel
        x0 = xindex
        x2 = (xindex % 128)
        x3 = xindex // 128
        xindex += XBLOCK
        tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), xmask, other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 128*x0), xmask, other=0.0).to(tl.int1)
        tmp6 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr3 + (r0_1 + 128*x3 + 64000*x2), xmask, other=0.0)
        tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
        tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
        tmp2 = tmp1.to(tl.float32)
        tmp3 = 1.1111111111111112
        tmp4 = tmp2 * tmp3
        tmp5 = tmp0 * tmp4
        tmp7 = tmp5 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp10 = tl.where(xmask, tmp8, 0)
        tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
        tmp14 = tmp12 - tmp13
        tmp16 = tmp14 * tmp15
        tmp17 = tmp7 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
        tmp20 = tl.where(xmask, tmp18, 0)
        tmp21 = tl.sum(tmp20, 1)[:, None].to(tl.float32)
        tmp22 = 0.0078125
        tmp23 = tmp15 * tmp22
        tmp24 = 128.0
        tmp25 = tmp7 * tmp24
        tmp26 = tmp25 - tmp11
        tmp27 = tmp16 * tmp21
        tmp28 = tmp26 - tmp27
        tmp29 = tmp23 * tmp28
        tmp30 = tmp5 * tmp16
        tl.store(out_ptr2 + (r0_1 + 128*x3 + 64000*x2), tmp29, xmask)
        tmp31 = tl.sum(tmp30, 0)
        tmp32 = accum0 + tmp31
        accum0 = tmp32
        tmp33 = tl.sum(tmp5, 0)
        tmp34 = accum1 + tmp33
        accum1 = tmp34
    tl.store(ws_ptr + (tl.program_id(0) + 0 * tl.num_programs(0)) * r0_numel + r0_index, accum0, r0_mask)
    tl.store(ws_ptr + (tl.program_id(0) + 1 * tl.num_programs(0)) * r0_numel + r0_index, accum1, r0_mask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/4k/c4k7hwxdypwoze2xrpf6jk7fbuqv3xhxmz4bsseboqptn7udqf2q.py
# Topologically Sorted Source Nodes: [convert_element_type_2, mul_47, clone_7, multi_head_attention_forward, transpose_1, sub_11, mul_49, mul_50, mul_51, mul_53, sub_12, sub_13, div_3, mul_54, permute_21, clone_8, view_23, sum_16], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.sum]
# Source node to ATen node mapping:
#   clone_7 => mul_48
#   clone_8 => clone_8
#   convert_element_type_2 => convert_element_type_2
#   div_3 => div_3
#   mul_47 => mul_47
#   mul_49 => mul_49
#   mul_50 => mul_50
#   mul_51 => mul_51
#   mul_53 => mul_53
#   mul_54 => mul_54
#   multi_head_attention_forward => view_11
#   permute_21 => permute_21
#   sub_11 => sub_11
#   sub_12 => sub_12
#   sub_13 => sub_13
#   sum_16 => sum_16
#   transpose_1 => permute_10
#   view_23 => view_23
# Graph fragment:
#   %clone_8 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0" = PlaceHolder[target=clone_8]
#   %convert_element_type_2 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%gt_2, torch.float32), kwargs = {})
#   %mul_47 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, 1.1111111111111112), kwargs = {})
#   %mul_48 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_16, %mul_47), kwargs = {})
#   %view_11 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_1, [128, 500, 128]), kwargs = {})
#   %permute_10 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_11, [1, 0, 2]), kwargs = {})
#   %sub_11 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_10, %getitem_3), kwargs = {})
#   %mul_49 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %rsqrt_1), kwargs = {})
#   %mul_50 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_48, %primals_13), kwargs = {})
#   %mul_51 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_50, 128), kwargs = {})
#   %mul_53 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_49, %sum_13), kwargs = {})
#   %sub_12 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_51, %sum_12), kwargs = {})
#   %sub_13 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_12, %mul_53), kwargs = {})
#   %div_3 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_1, 128), kwargs = {})
#   %mul_54 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, %sub_13), kwargs = {})
#   %permute_21 : Tensor "f32[128, 500, 128][128, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_54, [1, 0, 2]), kwargs = {})
#   %clone_8 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_21,), kwargs = {memory_format: torch.contiguous_format})
#   %view_23 : Tensor "f32[64000, 128][128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_8, [64000, 128]), kwargs = {})
#   %sum_16 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_23, [0], True), kwargs = {})
#   return %buf33
triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_7 = async_compile.triton('triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 33280000, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_7(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 64000
    r0_numel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 16384*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/d3/cd3aq2bbpfvcpyaxjb6qq5kvwcdxhxkrea5ds6tvhqpjkenf7kue.py
# Topologically Sorted Source Nodes: [convert_element_type_3, mul_56, clone_9, multi_head_attention_forward, mul_58, sum_17, neg, fma], Original ATen: [aten.native_dropout_backward, aten._softmax, aten._softmax_backward_data]
# Source node to ATen node mapping:
#   clone_9 => mul_57
#   convert_element_type_3 => convert_element_type_3
#   fma => fma
#   mul_56 => mul_56
#   mul_58 => mul_58
#   multi_head_attention_forward => div, exp, sub_1
#   neg => neg
#   sum_17 => sum_17
# Graph fragment:
#   %bmm_2 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=bmm_2]
#   %gt_1 : Tensor "b8[4000, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=gt_1]
#   %baddbmm : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=baddbmm]
#   %amax : Tensor "f32[4000, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=amax]
#   %sum_1 : Tensor "f32[4000, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=sum_1]
#   %mul_58 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=mul_58]
#   %sum_17 : Tensor "f32[4000, 128, 1][128, 1, 512000]cuda:0" = PlaceHolder[target=sum_17]
#   %convert_element_type_3 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%gt_1, torch.float32), kwargs = {})
#   %mul_56 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_3, 1.1111111111111112), kwargs = {})
#   %mul_57 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_2, %mul_56), kwargs = {})
#   %sub_1 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%baddbmm, %amax), kwargs = {})
#   %exp : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %div : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   %mul_58 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_57, %div), kwargs = {})
#   %sum_17 : Tensor "f32[4000, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_58, [-1], True), kwargs = {})
#   %neg : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%div,), kwargs = {})
#   %fma : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.fma.default](args = (%neg, %sum_17, %mul_58), kwargs = {})
#   return %mul_58,%sum_17,%fma
triton_per_fused__softmax__softmax_backward_data_native_dropout_backward_8 = async_compile.triton('triton_per_fused__softmax__softmax_backward_data_native_dropout_backward_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 524288, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__softmax_backward_data_native_dropout_backward_8', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4096000, 'r0_': 1114112000}}
)
@triton.jit
def triton_per_fused__softmax__softmax_backward_data_native_dropout_backward_8(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 512000
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 128*x0), None)
    tmp1 = tl.load(in_ptr0 + (r0_1 + 128*x0), None).to(tl.int1)
    tmp6 = tl.load(in_out_ptr1 + (r0_1 + 128*x0), None)
    tmp7 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp8 = tmp6 - tmp7
    tmp9 = libdevice.exp(tmp8)
    tmp11 = (tmp9 / tmp10)
    tmp12 = tmp5 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.sum(tmp13, 1)[:, None].to(tl.float32)
    tmp16 = -tmp11
    tmp17 = libdevice.fma(tmp16, tmp15, tmp12)
    tl.store(in_out_ptr1 + (r0_1 + 128*x0), tmp17, None)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/lm/clmrh5cd633bvcq5dlbk2hws4zximvtedwo6pdln3vorjeiosm5o.py
# Topologically Sorted Source Nodes: [permute_31, mul_59, permute_32, clone_10, view_26, permute_33, view_27, permute_34, clone_11, view_28, full, _generalized_scatter, _generalized_scatter_1, add_17, _generalized_scatter_2, add_18, unsqueeze_1, permute_35, squeeze_1, clone_12, view_29, sum_18], Original ATen: [aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.sum]
# Source node to ATen node mapping:
#   _generalized_scatter => select_scatter_default
#   _generalized_scatter_1 => select_scatter_default_1
#   _generalized_scatter_2 => select_scatter_default_2
#   add_17 => add_17
#   add_18 => add_18
#   clone_10 => clone_10
#   clone_11 => clone_11
#   clone_12 => clone_12
#   full => full_default
#   mul_59 => mul_59
#   permute_31 => permute_31
#   permute_32 => permute_32
#   permute_33 => permute_33
#   permute_34 => permute_34
#   permute_35 => permute_35
#   squeeze_1 => squeeze_1
#   sum_18 => sum_18
#   unsqueeze_1 => unsqueeze_1
#   view_26 => view_26
#   view_27 => view_27
#   view_28 => view_28
#   view_29 => view_29
# Graph fragment:
#   %bmm_1 : Tensor "f32[4000, 128, 16][2048, 16, 1]cuda:0" = PlaceHolder[target=bmm_1]
#   %bmm_4 : Tensor "f32[4000, 16, 128][2048, 128, 1]cuda:0" = PlaceHolder[target=bmm_4]
#   %bmm_3 : Tensor "f32[4000, 128, 16][2048, 16, 1]cuda:0" = PlaceHolder[target=bmm_3]
#   %permute_31 : Tensor "f32[4000, 128, 16][2048, 1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_4, [0, 2, 1]), kwargs = {})
#   %mul_59 : Tensor "f32[4000, 128, 16][2048, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_3, 0.25), kwargs = {})
#   %permute_32 : Tensor "f32[128, 4000, 16][16, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_1, [1, 0, 2]), kwargs = {})
#   %clone_10 : Tensor "f32[128, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_32,), kwargs = {memory_format: torch.contiguous_format})
#   %view_26 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_10, [128, 500, 128]), kwargs = {})
#   %permute_33 : Tensor "f32[128, 4000, 16][1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_31, [1, 0, 2]), kwargs = {})
#   %view_27 : Tensor "f32[128, 500, 128][1, 16384, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_33, [128, 500, 128]), kwargs = {})
#   %permute_34 : Tensor "f32[128, 4000, 16][16, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_59, [1, 0, 2]), kwargs = {})
#   %clone_11 : Tensor "f32[128, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_34,), kwargs = {memory_format: torch.contiguous_format})
#   %view_28 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_11, [128, 500, 128]), kwargs = {})
#   %full_default : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([3, 128, 500, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %view_26, 0, 2), kwargs = {})
#   %select_scatter_default_1 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %view_27, 0, 1), kwargs = {})
#   %add_17 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default, %select_scatter_default_1), kwargs = {})
#   %select_scatter_default_2 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %view_28, 0, 0), kwargs = {})
#   %add_18 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %select_scatter_default_2), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[3, 128, 500, 1, 128][8192000, 64000, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_18, 3), kwargs = {})
#   %permute_35 : Tensor "f32[1, 128, 500, 3, 128][128, 64000, 128, 8192000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_1, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze_1 : Tensor "f32[128, 500, 3, 128][64000, 128, 8192000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_35, 0), kwargs = {})
#   %clone_12 : Tensor "f32[128, 500, 3, 128][192000, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_1,), kwargs = {memory_format: torch.contiguous_format})
#   %view_29 : Tensor "f32[128, 500, 384][192000, 384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_12, [128, 500, 384]), kwargs = {})
#   %sum_18 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_29, [0, 1], True), kwargs = {})
#   return %buf42
triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_9 = async_compile.triton('triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 66519040, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_9(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 122880
    r0_numel = 200
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 384)
    x1 = xindex // 384
    _tmp19 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp3 = tl.load(in_ptr0 + (16*((r0_2 + 200*x1) // 500) + 16*((128*(((r0_2 + 200*x1) % 500)) + ((x0 % 128))) // 64000) + 2048*(((x0 % 128)) // 16) + 16384*(((r0_2 + 200*x1) % 500)) + ((((x0 % 128)) % 16))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr1 + (128*((x0 % 128)) + 16384*(((r0_2 + 200*x1) % 500)) + ((r0_2 + 200*x1) // 500) + ((128*(((r0_2 + 200*x1) % 500)) + ((x0 % 128))) // 64000)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr2 + (16*((r0_2 + 200*x1) // 500) + 16*((128*(((r0_2 + 200*x1) % 500)) + ((x0 % 128))) // 64000) + 2048*(((x0 % 128)) // 16) + 16384*(((r0_2 + 200*x1) % 500)) + ((((x0 % 128)) % 16))), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp0 = x0 // 128
        tmp1 = tl.full([1, 1], 2, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp4 = 0.0
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.full([1, 1], 1, tl.int32)
        tmp7 = tmp0 == tmp6
        tmp9 = tl.where(tmp7, tmp8, tmp4)
        tmp10 = tmp5 + tmp9
        tmp11 = tl.full([1, 1], 0, tl.int32)
        tmp12 = tmp0 == tmp11
        tmp14 = 0.25
        tmp15 = tmp13 * tmp14
        tmp16 = tl.where(tmp12, tmp15, tmp4)
        tmp17 = tmp10 + tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(r0_mask, tmp20, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/zt/czt4f3lbtr7b665z74dymfpp27u3h5h2uwwez2d2kvtav7yo6otl.py
# Topologically Sorted Source Nodes: [permute_31, mul_59, permute_32, clone_10, view_26, permute_33, view_27, permute_34, clone_11, view_28, full, _generalized_scatter, _generalized_scatter_1, add_17, _generalized_scatter_2, add_18, unsqueeze_1, permute_35, squeeze_1, clone_12, view_29, sum_18], Original ATen: [aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.sum]
# Source node to ATen node mapping:
#   _generalized_scatter => select_scatter_default
#   _generalized_scatter_1 => select_scatter_default_1
#   _generalized_scatter_2 => select_scatter_default_2
#   add_17 => add_17
#   add_18 => add_18
#   clone_10 => clone_10
#   clone_11 => clone_11
#   clone_12 => clone_12
#   full => full_default
#   mul_59 => mul_59
#   permute_31 => permute_31
#   permute_32 => permute_32
#   permute_33 => permute_33
#   permute_34 => permute_34
#   permute_35 => permute_35
#   squeeze_1 => squeeze_1
#   sum_18 => sum_18
#   unsqueeze_1 => unsqueeze_1
#   view_26 => view_26
#   view_27 => view_27
#   view_28 => view_28
#   view_29 => view_29
# Graph fragment:
#   %buf42 : Tensor "f32[1, 1, 384, 320][122880, 122880, 1, 384]cuda:0" = PlaceHolder[target=buf42]
#   %permute_31 : Tensor "f32[4000, 128, 16][2048, 1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_4, [0, 2, 1]), kwargs = {})
#   %mul_59 : Tensor "f32[4000, 128, 16][2048, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_3, 0.25), kwargs = {})
#   %permute_32 : Tensor "f32[128, 4000, 16][16, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_1, [1, 0, 2]), kwargs = {})
#   %clone_10 : Tensor "f32[128, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_32,), kwargs = {memory_format: torch.contiguous_format})
#   %view_26 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_10, [128, 500, 128]), kwargs = {})
#   %permute_33 : Tensor "f32[128, 4000, 16][1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_31, [1, 0, 2]), kwargs = {})
#   %view_27 : Tensor "f32[128, 500, 128][1, 16384, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_33, [128, 500, 128]), kwargs = {})
#   %permute_34 : Tensor "f32[128, 4000, 16][16, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_59, [1, 0, 2]), kwargs = {})
#   %clone_11 : Tensor "f32[128, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_34,), kwargs = {memory_format: torch.contiguous_format})
#   %view_28 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_11, [128, 500, 128]), kwargs = {})
#   %full_default : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([3, 128, 500, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %view_26, 0, 2), kwargs = {})
#   %select_scatter_default_1 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %view_27, 0, 1), kwargs = {})
#   %add_17 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default, %select_scatter_default_1), kwargs = {})
#   %select_scatter_default_2 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %view_28, 0, 0), kwargs = {})
#   %add_18 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %select_scatter_default_2), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[3, 128, 500, 1, 128][8192000, 64000, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_18, 3), kwargs = {})
#   %permute_35 : Tensor "f32[1, 128, 500, 3, 128][128, 64000, 128, 8192000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_1, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze_1 : Tensor "f32[128, 500, 3, 128][64000, 128, 8192000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_35, 0), kwargs = {})
#   %clone_12 : Tensor "f32[128, 500, 3, 128][192000, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_1,), kwargs = {memory_format: torch.contiguous_format})
#   %view_29 : Tensor "f32[128, 500, 384][192000, 384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_12, [128, 500, 384]), kwargs = {})
#   %sum_18 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_29, [0, 1], True), kwargs = {})
#   return %sum_18
triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_10 = async_compile.triton('triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 512},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 494592, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_10(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 384
    r0_numel = 320
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
        tmp0 = tl.load(in_ptr0 + (x0 + 384*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/ju/cju76kubnubejf6zakvqbylfpob4jtjku7cszile3rapytyetno3.py
# Topologically Sorted Source Nodes: [permute_31, mul_59, permute_32, clone_10, view_26, permute_33, view_27, permute_34, clone_11, view_28, full, _generalized_scatter, _generalized_scatter_1, add_17, _generalized_scatter_2, add_18, unsqueeze_1, permute_35, squeeze_1, clone_12], Original ATen: [aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze]
# Source node to ATen node mapping:
#   _generalized_scatter => select_scatter_default
#   _generalized_scatter_1 => select_scatter_default_1
#   _generalized_scatter_2 => select_scatter_default_2
#   add_17 => add_17
#   add_18 => add_18
#   clone_10 => clone_10
#   clone_11 => clone_11
#   clone_12 => clone_12
#   full => full_default
#   mul_59 => mul_59
#   permute_31 => permute_31
#   permute_32 => permute_32
#   permute_33 => permute_33
#   permute_34 => permute_34
#   permute_35 => permute_35
#   squeeze_1 => squeeze_1
#   unsqueeze_1 => unsqueeze_1
#   view_26 => view_26
#   view_27 => view_27
#   view_28 => view_28
# Graph fragment:
#   %bmm_1 : Tensor "f32[4000, 128, 16][2048, 16, 1]cuda:0" = PlaceHolder[target=bmm_1]
#   %bmm_4 : Tensor "f32[4000, 16, 128][2048, 128, 1]cuda:0" = PlaceHolder[target=bmm_4]
#   %bmm_3 : Tensor "f32[4000, 128, 16][2048, 16, 1]cuda:0" = PlaceHolder[target=bmm_3]
#   %permute_31 : Tensor "f32[4000, 128, 16][2048, 1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_4, [0, 2, 1]), kwargs = {})
#   %mul_59 : Tensor "f32[4000, 128, 16][2048, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_3, 0.25), kwargs = {})
#   %permute_32 : Tensor "f32[128, 4000, 16][16, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_1, [1, 0, 2]), kwargs = {})
#   %clone_10 : Tensor "f32[128, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_32,), kwargs = {memory_format: torch.contiguous_format})
#   %view_26 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_10, [128, 500, 128]), kwargs = {})
#   %permute_33 : Tensor "f32[128, 4000, 16][1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_31, [1, 0, 2]), kwargs = {})
#   %view_27 : Tensor "f32[128, 500, 128][1, 16384, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_33, [128, 500, 128]), kwargs = {})
#   %permute_34 : Tensor "f32[128, 4000, 16][16, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_59, [1, 0, 2]), kwargs = {})
#   %clone_11 : Tensor "f32[128, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_34,), kwargs = {memory_format: torch.contiguous_format})
#   %view_28 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_11, [128, 500, 128]), kwargs = {})
#   %full_default : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([3, 128, 500, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %view_26, 0, 2), kwargs = {})
#   %select_scatter_default_1 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %view_27, 0, 1), kwargs = {})
#   %add_17 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default, %select_scatter_default_1), kwargs = {})
#   %select_scatter_default_2 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %view_28, 0, 0), kwargs = {})
#   %add_18 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %select_scatter_default_2), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[3, 128, 500, 1, 128][8192000, 64000, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_18, 3), kwargs = {})
#   %permute_35 : Tensor "f32[1, 128, 500, 3, 128][128, 64000, 128, 8192000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_1, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze_1 : Tensor "f32[128, 500, 3, 128][64000, 128, 8192000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_35, 0), kwargs = {})
#   %clone_12 : Tensor "f32[128, 500, 3, 128][192000, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_1,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_12
triton_poi_fused__unsafe_view_add_clone_mul_select_backward_squeeze_transpose_unsqueeze_view_11 = async_compile.triton('triton_poi_fused__unsafe_view_add_clone_mul_select_backward_squeeze_transpose_unsqueeze_view_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 512}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_clone_mul_select_backward_squeeze_transpose_unsqueeze_view_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 32768000, 'x': 262144000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_clone_mul_select_backward_squeeze_transpose_unsqueeze_view_11(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64000
    xnumel = 384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex // 128
    x2 = (xindex % 128)
    y0 = (yindex % 128)
    y1 = yindex // 128
    x4 = xindex
    tmp3 = tl.load(in_ptr0 + (16*y0 + 16*((x2 + 128*y1) // 64000) + 2048*(x2 // 16) + 16384*y1 + ((x2 % 16))), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (y0 + 128*x2 + 16384*y1 + ((x2 + 128*y1) // 64000)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (16*y0 + 16*((x2 + 128*y1) // 64000) + 2048*(x2 // 16) + 16384*y1 + ((x2 % 16))), xmask & ymask, eviction_policy='evict_last')
    tmp0 = x3
    tmp1 = tl.full([1, 1], 2, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = tl.full([1, 1], 1, tl.int32)
    tmp7 = tmp0 == tmp6
    tmp9 = tl.where(tmp7, tmp8, tmp4)
    tmp10 = tmp5 + tmp9
    tmp11 = tl.full([1, 1], 0, tl.int32)
    tmp12 = tmp0 == tmp11
    tmp14 = 0.25
    tmp15 = tmp13 * tmp14
    tmp16 = tl.where(tmp12, tmp15, tmp4)
    tmp17 = tmp10 + tmp16
    tl.store(out_ptr0 + (x4 + 384*y1 + 192000*y0), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/x6/cx6oxavbtnwnmlawnu5bdxma5pqbjxjfyxvp62jo4olhxosh5vtm.py
# Topologically Sorted Source Nodes: [view_32, permute_40, mul_61, sum_19], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   mul_61 => mul_61
#   permute_40 => permute_40
#   sum_19 => sum_19
#   view_32 => view_32
# Graph fragment:
#   %mm_8 : Tensor "f32[64000, 128][128, 1]cuda:0" = PlaceHolder[target=mm_8]
#   %primals_5 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_5]
#   %view_32 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_8, [128, 500, 128]), kwargs = {})
#   %permute_40 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_32, [1, 0, 2]), kwargs = {})
#   %mul_61 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_40, %primals_5), kwargs = {})
#   %sum_19 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_61, [2], True), kwargs = {})
#   return %sum_19
triton_per_fused_native_layer_norm_backward_transpose_view_12 = async_compile.triton('triton_per_fused_native_layer_norm_backward_transpose_view_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_transpose_view_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 512000, 'r0_': 32768512}}
)
@triton.jit
def triton_per_fused_native_layer_norm_backward_transpose_view_12(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 64000
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


# kernel path: /traces/inductor_cache/go/cgolnw5dpaugb2widzuhzobodsaorucry4jlkjxf6iwoqbeyql72.py
# Topologically Sorted Source Nodes: [view_32, permute_40, mul_61, mul_62, mul_63, sum_20, mul_64, sub_15, sub_16, mul_65, mul_66, sum_21, sum_22, add_19, convert_element_type_4, mul_67, clone_13], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward, aten.add, aten.native_dropout_backward]
# Source node to ATen node mapping:
#   add_19 => add_19
#   clone_13 => mul_68
#   convert_element_type_4 => convert_element_type_4
#   mul_61 => mul_61
#   mul_62 => mul_62
#   mul_63 => mul_63
#   mul_64 => mul_64
#   mul_65 => mul_65
#   mul_66 => mul_66
#   mul_67 => mul_67
#   permute_40 => permute_40
#   sub_15 => sub_15
#   sub_16 => sub_16
#   sum_20 => sum_20
#   sum_21 => sum_21
#   sum_22 => sum_22
#   view_32 => view_32
# Graph fragment:
#   %mm_8 : Tensor "f32[64000, 128][128, 1]cuda:0" = PlaceHolder[target=mm_8]
#   %primals_5 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_5]
#   %mul_2 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=mul_2]
#   %add_16 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=add_16]
#   %div_4 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=div_4]
#   %sum_19 : Tensor "f32[500, 128, 1][1, 500, 64000]cuda:0" = PlaceHolder[target=sum_19]
#   %sum_20 : Tensor "f32[500, 128, 1][128, 1, 64000]cuda:0" = PlaceHolder[target=sum_20]
#   %add_19 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=add_19]
#   %gt : Tensor "b8[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=gt]
#   %view_32 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_8, [128, 500, 128]), kwargs = {})
#   %permute_40 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_32, [1, 0, 2]), kwargs = {})
#   %mul_61 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_40, %primals_5), kwargs = {})
#   %mul_62 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_61, 128), kwargs = {})
#   %mul_63 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_61, %mul_2), kwargs = {})
#   %sum_20 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_63, [2], True), kwargs = {})
#   %mul_64 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %sum_20), kwargs = {})
#   %sub_15 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_62, %sum_19), kwargs = {})
#   %sub_16 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_15, %mul_64), kwargs = {})
#   %mul_65 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_4, %sub_16), kwargs = {})
#   %mul_66 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_40, %mul_2), kwargs = {})
#   %sum_21 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_66, [0, 1]), kwargs = {})
#   %sum_22 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%permute_40, [0, 1]), kwargs = {})
#   %add_19 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %mul_65), kwargs = {})
#   %convert_element_type_4 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%gt, torch.float32), kwargs = {})
#   %mul_67 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_4, 1.1111111111111112), kwargs = {})
#   %mul_68 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_19, %mul_67), kwargs = {})
#   return %sum_20,%add_19,%mul_68
triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_transpose_view_13 = async_compile.triton('triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_transpose_view_13', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*i1', 'out_ptr1': '*fp32', 'ws_ptr': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'RSPLIT_SIZE': 'constexpr', 'NUM_STAGES': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'MixOrderReductionGrid', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_transpose_view_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 8, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'RSPLIT_SIZE': 64}
)
@triton.jit
def triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_transpose_view_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, RSPLIT_SIZE : tl.constexpr, NUM_STAGES : tl.constexpr):
    xnumel = 64000
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
        x0 = (xindex % 128)
        x1 = xindex // 128
        x3 = xindex
        xindex += XBLOCK
        tmp0 = tl.load(in_ptr0 + (r0_2 + 128*x1 + 64000*x0), xmask, other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_2), None, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr2 + (r0_2 + 128*x3), xmask, other=0.0)
        tmp9 = tl.load(in_out_ptr0 + (r0_2 + 128*x3), xmask, other=0.0)
        tmp10 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
        tmp13 = tl.load(in_ptr4 + (x1 + 500*x0), xmask, eviction_policy='evict_last')
        tmp19 = tl.load(in_ptr5 + (r0_2 + 128*x3), xmask, other=0.0).to(tl.int1)
        tmp25 = tl.load(in_ptr0 + (r0_2 + 128*x3), xmask, other=0.0)
        tmp2 = tmp0 * tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = tl.where(xmask, tmp5, 0)
        tmp8 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
        tmp11 = 128.0
        tmp12 = tmp2 * tmp11
        tmp14 = tmp12 - tmp13
        tmp15 = tmp3 * tmp8
        tmp16 = tmp14 - tmp15
        tmp17 = tmp10 * tmp16
        tmp18 = tmp9 + tmp17
        tmp20 = tmp19.to(tl.float32)
        tmp21 = 1.1111111111111112
        tmp22 = tmp20 * tmp21
        tmp23 = tmp18 * tmp22
        tmp24 = tmp0 * tmp3
        tl.store(in_out_ptr0 + (r0_2 + 128*x3), tmp18, xmask)
        tl.store(out_ptr1 + (r0_2 + 128*x3), tmp23, xmask)
        tmp26 = tl.sum(tmp24, 0)
        tmp27 = accum0 + tmp26
        accum0 = tmp27
        tmp28 = tl.sum(tmp25, 0)
        tmp29 = accum1 + tmp28
        accum1 = tmp29
    tl.store(ws_ptr + (tl.program_id(0) + 0 * tl.num_programs(0)) * r0_numel + r0_index, accum0, r0_mask)
    tl.store(ws_ptr + (tl.program_id(0) + 1 * tl.num_programs(0)) * r0_numel + r0_index, accum1, r0_mask)
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
        primals_2, primals_5, primals_7, primals_9, primals_13, primals_15, primals_17, primals_19, primals_21, view, gt, mul_2, view_2, baddbmm, amax, sum_1, gt_1, view_10, addmm_1, getitem_3, rsqrt_1, gt_2, mul_11, view_13, addmm_2, gt_3, getitem_7, rsqrt_3, view_15, gt_4, div_2, permute_27, permute_28, permute_29, permute_30, div_4, tangents_1 = args
        args.clear()
        assert_size_stride(primals_2, (128, 512), (512, 1))
        assert_size_stride(primals_5, (128, ), (1, ))
        assert_size_stride(primals_7, (384, 128), (128, 1))
        assert_size_stride(primals_9, (128, 128), (128, 1))
        assert_size_stride(primals_13, (128, ), (1, ))
        assert_size_stride(primals_15, (128, ), (1, ))
        assert_size_stride(primals_17, (512, 128), (128, 1))
        assert_size_stride(primals_19, (512, ), (1, ))
        assert_size_stride(primals_21, (128, 512), (512, 1))
        assert_size_stride(view, (64000, 512), (512, 1))
        assert_size_stride(gt, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(mul_2, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(view_2, (64000, 128), (128, 1))
        assert_size_stride(baddbmm, (4000, 128, 128), (16384, 128, 1))
        assert_size_stride(amax, (4000, 128, 1), (128, 1, 1))
        assert_size_stride(sum_1, (4000, 128, 1), (128, 1, 1))
        assert_size_stride(gt_1, (4000, 128, 128), (16384, 128, 1))
        assert_size_stride(view_10, (64000, 128), (128, 1))
        assert_size_stride(addmm_1, (64000, 128), (128, 1))
        assert_size_stride(getitem_3, (500, 128, 1), (128, 1, 1))
        assert_size_stride(rsqrt_1, (500, 128, 1), (128, 1, 1))
        assert_size_stride(gt_2, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(mul_11, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(view_13, (64000, 128), (128, 1))
        assert_size_stride(addmm_2, (64000, 512), (512, 1))
        assert_size_stride(gt_3, (500, 128, 512), (65536, 512, 1))
        assert_size_stride(getitem_7, (500, 128, 1), (128, 1, 1))
        assert_size_stride(rsqrt_3, (500, 128, 1), (128, 1, 1))
        assert_size_stride(view_15, (64000, 512), (512, 1))
        assert_size_stride(gt_4, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(div_2, (500, 128, 1), (128, 1, 1))
        assert_size_stride(permute_27, (4000, 128, 128), (16384, 1, 128))
        assert_size_stride(permute_28, (4000, 16, 128), (16, 1, 64000))
        assert_size_stride(permute_29, (4000, 128, 16), (16, 64000, 1))
        assert_size_stride(permute_30, (4000, 16, 128), (16, 1, 64000))
        assert_size_stride(div_4, (500, 128, 1), (128, 1, 1))
        assert_size_stride(tangents_1, (500, 128, 128), (16384, 128, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.float32)
            buf3 = empty_strided_cuda((1, 128, 500), (64000, 1, 128), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type, mul_22, clone_5, view_17, sum_2], Original ATen: [aten.native_dropout_backward, aten.view, aten.sum]
            # [Provenance debug handles] triton_red_fused_native_dropout_backward_sum_view_0:92
            stream0 = get_raw_stream(0)
            triton_red_fused_native_dropout_backward_sum_view_0.run(tangents_1, gt_4, buf0, buf3, 64000, 128, stream=stream0)
            del gt_4
            buf1 = empty_strided_cuda((64000, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type, mul_22, clone_5, view_17, linear_2, permute_13, mm_1], Original ATen: [aten.native_dropout_backward, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:93
            extern_kernels.mm(reinterpret_tensor(buf0, (64000, 128), (128, 1), 0), primals_21, out=buf1)
            del primals_21
            buf2 = empty_strided_cuda((128, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type, mul_22, clone_5, view_17, permute_14, permute_16], Original ATen: [aten.native_dropout_backward, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:94
            extern_kernels.mm(reinterpret_tensor(buf0, (128, 64000), (1, 128), 0), view_15, out=buf2)
            del buf0
            del view_15
            buf4 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type, mul_22, clone_5, view_17, sum_2], Original ATen: [aten.native_dropout_backward, aten.view, aten.sum]
            # [Provenance debug handles] triton_red_fused_native_dropout_backward_sum_view_1:95
            stream0 = get_raw_stream(0)
            triton_red_fused_native_dropout_backward_sum_view_1.run(buf3, buf4, 128, 500, stream=stream0)
            del buf3
            buf7 = empty_strided_cuda((500, 128, 512), (65536, 512, 1), torch.float32)
            buf12 = buf7; del buf7  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [view_19, mul_25, mul_26, sum_3, linear_1, gelu, dropout_2, layer_norm_3, mul_27, sum_4, mul_28, sub_6, sub_7, div_1, mul_29, mul_30, sum_5, sum_6, convert_element_type_1, mul_31, clone_6, mul_34, mul_35, mul_36, exp_1, mul_37, mul_38, add_15, mul_39], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.gelu, aten.native_dropout, aten.native_layer_norm, aten.native_dropout_backward, aten.gelu_backward]
            workspace_0 = empty_strided_cuda((1024000, ), (1, ), torch.float32)
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_gelu_backward_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_view_2.run(buf12, buf1, primals_19, gt_3, addmm_2, getitem_7, rsqrt_3, workspace_0, 64000, 512, stream=stream0)
            buf9 = workspace_0[0 * 1000 * 512 : (0 + 1) * 1000 * 512].view(1000, 512).sum(dim=0)
            buf11 = workspace_0[1 * 1000 * 512 : (1 + 1) * 1000 * 512].view(1000, 512).sum(dim=0)
            del workspace_0
            del addmm_2
            del buf1
            del getitem_7
            del gt_3
            del primals_19
            del rsqrt_3
            buf13 = empty_strided_cuda((64000, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_1, gelu, convert_element_type_1, mul_31, clone_6, mul_34, mul_35, mul_36, exp_1, mul_37, mul_38, add_15, mul_39, view_20, permute_17, mm_3], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:96
            extern_kernels.mm(reinterpret_tensor(buf12, (64000, 512), (512, 1), 0), primals_17, out=buf13)
            del primals_17
            buf14 = empty_strided_cuda((512, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_1, gelu, convert_element_type_1, mul_31, clone_6, mul_34, mul_35, mul_36, exp_1, mul_37, mul_38, add_15, mul_39, view_20, permute_18, permute_20], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:97
            extern_kernels.mm(reinterpret_tensor(buf12, (512, 64000), (1, 512), 0), view_13, out=buf14)
            del view_13
            buf15 = empty_strided_cuda((1, 512, 200), (102400, 1, 512), torch.float32)
            # Topologically Sorted Source Nodes: [linear_1, gelu, convert_element_type_1, mul_31, clone_6, mul_34, mul_35, mul_36, exp_1, mul_37, mul_38, add_15, mul_39, view_20, sum_7], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.sum]
            # [Provenance debug handles] triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_3:98
            stream0 = get_raw_stream(0)
            triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_3.run(buf12, buf15, 102400, 320, stream=stream0)
            del buf12
            buf16 = empty_strided_cuda((1, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_1, gelu, convert_element_type_1, mul_31, clone_6, mul_34, mul_35, mul_36, exp_1, mul_37, mul_38, add_15, mul_39, view_20, sum_7], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.sum]
            # [Provenance debug handles] triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_4:99
            stream0 = get_raw_stream(0)
            triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_4.run(buf15, buf16, 512, 200, stream=stream0)
            del buf15
            buf23 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.float32)
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [view_22, mul_41, mul_42, sum_8, mul_43, sum_9, mul_44, sub_9, sub_10, mul_45, mul_46, sum_10, sum_11, add_16], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            workspace_1 = empty_strided_cuda((256000, ), (1, ), torch.float32)
            stream0 = get_raw_stream(0)
            triton_per_fused_add_native_layer_norm_backward_view_5.run(buf13, primals_15, mul_11, tangents_1, div_2, buf23, workspace_1, 64000, 128, stream=stream0)
            buf20 = workspace_1[0 * 1000 * 128 : (0 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            buf22 = workspace_1[1 * 1000 * 128 : (1 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            del div_2
            del mul_11
            del primals_15
            del tangents_1
            buf30 = reinterpret_tensor(buf13, (128, 500, 128), (64000, 128, 1), 0); del buf13  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [convert_element_type_2, mul_47, clone_7, multi_head_attention_forward, transpose_1, sub_11, mul_49, mul_50, mul_51, sum_12, mul_52, sum_13, mul_53, sub_12, sub_13, div_3, mul_54, mul_55, sum_14, sum_15, permute_21, clone_8], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone]
            workspace_2 = workspace_1; del workspace_1  # reuse
            stream0 = get_raw_stream(0)
            triton_per_fused_clone_native_dropout_backward_native_layer_norm_backward_transpose_view_6.run(buf23, gt_2, primals_13, addmm_1, getitem_3, rsqrt_1, buf30, workspace_2, 64000, 128, stream=stream0)
            buf27 = workspace_2[0 * 1000 * 128 : (0 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            buf29 = workspace_2[1 * 1000 * 128 : (1 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            del addmm_1
            del getitem_3
            del gt_2
            del primals_13
            del rsqrt_1
            buf31 = empty_strided_cuda((64000, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2, mul_47, clone_7, multi_head_attention_forward, transpose_1, sub_11, mul_49, mul_50, mul_51, mul_53, sub_12, sub_13, div_3, mul_54, permute_21, clone_8, view_23, permute_22, mm_5], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:100
            extern_kernels.mm(reinterpret_tensor(buf30, (64000, 128), (128, 1), 0), primals_9, out=buf31)
            del primals_9
            buf32 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2, mul_47, clone_7, multi_head_attention_forward, transpose_1, sub_11, mul_49, mul_50, mul_51, mul_53, sub_12, sub_13, div_3, mul_54, permute_21, clone_8, view_23, permute_23, permute_25], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:101
            extern_kernels.mm(reinterpret_tensor(buf30, (128, 64000), (1, 128), 0), view_10, out=buf32)
            del view_10
            buf33 = empty_strided_cuda((1, 128, 500), (64000, 1, 128), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2, mul_47, clone_7, multi_head_attention_forward, transpose_1, sub_11, mul_49, mul_50, mul_51, mul_53, sub_12, sub_13, div_3, mul_54, permute_21, clone_8, view_23, sum_16], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_7:102
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_7.run(buf30, buf33, 64000, 128, stream=stream0)
            buf34 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2, mul_47, clone_7, multi_head_attention_forward, transpose_1, sub_11, mul_49, mul_50, mul_51, mul_53, sub_12, sub_13, div_3, mul_54, permute_21, clone_8, view_23, sum_16], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.sum]
            # [Provenance debug handles] triton_red_fused_native_dropout_backward_sum_view_1:103
            stream0 = get_raw_stream(0)
            triton_red_fused_native_dropout_backward_sum_view_1.run(buf33, buf34, 128, 500, stream=stream0)
            buf35 = reinterpret_tensor(buf30, (4000, 128, 16), (2048, 16, 1), 0); del buf30  # reuse
            # Topologically Sorted Source Nodes: [view_25, permute_26, bmm_1], Original ATen: [aten.view, aten.transpose, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:104
            extern_kernels.bmm(permute_27, reinterpret_tensor(buf31, (4000, 128, 16), (16, 64000, 1), 0), out=buf35)
            del permute_27
            buf36 = empty_strided_cuda((4000, 128, 128), (16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_25, permute_26, bmm_2], Original ATen: [aten.view, aten.transpose, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:105
            extern_kernels.bmm(reinterpret_tensor(buf31, (4000, 128, 16), (16, 64000, 1), 0), permute_28, out=buf36)
            del permute_28
            buf37 = buf36; del buf36  # reuse
            buf39 = baddbmm; del baddbmm  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_3, mul_56, clone_9, multi_head_attention_forward, mul_58, sum_17, neg, fma], Original ATen: [aten.native_dropout_backward, aten._softmax, aten._softmax_backward_data]
            # [Provenance debug handles] triton_per_fused__softmax__softmax_backward_data_native_dropout_backward_8:106
            stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data_native_dropout_backward_8.run(buf37, buf39, gt_1, amax, sum_1, 512000, 128, stream=stream0)
            del amax
            del buf37
            del gt_1
            del sum_1
            buf40 = reinterpret_tensor(buf31, (4000, 128, 16), (2048, 16, 1), 0); del buf31  # reuse
            # Topologically Sorted Source Nodes: [bmm_3], Original ATen: [aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:107
            extern_kernels.bmm(buf39, permute_29, out=buf40)
            del permute_29
            buf41 = empty_strided_cuda((4000, 16, 128), (2048, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [bmm_4], Original ATen: [aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:108
            extern_kernels.bmm(permute_30, buf39, out=buf41)
            del permute_30
            buf42 = empty_strided_cuda((1, 1, 384, 320), (122880, 122880, 1, 384), torch.float32)
            # Topologically Sorted Source Nodes: [permute_31, mul_59, permute_32, clone_10, view_26, permute_33, view_27, permute_34, clone_11, view_28, full, _generalized_scatter, _generalized_scatter_1, add_17, _generalized_scatter_2, add_18, unsqueeze_1, permute_35, squeeze_1, clone_12, view_29, sum_18], Original ATen: [aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_9:109
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf35, buf41, buf40, buf42, 122880, 200, stream=stream0)
            buf43 = empty_strided_cuda((1, 1, 384), (384, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_31, mul_59, permute_32, clone_10, view_26, permute_33, view_27, permute_34, clone_11, view_28, full, _generalized_scatter, _generalized_scatter_1, add_17, _generalized_scatter_2, add_18, unsqueeze_1, permute_35, squeeze_1, clone_12, view_29, sum_18], Original ATen: [aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_10:110
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_10.run(buf42, buf43, 384, 320, stream=stream0)
            del buf42
            buf44 = empty_strided_cuda((128, 500, 3, 128), (192000, 384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_31, mul_59, permute_32, clone_10, view_26, permute_33, view_27, permute_34, clone_11, view_28, full, _generalized_scatter, _generalized_scatter_1, add_17, _generalized_scatter_2, add_18, unsqueeze_1, permute_35, squeeze_1, clone_12], Original ATen: [aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_clone_mul_select_backward_squeeze_transpose_unsqueeze_view_11:111
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_mul_select_backward_squeeze_transpose_unsqueeze_view_11.run(buf35, buf41, buf40, buf44, 64000, 384, stream=stream0)
            del buf35
            buf45 = empty_strided_cuda((384, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_31, mul_59, permute_32, clone_10, view_26, permute_33, view_27, permute_34, clone_11, view_28, full, _generalized_scatter, _generalized_scatter_1, add_17, _generalized_scatter_2, add_18, unsqueeze_1, permute_35, squeeze_1, clone_12, view_29, view_31, permute_36, permute_39], Original ATen: [aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:112
            extern_kernels.mm(reinterpret_tensor(buf44, (384, 64000), (1, 384), 0), view_2, out=buf45)
            del view_2
            buf46 = reinterpret_tensor(buf41, (64000, 128), (128, 1), 0); del buf41  # reuse
            # Topologically Sorted Source Nodes: [permute_31, mul_59, permute_32, clone_10, view_26, permute_33, view_27, permute_34, clone_11, view_28, full, _generalized_scatter, _generalized_scatter_1, add_17, _generalized_scatter_2, add_18, unsqueeze_1, permute_35, squeeze_1, clone_12, view_29, view_31, multi_head_attention_forward, permute_38, mm_8], Original ATen: [aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:113
            extern_kernels.mm(reinterpret_tensor(buf44, (64000, 384), (384, 1), 0), primals_7, out=buf46)
            del buf44
            del primals_7
            buf47 = reinterpret_tensor(buf33, (500, 128, 1), (1, 500, 64000), 0); del buf33  # reuse
            # Topologically Sorted Source Nodes: [view_32, permute_40, mul_61, sum_19], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_backward_transpose_view_12:114
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_backward_transpose_view_12.run(buf46, primals_5, buf47, 64000, 128, stream=stream0)
            buf53 = buf23; del buf23  # reuse
            buf54 = reinterpret_tensor(buf40, (500, 128, 128), (16384, 128, 1), 0); del buf40  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [view_32, permute_40, mul_61, mul_62, mul_63, sum_20, mul_64, sub_15, sub_16, mul_65, mul_66, sum_21, sum_22, add_19, convert_element_type_4, mul_67, clone_13], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward, aten.add, aten.native_dropout_backward]
            workspace_3 = workspace_2; del workspace_2  # reuse
            stream0 = get_raw_stream(0)
            triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_transpose_view_13.run(buf53, buf46, primals_5, mul_2, div_4, buf47, gt, buf54, workspace_3, 64000, 128, stream=stream0)
            buf50 = workspace_3[0 * 1000 * 128 : (0 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            buf52 = workspace_3[1 * 1000 * 128 : (1 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            del workspace_3
            del buf46
            del div_4
            del gt
            del mul_2
            del primals_5
            buf55 = empty_strided_cuda((64000, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4, mul_67, clone_13, view_33, linear, permute_41, mm_9], Original ATen: [aten.native_dropout_backward, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:115
            extern_kernels.mm(reinterpret_tensor(buf54, (64000, 128), (128, 1), 0), primals_2, out=buf55)
            del primals_2
            buf56 = empty_strided_cuda((128, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4, mul_67, clone_13, view_33, permute_42, permute_44], Original ATen: [aten.native_dropout_backward, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:116
            extern_kernels.mm(reinterpret_tensor(buf54, (128, 64000), (1, 128), 0), view, out=buf56)
            del view
            buf57 = reinterpret_tensor(buf47, (1, 128, 500), (64000, 1, 128), 0); del buf47  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_4, mul_67, clone_13, view_33, sum_23], Original ATen: [aten.native_dropout_backward, aten.view, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_7:117
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_7.run(buf54, buf57, 64000, 128, stream=stream0)
            del buf54
            buf58 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4, mul_67, clone_13, view_33, sum_23], Original ATen: [aten.native_dropout_backward, aten.view, aten.sum]
            # [Provenance debug handles] triton_red_fused_native_dropout_backward_sum_view_1:118
            stream0 = get_raw_stream(0)
            triton_red_fused_native_dropout_backward_sum_view_1.run(buf57, buf58, 128, 500, stream=stream0)
            del buf57
        return (reinterpret_tensor(buf55, (500, 128, 512), (65536, 512, 1), 0), buf56, reinterpret_tensor(buf58, (128, ), (1, ), 0), buf53, buf50, buf52, buf45, reinterpret_tensor(buf43, (384, ), (1, ), 0), buf32, reinterpret_tensor(buf34, (128, ), (1, ), 0), None, buf39, buf27, buf29, buf20, buf22, buf14, reinterpret_tensor(buf16, (512, ), (1, ), 0), buf9, buf11, buf2, reinterpret_tensor(buf4, (128, ), (1, ), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_2 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view = rand_strided((64000, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    gt = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bool)
    mul_2 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    view_2 = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    baddbmm = rand_strided((4000, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax = rand_strided((4000, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_1 = rand_strided((4000, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_1 = rand_strided((4000, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_10 = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_1 = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_1 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_2 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bool)
    mul_11 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    view_13 = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_2 = rand_strided((64000, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    gt_3 = rand_strided((500, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_7 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_3 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_15 = rand_strided((64000, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    gt_4 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bool)
    div_2 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_27 = rand_strided((4000, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_28 = rand_strided((4000, 16, 128), (16, 1, 64000), device='cuda:0', dtype=torch.float32)
    permute_29 = rand_strided((4000, 128, 16), (16, 64000, 1), device='cuda:0', dtype=torch.float32)
    permute_30 = rand_strided((4000, 16, 128), (16, 1, 64000), device='cuda:0', dtype=torch.float32)
    div_4 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_2, primals_5, primals_7, primals_9, primals_13, primals_15, primals_17, primals_19, primals_21, view, gt, mul_2, view_2, baddbmm, amax, sum_1, gt_1, view_10, addmm_1, getitem_3, rsqrt_1, gt_2, mul_11, view_13, addmm_2, gt_3, getitem_7, rsqrt_3, view_15, gt_4, div_2, permute_27, permute_28, permute_29, permute_30, div_4, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
