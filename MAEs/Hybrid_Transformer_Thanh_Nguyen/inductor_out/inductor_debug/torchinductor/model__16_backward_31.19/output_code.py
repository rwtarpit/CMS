# AOT ID: ['16_backward']
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


# kernel path: /traces/inductor_cache/yi/cyibqgsru4xuygvpzfrflynusisnynhjnq3bewzqaej53slqcijo.py
# Topologically Sorted Source Nodes: [linear, gelu, layer_norm_3, mul_18, sum_4, sum_5], Original ATen: [aten.view, aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   gelu => add_9, erf, mul_7, mul_8, mul_9
#   layer_norm_3 => mul_10, sub_4
#   linear => view_14
#   mul_18 => mul_18
#   sum_4 => sum_4
#   sum_5 => sum_5
# Graph fragment:
#   %tangents_1 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0" = PlaceHolder[target=tangents_1]
#   %addmm_1 : Tensor "f32[500, 512][512, 1]cuda:0" = PlaceHolder[target=addmm_1]
#   %getitem_11 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=getitem_11]
#   %rsqrt_3 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_3]
#   %view_14 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_1, [500, 1, 512]), kwargs = {})
#   %mul_7 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, 0.5), kwargs = {})
#   %mul_8 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_8,), kwargs = {})
#   %add_9 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_9 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %add_9), kwargs = {})
#   %sub_4 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_9, %getitem_11), kwargs = {})
#   %mul_10 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_3), kwargs = {})
#   %mul_18 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %mul_10), kwargs = {})
#   %sum_4 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_18, [0, 1]), kwargs = {})
#   %sum_5 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%tangents_1, [0, 1]), kwargs = {})
#   return %buf2,%buf4
triton_red_fused_gelu_native_layer_norm_native_layer_norm_backward_view_0 = async_compile.triton('triton_red_fused_gelu_native_layer_norm_native_layer_norm_backward_view_0', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_gelu_native_layer_norm_native_layer_norm_backward_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 2, 'num_reduction': 2, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2080768, 'r0_': 4000}}
)
@triton.jit
def triton_red_fused_gelu_native_layer_norm_native_layer_norm_backward_view_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /traces/inductor_cache/gr/cgrkmotfulntunbeoeststenkm5fevkjppqywy6ev2tfdxke4m6c.py
# Topologically Sorted Source Nodes: [linear, gelu, layer_norm_3, mul_18, sum_4], Original ATen: [aten.view, aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   gelu => add_9, erf, mul_7, mul_8, mul_9
#   layer_norm_3 => mul_10, sub_4
#   linear => view_14
#   mul_18 => mul_18
#   sum_4 => sum_4
# Graph fragment:
#   %buf2 : Tensor "f32[512, 4][1, 512]cuda:0" = PlaceHolder[target=buf2]
#   %view_14 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_1, [500, 1, 512]), kwargs = {})
#   %mul_7 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, 0.5), kwargs = {})
#   %mul_8 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_8,), kwargs = {})
#   %add_9 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_9 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %add_9), kwargs = {})
#   %sub_4 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_9, %getitem_11), kwargs = {})
#   %mul_10 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_3), kwargs = {})
#   %mul_18 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %mul_10), kwargs = {})
#   %sum_4 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_18, [0, 1]), kwargs = {})
#   return %sum_4
triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_1 = async_compile.triton('triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_1', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 12288, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /traces/inductor_cache/wb/cwb7jdeuadeymdmcgik2mcyzaeu5swlistipgla4rkbuchkl5vqj.py
# Topologically Sorted Source Nodes: [mul_13, mul_14, sum_2, linear, gelu, layer_norm_3, mul_15, sum_3, mul_16, sub_6, sub_7, div_1, mul_17, mul_20, mul_21, mul_22, exp_1, mul_23, mul_24, add_13, mul_25], Original ATen: [aten.native_layer_norm_backward, aten.view, aten.gelu, aten.native_layer_norm, aten.gelu_backward]
# Source node to ATen node mapping:
#   add_13 => add_13
#   div_1 => div_1
#   exp_1 => exp_1
#   gelu => add_9, erf, mul_7, mul_8, mul_9
#   layer_norm_3 => mul_10, sub_4
#   linear => view_14
#   mul_13 => mul_13
#   mul_14 => mul_14
#   mul_15 => mul_15
#   mul_16 => mul_16
#   mul_17 => mul_17
#   mul_20 => mul_20
#   mul_21 => mul_21
#   mul_22 => mul_22
#   mul_23 => mul_23
#   mul_24 => mul_24
#   mul_25 => mul_25
#   sub_6 => sub_6
#   sub_7 => sub_7
#   sum_2 => sum_2
#   sum_3 => sum_3
# Graph fragment:
#   %tangents_1 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0" = PlaceHolder[target=tangents_1]
#   %primals_16 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_16]
#   %addmm_1 : Tensor "f32[500, 512][512, 1]cuda:0" = PlaceHolder[target=addmm_1]
#   %getitem_11 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=getitem_11]
#   %rsqrt_3 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_3]
#   %sum_2 : Tensor "f32[500, 1, 1][1, 500, 500]cuda:0" = PlaceHolder[target=sum_2]
#   %sum_3 : Tensor "f32[500, 1, 1][1, 500, 500]cuda:0" = PlaceHolder[target=sum_3]
#   %mul_13 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %primals_16), kwargs = {})
#   %mul_14 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, 512), kwargs = {})
#   %sum_2 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_13, [2], True), kwargs = {})
#   %view_14 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_1, [500, 1, 512]), kwargs = {})
#   %mul_7 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, 0.5), kwargs = {})
#   %mul_8 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_8,), kwargs = {})
#   %add_9 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_9 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %add_9), kwargs = {})
#   %sub_4 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_9, %getitem_11), kwargs = {})
#   %mul_10 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_3), kwargs = {})
#   %mul_15 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %mul_10), kwargs = {})
#   %sum_3 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_15, [2], True), kwargs = {})
#   %mul_16 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %sum_3), kwargs = {})
#   %sub_6 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_14, %sum_2), kwargs = {})
#   %sub_7 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_6, %mul_16), kwargs = {})
#   %div_1 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_3, 512), kwargs = {})
#   %mul_17 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %sub_7), kwargs = {})
#   %mul_20 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_9, 0.5), kwargs = {})
#   %mul_21 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, %view_14), kwargs = {})
#   %mul_22 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, -0.5), kwargs = {})
#   %exp_1 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_22,), kwargs = {})
#   %mul_23 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_1, 0.3989422804014327), kwargs = {})
#   %mul_24 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, %mul_23), kwargs = {})
#   %add_13 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %mul_24), kwargs = {})
#   %mul_25 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_17, %add_13), kwargs = {})
#   return %sum_2,%sum_3,%mul_25
triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_view_2 = async_compile.triton('triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_view_2', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_view_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 2, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4000, 'r0_': 4098048}}
)
@triton.jit
def triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_view_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_out_ptr0 + (r0_1 + 512*x0), xmask, other=0.0)
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


# kernel path: /traces/inductor_cache/yy/cyykplodxhjt55wfykwxjjwn7e2xmd2axjcn74z4rfn6ht6itje2.py
# Topologically Sorted Source Nodes: [view_15, sum_6], Original ATen: [aten.view, aten.sum]
# Source node to ATen node mapping:
#   sum_6 => sum_6
#   view_15 => view_15
# Graph fragment:
#   %mul_25 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0" = PlaceHolder[target=mul_25]
#   %view_15 : Tensor "f32[500, 512][512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_25, [500, 512]), kwargs = {})
#   %sum_6 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_15, [0], True), kwargs = {})
#   return %buf9
triton_red_fused_sum_view_3 = async_compile.triton('triton_red_fused_sum_view_3', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1040384, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_sum_view_3(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /traces/inductor_cache/la/clava6hog7b5bzv4u37iyapiktjoreqepwvxjiqcebst5jcg2ogw.py
# Topologically Sorted Source Nodes: [view_17, layer_norm_2, mul_32, sum_9, sum_10], Original ATen: [aten.view, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_2 => mul_5, sub_3
#   mul_32 => mul_32
#   sum_10 => sum_10
#   sum_9 => sum_9
#   view_17 => view_17
# Graph fragment:
#   %mm_2 : Tensor "f32[500, 128][128, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %add_6 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0" = PlaceHolder[target=add_6]
#   %getitem_9 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=getitem_9]
#   %rsqrt_2 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_2]
#   %view_17 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [500, 1, 128]), kwargs = {})
#   %sub_3 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_6, %getitem_9), kwargs = {})
#   %mul_5 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %mul_32 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_17, %mul_5), kwargs = {})
#   %sum_9 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_32, [0, 1]), kwargs = {})
#   %sum_10 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_17, [0, 1]), kwargs = {})
#   return %buf13,%buf15
triton_red_fused_native_layer_norm_native_layer_norm_backward_view_4 = async_compile.triton('triton_red_fused_native_layer_norm_native_layer_norm_backward_view_4', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 2, 'num_reduction': 2, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 520192, 'r0_': 4000}}
)
@triton.jit
def triton_red_fused_native_layer_norm_native_layer_norm_backward_view_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /traces/inductor_cache/i3/ci346bdcjfpxvi7mwramh7wqqw3l4n44ancnzn2d2lkq6erc6pyx.py
# Topologically Sorted Source Nodes: [view_17, mul_27, mul_28, sum_7, layer_norm_2, mul_29, sum_8, mul_30, sub_9, sub_10, div_2, mul_31, add_14, mul_34, mul_35, sum_11, multi_head_attention_forward, transpose_2, layer_norm_1, mul_36, sum_12, mul_37, sub_12, sub_13, div_3, mul_38], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.native_layer_norm, aten.add, aten.transpose]
# Source node to ATen node mapping:
#   add_14 => add_14
#   div_2 => div_2
#   div_3 => div_3
#   layer_norm_1 => mul_3, sub_2
#   layer_norm_2 => mul_5, sub_3
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
#   multi_head_attention_forward => view_11
#   sub_10 => sub_10
#   sub_12 => sub_12
#   sub_13 => sub_13
#   sub_9 => sub_9
#   sum_11 => sum_11
#   sum_12 => sum_12
#   sum_7 => sum_7
#   sum_8 => sum_8
#   transpose_2 => permute_11
#   view_17 => view_17
# Graph fragment:
#   %mm_2 : Tensor "f32[500, 128][128, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %primals_12 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_12]
#   %add_6 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0" = PlaceHolder[target=add_6]
#   %getitem_9 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=getitem_9]
#   %rsqrt_2 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_2]
#   %tangents_2 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0" = PlaceHolder[target=tangents_2]
#   %sum_7 : Tensor "f32[500, 1, 1][1, 500, 500]cuda:0" = PlaceHolder[target=sum_7]
#   %sum_8 : Tensor "f32[500, 1, 1][1, 500, 500]cuda:0" = PlaceHolder[target=sum_8]
#   %add_14 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0" = PlaceHolder[target=add_14]
#   %primals_10 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_10]
#   %addmm : Tensor "f32[500, 128][128, 1]cuda:0" = PlaceHolder[target=addmm]
#   %getitem_7 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=getitem_7]
#   %rsqrt_1 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_1]
#   %sum_11 : Tensor "f32[500, 1, 1][1, 500, 500]cuda:0" = PlaceHolder[target=sum_11]
#   %sum_12 : Tensor "f32[500, 1, 1][1, 500, 500]cuda:0" = PlaceHolder[target=sum_12]
#   %view_17 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [500, 1, 128]), kwargs = {})
#   %mul_27 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_17, %primals_12), kwargs = {})
#   %mul_28 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_27, 128), kwargs = {})
#   %sum_7 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_27, [2], True), kwargs = {})
#   %sub_3 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_6, %getitem_9), kwargs = {})
#   %mul_5 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %mul_29 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_27, %mul_5), kwargs = {})
#   %sum_8 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_29, [2], True), kwargs = {})
#   %mul_30 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %sum_8), kwargs = {})
#   %sub_9 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_28, %sum_7), kwargs = {})
#   %sub_10 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_9, %mul_30), kwargs = {})
#   %div_2 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_2, 128), kwargs = {})
#   %mul_31 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %sub_10), kwargs = {})
#   %add_14 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_2, %mul_31), kwargs = {})
#   %mul_34 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_14, %primals_10), kwargs = {})
#   %mul_35 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, 128), kwargs = {})
#   %sum_11 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_34, [2], True), kwargs = {})
#   %view_11 : Tensor "f32[1, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [1, 500, 128]), kwargs = {})
#   %permute_11 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_11, [1, 0, 2]), kwargs = {})
#   %sub_2 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_11, %getitem_7), kwargs = {})
#   %mul_3 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_36 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %mul_3), kwargs = {})
#   %sum_12 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_36, [2], True), kwargs = {})
#   %mul_37 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %sum_12), kwargs = {})
#   %sub_12 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_35, %sum_11), kwargs = {})
#   %sub_13 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_12, %mul_37), kwargs = {})
#   %div_3 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_1, 128), kwargs = {})
#   %mul_38 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, %sub_13), kwargs = {})
#   return %sum_7,%sum_8,%add_14,%sum_11,%sum_12,%mul_38
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_5 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 10, 'num_store': 2, 'num_reduction': 4, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 8000, 'r0_': 2049024}}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp8 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (r0_1 + 128*x0), xmask, other=0.0)
    tmp27 = tl.load(in_ptr5 + (r0_1), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr6 + (r0_1 + 128*x0), xmask, other=0.0)
    tmp34 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = tmp9 * tmp10
    tmp12 = tmp2 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None].to(tl.float32)
    tmp18 = 0.0078125
    tmp19 = tmp10 * tmp18
    tmp20 = 128.0
    tmp21 = tmp2 * tmp20
    tmp22 = tmp21 - tmp6
    tmp23 = tmp11 * tmp16
    tmp24 = tmp22 - tmp23
    tmp25 = tmp19 * tmp24
    tmp26 = tmp17 + tmp25
    tmp28 = tmp26 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, R0_BLOCK])
    tmp31 = tl.where(xmask, tmp29, 0)
    tmp32 = tl.sum(tmp31, 1)[:, None].to(tl.float32)
    tmp35 = tmp33 - tmp34
    tmp37 = tmp35 * tmp36
    tmp38 = tmp28 * tmp37
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK, R0_BLOCK])
    tmp41 = tl.where(xmask, tmp39, 0)
    tmp42 = tl.sum(tmp41, 1)[:, None].to(tl.float32)
    tmp43 = tmp36 * tmp18
    tmp44 = tmp28 * tmp20
    tmp45 = tmp44 - tmp32
    tmp46 = tmp37 * tmp42
    tmp47 = tmp45 - tmp46
    tmp48 = tmp43 * tmp47
    tl.store(in_out_ptr0 + (r0_1 + 128*x0), tmp26, xmask)
    tl.store(out_ptr4 + (r0_1 + 128*x0), tmp48, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/lx/clxclsu4elrwdoycbbftahzh3hye5jlbx7kuuafcawjeybnhzqyz.py
# Topologically Sorted Source Nodes: [view_17, layer_norm_2, mul_32, sum_9], Original ATen: [aten.view, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_2 => mul_5, sub_3
#   mul_32 => mul_32
#   sum_9 => sum_9
#   view_17 => view_17
# Graph fragment:
#   %buf13 : Tensor "f32[128, 4][1, 128]cuda:0" = PlaceHolder[target=buf13]
#   %view_17 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [500, 1, 128]), kwargs = {})
#   %sub_3 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_6, %getitem_9), kwargs = {})
#   %mul_5 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %mul_32 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_17, %mul_5), kwargs = {})
#   %sum_9 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_32, [0, 1]), kwargs = {})
#   return %sum_9
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_6 = async_compile.triton('triton_per_fused_native_layer_norm_native_layer_norm_backward_view_6', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 3072, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_layer_norm_native_layer_norm_backward_view_6(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /traces/inductor_cache/pj/cpjf2quspgicsjrtlyifzrkq4rv6cxo4p6lyaysdwf6q6dqqgvou.py
# Topologically Sorted Source Nodes: [mul_34, mul_35, multi_head_attention_forward, transpose_2, layer_norm_1, mul_37, sub_12, sub_13, div_3, mul_38, permute_17, view_18, sum_15], Original ATen: [aten.native_layer_norm_backward, aten.view, aten.transpose, aten.native_layer_norm, aten.sum]
# Source node to ATen node mapping:
#   div_3 => div_3
#   layer_norm_1 => mul_3, sub_2
#   mul_34 => mul_34
#   mul_35 => mul_35
#   mul_37 => mul_37
#   mul_38 => mul_38
#   multi_head_attention_forward => view_11
#   permute_17 => permute_17
#   sub_12 => sub_12
#   sub_13 => sub_13
#   sum_15 => sum_15
#   transpose_2 => permute_11
#   view_18 => view_18
# Graph fragment:
#   %mul_38 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0" = PlaceHolder[target=mul_38]
#   %mul_34 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_14, %primals_10), kwargs = {})
#   %mul_35 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, 128), kwargs = {})
#   %view_11 : Tensor "f32[1, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [1, 500, 128]), kwargs = {})
#   %permute_11 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_11, [1, 0, 2]), kwargs = {})
#   %sub_2 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_11, %getitem_7), kwargs = {})
#   %mul_3 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_37 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %sum_12), kwargs = {})
#   %sub_12 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_35, %sum_11), kwargs = {})
#   %sub_13 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_12, %mul_37), kwargs = {})
#   %div_3 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_1, 128), kwargs = {})
#   %mul_38 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, %sub_13), kwargs = {})
#   %permute_17 : Tensor "f32[1, 500, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_38, [1, 0, 2]), kwargs = {})
#   %view_18 : Tensor "f32[500, 128][128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_17, [500, 128]), kwargs = {})
#   %sum_15 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_18, [0], True), kwargs = {})
#   return %buf27
triton_red_fused_native_layer_norm_native_layer_norm_backward_sum_transpose_view_7 = async_compile.triton('triton_red_fused_native_layer_norm_native_layer_norm_backward_sum_transpose_view_7', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_sum_transpose_view_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 260096, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_native_layer_norm_native_layer_norm_backward_sum_transpose_view_7(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /traces/inductor_cache/vi/cvi6grxstqnkrur3aqodut7gkx7xzeth5fb3uac44cn2d7n67uen.py
# Topologically Sorted Source Nodes: [mul_40, sum_16, neg, fma], Original ATen: [aten._softmax_backward_data]
# Source node to ATen node mapping:
#   fma => fma
#   mul_40 => mul_40
#   neg => neg
#   sum_16 => sum_16
# Graph fragment:
#   %bmm_2 : Tensor "f32[4000, 1, 129][129, 129, 1]cuda:0" = PlaceHolder[target=bmm_2]
#   %div : Tensor "f32[4000, 1, 129][129, 129, 1]cuda:0" = PlaceHolder[target=div]
#   %sum_16 : Tensor "f32[4000, 1, 1][1, 4000, 4000]cuda:0" = PlaceHolder[target=sum_16]
#   %mul_40 : Tensor "f32[4000, 1, 129][129, 129, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_2, %div), kwargs = {})
#   %sum_16 : Tensor "f32[4000, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_40, [-1], True), kwargs = {})
#   %neg : Tensor "f32[4000, 1, 129][129, 129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%div,), kwargs = {})
#   %fma : Tensor "f32[4000, 1, 129][129, 129, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.fma.default](args = (%neg, %sum_16, %mul_40), kwargs = {})
#   return %sum_16,%fma
triton_per_fused__softmax_backward_data_8 = async_compile.triton('triton_per_fused__softmax_backward_data_8', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 8256000}}
)
@triton.jit
def triton_per_fused__softmax_backward_data_8(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /traces/inductor_cache/ae/cae3cjpxme3jtdksbt2rimbstrv7cg3msh7aela7qbmzhzwvb6g6.py
# Topologically Sorted Source Nodes: [mul_41, permute_30, view_23, sum_18], Original ATen: [aten.mul, aten.transpose, aten.view, aten.sum]
# Source node to ATen node mapping:
#   mul_41 => mul_41
#   permute_30 => permute_30
#   sum_18 => sum_18
#   view_23 => view_23
# Graph fragment:
#   %bmm_3 : Tensor "f32[4000, 1, 16][16, 16, 1]cuda:0" = PlaceHolder[target=bmm_3]
#   %mul_41 : Tensor "f32[4000, 1, 16][16, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_3, 0.25), kwargs = {})
#   %permute_30 : Tensor "f32[1, 4000, 16][16, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_41, [1, 0, 2]), kwargs = {})
#   %view_23 : Tensor "f32[1, 500, 128][16, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_30, [1, 500, 128]), kwargs = {})
#   %sum_18 : Tensor "f32[1, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_23, [0, 1], True), kwargs = {})
#   return %buf41
triton_red_fused_mul_sum_transpose_view_9 = async_compile.triton('triton_red_fused_mul_sum_transpose_view_9', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_transpose_view_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 260096, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_mul_sum_transpose_view_9(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r0_2 + 16000*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.25
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/6i/c6ietrvx3ztjh4xqq57bmwvpwg3lfazlevcmkf7lb7eupp3mr2zn.py
# Topologically Sorted Source Nodes: [mul_41, permute_30, view_23, view_25, sum_18, view_28, cat_2], Original ATen: [aten.mul, aten.transpose, aten.view, aten.sum, aten.cat]
# Source node to ATen node mapping:
#   cat_2 => cat_2
#   mul_41 => mul_41
#   permute_30 => permute_30
#   sum_18 => sum_18
#   view_23 => view_23
#   view_25 => view_25
#   view_28 => view_28
# Graph fragment:
#   %buf41 : Tensor "f32[1, 1, 128, 4][512, 512, 1, 128]cuda:0" = PlaceHolder[target=buf41]
#   %sum_18 : Tensor "f32[1, 1, 128][128, 128, 1]cuda:0" = PlaceHolder[target=sum_18]
#   %mul_41 : Tensor "f32[4000, 1, 16][16, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_3, 0.25), kwargs = {})
#   %permute_30 : Tensor "f32[1, 4000, 16][16, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_41, [1, 0, 2]), kwargs = {})
#   %view_23 : Tensor "f32[1, 500, 128][16, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_30, [1, 500, 128]), kwargs = {})
#   %view_25 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sum_17, [256]), kwargs = {})
#   %sum_18 : Tensor "f32[1, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_23, [0, 1], True), kwargs = {})
#   %view_28 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sum_18, [128]), kwargs = {})
#   %cat_2 : Tensor "f32[384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_28, %view_25],), kwargs = {})
#   return %sum_18,%buf46
triton_per_fused_cat_mul_sum_transpose_view_10 = async_compile.triton('triton_per_fused_cat_mul_sum_transpose_view_10', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_mul_sum_transpose_view_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 3072, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_cat_mul_sum_transpose_view_10(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /traces/inductor_cache/e2/ce2f4age6y6xl53x2t4schtiaauncp6ju3xekdlrh6egipbydnwz.py
# Topologically Sorted Source Nodes: [mul_41, permute_30, view_23, view_29], Original ATen: [aten.mul, aten.transpose, aten.view]
# Source node to ATen node mapping:
#   mul_41 => mul_41
#   permute_30 => permute_30
#   view_23 => view_23
#   view_29 => view_29
# Graph fragment:
#   %bmm_3 : Tensor "f32[4000, 1, 16][16, 16, 1]cuda:0" = PlaceHolder[target=bmm_3]
#   %mul_41 : Tensor "f32[4000, 1, 16][16, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_3, 0.25), kwargs = {})
#   %permute_30 : Tensor "f32[1, 4000, 16][16, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_41, [1, 0, 2]), kwargs = {})
#   %view_23 : Tensor "f32[1, 500, 128][16, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_30, [1, 500, 128]), kwargs = {})
#   %view_29 : Tensor "f32[500, 128][128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_23, [500, 128]), kwargs = {})
#   return %view_29
triton_poi_fused_mul_transpose_view_11 = async_compile.triton('triton_poi_fused_mul_transpose_view_11', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_transpose_view_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 768000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_transpose_view_11(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /traces/inductor_cache/hl/chlo6kw2dwjiksyt37hqxq4ckoelbgj57jvbf2tima4sffmxedgz.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
# Source node to ATen node mapping:
#    => constant_pad_nd_default
# Graph fragment:
#   %fma : Tensor "f32[4000, 1, 129][129, 516000, 1]cuda:0" = PlaceHolder[target=fma]
#   %constant_pad_nd_default : Tensor "f32[4000, 1, 132][132, 132, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%fma, [0, 3, 0, 0, 0, 0]), kwargs = {})
#   return %constant_pad_nd_default
triton_poi_fused_bmm_12 = async_compile.triton('triton_poi_fused_bmm_12', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 6288000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bmm_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /traces/inductor_cache/oe/coeqvzoes4tevtu4keufw5ywappeyzcfaetcfu4h74brisf2mxyj.py
# Topologically Sorted Source Nodes: [, permute_27, permute_28, clone_5, view_21, permute_29, view_22, full_1, _generalized_scatter, _generalized_scatter_1, add_15, unsqueeze_1, permute_31, squeeze_1, clone_6, view_24, sum_17], Original ATen: [aten.bmm, aten.transpose, aten.clone, aten._unsafe_view, aten.view, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.sum]
# Source node to ATen node mapping:
#    => slice_tensor
#   _generalized_scatter => select_scatter_default
#   _generalized_scatter_1 => select_scatter_default_1
#   add_15 => add_15
#   clone_5 => clone_5
#   clone_6 => clone_6
#   full_1 => full_default_1
#   permute_27 => permute_27
#   permute_28 => permute_28
#   permute_29 => permute_29
#   permute_31 => permute_31
#   squeeze_1 => squeeze_1
#   sum_17 => sum_17
#   unsqueeze_1 => unsqueeze_1
#   view_21 => view_21
#   view_22 => view_22
#   view_24 => view_24
# Graph fragment:
#   %bmm_1 : Tensor "f32[4000, 129, 16][2064, 16, 1]cuda:0" = PlaceHolder[target=bmm_1]
#   %bmm_default : Tensor "f32[4000, 16, 132][2112, 132, 1]cuda:0" = PlaceHolder[target=bmm_default]
#   %slice_tensor : Tensor "f32[4000, 16, 129][2112, 132, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%bmm_default, 2, 0, -3), kwargs = {})
#   %permute_27 : Tensor "f32[4000, 129, 16][2112, 1, 132]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%slice_tensor, [0, 2, 1]), kwargs = {})
#   %permute_28 : Tensor "f32[129, 4000, 16][16, 2064, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_1, [1, 0, 2]), kwargs = {})
#   %clone_5 : Tensor "f32[129, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_28,), kwargs = {memory_format: torch.contiguous_format})
#   %view_21 : Tensor "f32[129, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_5, [129, 500, 128]), kwargs = {})
#   %permute_29 : Tensor "f32[129, 4000, 16][1, 2112, 132]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_27, [1, 0, 2]), kwargs = {})
#   %view_22 : Tensor "f32[129, 500, 128][1, 16896, 132]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_29, [129, 500, 128]), kwargs = {})
#   %full_default_1 : Tensor "f32[2, 129, 500, 128][8256000, 64000, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([2, 129, 500, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : Tensor "f32[2, 129, 500, 128][8256000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_1, %view_21, 0, 1), kwargs = {})
#   %select_scatter_default_1 : Tensor "f32[2, 129, 500, 128][8256000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_1, %view_22, 0, 0), kwargs = {})
#   %add_15 : Tensor "f32[2, 129, 500, 128][8256000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default, %select_scatter_default_1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[2, 129, 500, 1, 128][8256000, 64000, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_15, 3), kwargs = {})
#   %permute_31 : Tensor "f32[1, 129, 500, 2, 128][128, 64000, 128, 8256000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_1, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze_1 : Tensor "f32[129, 500, 2, 128][64000, 128, 8256000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_31, 0), kwargs = {})
#   %clone_6 : Tensor "f32[129, 500, 2, 128][128000, 256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_1,), kwargs = {memory_format: torch.contiguous_format})
#   %view_24 : Tensor "f32[129, 500, 256][128000, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_6, [129, 500, 256]), kwargs = {})
#   %sum_17 : Tensor "f32[1, 1, 256][256, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_24, [0, 1], True), kwargs = {})
#   return %buf36
triton_red_fused__unsafe_view_add_bmm_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_13 = async_compile.triton('triton_red_fused__unsafe_view_add_bmm_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_13', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_view_add_bmm_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 33792000, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__unsafe_view_add_bmm_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_13(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /traces/inductor_cache/kq/ckq3rm6nvedbc4ny2b3u4t2cmytwdcdisimprxjutncqion6ezwv.py
# Topologically Sorted Source Nodes: [, permute_27, permute_28, clone_5, view_21, permute_29, view_22, full_1, _generalized_scatter, _generalized_scatter_1, add_15, unsqueeze_1, permute_31, squeeze_1, clone_6, view_24, sum_17, view_25, view_28, cat_2], Original ATen: [aten.bmm, aten.transpose, aten.clone, aten._unsafe_view, aten.view, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.sum, aten.cat]
# Source node to ATen node mapping:
#    => slice_tensor
#   _generalized_scatter => select_scatter_default
#   _generalized_scatter_1 => select_scatter_default_1
#   add_15 => add_15
#   cat_2 => cat_2
#   clone_5 => clone_5
#   clone_6 => clone_6
#   full_1 => full_default_1
#   permute_27 => permute_27
#   permute_28 => permute_28
#   permute_29 => permute_29
#   permute_31 => permute_31
#   squeeze_1 => squeeze_1
#   sum_17 => sum_17
#   unsqueeze_1 => unsqueeze_1
#   view_21 => view_21
#   view_22 => view_22
#   view_24 => view_24
#   view_25 => view_25
#   view_28 => view_28
# Graph fragment:
#   %buf36 : Tensor "f32[1, 1, 256, 375][96000, 96000, 1, 256]cuda:0" = PlaceHolder[target=buf36]
#   %sum_17 : Tensor "f32[1, 1, 256][256, 256, 1]cuda:0" = PlaceHolder[target=sum_17]
#   %slice_tensor : Tensor "f32[4000, 16, 129][2112, 132, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%bmm_default, 2, 0, -3), kwargs = {})
#   %permute_27 : Tensor "f32[4000, 129, 16][2112, 1, 132]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%slice_tensor, [0, 2, 1]), kwargs = {})
#   %permute_28 : Tensor "f32[129, 4000, 16][16, 2064, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_1, [1, 0, 2]), kwargs = {})
#   %clone_5 : Tensor "f32[129, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_28,), kwargs = {memory_format: torch.contiguous_format})
#   %view_21 : Tensor "f32[129, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_5, [129, 500, 128]), kwargs = {})
#   %permute_29 : Tensor "f32[129, 4000, 16][1, 2112, 132]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_27, [1, 0, 2]), kwargs = {})
#   %view_22 : Tensor "f32[129, 500, 128][1, 16896, 132]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_29, [129, 500, 128]), kwargs = {})
#   %full_default_1 : Tensor "f32[2, 129, 500, 128][8256000, 64000, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([2, 129, 500, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : Tensor "f32[2, 129, 500, 128][8256000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_1, %view_21, 0, 1), kwargs = {})
#   %select_scatter_default_1 : Tensor "f32[2, 129, 500, 128][8256000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_1, %view_22, 0, 0), kwargs = {})
#   %add_15 : Tensor "f32[2, 129, 500, 128][8256000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default, %select_scatter_default_1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[2, 129, 500, 1, 128][8256000, 64000, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_15, 3), kwargs = {})
#   %permute_31 : Tensor "f32[1, 129, 500, 2, 128][128, 64000, 128, 8256000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_1, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze_1 : Tensor "f32[129, 500, 2, 128][64000, 128, 8256000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_31, 0), kwargs = {})
#   %clone_6 : Tensor "f32[129, 500, 2, 128][128000, 256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_1,), kwargs = {memory_format: torch.contiguous_format})
#   %view_24 : Tensor "f32[129, 500, 256][128000, 256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_6, [129, 500, 256]), kwargs = {})
#   %sum_17 : Tensor "f32[1, 1, 256][256, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_24, [0, 1], True), kwargs = {})
#   %view_25 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sum_17, [256]), kwargs = {})
#   %view_28 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sum_18, [128]), kwargs = {})
#   %cat_2 : Tensor "f32[384][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_28, %view_25],), kwargs = {})
#   return %sum_17,%buf47
triton_red_fused__unsafe_view_add_bmm_cat_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_14 = async_compile.triton('triton_red_fused__unsafe_view_add_bmm_cat_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_14', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_view_add_bmm_cat_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 386048, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__unsafe_view_add_bmm_cat_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_14(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /traces/inductor_cache/bn/cbnzb26zn4653v5iaspy35azx2gu3b5duxjiuin5vh4kydzzpljo.py
# Topologically Sorted Source Nodes: [, permute_27, permute_28, clone_5, view_21, permute_29, view_22, full_1, _generalized_scatter, _generalized_scatter_1, add_15, unsqueeze_1, permute_31, squeeze_1, clone_6], Original ATen: [aten.bmm, aten.transpose, aten.clone, aten._unsafe_view, aten.view, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze]
# Source node to ATen node mapping:
#    => slice_tensor
#   _generalized_scatter => select_scatter_default
#   _generalized_scatter_1 => select_scatter_default_1
#   add_15 => add_15
#   clone_5 => clone_5
#   clone_6 => clone_6
#   full_1 => full_default_1
#   permute_27 => permute_27
#   permute_28 => permute_28
#   permute_29 => permute_29
#   permute_31 => permute_31
#   squeeze_1 => squeeze_1
#   unsqueeze_1 => unsqueeze_1
#   view_21 => view_21
#   view_22 => view_22
# Graph fragment:
#   %bmm_1 : Tensor "f32[4000, 129, 16][2064, 16, 1]cuda:0" = PlaceHolder[target=bmm_1]
#   %bmm_default : Tensor "f32[4000, 16, 132][2112, 132, 1]cuda:0" = PlaceHolder[target=bmm_default]
#   %slice_tensor : Tensor "f32[4000, 16, 129][2112, 132, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%bmm_default, 2, 0, -3), kwargs = {})
#   %permute_27 : Tensor "f32[4000, 129, 16][2112, 1, 132]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%slice_tensor, [0, 2, 1]), kwargs = {})
#   %permute_28 : Tensor "f32[129, 4000, 16][16, 2064, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_1, [1, 0, 2]), kwargs = {})
#   %clone_5 : Tensor "f32[129, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_28,), kwargs = {memory_format: torch.contiguous_format})
#   %view_21 : Tensor "f32[129, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_5, [129, 500, 128]), kwargs = {})
#   %permute_29 : Tensor "f32[129, 4000, 16][1, 2112, 132]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_27, [1, 0, 2]), kwargs = {})
#   %view_22 : Tensor "f32[129, 500, 128][1, 16896, 132]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_29, [129, 500, 128]), kwargs = {})
#   %full_default_1 : Tensor "f32[2, 129, 500, 128][8256000, 64000, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([2, 129, 500, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : Tensor "f32[2, 129, 500, 128][8256000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_1, %view_21, 0, 1), kwargs = {})
#   %select_scatter_default_1 : Tensor "f32[2, 129, 500, 128][8256000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_1, %view_22, 0, 0), kwargs = {})
#   %add_15 : Tensor "f32[2, 129, 500, 128][8256000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default, %select_scatter_default_1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[2, 129, 500, 1, 128][8256000, 64000, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_15, 3), kwargs = {})
#   %permute_31 : Tensor "f32[1, 129, 500, 2, 128][128, 64000, 128, 8256000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_1, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze_1 : Tensor "f32[129, 500, 2, 128][64000, 128, 8256000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_31, 0), kwargs = {})
#   %clone_6 : Tensor "f32[129, 500, 2, 128][128000, 256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_1,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_6
triton_poi_fused__unsafe_view_add_bmm_clone_select_backward_squeeze_transpose_unsqueeze_view_15 = async_compile.triton('triton_poi_fused__unsafe_view_add_bmm_clone_select_backward_squeeze_transpose_unsqueeze_view_15', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_bmm_clone_select_backward_squeeze_transpose_unsqueeze_view_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 33024000, 'x': 165120000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_bmm_clone_select_backward_squeeze_transpose_unsqueeze_view_15(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /traces/inductor_cache/e5/ce5lkou3bmbcqujg4ucqux2vm5dszwcgjztlq4imgqbr6dq7r3ge.py
# Topologically Sorted Source Nodes: [view_27, permute_40, mul_43, sum_19], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   mul_43 => mul_43
#   permute_40 => permute_40
#   sum_19 => sum_19
#   view_27 => view_27
# Graph fragment:
#   %mm_7 : Tensor "f32[64500, 128][128, 1]cuda:0" = PlaceHolder[target=mm_7]
#   %primals_4 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_4]
#   %view_27 : Tensor "f32[129, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_7, [129, 500, 128]), kwargs = {})
#   %permute_40 : Tensor "f32[500, 129, 128][128, 64000, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_27, [1, 0, 2]), kwargs = {})
#   %mul_43 : Tensor "f32[500, 129, 128][128, 64000, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_40, %primals_4), kwargs = {})
#   %sum_19 : Tensor "f32[500, 129, 1][129, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_43, [2], True), kwargs = {})
#   return %sum_19
triton_per_fused_native_layer_norm_backward_transpose_view_16 = async_compile.triton('triton_per_fused_native_layer_norm_backward_transpose_view_16', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_transpose_view_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 516000, 'r0_': 33024512}}
)
@triton.jit
def triton_per_fused_native_layer_norm_backward_transpose_view_16(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /traces/inductor_cache/qv/cqvvjcapgawf6hyoiyl6pwmbxsb5l5jbqdiwvtgdintnmq2aroyf.py
# Topologically Sorted Source Nodes: [view_27, permute_40, mul_43, mul_44, layer_norm, mul_45, sum_20, mul_46, sub_15, sub_16, div_4, mul_47, mul_48, sum_21, sum_22], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward, aten.native_layer_norm]
# Source node to ATen node mapping:
#   div_4 => div_4
#   layer_norm => mul, sub
#   mul_43 => mul_43
#   mul_44 => mul_44
#   mul_45 => mul_45
#   mul_46 => mul_46
#   mul_47 => mul_47
#   mul_48 => mul_48
#   permute_40 => permute_40
#   sub_15 => sub_15
#   sub_16 => sub_16
#   sum_20 => sum_20
#   sum_21 => sum_21
#   sum_22 => sum_22
#   view_27 => view_27
# Graph fragment:
#   %mm_7 : Tensor "f32[64500, 128][128, 1]cuda:0" = PlaceHolder[target=mm_7]
#   %primals_4 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_4]
#   %cat_1 : Tensor "f32[500, 129, 128][16512, 128, 1]cuda:0" = PlaceHolder[target=cat_1]
#   %getitem_1 : Tensor "f32[500, 129, 1][129, 1, 1]cuda:0" = PlaceHolder[target=getitem_1]
#   %rsqrt : Tensor "f32[500, 129, 1][129, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %sum_19 : Tensor "f32[500, 129, 1][1, 500, 64512]cuda:0" = PlaceHolder[target=sum_19]
#   %sum_20 : Tensor "f32[500, 129, 1][129, 1, 64512]cuda:0" = PlaceHolder[target=sum_20]
#   %view_27 : Tensor "f32[129, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_7, [129, 500, 128]), kwargs = {})
#   %permute_40 : Tensor "f32[500, 129, 128][128, 64000, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_27, [1, 0, 2]), kwargs = {})
#   %mul_43 : Tensor "f32[500, 129, 128][128, 64000, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_40, %primals_4), kwargs = {})
#   %mul_44 : Tensor "f32[500, 129, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, 128), kwargs = {})
#   %sub : Tensor "f32[500, 129, 128][16512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_1, %getitem_1), kwargs = {})
#   %mul : Tensor "f32[500, 129, 128][16512, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_45 : Tensor "f32[500, 129, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %mul), kwargs = {})
#   %sum_20 : Tensor "f32[500, 129, 1][129, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_45, [2], True), kwargs = {})
#   %mul_46 : Tensor "f32[500, 129, 128][16512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %sum_20), kwargs = {})
#   %sub_15 : Tensor "f32[500, 129, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_44, %sum_19), kwargs = {})
#   %sub_16 : Tensor "f32[500, 129, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_15, %mul_46), kwargs = {})
#   %div_4 : Tensor "f32[500, 129, 1][129, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt, 128), kwargs = {})
#   %mul_47 : Tensor "f32[500, 129, 128][16512, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_4, %sub_16), kwargs = {})
#   %mul_48 : Tensor "f32[500, 129, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_40, %mul), kwargs = {})
#   %sum_21 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_48, [0, 1]), kwargs = {})
#   %sum_22 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%permute_40, [0, 1]), kwargs = {})
#   return %sum_20,%mul_47
triton_per_fused_native_layer_norm_native_layer_norm_backward_transpose_view_17 = async_compile.triton('triton_per_fused_native_layer_norm_native_layer_norm_backward_transpose_view_17', '''
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
    inductor_meta={'grid_type': 'MixOrderReductionGrid', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_transpose_view_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 7, 'num_store': 0, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'RSPLIT_SIZE': 64}
)
@triton.jit
def triton_per_fused_native_layer_norm_native_layer_norm_backward_transpose_view_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, RSPLIT_SIZE : tl.constexpr, NUM_STAGES : tl.constexpr):
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


# kernel path: /traces/inductor_cache/vw/cvwrtqxhywsrwysy5wrbhcv6vgoa3xyroiac2u3ocl2yaruqnzyq.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward, transpose_2, layer_norm_1, mul_39, sum_13, sum_14, view_30, permute_41, add_16, slice_2, add_17, sum_23], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm, aten.native_layer_norm_backward, aten.add, aten.slice, aten.sum]
# Source node to ATen node mapping:
#   add_16 => add_16
#   add_17 => add_17
#   layer_norm_1 => mul_3, sub_2
#   mul_39 => mul_39
#   multi_head_attention_forward => view_11
#   permute_41 => permute_41
#   slice_2 => slice_2
#   sum_13 => sum_13
#   sum_14 => sum_14
#   sum_23 => sum_23
#   transpose_2 => permute_11
#   view_30 => view_30
# Graph fragment:
#   %add_14 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0" = PlaceHolder[target=add_14]
#   %addmm : Tensor "f32[500, 128][128, 1]cuda:0" = PlaceHolder[target=addmm]
#   %getitem_7 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=getitem_7]
#   %rsqrt_1 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_1]
#   %mm_9 : Tensor "f32[500, 128][128, 1]cuda:0" = PlaceHolder[target=mm_9]
#   %mul_47 : Tensor "f32[500, 129, 128][16512, 128, 1]cuda:0" = PlaceHolder[target=mul_47]
#   %view_11 : Tensor "f32[1, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [1, 500, 128]), kwargs = {})
#   %permute_11 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_11, [1, 0, 2]), kwargs = {})
#   %sub_2 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_11, %getitem_7), kwargs = {})
#   %mul_3 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_39 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_14, %mul_3), kwargs = {})
#   %sum_13 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_39, [0, 1]), kwargs = {})
#   %sum_14 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%add_14, [0, 1]), kwargs = {})
#   %view_30 : Tensor "f32[1, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_9, [1, 500, 128]), kwargs = {})
#   %permute_41 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_30, [1, 0, 2]), kwargs = {})
#   %add_16 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_14, %permute_41), kwargs = {})
#   %slice_2 : Tensor "f32[500, 1, 128][16512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_47, 1, 0, 1), kwargs = {})
#   %add_17 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %slice_2), kwargs = {})
#   %sum_23 : Tensor "f32[1, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%add_17, [0], True), kwargs = {})
#   return %buf20,%buf22,%buf57
triton_red_fused_add_native_layer_norm_native_layer_norm_backward_slice_sum_transpose_view_18 = async_compile.triton('triton_red_fused_add_native_layer_norm_native_layer_norm_backward_slice_sum_transpose_view_18', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_slice_sum_transpose_view_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 3, 'num_reduction': 3, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1036288, 'r0_': 4000}}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_native_layer_norm_backward_slice_sum_transpose_view_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp18 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
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
        tmp13 = tl.load(in_ptr4 + (x0 + 128*r0_2 + 16000*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr5 + (x0 + 16512*r0_2 + 2064000*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp5 = tmp3 * tmp4
        tmp6 = tmp0 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
        tmp10 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
        tmp14 = tmp0 + tmp13
        tmp16 = tmp14 + tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(r0_mask & xmask, tmp19, _tmp18)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
    tl.store(out_ptr1 + (x3), tmp11, xmask)
    tl.store(out_ptr2 + (x3), tmp18, xmask)
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
        primals_1, primals_4, primals_8, primals_10, primals_12, primals_14, primals_16, cat_1, getitem_1, rsqrt, view_2, div, view_10, addmm, getitem_7, rsqrt_1, add_6, getitem_9, rsqrt_2, view_13, addmm_1, getitem_11, rsqrt_3, permute_24, permute_25, permute_26, permute_34, permute_38, tangents_1, tangents_2 = args
        args.clear()
        assert_size_stride(primals_1, (1, 1, 128), (128, 128, 1))
        assert_size_stride(primals_4, (128, ), (1, ))
        assert_size_stride(primals_8, (128, 128), (128, 1))
        assert_size_stride(primals_10, (128, ), (1, ))
        assert_size_stride(primals_12, (128, ), (1, ))
        assert_size_stride(primals_14, (512, 128), (128, 1))
        assert_size_stride(primals_16, (512, ), (1, ))
        assert_size_stride(cat_1, (500, 129, 128), (16512, 128, 1))
        assert_size_stride(getitem_1, (500, 129, 1), (129, 1, 1))
        assert_size_stride(rsqrt, (500, 129, 1), (129, 1, 1))
        assert_size_stride(view_2, (64500, 128), (128, 1))
        assert_size_stride(div, (4000, 1, 129), (129, 129, 1))
        assert_size_stride(view_10, (500, 128), (128, 1))
        assert_size_stride(addmm, (500, 128), (128, 1))
        assert_size_stride(getitem_7, (500, 1, 1), (1, 1, 1))
        assert_size_stride(rsqrt_1, (500, 1, 1), (1, 1, 1))
        assert_size_stride(add_6, (500, 1, 128), (128, 64000, 1))
        assert_size_stride(getitem_9, (500, 1, 1), (1, 1, 1))
        assert_size_stride(rsqrt_2, (500, 1, 1), (1, 1, 1))
        assert_size_stride(view_13, (500, 128), (128, 1))
        assert_size_stride(addmm_1, (500, 512), (512, 1))
        assert_size_stride(getitem_11, (500, 1, 1), (1, 1, 1))
        assert_size_stride(rsqrt_3, (500, 1, 1), (1, 1, 1))
        assert_size_stride(permute_24, (4000, 16, 129), (16, 1, 64000))
        assert_size_stride(permute_25, (4000, 129, 16), (16, 64000, 1))
        assert_size_stride(permute_26, (4000, 16, 1), (16, 1, 16))
        assert_size_stride(permute_34, (256, 128), (128, 1))
        assert_size_stride(permute_38, (128, 128), (128, 1))
        assert_size_stride(tangents_1, (500, 1, 512), (512, 512, 1))
        assert_size_stride(tangents_2, (500, 1, 128), (128, 64000, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf2 = empty_strided_cuda((512, 4), (1, 512), torch.float32)
            buf4 = empty_strided_cuda((512, 4), (1, 512), torch.float32)
            # Topologically Sorted Source Nodes: [linear, gelu, layer_norm_3, mul_18, sum_4, sum_5], Original ATen: [aten.view, aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_red_fused_gelu_native_layer_norm_native_layer_norm_backward_view_0:55
            stream0 = get_raw_stream(0)
            triton_red_fused_gelu_native_layer_norm_native_layer_norm_backward_view_0.run(tangents_1, addmm_1, getitem_11, rsqrt_3, buf2, buf4, 2048, 125, stream=stream0)
            buf3 = empty_strided_cuda((512, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [linear, gelu, layer_norm_3, mul_18, sum_4], Original ATen: [aten.view, aten.gelu, aten.native_layer_norm, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_1:56
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_1.run(buf2, buf3, 512, 4, stream=stream0)
            del buf2
            buf5 = empty_strided_cuda((512, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [sum_5], Original ATen: [aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_1:57
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_1.run(buf4, buf5, 512, 4, stream=stream0)
            buf6 = reinterpret_tensor(addmm_1, (500, 1, 512), (512, 512, 1), 0); del addmm_1  # reuse
            # Topologically Sorted Source Nodes: [mul_13, mul_14, sum_2, linear, gelu, layer_norm_3, mul_15, sum_3, mul_16, sub_6, sub_7, div_1, mul_17, mul_20, mul_21, mul_22, exp_1, mul_23, mul_24, add_13, mul_25], Original ATen: [aten.native_layer_norm_backward, aten.view, aten.gelu, aten.native_layer_norm, aten.gelu_backward]
            # [Provenance debug handles] triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_view_2:58
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_view_2.run(buf6, tangents_1, primals_16, getitem_11, rsqrt_3, 500, 512, stream=stream0)
            del getitem_11
            del primals_16
            del rsqrt_3
            del tangents_1
            buf8 = empty_strided_cuda((512, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_15, permute_14, permute_16], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:59
            extern_kernels.mm(reinterpret_tensor(buf6, (512, 500), (1, 512), 0), view_13, out=buf8)
            del view_13
            buf9 = reinterpret_tensor(buf4, (1, 512, 4), (2048, 1, 512), 0); del buf4  # reuse
            # Topologically Sorted Source Nodes: [view_15, sum_6], Original ATen: [aten.view, aten.sum]
            # [Provenance debug handles] triton_red_fused_sum_view_3:60
            stream0 = get_raw_stream(0)
            triton_red_fused_sum_view_3.run(buf6, buf9, 2048, 125, stream=stream0)
            buf7 = empty_strided_cuda((500, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_15, linear, permute_13, mm_2], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:61
            extern_kernels.mm(reinterpret_tensor(buf6, (500, 512), (512, 1), 0), primals_14, out=buf7)
            del buf6
            del primals_14
            buf10 = empty_strided_cuda((1, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_15, sum_6], Original ATen: [aten.view, aten.sum]
            # [Provenance debug handles] triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_1:62
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_native_layer_norm_native_layer_norm_backward_view_1.run(buf9, buf10, 512, 4, stream=stream0)
            del buf9
            buf13 = empty_strided_cuda((128, 4), (1, 128), torch.float32)
            buf15 = empty_strided_cuda((128, 4), (1, 128), torch.float32)
            # Topologically Sorted Source Nodes: [view_17, layer_norm_2, mul_32, sum_9, sum_10], Original ATen: [aten.view, aten.native_layer_norm, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_red_fused_native_layer_norm_native_layer_norm_backward_view_4:63
            stream0 = get_raw_stream(0)
            triton_red_fused_native_layer_norm_native_layer_norm_backward_view_4.run(buf7, add_6, getitem_9, rsqrt_2, buf13, buf15, 512, 125, stream=stream0)
            buf17 = reinterpret_tensor(buf7, (500, 1, 128), (128, 64000, 1), 0); del buf7  # reuse
            buf24 = empty_strided_cuda((500, 1, 128), (128, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_17, mul_27, mul_28, sum_7, layer_norm_2, mul_29, sum_8, mul_30, sub_9, sub_10, div_2, mul_31, add_14, mul_34, mul_35, sum_11, multi_head_attention_forward, transpose_2, layer_norm_1, mul_36, sum_12, mul_37, sub_12, sub_13, div_3, mul_38], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.native_layer_norm, aten.add, aten.transpose]
            # [Provenance debug handles] triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_5:64
            stream0 = get_raw_stream(0)
            triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_5.run(buf17, primals_12, add_6, getitem_9, rsqrt_2, tangents_2, primals_10, addmm, getitem_7, rsqrt_1, buf24, 500, 128, stream=stream0)
            del add_6
            del getitem_9
            del primals_10
            del primals_12
            del rsqrt_2
            del tangents_2
            buf26 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mul_34, mul_35, multi_head_attention_forward, transpose_2, layer_norm_1, mul_37, sub_12, sub_13, div_3, mul_38, permute_17, view_18, permute_19, permute_21], Original ATen: [aten.native_layer_norm_backward, aten.view, aten.transpose, aten.native_layer_norm, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:65
            extern_kernels.mm(reinterpret_tensor(buf24, (128, 500), (1, 128), 0), view_10, out=buf26)
            del view_10
            buf14 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_17, layer_norm_2, mul_32, sum_9], Original ATen: [aten.view, aten.native_layer_norm, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_native_layer_norm_backward_view_6:66
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_native_layer_norm_backward_view_6.run(buf13, buf14, 128, 4, stream=stream0)
            del buf13
            buf16 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_17, sum_10], Original ATen: [aten.view, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_native_layer_norm_backward_view_6:67
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_native_layer_norm_backward_view_6.run(buf15, buf16, 128, 4, stream=stream0)
            buf27 = reinterpret_tensor(buf15, (1, 128, 4), (512, 1, 128), 0); del buf15  # reuse
            # Topologically Sorted Source Nodes: [mul_34, mul_35, multi_head_attention_forward, transpose_2, layer_norm_1, mul_37, sub_12, sub_13, div_3, mul_38, permute_17, view_18, sum_15], Original ATen: [aten.native_layer_norm_backward, aten.view, aten.transpose, aten.native_layer_norm, aten.sum]
            # [Provenance debug handles] triton_red_fused_native_layer_norm_native_layer_norm_backward_sum_transpose_view_7:68
            stream0 = get_raw_stream(0)
            triton_red_fused_native_layer_norm_native_layer_norm_backward_sum_transpose_view_7.run(buf24, buf27, 512, 125, stream=stream0)
            buf28 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mul_34, mul_35, multi_head_attention_forward, transpose_2, layer_norm_1, mul_37, sub_12, sub_13, div_3, mul_38, permute_17, view_18, sum_15], Original ATen: [aten.native_layer_norm_backward, aten.view, aten.transpose, aten.native_layer_norm, aten.sum]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_native_layer_norm_backward_view_6:69
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_native_layer_norm_backward_view_6.run(buf27, buf28, 128, 4, stream=stream0)
            buf25 = empty_strided_cuda((500, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mul_34, mul_35, multi_head_attention_forward, transpose_2, layer_norm_1, mul_37, sub_12, sub_13, div_3, mul_38, permute_17, view_18, permute_18, mm_4], Original ATen: [aten.native_layer_norm_backward, aten.view, aten.transpose, aten.native_layer_norm, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:70
            extern_kernels.mm(reinterpret_tensor(buf24, (500, 128), (128, 1), 0), primals_8, out=buf25)
            del primals_8
            buf30 = empty_strided_cuda((4000, 1, 129), (129, 129, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_20, permute_22, bmm_2], Original ATen: [aten.view, aten.transpose, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:71
            extern_kernels.bmm(reinterpret_tensor(buf25, (4000, 1, 16), (16, 64000, 1), 0), permute_24, out=buf30)
            del permute_24
            buf32 = reinterpret_tensor(buf30, (4000, 1, 129), (129, 516000, 1), 0); del buf30  # reuse
            # Topologically Sorted Source Nodes: [mul_40, sum_16, neg, fma], Original ATen: [aten._softmax_backward_data]
            # [Provenance debug handles] triton_per_fused__softmax_backward_data_8:72
            stream0 = get_raw_stream(0)
            triton_per_fused__softmax_backward_data_8.run(buf32, div, 4000, 129, stream=stream0)
            buf33 = reinterpret_tensor(buf24, (4000, 1, 16), (16, 16, 1), 0); del buf24  # reuse
            # Topologically Sorted Source Nodes: [bmm_3], Original ATen: [aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:73
            extern_kernels.bmm(buf32, permute_25, out=buf33)
            del permute_25
            buf41 = reinterpret_tensor(buf27, (1, 1, 128, 4), (512, 512, 1, 128), 0); del buf27  # reuse
            # Topologically Sorted Source Nodes: [mul_41, permute_30, view_23, sum_18], Original ATen: [aten.mul, aten.transpose, aten.view, aten.sum]
            # [Provenance debug handles] triton_red_fused_mul_sum_transpose_view_9:74
            stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_transpose_view_9.run(buf33, buf41, 512, 125, stream=stream0)
            buf48 = empty_strided_cuda((384, ), (1, ), torch.float32)
            buf46 = reinterpret_tensor(buf48, (128, ), (1, ), 0)  # alias
            # Topologically Sorted Source Nodes: [mul_41, permute_30, view_23, view_25, sum_18, view_28, cat_2], Original ATen: [aten.mul, aten.transpose, aten.view, aten.sum, aten.cat]
            # [Provenance debug handles] triton_per_fused_cat_mul_sum_transpose_view_10:75
            stream0 = get_raw_stream(0)
            triton_per_fused_cat_mul_sum_transpose_view_10.run(buf41, buf46, 128, 4, stream=stream0)
            del buf41
            buf43 = reinterpret_tensor(buf33, (500, 128), (128, 1), 0); del buf33  # reuse
            # Topologically Sorted Source Nodes: [mul_41, permute_30, view_23, view_29], Original ATen: [aten.mul, aten.transpose, aten.view]
            # [Provenance debug handles] triton_poi_fused_mul_transpose_view_11:76
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_transpose_view_11.run(buf43, 64000, stream=stream0)
            buf34 = empty_strided_cuda((4000, 1, 132), (132, 528000, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            # [Provenance debug handles] triton_poi_fused_bmm_12:77
            stream0 = get_raw_stream(0)
            triton_poi_fused_bmm_12.run(buf32, buf34, 528000, stream=stream0)
            del buf32
            buf49 = empty_strided_cuda((384, 128), (128, 1), torch.float32)
            buf44 = reinterpret_tensor(buf49, (128, 128), (128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [permute_36, expand, transpose, multi_head_attention_forward, permute_39], Original ATen: [aten.t, aten.expand, aten.transpose, aten.view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:78
            extern_kernels.mm(reinterpret_tensor(buf43, (128, 500), (1, 128), 0), reinterpret_tensor(primals_1, (500, 128), (0, 1), 0), out=buf44)
            del primals_1
            buf45 = empty_strided_cuda((500, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mm_9], Original ATen: [aten.mm]
            # [Provenance debug handles] extern_kernels.mm:79
            extern_kernels.mm(buf43, permute_38, out=buf45)
            del buf43
            del permute_38
            buf29 = empty_strided_cuda((4000, 129, 16), (2064, 16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_20, permute_22, permute_23, bmm_1], Original ATen: [aten.view, aten.transpose, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:80
            extern_kernels.bmm(reinterpret_tensor(div, (4000, 129, 1), (129, 1, 129), 0), reinterpret_tensor(buf25, (4000, 1, 16), (16, 64000, 1), 0), out=buf29)
            del buf25
            del div
            buf35 = empty_strided_cuda((4000, 16, 132), (2112, 132, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:81
            extern_kernels.bmm(permute_26, buf34, out=buf35)
            del buf34
            del permute_26
            buf36 = empty_strided_cuda((1, 1, 256, 375), (96000, 96000, 1, 256), torch.float32)
            # Topologically Sorted Source Nodes: [, permute_27, permute_28, clone_5, view_21, permute_29, view_22, full_1, _generalized_scatter, _generalized_scatter_1, add_15, unsqueeze_1, permute_31, squeeze_1, clone_6, view_24, sum_17], Original ATen: [aten.bmm, aten.transpose, aten.clone, aten._unsafe_view, aten.view, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_add_bmm_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_13:82
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_add_bmm_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_13.run(buf29, buf35, buf36, 96000, 172, stream=stream0)
            buf47 = reinterpret_tensor(buf48, (256, ), (1, ), 128)  # alias
            # Topologically Sorted Source Nodes: [, permute_27, permute_28, clone_5, view_21, permute_29, view_22, full_1, _generalized_scatter, _generalized_scatter_1, add_15, unsqueeze_1, permute_31, squeeze_1, clone_6, view_24, sum_17, view_25, view_28, cat_2], Original ATen: [aten.bmm, aten.transpose, aten.clone, aten._unsafe_view, aten.view, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.sum, aten.cat]
            # [Provenance debug handles] triton_red_fused__unsafe_view_add_bmm_cat_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_14:83
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_add_bmm_cat_clone_select_backward_squeeze_sum_transpose_unsqueeze_view_14.run(buf36, buf47, 256, 375, stream=stream0)
            del buf36
            buf38 = empty_strided_cuda((129, 500, 2, 128), (128000, 256, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, permute_27, permute_28, clone_5, view_21, permute_29, view_22, full_1, _generalized_scatter, _generalized_scatter_1, add_15, unsqueeze_1, permute_31, squeeze_1, clone_6], Original ATen: [aten.bmm, aten.transpose, aten.clone, aten._unsafe_view, aten.view, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_bmm_clone_select_backward_squeeze_transpose_unsqueeze_view_15:84
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_bmm_clone_select_backward_squeeze_transpose_unsqueeze_view_15.run(buf29, buf35, buf38, 64500, 256, stream=stream0)
            del buf35
            buf39 = reinterpret_tensor(buf49, (256, 128), (128, 1), 16384)  # alias
            # Topologically Sorted Source Nodes: [, permute_27, permute_28, clone_5, view_21, permute_29, view_22, full_1, _generalized_scatter, _generalized_scatter_1, add_15, unsqueeze_1, permute_31, squeeze_1, clone_6, view_24, view_26, permute_32, permute_35], Original ATen: [aten.bmm, aten.transpose, aten.clone, aten._unsafe_view, aten.view, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:85
            extern_kernels.mm(reinterpret_tensor(buf38, (256, 64500), (1, 256), 0), view_2, out=buf39)
            del view_2
            buf40 = reinterpret_tensor(buf29, (64500, 128), (128, 1), 0); del buf29  # reuse
            # Topologically Sorted Source Nodes: [, permute_27, permute_28, clone_5, view_21, permute_29, view_22, full_1, _generalized_scatter, _generalized_scatter_1, add_15, unsqueeze_1, permute_31, squeeze_1, clone_6, view_24, view_26, mm_7], Original ATen: [aten.bmm, aten.transpose, aten.clone, aten._unsafe_view, aten.view, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:86
            extern_kernels.mm(reinterpret_tensor(buf38, (64500, 256), (256, 1), 0), permute_34, out=buf40)
            del buf38
            del permute_34
            buf50 = empty_strided_cuda((500, 129, 1), (1, 500, 64512), torch.float32)
            # Topologically Sorted Source Nodes: [view_27, permute_40, mul_43, sum_19], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_backward_transpose_view_16:87
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_backward_transpose_view_16.run(buf40, primals_4, buf50, 64500, 128, stream=stream0)
            buf52 = empty_strided_cuda((500, 129, 128), (16512, 128, 1), torch.float32)
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [view_27, permute_40, mul_43, mul_44, layer_norm, mul_45, sum_20, mul_46, sub_15, sub_16, div_4, mul_47, mul_48, sum_21, sum_22], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward, aten.native_layer_norm]
            workspace_0 = empty_strided_cuda((258048, ), (1, ), torch.float32)
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_native_layer_norm_backward_transpose_view_17.run(buf40, primals_4, cat_1, getitem_1, rsqrt, buf50, buf52, workspace_0, 64500, 128, stream=stream0)
            buf54 = workspace_0[0 * 1008 * 128 : (0 + 1) * 1008 * 128].view(1008, 128).sum(dim=0)
            buf56 = workspace_0[1 * 1008 * 128 : (1 + 1) * 1008 * 128].view(1008, 128).sum(dim=0)
            del workspace_0
            del buf40
            del buf50
            del cat_1
            del getitem_1
            del primals_4
            del rsqrt
            buf20 = empty_strided_cuda((128, 4), (1, 128), torch.float32)
            buf22 = empty_strided_cuda((128, 4), (1, 128), torch.float32)
            buf57 = empty_strided_cuda((1, 1, 128, 4), (512, 512, 1, 128), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward, transpose_2, layer_norm_1, mul_39, sum_13, sum_14, view_30, permute_41, add_16, slice_2, add_17, sum_23], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm, aten.native_layer_norm_backward, aten.add, aten.slice, aten.sum]
            # [Provenance debug handles] triton_red_fused_add_native_layer_norm_native_layer_norm_backward_slice_sum_transpose_view_18:88
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_layer_norm_native_layer_norm_backward_slice_sum_transpose_view_18.run(buf17, addmm, getitem_7, rsqrt_1, buf45, buf52, buf20, buf22, buf57, 512, 125, stream=stream0)
            del addmm
            del buf17
            del buf45
            del getitem_7
            del rsqrt_1
            buf21 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward, transpose_2, layer_norm_1, mul_39, sum_13], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_native_layer_norm_backward_view_6:89
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_native_layer_norm_backward_view_6.run(buf20, buf21, 128, 4, stream=stream0)
            del buf20
            buf23 = empty_strided_cuda((128, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [sum_14], Original ATen: [aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_native_layer_norm_backward_view_6:90
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_native_layer_norm_backward_view_6.run(buf22, buf23, 128, 4, stream=stream0)
            del buf22
            buf58 = empty_strided_cuda((1, 1, 128), (128, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_30, permute_41, add_16, slice_2, add_17, sum_23], Original ATen: [aten.view, aten.transpose, aten.add, aten.slice, aten.sum]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_native_layer_norm_backward_view_6:91
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_native_layer_norm_backward_view_6.run(buf57, buf58, 128, 4, stream=stream0)
            del buf57
        return (buf58, None, reinterpret_tensor(buf52, (500, 128, 128), (16512, 128, 1), 128), buf54, buf56, buf49, buf48, buf26, reinterpret_tensor(buf28, (128, ), (1, ), 0), buf21, buf23, buf14, buf16, buf8, reinterpret_tensor(buf10, (512, ), (1, ), 0), buf3, buf5, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 1, 128), (128, 128, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_1 = rand_strided((500, 129, 128), (16512, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((500, 129, 1), (129, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt = rand_strided((500, 129, 1), (129, 1, 1), device='cuda:0', dtype=torch.float32)
    view_2 = rand_strided((64500, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div = rand_strided((4000, 1, 129), (129, 129, 1), device='cuda:0', dtype=torch.float32)
    view_10 = rand_strided((500, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm = rand_strided((500, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((500, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_1 = rand_strided((500, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    add_6 = rand_strided((500, 1, 128), (128, 64000, 1), device='cuda:0', dtype=torch.float32)
    getitem_9 = rand_strided((500, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_2 = rand_strided((500, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    view_13 = rand_strided((500, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_1 = rand_strided((500, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_11 = rand_strided((500, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_3 = rand_strided((500, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_24 = rand_strided((4000, 16, 129), (16, 1, 64000), device='cuda:0', dtype=torch.float32)
    permute_25 = rand_strided((4000, 129, 16), (16, 64000, 1), device='cuda:0', dtype=torch.float32)
    permute_26 = rand_strided((4000, 16, 1), (16, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_34 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_38 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((500, 1, 512), (512, 512, 1), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((500, 1, 128), (128, 64000, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_4, primals_8, primals_10, primals_12, primals_14, primals_16, cat_1, getitem_1, rsqrt, view_2, div, view_10, addmm, getitem_7, rsqrt_1, add_6, getitem_9, rsqrt_2, view_13, addmm_1, getitem_11, rsqrt_3, permute_24, permute_25, permute_26, permute_34, permute_38, tangents_1, tangents_2])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
