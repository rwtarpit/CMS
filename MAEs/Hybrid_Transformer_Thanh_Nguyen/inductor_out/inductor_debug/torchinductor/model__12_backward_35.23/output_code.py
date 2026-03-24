# AOT ID: ['12_backward']
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


# kernel path: /traces/inductor_cache/v5/cv5p6lqr75fuian35rnmyrqb45nlsvhhbskcdj65esfcozac3tas.py
# Topologically Sorted Source Nodes: [input_12], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_12 => add_27, mul_37, mul_43, sub_4, unsqueeze_8, unsqueeze_9
# Graph fragment:
#   %convolution_3 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0" = PlaceHolder[target=convolution_3]
#   %getitem_9 : Tensor "f32[1, 8, 1][8, 1, 1]cuda:0" = PlaceHolder[target=getitem_9]
#   %rsqrt_4 : Tensor "f32[1, 8, 1][8, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_4]
#   %primals_34 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=primals_34]
#   %primals_35 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=primals_35]
#   %sub_4 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %getitem_9), kwargs = {})
#   %mul_37 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_4), kwargs = {})
#   %unsqueeze_8 : Tensor "f32[8, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_34, -1), kwargs = {})
#   %mul_43 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_8), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[8, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_35, -1), kwargs = {})
#   %add_27 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_43, %unsqueeze_9), kwargs = {})
#   return %add_27
triton_poi_fused__native_batch_norm_legit_functional_0 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 786432000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x1 = ((xindex // 16384) % 8)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x3), tmp8, None)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/r3/cr36iqspcxjf4aicfgv4d7mvutqrsbr2xxhcskohzm55z37h2spk.py
# Topologically Sorted Source Nodes: [view_3, input_13, mul_48, mul_49, mul_50, exp, mul_51, mul_52, add_30, mul_53, input_12, unsqueeze_10, unsqueeze_11, sum_1, sub_5, mul_54, sum_2], Original ATen: [aten.view, aten.gelu, aten.gelu_backward, aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   add_30 => add_30
#   exp => exp
#   input_12 => squeeze_12
#   input_13 => add_28, erf_3, mul_45
#   mul_48 => mul_48
#   mul_49 => mul_49
#   mul_50 => mul_50
#   mul_51 => mul_51
#   mul_52 => mul_52
#   mul_53 => mul_53
#   mul_54 => mul_54
#   sub_5 => sub_5
#   sum_1 => sum_1
#   sum_2 => sum_2
#   unsqueeze_10 => unsqueeze_10
#   unsqueeze_11 => unsqueeze_11
#   view_3 => view_3
# Graph fragment:
#   %tangents_1 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=tangents_1]
#   %add_27 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0" = PlaceHolder[target=add_27]
#   %convolution_3 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0" = PlaceHolder[target=convolution_3]
#   %getitem_9 : Tensor "f32[1, 8, 1][8, 1, 1]cuda:0" = PlaceHolder[target=getitem_9]
#   %view_3 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%tangents_1, [500, 8, 16384]), kwargs = {})
#   %mul_45 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, 0.7071067811865476), kwargs = {})
#   %erf_3 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_45,), kwargs = {})
#   %add_28 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_3, 1), kwargs = {})
#   %mul_48 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_28, 0.5), kwargs = {})
#   %mul_49 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, %add_27), kwargs = {})
#   %mul_50 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_49, -0.5), kwargs = {})
#   %exp : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_50,), kwargs = {})
#   %mul_51 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp, 0.3989422804014327), kwargs = {})
#   %mul_52 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, %mul_51), kwargs = {})
#   %add_30 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_48, %mul_52), kwargs = {})
#   %mul_53 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_3, %add_30), kwargs = {})
#   %squeeze_12 : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_9, [0, 2]), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[1, 8][8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%squeeze_12, 0), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[1, 8, 1][8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_10, 2), kwargs = {})
#   %sum_1 : Tensor "f32[8][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_53, [0, 2]), kwargs = {})
#   %sub_5 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_11), kwargs = {})
#   %mul_54 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_53, %sub_5), kwargs = {})
#   %sum_2 : Tensor "f32[8][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_54, [0, 2]), kwargs = {})
#   return %buf1,%buf3
triton_red_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_view_1 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 262144},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 2, 'num_reduction': 2, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 6944, 'r0_': 786432000}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_view_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 432
    r0_numel = 151704
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x1 = xindex // 8
    x0 = (xindex % 8)
    _tmp24 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp33 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 151704*x1
        tmp1 = tl.full([1, 1], 8192000, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (16384*x0 + 131072*((((r0_2 + 151704*x1) // 16384) % 500)) + (((r0_2 + 151704*x1) % 16384))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (16384*x0 + 131072*((((r0_2 + 151704*x1) // 16384) % 500)) + (((r0_2 + 151704*x1) % 16384))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 0.7071067811865476
        tmp6 = tmp4 * tmp5
        tmp7 = libdevice.erf(tmp6)
        tmp8 = 1.0
        tmp9 = tmp7 + tmp8
        tmp10 = 0.5
        tmp11 = tmp9 * tmp10
        tmp12 = tmp4 * tmp4
        tmp13 = -0.5
        tmp14 = tmp12 * tmp13
        tmp15 = libdevice.exp(tmp14)
        tmp16 = 0.3989422804014327
        tmp17 = tmp15 * tmp16
        tmp18 = tmp4 * tmp17
        tmp19 = tmp11 + tmp18
        tmp20 = tmp3 * tmp19
        tmp21 = tl.full(tmp20.shape, 0, tmp20.dtype)
        tmp22 = tl.where(tmp2, tmp20, tmp21)
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, R0_BLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(r0_mask & xmask, tmp25, _tmp24)
        tmp26 = tl.load(in_ptr2 + (16384*x0 + 131072*((((r0_2 + 151704*x1) // 16384) % 500)) + (((r0_2 + 151704*x1) % 16384))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp27 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tmp26 - tmp27
        tmp29 = tmp20 * tmp28
        tmp30 = tl.full(tmp29.shape, 0, tmp29.dtype)
        tmp31 = tl.where(tmp2, tmp29, tmp30)
        tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
        tmp34 = _tmp33 + tmp32
        _tmp33 = tl.where(r0_mask & xmask, tmp34, _tmp33)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tmp33 = tl.sum(_tmp33, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp24, xmask)
    tl.store(out_ptr1 + (x3), tmp33, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/t2/ct2fwfvy533ximabctlcycspfvlyj2xjo6pb6uwkaeyzq3dfdfdw.py
# Topologically Sorted Source Nodes: [view_3, input_13, mul_48, mul_49, mul_50, exp, mul_51, mul_52, add_30, mul_53, sum_1], Original ATen: [aten.view, aten.gelu, aten.gelu_backward, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   add_30 => add_30
#   exp => exp
#   input_13 => add_28, erf_3, mul_45
#   mul_48 => mul_48
#   mul_49 => mul_49
#   mul_50 => mul_50
#   mul_51 => mul_51
#   mul_52 => mul_52
#   mul_53 => mul_53
#   sum_1 => sum_1
#   view_3 => view_3
# Graph fragment:
#   %buf1 : Tensor "f32[8, 54][1, 8]cuda:0" = PlaceHolder[target=buf1]
#   %view_3 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%tangents_1, [500, 8, 16384]), kwargs = {})
#   %mul_45 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, 0.7071067811865476), kwargs = {})
#   %erf_3 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_45,), kwargs = {})
#   %add_28 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_3, 1), kwargs = {})
#   %mul_48 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_28, 0.5), kwargs = {})
#   %mul_49 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, %add_27), kwargs = {})
#   %mul_50 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_49, -0.5), kwargs = {})
#   %exp : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_50,), kwargs = {})
#   %mul_51 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp, 0.3989422804014327), kwargs = {})
#   %mul_52 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, %mul_51), kwargs = {})
#   %add_30 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_48, %mul_52), kwargs = {})
#   %mul_53 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_3, %add_30), kwargs = {})
#   %sum_1 : Tensor "f32[8][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_53, [0, 2]), kwargs = {})
#   return %sum_1
triton_per_fused_gelu_gelu_backward_native_batch_norm_backward_view_2 = async_compile.triton('triton_per_fused_gelu_gelu_backward_native_batch_norm_backward_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_gelu_backward_native_batch_norm_backward_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1792, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_gelu_gelu_backward_native_batch_norm_backward_view_2(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 8
    r0_numel = 54
    R0_BLOCK: tl.constexpr = 64
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
    tmp0 = tl.load(in_ptr0 + (x0 + 8*r0_1), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/lh/clhvoibgayoq5yll7ttkvu4c7bsklsbfdzmizynqlavf54ylieg3.py
# Topologically Sorted Source Nodes: [view_3, input_13, mul_48, mul_49, mul_50, exp, mul_51, mul_52, add_30, mul_53, input_12, unsqueeze_10, unsqueeze_11, sub_5, mul_54, sum_2, mul_62], Original ATen: [aten.view, aten.gelu, aten.gelu_backward, aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   add_30 => add_30
#   exp => exp
#   input_12 => squeeze_12, squeeze_13
#   input_13 => add_28, erf_3, mul_45
#   mul_48 => mul_48
#   mul_49 => mul_49
#   mul_50 => mul_50
#   mul_51 => mul_51
#   mul_52 => mul_52
#   mul_53 => mul_53
#   mul_54 => mul_54
#   mul_62 => mul_62
#   sub_5 => sub_5
#   sum_2 => sum_2
#   unsqueeze_10 => unsqueeze_10
#   unsqueeze_11 => unsqueeze_11
#   view_3 => view_3
# Graph fragment:
#   %buf3 : Tensor "f32[8, 54][1, 8]cuda:0" = PlaceHolder[target=buf3]
#   %sum_2 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=sum_2]
#   %rsqrt_4 : Tensor "f32[1, 8, 1][8, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_4]
#   %view_3 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%tangents_1, [500, 8, 16384]), kwargs = {})
#   %mul_45 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, 0.7071067811865476), kwargs = {})
#   %erf_3 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_45,), kwargs = {})
#   %add_28 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_3, 1), kwargs = {})
#   %mul_48 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_28, 0.5), kwargs = {})
#   %mul_49 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, %add_27), kwargs = {})
#   %mul_50 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_49, -0.5), kwargs = {})
#   %exp : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_50,), kwargs = {})
#   %mul_51 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp, 0.3989422804014327), kwargs = {})
#   %mul_52 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, %mul_51), kwargs = {})
#   %add_30 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_48, %mul_52), kwargs = {})
#   %mul_53 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_3, %add_30), kwargs = {})
#   %squeeze_12 : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_9, [0, 2]), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[1, 8][8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%squeeze_12, 0), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[1, 8, 1][8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_10, 2), kwargs = {})
#   %sub_5 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_11), kwargs = {})
#   %mul_54 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_53, %sub_5), kwargs = {})
#   %sum_2 : Tensor "f32[8][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_54, [0, 2]), kwargs = {})
#   %squeeze_13 : Tensor "f32[8][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.squeeze.dims](args = (%rsqrt_4, [0, 2]), kwargs = {})
#   %mul_62 : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_2, %squeeze_13), kwargs = {})
#   return %sum_2,%mul_62
triton_per_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_view_3 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1888, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_view_3(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 8
    r0_numel = 54
    R0_BLOCK: tl.constexpr = 64
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
    tmp0 = tl.load(in_ptr0 + (x0 + 8*r0_1), r0_mask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/52/c52ukn6uncjpt5rsvigg3f5h4bgc4fuvcyaobpsys3btjyuhrgsb.py
# Topologically Sorted Source Nodes: [view_3, input_13, mul_48, mul_49, mul_50, exp, mul_51, mul_52, add_30, mul_53, input_12, unsqueeze_10, unsqueeze_11, sub_5, mul_55, unsqueeze_12, unsqueeze_13, mul_56, mul_57, mul_58, unsqueeze_14, unsqueeze_15, mul_59, unsqueeze_16, unsqueeze_17, mul_60, sub_7, sub_8, mul_61], Original ATen: [aten.view, aten.gelu, aten.gelu_backward, aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   add_30 => add_30
#   exp => exp
#   input_12 => squeeze_12, squeeze_13
#   input_13 => add_28, erf_3, mul_45
#   mul_48 => mul_48
#   mul_49 => mul_49
#   mul_50 => mul_50
#   mul_51 => mul_51
#   mul_52 => mul_52
#   mul_53 => mul_53
#   mul_55 => mul_55
#   mul_56 => mul_56
#   mul_57 => mul_57
#   mul_58 => mul_58
#   mul_59 => mul_59
#   mul_60 => mul_60
#   mul_61 => mul_61
#   sub_5 => sub_5
#   sub_7 => sub_7
#   sub_8 => sub_8
#   unsqueeze_10 => unsqueeze_10
#   unsqueeze_11 => unsqueeze_11
#   unsqueeze_12 => unsqueeze_12
#   unsqueeze_13 => unsqueeze_13
#   unsqueeze_14 => unsqueeze_14
#   unsqueeze_15 => unsqueeze_15
#   unsqueeze_16 => unsqueeze_16
#   unsqueeze_17 => unsqueeze_17
#   view_3 => view_3
# Graph fragment:
#   %tangents_1 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=tangents_1]
#   %add_27 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0" = PlaceHolder[target=add_27]
#   %convolution_3 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0" = PlaceHolder[target=convolution_3]
#   %getitem_9 : Tensor "f32[1, 8, 1][8, 1, 1]cuda:0" = PlaceHolder[target=getitem_9]
#   %sum_2 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=sum_2]
#   %rsqrt_4 : Tensor "f32[1, 8, 1][8, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_4]
#   %sum_1 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=sum_1]
#   %sub_8 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0" = PlaceHolder[target=sub_8]
#   %primals_34 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=primals_34]
#   %view_3 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%tangents_1, [500, 8, 16384]), kwargs = {})
#   %mul_45 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, 0.7071067811865476), kwargs = {})
#   %erf_3 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_45,), kwargs = {})
#   %add_28 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_3, 1), kwargs = {})
#   %mul_48 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_28, 0.5), kwargs = {})
#   %mul_49 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, %add_27), kwargs = {})
#   %mul_50 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_49, -0.5), kwargs = {})
#   %exp : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_50,), kwargs = {})
#   %mul_51 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp, 0.3989422804014327), kwargs = {})
#   %mul_52 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, %mul_51), kwargs = {})
#   %add_30 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_48, %mul_52), kwargs = {})
#   %mul_53 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_3, %add_30), kwargs = {})
#   %squeeze_12 : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_9, [0, 2]), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[1, 8][8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%squeeze_12, 0), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[1, 8, 1][8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_10, 2), kwargs = {})
#   %sub_5 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_11), kwargs = {})
#   %mul_55 : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_1, 1.220703125e-07), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[1, 8][8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_55, 0), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[1, 8, 1][8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_12, 2), kwargs = {})
#   %mul_56 : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_2, 1.220703125e-07), kwargs = {})
#   %squeeze_13 : Tensor "f32[8][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.squeeze.dims](args = (%rsqrt_4, [0, 2]), kwargs = {})
#   %mul_57 : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_13, %squeeze_13), kwargs = {})
#   %mul_58 : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_56, %mul_57), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[1, 8][8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_58, 0), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[1, 8, 1][8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_14, 2), kwargs = {})
#   %mul_59 : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_13, %primals_34), kwargs = {})
#   %unsqueeze_16 : Tensor "f32[1, 8][8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_59, 0), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[1, 8, 1][8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_16, 2), kwargs = {})
#   %mul_60 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_15), kwargs = {})
#   %sub_7 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_53, %mul_60), kwargs = {})
#   %sub_8 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_7, %unsqueeze_13), kwargs = {})
#   %mul_61 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_17), kwargs = {})
#   return %sub_8,%mul_61
triton_poi_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_view_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_view_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1310720000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_view_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x1 = ((xindex // 16384) % 8)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_out_ptr0 + (x3), None)
    tmp18 = tl.load(in_ptr1 + (x3), None)
    tmp19 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp2 = 0.7071067811865476
    tmp3 = tmp1 * tmp2
    tmp4 = libdevice.erf(tmp3)
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp1 * tmp1
    tmp10 = -0.5
    tmp11 = tmp9 * tmp10
    tmp12 = libdevice.exp(tmp11)
    tmp13 = 0.3989422804014327
    tmp14 = tmp12 * tmp13
    tmp15 = tmp1 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tmp0 * tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = 1.220703125e-07
    tmp23 = tmp21 * tmp22
    tmp25 = tmp24 * tmp24
    tmp26 = tmp23 * tmp25
    tmp27 = tmp20 * tmp26
    tmp28 = tmp17 - tmp27
    tmp30 = tmp29 * tmp22
    tmp31 = tmp28 - tmp30
    tmp33 = tmp24 * tmp32
    tmp34 = tmp31 * tmp33
    tl.store(in_out_ptr0 + (x3), tmp34, None)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/ki/ckih7kizsadq5nukupbwdryvugoxvoz6vs4t7fx3pli7dpxricd4.py
# Topologically Sorted Source Nodes: [sum_3], Original ATen: [aten.convolution_backward]
# Source node to ATen node mapping:
#   sum_3 => sum_3
# Graph fragment:
#   %mul_61 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0" = PlaceHolder[target=mul_61]
#   %sum_3 : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_61, [0, 2]), kwargs = {})
#   return %buf8
triton_red_fused_convolution_backward_5 = async_compile.triton('triton_red_fused_convolution_backward_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 262144},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 3456, 'r0_': 262144000}}
)
@triton.jit
def triton_red_fused_convolution_backward_5(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 432
    r0_numel = 151704
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x1 = xindex // 8
    x0 = (xindex % 8)
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 151704*x1
        tmp1 = tl.full([1, 1], 8192000, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (16384*x0 + 131072*((((r0_2 + 151704*x1) // 16384) % 500)) + (((r0_2 + 151704*x1) % 16384))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(r0_mask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/ie/cie5qb3khnjbmmjgdzzh5arwosyfze7y24ic7i7mcflhzx4dgirv.py
# Topologically Sorted Source Nodes: [input_9], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_9 => add_21, mul_27, mul_33, sub_3, unsqueeze_6, unsqueeze_7
# Graph fragment:
#   %convolution_2 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0" = PlaceHolder[target=convolution_2]
#   %getitem_7 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=getitem_7]
#   %rsqrt_3 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_3]
#   %primals_27 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=primals_27]
#   %primals_28 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=primals_28]
#   %sub_3 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %getitem_7), kwargs = {})
#   %mul_27 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_27, -1), kwargs = {})
#   %mul_33 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_27, %unsqueeze_6), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_28, -1), kwargs = {})
#   %add_21 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_33, %unsqueeze_7), kwargs = {})
#   return %add_21
triton_poi_fused__native_batch_norm_legit_functional_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 6291456000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x1 = ((xindex // 16384) % 64)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x3), tmp8, None)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/64/c64mtxl56jkr2lka6xkuz4vwc5tx3y6vwknsvl5n4xg5eevh4vms.py
# Topologically Sorted Source Nodes: [input_10, mul_64, mul_65, mul_66, exp_1, mul_67, mul_68, add_32, mul_69, input_9, unsqueeze_18, unsqueeze_19, sum_4, sub_9, mul_70, sum_5], Original ATen: [aten.gelu, aten.gelu_backward, aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   add_32 => add_32
#   exp_1 => exp_1
#   input_10 => add_22, erf_2, mul_35
#   input_9 => squeeze_9
#   mul_64 => mul_64
#   mul_65 => mul_65
#   mul_66 => mul_66
#   mul_67 => mul_67
#   mul_68 => mul_68
#   mul_69 => mul_69
#   mul_70 => mul_70
#   sub_9 => sub_9
#   sum_4 => sum_4
#   sum_5 => sum_5
#   unsqueeze_18 => unsqueeze_18
#   unsqueeze_19 => unsqueeze_19
# Graph fragment:
#   %getitem_10 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0" = PlaceHolder[target=getitem_10]
#   %add_21 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0" = PlaceHolder[target=add_21]
#   %convolution_2 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0" = PlaceHolder[target=convolution_2]
#   %getitem_7 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=getitem_7]
#   %mul_35 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, 0.7071067811865476), kwargs = {})
#   %erf_2 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_35,), kwargs = {})
#   %add_22 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_2, 1), kwargs = {})
#   %mul_64 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, 0.5), kwargs = {})
#   %mul_65 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %add_21), kwargs = {})
#   %mul_66 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_65, -0.5), kwargs = {})
#   %exp_1 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_66,), kwargs = {})
#   %mul_67 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_1, 0.3989422804014327), kwargs = {})
#   %mul_68 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %mul_67), kwargs = {})
#   %add_32 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_64, %mul_68), kwargs = {})
#   %mul_69 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_10, %add_32), kwargs = {})
#   %squeeze_9 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_7, [0, 2]), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[1, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%squeeze_9, 0), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, 2), kwargs = {})
#   %sum_4 : Tensor "f32[64][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_69, [0, 2]), kwargs = {})
#   %sub_9 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_19), kwargs = {})
#   %mul_70 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_69, %sub_9), kwargs = {})
#   %sum_5 : Tensor "f32[64][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_70, [0, 2]), kwargs = {})
#   return %buf14,%buf16
triton_red_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_7 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 131072},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 2, 'num_reduction': 2, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 64768, 'r0_': 6291456000}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4032
    r0_numel = 130032
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x1 = xindex // 64
    x0 = (xindex % 64)
    _tmp24 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp33 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 130032*x1
        tmp1 = tl.full([1, 1], 8192000, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (16384*x0 + 1048576*((((r0_2 + 130032*x1) // 16384) % 500)) + (((r0_2 + 130032*x1) % 16384))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (16384*x0 + 1048576*((((r0_2 + 130032*x1) // 16384) % 500)) + (((r0_2 + 130032*x1) % 16384))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 0.7071067811865476
        tmp6 = tmp4 * tmp5
        tmp7 = libdevice.erf(tmp6)
        tmp8 = 1.0
        tmp9 = tmp7 + tmp8
        tmp10 = 0.5
        tmp11 = tmp9 * tmp10
        tmp12 = tmp4 * tmp4
        tmp13 = -0.5
        tmp14 = tmp12 * tmp13
        tmp15 = libdevice.exp(tmp14)
        tmp16 = 0.3989422804014327
        tmp17 = tmp15 * tmp16
        tmp18 = tmp4 * tmp17
        tmp19 = tmp11 + tmp18
        tmp20 = tmp3 * tmp19
        tmp21 = tl.full(tmp20.shape, 0, tmp20.dtype)
        tmp22 = tl.where(tmp2, tmp20, tmp21)
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, R0_BLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(r0_mask & xmask, tmp25, _tmp24)
        tmp26 = tl.load(in_ptr2 + (16384*x0 + 1048576*((((r0_2 + 130032*x1) // 16384) % 500)) + (((r0_2 + 130032*x1) % 16384))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp27 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tmp26 - tmp27
        tmp29 = tmp20 * tmp28
        tmp30 = tl.full(tmp29.shape, 0, tmp29.dtype)
        tmp31 = tl.where(tmp2, tmp29, tmp30)
        tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
        tmp34 = _tmp33 + tmp32
        _tmp33 = tl.where(r0_mask & xmask, tmp34, _tmp33)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tmp33 = tl.sum(_tmp33, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp24, xmask)
    tl.store(out_ptr1 + (x3), tmp33, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/mi/cmisi7vu6yiu6jrr7r7bxk5pdyp5gg3wlzlpqzf4hprnyipfxw4t.py
# Topologically Sorted Source Nodes: [input_10, mul_64, mul_65, mul_66, exp_1, mul_67, mul_68, add_32, mul_69, sum_4], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   add_32 => add_32
#   exp_1 => exp_1
#   input_10 => add_22, erf_2, mul_35
#   mul_64 => mul_64
#   mul_65 => mul_65
#   mul_66 => mul_66
#   mul_67 => mul_67
#   mul_68 => mul_68
#   mul_69 => mul_69
#   sum_4 => sum_4
# Graph fragment:
#   %buf14 : Tensor "f32[64, 63][1, 64]cuda:0" = PlaceHolder[target=buf14]
#   %mul_35 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, 0.7071067811865476), kwargs = {})
#   %erf_2 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_35,), kwargs = {})
#   %add_22 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_2, 1), kwargs = {})
#   %mul_64 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, 0.5), kwargs = {})
#   %mul_65 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %add_21), kwargs = {})
#   %mul_66 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_65, -0.5), kwargs = {})
#   %exp_1 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_66,), kwargs = {})
#   %mul_67 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_1, 0.3989422804014327), kwargs = {})
#   %mul_68 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %mul_67), kwargs = {})
#   %add_32 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_64, %mul_68), kwargs = {})
#   %mul_69 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_10, %add_32), kwargs = {})
#   %sum_4 : Tensor "f32[64][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_69, [0, 2]), kwargs = {})
#   return %sum_4
triton_per_fused_gelu_gelu_backward_native_batch_norm_backward_8 = async_compile.triton('triton_per_fused_gelu_gelu_backward_native_batch_norm_backward_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_gelu_backward_native_batch_norm_backward_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 16640, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_gelu_gelu_backward_native_batch_norm_backward_8(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 63
    R0_BLOCK: tl.constexpr = 64
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
    tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_1), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/5x/c5xbyqpmcb33ffihod3dh6doyxfa2ocvvhcwodgykxwzc6dwf6ea.py
# Topologically Sorted Source Nodes: [input_10, mul_64, mul_65, mul_66, exp_1, mul_67, mul_68, add_32, mul_69, input_9, unsqueeze_18, unsqueeze_19, sub_9, mul_70, sum_5, mul_78], Original ATen: [aten.gelu, aten.gelu_backward, aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   add_32 => add_32
#   exp_1 => exp_1
#   input_10 => add_22, erf_2, mul_35
#   input_9 => squeeze_10, squeeze_9
#   mul_64 => mul_64
#   mul_65 => mul_65
#   mul_66 => mul_66
#   mul_67 => mul_67
#   mul_68 => mul_68
#   mul_69 => mul_69
#   mul_70 => mul_70
#   mul_78 => mul_78
#   sub_9 => sub_9
#   sum_5 => sum_5
#   unsqueeze_18 => unsqueeze_18
#   unsqueeze_19 => unsqueeze_19
# Graph fragment:
#   %buf16 : Tensor "f32[64, 63][1, 64]cuda:0" = PlaceHolder[target=buf16]
#   %sum_5 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=sum_5]
#   %rsqrt_3 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_3]
#   %mul_35 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, 0.7071067811865476), kwargs = {})
#   %erf_2 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_35,), kwargs = {})
#   %add_22 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_2, 1), kwargs = {})
#   %mul_64 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, 0.5), kwargs = {})
#   %mul_65 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %add_21), kwargs = {})
#   %mul_66 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_65, -0.5), kwargs = {})
#   %exp_1 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_66,), kwargs = {})
#   %mul_67 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_1, 0.3989422804014327), kwargs = {})
#   %mul_68 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %mul_67), kwargs = {})
#   %add_32 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_64, %mul_68), kwargs = {})
#   %mul_69 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_10, %add_32), kwargs = {})
#   %squeeze_9 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_7, [0, 2]), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[1, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%squeeze_9, 0), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, 2), kwargs = {})
#   %sub_9 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_19), kwargs = {})
#   %mul_70 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_69, %sub_9), kwargs = {})
#   %sum_5 : Tensor "f32[64][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_70, [0, 2]), kwargs = {})
#   %squeeze_10 : Tensor "f32[64][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.squeeze.dims](args = (%rsqrt_3, [0, 2]), kwargs = {})
#   %mul_78 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_5, %squeeze_10), kwargs = {})
#   return %sum_5,%mul_78
triton_per_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_9 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 17408, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_9(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 64
    r0_numel = 63
    R0_BLOCK: tl.constexpr = 64
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
    tmp0 = tl.load(in_ptr0 + (x0 + 64*r0_1), r0_mask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/jb/cjb4u6bd65lrsklyt6aainhp6krjsywejjpofg4reeymr7juzejz.py
# Topologically Sorted Source Nodes: [input_10, mul_64, mul_65, mul_66, exp_1, mul_67, mul_68, add_32, mul_69, input_9, unsqueeze_18, unsqueeze_19, sub_9, mul_71, unsqueeze_20, unsqueeze_21, mul_72, mul_73, mul_74, unsqueeze_22, unsqueeze_23, mul_75, unsqueeze_24, unsqueeze_25, mul_76, sub_11, sub_12, mul_77], Original ATen: [aten.gelu, aten.gelu_backward, aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   add_32 => add_32
#   exp_1 => exp_1
#   input_10 => add_22, erf_2, mul_35
#   input_9 => squeeze_10, squeeze_9
#   mul_64 => mul_64
#   mul_65 => mul_65
#   mul_66 => mul_66
#   mul_67 => mul_67
#   mul_68 => mul_68
#   mul_69 => mul_69
#   mul_71 => mul_71
#   mul_72 => mul_72
#   mul_73 => mul_73
#   mul_74 => mul_74
#   mul_75 => mul_75
#   mul_76 => mul_76
#   mul_77 => mul_77
#   sub_11 => sub_11
#   sub_12 => sub_12
#   sub_9 => sub_9
#   unsqueeze_18 => unsqueeze_18
#   unsqueeze_19 => unsqueeze_19
#   unsqueeze_20 => unsqueeze_20
#   unsqueeze_21 => unsqueeze_21
#   unsqueeze_22 => unsqueeze_22
#   unsqueeze_23 => unsqueeze_23
#   unsqueeze_24 => unsqueeze_24
#   unsqueeze_25 => unsqueeze_25
# Graph fragment:
#   %getitem_10 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0" = PlaceHolder[target=getitem_10]
#   %add_21 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0" = PlaceHolder[target=add_21]
#   %convolution_2 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0" = PlaceHolder[target=convolution_2]
#   %getitem_7 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=getitem_7]
#   %sum_5 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=sum_5]
#   %rsqrt_3 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_3]
#   %sum_4 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=sum_4]
#   %sub_12 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0" = PlaceHolder[target=sub_12]
#   %primals_27 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=primals_27]
#   %mul_35 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, 0.7071067811865476), kwargs = {})
#   %erf_2 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_35,), kwargs = {})
#   %add_22 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_2, 1), kwargs = {})
#   %mul_64 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, 0.5), kwargs = {})
#   %mul_65 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %add_21), kwargs = {})
#   %mul_66 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_65, -0.5), kwargs = {})
#   %exp_1 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_66,), kwargs = {})
#   %mul_67 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_1, 0.3989422804014327), kwargs = {})
#   %mul_68 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %mul_67), kwargs = {})
#   %add_32 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_64, %mul_68), kwargs = {})
#   %mul_69 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_10, %add_32), kwargs = {})
#   %squeeze_9 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_7, [0, 2]), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[1, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%squeeze_9, 0), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, 2), kwargs = {})
#   %sub_9 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_19), kwargs = {})
#   %mul_71 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_4, 1.220703125e-07), kwargs = {})
#   %unsqueeze_20 : Tensor "f32[1, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_71, 0), kwargs = {})
#   %unsqueeze_21 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_20, 2), kwargs = {})
#   %mul_72 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_5, 1.220703125e-07), kwargs = {})
#   %squeeze_10 : Tensor "f32[64][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.squeeze.dims](args = (%rsqrt_3, [0, 2]), kwargs = {})
#   %mul_73 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_10, %squeeze_10), kwargs = {})
#   %mul_74 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_72, %mul_73), kwargs = {})
#   %unsqueeze_22 : Tensor "f32[1, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_74, 0), kwargs = {})
#   %unsqueeze_23 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_22, 2), kwargs = {})
#   %mul_75 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_10, %primals_27), kwargs = {})
#   %unsqueeze_24 : Tensor "f32[1, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_75, 0), kwargs = {})
#   %unsqueeze_25 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_24, 2), kwargs = {})
#   %mul_76 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_23), kwargs = {})
#   %sub_11 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_69, %mul_76), kwargs = {})
#   %sub_12 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_11, %unsqueeze_21), kwargs = {})
#   %mul_77 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_25), kwargs = {})
#   return %sub_12,%mul_77
triton_poi_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 10485760000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x1 = ((xindex // 16384) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp18 = tl.load(in_ptr1 + (x3), None)
    tmp19 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp2 = 0.7071067811865476
    tmp3 = tmp1 * tmp2
    tmp4 = libdevice.erf(tmp3)
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp1 * tmp1
    tmp10 = -0.5
    tmp11 = tmp9 * tmp10
    tmp12 = libdevice.exp(tmp11)
    tmp13 = 0.3989422804014327
    tmp14 = tmp12 * tmp13
    tmp15 = tmp1 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tmp0 * tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = 1.220703125e-07
    tmp23 = tmp21 * tmp22
    tmp25 = tmp24 * tmp24
    tmp26 = tmp23 * tmp25
    tmp27 = tmp20 * tmp26
    tmp28 = tmp17 - tmp27
    tmp30 = tmp29 * tmp22
    tmp31 = tmp28 - tmp30
    tmp33 = tmp24 * tmp32
    tmp34 = tmp31 * tmp33
    tl.store(in_out_ptr0 + (x3), tmp34, None)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/yz/cyz7bbebe3nboe35igbmah3skjlaacn563snye7to42423oqhest.py
# Topologically Sorted Source Nodes: [sum_6], Original ATen: [aten.convolution_backward]
# Source node to ATen node mapping:
#   sum_6 => sum_6
# Graph fragment:
#   %mul_77 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0" = PlaceHolder[target=mul_77]
#   %sum_6 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_77, [0, 2]), kwargs = {})
#   return %buf21
triton_red_fused_convolution_backward_11 = async_compile.triton('triton_red_fused_convolution_backward_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 131072},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 32256, 'r0_': 2097152000}}
)
@triton.jit
def triton_red_fused_convolution_backward_11(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4032
    r0_numel = 130032
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x1 = xindex // 64
    x0 = (xindex % 64)
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 130032*x1
        tmp1 = tl.full([1, 1], 8192000, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (16384*x0 + 1048576*((((r0_2 + 130032*x1) // 16384) % 500)) + (((r0_2 + 130032*x1) % 16384))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(r0_mask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/ck/ccktpjgzikbzxz3xsljerdb4v3duqnrrw3gucv5iucix776ker5m.py
# Topologically Sorted Source Nodes: [sum_13], Original ATen: [aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   sum_13 => sum_13
# Graph fragment:
#   %getitem_19 : Tensor "f32[500, 4, 16384][65536, 16384, 1]cuda:0" = PlaceHolder[target=getitem_19]
#   %sum_13 : Tensor "f32[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%getitem_19, [0, 2]), kwargs = {})
#   return %buf52
triton_red_fused_native_batch_norm_backward_12 = async_compile.triton('triton_red_fused_native_batch_norm_backward_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 131072},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 3200, 'r0_': 131072000}}
)
@triton.jit
def triton_red_fused_native_batch_norm_backward_12(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 400
    r0_numel = 81920
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 4)
    x1 = xindex // 4
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (16384*x0 + 65536*(r0_2 // 16384) + 327680*x1 + ((r0_2 % 16384))), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/du/cduisooad65dvdtln2gaglhzxj7ovapjg2jitr7kfet3nha4durn.py
# Topologically Sorted Source Nodes: [sum_13], Original ATen: [aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   sum_13 => sum_13
# Graph fragment:
#   %buf52 : Tensor "f32[4, 100][1, 4]cuda:0" = PlaceHolder[target=buf52]
#   %sum_13 : Tensor "f32[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%getitem_19, [0, 2]), kwargs = {})
#   return %sum_13
triton_per_fused_native_batch_norm_backward_13 = async_compile.triton('triton_per_fused_native_batch_norm_backward_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 816, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_batch_norm_backward_13(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4
    r0_numel = 100
    R0_BLOCK: tl.constexpr = 128
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
    tmp0 = tl.load(in_ptr0 + (x0 + 4*r0_1), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/uj/cujt4dcv7waosaels3hno7gcku76ymvunbxbubxawgss47maihgy.py
# Topologically Sorted Source Nodes: [view, U, input_1, sub_21, mul_111, sum_14], Original ATen: [aten.view, aten.transpose, aten.clone, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   U => permute
#   input_1 => clone
#   mul_111 => mul_111
#   sub_21 => sub_21
#   sum_14 => sum_14
#   view => view
# Graph fragment:
#   %getitem_19 : Tensor "f32[500, 4, 16384][65536, 16384, 1]cuda:0" = PlaceHolder[target=getitem_19]
#   %primals_2 : Tensor "f32[500, 128, 128, 4][65536, 512, 4, 1]cuda:0" = PlaceHolder[target=primals_2]
#   %unsqueeze_43 : Tensor "f32[1, 4, 1][4, 1, 1]cuda:0" = PlaceHolder[target=unsqueeze_43]
#   %view : Tensor "f32[500, 16384, 4][65536, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_2, [500, 16384, 4]), kwargs = {})
#   %permute : Tensor "f32[500, 4, 16384][65536, 1, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view, [0, 2, 1]), kwargs = {})
#   %clone : Tensor "f32[500, 4, 16384][65536, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %sub_21 : Tensor "f32[500, 4, 16384][65536, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone, %unsqueeze_43), kwargs = {})
#   %mul_111 : Tensor "f32[500, 4, 16384][65536, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_19, %sub_21), kwargs = {})
#   %sum_14 : Tensor "f32[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_111, [0, 2]), kwargs = {})
#   return %buf54
triton_red_fused_clone_native_batch_norm_backward_transpose_view_14 = async_compile.triton('triton_red_fused_clone_native_batch_norm_backward_transpose_view_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'y': 4, 'x': 32768, 'r0_': 512},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_clone_native_batch_norm_backward_transpose_view_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 65536008, 'x': 884736, 'r0_': 131072000}}
)
@triton.jit
def triton_red_fused_clone_native_batch_norm_backward_transpose_view_14(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, r0_numel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    ynumel = 4
    xnumel = 27648
    r0_numel = 297
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, None, :]
    rbase = r0_base
    x1 = (xindex % 32)
    x2 = xindex // 32
    y0 = yindex
    _tmp17 = tl.full([YBLOCK, XBLOCK, R0_BLOCK], 0, tl.float32)
    x4 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp0 = r0_3 + 297*x1
        tmp1 = tl.full([1, 1, 1], 9482, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.broadcast_to(r0_3 + 297*x1 + 9482*x2, [YBLOCK, XBLOCK, R0_BLOCK])
        tmp4 = tl.full([1, 1, 1], 8192000, tl.int32)
        tmp5 = tmp3 < tmp4
        tmp6 = tmp5 & tmp2
        tmp7 = tl.load(in_ptr0 + (16384*y0 + 65536*((((r0_3 + 297*x1 + 9482*x2) // 16384) % 500)) + (((r0_3 + 297*x1 + 9482*x2) % 16384))), r0_mask & tmp6 & xmask & ymask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr1 + (y0 + 4*(((r0_3 + 297*x1 + 9482*x2) % 8192000))), r0_mask & tmp6 & xmask & ymask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [YBLOCK, XBLOCK, R0_BLOCK])), r0_mask & tmp6 & xmask & ymask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp7 * tmp10
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp6, tmp11, tmp12)
        tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
        tmp15 = tl.where(tmp2, tmp13, tmp14)
        tmp16 = tl.broadcast_to(tmp15, [YBLOCK, XBLOCK, R0_BLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(r0_mask & xmask & ymask, tmp18, _tmp17)
    tmp17 = tl.sum(_tmp17, 2)[:, :, None]
    tl.store(out_ptr0 + (x4 + 27648*y0), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/3v/c3vxle2xuyhiyzyfpzfau4c4iwkxcqovmat377agaajfsylulpvi.py
# Topologically Sorted Source Nodes: [view, U, input_1, sub_21, mul_111, sum_14], Original ATen: [aten.view, aten.transpose, aten.clone, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   U => permute
#   input_1 => clone
#   mul_111 => mul_111
#   sub_21 => sub_21
#   sum_14 => sum_14
#   view => view
# Graph fragment:
#   %buf54 : Tensor "f32[4, 864, 32][27648, 32, 1]cuda:0" = PlaceHolder[target=buf54]
#   %view : Tensor "f32[500, 16384, 4][65536, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_2, [500, 16384, 4]), kwargs = {})
#   %permute : Tensor "f32[500, 4, 16384][65536, 1, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view, [0, 2, 1]), kwargs = {})
#   %clone : Tensor "f32[500, 4, 16384][65536, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %sub_21 : Tensor "f32[500, 4, 16384][65536, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone, %unsqueeze_43), kwargs = {})
#   %mul_111 : Tensor "f32[500, 4, 16384][65536, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_19, %sub_21), kwargs = {})
#   %sum_14 : Tensor "f32[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_111, [0, 2]), kwargs = {})
#   return %buf55
triton_per_fused_clone_native_batch_norm_backward_transpose_view_15 = async_compile.triton('triton_per_fused_clone_native_batch_norm_backward_transpose_view_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clone_native_batch_norm_backward_transpose_view_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 27648, 'r0_': 442368}}
)
@triton.jit
def triton_per_fused_clone_native_batch_norm_backward_transpose_view_15(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 3456
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 32*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/ek/cekrifuhdzx3uqzucuiexqse52oo65v45exwo3yuz76byrlkfy3u.py
# Topologically Sorted Source Nodes: [view, U, input_1, sub_21, mul_111, sum_14, mul_119], Original ATen: [aten.view, aten.transpose, aten.clone, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   U => permute
#   input_1 => clone
#   mul_111 => mul_111
#   mul_119 => mul_119
#   sub_21 => sub_21
#   sum_14 => sum_14
#   view => view
# Graph fragment:
#   %buf55 : Tensor "f32[4, 864][864, 1]cuda:0" = PlaceHolder[target=buf55]
#   %sum_14 : Tensor "f32[4][1]cuda:0" = PlaceHolder[target=sum_14]
#   %squeeze_1 : Tensor "f32[4][1]cuda:0" = PlaceHolder[target=squeeze_1]
#   %view : Tensor "f32[500, 16384, 4][65536, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_2, [500, 16384, 4]), kwargs = {})
#   %permute : Tensor "f32[500, 4, 16384][65536, 1, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view, [0, 2, 1]), kwargs = {})
#   %clone : Tensor "f32[500, 4, 16384][65536, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %sub_21 : Tensor "f32[500, 4, 16384][65536, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone, %unsqueeze_43), kwargs = {})
#   %mul_111 : Tensor "f32[500, 4, 16384][65536, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_19, %sub_21), kwargs = {})
#   %sum_14 : Tensor "f32[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_111, [0, 2]), kwargs = {})
#   %mul_119 : Tensor "f32[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_14, %squeeze_1), kwargs = {})
#   return %sum_14,%mul_119
triton_per_fused_clone_native_batch_norm_backward_transpose_view_16 = async_compile.triton('triton_per_fused_clone_native_batch_norm_backward_transpose_view_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clone_native_batch_norm_backward_transpose_view_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 24, 'r0_': 13824}}
)
@triton.jit
def triton_per_fused_clone_native_batch_norm_backward_transpose_view_16(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4
    r0_numel = 864
    R0_BLOCK: tl.constexpr = 1024
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 864*x0), r0_mask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tmp6 = tmp4 * tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
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
        primals_2, primals_8, primals_13, primals_14, primals_15, primals_20, primals_21, primals_22, primals_27, primals_28, primals_29, primals_34, primals_35, squeeze_1, add_4, convolution, getitem_3, rsqrt_1, mul_16, convolution_1, getitem_5, rsqrt_2, mul_26, convolution_2, getitem_7, rsqrt_3, mul_36, convolution_3, getitem_9, rsqrt_4, unsqueeze_43, tangents_1 = args
        args.clear()
        assert_size_stride(primals_2, (500, 128, 128, 4), (65536, 512, 4, 1))
        assert_size_stride(primals_8, (64, 4, 1), (4, 1, 1))
        assert_size_stride(primals_13, (64, ), (1, ))
        assert_size_stride(primals_14, (64, ), (1, ))
        assert_size_stride(primals_15, (64, 64, 1), (64, 1, 1))
        assert_size_stride(primals_20, (64, ), (1, ))
        assert_size_stride(primals_21, (64, ), (1, ))
        assert_size_stride(primals_22, (64, 64, 1), (64, 1, 1))
        assert_size_stride(primals_27, (64, ), (1, ))
        assert_size_stride(primals_28, (64, ), (1, ))
        assert_size_stride(primals_29, (8, 64, 1), (64, 1, 1))
        assert_size_stride(primals_34, (8, ), (1, ))
        assert_size_stride(primals_35, (8, ), (1, ))
        assert_size_stride(squeeze_1, (4, ), (1, ))
        assert_size_stride(add_4, (500, 4, 16384), (65536, 16384, 1))
        assert_size_stride(convolution, (500, 64, 16384), (1048576, 16384, 1))
        assert_size_stride(getitem_3, (1, 64, 1), (64, 1, 1))
        assert_size_stride(rsqrt_1, (1, 64, 1), (64, 1, 1))
        assert_size_stride(mul_16, (500, 64, 16384), (1048576, 16384, 1))
        assert_size_stride(convolution_1, (500, 64, 16384), (1048576, 16384, 1))
        assert_size_stride(getitem_5, (1, 64, 1), (64, 1, 1))
        assert_size_stride(rsqrt_2, (1, 64, 1), (64, 1, 1))
        assert_size_stride(mul_26, (500, 64, 16384), (1048576, 16384, 1))
        assert_size_stride(convolution_2, (500, 64, 16384), (1048576, 16384, 1))
        assert_size_stride(getitem_7, (1, 64, 1), (64, 1, 1))
        assert_size_stride(rsqrt_3, (1, 64, 1), (64, 1, 1))
        assert_size_stride(mul_36, (500, 64, 16384), (1048576, 16384, 1))
        assert_size_stride(convolution_3, (500, 8, 16384), (131072, 16384, 1))
        assert_size_stride(getitem_9, (1, 8, 1), (8, 1, 1))
        assert_size_stride(rsqrt_4, (1, 8, 1), (8, 1, 1))
        assert_size_stride(unsqueeze_43, (1, 4, 1), (4, 1, 1))
        assert_size_stride(tangents_1, (4000, 128, 128), (16384, 128, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((500, 8, 16384), (131072, 16384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten._native_batch_norm_legit_functional]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_functional_0:282
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_0.run(convolution_3, getitem_9, rsqrt_4, primals_34, primals_35, buf0, 65536000, stream=stream0)
            del primals_35
            buf1 = empty_strided_cuda((8, 54), (1, 8), torch.float32)
            buf3 = empty_strided_cuda((8, 54), (1, 8), torch.float32)
            # Topologically Sorted Source Nodes: [view_3, input_13, mul_48, mul_49, mul_50, exp, mul_51, mul_52, add_30, mul_53, input_12, unsqueeze_10, unsqueeze_11, sum_1, sub_5, mul_54, sum_2], Original ATen: [aten.view, aten.gelu, aten.gelu_backward, aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
            # [Provenance debug handles] triton_red_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_view_1:283
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_view_1.run(tangents_1, buf0, convolution_3, getitem_9, buf1, buf3, 432, 151704, stream=stream0)
            buf2 = empty_strided_cuda((8, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_3, input_13, mul_48, mul_49, mul_50, exp, mul_51, mul_52, add_30, mul_53, sum_1], Original ATen: [aten.view, aten.gelu, aten.gelu_backward, aten.native_batch_norm_backward]
            # [Provenance debug handles] triton_per_fused_gelu_gelu_backward_native_batch_norm_backward_view_2:284
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_gelu_backward_native_batch_norm_backward_view_2.run(buf1, buf2, 8, 54, stream=stream0)
            del buf1
            buf4 = empty_strided_cuda((8, ), (1, ), torch.float32)
            buf7 = empty_strided_cuda((8, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [view_3, input_13, mul_48, mul_49, mul_50, exp, mul_51, mul_52, add_30, mul_53, input_12, unsqueeze_10, unsqueeze_11, sub_5, mul_54, sum_2, mul_62], Original ATen: [aten.view, aten.gelu, aten.gelu_backward, aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
            # [Provenance debug handles] triton_per_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_view_3:285
            stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_view_3.run(buf3, rsqrt_4, buf4, buf7, 8, 54, stream=stream0)
            buf5 = buf0; del buf0  # reuse
            buf6 = buf5; del buf5  # reuse
            # Topologically Sorted Source Nodes: [view_3, input_13, mul_48, mul_49, mul_50, exp, mul_51, mul_52, add_30, mul_53, input_12, unsqueeze_10, unsqueeze_11, sub_5, mul_55, unsqueeze_12, unsqueeze_13, mul_56, mul_57, mul_58, unsqueeze_14, unsqueeze_15, mul_59, unsqueeze_16, unsqueeze_17, mul_60, sub_7, sub_8, mul_61], Original ATen: [aten.view, aten.gelu, aten.gelu_backward, aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_view_4:286
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_view_4.run(buf6, tangents_1, convolution_3, getitem_9, buf4, rsqrt_4, buf2, primals_34, 65536000, stream=stream0)
            del convolution_3
            del getitem_9
            del primals_34
            del rsqrt_4
            del tangents_1
            buf8 = buf3; del buf3  # reuse
            # Topologically Sorted Source Nodes: [sum_3], Original ATen: [aten.convolution_backward]
            # [Provenance debug handles] triton_red_fused_convolution_backward_5:287
            stream0 = get_raw_stream(0)
            triton_red_fused_convolution_backward_5.run(buf6, buf8, 432, 151704, stream=stream0)
            buf9 = buf4; del buf4  # reuse
            # Topologically Sorted Source Nodes: [sum_3], Original ATen: [aten.convolution_backward]
            # [Provenance debug handles] triton_per_fused_gelu_gelu_backward_native_batch_norm_backward_view_2:288
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_gelu_backward_native_batch_norm_backward_view_2.run(buf8, buf9, 8, 54, stream=stream0)
            del buf8
            # Topologically Sorted Source Nodes: [convolution_backward], Original ATen: [aten.convolution_backward]
            # [Provenance debug handles] torch.ops.aten.convolution_backward.default:289
            buf10 = torch.ops.aten.convolution_backward.default(buf6, mul_36, primals_29, [8], [1], [0], [1], False, [0], 1, [True, True, False])
            del buf6
            del mul_36
            del primals_29
            buf11 = buf10[0]
            assert_size_stride(buf11, (500, 64, 16384), (1048576, 16384, 1), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf11, 16, 'torch.ops.aten.convolution_backward.default')
            buf12 = buf10[1]
            assert_size_stride(buf12, (8, 64, 1), (64, 1, 1), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf12, 16, 'torch.ops.aten.convolution_backward.default')
            del buf10
            buf13 = empty_strided_cuda((500, 64, 16384), (1048576, 16384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten._native_batch_norm_legit_functional]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_functional_6:290
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_6.run(convolution_2, getitem_7, rsqrt_3, primals_27, primals_28, buf13, 524288000, stream=stream0)
            del primals_28
            buf14 = empty_strided_cuda((64, 63), (1, 64), torch.float32)
            buf16 = empty_strided_cuda((64, 63), (1, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_10, mul_64, mul_65, mul_66, exp_1, mul_67, mul_68, add_32, mul_69, input_9, unsqueeze_18, unsqueeze_19, sum_4, sub_9, mul_70, sum_5], Original ATen: [aten.gelu, aten.gelu_backward, aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
            # [Provenance debug handles] triton_red_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_7:291
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_7.run(buf11, buf13, convolution_2, getitem_7, buf14, buf16, 4032, 130032, stream=stream0)
            buf15 = empty_strided_cuda((64, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [input_10, mul_64, mul_65, mul_66, exp_1, mul_67, mul_68, add_32, mul_69, sum_4], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_batch_norm_backward]
            # [Provenance debug handles] triton_per_fused_gelu_gelu_backward_native_batch_norm_backward_8:292
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_gelu_backward_native_batch_norm_backward_8.run(buf14, buf15, 64, 63, stream=stream0)
            del buf14
            buf17 = empty_strided_cuda((64, ), (1, ), torch.float32)
            buf20 = empty_strided_cuda((64, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [input_10, mul_64, mul_65, mul_66, exp_1, mul_67, mul_68, add_32, mul_69, input_9, unsqueeze_18, unsqueeze_19, sub_9, mul_70, sum_5, mul_78], Original ATen: [aten.gelu, aten.gelu_backward, aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
            # [Provenance debug handles] triton_per_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_9:293
            stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_9.run(buf16, rsqrt_3, buf17, buf20, 64, 63, stream=stream0)
            del buf16
            buf18 = buf11; del buf11  # reuse
            buf19 = buf18; del buf18  # reuse
            # Topologically Sorted Source Nodes: [input_10, mul_64, mul_65, mul_66, exp_1, mul_67, mul_68, add_32, mul_69, input_9, unsqueeze_18, unsqueeze_19, sub_9, mul_71, unsqueeze_20, unsqueeze_21, mul_72, mul_73, mul_74, unsqueeze_22, unsqueeze_23, mul_75, unsqueeze_24, unsqueeze_25, mul_76, sub_11, sub_12, mul_77], Original ATen: [aten.gelu, aten.gelu_backward, aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_10:294
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_10.run(buf19, buf13, convolution_2, getitem_7, buf17, rsqrt_3, buf15, primals_27, 524288000, stream=stream0)
            del buf13
            del convolution_2
            del getitem_7
            del primals_27
            del rsqrt_3
            buf21 = empty_strided_cuda((64, 63), (1, 64), torch.float32)
            # Topologically Sorted Source Nodes: [sum_6], Original ATen: [aten.convolution_backward]
            # [Provenance debug handles] triton_red_fused_convolution_backward_11:295
            stream0 = get_raw_stream(0)
            triton_red_fused_convolution_backward_11.run(buf19, buf21, 4032, 130032, stream=stream0)
            buf22 = buf17; del buf17  # reuse
            # Topologically Sorted Source Nodes: [sum_6], Original ATen: [aten.convolution_backward]
            # [Provenance debug handles] triton_per_fused_gelu_gelu_backward_native_batch_norm_backward_8:296
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_gelu_backward_native_batch_norm_backward_8.run(buf21, buf22, 64, 63, stream=stream0)
            # Topologically Sorted Source Nodes: [convolution_backward_1], Original ATen: [aten.convolution_backward]
            # [Provenance debug handles] torch.ops.aten.convolution_backward.default:297
            buf23 = torch.ops.aten.convolution_backward.default(buf19, mul_26, primals_22, [64], [1], [0], [1], False, [0], 1, [True, True, False])
            del mul_26
            del primals_22
            buf24 = buf23[0]
            assert_size_stride(buf24, (500, 64, 16384), (1048576, 16384, 1), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf24, 16, 'torch.ops.aten.convolution_backward.default')
            buf25 = buf23[1]
            assert_size_stride(buf25, (64, 64, 1), (64, 1, 1), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf25, 16, 'torch.ops.aten.convolution_backward.default')
            del buf23
            buf26 = buf19; del buf19  # reuse
            # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten._native_batch_norm_legit_functional]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_functional_6:298
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_6.run(convolution_1, getitem_5, rsqrt_2, primals_20, primals_21, buf26, 524288000, stream=stream0)
            del primals_21
            buf27 = buf21; del buf21  # reuse
            buf29 = empty_strided_cuda((64, 63), (1, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_7, mul_80, mul_81, mul_82, exp_2, mul_83, mul_84, add_34, mul_85, input_6, unsqueeze_26, unsqueeze_27, sum_7, sub_13, mul_86, sum_8], Original ATen: [aten.gelu, aten.gelu_backward, aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
            # [Provenance debug handles] triton_red_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_7:299
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_7.run(buf24, buf26, convolution_1, getitem_5, buf27, buf29, 4032, 130032, stream=stream0)
            buf28 = empty_strided_cuda((64, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [input_7, mul_80, mul_81, mul_82, exp_2, mul_83, mul_84, add_34, mul_85, sum_7], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_batch_norm_backward]
            # [Provenance debug handles] triton_per_fused_gelu_gelu_backward_native_batch_norm_backward_8:300
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_gelu_backward_native_batch_norm_backward_8.run(buf27, buf28, 64, 63, stream=stream0)
            buf30 = empty_strided_cuda((64, ), (1, ), torch.float32)
            buf33 = empty_strided_cuda((64, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [input_7, mul_80, mul_81, mul_82, exp_2, mul_83, mul_84, add_34, mul_85, input_6, unsqueeze_26, unsqueeze_27, sub_13, mul_86, sum_8, mul_94], Original ATen: [aten.gelu, aten.gelu_backward, aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
            # [Provenance debug handles] triton_per_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_9:301
            stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_9.run(buf29, rsqrt_2, buf30, buf33, 64, 63, stream=stream0)
            buf31 = buf24; del buf24  # reuse
            buf32 = buf31; del buf31  # reuse
            # Topologically Sorted Source Nodes: [input_7, mul_80, mul_81, mul_82, exp_2, mul_83, mul_84, add_34, mul_85, input_6, unsqueeze_26, unsqueeze_27, sub_13, mul_87, unsqueeze_28, unsqueeze_29, mul_88, mul_89, mul_90, unsqueeze_30, unsqueeze_31, mul_91, unsqueeze_32, unsqueeze_33, mul_92, sub_15, sub_16, mul_93], Original ATen: [aten.gelu, aten.gelu_backward, aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_10:302
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_10.run(buf32, buf26, convolution_1, getitem_5, buf30, rsqrt_2, buf28, primals_20, 524288000, stream=stream0)
            del buf26
            del convolution_1
            del getitem_5
            del primals_20
            del rsqrt_2
            buf34 = buf29; del buf29  # reuse
            # Topologically Sorted Source Nodes: [sum_9], Original ATen: [aten.convolution_backward]
            # [Provenance debug handles] triton_red_fused_convolution_backward_11:303
            stream0 = get_raw_stream(0)
            triton_red_fused_convolution_backward_11.run(buf32, buf34, 4032, 130032, stream=stream0)
            buf35 = buf30; del buf30  # reuse
            # Topologically Sorted Source Nodes: [sum_9], Original ATen: [aten.convolution_backward]
            # [Provenance debug handles] triton_per_fused_gelu_gelu_backward_native_batch_norm_backward_8:304
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_gelu_backward_native_batch_norm_backward_8.run(buf34, buf35, 64, 63, stream=stream0)
            # Topologically Sorted Source Nodes: [convolution_backward_2], Original ATen: [aten.convolution_backward]
            # [Provenance debug handles] torch.ops.aten.convolution_backward.default:305
            buf36 = torch.ops.aten.convolution_backward.default(buf32, mul_16, primals_15, [64], [1], [0], [1], False, [0], 1, [True, True, False])
            del mul_16
            del primals_15
            buf37 = buf36[0]
            assert_size_stride(buf37, (500, 64, 16384), (1048576, 16384, 1), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf37, 16, 'torch.ops.aten.convolution_backward.default')
            buf38 = buf36[1]
            assert_size_stride(buf38, (64, 64, 1), (64, 1, 1), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf38, 16, 'torch.ops.aten.convolution_backward.default')
            del buf36
            buf39 = buf32; del buf32  # reuse
            # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._native_batch_norm_legit_functional]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_functional_6:306
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_6.run(convolution, getitem_3, rsqrt_1, primals_13, primals_14, buf39, 524288000, stream=stream0)
            del primals_14
            buf40 = buf34; del buf34  # reuse
            buf42 = buf27; del buf27  # reuse
            # Topologically Sorted Source Nodes: [input_4, mul_96, mul_97, mul_98, exp_3, mul_99, mul_100, add_36, mul_101, input_3, unsqueeze_34, unsqueeze_35, sum_10, sub_17, mul_102, sum_11], Original ATen: [aten.gelu, aten.gelu_backward, aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
            # [Provenance debug handles] triton_red_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_7:307
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_7.run(buf37, buf39, convolution, getitem_3, buf40, buf42, 4032, 130032, stream=stream0)
            buf41 = empty_strided_cuda((64, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [input_4, mul_96, mul_97, mul_98, exp_3, mul_99, mul_100, add_36, mul_101, sum_10], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_batch_norm_backward]
            # [Provenance debug handles] triton_per_fused_gelu_gelu_backward_native_batch_norm_backward_8:308
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_gelu_backward_native_batch_norm_backward_8.run(buf40, buf41, 64, 63, stream=stream0)
            del buf40
            buf43 = empty_strided_cuda((64, ), (1, ), torch.float32)
            buf46 = empty_strided_cuda((64, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [input_4, mul_96, mul_97, mul_98, exp_3, mul_99, mul_100, add_36, mul_101, input_3, unsqueeze_34, unsqueeze_35, sub_17, mul_102, sum_11, mul_110], Original ATen: [aten.gelu, aten.gelu_backward, aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
            # [Provenance debug handles] triton_per_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_9:309
            stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_9.run(buf42, rsqrt_1, buf43, buf46, 64, 63, stream=stream0)
            buf44 = buf37; del buf37  # reuse
            buf45 = buf44; del buf44  # reuse
            # Topologically Sorted Source Nodes: [input_4, mul_96, mul_97, mul_98, exp_3, mul_99, mul_100, add_36, mul_101, input_3, unsqueeze_34, unsqueeze_35, sub_17, mul_103, unsqueeze_36, unsqueeze_37, mul_104, mul_105, mul_106, unsqueeze_38, unsqueeze_39, mul_107, unsqueeze_40, unsqueeze_41, mul_108, sub_19, sub_20, mul_109], Original ATen: [aten.gelu, aten.gelu_backward, aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_10:310
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_gelu_gelu_backward_native_batch_norm_backward_10.run(buf45, buf39, convolution, getitem_3, buf43, rsqrt_1, buf41, primals_13, 524288000, stream=stream0)
            del buf39
            del convolution
            del getitem_3
            del primals_13
            del rsqrt_1
            buf47 = buf42; del buf42  # reuse
            # Topologically Sorted Source Nodes: [sum_12], Original ATen: [aten.convolution_backward]
            # [Provenance debug handles] triton_red_fused_convolution_backward_11:311
            stream0 = get_raw_stream(0)
            triton_red_fused_convolution_backward_11.run(buf45, buf47, 4032, 130032, stream=stream0)
            buf48 = buf43; del buf43  # reuse
            # Topologically Sorted Source Nodes: [sum_12], Original ATen: [aten.convolution_backward]
            # [Provenance debug handles] triton_per_fused_gelu_gelu_backward_native_batch_norm_backward_8:312
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_gelu_backward_native_batch_norm_backward_8.run(buf47, buf48, 64, 63, stream=stream0)
            del buf47
            # Topologically Sorted Source Nodes: [convolution_backward_3], Original ATen: [aten.convolution_backward]
            # [Provenance debug handles] torch.ops.aten.convolution_backward.default:313
            buf49 = torch.ops.aten.convolution_backward.default(buf45, add_4, primals_8, [64], [1], [0], [1], False, [0], 1, [True, True, False])
            del add_4
            del buf45
            del primals_8
            buf50 = buf49[0]
            assert_size_stride(buf50, (500, 4, 16384), (65536, 16384, 1), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf50, 16, 'torch.ops.aten.convolution_backward.default')
            buf51 = buf49[1]
            assert_size_stride(buf51, (64, 4, 1), (4, 1, 1), 'torch.ops.aten.convolution_backward.default')
            assert_alignment(buf51, 16, 'torch.ops.aten.convolution_backward.default')
            del buf49
            buf52 = empty_strided_cuda((4, 100), (1, 4), torch.float32)
            # Topologically Sorted Source Nodes: [sum_13], Original ATen: [aten.native_batch_norm_backward]
            # [Provenance debug handles] triton_red_fused_native_batch_norm_backward_12:314
            stream0 = get_raw_stream(0)
            triton_red_fused_native_batch_norm_backward_12.run(buf50, buf52, 400, 81920, stream=stream0)
            buf53 = empty_strided_cuda((4, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [sum_13], Original ATen: [aten.native_batch_norm_backward]
            # [Provenance debug handles] triton_per_fused_native_batch_norm_backward_13:315
            stream0 = get_raw_stream(0)
            triton_per_fused_native_batch_norm_backward_13.run(buf52, buf53, 4, 100, stream=stream0)
            del buf52
            buf54 = empty_strided_cuda((4, 864, 32), (27648, 32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view, U, input_1, sub_21, mul_111, sum_14], Original ATen: [aten.view, aten.transpose, aten.clone, aten.native_batch_norm_backward]
            # [Provenance debug handles] triton_red_fused_clone_native_batch_norm_backward_transpose_view_14:316
            stream0 = get_raw_stream(0)
            triton_red_fused_clone_native_batch_norm_backward_transpose_view_14.run(buf50, primals_2, unsqueeze_43, buf54, 4, 27648, 297, stream=stream0)
            del buf50
            del primals_2
            del unsqueeze_43
            buf55 = empty_strided_cuda((4, 864), (864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view, U, input_1, sub_21, mul_111, sum_14], Original ATen: [aten.view, aten.transpose, aten.clone, aten.native_batch_norm_backward]
            # [Provenance debug handles] triton_per_fused_clone_native_batch_norm_backward_transpose_view_15:317
            stream0 = get_raw_stream(0)
            triton_per_fused_clone_native_batch_norm_backward_transpose_view_15.run(buf54, buf55, 3456, 32, stream=stream0)
            del buf54
            buf56 = empty_strided_cuda((4, ), (1, ), torch.float32)
            buf57 = buf56; del buf56  # reuse
            # Topologically Sorted Source Nodes: [view, U, input_1, sub_21, mul_111, sum_14, mul_119], Original ATen: [aten.view, aten.transpose, aten.clone, aten.native_batch_norm_backward]
            # [Provenance debug handles] triton_per_fused_clone_native_batch_norm_backward_transpose_view_16:318
            stream0 = get_raw_stream(0)
            triton_per_fused_clone_native_batch_norm_backward_transpose_view_16.run(buf57, buf55, squeeze_1, 4, 864, stream=stream0)
            del buf55
            del squeeze_1
        return (None, None, None, None, None, buf57, buf53, buf51, buf48, None, None, None, buf46, buf41, buf38, buf35, None, None, None, buf33, buf28, buf25, buf22, None, None, None, buf20, buf15, buf12, buf9, None, None, None, buf7, buf2, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_2 = rand_strided((500, 128, 128, 4), (65536, 512, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((8, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_4 = rand_strided((500, 4, 16384), (65536, 16384, 1), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((500, 64, 16384), (1048576, 16384, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((1, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_1 = rand_strided((1, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_16 = rand_strided((500, 64, 16384), (1048576, 16384, 1), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((500, 64, 16384), (1048576, 16384, 1), device='cuda:0', dtype=torch.float32)
    getitem_5 = rand_strided((1, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_2 = rand_strided((1, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_26 = rand_strided((500, 64, 16384), (1048576, 16384, 1), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((500, 64, 16384), (1048576, 16384, 1), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((1, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_3 = rand_strided((1, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_36 = rand_strided((500, 64, 16384), (1048576, 16384, 1), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((500, 8, 16384), (131072, 16384, 1), device='cuda:0', dtype=torch.float32)
    getitem_9 = rand_strided((1, 8, 1), (8, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_4 = rand_strided((1, 8, 1), (8, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_43 = rand_strided((1, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((4000, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_2, primals_8, primals_13, primals_14, primals_15, primals_20, primals_21, primals_22, primals_27, primals_28, primals_29, primals_34, primals_35, squeeze_1, add_4, convolution, getitem_3, rsqrt_1, mul_16, convolution_1, getitem_5, rsqrt_2, mul_26, convolution_2, getitem_7, rsqrt_3, mul_36, convolution_3, getitem_9, rsqrt_4, unsqueeze_43, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
