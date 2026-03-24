# AOT ID: ['14_backward']
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


# kernel path: /traces/inductor_cache/hs/chsdqa57sxvg5u73sb37ar6kbgi6xqufe765wvflem25teazwyg7.py
# Topologically Sorted Source Nodes: [mul_139, mul_140, sum_8, linear_13, gelu_6, dropout_19, layer_norm_27, mul_141, sum_9, mul_142, sub_36, sub_37, div_7, mul_143, mul_144, sum_10, sum_11, convert_element_type, mul_145, clone_35, mul_148, mul_149, mul_150, exp_7, mul_151, mul_152, add_91, mul_153], Original ATen: [aten.native_layer_norm_backward, aten.view, aten.gelu, aten.native_dropout, aten.native_layer_norm, aten.native_dropout_backward, aten.gelu_backward]
# Source node to ATen node mapping:
#   add_91 => add_91
#   clone_35 => mul_146
#   convert_element_type => convert_element_type
#   div_7 => div_7
#   dropout_19 => mul_134, mul_135
#   exp_7 => exp_7
#   gelu_6 => add_87, erf_6, mul_131, mul_132, mul_133
#   layer_norm_27 => mul_136, sub_34
#   linear_13 => view_105
#   mul_139 => mul_139
#   mul_140 => mul_140
#   mul_141 => mul_141
#   mul_142 => mul_142
#   mul_143 => mul_143
#   mul_144 => mul_144
#   mul_145 => mul_145
#   mul_148 => mul_148
#   mul_149 => mul_149
#   mul_150 => mul_150
#   mul_151 => mul_151
#   mul_152 => mul_152
#   mul_153 => mul_153
#   sub_36 => sub_36
#   sub_37 => sub_37
#   sum_10 => sum_10
#   sum_11 => sum_11
#   sum_8 => sum_8
#   sum_9 => sum_9
# Graph fragment:
#   %tangents_1 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0" = PlaceHolder[target=tangents_1]
#   %primals_114 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_114]
#   %gt_26 : Tensor "b8[500, 128, 512][65536, 512, 1]cuda:0" = PlaceHolder[target=gt_26]
#   %addmm_20 : Tensor "f32[64000, 512][512, 1]cuda:0" = PlaceHolder[target=addmm_20]
#   %getitem_55 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=getitem_55]
#   %rsqrt_27 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_27]
#   %sum_8 : Tensor "f32[500, 128, 1][128, 1, 64000]cuda:0" = PlaceHolder[target=sum_8]
#   %sum_9 : Tensor "f32[500, 128, 1][128, 1, 64000]cuda:0" = PlaceHolder[target=sum_9]
#   %mul_143 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0" = PlaceHolder[target=mul_143]
#   %mul_139 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %primals_114), kwargs = {})
#   %mul_140 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_139, 512), kwargs = {})
#   %sum_8 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_139, [2], True), kwargs = {})
#   %view_105 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_20, [500, 128, 512]), kwargs = {})
#   %mul_131 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_105, 0.5), kwargs = {})
#   %mul_132 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_105, 0.7071067811865476), kwargs = {})
#   %erf_6 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_132,), kwargs = {})
#   %add_87 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_6, 1), kwargs = {})
#   %mul_133 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_131, %add_87), kwargs = {})
#   %mul_134 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt_26, %mul_133), kwargs = {})
#   %mul_135 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_134, 1.1111111111111112), kwargs = {})
#   %sub_34 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_135, %getitem_55), kwargs = {})
#   %mul_136 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_34, %rsqrt_27), kwargs = {})
#   %mul_141 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_139, %mul_136), kwargs = {})
#   %sum_9 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_141, [2], True), kwargs = {})
#   %mul_142 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_136, %sum_9), kwargs = {})
#   %sub_36 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_140, %sum_8), kwargs = {})
#   %sub_37 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_36, %mul_142), kwargs = {})
#   %div_7 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_27, 512), kwargs = {})
#   %mul_143 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_7, %sub_37), kwargs = {})
#   %mul_144 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %mul_136), kwargs = {})
#   %sum_10 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_144, [0, 1]), kwargs = {})
#   %sum_11 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%tangents_1, [0, 1]), kwargs = {})
#   %convert_element_type : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%gt_26, torch.float32), kwargs = {})
#   %mul_145 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, 1.1111111111111112), kwargs = {})
#   %mul_146 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_143, %mul_145), kwargs = {})
#   %mul_148 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_87, 0.5), kwargs = {})
#   %mul_149 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_105, %view_105), kwargs = {})
#   %mul_150 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_149, -0.5), kwargs = {})
#   %exp_7 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_150,), kwargs = {})
#   %mul_151 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_7, 0.3989422804014327), kwargs = {})
#   %mul_152 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_105, %mul_151), kwargs = {})
#   %add_91 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_148, %mul_152), kwargs = {})
#   %mul_153 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_146, %add_91), kwargs = {})
#   return %sum_8,%sum_9,%mul_143,%mul_153
triton_per_fused_gelu_gelu_backward_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_view_0 = async_compile.triton('triton_per_fused_gelu_gelu_backward_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_view_0', '''
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
    inductor_meta={'grid_type': 'MixOrderReductionGrid', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_gelu_backward_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_view_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 6, 'num_store': -2, 'num_reduction': 2, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'RSPLIT_SIZE': 64}
)
@triton.jit
def triton_per_fused_gelu_gelu_backward_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_view_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, RSPLIT_SIZE : tl.constexpr, NUM_STAGES : tl.constexpr):
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


# kernel path: /traces/inductor_cache/wl/cwlyff22qc7jakq64otcybhwvsmieul2l5ono7qqbx6lcdzqcqwe.py
# Topologically Sorted Source Nodes: [linear_13, gelu_6, convert_element_type, mul_145, clone_35, mul_148, mul_149, mul_150, exp_7, mul_151, mul_152, add_91, mul_153, view_106, sum_12], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.sum]
# Source node to ATen node mapping:
#   add_91 => add_91
#   clone_35 => mul_146
#   convert_element_type => convert_element_type
#   exp_7 => exp_7
#   gelu_6 => add_87, erf_6, mul_132
#   linear_13 => view_105
#   mul_145 => mul_145
#   mul_148 => mul_148
#   mul_149 => mul_149
#   mul_150 => mul_150
#   mul_151 => mul_151
#   mul_152 => mul_152
#   mul_153 => mul_153
#   sum_12 => sum_12
#   view_106 => view_106
# Graph fragment:
#   %mul_153 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0" = PlaceHolder[target=mul_153]
#   %view_105 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_20, [500, 128, 512]), kwargs = {})
#   %mul_132 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_105, 0.7071067811865476), kwargs = {})
#   %erf_6 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_132,), kwargs = {})
#   %add_87 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_6, 1), kwargs = {})
#   %convert_element_type : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%gt_26, torch.float32), kwargs = {})
#   %mul_145 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, 1.1111111111111112), kwargs = {})
#   %mul_146 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_143, %mul_145), kwargs = {})
#   %mul_148 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_87, 0.5), kwargs = {})
#   %mul_149 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_105, %view_105), kwargs = {})
#   %mul_150 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_149, -0.5), kwargs = {})
#   %exp_7 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_150,), kwargs = {})
#   %mul_151 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_7, 0.3989422804014327), kwargs = {})
#   %mul_152 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_105, %mul_151), kwargs = {})
#   %add_91 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_148, %mul_152), kwargs = {})
#   %mul_153 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_146, %add_91), kwargs = {})
#   %view_106 : Tensor "f32[64000, 512][512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_153, [64000, 512]), kwargs = {})
#   %sum_12 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_106, [0], True), kwargs = {})
#   return %buf10
triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_1 = async_compile.triton('triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_1', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 131891200, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /traces/inductor_cache/yn/cynpmuz4nh5ftap46g3nlx4sr7lw4tn3h2hlbvxlsohssgxfk7fd.py
# Topologically Sorted Source Nodes: [linear_13, gelu_6, convert_element_type, mul_145, clone_35, mul_148, mul_149, mul_150, exp_7, mul_151, mul_152, add_91, mul_153, view_106, sum_12], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.sum]
# Source node to ATen node mapping:
#   add_91 => add_91
#   clone_35 => mul_146
#   convert_element_type => convert_element_type
#   exp_7 => exp_7
#   gelu_6 => add_87, erf_6, mul_132
#   linear_13 => view_105
#   mul_145 => mul_145
#   mul_148 => mul_148
#   mul_149 => mul_149
#   mul_150 => mul_150
#   mul_151 => mul_151
#   mul_152 => mul_152
#   mul_153 => mul_153
#   sum_12 => sum_12
#   view_106 => view_106
# Graph fragment:
#   %buf10 : Tensor "f32[1, 512, 200][102400, 1, 512]cuda:0" = PlaceHolder[target=buf10]
#   %view_105 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_20, [500, 128, 512]), kwargs = {})
#   %mul_132 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_105, 0.7071067811865476), kwargs = {})
#   %erf_6 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_132,), kwargs = {})
#   %add_87 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_6, 1), kwargs = {})
#   %convert_element_type : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%gt_26, torch.float32), kwargs = {})
#   %mul_145 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, 1.1111111111111112), kwargs = {})
#   %mul_146 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_143, %mul_145), kwargs = {})
#   %mul_148 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_87, 0.5), kwargs = {})
#   %mul_149 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_105, %view_105), kwargs = {})
#   %mul_150 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_149, -0.5), kwargs = {})
#   %exp_7 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_150,), kwargs = {})
#   %mul_151 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_7, 0.3989422804014327), kwargs = {})
#   %mul_152 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_105, %mul_151), kwargs = {})
#   %add_91 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_148, %mul_152), kwargs = {})
#   %mul_153 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_146, %add_91), kwargs = {})
#   %view_106 : Tensor "f32[64000, 512][512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_153, [64000, 512]), kwargs = {})
#   %sum_12 : Tensor "f32[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_106, [0], True), kwargs = {})
#   return %sum_12
triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_2 = async_compile.triton('triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_2', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 413696, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_2(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /traces/inductor_cache/nq/cnq6i5kwp43n2xio2sy5tye662vsntvmr33s44cdig6xzulnqptz.py
# Topologically Sorted Source Nodes: [view_108, mul_155, mul_156, sum_13, layer_norm_26, mul_157, sum_14, mul_158, sub_39, sub_40, div_8, mul_159, mul_160, sum_15, sum_16, add_92], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.native_layer_norm, aten.add]
# Source node to ATen node mapping:
#   add_92 => add_92
#   div_8 => div_8
#   layer_norm_26 => mul_129, sub_33
#   mul_155 => mul_155
#   mul_156 => mul_156
#   mul_157 => mul_157
#   mul_158 => mul_158
#   mul_159 => mul_159
#   mul_160 => mul_160
#   sub_39 => sub_39
#   sub_40 => sub_40
#   sum_13 => sum_13
#   sum_14 => sum_14
#   sum_15 => sum_15
#   sum_16 => sum_16
#   view_108 => view_108
# Graph fragment:
#   %mm_7 : Tensor "f32[64000, 128][128, 1]cuda:0" = PlaceHolder[target=mm_7]
#   %primals_110 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_110]
#   %add_84 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=add_84]
#   %getitem_53 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=getitem_53]
#   %rsqrt_26 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_26]
#   %tangents_2 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=tangents_2]
#   %sum_13 : Tensor "f32[500, 128, 1][128, 1, 64000]cuda:0" = PlaceHolder[target=sum_13]
#   %sum_14 : Tensor "f32[500, 128, 1][128, 1, 64000]cuda:0" = PlaceHolder[target=sum_14]
#   %view_108 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_7, [500, 128, 128]), kwargs = {})
#   %mul_155 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_108, %primals_110), kwargs = {})
#   %mul_156 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_155, 128), kwargs = {})
#   %sum_13 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_155, [2], True), kwargs = {})
#   %sub_33 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_84, %getitem_53), kwargs = {})
#   %mul_129 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_33, %rsqrt_26), kwargs = {})
#   %mul_157 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_155, %mul_129), kwargs = {})
#   %sum_14 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_157, [2], True), kwargs = {})
#   %mul_158 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_129, %sum_14), kwargs = {})
#   %sub_39 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_156, %sum_13), kwargs = {})
#   %sub_40 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_39, %mul_158), kwargs = {})
#   %div_8 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_26, 128), kwargs = {})
#   %mul_159 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_8, %sub_40), kwargs = {})
#   %mul_160 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_108, %mul_129), kwargs = {})
#   %sum_15 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_160, [0, 1]), kwargs = {})
#   %sum_16 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_108, [0, 1]), kwargs = {})
#   %add_92 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_2, %mul_159), kwargs = {})
#   return %sum_13,%sum_14,%add_92
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_3 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'ws_ptr': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'RSPLIT_SIZE': 'constexpr', 'NUM_STAGES': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'MixOrderReductionGrid', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 6, 'num_store': -1, 'num_reduction': 2, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'RSPLIT_SIZE': 64}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, RSPLIT_SIZE : tl.constexpr, NUM_STAGES : tl.constexpr):
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
        tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
        tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
        tmp17 = tl.load(in_ptr5 + (r0_1 + 128*x0), xmask, other=0.0)
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
        tmp27 = tmp0 * tmp11
        tl.store(out_ptr2 + (r0_1 + 128*x0), tmp26, xmask)
        tmp28 = tl.sum(tmp27, 0)
        tmp29 = accum0 + tmp28
        accum0 = tmp29
        tmp30 = tl.sum(tmp0, 0)
        tmp31 = accum1 + tmp30
        accum1 = tmp31
    tl.store(ws_ptr + (tl.program_id(0) + 0 * tl.num_programs(0)) * r0_numel + r0_index, accum0, r0_mask)
    tl.store(ws_ptr + (tl.program_id(0) + 1 * tl.num_programs(0)) * r0_numel + r0_index, accum1, r0_mask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/5r/c5rbkxycnm5z2i46jui3pq7vgpmhefmrewrpuntpc4sidvhne74v.py
# Topologically Sorted Source Nodes: [convert_element_type_1, mul_161, clone_36, multi_head_attention_forward_6, transpose_13, sub_41, mul_163, mul_164, mul_165, sum_17, mul_166, sum_18, mul_167, sub_42, sub_43, div_9, mul_168, mul_169, sum_19, sum_20, permute_88, clone_37], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone]
# Source node to ATen node mapping:
#   clone_36 => mul_162
#   clone_37 => clone_37
#   convert_element_type_1 => convert_element_type_1
#   div_9 => div_9
#   mul_161 => mul_161
#   mul_163 => mul_163
#   mul_164 => mul_164
#   mul_165 => mul_165
#   mul_166 => mul_166
#   mul_167 => mul_167
#   mul_168 => mul_168
#   mul_169 => mul_169
#   multi_head_attention_forward_6 => view_102
#   permute_88 => permute_88
#   sub_41 => sub_41
#   sub_42 => sub_42
#   sub_43 => sub_43
#   sum_17 => sum_17
#   sum_18 => sum_18
#   sum_19 => sum_19
#   sum_20 => sum_20
#   transpose_13 => permute_82
# Graph fragment:
#   %add_92 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=add_92]
#   %gt_25 : Tensor "b8[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=gt_25]
#   %primals_108 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_108]
#   %addmm_19 : Tensor "f32[64000, 128][128, 1]cuda:0" = PlaceHolder[target=addmm_19]
#   %getitem_51 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=getitem_51]
#   %rsqrt_25 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_25]
#   %sum_17 : Tensor "f32[500, 128, 1][128, 1, 64000]cuda:0" = PlaceHolder[target=sum_17]
#   %sum_18 : Tensor "f32[500, 128, 1][128, 1, 64000]cuda:0" = PlaceHolder[target=sum_18]
#   %convert_element_type_1 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%gt_25, torch.float32), kwargs = {})
#   %mul_161 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1, 1.1111111111111112), kwargs = {})
#   %mul_162 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_92, %mul_161), kwargs = {})
#   %view_102 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_19, [128, 500, 128]), kwargs = {})
#   %permute_82 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_102, [1, 0, 2]), kwargs = {})
#   %sub_41 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_82, %getitem_51), kwargs = {})
#   %mul_163 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_41, %rsqrt_25), kwargs = {})
#   %mul_164 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_162, %primals_108), kwargs = {})
#   %mul_165 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_164, 128), kwargs = {})
#   %sum_17 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_164, [2], True), kwargs = {})
#   %mul_166 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_164, %mul_163), kwargs = {})
#   %sum_18 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_166, [2], True), kwargs = {})
#   %mul_167 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_163, %sum_18), kwargs = {})
#   %sub_42 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_165, %sum_17), kwargs = {})
#   %sub_43 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_42, %mul_167), kwargs = {})
#   %div_9 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_25, 128), kwargs = {})
#   %mul_168 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_9, %sub_43), kwargs = {})
#   %mul_169 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_162, %mul_163), kwargs = {})
#   %sum_19 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_169, [0, 1]), kwargs = {})
#   %sum_20 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_162, [0, 1]), kwargs = {})
#   %permute_88 : Tensor "f32[128, 500, 128][128, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_168, [1, 0, 2]), kwargs = {})
#   %clone_37 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_88,), kwargs = {memory_format: torch.contiguous_format})
#   return %sum_17,%sum_18,%clone_37
triton_per_fused_clone_native_dropout_backward_native_layer_norm_backward_transpose_view_4 = async_compile.triton('triton_per_fused_clone_native_dropout_backward_native_layer_norm_backward_transpose_view_4', '''
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
    inductor_meta={'grid_type': 'MixOrderReductionGrid', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clone_native_dropout_backward_native_layer_norm_backward_transpose_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 6, 'num_store': -1, 'num_reduction': 2, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'RSPLIT_SIZE': 64}
)
@triton.jit
def triton_per_fused_clone_native_dropout_backward_native_layer_norm_backward_transpose_view_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, RSPLIT_SIZE : tl.constexpr, NUM_STAGES : tl.constexpr):
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


# kernel path: /traces/inductor_cache/ec/cecdatd5fyhl2ruvp2f5bdqjxojgm5txeqxpj5sjapnnshqdl4yo.py
# Topologically Sorted Source Nodes: [convert_element_type_1, mul_161, clone_36, multi_head_attention_forward_6, transpose_13, sub_41, mul_163, mul_164, mul_165, mul_167, sub_42, sub_43, div_9, mul_168, permute_88, clone_37, view_109, sum_21], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.sum]
# Source node to ATen node mapping:
#   clone_36 => mul_162
#   clone_37 => clone_37
#   convert_element_type_1 => convert_element_type_1
#   div_9 => div_9
#   mul_161 => mul_161
#   mul_163 => mul_163
#   mul_164 => mul_164
#   mul_165 => mul_165
#   mul_167 => mul_167
#   mul_168 => mul_168
#   multi_head_attention_forward_6 => view_102
#   permute_88 => permute_88
#   sub_41 => sub_41
#   sub_42 => sub_42
#   sub_43 => sub_43
#   sum_21 => sum_21
#   transpose_13 => permute_82
#   view_109 => view_109
# Graph fragment:
#   %clone_37 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0" = PlaceHolder[target=clone_37]
#   %convert_element_type_1 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%gt_25, torch.float32), kwargs = {})
#   %mul_161 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1, 1.1111111111111112), kwargs = {})
#   %mul_162 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_92, %mul_161), kwargs = {})
#   %view_102 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_19, [128, 500, 128]), kwargs = {})
#   %permute_82 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_102, [1, 0, 2]), kwargs = {})
#   %sub_41 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_82, %getitem_51), kwargs = {})
#   %mul_163 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_41, %rsqrt_25), kwargs = {})
#   %mul_164 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_162, %primals_108), kwargs = {})
#   %mul_165 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_164, 128), kwargs = {})
#   %mul_167 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_163, %sum_18), kwargs = {})
#   %sub_42 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_165, %sum_17), kwargs = {})
#   %sub_43 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_42, %mul_167), kwargs = {})
#   %div_9 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_25, 128), kwargs = {})
#   %mul_168 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_9, %sub_43), kwargs = {})
#   %permute_88 : Tensor "f32[128, 500, 128][128, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_168, [1, 0, 2]), kwargs = {})
#   %clone_37 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_88,), kwargs = {memory_format: torch.contiguous_format})
#   %view_109 : Tensor "f32[64000, 128][128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_37, [64000, 128]), kwargs = {})
#   %sum_21 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_109, [0], True), kwargs = {})
#   return %buf28
triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_5 = async_compile.triton('triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_5', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 33280000, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_5(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /traces/inductor_cache/r4/cr4pvixh7klahgjabxdzauogjfi7pcrkss5gz7cmr3hphytlwwzs.py
# Topologically Sorted Source Nodes: [convert_element_type_1, mul_161, clone_36, multi_head_attention_forward_6, transpose_13, sub_41, mul_163, mul_164, mul_165, mul_167, sub_42, sub_43, div_9, mul_168, permute_88, clone_37, view_109, sum_21], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.sum]
# Source node to ATen node mapping:
#   clone_36 => mul_162
#   clone_37 => clone_37
#   convert_element_type_1 => convert_element_type_1
#   div_9 => div_9
#   mul_161 => mul_161
#   mul_163 => mul_163
#   mul_164 => mul_164
#   mul_165 => mul_165
#   mul_167 => mul_167
#   mul_168 => mul_168
#   multi_head_attention_forward_6 => view_102
#   permute_88 => permute_88
#   sub_41 => sub_41
#   sub_42 => sub_42
#   sub_43 => sub_43
#   sum_21 => sum_21
#   transpose_13 => permute_82
#   view_109 => view_109
# Graph fragment:
#   %buf28 : Tensor "f32[1, 128, 500][64000, 1, 128]cuda:0" = PlaceHolder[target=buf28]
#   %convert_element_type_1 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%gt_25, torch.float32), kwargs = {})
#   %mul_161 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1, 1.1111111111111112), kwargs = {})
#   %mul_162 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_92, %mul_161), kwargs = {})
#   %view_102 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_19, [128, 500, 128]), kwargs = {})
#   %permute_82 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_102, [1, 0, 2]), kwargs = {})
#   %sub_41 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_82, %getitem_51), kwargs = {})
#   %mul_163 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_41, %rsqrt_25), kwargs = {})
#   %mul_164 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_162, %primals_108), kwargs = {})
#   %mul_165 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_164, 128), kwargs = {})
#   %mul_167 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_163, %sum_18), kwargs = {})
#   %sub_42 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_165, %sum_17), kwargs = {})
#   %sub_43 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_42, %mul_167), kwargs = {})
#   %div_9 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_25, 128), kwargs = {})
#   %mul_168 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_9, %sub_43), kwargs = {})
#   %permute_88 : Tensor "f32[128, 500, 128][128, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_168, [1, 0, 2]), kwargs = {})
#   %clone_37 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_88,), kwargs = {memory_format: torch.contiguous_format})
#   %view_109 : Tensor "f32[64000, 128][128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_37, [64000, 128]), kwargs = {})
#   %sum_21 : Tensor "f32[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_109, [0], True), kwargs = {})
#   return %sum_21
triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_6 = async_compile.triton('triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_6', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 257024, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_6(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /traces/inductor_cache/ij/cijjnzbtb3cvckcrgqj4gkdad35rtz2tej2akg2nr7bsjl3ugw4d.py
# Topologically Sorted Source Nodes: [convert_element_type_2, mul_170, clone_38, multi_head_attention_forward_6, mul_172, sum_22, neg, fma], Original ATen: [aten.native_dropout_backward, aten._softmax, aten._softmax_backward_data]
# Source node to ATen node mapping:
#   clone_38 => mul_171
#   convert_element_type_2 => convert_element_type_2
#   fma => fma
#   mul_170 => mul_170
#   mul_172 => mul_172
#   multi_head_attention_forward_6 => div_6, exp_6, sub_31
#   neg => neg
#   sum_22 => sum_22
# Graph fragment:
#   %bmm_8 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=bmm_8]
#   %gt_24 : Tensor "b8[4000, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=gt_24]
#   %baddbmm_6 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=baddbmm_6]
#   %amax_6 : Tensor "f32[4000, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=amax_6]
#   %sum_7 : Tensor "f32[4000, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=sum_7]
#   %mul_172 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=mul_172]
#   %sum_22 : Tensor "f32[4000, 128, 1][128, 1, 512000]cuda:0" = PlaceHolder[target=sum_22]
#   %convert_element_type_2 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%gt_24, torch.float32), kwargs = {})
#   %mul_170 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, 1.1111111111111112), kwargs = {})
#   %mul_171 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_8, %mul_170), kwargs = {})
#   %sub_31 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%baddbmm_6, %amax_6), kwargs = {})
#   %exp_6 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_31,), kwargs = {})
#   %div_6 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_6, %sum_7), kwargs = {})
#   %mul_172 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_171, %div_6), kwargs = {})
#   %sum_22 : Tensor "f32[4000, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_172, [-1], True), kwargs = {})
#   %neg : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%div_6,), kwargs = {})
#   %fma : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.fma.default](args = (%neg, %sum_22, %mul_172), kwargs = {})
#   return %mul_172,%sum_22,%fma
triton_per_fused__softmax__softmax_backward_data_native_dropout_backward_7 = async_compile.triton('triton_per_fused__softmax__softmax_backward_data_native_dropout_backward_7', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__softmax_backward_data_native_dropout_backward_7', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4096000, 'r0_': 1114112000}}
)
@triton.jit
def triton_per_fused__softmax__softmax_backward_data_native_dropout_backward_7(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /traces/inductor_cache/4z/c4zc53amtkqyypent3lrswt6zi3dlvibavz6qf4rwqzinajhdris.py
# Topologically Sorted Source Nodes: [permute_98, mul_173, permute_99, clone_39, view_112, permute_100, view_113, permute_101, clone_40, view_114, full, _generalized_scatter, _generalized_scatter_1, add_93, _generalized_scatter_2, add_94, unsqueeze_7, permute_102, squeeze_7, clone_41, view_115, sum_23], Original ATen: [aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.sum]
# Source node to ATen node mapping:
#   _generalized_scatter => select_scatter_default
#   _generalized_scatter_1 => select_scatter_default_1
#   _generalized_scatter_2 => select_scatter_default_2
#   add_93 => add_93
#   add_94 => add_94
#   clone_39 => clone_39
#   clone_40 => clone_40
#   clone_41 => clone_41
#   full => full_default
#   mul_173 => mul_173
#   permute_100 => permute_100
#   permute_101 => permute_101
#   permute_102 => permute_102
#   permute_98 => permute_98
#   permute_99 => permute_99
#   squeeze_7 => squeeze_7
#   sum_23 => sum_23
#   unsqueeze_7 => unsqueeze_7
#   view_112 => view_112
#   view_113 => view_113
#   view_114 => view_114
#   view_115 => view_115
# Graph fragment:
#   %bmm_7 : Tensor "f32[4000, 128, 16][2048, 16, 1]cuda:0" = PlaceHolder[target=bmm_7]
#   %bmm_10 : Tensor "f32[4000, 16, 128][2048, 128, 1]cuda:0" = PlaceHolder[target=bmm_10]
#   %bmm_9 : Tensor "f32[4000, 128, 16][2048, 16, 1]cuda:0" = PlaceHolder[target=bmm_9]
#   %permute_98 : Tensor "f32[4000, 128, 16][2048, 1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_10, [0, 2, 1]), kwargs = {})
#   %mul_173 : Tensor "f32[4000, 128, 16][2048, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_9, 0.25), kwargs = {})
#   %permute_99 : Tensor "f32[128, 4000, 16][16, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_7, [1, 0, 2]), kwargs = {})
#   %clone_39 : Tensor "f32[128, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_99,), kwargs = {memory_format: torch.contiguous_format})
#   %view_112 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_39, [128, 500, 128]), kwargs = {})
#   %permute_100 : Tensor "f32[128, 4000, 16][1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_98, [1, 0, 2]), kwargs = {})
#   %view_113 : Tensor "f32[128, 500, 128][1, 16384, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_100, [128, 500, 128]), kwargs = {})
#   %permute_101 : Tensor "f32[128, 4000, 16][16, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_173, [1, 0, 2]), kwargs = {})
#   %clone_40 : Tensor "f32[128, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_101,), kwargs = {memory_format: torch.contiguous_format})
#   %view_114 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_40, [128, 500, 128]), kwargs = {})
#   %full_default : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=21] = call_function[target=torch.ops.aten.full.default](args = ([3, 128, 500, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %view_112, 0, 2), kwargs = {})
#   %select_scatter_default_1 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %view_113, 0, 1), kwargs = {})
#   %add_93 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default, %select_scatter_default_1), kwargs = {})
#   %select_scatter_default_2 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %view_114, 0, 0), kwargs = {})
#   %add_94 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_93, %select_scatter_default_2), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[3, 128, 500, 1, 128][8192000, 64000, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_94, 3), kwargs = {})
#   %permute_102 : Tensor "f32[1, 128, 500, 3, 128][128, 64000, 128, 8192000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_7, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze_7 : Tensor "f32[128, 500, 3, 128][64000, 128, 8192000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_102, 0), kwargs = {})
#   %clone_41 : Tensor "f32[128, 500, 3, 128][192000, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_7,), kwargs = {memory_format: torch.contiguous_format})
#   %view_115 : Tensor "f32[128, 500, 384][192000, 384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_41, [128, 500, 384]), kwargs = {})
#   %sum_23 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_115, [0, 1], True), kwargs = {})
#   return %buf37
triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_8 = async_compile.triton('triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_8', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 66519040, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_8(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /traces/inductor_cache/bi/cbia6c5frsvz4zhw55ntgaepdhyqb2bixxj7m5664w7nb4tdbrk6.py
# Topologically Sorted Source Nodes: [permute_98, mul_173, permute_99, clone_39, view_112, permute_100, view_113, permute_101, clone_40, view_114, full, _generalized_scatter, _generalized_scatter_1, add_93, _generalized_scatter_2, add_94, unsqueeze_7, permute_102, squeeze_7, clone_41, view_115, sum_23], Original ATen: [aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.sum]
# Source node to ATen node mapping:
#   _generalized_scatter => select_scatter_default
#   _generalized_scatter_1 => select_scatter_default_1
#   _generalized_scatter_2 => select_scatter_default_2
#   add_93 => add_93
#   add_94 => add_94
#   clone_39 => clone_39
#   clone_40 => clone_40
#   clone_41 => clone_41
#   full => full_default
#   mul_173 => mul_173
#   permute_100 => permute_100
#   permute_101 => permute_101
#   permute_102 => permute_102
#   permute_98 => permute_98
#   permute_99 => permute_99
#   squeeze_7 => squeeze_7
#   sum_23 => sum_23
#   unsqueeze_7 => unsqueeze_7
#   view_112 => view_112
#   view_113 => view_113
#   view_114 => view_114
#   view_115 => view_115
# Graph fragment:
#   %buf37 : Tensor "f32[1, 1, 384, 320][122880, 122880, 1, 384]cuda:0" = PlaceHolder[target=buf37]
#   %permute_98 : Tensor "f32[4000, 128, 16][2048, 1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_10, [0, 2, 1]), kwargs = {})
#   %mul_173 : Tensor "f32[4000, 128, 16][2048, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_9, 0.25), kwargs = {})
#   %permute_99 : Tensor "f32[128, 4000, 16][16, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_7, [1, 0, 2]), kwargs = {})
#   %clone_39 : Tensor "f32[128, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_99,), kwargs = {memory_format: torch.contiguous_format})
#   %view_112 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_39, [128, 500, 128]), kwargs = {})
#   %permute_100 : Tensor "f32[128, 4000, 16][1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_98, [1, 0, 2]), kwargs = {})
#   %view_113 : Tensor "f32[128, 500, 128][1, 16384, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_100, [128, 500, 128]), kwargs = {})
#   %permute_101 : Tensor "f32[128, 4000, 16][16, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_173, [1, 0, 2]), kwargs = {})
#   %clone_40 : Tensor "f32[128, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_101,), kwargs = {memory_format: torch.contiguous_format})
#   %view_114 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_40, [128, 500, 128]), kwargs = {})
#   %full_default : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=21] = call_function[target=torch.ops.aten.full.default](args = ([3, 128, 500, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %view_112, 0, 2), kwargs = {})
#   %select_scatter_default_1 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %view_113, 0, 1), kwargs = {})
#   %add_93 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default, %select_scatter_default_1), kwargs = {})
#   %select_scatter_default_2 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %view_114, 0, 0), kwargs = {})
#   %add_94 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_93, %select_scatter_default_2), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[3, 128, 500, 1, 128][8192000, 64000, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_94, 3), kwargs = {})
#   %permute_102 : Tensor "f32[1, 128, 500, 3, 128][128, 64000, 128, 8192000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_7, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze_7 : Tensor "f32[128, 500, 3, 128][64000, 128, 8192000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_102, 0), kwargs = {})
#   %clone_41 : Tensor "f32[128, 500, 3, 128][192000, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_7,), kwargs = {memory_format: torch.contiguous_format})
#   %view_115 : Tensor "f32[128, 500, 384][192000, 384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_41, [128, 500, 384]), kwargs = {})
#   %sum_23 : Tensor "f32[1, 1, 384][384, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_115, [0, 1], True), kwargs = {})
#   return %sum_23
triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_9 = async_compile.triton('triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_9', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 494592, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_9(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /traces/inductor_cache/wk/cwk2jcaplk76vughdpfgmys3whjomwuynedjudbh6hieqadqthth.py
# Topologically Sorted Source Nodes: [permute_98, mul_173, permute_99, clone_39, view_112, permute_100, view_113, permute_101, clone_40, view_114, full, _generalized_scatter, _generalized_scatter_1, add_93, _generalized_scatter_2, add_94, unsqueeze_7, permute_102, squeeze_7, clone_41], Original ATen: [aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze]
# Source node to ATen node mapping:
#   _generalized_scatter => select_scatter_default
#   _generalized_scatter_1 => select_scatter_default_1
#   _generalized_scatter_2 => select_scatter_default_2
#   add_93 => add_93
#   add_94 => add_94
#   clone_39 => clone_39
#   clone_40 => clone_40
#   clone_41 => clone_41
#   full => full_default
#   mul_173 => mul_173
#   permute_100 => permute_100
#   permute_101 => permute_101
#   permute_102 => permute_102
#   permute_98 => permute_98
#   permute_99 => permute_99
#   squeeze_7 => squeeze_7
#   unsqueeze_7 => unsqueeze_7
#   view_112 => view_112
#   view_113 => view_113
#   view_114 => view_114
# Graph fragment:
#   %bmm_7 : Tensor "f32[4000, 128, 16][2048, 16, 1]cuda:0" = PlaceHolder[target=bmm_7]
#   %bmm_10 : Tensor "f32[4000, 16, 128][2048, 128, 1]cuda:0" = PlaceHolder[target=bmm_10]
#   %bmm_9 : Tensor "f32[4000, 128, 16][2048, 16, 1]cuda:0" = PlaceHolder[target=bmm_9]
#   %permute_98 : Tensor "f32[4000, 128, 16][2048, 1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_10, [0, 2, 1]), kwargs = {})
#   %mul_173 : Tensor "f32[4000, 128, 16][2048, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_9, 0.25), kwargs = {})
#   %permute_99 : Tensor "f32[128, 4000, 16][16, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm_7, [1, 0, 2]), kwargs = {})
#   %clone_39 : Tensor "f32[128, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_99,), kwargs = {memory_format: torch.contiguous_format})
#   %view_112 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_39, [128, 500, 128]), kwargs = {})
#   %permute_100 : Tensor "f32[128, 4000, 16][1, 2048, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_98, [1, 0, 2]), kwargs = {})
#   %view_113 : Tensor "f32[128, 500, 128][1, 16384, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_100, [128, 500, 128]), kwargs = {})
#   %permute_101 : Tensor "f32[128, 4000, 16][16, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_173, [1, 0, 2]), kwargs = {})
#   %clone_40 : Tensor "f32[128, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_101,), kwargs = {memory_format: torch.contiguous_format})
#   %view_114 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_40, [128, 500, 128]), kwargs = {})
#   %full_default : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=21] = call_function[target=torch.ops.aten.full.default](args = ([3, 128, 500, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %view_112, 0, 2), kwargs = {})
#   %select_scatter_default_1 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %view_113, 0, 1), kwargs = {})
#   %add_93 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default, %select_scatter_default_1), kwargs = {})
#   %select_scatter_default_2 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %view_114, 0, 0), kwargs = {})
#   %add_94 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_93, %select_scatter_default_2), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[3, 128, 500, 1, 128][8192000, 64000, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_94, 3), kwargs = {})
#   %permute_102 : Tensor "f32[1, 128, 500, 3, 128][128, 64000, 128, 8192000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_7, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze_7 : Tensor "f32[128, 500, 3, 128][64000, 128, 8192000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_102, 0), kwargs = {})
#   %clone_41 : Tensor "f32[128, 500, 3, 128][192000, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_7,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_41
triton_poi_fused__unsafe_view_add_clone_mul_select_backward_squeeze_transpose_unsqueeze_view_10 = async_compile.triton('triton_poi_fused__unsafe_view_add_clone_mul_select_backward_squeeze_transpose_unsqueeze_view_10', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_clone_mul_select_backward_squeeze_transpose_unsqueeze_view_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 32768000, 'x': 262144000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_clone_mul_select_backward_squeeze_transpose_unsqueeze_view_10(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /traces/inductor_cache/bi/cbihvzsvdqgjr3n52ipnlfduzmeiub4igy6o343oqk7lqhk7nsho.py
# Topologically Sorted Source Nodes: [view_118, permute_107, mul_175, sum_24], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   mul_175 => mul_175
#   permute_107 => permute_107
#   sum_24 => sum_24
#   view_118 => view_118
# Graph fragment:
#   %mm_12 : Tensor "f32[64000, 128][128, 1]cuda:0" = PlaceHolder[target=mm_12]
#   %primals_102 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_102]
#   %view_118 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_12, [128, 500, 128]), kwargs = {})
#   %permute_107 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_118, [1, 0, 2]), kwargs = {})
#   %mul_175 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_107, %primals_102), kwargs = {})
#   %sum_24 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_175, [2], True), kwargs = {})
#   return %sum_24
triton_per_fused_native_layer_norm_backward_transpose_view_11 = async_compile.triton('triton_per_fused_native_layer_norm_backward_transpose_view_11', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_transpose_view_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 512000, 'r0_': 32768512}}
)
@triton.jit
def triton_per_fused_native_layer_norm_backward_transpose_view_11(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /traces/inductor_cache/i3/ci3grosgdzwl73dhpo4bad5omxkwfoycdbeahmncyn56fou6m56n.py
# Topologically Sorted Source Nodes: [view_118, permute_107, mul_175, mul_176, mul_177, sum_25, mul_178, sub_45, sub_46, mul_179, mul_180, sum_26, sum_27, add_95, convert_element_type_3, mul_181, clone_42], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward, aten.add, aten.native_dropout_backward]
# Source node to ATen node mapping:
#   add_95 => add_95
#   clone_42 => mul_182
#   convert_element_type_3 => convert_element_type_3
#   mul_175 => mul_175
#   mul_176 => mul_176
#   mul_177 => mul_177
#   mul_178 => mul_178
#   mul_179 => mul_179
#   mul_180 => mul_180
#   mul_181 => mul_181
#   permute_107 => permute_107
#   sub_45 => sub_45
#   sub_46 => sub_46
#   sum_25 => sum_25
#   sum_26 => sum_26
#   sum_27 => sum_27
#   view_118 => view_118
# Graph fragment:
#   %mm_12 : Tensor "f32[64000, 128][128, 1]cuda:0" = PlaceHolder[target=mm_12]
#   %primals_102 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_102]
#   %mul_120 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=mul_120]
#   %add_92 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=add_92]
#   %div_10 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=div_10]
#   %sum_24 : Tensor "f32[500, 128, 1][1, 500, 64000]cuda:0" = PlaceHolder[target=sum_24]
#   %sum_25 : Tensor "f32[500, 128, 1][128, 1, 64000]cuda:0" = PlaceHolder[target=sum_25]
#   %add_95 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=add_95]
#   %gt_23 : Tensor "b8[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=gt_23]
#   %view_118 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_12, [128, 500, 128]), kwargs = {})
#   %permute_107 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_118, [1, 0, 2]), kwargs = {})
#   %mul_175 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_107, %primals_102), kwargs = {})
#   %mul_176 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_175, 128), kwargs = {})
#   %mul_177 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_175, %mul_120), kwargs = {})
#   %sum_25 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_177, [2], True), kwargs = {})
#   %mul_178 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_120, %sum_25), kwargs = {})
#   %sub_45 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_176, %sum_24), kwargs = {})
#   %sub_46 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_45, %mul_178), kwargs = {})
#   %mul_179 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_10, %sub_46), kwargs = {})
#   %mul_180 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_107, %mul_120), kwargs = {})
#   %sum_26 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_180, [0, 1]), kwargs = {})
#   %sum_27 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%permute_107, [0, 1]), kwargs = {})
#   %add_95 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_92, %mul_179), kwargs = {})
#   %convert_element_type_3 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%gt_23, torch.float32), kwargs = {})
#   %mul_181 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_3, 1.1111111111111112), kwargs = {})
#   %mul_182 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_95, %mul_181), kwargs = {})
#   return %sum_25,%add_95,%mul_182
triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_transpose_view_12 = async_compile.triton('triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_transpose_view_12', '''
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
    inductor_meta={'grid_type': 'MixOrderReductionGrid', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_transpose_view_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 8, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'RSPLIT_SIZE': 64}
)
@triton.jit
def triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_transpose_view_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, RSPLIT_SIZE : tl.constexpr, NUM_STAGES : tl.constexpr):
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


# kernel path: /traces/inductor_cache/fb/cfb2bqargtgxoe47cpj7zyhxhgee4ryrlzoqy6vqsdihc7if47tx.py
# Topologically Sorted Source Nodes: [view_124, mul_200, mul_201, sum_34, mul_202, sum_35, mul_203, sub_51, sub_52, mul_204, mul_205, sum_36, sum_37, add_98], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
# Source node to ATen node mapping:
#   add_98 => add_98
#   mul_200 => mul_200
#   mul_201 => mul_201
#   mul_202 => mul_202
#   mul_203 => mul_203
#   mul_204 => mul_204
#   mul_205 => mul_205
#   sub_51 => sub_51
#   sub_52 => sub_52
#   sum_34 => sum_34
#   sum_35 => sum_35
#   sum_36 => sum_36
#   sum_37 => sum_37
#   view_124 => view_124
# Graph fragment:
#   %mm_15 : Tensor "f32[64000, 128][128, 1]cuda:0" = PlaceHolder[target=mm_15]
#   %primals_94 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_94]
#   %mul_109 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=mul_109]
#   %add_95 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=add_95]
#   %div_12 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=div_12]
#   %sum_34 : Tensor "f32[500, 128, 1][128, 1, 64000]cuda:0" = PlaceHolder[target=sum_34]
#   %sum_35 : Tensor "f32[500, 128, 1][128, 1, 64000]cuda:0" = PlaceHolder[target=sum_35]
#   %view_124 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_15, [500, 128, 128]), kwargs = {})
#   %mul_200 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_124, %primals_94), kwargs = {})
#   %mul_201 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_200, 128), kwargs = {})
#   %sum_34 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_200, [2], True), kwargs = {})
#   %mul_202 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_200, %mul_109), kwargs = {})
#   %sum_35 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_202, [2], True), kwargs = {})
#   %mul_203 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_109, %sum_35), kwargs = {})
#   %sub_51 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_201, %sum_34), kwargs = {})
#   %sub_52 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_51, %mul_203), kwargs = {})
#   %mul_204 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_12, %sub_52), kwargs = {})
#   %mul_205 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_124, %mul_109), kwargs = {})
#   %sum_36 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_205, [0, 1]), kwargs = {})
#   %sum_37 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_124, [0, 1]), kwargs = {})
#   %add_98 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_95, %mul_204), kwargs = {})
#   return %sum_34,%sum_35,%add_98
triton_per_fused_add_native_layer_norm_backward_view_13 = async_compile.triton('triton_per_fused_add_native_layer_norm_backward_view_13', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'ws_ptr': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'RSPLIT_SIZE': 'constexpr', 'NUM_STAGES': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'MixOrderReductionGrid', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_view_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': -1, 'num_reduction': 2, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'RSPLIT_SIZE': 64}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_backward_view_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, RSPLIT_SIZE : tl.constexpr, NUM_STAGES : tl.constexpr):
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
        tmp13 = tl.load(in_out_ptr0 + (r0_1 + 128*x0), xmask, other=0.0)
        tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
        tl.store(in_out_ptr0 + (r0_1 + 128*x0), tmp21, xmask)
        tmp23 = tl.sum(tmp22, 0)
        tmp24 = accum0 + tmp23
        accum0 = tmp24
        tmp25 = tl.sum(tmp0, 0)
        tmp26 = accum1 + tmp25
        accum1 = tmp26
    tl.store(ws_ptr + (tl.program_id(0) + 0 * tl.num_programs(0)) * r0_numel + r0_index, accum0, r0_mask)
    tl.store(ws_ptr + (tl.program_id(0) + 1 * tl.num_programs(0)) * r0_numel + r0_index, accum1, r0_mask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/oe/coeshjmdxb26a5bgwvrj5c7kj7cduzg3jg5rpfd2umcnbutvsydl.py
# Topologically Sorted Source Nodes: [add_99, add_106, add_113, add_120, add_127, convert_element_type_26, mul_440, clone_86, multi_head_attention_forward, mul_442, sum_148, neg_6, fma_6, add_134], Original ATen: [aten.add, aten.native_dropout_backward, aten._softmax, aten._softmax_backward_data]
# Source node to ATen node mapping:
#   add_106 => add_106
#   add_113 => add_113
#   add_120 => add_120
#   add_127 => add_127
#   add_134 => add_134
#   add_99 => add_99
#   clone_86 => mul_441
#   convert_element_type_26 => convert_element_type_26
#   fma_6 => fma_6
#   mul_440 => mul_440
#   mul_442 => mul_442
#   multi_head_attention_forward => div, exp, sub_1
#   neg_6 => neg_6
#   sum_148 => sum_148
# Graph fragment:
#   %bmm_32 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=bmm_32]
#   %gt : Tensor "b8[4000, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=gt]
#   %baddbmm : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=baddbmm]
#   %amax : Tensor "f32[4000, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=amax]
#   %sum_1 : Tensor "f32[4000, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=sum_1]
#   %mul_442 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=mul_442]
#   %sum_148 : Tensor "f32[4000, 128, 1][128, 1, 512000]cuda:0" = PlaceHolder[target=sum_148]
#   %fma : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=fma]
#   %fma_1 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=fma_1]
#   %fma_2 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=fma_2]
#   %fma_3 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=fma_3]
#   %fma_4 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=fma_4]
#   %fma_5 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=fma_5]
#   %fma_6 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=fma_6]
#   %add_99 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%fma, %fma_1), kwargs = {})
#   %add_106 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_99, %fma_2), kwargs = {})
#   %add_113 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_106, %fma_3), kwargs = {})
#   %add_120 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_113, %fma_4), kwargs = {})
#   %add_127 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_120, %fma_5), kwargs = {})
#   %convert_element_type_26 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%gt, torch.float32), kwargs = {})
#   %mul_440 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_26, 1.1111111111111112), kwargs = {})
#   %mul_441 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_32, %mul_440), kwargs = {})
#   %sub_1 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%baddbmm, %amax), kwargs = {})
#   %exp : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %div : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   %mul_442 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_441, %div), kwargs = {})
#   %sum_148 : Tensor "f32[4000, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_442, [-1], True), kwargs = {})
#   %neg_6 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%div,), kwargs = {})
#   %fma_6 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.fma.default](args = (%neg_6, %sum_148, %mul_442), kwargs = {})
#   %add_134 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_127, %fma_6), kwargs = {})
#   return %mul_442,%sum_148,%fma_6,%add_134
triton_per_fused__softmax__softmax_backward_data_add_native_dropout_backward_14 = async_compile.triton('triton_per_fused__softmax__softmax_backward_data_add_native_dropout_backward_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__softmax_backward_data_add_native_dropout_backward_14', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 11, 'num_store': 2, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4096000, 'r0_': 3211264000}}
)
@triton.jit
def triton_per_fused__softmax__softmax_backward_data_add_native_dropout_backward_14(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp18 = tl.load(in_out_ptr2 + (r0_1 + 128*x0), None)
    tmp19 = tl.load(in_ptr3 + (r0_1 + 128*x0), None)
    tmp21 = tl.load(in_ptr4 + (r0_1 + 128*x0), None)
    tmp23 = tl.load(in_ptr5 + (r0_1 + 128*x0), None)
    tmp25 = tl.load(in_ptr6 + (r0_1 + 128*x0), None)
    tmp27 = tl.load(in_ptr7 + (r0_1 + 128*x0), None)
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
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 + tmp23
    tmp26 = tmp24 + tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp28 + tmp17
    tl.store(in_out_ptr1 + (r0_1 + 128*x0), tmp17, None)
    tl.store(in_out_ptr2 + (r0_1 + 128*x0), tmp29, None)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/z7/cz7rdhbz3zv2trjwk4rmj7oh4pziljesqqebllpnkcsfzrcbhav7.py
# Topologically Sorted Source Nodes: [view_214, permute_275, mul_445, mul_446, linear, layer_norm, mul_447, sum_151, mul_448, sub_117, sub_118, div_34, mul_449, mul_450, sum_152, sum_153, add_137], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward, aten.native_layer_norm, aten.add]
# Source node to ATen node mapping:
#   add_137 => add_137
#   div_34 => div_34
#   layer_norm => mul, sub
#   linear => view_2
#   mul_445 => mul_445
#   mul_446 => mul_446
#   mul_447 => mul_447
#   mul_448 => mul_448
#   mul_449 => mul_449
#   mul_450 => mul_450
#   permute_275 => permute_275
#   sub_117 => sub_117
#   sub_118 => sub_118
#   sum_151 => sum_151
#   sum_152 => sum_152
#   sum_153 => sum_153
#   view_214 => view_214
# Graph fragment:
#   %mm_60 : Tensor "f32[64000, 128][128, 1]cuda:0" = PlaceHolder[target=mm_60]
#   %primals_4 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_4]
#   %addmm : Tensor "f32[64000, 128][128, 1]cuda:0" = PlaceHolder[target=addmm]
#   %getitem_1 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=getitem_1]
#   %rsqrt : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %add_133 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=add_133]
#   %sum_150 : Tensor "f32[500, 128, 1][1, 500, 64000]cuda:0" = PlaceHolder[target=sum_150]
#   %sum_151 : Tensor "f32[500, 128, 1][128, 1, 64000]cuda:0" = PlaceHolder[target=sum_151]
#   %view_214 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_60, [128, 500, 128]), kwargs = {})
#   %permute_275 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_214, [1, 0, 2]), kwargs = {})
#   %mul_445 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_275, %primals_4), kwargs = {})
#   %mul_446 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_445, 128), kwargs = {})
#   %view_2 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [500, 128, 128]), kwargs = {})
#   %sub : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_2, %getitem_1), kwargs = {})
#   %mul : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_447 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_445, %mul), kwargs = {})
#   %sum_151 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_447, [2], True), kwargs = {})
#   %mul_448 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %sum_151), kwargs = {})
#   %sub_117 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_446, %sum_150), kwargs = {})
#   %sub_118 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_117, %mul_448), kwargs = {})
#   %div_34 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt, 128), kwargs = {})
#   %mul_449 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_34, %sub_118), kwargs = {})
#   %mul_450 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_275, %mul), kwargs = {})
#   %sum_152 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_450, [0, 1]), kwargs = {})
#   %sum_153 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%permute_275, [0, 1]), kwargs = {})
#   %add_137 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_133, %mul_449), kwargs = {})
#   return %sum_151,%add_137
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_15 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_15', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'ws_ptr': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'RSPLIT_SIZE': 'constexpr', 'NUM_STAGES': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'MixOrderReductionGrid', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 8, 'num_store': 0, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'RSPLIT_SIZE': 64}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ws_ptr, xnumel, r0_numel, XBLOCK : tl.constexpr, RSPLIT_SIZE : tl.constexpr, NUM_STAGES : tl.constexpr):
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
        x5 = xindex
        xindex += XBLOCK
        tmp0 = tl.load(in_ptr0 + (r0_2 + 128*x1 + 64000*x0), xmask, other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_2), None, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr2 + (r0_2 + 128*x5), xmask, other=0.0)
        tmp4 = tl.load(in_ptr3 + (x5), xmask, eviction_policy='evict_last')
        tmp6 = tl.load(in_ptr4 + (x5), xmask, eviction_policy='evict_last')
        tmp13 = tl.load(in_out_ptr0 + (r0_2 + 128*x5), xmask, other=0.0)
        tmp18 = tl.load(in_ptr5 + (x1 + 500*x0), xmask, eviction_policy='evict_last')
        tmp25 = tl.load(in_ptr0 + (r0_2 + 128*x5), xmask, other=0.0)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp3 - tmp4
        tmp7 = tmp5 * tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = tl.where(xmask, tmp9, 0)
        tmp12 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
        tmp14 = 0.0078125
        tmp15 = tmp6 * tmp14
        tmp16 = 128.0
        tmp17 = tmp2 * tmp16
        tmp19 = tmp17 - tmp18
        tmp20 = tmp7 * tmp12
        tmp21 = tmp19 - tmp20
        tmp22 = tmp15 * tmp21
        tmp23 = tmp13 + tmp22
        tmp24 = tmp0 * tmp7
        tl.store(in_out_ptr0 + (r0_2 + 128*x5), tmp23, xmask)
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
        primals_2, primals_4, primals_6, primals_8, primals_12, primals_14, primals_16, primals_18, primals_20, primals_22, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_38, primals_40, primals_42, primals_44, primals_46, primals_48, primals_50, primals_52, primals_54, primals_56, primals_58, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_106, primals_108, primals_110, primals_112, primals_114, view_1, addmm, getitem_1, rsqrt, view_3, baddbmm, amax, sum_1, gt, view_11, addmm_1, getitem_3, rsqrt_1, gt_1, mul_9, view_14, addmm_2, gt_2, getitem_7, rsqrt_3, view_16, gt_3, mul_20, view_18, baddbmm_1, amax_1, sum_2, gt_4, view_26, addmm_4, getitem_11, rsqrt_5, gt_5, mul_29, view_29, addmm_5, gt_6, getitem_15, rsqrt_7, view_31, gt_7, mul_40, view_33, baddbmm_2, amax_2, sum_3, gt_8, view_41, addmm_7, getitem_19, rsqrt_9, gt_9, mul_49, view_44, addmm_8, gt_10, getitem_23, rsqrt_11, view_46, gt_11, mul_60, view_48, baddbmm_3, amax_3, sum_4, gt_12, view_56, addmm_10, getitem_27, rsqrt_13, gt_13, mul_69, view_59, addmm_11, gt_14, getitem_31, rsqrt_15, view_61, gt_15, mul_80, view_63, baddbmm_4, amax_4, sum_5, gt_16, view_71, addmm_13, getitem_35, rsqrt_17, gt_17, mul_89, view_74, addmm_14, gt_18, getitem_39, rsqrt_19, view_76, gt_19, mul_100, view_78, baddbmm_5, amax_5, sum_6, gt_20, view_86, addmm_16, getitem_43, rsqrt_21, gt_21, mul_109, view_89, addmm_17, gt_22, getitem_47, rsqrt_23, view_91, gt_23, mul_120, view_93, baddbmm_6, amax_6, sum_7, gt_24, view_101, addmm_19, getitem_51, rsqrt_25, gt_25, add_84, getitem_53, rsqrt_26, view_104, addmm_20, gt_26, getitem_55, rsqrt_27, permute_94, permute_95, permute_96, permute_97, div_10, div_12, permute_122, permute_123, permute_124, permute_125, div_14, div_16, permute_150, permute_151, permute_152, permute_153, div_18, div_20, permute_178, permute_179, permute_180, permute_181, div_22, div_24, permute_206, permute_207, permute_208, permute_209, div_26, div_28, permute_234, permute_235, permute_236, permute_237, div_30, div_32, permute_262, permute_263, permute_264, permute_265, tangents_1, tangents_2 = args
        args.clear()
        assert_size_stride(primals_2, (128, 16), (16, 1))
        assert_size_stride(primals_4, (128, ), (1, ))
        assert_size_stride(primals_6, (384, 128), (128, 1))
        assert_size_stride(primals_8, (128, 128), (128, 1))
        assert_size_stride(primals_12, (128, ), (1, ))
        assert_size_stride(primals_14, (128, ), (1, ))
        assert_size_stride(primals_16, (512, 128), (128, 1))
        assert_size_stride(primals_18, (512, ), (1, ))
        assert_size_stride(primals_20, (128, 512), (512, 1))
        assert_size_stride(primals_22, (128, ), (1, ))
        assert_size_stride(primals_24, (384, 128), (128, 1))
        assert_size_stride(primals_26, (128, 128), (128, 1))
        assert_size_stride(primals_28, (128, ), (1, ))
        assert_size_stride(primals_30, (128, ), (1, ))
        assert_size_stride(primals_32, (512, 128), (128, 1))
        assert_size_stride(primals_34, (512, ), (1, ))
        assert_size_stride(primals_36, (128, 512), (512, 1))
        assert_size_stride(primals_38, (128, ), (1, ))
        assert_size_stride(primals_40, (384, 128), (128, 1))
        assert_size_stride(primals_42, (128, 128), (128, 1))
        assert_size_stride(primals_44, (128, ), (1, ))
        assert_size_stride(primals_46, (128, ), (1, ))
        assert_size_stride(primals_48, (512, 128), (128, 1))
        assert_size_stride(primals_50, (512, ), (1, ))
        assert_size_stride(primals_52, (128, 512), (512, 1))
        assert_size_stride(primals_54, (128, ), (1, ))
        assert_size_stride(primals_56, (384, 128), (128, 1))
        assert_size_stride(primals_58, (128, 128), (128, 1))
        assert_size_stride(primals_60, (128, ), (1, ))
        assert_size_stride(primals_62, (128, ), (1, ))
        assert_size_stride(primals_64, (512, 128), (128, 1))
        assert_size_stride(primals_66, (512, ), (1, ))
        assert_size_stride(primals_68, (128, 512), (512, 1))
        assert_size_stride(primals_70, (128, ), (1, ))
        assert_size_stride(primals_72, (384, 128), (128, 1))
        assert_size_stride(primals_74, (128, 128), (128, 1))
        assert_size_stride(primals_76, (128, ), (1, ))
        assert_size_stride(primals_78, (128, ), (1, ))
        assert_size_stride(primals_80, (512, 128), (128, 1))
        assert_size_stride(primals_82, (512, ), (1, ))
        assert_size_stride(primals_84, (128, 512), (512, 1))
        assert_size_stride(primals_86, (128, ), (1, ))
        assert_size_stride(primals_88, (384, 128), (128, 1))
        assert_size_stride(primals_90, (128, 128), (128, 1))
        assert_size_stride(primals_92, (128, ), (1, ))
        assert_size_stride(primals_94, (128, ), (1, ))
        assert_size_stride(primals_96, (512, 128), (128, 1))
        assert_size_stride(primals_98, (512, ), (1, ))
        assert_size_stride(primals_100, (128, 512), (512, 1))
        assert_size_stride(primals_102, (128, ), (1, ))
        assert_size_stride(primals_104, (384, 128), (128, 1))
        assert_size_stride(primals_106, (128, 128), (128, 1))
        assert_size_stride(primals_108, (128, ), (1, ))
        assert_size_stride(primals_110, (128, ), (1, ))
        assert_size_stride(primals_112, (512, 128), (128, 1))
        assert_size_stride(primals_114, (512, ), (1, ))
        assert_size_stride(view_1, (64000, 16), (16, 1))
        assert_size_stride(addmm, (64000, 128), (128, 1))
        assert_size_stride(getitem_1, (500, 128, 1), (128, 1, 1))
        assert_size_stride(rsqrt, (500, 128, 1), (128, 1, 1))
        assert_size_stride(view_3, (64000, 128), (128, 1))
        assert_size_stride(baddbmm, (4000, 128, 128), (16384, 128, 1))
        assert_size_stride(amax, (4000, 128, 1), (128, 1, 1))
        assert_size_stride(sum_1, (4000, 128, 1), (128, 1, 1))
        assert_size_stride(gt, (4000, 128, 128), (16384, 128, 1))
        assert_size_stride(view_11, (64000, 128), (128, 1))
        assert_size_stride(addmm_1, (64000, 128), (128, 1))
        assert_size_stride(getitem_3, (500, 128, 1), (128, 1, 1))
        assert_size_stride(rsqrt_1, (500, 128, 1), (128, 1, 1))
        assert_size_stride(gt_1, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(mul_9, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(view_14, (64000, 128), (128, 1))
        assert_size_stride(addmm_2, (64000, 512), (512, 1))
        assert_size_stride(gt_2, (500, 128, 512), (65536, 512, 1))
        assert_size_stride(getitem_7, (500, 128, 1), (128, 1, 1))
        assert_size_stride(rsqrt_3, (500, 128, 1), (128, 1, 1))
        assert_size_stride(view_16, (64000, 512), (512, 1))
        assert_size_stride(gt_3, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(mul_20, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(view_18, (64000, 128), (128, 1))
        assert_size_stride(baddbmm_1, (4000, 128, 128), (16384, 128, 1))
        assert_size_stride(amax_1, (4000, 128, 1), (128, 1, 1))
        assert_size_stride(sum_2, (4000, 128, 1), (128, 1, 1))
        assert_size_stride(gt_4, (4000, 128, 128), (16384, 128, 1))
        assert_size_stride(view_26, (64000, 128), (128, 1))
        assert_size_stride(addmm_4, (64000, 128), (128, 1))
        assert_size_stride(getitem_11, (500, 128, 1), (128, 1, 1))
        assert_size_stride(rsqrt_5, (500, 128, 1), (128, 1, 1))
        assert_size_stride(gt_5, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(mul_29, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(view_29, (64000, 128), (128, 1))
        assert_size_stride(addmm_5, (64000, 512), (512, 1))
        assert_size_stride(gt_6, (500, 128, 512), (65536, 512, 1))
        assert_size_stride(getitem_15, (500, 128, 1), (128, 1, 1))
        assert_size_stride(rsqrt_7, (500, 128, 1), (128, 1, 1))
        assert_size_stride(view_31, (64000, 512), (512, 1))
        assert_size_stride(gt_7, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(mul_40, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(view_33, (64000, 128), (128, 1))
        assert_size_stride(baddbmm_2, (4000, 128, 128), (16384, 128, 1))
        assert_size_stride(amax_2, (4000, 128, 1), (128, 1, 1))
        assert_size_stride(sum_3, (4000, 128, 1), (128, 1, 1))
        assert_size_stride(gt_8, (4000, 128, 128), (16384, 128, 1))
        assert_size_stride(view_41, (64000, 128), (128, 1))
        assert_size_stride(addmm_7, (64000, 128), (128, 1))
        assert_size_stride(getitem_19, (500, 128, 1), (128, 1, 1))
        assert_size_stride(rsqrt_9, (500, 128, 1), (128, 1, 1))
        assert_size_stride(gt_9, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(mul_49, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(view_44, (64000, 128), (128, 1))
        assert_size_stride(addmm_8, (64000, 512), (512, 1))
        assert_size_stride(gt_10, (500, 128, 512), (65536, 512, 1))
        assert_size_stride(getitem_23, (500, 128, 1), (128, 1, 1))
        assert_size_stride(rsqrt_11, (500, 128, 1), (128, 1, 1))
        assert_size_stride(view_46, (64000, 512), (512, 1))
        assert_size_stride(gt_11, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(mul_60, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(view_48, (64000, 128), (128, 1))
        assert_size_stride(baddbmm_3, (4000, 128, 128), (16384, 128, 1))
        assert_size_stride(amax_3, (4000, 128, 1), (128, 1, 1))
        assert_size_stride(sum_4, (4000, 128, 1), (128, 1, 1))
        assert_size_stride(gt_12, (4000, 128, 128), (16384, 128, 1))
        assert_size_stride(view_56, (64000, 128), (128, 1))
        assert_size_stride(addmm_10, (64000, 128), (128, 1))
        assert_size_stride(getitem_27, (500, 128, 1), (128, 1, 1))
        assert_size_stride(rsqrt_13, (500, 128, 1), (128, 1, 1))
        assert_size_stride(gt_13, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(mul_69, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(view_59, (64000, 128), (128, 1))
        assert_size_stride(addmm_11, (64000, 512), (512, 1))
        assert_size_stride(gt_14, (500, 128, 512), (65536, 512, 1))
        assert_size_stride(getitem_31, (500, 128, 1), (128, 1, 1))
        assert_size_stride(rsqrt_15, (500, 128, 1), (128, 1, 1))
        assert_size_stride(view_61, (64000, 512), (512, 1))
        assert_size_stride(gt_15, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(mul_80, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(view_63, (64000, 128), (128, 1))
        assert_size_stride(baddbmm_4, (4000, 128, 128), (16384, 128, 1))
        assert_size_stride(amax_4, (4000, 128, 1), (128, 1, 1))
        assert_size_stride(sum_5, (4000, 128, 1), (128, 1, 1))
        assert_size_stride(gt_16, (4000, 128, 128), (16384, 128, 1))
        assert_size_stride(view_71, (64000, 128), (128, 1))
        assert_size_stride(addmm_13, (64000, 128), (128, 1))
        assert_size_stride(getitem_35, (500, 128, 1), (128, 1, 1))
        assert_size_stride(rsqrt_17, (500, 128, 1), (128, 1, 1))
        assert_size_stride(gt_17, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(mul_89, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(view_74, (64000, 128), (128, 1))
        assert_size_stride(addmm_14, (64000, 512), (512, 1))
        assert_size_stride(gt_18, (500, 128, 512), (65536, 512, 1))
        assert_size_stride(getitem_39, (500, 128, 1), (128, 1, 1))
        assert_size_stride(rsqrt_19, (500, 128, 1), (128, 1, 1))
        assert_size_stride(view_76, (64000, 512), (512, 1))
        assert_size_stride(gt_19, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(mul_100, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(view_78, (64000, 128), (128, 1))
        assert_size_stride(baddbmm_5, (4000, 128, 128), (16384, 128, 1))
        assert_size_stride(amax_5, (4000, 128, 1), (128, 1, 1))
        assert_size_stride(sum_6, (4000, 128, 1), (128, 1, 1))
        assert_size_stride(gt_20, (4000, 128, 128), (16384, 128, 1))
        assert_size_stride(view_86, (64000, 128), (128, 1))
        assert_size_stride(addmm_16, (64000, 128), (128, 1))
        assert_size_stride(getitem_43, (500, 128, 1), (128, 1, 1))
        assert_size_stride(rsqrt_21, (500, 128, 1), (128, 1, 1))
        assert_size_stride(gt_21, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(mul_109, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(view_89, (64000, 128), (128, 1))
        assert_size_stride(addmm_17, (64000, 512), (512, 1))
        assert_size_stride(gt_22, (500, 128, 512), (65536, 512, 1))
        assert_size_stride(getitem_47, (500, 128, 1), (128, 1, 1))
        assert_size_stride(rsqrt_23, (500, 128, 1), (128, 1, 1))
        assert_size_stride(view_91, (64000, 512), (512, 1))
        assert_size_stride(gt_23, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(mul_120, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(view_93, (64000, 128), (128, 1))
        assert_size_stride(baddbmm_6, (4000, 128, 128), (16384, 128, 1))
        assert_size_stride(amax_6, (4000, 128, 1), (128, 1, 1))
        assert_size_stride(sum_7, (4000, 128, 1), (128, 1, 1))
        assert_size_stride(gt_24, (4000, 128, 128), (16384, 128, 1))
        assert_size_stride(view_101, (64000, 128), (128, 1))
        assert_size_stride(addmm_19, (64000, 128), (128, 1))
        assert_size_stride(getitem_51, (500, 128, 1), (128, 1, 1))
        assert_size_stride(rsqrt_25, (500, 128, 1), (128, 1, 1))
        assert_size_stride(gt_25, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(add_84, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(getitem_53, (500, 128, 1), (128, 1, 1))
        assert_size_stride(rsqrt_26, (500, 128, 1), (128, 1, 1))
        assert_size_stride(view_104, (64000, 128), (128, 1))
        assert_size_stride(addmm_20, (64000, 512), (512, 1))
        assert_size_stride(gt_26, (500, 128, 512), (65536, 512, 1))
        assert_size_stride(getitem_55, (500, 128, 1), (128, 1, 1))
        assert_size_stride(rsqrt_27, (500, 128, 1), (128, 1, 1))
        assert_size_stride(permute_94, (4000, 128, 128), (16384, 1, 128))
        assert_size_stride(permute_95, (4000, 16, 128), (16, 1, 64000))
        assert_size_stride(permute_96, (4000, 128, 16), (16, 64000, 1))
        assert_size_stride(permute_97, (4000, 16, 128), (16, 1, 64000))
        assert_size_stride(div_10, (500, 128, 1), (128, 1, 1))
        assert_size_stride(div_12, (500, 128, 1), (128, 1, 1))
        assert_size_stride(permute_122, (4000, 128, 128), (16384, 1, 128))
        assert_size_stride(permute_123, (4000, 16, 128), (16, 1, 64000))
        assert_size_stride(permute_124, (4000, 128, 16), (16, 64000, 1))
        assert_size_stride(permute_125, (4000, 16, 128), (16, 1, 64000))
        assert_size_stride(div_14, (500, 128, 1), (128, 1, 1))
        assert_size_stride(div_16, (500, 128, 1), (128, 1, 1))
        assert_size_stride(permute_150, (4000, 128, 128), (16384, 1, 128))
        assert_size_stride(permute_151, (4000, 16, 128), (16, 1, 64000))
        assert_size_stride(permute_152, (4000, 128, 16), (16, 64000, 1))
        assert_size_stride(permute_153, (4000, 16, 128), (16, 1, 64000))
        assert_size_stride(div_18, (500, 128, 1), (128, 1, 1))
        assert_size_stride(div_20, (500, 128, 1), (128, 1, 1))
        assert_size_stride(permute_178, (4000, 128, 128), (16384, 1, 128))
        assert_size_stride(permute_179, (4000, 16, 128), (16, 1, 64000))
        assert_size_stride(permute_180, (4000, 128, 16), (16, 64000, 1))
        assert_size_stride(permute_181, (4000, 16, 128), (16, 1, 64000))
        assert_size_stride(div_22, (500, 128, 1), (128, 1, 1))
        assert_size_stride(div_24, (500, 128, 1), (128, 1, 1))
        assert_size_stride(permute_206, (4000, 128, 128), (16384, 1, 128))
        assert_size_stride(permute_207, (4000, 16, 128), (16, 1, 64000))
        assert_size_stride(permute_208, (4000, 128, 16), (16, 64000, 1))
        assert_size_stride(permute_209, (4000, 16, 128), (16, 1, 64000))
        assert_size_stride(div_26, (500, 128, 1), (128, 1, 1))
        assert_size_stride(div_28, (500, 128, 1), (128, 1, 1))
        assert_size_stride(permute_234, (4000, 128, 128), (16384, 1, 128))
        assert_size_stride(permute_235, (4000, 16, 128), (16, 1, 64000))
        assert_size_stride(permute_236, (4000, 128, 16), (16, 64000, 1))
        assert_size_stride(permute_237, (4000, 16, 128), (16, 1, 64000))
        assert_size_stride(div_30, (500, 128, 1), (128, 1, 1))
        assert_size_stride(div_32, (500, 128, 1), (128, 1, 1))
        assert_size_stride(permute_262, (4000, 128, 128), (16384, 1, 128))
        assert_size_stride(permute_263, (4000, 16, 128), (16, 1, 64000))
        assert_size_stride(permute_264, (4000, 128, 16), (16, 64000, 1))
        assert_size_stride(permute_265, (4000, 16, 128), (16, 1, 64000))
        assert_size_stride(tangents_1, (500, 128, 512), (65536, 512, 1))
        assert_size_stride(tangents_2, (500, 128, 128), (16384, 128, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf2 = empty_strided_cuda((500, 128, 512), (65536, 512, 1), torch.float32)
            buf7 = buf2; del buf2  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [mul_139, mul_140, sum_8, linear_13, gelu_6, dropout_19, layer_norm_27, mul_141, sum_9, mul_142, sub_36, sub_37, div_7, mul_143, mul_144, sum_10, sum_11, convert_element_type, mul_145, clone_35, mul_148, mul_149, mul_150, exp_7, mul_151, mul_152, add_91, mul_153], Original ATen: [aten.native_layer_norm_backward, aten.view, aten.gelu, aten.native_dropout, aten.native_layer_norm, aten.native_dropout_backward, aten.gelu_backward]
            workspace_0 = empty_strided_cuda((1024000, ), (1, ), torch.float32)
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_gelu_backward_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_view_0.run(buf7, tangents_1, primals_114, gt_26, addmm_20, getitem_55, rsqrt_27, workspace_0, 64000, 512, stream=stream0)
            buf4 = workspace_0[0 * 1000 * 512 : (0 + 1) * 1000 * 512].view(1000, 512).sum(dim=0)
            buf6 = workspace_0[1 * 1000 * 512 : (1 + 1) * 1000 * 512].view(1000, 512).sum(dim=0)
            del addmm_20
            del getitem_55
            del gt_26
            del primals_114
            del rsqrt_27
            del tangents_1
            buf8 = empty_strided_cuda((64000, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_13, gelu_6, convert_element_type, mul_145, clone_35, mul_148, mul_149, mul_150, exp_7, mul_151, mul_152, add_91, mul_153, view_106, permute_84, mm_7], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:119
            extern_kernels.mm(reinterpret_tensor(buf7, (64000, 512), (512, 1), 0), primals_112, out=buf8)
            del primals_112
            buf9 = empty_strided_cuda((512, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_13, gelu_6, convert_element_type, mul_145, clone_35, mul_148, mul_149, mul_150, exp_7, mul_151, mul_152, add_91, mul_153, view_106, permute_85, permute_87], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:120
            extern_kernels.mm(reinterpret_tensor(buf7, (512, 64000), (1, 512), 0), view_104, out=buf9)
            del view_104
            buf10 = empty_strided_cuda((1, 512, 200), (102400, 1, 512), torch.float32)
            # Topologically Sorted Source Nodes: [linear_13, gelu_6, convert_element_type, mul_145, clone_35, mul_148, mul_149, mul_150, exp_7, mul_151, mul_152, add_91, mul_153, view_106, sum_12], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.sum]
            # [Provenance debug handles] triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_1:121
            stream0 = get_raw_stream(0)
            triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_1.run(buf7, buf10, 102400, 320, stream=stream0)
            del buf7
            buf11 = empty_strided_cuda((1, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_13, gelu_6, convert_element_type, mul_145, clone_35, mul_148, mul_149, mul_150, exp_7, mul_151, mul_152, add_91, mul_153, view_106, sum_12], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.sum]
            # [Provenance debug handles] triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_2:122
            stream0 = get_raw_stream(0)
            triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_2.run(buf10, buf11, 512, 200, stream=stream0)
            buf18 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.float32)
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [view_108, mul_155, mul_156, sum_13, layer_norm_26, mul_157, sum_14, mul_158, sub_39, sub_40, div_8, mul_159, mul_160, sum_15, sum_16, add_92], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.native_layer_norm, aten.add]
            workspace_1 = empty_strided_cuda((256000, ), (1, ), torch.float32)
            stream0 = get_raw_stream(0)
            triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_3.run(buf8, primals_110, add_84, getitem_53, rsqrt_26, tangents_2, buf18, workspace_1, 64000, 128, stream=stream0)
            buf15 = workspace_1[0 * 1000 * 128 : (0 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            buf17 = workspace_1[1 * 1000 * 128 : (1 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            del add_84
            del getitem_53
            del primals_110
            del rsqrt_26
            del tangents_2
            buf25 = reinterpret_tensor(buf8, (128, 500, 128), (64000, 128, 1), 0); del buf8  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [convert_element_type_1, mul_161, clone_36, multi_head_attention_forward_6, transpose_13, sub_41, mul_163, mul_164, mul_165, sum_17, mul_166, sum_18, mul_167, sub_42, sub_43, div_9, mul_168, mul_169, sum_19, sum_20, permute_88, clone_37], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone]
            workspace_2 = workspace_1; del workspace_1  # reuse
            stream0 = get_raw_stream(0)
            triton_per_fused_clone_native_dropout_backward_native_layer_norm_backward_transpose_view_4.run(buf18, gt_25, primals_108, addmm_19, getitem_51, rsqrt_25, buf25, workspace_2, 64000, 128, stream=stream0)
            buf22 = workspace_2[0 * 1000 * 128 : (0 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            buf24 = workspace_2[1 * 1000 * 128 : (1 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            del addmm_19
            del getitem_51
            del gt_25
            del primals_108
            del rsqrt_25
            buf26 = empty_strided_cuda((64000, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_1, mul_161, clone_36, multi_head_attention_forward_6, transpose_13, sub_41, mul_163, mul_164, mul_165, mul_167, sub_42, sub_43, div_9, mul_168, permute_88, clone_37, view_109, permute_89, mm_9], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:123
            extern_kernels.mm(reinterpret_tensor(buf25, (64000, 128), (128, 1), 0), primals_106, out=buf26)
            del primals_106
            buf27 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_1, mul_161, clone_36, multi_head_attention_forward_6, transpose_13, sub_41, mul_163, mul_164, mul_165, mul_167, sub_42, sub_43, div_9, mul_168, permute_88, clone_37, view_109, permute_90, permute_92], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:124
            extern_kernels.mm(reinterpret_tensor(buf25, (128, 64000), (1, 128), 0), view_101, out=buf27)
            del view_101
            buf28 = empty_strided_cuda((1, 128, 500), (64000, 1, 128), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_1, mul_161, clone_36, multi_head_attention_forward_6, transpose_13, sub_41, mul_163, mul_164, mul_165, mul_167, sub_42, sub_43, div_9, mul_168, permute_88, clone_37, view_109, sum_21], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_5:125
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_5.run(buf25, buf28, 64000, 128, stream=stream0)
            buf29 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_1, mul_161, clone_36, multi_head_attention_forward_6, transpose_13, sub_41, mul_163, mul_164, mul_165, mul_167, sub_42, sub_43, div_9, mul_168, permute_88, clone_37, view_109, sum_21], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_6:126
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_6.run(buf28, buf29, 128, 500, stream=stream0)
            buf30 = reinterpret_tensor(buf25, (4000, 128, 16), (2048, 16, 1), 0); del buf25  # reuse
            # Topologically Sorted Source Nodes: [view_111, permute_93, bmm_7], Original ATen: [aten.view, aten.transpose, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:127
            extern_kernels.bmm(permute_94, reinterpret_tensor(buf26, (4000, 128, 16), (16, 64000, 1), 0), out=buf30)
            del permute_94
            buf31 = empty_strided_cuda((4000, 128, 128), (16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_111, permute_93, bmm_8], Original ATen: [aten.view, aten.transpose, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:128
            extern_kernels.bmm(reinterpret_tensor(buf26, (4000, 128, 16), (16, 64000, 1), 0), permute_95, out=buf31)
            del buf26
            del permute_95
            buf32 = buf31; del buf31  # reuse
            buf34 = baddbmm_6; del baddbmm_6  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_2, mul_170, clone_38, multi_head_attention_forward_6, mul_172, sum_22, neg, fma], Original ATen: [aten.native_dropout_backward, aten._softmax, aten._softmax_backward_data]
            # [Provenance debug handles] triton_per_fused__softmax__softmax_backward_data_native_dropout_backward_7:129
            stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data_native_dropout_backward_7.run(buf32, buf34, gt_24, amax_6, sum_7, 512000, 128, stream=stream0)
            del amax_6
            del gt_24
            del sum_7
            buf35 = empty_strided_cuda((4000, 128, 16), (2048, 16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [bmm_9], Original ATen: [aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:130
            extern_kernels.bmm(buf34, permute_96, out=buf35)
            del permute_96
            buf36 = empty_strided_cuda((4000, 16, 128), (2048, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [bmm_10], Original ATen: [aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:131
            extern_kernels.bmm(permute_97, buf34, out=buf36)
            del permute_97
            buf37 = empty_strided_cuda((1, 1, 384, 320), (122880, 122880, 1, 384), torch.float32)
            # Topologically Sorted Source Nodes: [permute_98, mul_173, permute_99, clone_39, view_112, permute_100, view_113, permute_101, clone_40, view_114, full, _generalized_scatter, _generalized_scatter_1, add_93, _generalized_scatter_2, add_94, unsqueeze_7, permute_102, squeeze_7, clone_41, view_115, sum_23], Original ATen: [aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_8:132
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_8.run(buf30, buf36, buf35, buf37, 122880, 200, stream=stream0)
            buf38 = empty_strided_cuda((1, 1, 384), (384, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_98, mul_173, permute_99, clone_39, view_112, permute_100, view_113, permute_101, clone_40, view_114, full, _generalized_scatter, _generalized_scatter_1, add_93, _generalized_scatter_2, add_94, unsqueeze_7, permute_102, squeeze_7, clone_41, view_115, sum_23], Original ATen: [aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_9:133
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf37, buf38, 384, 320, stream=stream0)
            buf39 = empty_strided_cuda((128, 500, 3, 128), (192000, 384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_98, mul_173, permute_99, clone_39, view_112, permute_100, view_113, permute_101, clone_40, view_114, full, _generalized_scatter, _generalized_scatter_1, add_93, _generalized_scatter_2, add_94, unsqueeze_7, permute_102, squeeze_7, clone_41], Original ATen: [aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_clone_mul_select_backward_squeeze_transpose_unsqueeze_view_10:134
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_mul_select_backward_squeeze_transpose_unsqueeze_view_10.run(buf30, buf36, buf35, buf39, 64000, 384, stream=stream0)
            buf40 = empty_strided_cuda((384, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [permute_98, mul_173, permute_99, clone_39, view_112, permute_100, view_113, permute_101, clone_40, view_114, full, _generalized_scatter, _generalized_scatter_1, add_93, _generalized_scatter_2, add_94, unsqueeze_7, permute_102, squeeze_7, clone_41, view_115, view_117, permute_103, permute_106], Original ATen: [aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:135
            extern_kernels.mm(reinterpret_tensor(buf39, (384, 64000), (1, 384), 0), view_93, out=buf40)
            del view_93
            buf41 = reinterpret_tensor(buf36, (64000, 128), (128, 1), 0); del buf36  # reuse
            # Topologically Sorted Source Nodes: [permute_98, mul_173, permute_99, clone_39, view_112, permute_100, view_113, permute_101, clone_40, view_114, full, _generalized_scatter, _generalized_scatter_1, add_93, _generalized_scatter_2, add_94, unsqueeze_7, permute_102, squeeze_7, clone_41, view_115, view_117, multi_head_attention_forward_6, permute_105, mm_12], Original ATen: [aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.select_backward, aten.add, aten.unsqueeze, aten.squeeze, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:136
            extern_kernels.mm(reinterpret_tensor(buf39, (64000, 384), (384, 1), 0), primals_104, out=buf41)
            del primals_104
            buf42 = reinterpret_tensor(buf28, (500, 128, 1), (1, 500, 64000), 0); del buf28  # reuse
            # Topologically Sorted Source Nodes: [view_118, permute_107, mul_175, sum_24], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_backward_transpose_view_11:137
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_backward_transpose_view_11.run(buf41, primals_102, buf42, 64000, 128, stream=stream0)
            buf48 = buf18; del buf18  # reuse
            buf49 = reinterpret_tensor(buf35, (500, 128, 128), (16384, 128, 1), 0); del buf35  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [view_118, permute_107, mul_175, mul_176, mul_177, sum_25, mul_178, sub_45, sub_46, mul_179, mul_180, sum_26, sum_27, add_95, convert_element_type_3, mul_181, clone_42], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward, aten.add, aten.native_dropout_backward]
            workspace_3 = workspace_2; del workspace_2  # reuse
            stream0 = get_raw_stream(0)
            triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_transpose_view_12.run(buf48, buf41, primals_102, mul_120, div_10, buf42, gt_23, buf49, workspace_3, 64000, 128, stream=stream0)
            buf45 = workspace_3[0 * 1000 * 128 : (0 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            buf47 = workspace_3[1 * 1000 * 128 : (1 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            del div_10
            del gt_23
            del mul_120
            del primals_102
            buf50 = empty_strided_cuda((64000, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3, mul_181, clone_42, view_119, linear_12, permute_108, mm_13], Original ATen: [aten.native_dropout_backward, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:138
            extern_kernels.mm(reinterpret_tensor(buf49, (64000, 128), (128, 1), 0), primals_100, out=buf50)
            del primals_100
            buf51 = empty_strided_cuda((128, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3, mul_181, clone_42, view_119, permute_109, permute_111], Original ATen: [aten.native_dropout_backward, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:139
            extern_kernels.mm(reinterpret_tensor(buf49, (128, 64000), (1, 128), 0), view_91, out=buf51)
            del view_91
            buf52 = reinterpret_tensor(buf42, (1, 128, 500), (64000, 1, 128), 0); del buf42  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_3, mul_181, clone_42, view_119, sum_28], Original ATen: [aten.native_dropout_backward, aten.view, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_5:140
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_5.run(buf49, buf52, 64000, 128, stream=stream0)
            buf53 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3, mul_181, clone_42, view_119, sum_28], Original ATen: [aten.native_dropout_backward, aten.view, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_6:141
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_6.run(buf52, buf53, 128, 500, stream=stream0)
            buf56 = empty_strided_cuda((500, 128, 512), (65536, 512, 1), torch.float32)
            buf61 = buf56; del buf56  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [view_121, mul_184, mul_185, sum_29, linear_11, gelu_5, dropout_16, layer_norm_23, mul_186, sum_30, mul_187, sub_48, sub_49, div_11, mul_188, mul_189, sum_31, sum_32, convert_element_type_4, mul_190, clone_43, mul_193, mul_194, mul_195, exp_8, mul_196, mul_197, add_97, mul_198], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.gelu, aten.native_dropout, aten.native_layer_norm, aten.native_dropout_backward, aten.gelu_backward]
            workspace_4 = workspace_0; del workspace_0  # reuse
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_gelu_backward_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_view_0.run(buf61, buf50, primals_98, gt_22, addmm_17, getitem_47, rsqrt_23, workspace_4, 64000, 512, stream=stream0)
            buf58 = workspace_4[0 * 1000 * 512 : (0 + 1) * 1000 * 512].view(1000, 512).sum(dim=0)
            buf60 = workspace_4[1 * 1000 * 512 : (1 + 1) * 1000 * 512].view(1000, 512).sum(dim=0)
            del addmm_17
            del getitem_47
            del gt_22
            del primals_98
            del rsqrt_23
            buf62 = reinterpret_tensor(buf49, (64000, 128), (128, 1), 0); del buf49  # reuse
            # Topologically Sorted Source Nodes: [linear_11, gelu_5, convert_element_type_4, mul_190, clone_43, mul_193, mul_194, mul_195, exp_8, mul_196, mul_197, add_97, mul_198, view_122, permute_112, mm_15], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:142
            extern_kernels.mm(reinterpret_tensor(buf61, (64000, 512), (512, 1), 0), primals_96, out=buf62)
            del primals_96
            buf63 = empty_strided_cuda((512, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_11, gelu_5, convert_element_type_4, mul_190, clone_43, mul_193, mul_194, mul_195, exp_8, mul_196, mul_197, add_97, mul_198, view_122, permute_113, permute_115], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:143
            extern_kernels.mm(reinterpret_tensor(buf61, (512, 64000), (1, 512), 0), view_89, out=buf63)
            del view_89
            buf64 = buf10; del buf10  # reuse
            # Topologically Sorted Source Nodes: [linear_11, gelu_5, convert_element_type_4, mul_190, clone_43, mul_193, mul_194, mul_195, exp_8, mul_196, mul_197, add_97, mul_198, view_122, sum_33], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.sum]
            # [Provenance debug handles] triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_1:144
            stream0 = get_raw_stream(0)
            triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_1.run(buf61, buf64, 102400, 320, stream=stream0)
            buf65 = empty_strided_cuda((1, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_11, gelu_5, convert_element_type_4, mul_190, clone_43, mul_193, mul_194, mul_195, exp_8, mul_196, mul_197, add_97, mul_198, view_122, sum_33], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.sum]
            # [Provenance debug handles] triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_2:145
            stream0 = get_raw_stream(0)
            triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_2.run(buf64, buf65, 512, 200, stream=stream0)
            buf72 = buf48; del buf48  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [view_124, mul_200, mul_201, sum_34, mul_202, sum_35, mul_203, sub_51, sub_52, mul_204, mul_205, sum_36, sum_37, add_98], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            workspace_5 = workspace_3; del workspace_3  # reuse
            stream0 = get_raw_stream(0)
            triton_per_fused_add_native_layer_norm_backward_view_13.run(buf72, buf62, primals_94, mul_109, div_12, workspace_5, 64000, 128, stream=stream0)
            buf69 = workspace_5[0 * 1000 * 128 : (0 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            buf71 = workspace_5[1 * 1000 * 128 : (1 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            del div_12
            del mul_109
            del primals_94
            buf79 = reinterpret_tensor(buf62, (128, 500, 128), (64000, 128, 1), 0); del buf62  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [convert_element_type_5, mul_206, clone_44, multi_head_attention_forward_5, transpose_11, sub_53, mul_208, mul_209, mul_210, sum_38, mul_211, sum_39, mul_212, sub_54, sub_55, div_13, mul_213, mul_214, sum_40, sum_41, permute_116, clone_45], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone]
            workspace_6 = workspace_5; del workspace_5  # reuse
            stream0 = get_raw_stream(0)
            triton_per_fused_clone_native_dropout_backward_native_layer_norm_backward_transpose_view_4.run(buf72, gt_21, primals_92, addmm_16, getitem_43, rsqrt_21, buf79, workspace_6, 64000, 128, stream=stream0)
            buf76 = workspace_6[0 * 1000 * 128 : (0 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            buf78 = workspace_6[1 * 1000 * 128 : (1 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            del addmm_16
            del getitem_43
            del gt_21
            del primals_92
            del rsqrt_21
            buf80 = buf41; del buf41  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_5, mul_206, clone_44, multi_head_attention_forward_5, transpose_11, sub_53, mul_208, mul_209, mul_210, mul_212, sub_54, sub_55, div_13, mul_213, permute_116, clone_45, view_125, permute_117, mm_17], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:146
            extern_kernels.mm(reinterpret_tensor(buf79, (64000, 128), (128, 1), 0), primals_90, out=buf80)
            del primals_90
            buf81 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_5, mul_206, clone_44, multi_head_attention_forward_5, transpose_11, sub_53, mul_208, mul_209, mul_210, mul_212, sub_54, sub_55, div_13, mul_213, permute_116, clone_45, view_125, permute_118, permute_120], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:147
            extern_kernels.mm(reinterpret_tensor(buf79, (128, 64000), (1, 128), 0), view_86, out=buf81)
            del view_86
            buf82 = buf52; del buf52  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_5, mul_206, clone_44, multi_head_attention_forward_5, transpose_11, sub_53, mul_208, mul_209, mul_210, mul_212, sub_54, sub_55, div_13, mul_213, permute_116, clone_45, view_125, sum_42], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_5:148
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_5.run(buf79, buf82, 64000, 128, stream=stream0)
            buf83 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_5, mul_206, clone_44, multi_head_attention_forward_5, transpose_11, sub_53, mul_208, mul_209, mul_210, mul_212, sub_54, sub_55, div_13, mul_213, permute_116, clone_45, view_125, sum_42], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_6:149
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_6.run(buf82, buf83, 128, 500, stream=stream0)
            buf84 = reinterpret_tensor(buf79, (4000, 128, 16), (2048, 16, 1), 0); del buf79  # reuse
            # Topologically Sorted Source Nodes: [view_127, permute_121, bmm_11], Original ATen: [aten.view, aten.transpose, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:150
            extern_kernels.bmm(permute_122, reinterpret_tensor(buf80, (4000, 128, 16), (16, 64000, 1), 0), out=buf84)
            del permute_122
            buf85 = buf32; del buf32  # reuse
            # Topologically Sorted Source Nodes: [view_127, permute_121, bmm_12], Original ATen: [aten.view, aten.transpose, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:151
            extern_kernels.bmm(reinterpret_tensor(buf80, (4000, 128, 16), (16, 64000, 1), 0), permute_123, out=buf85)
            del permute_123
            buf86 = buf85; del buf85  # reuse
            buf88 = baddbmm_5; del baddbmm_5  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_6, mul_215, clone_46, multi_head_attention_forward_5, mul_217, sum_43, neg_1, fma_1], Original ATen: [aten.native_dropout_backward, aten._softmax, aten._softmax_backward_data]
            # [Provenance debug handles] triton_per_fused__softmax__softmax_backward_data_native_dropout_backward_7:152
            stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data_native_dropout_backward_7.run(buf86, buf88, gt_20, amax_5, sum_6, 512000, 128, stream=stream0)
            del amax_5
            del gt_20
            del sum_6
            buf89 = reinterpret_tensor(buf80, (4000, 128, 16), (2048, 16, 1), 0); del buf80  # reuse
            # Topologically Sorted Source Nodes: [bmm_13], Original ATen: [aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:153
            extern_kernels.bmm(buf88, permute_124, out=buf89)
            del permute_124
            buf90 = reinterpret_tensor(buf30, (4000, 16, 128), (2048, 128, 1), 0); del buf30  # reuse
            # Topologically Sorted Source Nodes: [bmm_14], Original ATen: [aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:154
            extern_kernels.bmm(permute_125, buf88, out=buf90)
            del permute_125
            buf91 = buf37; del buf37  # reuse
            # Topologically Sorted Source Nodes: [full, permute_126, mul_218, permute_127, clone_47, view_128, permute_128, view_129, permute_129, clone_48, view_130, _generalized_scatter_3, _generalized_scatter_4, add_100, _generalized_scatter_5, add_101, unsqueeze_8, permute_130, squeeze_8, clone_49, view_131, sum_44], Original ATen: [aten.select_backward, aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.add, aten.unsqueeze, aten.squeeze, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_8:155
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_8.run(buf84, buf90, buf89, buf91, 122880, 200, stream=stream0)
            buf92 = empty_strided_cuda((1, 1, 384), (384, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full, permute_126, mul_218, permute_127, clone_47, view_128, permute_128, view_129, permute_129, clone_48, view_130, _generalized_scatter_3, _generalized_scatter_4, add_100, _generalized_scatter_5, add_101, unsqueeze_8, permute_130, squeeze_8, clone_49, view_131, sum_44], Original ATen: [aten.select_backward, aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.add, aten.unsqueeze, aten.squeeze, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_9:156
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf91, buf92, 384, 320, stream=stream0)
            buf93 = buf39; del buf39  # reuse
            # Topologically Sorted Source Nodes: [full, permute_126, mul_218, permute_127, clone_47, view_128, permute_128, view_129, permute_129, clone_48, view_130, _generalized_scatter_3, _generalized_scatter_4, add_100, _generalized_scatter_5, add_101, unsqueeze_8, permute_130, squeeze_8, clone_49], Original ATen: [aten.select_backward, aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.add, aten.unsqueeze, aten.squeeze]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_clone_mul_select_backward_squeeze_transpose_unsqueeze_view_10:157
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_mul_select_backward_squeeze_transpose_unsqueeze_view_10.run(buf84, buf90, buf89, buf93, 64000, 384, stream=stream0)
            buf94 = empty_strided_cuda((384, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full, permute_126, mul_218, permute_127, clone_47, view_128, permute_128, view_129, permute_129, clone_48, view_130, _generalized_scatter_3, _generalized_scatter_4, add_100, _generalized_scatter_5, add_101, unsqueeze_8, permute_130, squeeze_8, clone_49, view_131, view_133, permute_131, permute_134], Original ATen: [aten.select_backward, aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.add, aten.unsqueeze, aten.squeeze, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:158
            extern_kernels.mm(reinterpret_tensor(buf93, (384, 64000), (1, 384), 0), view_78, out=buf94)
            del view_78
            buf95 = reinterpret_tensor(buf90, (64000, 128), (128, 1), 0); del buf90  # reuse
            # Topologically Sorted Source Nodes: [full, permute_126, mul_218, permute_127, clone_47, view_128, permute_128, view_129, permute_129, clone_48, view_130, _generalized_scatter_3, _generalized_scatter_4, add_100, _generalized_scatter_5, add_101, unsqueeze_8, permute_130, squeeze_8, clone_49, view_131, view_133, multi_head_attention_forward_5, permute_133, mm_20], Original ATen: [aten.select_backward, aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.add, aten.unsqueeze, aten.squeeze, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:159
            extern_kernels.mm(reinterpret_tensor(buf93, (64000, 384), (384, 1), 0), primals_88, out=buf95)
            del primals_88
            buf96 = reinterpret_tensor(buf82, (500, 128, 1), (1, 500, 64000), 0); del buf82  # reuse
            # Topologically Sorted Source Nodes: [view_134, permute_135, mul_220, sum_45], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_backward_transpose_view_11:160
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_backward_transpose_view_11.run(buf95, primals_86, buf96, 64000, 128, stream=stream0)
            buf102 = buf72; del buf72  # reuse
            buf103 = reinterpret_tensor(buf89, (500, 128, 128), (16384, 128, 1), 0); del buf89  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [view_134, permute_135, mul_220, mul_221, mul_222, sum_46, mul_223, sub_57, sub_58, mul_224, mul_225, sum_47, sum_48, add_102, convert_element_type_7, mul_226, clone_50], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward, aten.add, aten.native_dropout_backward]
            workspace_7 = workspace_6; del workspace_6  # reuse
            stream0 = get_raw_stream(0)
            triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_transpose_view_12.run(buf102, buf95, primals_86, mul_100, div_14, buf96, gt_19, buf103, workspace_7, 64000, 128, stream=stream0)
            buf99 = workspace_7[0 * 1000 * 128 : (0 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            buf101 = workspace_7[1 * 1000 * 128 : (1 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            del div_14
            del gt_19
            del mul_100
            del primals_86
            buf104 = reinterpret_tensor(buf61, (64000, 512), (512, 1), 0); del buf61  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_7, mul_226, clone_50, view_135, linear_10, permute_136, mm_21], Original ATen: [aten.native_dropout_backward, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:161
            extern_kernels.mm(reinterpret_tensor(buf103, (64000, 128), (128, 1), 0), primals_84, out=buf104)
            del primals_84
            buf105 = empty_strided_cuda((128, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_7, mul_226, clone_50, view_135, permute_137, permute_139], Original ATen: [aten.native_dropout_backward, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:162
            extern_kernels.mm(reinterpret_tensor(buf103, (128, 64000), (1, 128), 0), view_76, out=buf105)
            del view_76
            buf106 = reinterpret_tensor(buf96, (1, 128, 500), (64000, 1, 128), 0); del buf96  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_7, mul_226, clone_50, view_135, sum_49], Original ATen: [aten.native_dropout_backward, aten.view, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_5:163
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_5.run(buf103, buf106, 64000, 128, stream=stream0)
            buf107 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_7, mul_226, clone_50, view_135, sum_49], Original ATen: [aten.native_dropout_backward, aten.view, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_6:164
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_6.run(buf106, buf107, 128, 500, stream=stream0)
            buf110 = reinterpret_tensor(buf50, (500, 128, 512), (65536, 512, 1), 0); del buf50  # reuse
            buf115 = buf110; del buf110  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [view_137, mul_229, mul_230, sum_50, linear_9, gelu_4, dropout_13, layer_norm_19, mul_231, sum_51, mul_232, sub_60, sub_61, div_15, mul_233, mul_234, sum_52, sum_53, convert_element_type_8, mul_235, clone_51, mul_238, mul_239, mul_240, exp_9, mul_241, mul_242, add_104, mul_243], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.gelu, aten.native_dropout, aten.native_layer_norm, aten.native_dropout_backward, aten.gelu_backward]
            workspace_8 = workspace_4; del workspace_4  # reuse
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_gelu_backward_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_view_0.run(buf115, buf104, primals_82, gt_18, addmm_14, getitem_39, rsqrt_19, workspace_8, 64000, 512, stream=stream0)
            buf112 = workspace_8[0 * 1000 * 512 : (0 + 1) * 1000 * 512].view(1000, 512).sum(dim=0)
            buf114 = workspace_8[1 * 1000 * 512 : (1 + 1) * 1000 * 512].view(1000, 512).sum(dim=0)
            del addmm_14
            del getitem_39
            del gt_18
            del primals_82
            del rsqrt_19
            buf116 = reinterpret_tensor(buf103, (64000, 128), (128, 1), 0); del buf103  # reuse
            # Topologically Sorted Source Nodes: [linear_9, gelu_4, convert_element_type_8, mul_235, clone_51, mul_238, mul_239, mul_240, exp_9, mul_241, mul_242, add_104, mul_243, view_138, permute_140, mm_23], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:165
            extern_kernels.mm(reinterpret_tensor(buf115, (64000, 512), (512, 1), 0), primals_80, out=buf116)
            del primals_80
            buf117 = empty_strided_cuda((512, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_9, gelu_4, convert_element_type_8, mul_235, clone_51, mul_238, mul_239, mul_240, exp_9, mul_241, mul_242, add_104, mul_243, view_138, permute_141, permute_143], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:166
            extern_kernels.mm(reinterpret_tensor(buf115, (512, 64000), (1, 512), 0), view_74, out=buf117)
            del view_74
            buf118 = buf64; del buf64  # reuse
            # Topologically Sorted Source Nodes: [linear_9, gelu_4, convert_element_type_8, mul_235, clone_51, mul_238, mul_239, mul_240, exp_9, mul_241, mul_242, add_104, mul_243, view_138, sum_54], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.sum]
            # [Provenance debug handles] triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_1:167
            stream0 = get_raw_stream(0)
            triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_1.run(buf115, buf118, 102400, 320, stream=stream0)
            buf119 = empty_strided_cuda((1, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_9, gelu_4, convert_element_type_8, mul_235, clone_51, mul_238, mul_239, mul_240, exp_9, mul_241, mul_242, add_104, mul_243, view_138, sum_54], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.sum]
            # [Provenance debug handles] triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_2:168
            stream0 = get_raw_stream(0)
            triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_2.run(buf118, buf119, 512, 200, stream=stream0)
            buf126 = buf102; del buf102  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [view_140, mul_245, mul_246, sum_55, mul_247, sum_56, mul_248, sub_63, sub_64, mul_249, mul_250, sum_57, sum_58, add_105], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            workspace_9 = workspace_7; del workspace_7  # reuse
            stream0 = get_raw_stream(0)
            triton_per_fused_add_native_layer_norm_backward_view_13.run(buf126, buf116, primals_78, mul_89, div_16, workspace_9, 64000, 128, stream=stream0)
            buf123 = workspace_9[0 * 1000 * 128 : (0 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            buf125 = workspace_9[1 * 1000 * 128 : (1 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            del div_16
            del mul_89
            del primals_78
            buf133 = reinterpret_tensor(buf116, (128, 500, 128), (64000, 128, 1), 0); del buf116  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [convert_element_type_9, mul_251, clone_52, multi_head_attention_forward_4, transpose_9, sub_65, mul_253, mul_254, mul_255, sum_59, mul_256, sum_60, mul_257, sub_66, sub_67, div_17, mul_258, mul_259, sum_61, sum_62, permute_144, clone_53], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone]
            workspace_10 = workspace_9; del workspace_9  # reuse
            stream0 = get_raw_stream(0)
            triton_per_fused_clone_native_dropout_backward_native_layer_norm_backward_transpose_view_4.run(buf126, gt_17, primals_76, addmm_13, getitem_35, rsqrt_17, buf133, workspace_10, 64000, 128, stream=stream0)
            buf130 = workspace_10[0 * 1000 * 128 : (0 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            buf132 = workspace_10[1 * 1000 * 128 : (1 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            del addmm_13
            del getitem_35
            del gt_17
            del primals_76
            del rsqrt_17
            buf134 = buf95; del buf95  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_9, mul_251, clone_52, multi_head_attention_forward_4, transpose_9, sub_65, mul_253, mul_254, mul_255, mul_257, sub_66, sub_67, div_17, mul_258, permute_144, clone_53, view_141, permute_145, mm_25], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:169
            extern_kernels.mm(reinterpret_tensor(buf133, (64000, 128), (128, 1), 0), primals_74, out=buf134)
            del primals_74
            buf135 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_9, mul_251, clone_52, multi_head_attention_forward_4, transpose_9, sub_65, mul_253, mul_254, mul_255, mul_257, sub_66, sub_67, div_17, mul_258, permute_144, clone_53, view_141, permute_146, permute_148], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:170
            extern_kernels.mm(reinterpret_tensor(buf133, (128, 64000), (1, 128), 0), view_71, out=buf135)
            del view_71
            buf136 = buf106; del buf106  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_9, mul_251, clone_52, multi_head_attention_forward_4, transpose_9, sub_65, mul_253, mul_254, mul_255, mul_257, sub_66, sub_67, div_17, mul_258, permute_144, clone_53, view_141, sum_63], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_5:171
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_5.run(buf133, buf136, 64000, 128, stream=stream0)
            buf137 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_9, mul_251, clone_52, multi_head_attention_forward_4, transpose_9, sub_65, mul_253, mul_254, mul_255, mul_257, sub_66, sub_67, div_17, mul_258, permute_144, clone_53, view_141, sum_63], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_6:172
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_6.run(buf136, buf137, 128, 500, stream=stream0)
            buf138 = reinterpret_tensor(buf133, (4000, 128, 16), (2048, 16, 1), 0); del buf133  # reuse
            # Topologically Sorted Source Nodes: [view_143, permute_149, bmm_15], Original ATen: [aten.view, aten.transpose, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:173
            extern_kernels.bmm(permute_150, reinterpret_tensor(buf134, (4000, 128, 16), (16, 64000, 1), 0), out=buf138)
            del permute_150
            buf139 = buf86; del buf86  # reuse
            # Topologically Sorted Source Nodes: [view_143, permute_149, bmm_16], Original ATen: [aten.view, aten.transpose, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:174
            extern_kernels.bmm(reinterpret_tensor(buf134, (4000, 128, 16), (16, 64000, 1), 0), permute_151, out=buf139)
            del permute_151
            buf140 = buf139; del buf139  # reuse
            buf142 = baddbmm_4; del baddbmm_4  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_10, mul_260, clone_54, multi_head_attention_forward_4, mul_262, sum_64, neg_2, fma_2], Original ATen: [aten.native_dropout_backward, aten._softmax, aten._softmax_backward_data]
            # [Provenance debug handles] triton_per_fused__softmax__softmax_backward_data_native_dropout_backward_7:175
            stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data_native_dropout_backward_7.run(buf140, buf142, gt_16, amax_4, sum_5, 512000, 128, stream=stream0)
            del amax_4
            del gt_16
            del sum_5
            buf143 = reinterpret_tensor(buf134, (4000, 128, 16), (2048, 16, 1), 0); del buf134  # reuse
            # Topologically Sorted Source Nodes: [bmm_17], Original ATen: [aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:176
            extern_kernels.bmm(buf142, permute_152, out=buf143)
            del permute_152
            buf144 = reinterpret_tensor(buf84, (4000, 16, 128), (2048, 128, 1), 0); del buf84  # reuse
            # Topologically Sorted Source Nodes: [bmm_18], Original ATen: [aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:177
            extern_kernels.bmm(permute_153, buf142, out=buf144)
            del permute_153
            buf145 = buf91; del buf91  # reuse
            # Topologically Sorted Source Nodes: [full, permute_154, mul_263, permute_155, clone_55, view_144, permute_156, view_145, permute_157, clone_56, view_146, _generalized_scatter_6, _generalized_scatter_7, add_107, _generalized_scatter_8, add_108, unsqueeze_9, permute_158, squeeze_9, clone_57, view_147, sum_65], Original ATen: [aten.select_backward, aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.add, aten.unsqueeze, aten.squeeze, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_8:178
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_8.run(buf138, buf144, buf143, buf145, 122880, 200, stream=stream0)
            buf146 = empty_strided_cuda((1, 1, 384), (384, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full, permute_154, mul_263, permute_155, clone_55, view_144, permute_156, view_145, permute_157, clone_56, view_146, _generalized_scatter_6, _generalized_scatter_7, add_107, _generalized_scatter_8, add_108, unsqueeze_9, permute_158, squeeze_9, clone_57, view_147, sum_65], Original ATen: [aten.select_backward, aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.add, aten.unsqueeze, aten.squeeze, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_9:179
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf145, buf146, 384, 320, stream=stream0)
            buf147 = buf93; del buf93  # reuse
            # Topologically Sorted Source Nodes: [full, permute_154, mul_263, permute_155, clone_55, view_144, permute_156, view_145, permute_157, clone_56, view_146, _generalized_scatter_6, _generalized_scatter_7, add_107, _generalized_scatter_8, add_108, unsqueeze_9, permute_158, squeeze_9, clone_57], Original ATen: [aten.select_backward, aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.add, aten.unsqueeze, aten.squeeze]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_clone_mul_select_backward_squeeze_transpose_unsqueeze_view_10:180
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_mul_select_backward_squeeze_transpose_unsqueeze_view_10.run(buf138, buf144, buf143, buf147, 64000, 384, stream=stream0)
            buf148 = empty_strided_cuda((384, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full, permute_154, mul_263, permute_155, clone_55, view_144, permute_156, view_145, permute_157, clone_56, view_146, _generalized_scatter_6, _generalized_scatter_7, add_107, _generalized_scatter_8, add_108, unsqueeze_9, permute_158, squeeze_9, clone_57, view_147, view_149, permute_159, permute_162], Original ATen: [aten.select_backward, aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.add, aten.unsqueeze, aten.squeeze, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:181
            extern_kernels.mm(reinterpret_tensor(buf147, (384, 64000), (1, 384), 0), view_63, out=buf148)
            del view_63
            buf149 = reinterpret_tensor(buf144, (64000, 128), (128, 1), 0); del buf144  # reuse
            # Topologically Sorted Source Nodes: [full, permute_154, mul_263, permute_155, clone_55, view_144, permute_156, view_145, permute_157, clone_56, view_146, _generalized_scatter_6, _generalized_scatter_7, add_107, _generalized_scatter_8, add_108, unsqueeze_9, permute_158, squeeze_9, clone_57, view_147, view_149, multi_head_attention_forward_4, permute_161, mm_28], Original ATen: [aten.select_backward, aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.add, aten.unsqueeze, aten.squeeze, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:182
            extern_kernels.mm(reinterpret_tensor(buf147, (64000, 384), (384, 1), 0), primals_72, out=buf149)
            del primals_72
            buf150 = reinterpret_tensor(buf136, (500, 128, 1), (1, 500, 64000), 0); del buf136  # reuse
            # Topologically Sorted Source Nodes: [view_150, permute_163, mul_265, sum_66], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_backward_transpose_view_11:183
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_backward_transpose_view_11.run(buf149, primals_70, buf150, 64000, 128, stream=stream0)
            buf156 = buf126; del buf126  # reuse
            buf157 = reinterpret_tensor(buf143, (500, 128, 128), (16384, 128, 1), 0); del buf143  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [view_150, permute_163, mul_265, mul_266, mul_267, sum_67, mul_268, sub_69, sub_70, mul_269, mul_270, sum_68, sum_69, add_109, convert_element_type_11, mul_271, clone_58], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward, aten.add, aten.native_dropout_backward]
            workspace_11 = workspace_10; del workspace_10  # reuse
            stream0 = get_raw_stream(0)
            triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_transpose_view_12.run(buf156, buf149, primals_70, mul_80, div_18, buf150, gt_15, buf157, workspace_11, 64000, 128, stream=stream0)
            buf153 = workspace_11[0 * 1000 * 128 : (0 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            buf155 = workspace_11[1 * 1000 * 128 : (1 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            del div_18
            del gt_15
            del mul_80
            del primals_70
            buf158 = reinterpret_tensor(buf115, (64000, 512), (512, 1), 0); del buf115  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_11, mul_271, clone_58, view_151, linear_8, permute_164, mm_29], Original ATen: [aten.native_dropout_backward, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:184
            extern_kernels.mm(reinterpret_tensor(buf157, (64000, 128), (128, 1), 0), primals_68, out=buf158)
            del primals_68
            buf159 = empty_strided_cuda((128, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_11, mul_271, clone_58, view_151, permute_165, permute_167], Original ATen: [aten.native_dropout_backward, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:185
            extern_kernels.mm(reinterpret_tensor(buf157, (128, 64000), (1, 128), 0), view_61, out=buf159)
            del view_61
            buf160 = reinterpret_tensor(buf150, (1, 128, 500), (64000, 1, 128), 0); del buf150  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_11, mul_271, clone_58, view_151, sum_70], Original ATen: [aten.native_dropout_backward, aten.view, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_5:186
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_5.run(buf157, buf160, 64000, 128, stream=stream0)
            buf161 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_11, mul_271, clone_58, view_151, sum_70], Original ATen: [aten.native_dropout_backward, aten.view, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_6:187
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_6.run(buf160, buf161, 128, 500, stream=stream0)
            buf164 = reinterpret_tensor(buf104, (500, 128, 512), (65536, 512, 1), 0); del buf104  # reuse
            buf169 = buf164; del buf164  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [view_153, mul_274, mul_275, sum_71, linear_7, gelu_3, dropout_10, layer_norm_15, mul_276, sum_72, mul_277, sub_72, sub_73, div_19, mul_278, mul_279, sum_73, sum_74, convert_element_type_12, mul_280, clone_59, mul_283, mul_284, mul_285, exp_10, mul_286, mul_287, add_111, mul_288], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.gelu, aten.native_dropout, aten.native_layer_norm, aten.native_dropout_backward, aten.gelu_backward]
            workspace_12 = workspace_8; del workspace_8  # reuse
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_gelu_backward_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_view_0.run(buf169, buf158, primals_66, gt_14, addmm_11, getitem_31, rsqrt_15, workspace_12, 64000, 512, stream=stream0)
            buf166 = workspace_12[0 * 1000 * 512 : (0 + 1) * 1000 * 512].view(1000, 512).sum(dim=0)
            buf168 = workspace_12[1 * 1000 * 512 : (1 + 1) * 1000 * 512].view(1000, 512).sum(dim=0)
            del addmm_11
            del getitem_31
            del gt_14
            del primals_66
            del rsqrt_15
            buf170 = reinterpret_tensor(buf157, (64000, 128), (128, 1), 0); del buf157  # reuse
            # Topologically Sorted Source Nodes: [linear_7, gelu_3, convert_element_type_12, mul_280, clone_59, mul_283, mul_284, mul_285, exp_10, mul_286, mul_287, add_111, mul_288, view_154, permute_168, mm_31], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:188
            extern_kernels.mm(reinterpret_tensor(buf169, (64000, 512), (512, 1), 0), primals_64, out=buf170)
            del primals_64
            buf171 = empty_strided_cuda((512, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_7, gelu_3, convert_element_type_12, mul_280, clone_59, mul_283, mul_284, mul_285, exp_10, mul_286, mul_287, add_111, mul_288, view_154, permute_169, permute_171], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:189
            extern_kernels.mm(reinterpret_tensor(buf169, (512, 64000), (1, 512), 0), view_59, out=buf171)
            del view_59
            buf172 = buf118; del buf118  # reuse
            # Topologically Sorted Source Nodes: [linear_7, gelu_3, convert_element_type_12, mul_280, clone_59, mul_283, mul_284, mul_285, exp_10, mul_286, mul_287, add_111, mul_288, view_154, sum_75], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.sum]
            # [Provenance debug handles] triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_1:190
            stream0 = get_raw_stream(0)
            triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_1.run(buf169, buf172, 102400, 320, stream=stream0)
            buf173 = empty_strided_cuda((1, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_7, gelu_3, convert_element_type_12, mul_280, clone_59, mul_283, mul_284, mul_285, exp_10, mul_286, mul_287, add_111, mul_288, view_154, sum_75], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.sum]
            # [Provenance debug handles] triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_2:191
            stream0 = get_raw_stream(0)
            triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_2.run(buf172, buf173, 512, 200, stream=stream0)
            buf180 = buf156; del buf156  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [view_156, mul_290, mul_291, sum_76, mul_292, sum_77, mul_293, sub_75, sub_76, mul_294, mul_295, sum_78, sum_79, add_112], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            workspace_13 = workspace_11; del workspace_11  # reuse
            stream0 = get_raw_stream(0)
            triton_per_fused_add_native_layer_norm_backward_view_13.run(buf180, buf170, primals_62, mul_69, div_20, workspace_13, 64000, 128, stream=stream0)
            buf177 = workspace_13[0 * 1000 * 128 : (0 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            buf179 = workspace_13[1 * 1000 * 128 : (1 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            del div_20
            del mul_69
            del primals_62
            buf187 = reinterpret_tensor(buf170, (128, 500, 128), (64000, 128, 1), 0); del buf170  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [convert_element_type_13, mul_296, clone_60, multi_head_attention_forward_3, transpose_7, sub_77, mul_298, mul_299, mul_300, sum_80, mul_301, sum_81, mul_302, sub_78, sub_79, div_21, mul_303, mul_304, sum_82, sum_83, permute_172, clone_61], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone]
            workspace_14 = workspace_13; del workspace_13  # reuse
            stream0 = get_raw_stream(0)
            triton_per_fused_clone_native_dropout_backward_native_layer_norm_backward_transpose_view_4.run(buf180, gt_13, primals_60, addmm_10, getitem_27, rsqrt_13, buf187, workspace_14, 64000, 128, stream=stream0)
            buf184 = workspace_14[0 * 1000 * 128 : (0 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            buf186 = workspace_14[1 * 1000 * 128 : (1 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            del addmm_10
            del getitem_27
            del gt_13
            del primals_60
            del rsqrt_13
            buf188 = buf149; del buf149  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_13, mul_296, clone_60, multi_head_attention_forward_3, transpose_7, sub_77, mul_298, mul_299, mul_300, mul_302, sub_78, sub_79, div_21, mul_303, permute_172, clone_61, view_157, permute_173, mm_33], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:192
            extern_kernels.mm(reinterpret_tensor(buf187, (64000, 128), (128, 1), 0), primals_58, out=buf188)
            del primals_58
            buf189 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_13, mul_296, clone_60, multi_head_attention_forward_3, transpose_7, sub_77, mul_298, mul_299, mul_300, mul_302, sub_78, sub_79, div_21, mul_303, permute_172, clone_61, view_157, permute_174, permute_176], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:193
            extern_kernels.mm(reinterpret_tensor(buf187, (128, 64000), (1, 128), 0), view_56, out=buf189)
            del view_56
            buf190 = buf160; del buf160  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_13, mul_296, clone_60, multi_head_attention_forward_3, transpose_7, sub_77, mul_298, mul_299, mul_300, mul_302, sub_78, sub_79, div_21, mul_303, permute_172, clone_61, view_157, sum_84], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_5:194
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_5.run(buf187, buf190, 64000, 128, stream=stream0)
            buf191 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_13, mul_296, clone_60, multi_head_attention_forward_3, transpose_7, sub_77, mul_298, mul_299, mul_300, mul_302, sub_78, sub_79, div_21, mul_303, permute_172, clone_61, view_157, sum_84], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_6:195
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_6.run(buf190, buf191, 128, 500, stream=stream0)
            buf192 = reinterpret_tensor(buf187, (4000, 128, 16), (2048, 16, 1), 0); del buf187  # reuse
            # Topologically Sorted Source Nodes: [view_159, permute_177, bmm_19], Original ATen: [aten.view, aten.transpose, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:196
            extern_kernels.bmm(permute_178, reinterpret_tensor(buf188, (4000, 128, 16), (16, 64000, 1), 0), out=buf192)
            del permute_178
            buf193 = buf140; del buf140  # reuse
            # Topologically Sorted Source Nodes: [view_159, permute_177, bmm_20], Original ATen: [aten.view, aten.transpose, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:197
            extern_kernels.bmm(reinterpret_tensor(buf188, (4000, 128, 16), (16, 64000, 1), 0), permute_179, out=buf193)
            del permute_179
            buf194 = buf193; del buf193  # reuse
            buf196 = baddbmm_3; del baddbmm_3  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_14, mul_305, clone_62, multi_head_attention_forward_3, mul_307, sum_85, neg_3, fma_3], Original ATen: [aten.native_dropout_backward, aten._softmax, aten._softmax_backward_data]
            # [Provenance debug handles] triton_per_fused__softmax__softmax_backward_data_native_dropout_backward_7:198
            stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data_native_dropout_backward_7.run(buf194, buf196, gt_12, amax_3, sum_4, 512000, 128, stream=stream0)
            del amax_3
            del gt_12
            del sum_4
            buf197 = reinterpret_tensor(buf188, (4000, 128, 16), (2048, 16, 1), 0); del buf188  # reuse
            # Topologically Sorted Source Nodes: [bmm_21], Original ATen: [aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:199
            extern_kernels.bmm(buf196, permute_180, out=buf197)
            del permute_180
            buf198 = reinterpret_tensor(buf138, (4000, 16, 128), (2048, 128, 1), 0); del buf138  # reuse
            # Topologically Sorted Source Nodes: [bmm_22], Original ATen: [aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:200
            extern_kernels.bmm(permute_181, buf196, out=buf198)
            del permute_181
            buf199 = buf145; del buf145  # reuse
            # Topologically Sorted Source Nodes: [full, permute_182, mul_308, permute_183, clone_63, view_160, permute_184, view_161, permute_185, clone_64, view_162, _generalized_scatter_9, _generalized_scatter_10, add_114, _generalized_scatter_11, add_115, unsqueeze_10, permute_186, squeeze_10, clone_65, view_163, sum_86], Original ATen: [aten.select_backward, aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.add, aten.unsqueeze, aten.squeeze, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_8:201
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_8.run(buf192, buf198, buf197, buf199, 122880, 200, stream=stream0)
            buf200 = empty_strided_cuda((1, 1, 384), (384, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full, permute_182, mul_308, permute_183, clone_63, view_160, permute_184, view_161, permute_185, clone_64, view_162, _generalized_scatter_9, _generalized_scatter_10, add_114, _generalized_scatter_11, add_115, unsqueeze_10, permute_186, squeeze_10, clone_65, view_163, sum_86], Original ATen: [aten.select_backward, aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.add, aten.unsqueeze, aten.squeeze, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_9:202
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf199, buf200, 384, 320, stream=stream0)
            buf201 = buf147; del buf147  # reuse
            # Topologically Sorted Source Nodes: [full, permute_182, mul_308, permute_183, clone_63, view_160, permute_184, view_161, permute_185, clone_64, view_162, _generalized_scatter_9, _generalized_scatter_10, add_114, _generalized_scatter_11, add_115, unsqueeze_10, permute_186, squeeze_10, clone_65], Original ATen: [aten.select_backward, aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.add, aten.unsqueeze, aten.squeeze]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_clone_mul_select_backward_squeeze_transpose_unsqueeze_view_10:203
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_mul_select_backward_squeeze_transpose_unsqueeze_view_10.run(buf192, buf198, buf197, buf201, 64000, 384, stream=stream0)
            buf202 = empty_strided_cuda((384, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full, permute_182, mul_308, permute_183, clone_63, view_160, permute_184, view_161, permute_185, clone_64, view_162, _generalized_scatter_9, _generalized_scatter_10, add_114, _generalized_scatter_11, add_115, unsqueeze_10, permute_186, squeeze_10, clone_65, view_163, view_165, permute_187, permute_190], Original ATen: [aten.select_backward, aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.add, aten.unsqueeze, aten.squeeze, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:204
            extern_kernels.mm(reinterpret_tensor(buf201, (384, 64000), (1, 384), 0), view_48, out=buf202)
            del view_48
            buf203 = reinterpret_tensor(buf198, (64000, 128), (128, 1), 0); del buf198  # reuse
            # Topologically Sorted Source Nodes: [full, permute_182, mul_308, permute_183, clone_63, view_160, permute_184, view_161, permute_185, clone_64, view_162, _generalized_scatter_9, _generalized_scatter_10, add_114, _generalized_scatter_11, add_115, unsqueeze_10, permute_186, squeeze_10, clone_65, view_163, view_165, multi_head_attention_forward_3, permute_189, mm_36], Original ATen: [aten.select_backward, aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.add, aten.unsqueeze, aten.squeeze, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:205
            extern_kernels.mm(reinterpret_tensor(buf201, (64000, 384), (384, 1), 0), primals_56, out=buf203)
            del primals_56
            buf204 = reinterpret_tensor(buf190, (500, 128, 1), (1, 500, 64000), 0); del buf190  # reuse
            # Topologically Sorted Source Nodes: [view_166, permute_191, mul_310, sum_87], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_backward_transpose_view_11:206
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_backward_transpose_view_11.run(buf203, primals_54, buf204, 64000, 128, stream=stream0)
            buf210 = buf180; del buf180  # reuse
            buf211 = reinterpret_tensor(buf197, (500, 128, 128), (16384, 128, 1), 0); del buf197  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [view_166, permute_191, mul_310, mul_311, mul_312, sum_88, mul_313, sub_81, sub_82, mul_314, mul_315, sum_89, sum_90, add_116, convert_element_type_15, mul_316, clone_66], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward, aten.add, aten.native_dropout_backward]
            workspace_15 = workspace_14; del workspace_14  # reuse
            stream0 = get_raw_stream(0)
            triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_transpose_view_12.run(buf210, buf203, primals_54, mul_60, div_22, buf204, gt_11, buf211, workspace_15, 64000, 128, stream=stream0)
            buf207 = workspace_15[0 * 1000 * 128 : (0 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            buf209 = workspace_15[1 * 1000 * 128 : (1 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            del div_22
            del gt_11
            del mul_60
            del primals_54
            buf212 = reinterpret_tensor(buf169, (64000, 512), (512, 1), 0); del buf169  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_15, mul_316, clone_66, view_167, linear_6, permute_192, mm_37], Original ATen: [aten.native_dropout_backward, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:207
            extern_kernels.mm(reinterpret_tensor(buf211, (64000, 128), (128, 1), 0), primals_52, out=buf212)
            del primals_52
            buf213 = empty_strided_cuda((128, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_15, mul_316, clone_66, view_167, permute_193, permute_195], Original ATen: [aten.native_dropout_backward, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:208
            extern_kernels.mm(reinterpret_tensor(buf211, (128, 64000), (1, 128), 0), view_46, out=buf213)
            del view_46
            buf214 = reinterpret_tensor(buf204, (1, 128, 500), (64000, 1, 128), 0); del buf204  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_15, mul_316, clone_66, view_167, sum_91], Original ATen: [aten.native_dropout_backward, aten.view, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_5:209
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_5.run(buf211, buf214, 64000, 128, stream=stream0)
            buf215 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_15, mul_316, clone_66, view_167, sum_91], Original ATen: [aten.native_dropout_backward, aten.view, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_6:210
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_6.run(buf214, buf215, 128, 500, stream=stream0)
            buf218 = reinterpret_tensor(buf158, (500, 128, 512), (65536, 512, 1), 0); del buf158  # reuse
            buf223 = buf218; del buf218  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [view_169, mul_319, mul_320, sum_92, linear_5, gelu_2, dropout_7, layer_norm_11, mul_321, sum_93, mul_322, sub_84, sub_85, div_23, mul_323, mul_324, sum_94, sum_95, convert_element_type_16, mul_325, clone_67, mul_328, mul_329, mul_330, exp_11, mul_331, mul_332, add_118, mul_333], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.gelu, aten.native_dropout, aten.native_layer_norm, aten.native_dropout_backward, aten.gelu_backward]
            workspace_16 = workspace_12; del workspace_12  # reuse
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_gelu_backward_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_view_0.run(buf223, buf212, primals_50, gt_10, addmm_8, getitem_23, rsqrt_11, workspace_16, 64000, 512, stream=stream0)
            buf220 = workspace_16[0 * 1000 * 512 : (0 + 1) * 1000 * 512].view(1000, 512).sum(dim=0)
            buf222 = workspace_16[1 * 1000 * 512 : (1 + 1) * 1000 * 512].view(1000, 512).sum(dim=0)
            del addmm_8
            del getitem_23
            del gt_10
            del primals_50
            del rsqrt_11
            buf224 = reinterpret_tensor(buf211, (64000, 128), (128, 1), 0); del buf211  # reuse
            # Topologically Sorted Source Nodes: [linear_5, gelu_2, convert_element_type_16, mul_325, clone_67, mul_328, mul_329, mul_330, exp_11, mul_331, mul_332, add_118, mul_333, view_170, permute_196, mm_39], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:211
            extern_kernels.mm(reinterpret_tensor(buf223, (64000, 512), (512, 1), 0), primals_48, out=buf224)
            del primals_48
            buf225 = empty_strided_cuda((512, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_5, gelu_2, convert_element_type_16, mul_325, clone_67, mul_328, mul_329, mul_330, exp_11, mul_331, mul_332, add_118, mul_333, view_170, permute_197, permute_199], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:212
            extern_kernels.mm(reinterpret_tensor(buf223, (512, 64000), (1, 512), 0), view_44, out=buf225)
            del view_44
            buf226 = buf172; del buf172  # reuse
            # Topologically Sorted Source Nodes: [linear_5, gelu_2, convert_element_type_16, mul_325, clone_67, mul_328, mul_329, mul_330, exp_11, mul_331, mul_332, add_118, mul_333, view_170, sum_96], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.sum]
            # [Provenance debug handles] triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_1:213
            stream0 = get_raw_stream(0)
            triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_1.run(buf223, buf226, 102400, 320, stream=stream0)
            buf227 = empty_strided_cuda((1, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_5, gelu_2, convert_element_type_16, mul_325, clone_67, mul_328, mul_329, mul_330, exp_11, mul_331, mul_332, add_118, mul_333, view_170, sum_96], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.sum]
            # [Provenance debug handles] triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_2:214
            stream0 = get_raw_stream(0)
            triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_2.run(buf226, buf227, 512, 200, stream=stream0)
            buf234 = buf210; del buf210  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [view_172, mul_335, mul_336, sum_97, mul_337, sum_98, mul_338, sub_87, sub_88, mul_339, mul_340, sum_99, sum_100, add_119], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            workspace_17 = workspace_15; del workspace_15  # reuse
            stream0 = get_raw_stream(0)
            triton_per_fused_add_native_layer_norm_backward_view_13.run(buf234, buf224, primals_46, mul_49, div_24, workspace_17, 64000, 128, stream=stream0)
            buf231 = workspace_17[0 * 1000 * 128 : (0 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            buf233 = workspace_17[1 * 1000 * 128 : (1 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            del div_24
            del mul_49
            del primals_46
            buf241 = reinterpret_tensor(buf224, (128, 500, 128), (64000, 128, 1), 0); del buf224  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [convert_element_type_17, mul_341, clone_68, multi_head_attention_forward_2, transpose_5, sub_89, mul_343, mul_344, mul_345, sum_101, mul_346, sum_102, mul_347, sub_90, sub_91, div_25, mul_348, mul_349, sum_103, sum_104, permute_200, clone_69], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone]
            workspace_18 = workspace_17; del workspace_17  # reuse
            stream0 = get_raw_stream(0)
            triton_per_fused_clone_native_dropout_backward_native_layer_norm_backward_transpose_view_4.run(buf234, gt_9, primals_44, addmm_7, getitem_19, rsqrt_9, buf241, workspace_18, 64000, 128, stream=stream0)
            buf238 = workspace_18[0 * 1000 * 128 : (0 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            buf240 = workspace_18[1 * 1000 * 128 : (1 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            del addmm_7
            del getitem_19
            del gt_9
            del primals_44
            del rsqrt_9
            buf242 = buf203; del buf203  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_17, mul_341, clone_68, multi_head_attention_forward_2, transpose_5, sub_89, mul_343, mul_344, mul_345, mul_347, sub_90, sub_91, div_25, mul_348, permute_200, clone_69, view_173, permute_201, mm_41], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:215
            extern_kernels.mm(reinterpret_tensor(buf241, (64000, 128), (128, 1), 0), primals_42, out=buf242)
            del primals_42
            buf243 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_17, mul_341, clone_68, multi_head_attention_forward_2, transpose_5, sub_89, mul_343, mul_344, mul_345, mul_347, sub_90, sub_91, div_25, mul_348, permute_200, clone_69, view_173, permute_202, permute_204], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:216
            extern_kernels.mm(reinterpret_tensor(buf241, (128, 64000), (1, 128), 0), view_41, out=buf243)
            del view_41
            buf244 = buf214; del buf214  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_17, mul_341, clone_68, multi_head_attention_forward_2, transpose_5, sub_89, mul_343, mul_344, mul_345, mul_347, sub_90, sub_91, div_25, mul_348, permute_200, clone_69, view_173, sum_105], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_5:217
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_5.run(buf241, buf244, 64000, 128, stream=stream0)
            buf245 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_17, mul_341, clone_68, multi_head_attention_forward_2, transpose_5, sub_89, mul_343, mul_344, mul_345, mul_347, sub_90, sub_91, div_25, mul_348, permute_200, clone_69, view_173, sum_105], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_6:218
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_6.run(buf244, buf245, 128, 500, stream=stream0)
            buf246 = reinterpret_tensor(buf241, (4000, 128, 16), (2048, 16, 1), 0); del buf241  # reuse
            # Topologically Sorted Source Nodes: [view_175, permute_205, bmm_23], Original ATen: [aten.view, aten.transpose, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:219
            extern_kernels.bmm(permute_206, reinterpret_tensor(buf242, (4000, 128, 16), (16, 64000, 1), 0), out=buf246)
            del permute_206
            buf247 = buf194; del buf194  # reuse
            # Topologically Sorted Source Nodes: [view_175, permute_205, bmm_24], Original ATen: [aten.view, aten.transpose, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:220
            extern_kernels.bmm(reinterpret_tensor(buf242, (4000, 128, 16), (16, 64000, 1), 0), permute_207, out=buf247)
            del permute_207
            buf248 = buf247; del buf247  # reuse
            buf250 = baddbmm_2; del baddbmm_2  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_18, mul_350, clone_70, multi_head_attention_forward_2, mul_352, sum_106, neg_4, fma_4], Original ATen: [aten.native_dropout_backward, aten._softmax, aten._softmax_backward_data]
            # [Provenance debug handles] triton_per_fused__softmax__softmax_backward_data_native_dropout_backward_7:221
            stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data_native_dropout_backward_7.run(buf248, buf250, gt_8, amax_2, sum_3, 512000, 128, stream=stream0)
            del amax_2
            del gt_8
            del sum_3
            buf251 = reinterpret_tensor(buf242, (4000, 128, 16), (2048, 16, 1), 0); del buf242  # reuse
            # Topologically Sorted Source Nodes: [bmm_25], Original ATen: [aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:222
            extern_kernels.bmm(buf250, permute_208, out=buf251)
            del permute_208
            buf252 = reinterpret_tensor(buf192, (4000, 16, 128), (2048, 128, 1), 0); del buf192  # reuse
            # Topologically Sorted Source Nodes: [bmm_26], Original ATen: [aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:223
            extern_kernels.bmm(permute_209, buf250, out=buf252)
            del permute_209
            buf253 = buf199; del buf199  # reuse
            # Topologically Sorted Source Nodes: [full, permute_210, mul_353, permute_211, clone_71, view_176, permute_212, view_177, permute_213, clone_72, view_178, _generalized_scatter_12, _generalized_scatter_13, add_121, _generalized_scatter_14, add_122, unsqueeze_11, permute_214, squeeze_11, clone_73, view_179, sum_107], Original ATen: [aten.select_backward, aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.add, aten.unsqueeze, aten.squeeze, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_8:224
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_8.run(buf246, buf252, buf251, buf253, 122880, 200, stream=stream0)
            buf254 = empty_strided_cuda((1, 1, 384), (384, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full, permute_210, mul_353, permute_211, clone_71, view_176, permute_212, view_177, permute_213, clone_72, view_178, _generalized_scatter_12, _generalized_scatter_13, add_121, _generalized_scatter_14, add_122, unsqueeze_11, permute_214, squeeze_11, clone_73, view_179, sum_107], Original ATen: [aten.select_backward, aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.add, aten.unsqueeze, aten.squeeze, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_9:225
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf253, buf254, 384, 320, stream=stream0)
            buf255 = buf201; del buf201  # reuse
            # Topologically Sorted Source Nodes: [full, permute_210, mul_353, permute_211, clone_71, view_176, permute_212, view_177, permute_213, clone_72, view_178, _generalized_scatter_12, _generalized_scatter_13, add_121, _generalized_scatter_14, add_122, unsqueeze_11, permute_214, squeeze_11, clone_73], Original ATen: [aten.select_backward, aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.add, aten.unsqueeze, aten.squeeze]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_clone_mul_select_backward_squeeze_transpose_unsqueeze_view_10:226
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_mul_select_backward_squeeze_transpose_unsqueeze_view_10.run(buf246, buf252, buf251, buf255, 64000, 384, stream=stream0)
            buf256 = empty_strided_cuda((384, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full, permute_210, mul_353, permute_211, clone_71, view_176, permute_212, view_177, permute_213, clone_72, view_178, _generalized_scatter_12, _generalized_scatter_13, add_121, _generalized_scatter_14, add_122, unsqueeze_11, permute_214, squeeze_11, clone_73, view_179, view_181, permute_215, permute_218], Original ATen: [aten.select_backward, aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.add, aten.unsqueeze, aten.squeeze, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:227
            extern_kernels.mm(reinterpret_tensor(buf255, (384, 64000), (1, 384), 0), view_33, out=buf256)
            del view_33
            buf257 = reinterpret_tensor(buf252, (64000, 128), (128, 1), 0); del buf252  # reuse
            # Topologically Sorted Source Nodes: [full, permute_210, mul_353, permute_211, clone_71, view_176, permute_212, view_177, permute_213, clone_72, view_178, _generalized_scatter_12, _generalized_scatter_13, add_121, _generalized_scatter_14, add_122, unsqueeze_11, permute_214, squeeze_11, clone_73, view_179, view_181, multi_head_attention_forward_2, permute_217, mm_44], Original ATen: [aten.select_backward, aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.add, aten.unsqueeze, aten.squeeze, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:228
            extern_kernels.mm(reinterpret_tensor(buf255, (64000, 384), (384, 1), 0), primals_40, out=buf257)
            del primals_40
            buf258 = reinterpret_tensor(buf244, (500, 128, 1), (1, 500, 64000), 0); del buf244  # reuse
            # Topologically Sorted Source Nodes: [view_182, permute_219, mul_355, sum_108], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_backward_transpose_view_11:229
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_backward_transpose_view_11.run(buf257, primals_38, buf258, 64000, 128, stream=stream0)
            buf264 = buf234; del buf234  # reuse
            buf265 = reinterpret_tensor(buf251, (500, 128, 128), (16384, 128, 1), 0); del buf251  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [view_182, permute_219, mul_355, mul_356, mul_357, sum_109, mul_358, sub_93, sub_94, mul_359, mul_360, sum_110, sum_111, add_123, convert_element_type_19, mul_361, clone_74], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward, aten.add, aten.native_dropout_backward]
            workspace_19 = workspace_18; del workspace_18  # reuse
            stream0 = get_raw_stream(0)
            triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_transpose_view_12.run(buf264, buf257, primals_38, mul_40, div_26, buf258, gt_7, buf265, workspace_19, 64000, 128, stream=stream0)
            buf261 = workspace_19[0 * 1000 * 128 : (0 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            buf263 = workspace_19[1 * 1000 * 128 : (1 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            del div_26
            del gt_7
            del mul_40
            del primals_38
            buf266 = reinterpret_tensor(buf223, (64000, 512), (512, 1), 0); del buf223  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_19, mul_361, clone_74, view_183, linear_4, permute_220, mm_45], Original ATen: [aten.native_dropout_backward, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:230
            extern_kernels.mm(reinterpret_tensor(buf265, (64000, 128), (128, 1), 0), primals_36, out=buf266)
            del primals_36
            buf267 = empty_strided_cuda((128, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_19, mul_361, clone_74, view_183, permute_221, permute_223], Original ATen: [aten.native_dropout_backward, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:231
            extern_kernels.mm(reinterpret_tensor(buf265, (128, 64000), (1, 128), 0), view_31, out=buf267)
            del view_31
            buf268 = reinterpret_tensor(buf258, (1, 128, 500), (64000, 1, 128), 0); del buf258  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_19, mul_361, clone_74, view_183, sum_112], Original ATen: [aten.native_dropout_backward, aten.view, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_5:232
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_5.run(buf265, buf268, 64000, 128, stream=stream0)
            buf269 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_19, mul_361, clone_74, view_183, sum_112], Original ATen: [aten.native_dropout_backward, aten.view, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_6:233
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_6.run(buf268, buf269, 128, 500, stream=stream0)
            buf272 = reinterpret_tensor(buf212, (500, 128, 512), (65536, 512, 1), 0); del buf212  # reuse
            buf277 = buf272; del buf272  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [view_185, mul_364, mul_365, sum_113, linear_3, gelu_1, dropout_4, layer_norm_7, mul_366, sum_114, mul_367, sub_96, sub_97, div_27, mul_368, mul_369, sum_115, sum_116, convert_element_type_20, mul_370, clone_75, mul_373, mul_374, mul_375, exp_12, mul_376, mul_377, add_125, mul_378], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.gelu, aten.native_dropout, aten.native_layer_norm, aten.native_dropout_backward, aten.gelu_backward]
            workspace_20 = workspace_16; del workspace_16  # reuse
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_gelu_backward_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_view_0.run(buf277, buf266, primals_34, gt_6, addmm_5, getitem_15, rsqrt_7, workspace_20, 64000, 512, stream=stream0)
            buf274 = workspace_20[0 * 1000 * 512 : (0 + 1) * 1000 * 512].view(1000, 512).sum(dim=0)
            buf276 = workspace_20[1 * 1000 * 512 : (1 + 1) * 1000 * 512].view(1000, 512).sum(dim=0)
            del addmm_5
            del getitem_15
            del gt_6
            del primals_34
            del rsqrt_7
            buf278 = reinterpret_tensor(buf265, (64000, 128), (128, 1), 0); del buf265  # reuse
            # Topologically Sorted Source Nodes: [linear_3, gelu_1, convert_element_type_20, mul_370, clone_75, mul_373, mul_374, mul_375, exp_12, mul_376, mul_377, add_125, mul_378, view_186, permute_224, mm_47], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:234
            extern_kernels.mm(reinterpret_tensor(buf277, (64000, 512), (512, 1), 0), primals_32, out=buf278)
            del primals_32
            buf279 = empty_strided_cuda((512, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_3, gelu_1, convert_element_type_20, mul_370, clone_75, mul_373, mul_374, mul_375, exp_12, mul_376, mul_377, add_125, mul_378, view_186, permute_225, permute_227], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:235
            extern_kernels.mm(reinterpret_tensor(buf277, (512, 64000), (1, 512), 0), view_29, out=buf279)
            del view_29
            buf280 = buf226; del buf226  # reuse
            # Topologically Sorted Source Nodes: [linear_3, gelu_1, convert_element_type_20, mul_370, clone_75, mul_373, mul_374, mul_375, exp_12, mul_376, mul_377, add_125, mul_378, view_186, sum_117], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.sum]
            # [Provenance debug handles] triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_1:236
            stream0 = get_raw_stream(0)
            triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_1.run(buf277, buf280, 102400, 320, stream=stream0)
            buf281 = empty_strided_cuda((1, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_3, gelu_1, convert_element_type_20, mul_370, clone_75, mul_373, mul_374, mul_375, exp_12, mul_376, mul_377, add_125, mul_378, view_186, sum_117], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.sum]
            # [Provenance debug handles] triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_2:237
            stream0 = get_raw_stream(0)
            triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_2.run(buf280, buf281, 512, 200, stream=stream0)
            buf288 = buf264; del buf264  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [view_188, mul_380, mul_381, sum_118, mul_382, sum_119, mul_383, sub_99, sub_100, mul_384, mul_385, sum_120, sum_121, add_126], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            workspace_21 = workspace_19; del workspace_19  # reuse
            stream0 = get_raw_stream(0)
            triton_per_fused_add_native_layer_norm_backward_view_13.run(buf288, buf278, primals_30, mul_29, div_28, workspace_21, 64000, 128, stream=stream0)
            buf285 = workspace_21[0 * 1000 * 128 : (0 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            buf287 = workspace_21[1 * 1000 * 128 : (1 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            del div_28
            del mul_29
            del primals_30
            buf295 = reinterpret_tensor(buf278, (128, 500, 128), (64000, 128, 1), 0); del buf278  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [convert_element_type_21, mul_386, clone_76, multi_head_attention_forward_1, transpose_3, sub_101, mul_388, mul_389, mul_390, sum_122, mul_391, sum_123, mul_392, sub_102, sub_103, div_29, mul_393, mul_394, sum_124, sum_125, permute_228, clone_77], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone]
            workspace_22 = workspace_21; del workspace_21  # reuse
            stream0 = get_raw_stream(0)
            triton_per_fused_clone_native_dropout_backward_native_layer_norm_backward_transpose_view_4.run(buf288, gt_5, primals_28, addmm_4, getitem_11, rsqrt_5, buf295, workspace_22, 64000, 128, stream=stream0)
            buf292 = workspace_22[0 * 1000 * 128 : (0 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            buf294 = workspace_22[1 * 1000 * 128 : (1 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            del addmm_4
            del getitem_11
            del gt_5
            del primals_28
            del rsqrt_5
            buf296 = buf257; del buf257  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_21, mul_386, clone_76, multi_head_attention_forward_1, transpose_3, sub_101, mul_388, mul_389, mul_390, mul_392, sub_102, sub_103, div_29, mul_393, permute_228, clone_77, view_189, permute_229, mm_49], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:238
            extern_kernels.mm(reinterpret_tensor(buf295, (64000, 128), (128, 1), 0), primals_26, out=buf296)
            del primals_26
            buf297 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_21, mul_386, clone_76, multi_head_attention_forward_1, transpose_3, sub_101, mul_388, mul_389, mul_390, mul_392, sub_102, sub_103, div_29, mul_393, permute_228, clone_77, view_189, permute_230, permute_232], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:239
            extern_kernels.mm(reinterpret_tensor(buf295, (128, 64000), (1, 128), 0), view_26, out=buf297)
            del view_26
            buf298 = buf268; del buf268  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_21, mul_386, clone_76, multi_head_attention_forward_1, transpose_3, sub_101, mul_388, mul_389, mul_390, mul_392, sub_102, sub_103, div_29, mul_393, permute_228, clone_77, view_189, sum_126], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_5:240
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_5.run(buf295, buf298, 64000, 128, stream=stream0)
            buf299 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_21, mul_386, clone_76, multi_head_attention_forward_1, transpose_3, sub_101, mul_388, mul_389, mul_390, mul_392, sub_102, sub_103, div_29, mul_393, permute_228, clone_77, view_189, sum_126], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_6:241
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_6.run(buf298, buf299, 128, 500, stream=stream0)
            buf300 = reinterpret_tensor(buf295, (4000, 128, 16), (2048, 16, 1), 0); del buf295  # reuse
            # Topologically Sorted Source Nodes: [view_191, permute_233, bmm_27], Original ATen: [aten.view, aten.transpose, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:242
            extern_kernels.bmm(permute_234, reinterpret_tensor(buf296, (4000, 128, 16), (16, 64000, 1), 0), out=buf300)
            del permute_234
            buf301 = buf248; del buf248  # reuse
            # Topologically Sorted Source Nodes: [view_191, permute_233, bmm_28], Original ATen: [aten.view, aten.transpose, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:243
            extern_kernels.bmm(reinterpret_tensor(buf296, (4000, 128, 16), (16, 64000, 1), 0), permute_235, out=buf301)
            del permute_235
            buf302 = buf301; del buf301  # reuse
            buf304 = baddbmm_1; del baddbmm_1  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_22, mul_395, clone_78, multi_head_attention_forward_1, mul_397, sum_127, neg_5, fma_5], Original ATen: [aten.native_dropout_backward, aten._softmax, aten._softmax_backward_data]
            # [Provenance debug handles] triton_per_fused__softmax__softmax_backward_data_native_dropout_backward_7:244
            stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data_native_dropout_backward_7.run(buf302, buf304, gt_4, amax_1, sum_2, 512000, 128, stream=stream0)
            del amax_1
            del gt_4
            del sum_2
            buf305 = reinterpret_tensor(buf296, (4000, 128, 16), (2048, 16, 1), 0); del buf296  # reuse
            # Topologically Sorted Source Nodes: [bmm_29], Original ATen: [aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:245
            extern_kernels.bmm(buf304, permute_236, out=buf305)
            del permute_236
            buf306 = reinterpret_tensor(buf246, (4000, 16, 128), (2048, 128, 1), 0); del buf246  # reuse
            # Topologically Sorted Source Nodes: [bmm_30], Original ATen: [aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:246
            extern_kernels.bmm(permute_237, buf304, out=buf306)
            del permute_237
            buf307 = buf253; del buf253  # reuse
            # Topologically Sorted Source Nodes: [full, permute_238, mul_398, permute_239, clone_79, view_192, permute_240, view_193, permute_241, clone_80, view_194, _generalized_scatter_15, _generalized_scatter_16, add_128, _generalized_scatter_17, add_129, unsqueeze_12, permute_242, squeeze_12, clone_81, view_195, sum_128], Original ATen: [aten.select_backward, aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.add, aten.unsqueeze, aten.squeeze, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_8:247
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_8.run(buf300, buf306, buf305, buf307, 122880, 200, stream=stream0)
            buf308 = empty_strided_cuda((1, 1, 384), (384, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full, permute_238, mul_398, permute_239, clone_79, view_192, permute_240, view_193, permute_241, clone_80, view_194, _generalized_scatter_15, _generalized_scatter_16, add_128, _generalized_scatter_17, add_129, unsqueeze_12, permute_242, squeeze_12, clone_81, view_195, sum_128], Original ATen: [aten.select_backward, aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.add, aten.unsqueeze, aten.squeeze, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_9:248
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf307, buf308, 384, 320, stream=stream0)
            buf309 = buf255; del buf255  # reuse
            # Topologically Sorted Source Nodes: [full, permute_238, mul_398, permute_239, clone_79, view_192, permute_240, view_193, permute_241, clone_80, view_194, _generalized_scatter_15, _generalized_scatter_16, add_128, _generalized_scatter_17, add_129, unsqueeze_12, permute_242, squeeze_12, clone_81], Original ATen: [aten.select_backward, aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.add, aten.unsqueeze, aten.squeeze]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_clone_mul_select_backward_squeeze_transpose_unsqueeze_view_10:249
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_mul_select_backward_squeeze_transpose_unsqueeze_view_10.run(buf300, buf306, buf305, buf309, 64000, 384, stream=stream0)
            buf310 = empty_strided_cuda((384, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full, permute_238, mul_398, permute_239, clone_79, view_192, permute_240, view_193, permute_241, clone_80, view_194, _generalized_scatter_15, _generalized_scatter_16, add_128, _generalized_scatter_17, add_129, unsqueeze_12, permute_242, squeeze_12, clone_81, view_195, view_197, permute_243, permute_246], Original ATen: [aten.select_backward, aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.add, aten.unsqueeze, aten.squeeze, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:250
            extern_kernels.mm(reinterpret_tensor(buf309, (384, 64000), (1, 384), 0), view_18, out=buf310)
            del view_18
            buf311 = reinterpret_tensor(buf306, (64000, 128), (128, 1), 0); del buf306  # reuse
            # Topologically Sorted Source Nodes: [full, permute_238, mul_398, permute_239, clone_79, view_192, permute_240, view_193, permute_241, clone_80, view_194, _generalized_scatter_15, _generalized_scatter_16, add_128, _generalized_scatter_17, add_129, unsqueeze_12, permute_242, squeeze_12, clone_81, view_195, view_197, multi_head_attention_forward_1, permute_245, mm_52], Original ATen: [aten.select_backward, aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.add, aten.unsqueeze, aten.squeeze, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:251
            extern_kernels.mm(reinterpret_tensor(buf309, (64000, 384), (384, 1), 0), primals_24, out=buf311)
            del primals_24
            buf312 = reinterpret_tensor(buf298, (500, 128, 1), (1, 500, 64000), 0); del buf298  # reuse
            # Topologically Sorted Source Nodes: [view_198, permute_247, mul_400, sum_129], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_backward_transpose_view_11:252
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_backward_transpose_view_11.run(buf311, primals_22, buf312, 64000, 128, stream=stream0)
            buf318 = buf288; del buf288  # reuse
            buf319 = reinterpret_tensor(buf305, (500, 128, 128), (16384, 128, 1), 0); del buf305  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [view_198, permute_247, mul_400, mul_401, mul_402, sum_130, mul_403, sub_105, sub_106, mul_404, mul_405, sum_131, sum_132, add_130, convert_element_type_23, mul_406, clone_82], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward, aten.add, aten.native_dropout_backward]
            workspace_23 = workspace_22; del workspace_22  # reuse
            stream0 = get_raw_stream(0)
            triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_transpose_view_12.run(buf318, buf311, primals_22, mul_20, div_30, buf312, gt_3, buf319, workspace_23, 64000, 128, stream=stream0)
            buf315 = workspace_23[0 * 1000 * 128 : (0 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            buf317 = workspace_23[1 * 1000 * 128 : (1 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            del div_30
            del gt_3
            del mul_20
            del primals_22
            buf320 = reinterpret_tensor(buf277, (64000, 512), (512, 1), 0); del buf277  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_23, mul_406, clone_82, view_199, linear_2, permute_248, mm_53], Original ATen: [aten.native_dropout_backward, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:253
            extern_kernels.mm(reinterpret_tensor(buf319, (64000, 128), (128, 1), 0), primals_20, out=buf320)
            del primals_20
            buf321 = empty_strided_cuda((128, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_23, mul_406, clone_82, view_199, permute_249, permute_251], Original ATen: [aten.native_dropout_backward, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:254
            extern_kernels.mm(reinterpret_tensor(buf319, (128, 64000), (1, 128), 0), view_16, out=buf321)
            del view_16
            buf322 = reinterpret_tensor(buf312, (1, 128, 500), (64000, 1, 128), 0); del buf312  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_23, mul_406, clone_82, view_199, sum_133], Original ATen: [aten.native_dropout_backward, aten.view, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_5:255
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_5.run(buf319, buf322, 64000, 128, stream=stream0)
            buf323 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_23, mul_406, clone_82, view_199, sum_133], Original ATen: [aten.native_dropout_backward, aten.view, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_6:256
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_6.run(buf322, buf323, 128, 500, stream=stream0)
            buf326 = reinterpret_tensor(buf266, (500, 128, 512), (65536, 512, 1), 0); del buf266  # reuse
            buf331 = buf326; del buf326  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [view_201, mul_409, mul_410, sum_134, linear_1, gelu, dropout_1, layer_norm_3, mul_411, sum_135, mul_412, sub_108, sub_109, div_31, mul_413, mul_414, sum_136, sum_137, convert_element_type_24, mul_415, clone_83, mul_418, mul_419, mul_420, exp_13, mul_421, mul_422, add_132, mul_423], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.gelu, aten.native_dropout, aten.native_layer_norm, aten.native_dropout_backward, aten.gelu_backward]
            workspace_24 = workspace_20; del workspace_20  # reuse
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_gelu_backward_native_dropout_native_dropout_backward_native_layer_norm_native_layer_norm_backward_view_0.run(buf331, buf320, primals_18, gt_2, addmm_2, getitem_7, rsqrt_3, workspace_24, 64000, 512, stream=stream0)
            buf328 = workspace_24[0 * 1000 * 512 : (0 + 1) * 1000 * 512].view(1000, 512).sum(dim=0)
            buf330 = workspace_24[1 * 1000 * 512 : (1 + 1) * 1000 * 512].view(1000, 512).sum(dim=0)
            del addmm_2
            del buf320
            del getitem_7
            del gt_2
            del primals_18
            del rsqrt_3
            buf332 = reinterpret_tensor(buf319, (64000, 128), (128, 1), 0); del buf319  # reuse
            # Topologically Sorted Source Nodes: [linear_1, gelu, convert_element_type_24, mul_415, clone_83, mul_418, mul_419, mul_420, exp_13, mul_421, mul_422, add_132, mul_423, view_202, permute_252, mm_55], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:257
            extern_kernels.mm(reinterpret_tensor(buf331, (64000, 512), (512, 1), 0), primals_16, out=buf332)
            del primals_16
            buf333 = empty_strided_cuda((512, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_1, gelu, convert_element_type_24, mul_415, clone_83, mul_418, mul_419, mul_420, exp_13, mul_421, mul_422, add_132, mul_423, view_202, permute_253, permute_255], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:258
            extern_kernels.mm(reinterpret_tensor(buf331, (512, 64000), (1, 512), 0), view_14, out=buf333)
            del view_14
            buf334 = buf280; del buf280  # reuse
            # Topologically Sorted Source Nodes: [linear_1, gelu, convert_element_type_24, mul_415, clone_83, mul_418, mul_419, mul_420, exp_13, mul_421, mul_422, add_132, mul_423, view_202, sum_138], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.sum]
            # [Provenance debug handles] triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_1:259
            stream0 = get_raw_stream(0)
            triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_1.run(buf331, buf334, 102400, 320, stream=stream0)
            del buf331
            buf335 = empty_strided_cuda((1, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_1, gelu, convert_element_type_24, mul_415, clone_83, mul_418, mul_419, mul_420, exp_13, mul_421, mul_422, add_132, mul_423, view_202, sum_138], Original ATen: [aten.view, aten.gelu, aten.native_dropout_backward, aten.gelu_backward, aten.sum]
            # [Provenance debug handles] triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_2:260
            stream0 = get_raw_stream(0)
            triton_red_fused_gelu_gelu_backward_native_dropout_backward_sum_view_2.run(buf334, buf335, 512, 200, stream=stream0)
            del buf334
            buf342 = buf318; del buf318  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [view_204, mul_425, mul_426, sum_139, mul_427, sum_140, mul_428, sub_111, sub_112, mul_429, mul_430, sum_141, sum_142, add_133], Original ATen: [aten.view, aten.native_layer_norm_backward, aten.add]
            workspace_25 = workspace_23; del workspace_23  # reuse
            stream0 = get_raw_stream(0)
            triton_per_fused_add_native_layer_norm_backward_view_13.run(buf342, buf332, primals_14, mul_9, div_32, workspace_25, 64000, 128, stream=stream0)
            buf339 = workspace_25[0 * 1000 * 128 : (0 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            buf341 = workspace_25[1 * 1000 * 128 : (1 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            del div_32
            del mul_9
            del primals_14
            buf349 = reinterpret_tensor(buf332, (128, 500, 128), (64000, 128, 1), 0); del buf332  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [convert_element_type_25, mul_431, clone_84, multi_head_attention_forward, transpose_1, sub_113, mul_433, mul_434, mul_435, sum_143, mul_436, sum_144, mul_437, sub_114, sub_115, div_33, mul_438, mul_439, sum_145, sum_146, permute_256, clone_85], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone]
            workspace_26 = workspace_25; del workspace_25  # reuse
            stream0 = get_raw_stream(0)
            triton_per_fused_clone_native_dropout_backward_native_layer_norm_backward_transpose_view_4.run(buf342, gt_1, primals_12, addmm_1, getitem_3, rsqrt_1, buf349, workspace_26, 64000, 128, stream=stream0)
            buf346 = workspace_26[0 * 1000 * 128 : (0 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            buf348 = workspace_26[1 * 1000 * 128 : (1 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            del addmm_1
            del getitem_3
            del gt_1
            del primals_12
            del rsqrt_1
            buf350 = buf311; del buf311  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_25, mul_431, clone_84, multi_head_attention_forward, transpose_1, sub_113, mul_433, mul_434, mul_435, mul_437, sub_114, sub_115, div_33, mul_438, permute_256, clone_85, view_205, permute_257, mm_57], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:261
            extern_kernels.mm(reinterpret_tensor(buf349, (64000, 128), (128, 1), 0), primals_8, out=buf350)
            del primals_8
            buf351 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_25, mul_431, clone_84, multi_head_attention_forward, transpose_1, sub_113, mul_433, mul_434, mul_435, mul_437, sub_114, sub_115, div_33, mul_438, permute_256, clone_85, view_205, permute_258, permute_260], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:262
            extern_kernels.mm(reinterpret_tensor(buf349, (128, 64000), (1, 128), 0), view_11, out=buf351)
            del view_11
            buf352 = buf322; del buf322  # reuse
            # Topologically Sorted Source Nodes: [convert_element_type_25, mul_431, clone_84, multi_head_attention_forward, transpose_1, sub_113, mul_433, mul_434, mul_435, mul_437, sub_114, sub_115, div_33, mul_438, permute_256, clone_85, view_205, sum_147], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_5:263
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_5.run(buf349, buf352, 64000, 128, stream=stream0)
            buf353 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_25, mul_431, clone_84, multi_head_attention_forward, transpose_1, sub_113, mul_433, mul_434, mul_435, mul_437, sub_114, sub_115, div_33, mul_438, permute_256, clone_85, view_205, sum_147], Original ATen: [aten.native_dropout_backward, aten.view, aten.transpose, aten.native_layer_norm_backward, aten.clone, aten._unsafe_view, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_6:264
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_6.run(buf352, buf353, 128, 500, stream=stream0)
            buf354 = reinterpret_tensor(buf349, (4000, 128, 16), (2048, 16, 1), 0); del buf349  # reuse
            # Topologically Sorted Source Nodes: [view_207, permute_261, bmm_31], Original ATen: [aten.view, aten.transpose, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:265
            extern_kernels.bmm(permute_262, reinterpret_tensor(buf350, (4000, 128, 16), (16, 64000, 1), 0), out=buf354)
            del permute_262
            buf355 = buf302; del buf302  # reuse
            # Topologically Sorted Source Nodes: [view_207, permute_261, bmm_32], Original ATen: [aten.view, aten.transpose, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:266
            extern_kernels.bmm(reinterpret_tensor(buf350, (4000, 128, 16), (16, 64000, 1), 0), permute_263, out=buf355)
            del permute_263
            buf356 = buf355; del buf355  # reuse
            buf358 = baddbmm; del baddbmm  # reuse
            buf361 = buf34; del buf34  # reuse
            # Topologically Sorted Source Nodes: [add_99, add_106, add_113, add_120, add_127, convert_element_type_26, mul_440, clone_86, multi_head_attention_forward, mul_442, sum_148, neg_6, fma_6, add_134], Original ATen: [aten.add, aten.native_dropout_backward, aten._softmax, aten._softmax_backward_data]
            # [Provenance debug handles] triton_per_fused__softmax__softmax_backward_data_add_native_dropout_backward_14:267
            stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data_add_native_dropout_backward_14.run(buf356, buf358, buf361, gt, amax, sum_1, buf88, buf142, buf196, buf250, buf304, 512000, 128, stream=stream0)
            del amax
            del buf142
            del buf196
            del buf250
            del buf304
            del buf356
            del buf88
            del gt
            del sum_1
            buf359 = reinterpret_tensor(buf350, (4000, 128, 16), (2048, 16, 1), 0); del buf350  # reuse
            # Topologically Sorted Source Nodes: [bmm_33], Original ATen: [aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:268
            extern_kernels.bmm(buf358, permute_264, out=buf359)
            del permute_264
            buf360 = reinterpret_tensor(buf300, (4000, 16, 128), (2048, 128, 1), 0); del buf300  # reuse
            # Topologically Sorted Source Nodes: [bmm_34], Original ATen: [aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:269
            extern_kernels.bmm(permute_265, buf358, out=buf360)
            del buf358
            del permute_265
            buf362 = buf307; del buf307  # reuse
            # Topologically Sorted Source Nodes: [full, permute_266, mul_443, permute_267, clone_87, view_208, permute_268, view_209, permute_269, clone_88, view_210, _generalized_scatter_18, _generalized_scatter_19, add_135, _generalized_scatter_20, add_136, unsqueeze_13, permute_270, squeeze_13, clone_89, view_211, sum_149], Original ATen: [aten.select_backward, aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.add, aten.unsqueeze, aten.squeeze, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_8:270
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_8.run(buf354, buf360, buf359, buf362, 122880, 200, stream=stream0)
            buf363 = empty_strided_cuda((1, 1, 384), (384, 384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full, permute_266, mul_443, permute_267, clone_87, view_208, permute_268, view_209, permute_269, clone_88, view_210, _generalized_scatter_18, _generalized_scatter_19, add_135, _generalized_scatter_20, add_136, unsqueeze_13, permute_270, squeeze_13, clone_89, view_211, sum_149], Original ATen: [aten.select_backward, aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.add, aten.unsqueeze, aten.squeeze, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_9:271
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_add_clone_mul_select_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf362, buf363, 384, 320, stream=stream0)
            del buf362
            buf364 = buf309; del buf309  # reuse
            # Topologically Sorted Source Nodes: [full, permute_266, mul_443, permute_267, clone_87, view_208, permute_268, view_209, permute_269, clone_88, view_210, _generalized_scatter_18, _generalized_scatter_19, add_135, _generalized_scatter_20, add_136, unsqueeze_13, permute_270, squeeze_13, clone_89], Original ATen: [aten.select_backward, aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.add, aten.unsqueeze, aten.squeeze]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_clone_mul_select_backward_squeeze_transpose_unsqueeze_view_10:272
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_mul_select_backward_squeeze_transpose_unsqueeze_view_10.run(buf354, buf360, buf359, buf364, 64000, 384, stream=stream0)
            del buf354
            del buf359
            buf365 = empty_strided_cuda((384, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full, permute_266, mul_443, permute_267, clone_87, view_208, permute_268, view_209, permute_269, clone_88, view_210, _generalized_scatter_18, _generalized_scatter_19, add_135, _generalized_scatter_20, add_136, unsqueeze_13, permute_270, squeeze_13, clone_89, view_211, view_213, permute_271, permute_274], Original ATen: [aten.select_backward, aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.add, aten.unsqueeze, aten.squeeze, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:273
            extern_kernels.mm(reinterpret_tensor(buf364, (384, 64000), (1, 384), 0), view_3, out=buf365)
            del view_3
            buf366 = reinterpret_tensor(buf360, (64000, 128), (128, 1), 0); del buf360  # reuse
            # Topologically Sorted Source Nodes: [full, permute_266, mul_443, permute_267, clone_87, view_208, permute_268, view_209, permute_269, clone_88, view_210, _generalized_scatter_18, _generalized_scatter_19, add_135, _generalized_scatter_20, add_136, unsqueeze_13, permute_270, squeeze_13, clone_89, view_211, view_213, multi_head_attention_forward, permute_273, mm_60], Original ATen: [aten.select_backward, aten.transpose, aten.mul, aten.clone, aten._unsafe_view, aten.view, aten.add, aten.unsqueeze, aten.squeeze, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:274
            extern_kernels.mm(reinterpret_tensor(buf364, (64000, 384), (384, 1), 0), primals_6, out=buf366)
            del buf364
            del primals_6
            buf367 = reinterpret_tensor(buf352, (500, 128, 1), (1, 500, 64000), 0); del buf352  # reuse
            # Topologically Sorted Source Nodes: [view_214, permute_275, mul_445, sum_150], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_backward_transpose_view_11:275
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_backward_transpose_view_11.run(buf366, primals_4, buf367, 64000, 128, stream=stream0)
            buf373 = buf342; del buf342  # reuse
            # Call mix order reduction kernel
            # Topologically Sorted Source Nodes: [view_214, permute_275, mul_445, mul_446, linear, layer_norm, mul_447, sum_151, mul_448, sub_117, sub_118, div_34, mul_449, mul_450, sum_152, sum_153, add_137], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward, aten.native_layer_norm, aten.add]
            workspace_27 = workspace_26; del workspace_26  # reuse
            stream0 = get_raw_stream(0)
            triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_15.run(buf373, buf366, primals_4, addmm, getitem_1, rsqrt, buf367, workspace_27, 64000, 128, stream=stream0)
            buf370 = workspace_27[0 * 1000 * 128 : (0 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            buf372 = workspace_27[1 * 1000 * 128 : (1 + 1) * 1000 * 128].view(1000, 128).sum(dim=0)
            del workspace_27
            del addmm
            del buf366
            del getitem_1
            del primals_4
            del rsqrt
            buf374 = reinterpret_tensor(workspace_24, (64000, 16), (16, 1), 0); del workspace_24  # reuse
            # Topologically Sorted Source Nodes: [view_214, permute_275, mul_445, mul_446, linear, layer_norm, mul_448, sub_117, sub_118, div_34, mul_449, add_137, view_215, permute_276, mm_61], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward, aten.native_layer_norm, aten.add, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:276
            extern_kernels.mm(reinterpret_tensor(buf373, (64000, 128), (128, 1), 0), primals_2, out=buf374)
            del primals_2
            buf375 = empty_strided_cuda((128, 16), (16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_214, permute_275, mul_445, mul_446, linear, layer_norm, mul_448, sub_117, sub_118, div_34, mul_449, add_137, view_215, permute_277, permute_279], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward, aten.native_layer_norm, aten.add, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:277
            extern_kernels.mm(reinterpret_tensor(buf373, (128, 64000), (1, 128), 0), view_1, out=buf375)
            del view_1
            buf376 = reinterpret_tensor(buf367, (1, 128, 500), (64000, 1, 128), 0); del buf367  # reuse
            # Topologically Sorted Source Nodes: [view_214, permute_275, mul_445, mul_446, linear, layer_norm, mul_448, sub_117, sub_118, div_34, mul_449, add_137, view_215, sum_154], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward, aten.native_layer_norm, aten.add, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_5:278
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_5.run(buf373, buf376, 64000, 128, stream=stream0)
            del buf373
            buf377 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view_214, permute_275, mul_445, mul_446, linear, layer_norm, mul_448, sub_117, sub_118, div_34, mul_449, add_137, view_215, sum_154], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm_backward, aten.native_layer_norm, aten.add, aten.sum]
            # [Provenance debug handles] triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_6:279
            stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clone_native_dropout_backward_native_layer_norm_backward_sum_transpose_view_6.run(buf376, buf377, 128, 500, stream=stream0)
            del buf376
        return (reinterpret_tensor(buf374, (500, 128, 1, 16), (2048, 16, 16, 1), 0), buf375, reinterpret_tensor(buf377, (128, ), (1, ), 0), buf370, buf372, buf365, reinterpret_tensor(buf363, (384, ), (1, ), 0), buf351, reinterpret_tensor(buf353, (128, ), (1, ), 0), None, buf361, buf346, buf348, buf339, buf341, buf333, reinterpret_tensor(buf335, (512, ), (1, ), 0), buf328, buf330, buf321, reinterpret_tensor(buf323, (128, ), (1, ), 0), buf315, buf317, buf310, reinterpret_tensor(buf308, (384, ), (1, ), 0), buf297, reinterpret_tensor(buf299, (128, ), (1, ), 0), buf292, buf294, buf285, buf287, buf279, reinterpret_tensor(buf281, (512, ), (1, ), 0), buf274, buf276, buf267, reinterpret_tensor(buf269, (128, ), (1, ), 0), buf261, buf263, buf256, reinterpret_tensor(buf254, (384, ), (1, ), 0), buf243, reinterpret_tensor(buf245, (128, ), (1, ), 0), buf238, buf240, buf231, buf233, buf225, reinterpret_tensor(buf227, (512, ), (1, ), 0), buf220, buf222, buf213, reinterpret_tensor(buf215, (128, ), (1, ), 0), buf207, buf209, buf202, reinterpret_tensor(buf200, (384, ), (1, ), 0), buf189, reinterpret_tensor(buf191, (128, ), (1, ), 0), buf184, buf186, buf177, buf179, buf171, reinterpret_tensor(buf173, (512, ), (1, ), 0), buf166, buf168, buf159, reinterpret_tensor(buf161, (128, ), (1, ), 0), buf153, buf155, buf148, reinterpret_tensor(buf146, (384, ), (1, ), 0), buf135, reinterpret_tensor(buf137, (128, ), (1, ), 0), buf130, buf132, buf123, buf125, buf117, reinterpret_tensor(buf119, (512, ), (1, ), 0), buf112, buf114, buf105, reinterpret_tensor(buf107, (128, ), (1, ), 0), buf99, buf101, buf94, reinterpret_tensor(buf92, (384, ), (1, ), 0), buf81, reinterpret_tensor(buf83, (128, ), (1, ), 0), buf76, buf78, buf69, buf71, buf63, reinterpret_tensor(buf65, (512, ), (1, ), 0), buf58, buf60, buf51, reinterpret_tensor(buf53, (128, ), (1, ), 0), buf45, buf47, buf40, reinterpret_tensor(buf38, (384, ), (1, ), 0), buf27, reinterpret_tensor(buf29, (128, ), (1, ), 0), buf22, buf24, buf15, buf17, buf9, reinterpret_tensor(buf11, (512, ), (1, ), 0), buf4, buf6, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_2 = rand_strided((128, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_1 = rand_strided((64000, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    addmm = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_3 = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    baddbmm = rand_strided((4000, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax = rand_strided((4000, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_1 = rand_strided((4000, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    gt = rand_strided((4000, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_11 = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_1 = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_1 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_1 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bool)
    mul_9 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    view_14 = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_2 = rand_strided((64000, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    gt_2 = rand_strided((500, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_7 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_3 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_16 = rand_strided((64000, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    gt_3 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bool)
    mul_20 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    view_18 = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    baddbmm_1 = rand_strided((4000, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax_1 = rand_strided((4000, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_2 = rand_strided((4000, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_4 = rand_strided((4000, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_26 = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_4 = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    getitem_11 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_5 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_5 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bool)
    mul_29 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    view_29 = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_5 = rand_strided((64000, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    gt_6 = rand_strided((500, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_15 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_7 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_31 = rand_strided((64000, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    gt_7 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bool)
    mul_40 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    view_33 = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    baddbmm_2 = rand_strided((4000, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax_2 = rand_strided((4000, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_3 = rand_strided((4000, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_8 = rand_strided((4000, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_41 = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_7 = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    getitem_19 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_9 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_9 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bool)
    mul_49 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    view_44 = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_8 = rand_strided((64000, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    gt_10 = rand_strided((500, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_23 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_11 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_46 = rand_strided((64000, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    gt_11 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bool)
    mul_60 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    view_48 = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    baddbmm_3 = rand_strided((4000, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax_3 = rand_strided((4000, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_4 = rand_strided((4000, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_12 = rand_strided((4000, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_56 = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    getitem_27 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_13 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_13 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bool)
    mul_69 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    view_59 = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_11 = rand_strided((64000, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    gt_14 = rand_strided((500, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_31 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_15 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_61 = rand_strided((64000, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    gt_15 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bool)
    mul_80 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    view_63 = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    baddbmm_4 = rand_strided((4000, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax_4 = rand_strided((4000, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_5 = rand_strided((4000, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_16 = rand_strided((4000, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_71 = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_13 = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    getitem_35 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_17 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_17 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bool)
    mul_89 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    view_74 = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_14 = rand_strided((64000, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    gt_18 = rand_strided((500, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_39 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_19 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_76 = rand_strided((64000, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    gt_19 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bool)
    mul_100 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    view_78 = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    baddbmm_5 = rand_strided((4000, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax_5 = rand_strided((4000, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_6 = rand_strided((4000, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_20 = rand_strided((4000, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_86 = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_16 = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    getitem_43 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_21 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_21 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bool)
    mul_109 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    view_89 = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_17 = rand_strided((64000, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    gt_22 = rand_strided((500, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_47 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_23 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_91 = rand_strided((64000, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    gt_23 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bool)
    mul_120 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    view_93 = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    baddbmm_6 = rand_strided((4000, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    amax_6 = rand_strided((4000, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_7 = rand_strided((4000, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_24 = rand_strided((4000, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bool)
    view_101 = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_19 = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    getitem_51 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_25 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    gt_25 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bool)
    add_84 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_53 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_26 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_104 = rand_strided((64000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_20 = rand_strided((64000, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    gt_26 = rand_strided((500, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bool)
    getitem_55 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_27 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_94 = rand_strided((4000, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_95 = rand_strided((4000, 16, 128), (16, 1, 64000), device='cuda:0', dtype=torch.float32)
    permute_96 = rand_strided((4000, 128, 16), (16, 64000, 1), device='cuda:0', dtype=torch.float32)
    permute_97 = rand_strided((4000, 16, 128), (16, 1, 64000), device='cuda:0', dtype=torch.float32)
    div_10 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    div_12 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_122 = rand_strided((4000, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_123 = rand_strided((4000, 16, 128), (16, 1, 64000), device='cuda:0', dtype=torch.float32)
    permute_124 = rand_strided((4000, 128, 16), (16, 64000, 1), device='cuda:0', dtype=torch.float32)
    permute_125 = rand_strided((4000, 16, 128), (16, 1, 64000), device='cuda:0', dtype=torch.float32)
    div_14 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    div_16 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_150 = rand_strided((4000, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_151 = rand_strided((4000, 16, 128), (16, 1, 64000), device='cuda:0', dtype=torch.float32)
    permute_152 = rand_strided((4000, 128, 16), (16, 64000, 1), device='cuda:0', dtype=torch.float32)
    permute_153 = rand_strided((4000, 16, 128), (16, 1, 64000), device='cuda:0', dtype=torch.float32)
    div_18 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    div_20 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_178 = rand_strided((4000, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_179 = rand_strided((4000, 16, 128), (16, 1, 64000), device='cuda:0', dtype=torch.float32)
    permute_180 = rand_strided((4000, 128, 16), (16, 64000, 1), device='cuda:0', dtype=torch.float32)
    permute_181 = rand_strided((4000, 16, 128), (16, 1, 64000), device='cuda:0', dtype=torch.float32)
    div_22 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    div_24 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_206 = rand_strided((4000, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_207 = rand_strided((4000, 16, 128), (16, 1, 64000), device='cuda:0', dtype=torch.float32)
    permute_208 = rand_strided((4000, 128, 16), (16, 64000, 1), device='cuda:0', dtype=torch.float32)
    permute_209 = rand_strided((4000, 16, 128), (16, 1, 64000), device='cuda:0', dtype=torch.float32)
    div_26 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    div_28 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_234 = rand_strided((4000, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_235 = rand_strided((4000, 16, 128), (16, 1, 64000), device='cuda:0', dtype=torch.float32)
    permute_236 = rand_strided((4000, 128, 16), (16, 64000, 1), device='cuda:0', dtype=torch.float32)
    permute_237 = rand_strided((4000, 16, 128), (16, 1, 64000), device='cuda:0', dtype=torch.float32)
    div_30 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    div_32 = rand_strided((500, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_262 = rand_strided((4000, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.float32)
    permute_263 = rand_strided((4000, 16, 128), (16, 1, 64000), device='cuda:0', dtype=torch.float32)
    permute_264 = rand_strided((4000, 128, 16), (16, 64000, 1), device='cuda:0', dtype=torch.float32)
    permute_265 = rand_strided((4000, 16, 128), (16, 1, 64000), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((500, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_2, primals_4, primals_6, primals_8, primals_12, primals_14, primals_16, primals_18, primals_20, primals_22, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_38, primals_40, primals_42, primals_44, primals_46, primals_48, primals_50, primals_52, primals_54, primals_56, primals_58, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_106, primals_108, primals_110, primals_112, primals_114, view_1, addmm, getitem_1, rsqrt, view_3, baddbmm, amax, sum_1, gt, view_11, addmm_1, getitem_3, rsqrt_1, gt_1, mul_9, view_14, addmm_2, gt_2, getitem_7, rsqrt_3, view_16, gt_3, mul_20, view_18, baddbmm_1, amax_1, sum_2, gt_4, view_26, addmm_4, getitem_11, rsqrt_5, gt_5, mul_29, view_29, addmm_5, gt_6, getitem_15, rsqrt_7, view_31, gt_7, mul_40, view_33, baddbmm_2, amax_2, sum_3, gt_8, view_41, addmm_7, getitem_19, rsqrt_9, gt_9, mul_49, view_44, addmm_8, gt_10, getitem_23, rsqrt_11, view_46, gt_11, mul_60, view_48, baddbmm_3, amax_3, sum_4, gt_12, view_56, addmm_10, getitem_27, rsqrt_13, gt_13, mul_69, view_59, addmm_11, gt_14, getitem_31, rsqrt_15, view_61, gt_15, mul_80, view_63, baddbmm_4, amax_4, sum_5, gt_16, view_71, addmm_13, getitem_35, rsqrt_17, gt_17, mul_89, view_74, addmm_14, gt_18, getitem_39, rsqrt_19, view_76, gt_19, mul_100, view_78, baddbmm_5, amax_5, sum_6, gt_20, view_86, addmm_16, getitem_43, rsqrt_21, gt_21, mul_109, view_89, addmm_17, gt_22, getitem_47, rsqrt_23, view_91, gt_23, mul_120, view_93, baddbmm_6, amax_6, sum_7, gt_24, view_101, addmm_19, getitem_51, rsqrt_25, gt_25, add_84, getitem_53, rsqrt_26, view_104, addmm_20, gt_26, getitem_55, rsqrt_27, permute_94, permute_95, permute_96, permute_97, div_10, div_12, permute_122, permute_123, permute_124, permute_125, div_14, div_16, permute_150, permute_151, permute_152, permute_153, div_18, div_20, permute_178, permute_179, permute_180, permute_181, div_22, div_24, permute_206, permute_207, permute_208, permute_209, div_26, div_28, permute_234, permute_235, permute_236, permute_237, div_30, div_32, permute_262, permute_263, permute_264, permute_265, tangents_1, tangents_2])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
