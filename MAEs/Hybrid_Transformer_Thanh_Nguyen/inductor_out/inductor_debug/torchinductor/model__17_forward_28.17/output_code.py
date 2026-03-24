# AOT ID: ['17_forward']
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


# kernel path: /traces/inductor_cache/ai/caiscboapn5nibq5x6sxa4uolj35bxynrx4mkoxa4ujv2clkukb5.py
# Topologically Sorted Source Nodes: [, linear, iadd, cat_1, layer_norm, transpose_1, multi_head_attention_forward], Original ATen: [aten.addmm, aten.view, aten.add, aten.cat, aten.native_layer_norm, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#    => add_tensor_2
#   cat_1 => cat_1
#   iadd => add
#   layer_norm => add_1, add_2, mul, mul_1, rsqrt, sub, var_mean
#   linear => view_1
#   multi_head_attention_forward => clone_1
#   transpose_1 => permute_2
# Graph fragment:
#   %primals_3 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_3]
#   %mm_default_2 : Tensor "f32[500, 128][128, 1]cuda:0" = PlaceHolder[target=mm_default_2]
#   %primals_4 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0" = PlaceHolder[target=primals_4]
#   %primals_6 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=primals_6]
#   %cat_1 : Tensor "f32[500, 129, 128][16512, 128, 1]cuda:0" = PlaceHolder[target=cat_1]
#   %buf3 : Tensor "f32[500, 129, 1][129, 1, 64512]cuda:0" = PlaceHolder[target=buf3]
#   %getitem_1 : Tensor "f32[500, 129, 1][129, 1, 1]cuda:0" = PlaceHolder[target=getitem_1]
#   %rsqrt : Tensor "f32[500, 129, 1][129, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %primals_7 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_7]
#   %primals_8 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_8]
#   %add_tensor_2 : Tensor "f32[500, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_3, %mm_default_2), kwargs = {})
#   %view_1 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_2, [500, 1, 128]), kwargs = {})
#   %add : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %primals_4), kwargs = {})
#   %cat_1 : Tensor "f32[500, 129, 128][16512, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%add, %primals_6], 1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%cat_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_1 : Tensor "f32[500, 129, 1][129, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[500, 129, 1][129, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %sub : Tensor "f32[500, 129, 128][16512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_1, %getitem_1), kwargs = {})
#   %mul : Tensor "f32[500, 129, 128][16512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[500, 129, 128][16512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %primals_7), kwargs = {})
#   %add_2 : Tensor "f32[500, 129, 128][16512, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %primals_8), kwargs = {})
#   %permute_2 : Tensor "f32[129, 500, 128][128, 16512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_2, [1, 0, 2]), kwargs = {})
#   %clone_1 : Tensor "f32[129, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_2,), kwargs = {memory_format: torch.contiguous_format})
#   return %cat_1,%getitem_1,%buf3,%rsqrt,%clone_1
triton_per_fused_add_addmm_cat_clone_native_layer_norm_transpose_view_0 = async_compile.triton('triton_per_fused_add_addmm_cat_clone_native_layer_norm_transpose_view_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_cat_clone_native_layer_norm_transpose_view_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 6, 'num_store': 4, 'num_reduction': 4, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1032000, 'r0_': 165377536}}
)
@triton.jit
def triton_per_fused_add_addmm_cat_clone_native_layer_norm_transpose_view_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    x0 = (xindex % 129)
    r0_2 = r0_index
    x1 = xindex // 129
    x3 = xindex
    tmp40 = tl.load(in_ptr4 + (r0_2), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr5 + (r0_2), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(r0_2, [XBLOCK, R0_BLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (r0_2 + 128*x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (r0_2 + 128*x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1, 1], 129, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr3 + (r0_2 + 128*((-1) + x0) + 16384*x1), tmp12 & xmask, other=0.0)
    tmp16 = tl.where(tmp4, tmp11, tmp15)
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp22 = tl.where(xmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None].to(tl.float32)
    tmp24 = tl.full([1, 1], 128, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = (tmp23 / tmp25)
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, R0_BLOCK])
    tmp31 = tl.where(xmask, tmp29, 0)
    tmp32 = tl.sum(tmp31, 1)[:, None].to(tl.float32)
    tmp33 = 128.0
    tmp34 = (tmp32 / tmp33)
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = libdevice.rsqrt(tmp36)
    tmp38 = tmp16 - tmp26
    tmp39 = tmp38 * tmp37
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tl.store(out_ptr0 + (r0_2 + 128*x3), tmp16, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp37, xmask)
    tl.store(out_ptr2 + (r0_2 + 128*x1 + 64000*x0), tmp43, xmask)
    tl.store(out_ptr1 + (x3), tmp26, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/xx/cxxp5gdqd2zrqnyxyi5fne2r3zhdz43akqigg22bdxy72xmqq2zy.py
# Topologically Sorted Source Nodes: [, linear, iadd], Original ATen: [aten.addmm, aten.view, aten.add]
# Source node to ATen node mapping:
#    => add_tensor_2
#   iadd => add
#   linear => view_1
# Graph fragment:
#   %primals_3 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_3]
#   %mm_default_2 : Tensor "f32[500, 128][128, 1]cuda:0" = PlaceHolder[target=mm_default_2]
#   %primals_4 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0" = PlaceHolder[target=primals_4]
#   %add_tensor_2 : Tensor "f32[500, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_3, %mm_default_2), kwargs = {})
#   %view_1 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_2, [500, 1, 128]), kwargs = {})
#   %add : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %primals_4), kwargs = {})
#   return %add
triton_poi_fused_add_addmm_view_1 = async_compile.triton('triton_poi_fused_add_addmm_view_1', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_view_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1024512}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_view_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/hi/chil62e36hkq7thomxv42qh2t26i35znmxmnhconecq6yvojxw4e.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.split_with_sizes, aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => add_3, clone_2, permute_7, split_with_sizes_1, squeeze, unsqueeze, view_6, view_7
# Graph fragment:
#   %mm : Tensor "f32[64500, 256][256, 1]cuda:0" = PlaceHolder[target=mm]
#   %primals_10 : Tensor "f32[384][1]cuda:0" = PlaceHolder[target=primals_10]
#   %split_with_sizes_1 : [num_users=2] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%primals_10, [128, 256]), kwargs = {})
#   %view_6 : Tensor "f32[129, 500, 256][128000, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [129, 500, 256]), kwargs = {})
#   %add_3 : Tensor "f32[129, 500, 256][128000, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_6, %getitem_5), kwargs = {})
#   %view_7 : Tensor "f32[129, 500, 2, 128][128000, 256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_3, [129, 500, 2, 128]), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 129, 500, 2, 128][16512000, 128000, 256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_7, 0), kwargs = {})
#   %permute_7 : Tensor "f32[2, 129, 500, 1, 128][128, 128000, 256, 16512000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze : Tensor "f32[2, 129, 500, 128][128, 128000, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_7, -2), kwargs = {})
#   %clone_2 : Tensor "f32[2, 129, 500, 128][8256000, 64000, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_2
triton_poi_fused__unsafe_view_add_clone_split_with_sizes_squeeze_transpose_unsqueeze_view_2 = async_compile.triton('triton_poi_fused__unsafe_view_add_clone_split_with_sizes_squeeze_transpose_unsqueeze_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_clone_split_with_sizes_squeeze_transpose_unsqueeze_view_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 198145024}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_clone_split_with_sizes_squeeze_transpose_unsqueeze_view_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16512000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 64500)
    x2 = xindex // 8256000
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x2 + 256*x1), xmask)
    tmp1 = tl.load(in_ptr1 + (128 + x0 + 128*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/ie/ciehrpi5vezzzgcorpaksbqppewk7hxsjep2std6dgendokerzwn.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward, ], Original ATen: [aten.split_with_sizes, aten.addmm, aten.view, aten.transpose, aten.mul]
# Source node to ATen node mapping:
#    => add_tensor_1
#   multi_head_attention_forward => mul_2, permute_8, split_with_sizes_1, view_4, view_8
# Graph fragment:
#   %primals_10 : Tensor "f32[384][1]cuda:0" = PlaceHolder[target=primals_10]
#   %mm_default_1 : Tensor "f32[500, 128][128, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %split_with_sizes_1 : [num_users=2] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%primals_10, [128, 256]), kwargs = {})
#   %add_tensor_1 : Tensor "f32[500, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, %mm_default_1), kwargs = {})
#   %view_4 : Tensor "f32[1, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [1, 500, 128]), kwargs = {})
#   %view_8 : Tensor "f32[1, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_4, [1, 4000, 16]), kwargs = {})
#   %permute_8 : Tensor "f32[4000, 1, 16][16, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_8, [1, 0, 2]), kwargs = {})
#   %mul_2 : Tensor "f32[4000, 1, 16][16, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_8, 0.25), kwargs = {})
#   return %mul_2
triton_poi_fused_addmm_mul_split_with_sizes_transpose_view_3 = async_compile.triton('triton_poi_fused_addmm_mul_split_with_sizes_transpose_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_mul_split_with_sizes_transpose_view_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 769536}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_mul_split_with_sizes_transpose_view_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + ((x0 % 128)), xmask)
    tmp1 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.25
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/tr/ctrr4ku6yrmh7qf6yr3xaz7x5f56ch6egfnu3dhv4p5medn2qut4.py
# Topologically Sorted Source Nodes: [zeros_like, cat, multi_head_attention_forward, , permute_38], Original ATen: [aten.zeros_like, aten.cat, aten.split_with_sizes, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.expand, aten.mul, aten.baddbmm]
# Source node to ATen node mapping:
#    => add_tensor_1
#   cat => cat
#   multi_head_attention_forward => add_3, baddbmm, clone_2, clone_3, expand, mul_2, permute_11, permute_7, permute_8, permute_9, select, split_with_sizes_1, squeeze, unsqueeze, view_11, view_12, view_4, view_6, view_7, view_8, view_9
#   permute_38 => permute_38
#   zeros_like => full_default
# Graph fragment:
#   %clone_2 : Tensor "f32[2, 129, 500, 128][8256000, 64000, 128, 1]cuda:0" = PlaceHolder[target=clone_2]
#   %full_default : Tensor "f32[500, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([500, 1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %cat : Tensor "f32[500, 129][129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%full_default, %primals_5], 1), kwargs = {})
#   %split_with_sizes_1 : [num_users=2] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%primals_10, [128, 256]), kwargs = {})
#   %add_tensor_1 : Tensor "f32[500, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, %mm_default_1), kwargs = {})
#   %view_4 : Tensor "f32[1, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [1, 500, 128]), kwargs = {})
#   %view_6 : Tensor "f32[129, 500, 256][128000, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [129, 500, 256]), kwargs = {})
#   %add_3 : Tensor "f32[129, 500, 256][128000, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_6, %getitem_5), kwargs = {})
#   %view_7 : Tensor "f32[129, 500, 2, 128][128000, 256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_3, [129, 500, 2, 128]), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 129, 500, 2, 128][16512000, 128000, 256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_7, 0), kwargs = {})
#   %permute_7 : Tensor "f32[2, 129, 500, 1, 128][128, 128000, 256, 16512000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze : Tensor "f32[2, 129, 500, 128][128, 128000, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_7, -2), kwargs = {})
#   %clone_2 : Tensor "f32[2, 129, 500, 128][8256000, 64000, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
#   %select : Tensor "f32[129, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_2, 0, 0), kwargs = {})
#   %view_8 : Tensor "f32[1, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_4, [1, 4000, 16]), kwargs = {})
#   %permute_8 : Tensor "f32[4000, 1, 16][16, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_8, [1, 0, 2]), kwargs = {})
#   %view_9 : Tensor "f32[129, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select, [129, 4000, 16]), kwargs = {})
#   %permute_9 : Tensor "f32[4000, 129, 16][16, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_9, [1, 0, 2]), kwargs = {})
#   %view_11 : Tensor "f32[500, 1, 1, 129][129, 129, 129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%cat, [500, 1, 1, 129]), kwargs = {})
#   %expand : Tensor "f32[500, 8, 1, 129][129, 0, 129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%view_11, [-1, 8, -1, -1]), kwargs = {})
#   %clone_3 : Tensor "f32[500, 8, 1, 129][1032, 129, 129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
#   %view_12 : Tensor "f32[4000, 1, 129][129, 129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_3, [4000, 1, 129]), kwargs = {})
#   %mul_2 : Tensor "f32[4000, 1, 16][16, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_8, 0.25), kwargs = {})
#   %permute_11 : Tensor "f32[4000, 16, 129][16, 1, 64000]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%permute_9, [0, 2, 1]), kwargs = {})
#   %baddbmm : Tensor "f32[4000, 1, 129][129, 129, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.baddbmm.default](args = (%view_12, %mul_2, %permute_11), kwargs = {})
#   %permute_38 : Tensor "f32[4000, 129, 16][16, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_11, [0, 2, 1]), kwargs = {})
#   return %buf12,%permute_38
triton_poi_fused__unsafe_view_add_addmm_baddbmm_cat_clone_expand_mul_select_split_with_sizes_squeeze_transpose_unsqueeze_view_zeros_like_4 = async_compile.triton('triton_poi_fused__unsafe_view_add_addmm_baddbmm_cat_clone_expand_mul_select_split_with_sizes_squeeze_transpose_unsqueeze_view_zeros_like_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_addmm_baddbmm_cat_clone_expand_mul_select_split_with_sizes_squeeze_transpose_unsqueeze_view_zeros_like_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 165120000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_baddbmm_cat_clone_expand_mul_select_split_with_sizes_squeeze_transpose_unsqueeze_view_zeros_like_4(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8256000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
    tl.store(out_ptr1 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/jh/cjhstzzkyo4piur7xgj7zmgjoihk2d7dfldliavlzde2dxgbjdmf.py
# Topologically Sorted Source Nodes: [zeros_like, cat, multi_head_attention_forward], Original ATen: [aten.zeros_like, aten.cat, aten.view, aten.expand, aten.clone]
# Source node to ATen node mapping:
#   cat => cat
#   multi_head_attention_forward => clone_3, expand, view_11
#   zeros_like => full_default
# Graph fragment:
#   %primals_5 : Tensor "f32[500, 128][128, 1]cuda:0" = PlaceHolder[target=primals_5]
#   %full_default : Tensor "f32[500, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([500, 1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %cat : Tensor "f32[500, 129][129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%full_default, %primals_5], 1), kwargs = {})
#   %view_11 : Tensor "f32[500, 1, 1, 129][129, 129, 129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%cat, [500, 1, 1, 129]), kwargs = {})
#   %expand : Tensor "f32[500, 8, 1, 129][129, 0, 129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%view_11, [-1, 8, -1, -1]), kwargs = {})
#   %clone_3 : Tensor "f32[500, 8, 1, 129][1032, 129, 129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_3
triton_poi_fused_cat_clone_expand_view_zeros_like_5 = async_compile.triton('triton_poi_fused_cat_clone_expand_view_zeros_like_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_clone_expand_view_zeros_like_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4384000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_clone_expand_view_zeros_like_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 516000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 129)
    x2 = xindex // 1032
    x3 = (xindex % 1032)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 0.0
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 129, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr0 + (128*x2 + ((-1) + x0)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.where(tmp4, tmp7, tmp11)
    tl.store(out_ptr0 + (x3 + 1056*x2), tmp12, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/ez/ceztnzcvf5d7xr3r7owjqabfoyskwzigtizxo3mjzl5egzz5mgtv.py
# Topologically Sorted Source Nodes: [zeros_like, cat, multi_head_attention_forward, ], Original ATen: [aten.zeros_like, aten.cat, aten.split_with_sizes, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.expand, aten.mul, aten.baddbmm]
# Source node to ATen node mapping:
#    => add_tensor_1
#   cat => cat
#   multi_head_attention_forward => add_3, baddbmm, clone_2, clone_3, expand, mul_2, permute_11, permute_7, permute_8, permute_9, select, split_with_sizes_1, squeeze, unsqueeze, view_11, view_12, view_4, view_6, view_7, view_8, view_9
#   zeros_like => full_default
# Graph fragment:
#   %clone_3 : Tensor "f32[500, 8, 1, 129][1056, 129, 129, 1]cuda:0" = PlaceHolder[target=clone_3]
#   %full_default : Tensor "f32[500, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([500, 1], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %cat : Tensor "f32[500, 129][129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%full_default, %primals_5], 1), kwargs = {})
#   %split_with_sizes_1 : [num_users=2] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%primals_10, [128, 256]), kwargs = {})
#   %add_tensor_1 : Tensor "f32[500, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, %mm_default_1), kwargs = {})
#   %view_4 : Tensor "f32[1, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [1, 500, 128]), kwargs = {})
#   %view_6 : Tensor "f32[129, 500, 256][128000, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [129, 500, 256]), kwargs = {})
#   %add_3 : Tensor "f32[129, 500, 256][128000, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_6, %getitem_5), kwargs = {})
#   %view_7 : Tensor "f32[129, 500, 2, 128][128000, 256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_3, [129, 500, 2, 128]), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 129, 500, 2, 128][16512000, 128000, 256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_7, 0), kwargs = {})
#   %permute_7 : Tensor "f32[2, 129, 500, 1, 128][128, 128000, 256, 16512000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze : Tensor "f32[2, 129, 500, 128][128, 128000, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_7, -2), kwargs = {})
#   %clone_2 : Tensor "f32[2, 129, 500, 128][8256000, 64000, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
#   %select : Tensor "f32[129, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_2, 0, 0), kwargs = {})
#   %view_8 : Tensor "f32[1, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_4, [1, 4000, 16]), kwargs = {})
#   %permute_8 : Tensor "f32[4000, 1, 16][16, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_8, [1, 0, 2]), kwargs = {})
#   %view_9 : Tensor "f32[129, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select, [129, 4000, 16]), kwargs = {})
#   %permute_9 : Tensor "f32[4000, 129, 16][16, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_9, [1, 0, 2]), kwargs = {})
#   %view_11 : Tensor "f32[500, 1, 1, 129][129, 129, 129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%cat, [500, 1, 1, 129]), kwargs = {})
#   %expand : Tensor "f32[500, 8, 1, 129][129, 0, 129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%view_11, [-1, 8, -1, -1]), kwargs = {})
#   %clone_3 : Tensor "f32[500, 8, 1, 129][1032, 129, 129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
#   %view_12 : Tensor "f32[4000, 1, 129][129, 129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_3, [4000, 1, 129]), kwargs = {})
#   %mul_2 : Tensor "f32[4000, 1, 16][16, 16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_8, 0.25), kwargs = {})
#   %permute_11 : Tensor "f32[4000, 16, 129][16, 1, 64000]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%permute_9, [0, 2, 1]), kwargs = {})
#   %baddbmm : Tensor "f32[4000, 1, 129][129, 129, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.baddbmm.default](args = (%view_12, %mul_2, %permute_11), kwargs = {})
#   return %buf14
triton_poi_fused__unsafe_view_add_addmm_baddbmm_cat_clone_expand_mul_select_split_with_sizes_squeeze_transpose_unsqueeze_view_zeros_like_6 = async_compile.triton('triton_poi_fused__unsafe_view_add_addmm_baddbmm_cat_clone_expand_mul_select_split_with_sizes_squeeze_transpose_unsqueeze_view_zeros_like_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_addmm_baddbmm_cat_clone_expand_mul_select_split_with_sizes_squeeze_transpose_unsqueeze_view_zeros_like_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 6192000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_baddbmm_cat_clone_expand_mul_select_split_with_sizes_squeeze_transpose_unsqueeze_view_zeros_like_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 516000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 129)
    x1 = xindex // 129
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 129*((x1 % 8)) + 1056*(x1 // 8)), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/uk/cukwimjdsxhp3a7x4vxljtytiuexyx3vmb44xtcd7wvduthns4rt.py
# Topologically Sorted Source Nodes: [, multi_head_attention_forward], Original ATen: [prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax]
# Source node to ATen node mapping:
#    => exp_default, prepare_softmax_online_default, sub_tensor
#   multi_head_attention_forward => div
# Graph fragment:
#   %baddbmm : Tensor "f32[4000, 1, 129][129, 129, 1]cuda:0" = PlaceHolder[target=baddbmm]
#   %getitem_14 : Tensor "f32[4000, 1, 1][1, 4000, 4000]cuda:0" = PlaceHolder[target=getitem_14]
#   %getitem_15 : Tensor "f32[4000, 1, 1][1, 4000, 4000]cuda:0" = PlaceHolder[target=getitem_15]
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%baddbmm, -1), kwargs = {})
#   %sub_tensor : Tensor "f32[4000, 1, 129][129, 129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%baddbmm, %getitem_14), kwargs = {})
#   %exp_default : Tensor "f32[4000, 1, 129][129, 129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div : Tensor "f32[4000, 1, 129][129, 129, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_15), kwargs = {})
#   return %getitem_14,%getitem_15,%div
triton_per_fused__softmax_exp_prepare_softmax_online_sub_7 = async_compile.triton('triton_per_fused__softmax_exp_prepare_softmax_online_sub_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_exp_prepare_softmax_online_sub_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 4, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 6192000}}
)
@triton.jit
def triton_per_fused__softmax_exp_prepare_softmax_online_sub_7(in_out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(r0_mask & xmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None].to(tl.float32)
    tmp7 = tmp1 - tmp6
    tmp8 = libdevice.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(r0_mask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
    tmp13 = tmp0 - tmp6
    tmp14 = libdevice.exp(tmp13)
    tmp15 = (tmp14 / tmp12)
    tl.store(in_out_ptr0 + (r0_1 + 129*x0), tmp15, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/63/c63adkl6wd6nzgihdqfyutmqt3llqveuhedb5o7nnnugvry5dfml.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward, permute_37], Original ATen: [aten.split_with_sizes, aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.bmm]
# Source node to ATen node mapping:
#   multi_head_attention_forward => add_3, bmm, clone_2, permute_10, permute_7, select_1, split_with_sizes_1, squeeze, unsqueeze, view_10, view_6, view_7
#   permute_37 => permute_37
# Graph fragment:
#   %clone_2 : Tensor "f32[2, 129, 500, 128][8256000, 64000, 128, 1]cuda:0" = PlaceHolder[target=clone_2]
#   %split_with_sizes_1 : [num_users=2] = call_function[target=torch.ops.aten.split_with_sizes.default](args = (%primals_10, [128, 256]), kwargs = {})
#   %view_6 : Tensor "f32[129, 500, 256][128000, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [129, 500, 256]), kwargs = {})
#   %add_3 : Tensor "f32[129, 500, 256][128000, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_6, %getitem_5), kwargs = {})
#   %view_7 : Tensor "f32[129, 500, 2, 128][128000, 256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_3, [129, 500, 2, 128]), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 129, 500, 2, 128][16512000, 128000, 256, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_7, 0), kwargs = {})
#   %permute_7 : Tensor "f32[2, 129, 500, 1, 128][128, 128000, 256, 16512000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze : Tensor "f32[2, 129, 500, 128][128, 128000, 256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_7, -2), kwargs = {})
#   %clone_2 : Tensor "f32[2, 129, 500, 128][8256000, 64000, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
#   %select_1 : Tensor "f32[129, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_2, 0, 1), kwargs = {})
#   %view_10 : Tensor "f32[129, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_1, [129, 4000, 16]), kwargs = {})
#   %permute_10 : Tensor "f32[4000, 129, 16][16, 64000, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_10, [1, 0, 2]), kwargs = {})
#   %bmm : Tensor "f32[4000, 1, 16][16, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%div, %permute_10), kwargs = {})
#   %permute_37 : Tensor "f32[4000, 16, 129][16, 1, 64000]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_10, [0, 2, 1]), kwargs = {})
#   return %buf19,%permute_37
triton_poi_fused__unsafe_view_add_bmm_clone_select_split_with_sizes_squeeze_transpose_unsqueeze_view_8 = async_compile.triton('triton_poi_fused__unsafe_view_add_bmm_clone_select_split_with_sizes_squeeze_transpose_unsqueeze_view_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_bmm_clone_select_split_with_sizes_squeeze_transpose_unsqueeze_view_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 165120000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_bmm_clone_select_split_with_sizes_squeeze_transpose_unsqueeze_view_8(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8256000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (8256000 + x2), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
    tl.store(out_ptr1 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/4s/c4shcmiqvxu6o4qnbpu7xas3druk2b2ost3paw6z6q7jcwslodj3.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward, transpose_2, layer_norm_1, iadd_1, layer_norm_2, div_3], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm, aten.add, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   div_3 => div_3
#   iadd_1 => add_6
#   layer_norm_1 => add_4, add_5, mul_3, mul_4, rsqrt_1, sub_2, var_mean_1
#   layer_norm_2 => add_7, add_8, mul_5, mul_6, rsqrt_2, sub_3, var_mean_2
#   multi_head_attention_forward => view_14
#   transpose_2 => permute_14
# Graph fragment:
#   %addmm_2 : Tensor "f32[500, 128][128, 1]cuda:0" = PlaceHolder[target=addmm_2]
#   %buf23 : Tensor "f32[500, 1, 1][1, 500, 500]cuda:0" = PlaceHolder[target=buf23]
#   %getitem_7 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=getitem_7]
#   %rsqrt_1 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_1]
#   %primals_13 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_13]
#   %primals_14 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_14]
#   %add : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0" = PlaceHolder[target=add]
#   %add_6 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0" = PlaceHolder[target=add_6]
#   %getitem_9 : Tensor "f32[500, 1, 1][1, 500, 500]cuda:0" = PlaceHolder[target=getitem_9]
#   %buf28 : Tensor "f32[500, 1, 1][1, 500, 500]cuda:0" = PlaceHolder[target=buf28]
#   %mul_5 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0" = PlaceHolder[target=mul_5]
#   %primals_15 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_15]
#   %primals_16 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_16]
#   %view_14 : Tensor "f32[1, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_2, [1, 500, 128]), kwargs = {})
#   %permute_14 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_14, [1, 0, 2]), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%permute_14, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_4 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %sub_2 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_14, %getitem_7), kwargs = {})
#   %mul_3 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_4 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %primals_13), kwargs = {})
#   %add_5 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %primals_14), kwargs = {})
#   %add_6 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %add), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_6, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_7 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_2 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_7,), kwargs = {})
#   %sub_3 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_6, %getitem_9), kwargs = {})
#   %mul_5 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %mul_6 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %primals_15), kwargs = {})
#   %add_8 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %primals_16), kwargs = {})
#   %div_3 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_2, 128), kwargs = {})
#   return %getitem_7,%buf23,%rsqrt_1,%add_6,%getitem_9,%buf28,%mul_5,%add_8,%div_3
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_9 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 6, 'num_store': 6, 'num_reduction': 8, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 12000, 'r0_': 2050048}}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr4, out_ptr5, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp24 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr3 + (r0_1 + 128*x0), xmask, other=0.0)
    tmp49 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr5 + (r0_1), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None].to(tl.float32)
    tmp8 = tl.full([1, 1], 128, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = (tmp7 / tmp9)
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None].to(tl.float32)
    tmp17 = 128.0
    tmp18 = (tmp16 / tmp17)
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp0 - tmp10
    tmp23 = tmp22 * tmp21
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp29 = tmp27 + tmp28
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, R0_BLOCK])
    tmp32 = tl.where(xmask, tmp30, 0)
    tmp33 = tl.broadcast_to(tmp30, [XBLOCK, R0_BLOCK])
    tmp35 = tl.where(xmask, tmp33, 0)
    tmp36 = tl.sum(tmp35, 1)[:, None].to(tl.float32)
    tmp37 = (tmp36 / tmp9)
    tmp38 = tmp30 - tmp37
    tmp39 = tmp38 * tmp38
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK, R0_BLOCK])
    tmp42 = tl.where(xmask, tmp40, 0)
    tmp43 = tl.sum(tmp42, 1)[:, None].to(tl.float32)
    tmp44 = tmp29 - tmp37
    tmp45 = (tmp43 / tmp17)
    tmp46 = tmp45 + tmp19
    tmp47 = libdevice.rsqrt(tmp46)
    tmp48 = tmp44 * tmp47
    tmp50 = tmp48 * tmp49
    tmp52 = tmp50 + tmp51
    tmp53 = 0.0078125
    tmp54 = tmp47 * tmp53
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, xmask)
    tl.store(out_ptr1 + (r0_1 + 128*x0), tmp29, xmask)
    tl.store(out_ptr4 + (r0_1 + 128*x0), tmp48, xmask)
    tl.store(out_ptr5 + (r0_1 + 128*x0), tmp52, xmask)
    tl.store(out_ptr6 + (x0), tmp54, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/hd/chdypqbx3kddn5ehuk3n324ywe4t36zfd4n6du66tkeq6tspnj72.py
# Topologically Sorted Source Nodes: [linear_1, gelu, layer_norm_3], Original ATen: [aten.view, aten.gelu, aten.native_layer_norm]
# Source node to ATen node mapping:
#   gelu => add_9, erf, mul_7, mul_8, mul_9
#   layer_norm_3 => add_10, add_11, mul_10, mul_11, rsqrt_3, sub_4, var_mean_3
#   linear_1 => view_17
# Graph fragment:
#   %addmm_3 : Tensor "f32[500, 512][512, 1]cuda:0" = PlaceHolder[target=addmm_3]
#   %buf34 : Tensor "f32[500, 1, 1][1, 500, 500]cuda:0" = PlaceHolder[target=buf34]
#   %getitem_11 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=getitem_11]
#   %rsqrt_3 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_3]
#   %primals_19 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_19]
#   %primals_20 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_20]
#   %view_17 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_3, [500, 1, 512]), kwargs = {})
#   %mul_7 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_17, 0.5), kwargs = {})
#   %mul_8 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_17, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_8,), kwargs = {})
#   %add_9 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_9 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %add_9), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%mul_9, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_10 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_10, 1e-05), kwargs = {})
#   %rsqrt_3 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_10,), kwargs = {})
#   %sub_4 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_9, %getitem_11), kwargs = {})
#   %mul_10 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_3), kwargs = {})
#   %mul_11 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %primals_19), kwargs = {})
#   %add_11 : Tensor "f32[500, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %primals_20), kwargs = {})
#   return %getitem_11,%buf34,%rsqrt_3,%add_11
triton_per_fused_gelu_native_layer_norm_view_10 = async_compile.triton('triton_per_fused_gelu_native_layer_norm_view_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_native_layer_norm_view_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 3, 'num_reduction': 4, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 8000, 'r0_': 3076096}}
)
@triton.jit
def triton_per_fused_gelu_native_layer_norm_view_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp32 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp14 = tl.where(xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
    tmp16 = tl.full([1, 1], 512, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = (tmp15 / tmp17)
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
    tmp23 = tl.where(xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None].to(tl.float32)
    tmp25 = 512.0
    tmp26 = (tmp24 / tmp25)
    tmp27 = 1e-05
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tmp30 = tmp8 - tmp18
    tmp31 = tmp30 * tmp29
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp29, xmask)
    tl.store(out_ptr1 + (r0_1 + 512*x0), tmp35, xmask)
    tl.store(out_ptr0 + (x0), tmp18, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/3j/c3jz4hncvs6a3mz25katusosnwq3fskzhwodukwt4tmrungbqmia.py
# Topologically Sorted Source Nodes: [, linear_2, iadd_2, layer_norm_4, div_1], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#    => add_tensor
#   div_1 => div_1
#   iadd_2 => add_12
#   layer_norm_4 => add_13, add_14, mul_12, mul_13, rsqrt_4, sub_5, var_mean_4
#   linear_2 => view_19
# Graph fragment:
#   %primals_22 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_22]
#   %mm_default : Tensor "f32[500, 128][128, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %add_6 : Tensor "f32[500, 1, 128][128, 64000, 1]cuda:0" = PlaceHolder[target=add_6]
#   %getitem_13 : Tensor "f32[500, 1, 1][1, 500, 500]cuda:0" = PlaceHolder[target=getitem_13]
#   %buf40 : Tensor "f32[500, 1, 1][1, 500, 500]cuda:0" = PlaceHolder[target=buf40]
#   %mul_12 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0" = PlaceHolder[target=mul_12]
#   %primals_23 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_23]
#   %primals_24 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_24]
#   %add_tensor : Tensor "f32[500, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_22, %mm_default), kwargs = {})
#   %view_19 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor, [500, 1, 128]), kwargs = {})
#   %add_12 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_19, %add_6), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_12, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_13 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-05), kwargs = {})
#   %rsqrt_4 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_13,), kwargs = {})
#   %sub_5 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_12, %getitem_13), kwargs = {})
#   %mul_12 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_4), kwargs = {})
#   %mul_13 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_12, %primals_23), kwargs = {})
#   %add_14 : Tensor "f32[500, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %primals_24), kwargs = {})
#   %div_1 : Tensor "f32[500, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_4, 128), kwargs = {})
#   return %getitem_13,%buf40,%mul_12,%add_14,%div_1
triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_11 = async_compile.triton('triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 3, 'num_reduction': 4, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4000, 'r0_': 1537536}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 128*x0), xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r0_1 + 128*x0), xmask, other=0.0)
    tmp28 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp12 = tl.full([1, 1], 128, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp21 = tmp4 - tmp14
    tmp22 = 128.0
    tmp23 = (tmp20 / tmp22)
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = 0.0078125
    tmp33 = tmp26 * tmp32
    tl.store(in_out_ptr0 + (r0_1 + 128*x0), tmp27, xmask)
    tl.store(out_ptr2 + (r0_1 + 128*x0), tmp31, xmask)
    tl.store(out_ptr3 + (x0), tmp33, xmask)
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
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26 = args
        args.clear()
        assert_size_stride(primals_1, (500, 1, 512), (512, 512, 1))
        assert_size_stride(primals_2, (128, 512), (512, 1))
        assert_size_stride(primals_3, (128, ), (1, ))
        assert_size_stride(primals_4, (500, 1, 128), (128, 64000, 1))
        assert_size_stride(primals_5, (500, 128), (128, 1))
        assert_size_stride(primals_6, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(primals_7, (128, ), (1, ))
        assert_size_stride(primals_8, (128, ), (1, ))
        assert_size_stride(primals_9, (384, 128), (128, 1))
        assert_size_stride(primals_10, (384, ), (1, ))
        assert_size_stride(primals_11, (128, 128), (128, 1))
        assert_size_stride(primals_12, (128, ), (1, ))
        assert_size_stride(primals_13, (128, ), (1, ))
        assert_size_stride(primals_14, (128, ), (1, ))
        assert_size_stride(primals_15, (128, ), (1, ))
        assert_size_stride(primals_16, (128, ), (1, ))
        assert_size_stride(primals_17, (512, 128), (128, 1))
        assert_size_stride(primals_18, (512, ), (1, ))
        assert_size_stride(primals_19, (512, ), (1, ))
        assert_size_stride(primals_20, (512, ), (1, ))
        assert_size_stride(primals_21, (128, 512), (512, 1))
        assert_size_stride(primals_22, (128, ), (1, ))
        assert_size_stride(primals_23, (128, ), (1, ))
        assert_size_stride(primals_24, (128, ), (1, ))
        assert_size_stride(primals_25, (10, 128), (128, 1))
        assert_size_stride(primals_26, (10, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((500, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:1
            extern_kernels.mm(reinterpret_tensor(primals_1, (500, 512), (512, 1), 0), reinterpret_tensor(primals_2, (512, 128), (1, 512), 0), out=buf0)
            buf1 = empty_strided_cuda((500, 129, 128), (16512, 128, 1), torch.float32)
            buf2 = empty_strided_cuda((500, 129, 1), (129, 1, 1), torch.float32)
            buf3 = empty_strided_cuda((500, 129, 1), (129, 1, 64512), torch.float32)
            buf5 = reinterpret_tensor(buf3, (500, 129, 1), (129, 1, 1), 0); del buf3  # reuse
            buf8 = empty_strided_cuda((129, 500, 128), (64000, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear, iadd, cat_1, layer_norm, transpose_1, multi_head_attention_forward], Original ATen: [aten.addmm, aten.view, aten.add, aten.cat, aten.native_layer_norm, aten.transpose, aten.clone]
            # [Provenance debug handles] triton_per_fused_add_addmm_cat_clone_native_layer_norm_transpose_view_0:2
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_cat_clone_native_layer_norm_transpose_view_0.run(buf5, primals_3, buf0, primals_4, primals_6, primals_7, primals_8, buf1, buf2, buf8, 64500, 128, stream=stream0)
            del primals_6
            del primals_8
            buf6 = reinterpret_tensor(buf0, (500, 1, 128), (128, 128, 1), 0); del buf0  # reuse
            # Topologically Sorted Source Nodes: [, linear, iadd], Original ATen: [aten.addmm, aten.view, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_addmm_view_1:3
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_view_1.run(buf6, primals_3, primals_4, 64000, stream=stream0)
            del primals_3
            del primals_4
            buf7 = empty_strided_cuda((500, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear, iadd, multi_head_attention_forward], Original ATen: [aten.addmm, aten.view, aten.add, aten.split_with_sizes, aten.t, aten.transpose]
            # [Provenance debug handles] extern_kernels.mm:4
            extern_kernels.mm(reinterpret_tensor(buf6, (500, 128), (128, 1), 0), reinterpret_tensor(primals_9, (128, 128), (1, 128), 0), out=buf7)
            buf9 = empty_strided_cuda((64500, 256), (256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [layer_norm, transpose_1, multi_head_attention_forward], Original ATen: [aten.native_layer_norm, aten.transpose, aten.split_with_sizes, aten.t, aten.clone, aten._unsafe_view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:5
            extern_kernels.mm(reinterpret_tensor(buf8, (64500, 128), (128, 1), 0), reinterpret_tensor(primals_9, (128, 256), (1, 128), 16384), out=buf9)
            buf10 = empty_strided_cuda((2, 129, 500, 128), (8256000, 64000, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.split_with_sizes, aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_clone_split_with_sizes_squeeze_transpose_unsqueeze_view_2:6
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_split_with_sizes_squeeze_transpose_unsqueeze_view_2.run(buf9, primals_10, buf10, 16512000, stream=stream0)
            del buf9
            buf11 = reinterpret_tensor(buf7, (4000, 1, 16), (16, 64000, 1), 0); del buf7  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward, ], Original ATen: [aten.split_with_sizes, aten.addmm, aten.view, aten.transpose, aten.mul]
            # [Provenance debug handles] triton_poi_fused_addmm_mul_split_with_sizes_transpose_view_3:7
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_mul_split_with_sizes_transpose_view_3.run(buf11, primals_10, 64000, stream=stream0)
            del primals_10
            buf12 = empty_strided_cuda((4000, 16, 129), (16, 1, 64000), torch.float32)
            buf48 = empty_strided_cuda((4000, 129, 16), (16, 64000, 1), torch.float32)
            # Topologically Sorted Source Nodes: [zeros_like, cat, multi_head_attention_forward, , permute_38], Original ATen: [aten.zeros_like, aten.cat, aten.split_with_sizes, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.expand, aten.mul, aten.baddbmm]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_addmm_baddbmm_cat_clone_expand_mul_select_split_with_sizes_squeeze_transpose_unsqueeze_view_zeros_like_4:8
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_baddbmm_cat_clone_expand_mul_select_split_with_sizes_squeeze_transpose_unsqueeze_view_zeros_like_4.run(buf10, buf12, buf48, 8256000, stream=stream0)
            buf13 = empty_strided_cuda((500, 8, 1, 129), (1056, 129, 129, 1), torch.float32)
            # Topologically Sorted Source Nodes: [zeros_like, cat, multi_head_attention_forward], Original ATen: [aten.zeros_like, aten.cat, aten.view, aten.expand, aten.clone]
            # [Provenance debug handles] triton_poi_fused_cat_clone_expand_view_zeros_like_5:9
            stream0 = get_raw_stream(0)
            triton_poi_fused_cat_clone_expand_view_zeros_like_5.run(primals_5, buf13, 516000, stream=stream0)
            del primals_5
            buf14 = empty_strided_cuda((4000, 1, 129), (129, 516000, 1), torch.float32)
            # Topologically Sorted Source Nodes: [zeros_like, cat, multi_head_attention_forward, ], Original ATen: [aten.zeros_like, aten.cat, aten.split_with_sizes, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.expand, aten.mul, aten.baddbmm]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_addmm_baddbmm_cat_clone_expand_mul_select_split_with_sizes_squeeze_transpose_unsqueeze_view_zeros_like_6:10
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_baddbmm_cat_clone_expand_mul_select_split_with_sizes_squeeze_transpose_unsqueeze_view_zeros_like_6.run(buf13, buf14, 516000, stream=stream0)
            del buf13
            buf15 = empty_strided_cuda((4000, 1, 129), (129, 129, 1), torch.float32)
            # Topologically Sorted Source Nodes: [zeros_like, cat, multi_head_attention_forward, ], Original ATen: [aten.zeros_like, aten.cat, aten.split_with_sizes, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.expand, aten.mul, aten.baddbmm]
            # [Provenance debug handles] extern_kernels.baddbmm:11
            extern_kernels.baddbmm(buf14, buf11, buf12, alpha=1, beta=1, out=buf15)
            del buf14
            buf18 = buf15; del buf15  # reuse
            # Topologically Sorted Source Nodes: [, multi_head_attention_forward], Original ATen: [prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax]
            # [Provenance debug handles] triton_per_fused__softmax_exp_prepare_softmax_online_sub_7:12
            stream0 = get_raw_stream(0)
            triton_per_fused__softmax_exp_prepare_softmax_online_sub_7.run(buf18, 4000, 129, stream=stream0)
            buf19 = reinterpret_tensor(buf12, (4000, 129, 16), (16, 64000, 1), 0); del buf12  # reuse
            buf47 = empty_strided_cuda((4000, 16, 129), (16, 1, 64000), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward, permute_37], Original ATen: [aten.split_with_sizes, aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.bmm]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_bmm_clone_select_split_with_sizes_squeeze_transpose_unsqueeze_view_8:13
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_bmm_clone_select_split_with_sizes_squeeze_transpose_unsqueeze_view_8.run(buf10, buf19, buf47, 8256000, stream=stream0)
            del buf10
            buf20 = empty_strided_cuda((4000, 1, 16), (16, 16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.split_with_sizes, aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:14
            extern_kernels.bmm(buf18, buf19, out=buf20)
            del buf19
            buf21 = empty_strided_cuda((500, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:15
            extern_kernels.addmm(primals_12, reinterpret_tensor(buf20, (500, 128), (128, 1), 0), reinterpret_tensor(primals_11, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf21)
            del primals_12
            buf22 = empty_strided_cuda((500, 1, 1), (1, 1, 1), torch.float32)
            buf23 = empty_strided_cuda((500, 1, 1), (1, 500, 500), torch.float32)
            buf25 = reinterpret_tensor(buf23, (500, 1, 1), (1, 1, 1), 0); del buf23  # reuse
            buf26 = empty_strided_cuda((500, 1, 128), (128, 64000, 1), torch.float32)
            buf30 = empty_strided_cuda((500, 1, 128), (128, 128, 1), torch.float32)
            buf31 = empty_strided_cuda((500, 1, 128), (128, 128, 1), torch.float32)
            buf46 = empty_strided_cuda((500, 1, 1), (1, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward, transpose_2, layer_norm_1, iadd_1, layer_norm_2, div_3], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm, aten.add, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_9:16
            stream0 = get_raw_stream(0)
            triton_per_fused_add_native_layer_norm_native_layer_norm_backward_transpose_view_9.run(buf25, buf21, primals_13, primals_14, buf6, primals_15, primals_16, buf22, buf26, buf30, buf31, buf46, 500, 128, stream=stream0)
            del primals_14
            del primals_16
            buf32 = empty_strided_cuda((500, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [layer_norm_2, linear_1], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:17
            extern_kernels.addmm(primals_18, reinterpret_tensor(buf31, (500, 128), (128, 1), 0), reinterpret_tensor(primals_17, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf32)
            del primals_18
            buf33 = empty_strided_cuda((500, 1, 1), (1, 1, 1), torch.float32)
            buf34 = empty_strided_cuda((500, 1, 1), (1, 500, 500), torch.float32)
            buf36 = reinterpret_tensor(buf34, (500, 1, 1), (1, 1, 1), 0); del buf34  # reuse
            buf37 = empty_strided_cuda((500, 1, 512), (512, 512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_1, gelu, layer_norm_3], Original ATen: [aten.view, aten.gelu, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_gelu_native_layer_norm_view_10:18
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_native_layer_norm_view_10.run(buf36, buf32, primals_19, primals_20, buf33, buf37, 500, 512, stream=stream0)
            del primals_20
            buf38 = empty_strided_cuda((500, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_1, gelu, layer_norm_3, linear_2, ], Original ATen: [aten.view, aten.gelu, aten.native_layer_norm, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:19
            extern_kernels.mm(reinterpret_tensor(buf37, (500, 512), (512, 1), 0), reinterpret_tensor(primals_21, (512, 128), (1, 512), 0), out=buf38)
            buf42 = reinterpret_tensor(buf38, (500, 1, 128), (128, 128, 1), 0); del buf38  # reuse
            buf43 = empty_strided_cuda((500, 1, 128), (128, 128, 1), torch.float32)
            buf45 = empty_strided_cuda((500, 1, 1), (1, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear_2, iadd_2, layer_norm_4, div_1], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_11:20
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_native_layer_norm_backward_view_11.run(buf42, primals_22, buf26, primals_23, primals_24, buf43, buf45, 500, 128, stream=stream0)
            del buf26
            del primals_22
            del primals_24
            buf44 = empty_strided_cuda((500, 10), (10, 1), torch.float32)
            # Topologically Sorted Source Nodes: [layer_norm_4, squeeze, linear_3], Original ATen: [aten.native_layer_norm, aten.squeeze, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:21
            extern_kernels.addmm(primals_26, reinterpret_tensor(buf43, (500, 128), (128, 1), 0), reinterpret_tensor(primals_25, (128, 10), (1, 128), 0), alpha=1, beta=1, out=buf44)
            del primals_26
        return (buf44, primals_2, primals_7, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, reinterpret_tensor(primals_1, (500, 512), (512, 1), 0), buf1, buf2, buf5, reinterpret_tensor(buf6, (500, 128), (128, 1), 0), reinterpret_tensor(buf8, (64500, 128), (128, 1), 0), buf18, reinterpret_tensor(buf20, (500, 128), (128, 1), 0), buf21, buf22, buf25, buf30, reinterpret_tensor(buf31, (500, 128), (128, 1), 0), buf32, buf33, buf36, reinterpret_tensor(buf37, (500, 512), (512, 1), 0), buf42, reinterpret_tensor(buf43, (500, 128), (128, 1), 0), buf45, buf46, buf47, buf48, reinterpret_tensor(buf11, (4000, 16, 1), (16, 1, 16), 0), reinterpret_tensor(primals_9, (256, 128), (128, 1), 16384), reinterpret_tensor(primals_9, (128, 128), (128, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((500, 1, 512), (512, 512, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((500, 1, 128), (128, 64000, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((500, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((10, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
