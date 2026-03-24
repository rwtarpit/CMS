# AOT ID: ['15_forward']
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


# kernel path: /traces/inductor_cache/6a/c6ah7dw3o3mrtoqqnpllji2ljpvkifnkkjhi5mhxvv3aucymnb6g.py
# Topologically Sorted Source Nodes: [, linear, inductor_random, dropout, iadd, layer_norm, transpose, multi_head_attention_forward, div_4], Original ATen: [aten.addmm, aten.view, aten.native_dropout, aten.add, aten.native_layer_norm, aten.transpose, aten.clone, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#    => add_tensor_1, inductor_lookup_seed_default
#   div_4 => div_4
#   dropout => gt, mul, mul_1
#   iadd => add
#   inductor_random => inductor_random_default_4
#   layer_norm => add_1, add_2, mul_2, mul_3, rsqrt, sub, var_mean
#   linear => view_1
#   multi_head_attention_forward => clone
#   transpose => permute_1
# Graph fragment:
#   %inductor_seeds_default : Tensor "i64[5][1]cuda:0" = PlaceHolder[target=inductor_seeds_default]
#   %inductor_random_default_4 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=inductor_random_default_4]
#   %gt : Tensor "b8[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=gt]
#   %primals_3 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_3]
#   %mm_default_1 : Tensor "f32[64000, 128][128, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %primals_4 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=primals_4]
#   %getitem_1 : Tensor "f32[500, 128, 1][128, 1, 64000]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf5 : Tensor "f32[500, 128, 1][128, 1, 64000]cuda:0" = PlaceHolder[target=buf5]
#   %mul_2 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=mul_2]
#   %primals_5 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_5]
#   %primals_6 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_6]
#   %add_tensor_1 : Tensor "f32[64000, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_3, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [500, 128, 128]), kwargs = {})
#   %inductor_lookup_seed_default : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_4 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([500, 128, 128], %inductor_lookup_seed_default, rand), kwargs = {})
#   %gt : Tensor "b8[500, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%inductor_random_default_4, 0.1), kwargs = {})
#   %mul : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt, %view_1), kwargs = {})
#   %mul_1 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, 1.1111111111111112), kwargs = {})
#   %add : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %primals_4), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_1 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %sub : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem_1), kwargs = {})
#   %mul_2 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_3 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %primals_5), kwargs = {})
#   %add_2 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %primals_6), kwargs = {})
#   %permute_1 : Tensor "f32[128, 500, 128][128, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_2, [1, 0, 2]), kwargs = {})
#   %clone : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_1,), kwargs = {memory_format: torch.contiguous_format})
#   %div_4 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt, 128), kwargs = {})
#   return %inductor_random_default_4,%gt,%getitem_1,%buf5,%mul_2,%div_4,%clone
triton_per_fused_add_addmm_clone_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_0 = async_compile.triton('triton_per_fused_add_addmm_clone_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*i1', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'load_seed_offset': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_clone_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 4, 'num_reduction': 4, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 512000, 'r0_': 212993536}}
)
@triton.jit
def triton_per_fused_add_addmm_clone_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr4, out_ptr5, out_ptr6, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    x2 = (xindex % 128)
    x3 = xindex // 128
    tmp6 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (r0_1 + 128*x0), xmask, other=0.0)
    tmp12 = tl.load(in_ptr3 + (r0_1 + 128*x0), xmask, other=0.0)
    tmp39 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr5 + (r0_1), None, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r0_1 + 128*x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(xmask, tmp14, 0)
    tmp17 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp21 = tl.full([1, 1], 128, tl.int32)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = (tmp20 / tmp22)
    tmp24 = tmp14 - tmp23
    tmp25 = tmp24 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
    tmp28 = tl.where(xmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None].to(tl.float32)
    tmp30 = tmp13 - tmp23
    tmp31 = 128.0
    tmp32 = (tmp29 / tmp31)
    tmp33 = 1e-05
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.rsqrt(tmp34)
    tmp36 = tmp30 * tmp35
    tmp37 = 0.0078125
    tmp38 = tmp35 * tmp37
    tmp40 = tmp36 * tmp39
    tmp42 = tmp40 + tmp41
    tl.store(out_ptr1 + (r0_1 + 128*x0), tmp4, xmask)
    tl.store(out_ptr4 + (r0_1 + 128*x0), tmp36, xmask)
    tl.store(out_ptr5 + (x0), tmp38, xmask)
    tl.store(out_ptr6 + (r0_1 + 128*x3 + 64000*x2), tmp42, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/l5/cl5j7vp4nqavqvxkj5xenqkcx46hc6rehraqup6ekokbixxotarf.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => add_3, clone_1, permute_3, squeeze, unsqueeze, view_3, view_4
# Graph fragment:
#   %mm : Tensor "f32[64000, 384][384, 1]cuda:0" = PlaceHolder[target=mm]
#   %primals_8 : Tensor "f32[384][1]cuda:0" = PlaceHolder[target=primals_8]
#   %view_3 : Tensor "f32[128, 500, 384][192000, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [128, 500, 384]), kwargs = {})
#   %add_3 : Tensor "f32[128, 500, 384][192000, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_3, %primals_8), kwargs = {})
#   %view_4 : Tensor "f32[128, 500, 3, 128][192000, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_3, [128, 500, 3, 128]), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 128, 500, 3, 128][24576000, 192000, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_4, 0), kwargs = {})
#   %permute_3 : Tensor "f32[3, 128, 500, 1, 128][128, 192000, 384, 24576000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze : Tensor "f32[3, 128, 500, 128][128, 192000, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_3, -2), kwargs = {})
#   %clone_1 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_1
triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_1 = async_compile.triton('triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 294913536}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 64000)
    x2 = xindex // 8192000
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x2 + 384*x1), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 128*x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/jk/cjknggepg4ydf2cdyjiphl5qd7p6rgx5wlg3ttgtvpzwpwxfmzk7.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.mul]
# Source node to ATen node mapping:
#   multi_head_attention_forward => add_3, clone_1, mul_4, permute_3, permute_4, select, squeeze, unsqueeze, view_3, view_4, view_5
# Graph fragment:
#   %clone_1 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0" = PlaceHolder[target=clone_1]
#   %view_3 : Tensor "f32[128, 500, 384][192000, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [128, 500, 384]), kwargs = {})
#   %add_3 : Tensor "f32[128, 500, 384][192000, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_3, %primals_8), kwargs = {})
#   %view_4 : Tensor "f32[128, 500, 3, 128][192000, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_3, [128, 500, 3, 128]), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 128, 500, 3, 128][24576000, 192000, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_4, 0), kwargs = {})
#   %permute_3 : Tensor "f32[3, 128, 500, 1, 128][128, 192000, 384, 24576000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze : Tensor "f32[3, 128, 500, 128][128, 192000, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_3, -2), kwargs = {})
#   %clone_1 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
#   %select : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_1, 0, 0), kwargs = {})
#   %view_5 : Tensor "f32[128, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select, [128, 4000, 16]), kwargs = {})
#   %permute_4 : Tensor "f32[4000, 128, 16][16, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_5, [1, 0, 2]), kwargs = {})
#   %mul_4 : Tensor "f32[4000, 128, 16][16, 64000, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_4, 0.25), kwargs = {})
#   return %mul_4
triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2 = async_compile.triton('triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 98304000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = 0.25
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/ko/ckoaw7tjuz7io2yaxyrvmr3cdjeurb5awjb45qej63so5kx3tqr3.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward, permute_29], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.expand, aten.mul, aten.baddbmm]
# Source node to ATen node mapping:
#   multi_head_attention_forward => add_3, add_4, baddbmm, clone_1, clone_2, expand, mul_4, permute_3, permute_4, permute_5, permute_7, select, select_1, squeeze, unsqueeze, view_3, view_4, view_5, view_6, view_8, view_9
#   permute_29 => permute_29
# Graph fragment:
#   %clone_1 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0" = PlaceHolder[target=clone_1]
#   %view_3 : Tensor "f32[128, 500, 384][192000, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [128, 500, 384]), kwargs = {})
#   %add_3 : Tensor "f32[128, 500, 384][192000, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_3, %primals_8), kwargs = {})
#   %view_4 : Tensor "f32[128, 500, 3, 128][192000, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_3, [128, 500, 3, 128]), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 128, 500, 3, 128][24576000, 192000, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_4, 0), kwargs = {})
#   %permute_3 : Tensor "f32[3, 128, 500, 1, 128][128, 192000, 384, 24576000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze : Tensor "f32[3, 128, 500, 128][128, 192000, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_3, -2), kwargs = {})
#   %clone_1 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
#   %select : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_1, 0, 0), kwargs = {})
#   %select_1 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_1, 0, 1), kwargs = {})
#   %view_5 : Tensor "f32[128, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select, [128, 4000, 16]), kwargs = {})
#   %permute_4 : Tensor "f32[4000, 128, 16][16, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_5, [1, 0, 2]), kwargs = {})
#   %view_6 : Tensor "f32[128, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_1, [128, 4000, 16]), kwargs = {})
#   %permute_5 : Tensor "f32[4000, 128, 16][16, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_6, [1, 0, 2]), kwargs = {})
#   %view_8 : Tensor "f32[500, 1, 1, 128][128, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_11, [500, 1, 1, 128]), kwargs = {})
#   %expand : Tensor "f32[500, 8, 1, 128][128, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%view_8, [-1, 8, -1, -1]), kwargs = {})
#   %clone_2 : Tensor "f32[500, 8, 1, 128][1024, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
#   %view_9 : Tensor "f32[4000, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_2, [4000, 1, 128]), kwargs = {})
#   %add_4 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_12, %view_9), kwargs = {})
#   %mul_4 : Tensor "f32[4000, 128, 16][16, 64000, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_4, 0.25), kwargs = {})
#   %permute_7 : Tensor "f32[4000, 16, 128][16, 1, 64000]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%permute_5, [0, 2, 1]), kwargs = {})
#   %baddbmm : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.baddbmm.default](args = (%add_4, %mul_4, %permute_7), kwargs = {})
#   %permute_29 : Tensor "f32[4000, 128, 16][16, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_7, [0, 2, 1]), kwargs = {})
#   return %buf12,%permute_29
triton_poi_fused__unsafe_view_add_baddbmm_clone_expand_mul_select_squeeze_transpose_unsqueeze_view_3 = async_compile.triton('triton_poi_fused__unsafe_view_add_baddbmm_clone_expand_mul_select_squeeze_transpose_unsqueeze_view_3', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_baddbmm_clone_expand_mul_select_squeeze_transpose_unsqueeze_view_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 163840000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_baddbmm_clone_expand_mul_select_squeeze_transpose_unsqueeze_view_3(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (8192000 + x2), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
    tl.store(out_ptr1 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/i7/ci7wvf3mbk2yuvk7nwvw5xzh6gstmyiz727ehyku6qmehyfubrhx.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view, aten.expand, aten.clone, aten._unsafe_view, aten.add]
# Source node to ATen node mapping:
#   multi_head_attention_forward => add_4, clone_2, expand, view_8, view_9
# Graph fragment:
#   %primals_12 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=primals_12]
#   %primals_11 : Tensor "f32[500, 128][128, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %view_8 : Tensor "f32[500, 1, 1, 128][128, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_11, [500, 1, 1, 128]), kwargs = {})
#   %expand : Tensor "f32[500, 8, 1, 128][128, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%view_8, [-1, 8, -1, -1]), kwargs = {})
#   %clone_2 : Tensor "f32[500, 8, 1, 128][1024, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
#   %view_9 : Tensor "f32[4000, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_2, [4000, 1, 128]), kwargs = {})
#   %add_4 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_12, %view_9), kwargs = {})
#   return %add_4
triton_poi_fused__unsafe_view_add_clone_expand_view_4 = async_compile.triton('triton_poi_fused__unsafe_view_add_clone_expand_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_clone_expand_view_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 786688000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_clone_expand_view_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x0 = (xindex % 128)
    x2 = xindex // 16384
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 128*(x2 // 8)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/52/c52r3zanfebm7625kssh2mkjedjfp3aq4t3ff3at5bqpqps46k4z.py
# Topologically Sorted Source Nodes: [, multi_head_attention_forward, inductor_random], Original ATen: [prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten.native_dropout]
# Source node to ATen node mapping:
#    => exp_default, inductor_lookup_seed_default_1, prepare_softmax_online_default, sub_tensor
#   inductor_random => inductor_random_default_3
#   multi_head_attention_forward => div, gt_1, mul_5, mul_6
# Graph fragment:
#   %baddbmm : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=baddbmm]
#   %inductor_seeds_default : Tensor "i64[5][1]cuda:0" = PlaceHolder[target=inductor_seeds_default]
#   %inductor_random_default_3 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=inductor_random_default_3]
#   %gt_1 : Tensor "b8[4000, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=gt_1]
#   %getitem_8 : Tensor "f32[4000, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=getitem_8]
#   %getitem_9 : Tensor "f32[4000, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=getitem_9]
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%baddbmm, -1), kwargs = {})
#   %sub_tensor : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%baddbmm, %getitem_8), kwargs = {})
#   %exp_default : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_9), kwargs = {})
#   %inductor_lookup_seed_default_1 : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default_3 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([4000, 128, 128], %inductor_lookup_seed_default_1, rand), kwargs = {})
#   %gt_1 : Tensor "b8[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%inductor_random_default_3, 0.1), kwargs = {})
#   %mul_5 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt_1, %div), kwargs = {})
#   %mul_6 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, 1.1111111111111112), kwargs = {})
#   return %getitem_8,%getitem_9,%inductor_random_default_3,%gt_1,%mul_6
triton_per_fused__softmax_exp_native_dropout_prepare_softmax_online_sub_5 = async_compile.triton('triton_per_fused__softmax_exp_native_dropout_prepare_softmax_online_sub_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr3': '*i1', 'out_ptr4': '*fp32', 'load_seed_offset': 'constexpr', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'load_seed_offset': 1}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_exp_native_dropout_prepare_softmax_online_sub_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 4, 'num_reduction': 4, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 8192000, 'r0_': 917504000}}
)
@triton.jit
def triton_per_fused__softmax_exp_native_dropout_prepare_softmax_online_sub_5(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr3, out_ptr4, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = triton_helpers.max2(tmp3, 1)[:, None].to(tl.float32)
    tmp6 = tmp1 - tmp5
    tmp7 = libdevice.exp(tmp6)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.sum(tmp8, 1)[:, None].to(tl.float32)
    tmp11 = tl.load(in_ptr1 + load_seed_offset)
    tmp12 = r0_1 + 128*x0
    tmp13 = tl.rand(tmp11, (tmp12).to(tl.uint32))
    tmp14 = 0.1
    tmp15 = tmp13 > tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp0 - tmp5
    tmp18 = libdevice.exp(tmp17)
    tmp19 = (tmp18 / tmp10)
    tmp20 = tmp16 * tmp19
    tmp21 = 1.1111111111111112
    tmp22 = tmp20 * tmp21
    tl.store(out_ptr3 + (r0_1 + 128*x0), tmp15, None)
    tl.store(out_ptr4 + (r0_1 + 128*x0), tmp22, None)
    tl.store(out_ptr0 + (x0), tmp5, None)
    tl.store(out_ptr1 + (x0), tmp10, None)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/rt/crtkrth3nlxntymo6fgzwrpgxfqy5sknjigck6j2jua2e6vixscp.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward, permute_28], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.bmm]
# Source node to ATen node mapping:
#   multi_head_attention_forward => add_3, bmm, clone_1, permute_3, permute_6, select_2, squeeze, unsqueeze, view_3, view_4, view_7
#   permute_28 => permute_28
# Graph fragment:
#   %clone_1 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0" = PlaceHolder[target=clone_1]
#   %view_3 : Tensor "f32[128, 500, 384][192000, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [128, 500, 384]), kwargs = {})
#   %add_3 : Tensor "f32[128, 500, 384][192000, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_3, %primals_8), kwargs = {})
#   %view_4 : Tensor "f32[128, 500, 3, 128][192000, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_3, [128, 500, 3, 128]), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 128, 500, 3, 128][24576000, 192000, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_4, 0), kwargs = {})
#   %permute_3 : Tensor "f32[3, 128, 500, 1, 128][128, 192000, 384, 24576000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze : Tensor "f32[3, 128, 500, 128][128, 192000, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_3, -2), kwargs = {})
#   %clone_1 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
#   %select_2 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_1, 0, 2), kwargs = {})
#   %view_7 : Tensor "f32[128, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_2, [128, 4000, 16]), kwargs = {})
#   %permute_6 : Tensor "f32[4000, 128, 16][16, 64000, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_7, [1, 0, 2]), kwargs = {})
#   %bmm : Tensor "f32[4000, 128, 16][2048, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%mul_6, %permute_6), kwargs = {})
#   %permute_28 : Tensor "f32[4000, 16, 128][16, 1, 64000]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_6, [0, 2, 1]), kwargs = {})
#   return %buf20,%permute_28
triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_6 = async_compile.triton('triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_6', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 163840000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_6(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (16384000 + x2), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
    tl.store(out_ptr1 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/6d/c6da4hvjvau3sazwh74pfagmzd54wnfbmwxr3gns3ehaa5r4ewaq.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => clone_3, permute_8
# Graph fragment:
#   %bmm : Tensor "f32[4000, 128, 16][2048, 16, 1]cuda:0" = PlaceHolder[target=bmm]
#   %permute_8 : Tensor "f32[128, 4000, 16][16, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm, [1, 0, 2]), kwargs = {})
#   %clone_3 : Tensor "f32[128, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_8,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_3
triton_poi_fused_clone_transpose_7 = async_compile.triton('triton_poi_fused_clone_transpose_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_transpose_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 98304000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_transpose_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 16)
    x1 = ((xindex // 16) % 4000)
    x2 = xindex // 64000
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16*x2 + 2048*x1), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/o6/co6wr2wwhx5zxrmcghbhmcc64poav4vf7mtm3bykxbzw2mawpixg.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward, transpose_1, layer_norm_1], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_1 => clone_4, var_mean_1
#   multi_head_attention_forward => view_11
#   transpose_1 => permute_10
# Graph fragment:
#   %addmm_1 : Tensor "f32[64000, 128][128, 1]cuda:0" = PlaceHolder[target=addmm_1]
#   %view_11 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_1, [128, 500, 128]), kwargs = {})
#   %permute_10 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_11, [1, 0, 2]), kwargs = {})
#   %clone_4 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_10,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_4, [2]), kwargs = {correction: 0, keepdim: True})
#   return %buf25
triton_per_fused_native_layer_norm_transpose_view_8 = async_compile.triton('triton_per_fused_native_layer_norm_transpose_view_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_transpose_view_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 3, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 512000, 'r0_': 32768000}}
)
@triton.jit
def triton_per_fused_native_layer_norm_transpose_view_8(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x0), tmp16, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/4y/c4yhhkhcvadlepxnb5nvagc7j4ocojkcoejhnzgvp6klrbhaj6uo.py
# Topologically Sorted Source Nodes: [, linear, dropout, iadd, multi_head_attention_forward, transpose_1, layer_norm_1, inductor_random, dropout_1, iadd_1, layer_norm_2, div_2], Original ATen: [aten.addmm, aten.view, aten.native_dropout, aten.add, aten.transpose, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#    => add_tensor_1, inductor_lookup_seed_default_2
#   div_2 => div_2
#   dropout => mul, mul_1
#   dropout_1 => gt_2, mul_10, mul_9
#   iadd => add
#   iadd_1 => add_7
#   inductor_random => inductor_random_default_2
#   layer_norm_1 => add_5, add_6, clone_4, mul_7, mul_8, rsqrt_1, sub_2, var_mean_1
#   layer_norm_2 => add_8, add_9, mul_11, mul_12, rsqrt_2, sub_3, var_mean_2
#   linear => view_1
#   multi_head_attention_forward => view_11
#   transpose_1 => permute_10
# Graph fragment:
#   %buf25 : Tensor "f32[500, 128, 1][1, 500, 64000]cuda:0" = PlaceHolder[target=buf25]
#   %addmm_1 : Tensor "f32[64000, 128][128, 1]cuda:0" = PlaceHolder[target=addmm_1]
#   %inductor_seeds_default : Tensor "i64[5][1]cuda:0" = PlaceHolder[target=inductor_seeds_default]
#   %inductor_random_default_2 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=inductor_random_default_2]
#   %gt_2 : Tensor "b8[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=gt_2]
#   %getitem_3 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=getitem_3]
#   %rsqrt_1 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_1]
#   %primals_13 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_13]
#   %primals_14 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_14]
#   %gt : Tensor "b8[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=gt]
#   %primals_3 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_3]
#   %mm_default_1 : Tensor "f32[64000, 128][128, 1]cuda:0" = PlaceHolder[target=mm_default_1]
#   %primals_4 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=primals_4]
#   %add_7 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=add_7]
#   %getitem_5 : Tensor "f32[500, 128, 1][128, 1, 64000]cuda:0" = PlaceHolder[target=getitem_5]
#   %buf32 : Tensor "f32[500, 128, 1][128, 1, 64000]cuda:0" = PlaceHolder[target=buf32]
#   %mul_11 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=mul_11]
#   %primals_15 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_15]
#   %primals_16 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_16]
#   %add_tensor_1 : Tensor "f32[64000, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_3, %mm_default_1), kwargs = {})
#   %view_1 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_1, [500, 128, 128]), kwargs = {})
#   %mul : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt, %view_1), kwargs = {})
#   %mul_1 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, 1.1111111111111112), kwargs = {})
#   %add : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %primals_4), kwargs = {})
#   %view_11 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_1, [128, 500, 128]), kwargs = {})
#   %permute_10 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_11, [1, 0, 2]), kwargs = {})
#   %clone_4 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_10,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_4, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_5 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %sub_2 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_4, %getitem_3), kwargs = {})
#   %mul_7 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_8 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %primals_13), kwargs = {})
#   %add_6 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %primals_14), kwargs = {})
#   %inductor_lookup_seed_default_2 : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 2), kwargs = {})
#   %inductor_random_default_2 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([500, 128, 128], %inductor_lookup_seed_default_2, rand), kwargs = {})
#   %gt_2 : Tensor "b8[500, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%inductor_random_default_2, 0.1), kwargs = {})
#   %mul_9 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt_2, %add_6), kwargs = {})
#   %mul_10 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, 1.1111111111111112), kwargs = {})
#   %add_7 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %add), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_7, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_8 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt_2 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_8,), kwargs = {})
#   %sub_3 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_7, %getitem_5), kwargs = {})
#   %mul_11 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %mul_12 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_11, %primals_15), kwargs = {})
#   %add_9 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %primals_16), kwargs = {})
#   %div_2 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_2, 128), kwargs = {})
#   return %rsqrt_1,%getitem_3,%inductor_random_default_2,%gt_2,%add_7,%getitem_5,%buf32,%mul_11,%add_9,%div_2
triton_red_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_9 = async_compile.triton('triton_red_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*i1', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr3': '*i1', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'out_ptr8': '*fp32', 'load_seed_offset': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]], (19,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 13, 'num_store': 7, 'num_reduction': 3, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 512000, 'r0_': 319490560}}
)
@triton.jit
def triton_red_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, out_ptr3, out_ptr6, out_ptr7, out_ptr8, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 500*x0), xmask, eviction_policy='evict_last')
    tmp1 = 128.0
    tmp2 = (tmp0 / tmp1)
    tmp3 = 1e-05
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.rsqrt(tmp4)
    tl.store(out_ptr0 + (x3), tmp5, xmask)
    tmp8_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp6 = tl.load(in_ptr1 + (r0_2 + 128*x1 + 64000*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight, roffset == 0
        )
        tmp8_mean = tl.where(r0_mask & xmask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(r0_mask & xmask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(r0_mask & xmask, tmp8_weight_next, tmp8_weight)
        tmp14 = tl.load(in_ptr2 + load_seed_offset)
        tmp15 = r0_2 + 128*x3
        tmp16 = tl.rand(tmp14, (tmp15).to(tl.uint32))
        tmp17 = 0.1
        tmp18 = tmp16 > tmp17
        tl.store(out_ptr3 + (r0_2 + 128*x3), tmp18, r0_mask & xmask)
    tmp9, tmp10, tmp11 = triton_helpers.welford(tmp8_mean, tmp8_m2, tmp8_weight, 1)
    tmp8 = tmp9[:, None]
    tmp12 = tmp10[:, None]
    tmp13 = tmp11[:, None]
    tl.store(out_ptr1 + (x3), tmp8, xmask)
    tmp42_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp42_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp42_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp19 = tl.load(out_ptr3 + (r0_2 + 128*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.int1)
        tmp21 = tl.load(in_ptr1 + (r0_2 + 128*x1 + 64000*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp24 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp26 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp31 = tl.load(in_ptr5 + (r0_2 + 128*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.int1)
        tmp33 = tl.load(in_ptr6 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp34 = tl.load(in_out_ptr0 + (r0_2 + 128*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp38 = tl.load(in_ptr7 + (r0_2 + 128*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tmp19.to(tl.float32)
        tmp22 = tmp21 - tmp8
        tmp23 = tmp22 * tmp5
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tmp28 = tmp20 * tmp27
        tmp29 = 1.1111111111111112
        tmp30 = tmp28 * tmp29
        tmp32 = tmp31.to(tl.float32)
        tmp35 = tmp33 + tmp34
        tmp36 = tmp32 * tmp35
        tmp37 = tmp36 * tmp29
        tmp39 = tmp37 + tmp38
        tmp40 = tmp30 + tmp39
        tmp41 = tl.broadcast_to(tmp40, [XBLOCK, R0_BLOCK])
        tmp42_mean_next, tmp42_m2_next, tmp42_weight_next = triton_helpers.welford_reduce(
            tmp41, tmp42_mean, tmp42_m2, tmp42_weight, roffset == 0
        )
        tmp42_mean = tl.where(r0_mask & xmask, tmp42_mean_next, tmp42_mean)
        tmp42_m2 = tl.where(r0_mask & xmask, tmp42_m2_next, tmp42_m2)
        tmp42_weight = tl.where(r0_mask & xmask, tmp42_weight_next, tmp42_weight)
        tl.store(in_out_ptr0 + (r0_2 + 128*x3), tmp40, r0_mask & xmask)
    tmp43, tmp44, tmp45 = triton_helpers.welford(tmp42_mean, tmp42_m2, tmp42_weight, 1)
    tmp42 = tmp43[:, None]
    tmp46 = tmp44[:, None]
    tmp47 = tmp45[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp48 = tl.load(in_out_ptr0 + (r0_2 + 128*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp56 = tl.load(in_ptr8 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp58 = tl.load(in_ptr9 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp49 = tmp48 - tmp42
        tmp50 = 128.0
        tmp51 = (tmp46 / tmp50)
        tmp52 = 1e-05
        tmp53 = tmp51 + tmp52
        tmp54 = libdevice.rsqrt(tmp53)
        tmp55 = tmp49 * tmp54
        tmp57 = tmp55 * tmp56
        tmp59 = tmp57 + tmp58
        tl.store(out_ptr6 + (r0_2 + 128*x3), tmp55, r0_mask & xmask)
        tl.store(out_ptr7 + (r0_2 + 128*x3), tmp59, r0_mask & xmask)
    tmp60 = 128.0
    tmp61 = (tmp46 / tmp60)
    tmp62 = 1e-05
    tmp63 = tmp61 + tmp62
    tmp64 = libdevice.rsqrt(tmp63)
    tmp65 = 0.0078125
    tmp66 = tmp64 * tmp65
    tl.store(out_ptr8 + (x3), tmp66, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/wp/cwpossgbvevto4tujzvs4wbjf3ij3vvqp32ozisw47g6ariz7a3n.py
# Topologically Sorted Source Nodes: [linear_1, gelu, , inductor_random, dropout_2, layer_norm_3], Original ATen: [aten.view, aten.gelu, aten.native_dropout, aten.native_layer_norm]
# Source node to ATen node mapping:
#    => inductor_lookup_seed_default_3
#   dropout_2 => gt_3, mul_16, mul_17
#   gelu => add_10, erf, mul_13, mul_14, mul_15
#   inductor_random => inductor_random_default_1
#   layer_norm_3 => add_11, add_12, mul_18, mul_19, rsqrt_3, sub_4, var_mean_3
#   linear_1 => view_14
# Graph fragment:
#   %inductor_seeds_default : Tensor "i64[5][1]cuda:0" = PlaceHolder[target=inductor_seeds_default]
#   %inductor_random_default_1 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0" = PlaceHolder[target=inductor_random_default_1]
#   %gt_3 : Tensor "b8[500, 128, 512][65536, 512, 1]cuda:0" = PlaceHolder[target=gt_3]
#   %addmm_2 : Tensor "f32[64000, 512][512, 1]cuda:0" = PlaceHolder[target=addmm_2]
#   %buf40 : Tensor "f32[500, 128, 1][128, 1, 64000]cuda:0" = PlaceHolder[target=buf40]
#   %getitem_7 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=getitem_7]
#   %rsqrt_3 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_3]
#   %primals_19 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_19]
#   %primals_20 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_20]
#   %view_14 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_2, [500, 128, 512]), kwargs = {})
#   %mul_13 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, 0.5), kwargs = {})
#   %mul_14 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_14,), kwargs = {})
#   %add_10 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_15 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %add_10), kwargs = {})
#   %inductor_lookup_seed_default_3 : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 3), kwargs = {})
#   %inductor_random_default_1 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([500, 128, 512], %inductor_lookup_seed_default_3, rand), kwargs = {})
#   %gt_3 : Tensor "b8[500, 128, 512][65536, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%inductor_random_default_1, 0.1), kwargs = {})
#   %mul_16 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt_3, %mul_15), kwargs = {})
#   %mul_17 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, 1.1111111111111112), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%mul_17, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_11 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_3 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_11,), kwargs = {})
#   %sub_4 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_17, %getitem_7), kwargs = {})
#   %mul_18 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_3), kwargs = {})
#   %mul_19 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, %primals_19), kwargs = {})
#   %add_12 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %primals_20), kwargs = {})
#   return %inductor_random_default_1,%gt_3,%getitem_7,%buf40,%rsqrt_3,%add_12
triton_per_fused_gelu_native_dropout_native_layer_norm_view_10 = async_compile.triton('triton_per_fused_gelu_native_dropout_native_layer_norm_view_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'load_seed_offset': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_native_dropout_native_layer_norm_view_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 4, 'num_reduction': 4, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1024000, 'r0_': 458756096}}
)
@triton.jit
def triton_per_fused_gelu_native_dropout_native_layer_norm_view_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, out_ptr3, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 64000
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
    tmp6 = tl.load(in_ptr1 + (r0_1 + 512*x0), xmask, other=0.0)
    tmp41 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r0_1 + 512*x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = 0.7071067811865476
    tmp10 = tmp6 * tmp9
    tmp11 = libdevice.erf(tmp10)
    tmp12 = 1.0
    tmp13 = tmp11 + tmp12
    tmp14 = tmp8 * tmp13
    tmp15 = tmp5 * tmp14
    tmp16 = 1.1111111111111112
    tmp17 = tmp15 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp20 = tl.where(xmask, tmp18, 0)
    tmp21 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp23 = tl.where(xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None].to(tl.float32)
    tmp25 = tl.full([1, 1], 512, tl.int32)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = (tmp24 / tmp26)
    tmp28 = tmp18 - tmp27
    tmp29 = tmp28 * tmp28
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, R0_BLOCK])
    tmp32 = tl.where(xmask, tmp30, 0)
    tmp33 = tl.sum(tmp32, 1)[:, None].to(tl.float32)
    tmp34 = 512.0
    tmp35 = (tmp33 / tmp34)
    tmp36 = 1e-05
    tmp37 = tmp35 + tmp36
    tmp38 = libdevice.rsqrt(tmp37)
    tmp39 = tmp17 - tmp27
    tmp40 = tmp39 * tmp38
    tmp42 = tmp40 * tmp41
    tmp44 = tmp42 + tmp43
    tl.store(out_ptr1 + (r0_1 + 512*x0), tmp4, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp38, xmask)
    tl.store(out_ptr3 + (r0_1 + 512*x0), tmp44, xmask)
    tl.store(out_ptr2 + (x0), tmp27, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/lo/clo4kncg4hkiywl7wkirm55f5slxa3zzn4zj4ccuxxd5deb63own.py
# Topologically Sorted Source Nodes: [, linear_2, inductor_random, dropout_3, iadd_2], Original ATen: [aten.addmm, aten.view, aten.native_dropout, aten.add]
# Source node to ATen node mapping:
#    => add_tensor, inductor_lookup_seed_default_4
#   dropout_3 => gt_4, mul_20, mul_21
#   iadd_2 => add_13
#   inductor_random => inductor_random_default
#   linear_2 => view_16
# Graph fragment:
#   %inductor_seeds_default : Tensor "i64[5][1]cuda:0" = PlaceHolder[target=inductor_seeds_default]
#   %inductor_random_default : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=inductor_random_default]
#   %gt_4 : Tensor "b8[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=gt_4]
#   %primals_22 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_22]
#   %mm_default : Tensor "f32[64000, 128][128, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %add_7 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=add_7]
#   %add_tensor : Tensor "f32[64000, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_22, %mm_default), kwargs = {})
#   %view_16 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor, [500, 128, 128]), kwargs = {})
#   %inductor_lookup_seed_default_4 : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 4), kwargs = {})
#   %inductor_random_default : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([500, 128, 128], %inductor_lookup_seed_default_4, rand), kwargs = {})
#   %gt_4 : Tensor "b8[500, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%inductor_random_default, 0.1), kwargs = {})
#   %mul_20 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt_4, %view_16), kwargs = {})
#   %mul_21 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, 1.1111111111111112), kwargs = {})
#   %add_13 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_21, %add_7), kwargs = {})
#   return %inductor_random_default,%gt_4,%add_13
triton_poi_fused_add_addmm_native_dropout_view_11 = async_compile.triton('triton_poi_fused_add_addmm_native_dropout_view_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr1': '*i1', 'load_seed_offset': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_native_dropout_view_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 147456512}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_native_dropout_view_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr1, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    x1 = (xindex % 128)
    tmp6 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_out_ptr0 + (x0), None)
    tmp12 = tl.load(in_ptr2 + (x0), None)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr1 + (x0), tmp4, None)
    tl.store(in_out_ptr0 + (x0), tmp13, None)
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
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22 = args
        args.clear()
        assert_size_stride(primals_1, (500, 128, 512), (65536, 512, 1))
        assert_size_stride(primals_2, (128, 512), (512, 1))
        assert_size_stride(primals_3, (128, ), (1, ))
        assert_size_stride(primals_4, (500, 128, 128), (16384, 128, 1))
        assert_size_stride(primals_5, (128, ), (1, ))
        assert_size_stride(primals_6, (128, ), (1, ))
        assert_size_stride(primals_7, (384, 128), (128, 1))
        assert_size_stride(primals_8, (384, ), (1, ))
        assert_size_stride(primals_9, (128, 128), (128, 1))
        assert_size_stride(primals_10, (128, ), (1, ))
        assert_size_stride(primals_11, (500, 128), (128, 1))
        assert_size_stride(primals_12, (4000, 128, 128), (16384, 128, 1))
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
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((64000, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear, ], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:1
            extern_kernels.mm(reinterpret_tensor(primals_1, (64000, 512), (512, 1), 0), reinterpret_tensor(primals_2, (512, 128), (1, 512), 0), out=buf0)
            buf1 = empty_strided_cuda((5, ), (1, ), torch.int64)
            # Topologically Sorted Source Nodes: [], Original ATen: []
            # [Provenance debug handles] aten.randint.low_out:2
            aten.randint.low_out(-9223372036854775808, 9223372036854775807, [5], out=buf1)
            buf3 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.bool)
            buf7 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.float32)
            buf51 = empty_strided_cuda((500, 128, 1), (128, 1, 1), torch.float32)
            buf8 = empty_strided_cuda((128, 500, 128), (64000, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear, inductor_random, dropout, iadd, layer_norm, transpose, multi_head_attention_forward, div_4], Original ATen: [aten.addmm, aten.view, aten.native_dropout, aten.add, aten.native_layer_norm, aten.transpose, aten.clone, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_add_addmm_clone_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_0:3
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_clone_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_0.run(buf1, primals_3, buf0, primals_4, primals_5, primals_6, buf3, buf7, buf51, buf8, 0, 64000, 128, stream=stream0)
            del primals_6
            buf9 = empty_strided_cuda((64000, 384), (384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [layer_norm, transpose, multi_head_attention_forward], Original ATen: [aten.native_layer_norm, aten.transpose, aten.t, aten.clone, aten._unsafe_view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:4
            extern_kernels.mm(reinterpret_tensor(buf8, (64000, 128), (128, 1), 0), reinterpret_tensor(primals_7, (128, 384), (1, 128), 0), out=buf9)
            buf10 = empty_strided_cuda((3, 128, 500, 128), (8192000, 64000, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_1:5
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_1.run(buf9, primals_8, buf10, 24576000, stream=stream0)
            del buf9
            del primals_8
            buf11 = empty_strided_cuda((4000, 128, 16), (16, 64000, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2:6
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2.run(buf10, buf11, 8192000, stream=stream0)
            buf12 = empty_strided_cuda((4000, 16, 128), (16, 1, 64000), torch.float32)
            buf50 = empty_strided_cuda((4000, 128, 16), (16, 64000, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward, permute_29], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.expand, aten.mul, aten.baddbmm]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_baddbmm_clone_expand_mul_select_squeeze_transpose_unsqueeze_view_3:7
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_baddbmm_clone_expand_mul_select_squeeze_transpose_unsqueeze_view_3.run(buf10, buf12, buf50, 8192000, stream=stream0)
            buf13 = empty_strided_cuda((4000, 128, 128), (16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view, aten.expand, aten.clone, aten._unsafe_view, aten.add]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_clone_expand_view_4:8
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_expand_view_4.run(primals_12, primals_11, buf13, 65536000, stream=stream0)
            del primals_11
            del primals_12
            buf14 = empty_strided_cuda((4000, 128, 128), (16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.expand, aten.mul, aten.baddbmm]
            # [Provenance debug handles] extern_kernels.baddbmm:9
            extern_kernels.baddbmm(buf13, buf11, buf12, alpha=1, beta=1, out=buf14)
            buf15 = empty_strided_cuda((4000, 128, 1), (128, 1, 1), torch.float32)
            buf16 = empty_strided_cuda((4000, 128, 1), (128, 1, 1), torch.float32)
            buf18 = empty_strided_cuda((4000, 128, 128), (16384, 128, 1), torch.bool)
            buf19 = buf13; del buf13  # reuse
            # Topologically Sorted Source Nodes: [, multi_head_attention_forward, inductor_random], Original ATen: [prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten.native_dropout]
            # [Provenance debug handles] triton_per_fused__softmax_exp_native_dropout_prepare_softmax_online_sub_5:10
            stream0 = get_raw_stream(0)
            triton_per_fused__softmax_exp_native_dropout_prepare_softmax_online_sub_5.run(buf14, buf1, buf15, buf16, buf18, buf19, 1, 512000, 128, stream=stream0)
            buf20 = reinterpret_tensor(buf12, (4000, 128, 16), (16, 64000, 1), 0); del buf12  # reuse
            buf49 = empty_strided_cuda((4000, 16, 128), (16, 1, 64000), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward, permute_28], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.bmm]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_6:11
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_6.run(buf10, buf20, buf49, 8192000, stream=stream0)
            del buf10
            buf21 = empty_strided_cuda((4000, 128, 16), (2048, 16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:12
            extern_kernels.bmm(buf19, buf20, out=buf21)
            buf22 = reinterpret_tensor(buf20, (128, 4000, 16), (64000, 16, 1), 0); del buf20  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.transpose, aten.clone]
            # [Provenance debug handles] triton_poi_fused_clone_transpose_7:13
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_7.run(buf21, buf22, 8192000, stream=stream0)
            buf23 = reinterpret_tensor(buf21, (64000, 128), (128, 1), 0); del buf21  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:14
            extern_kernels.addmm(primals_10, reinterpret_tensor(buf22, (64000, 128), (128, 1), 0), reinterpret_tensor(primals_9, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf23)
            del primals_10
            buf25 = empty_strided_cuda((500, 128, 1), (1, 500, 64000), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward, transpose_1, layer_norm_1], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_transpose_view_8:15
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_transpose_view_8.run(buf23, buf25, 64000, 128, stream=stream0)
            buf27 = empty_strided_cuda((500, 128, 1), (128, 1, 1), torch.float32)
            buf24 = empty_strided_cuda((500, 128, 1), (128, 1, 1), torch.float32)
            buf29 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.bool)
            buf30 = reinterpret_tensor(buf0, (500, 128, 128), (16384, 128, 1), 0); del buf0  # reuse
            buf34 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.float32)
            buf35 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.float32)
            buf48 = empty_strided_cuda((500, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear, dropout, iadd, multi_head_attention_forward, transpose_1, layer_norm_1, inductor_random, dropout_1, iadd_1, layer_norm_2, div_2], Original ATen: [aten.addmm, aten.view, aten.native_dropout, aten.add, aten.transpose, aten.native_layer_norm, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_red_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_9:16
            stream0 = get_raw_stream(0)
            triton_red_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_9.run(buf30, buf25, buf23, buf1, primals_13, primals_14, buf3, primals_3, primals_4, primals_15, primals_16, buf27, buf24, buf29, buf34, buf35, buf48, 2, 64000, 128, stream=stream0)
            del primals_14
            del primals_16
            del primals_3
            del primals_4
            buf36 = empty_strided_cuda((64000, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [layer_norm_2, linear_1], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:17
            extern_kernels.addmm(primals_18, reinterpret_tensor(buf35, (64000, 128), (128, 1), 0), reinterpret_tensor(primals_17, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf36)
            del primals_18
            buf38 = empty_strided_cuda((500, 128, 512), (65536, 512, 1), torch.bool)
            buf39 = reinterpret_tensor(buf25, (500, 128, 1), (128, 1, 1), 0); del buf25  # reuse
            buf40 = empty_strided_cuda((500, 128, 1), (128, 1, 64000), torch.float32)
            buf42 = reinterpret_tensor(buf40, (500, 128, 1), (128, 1, 1), 0); del buf40  # reuse
            buf43 = empty_strided_cuda((500, 128, 512), (65536, 512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_1, gelu, , inductor_random, dropout_2, layer_norm_3], Original ATen: [aten.view, aten.gelu, aten.native_dropout, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_gelu_native_dropout_native_layer_norm_view_10:18
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_native_dropout_native_layer_norm_view_10.run(buf42, buf1, buf36, primals_19, primals_20, buf38, buf39, buf43, 3, 64000, 512, stream=stream0)
            del primals_20
            buf44 = empty_strided_cuda((64000, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_1, gelu, dropout_2, layer_norm_3, linear_2, ], Original ATen: [aten.view, aten.gelu, aten.native_dropout, aten.native_layer_norm, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:19
            extern_kernels.mm(reinterpret_tensor(buf43, (64000, 512), (512, 1), 0), reinterpret_tensor(primals_21, (512, 128), (1, 512), 0), out=buf44)
            buf46 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.bool)
            buf47 = reinterpret_tensor(buf44, (500, 128, 128), (16384, 128, 1), 0); del buf44  # reuse
            # Topologically Sorted Source Nodes: [, linear_2, inductor_random, dropout_3, iadd_2], Original ATen: [aten.addmm, aten.view, aten.native_dropout, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_addmm_native_dropout_view_11:20
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_native_dropout_view_11.run(buf47, buf1, primals_22, buf30, buf46, 4, 8192000, stream=stream0)
            del buf1
            del buf30
            del primals_22
        return (buf47, primals_2, primals_5, primals_7, primals_9, primals_13, primals_15, primals_17, primals_19, primals_21, reinterpret_tensor(primals_1, (64000, 512), (512, 1), 0), buf3, buf7, reinterpret_tensor(buf8, (64000, 128), (128, 1), 0), buf14, buf15, buf16, buf18, reinterpret_tensor(buf22, (64000, 128), (128, 1), 0), buf23, buf24, buf27, buf29, buf34, reinterpret_tensor(buf35, (64000, 128), (128, 1), 0), buf36, buf38, buf39, buf42, reinterpret_tensor(buf43, (64000, 512), (512, 1), 0), buf46, buf48, reinterpret_tensor(buf19, (4000, 128, 128), (16384, 1, 128), 0), buf49, buf50, reinterpret_tensor(buf11, (4000, 16, 128), (16, 1, 64000), 0), buf51, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((500, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((500, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((500, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4000, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
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
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
