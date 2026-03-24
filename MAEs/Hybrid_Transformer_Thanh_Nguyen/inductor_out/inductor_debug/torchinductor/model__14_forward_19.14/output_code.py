# AOT ID: ['14_forward']
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


# kernel path: /traces/inductor_cache/hr/chrvtgfacx7ozd3sbx2z3cme43ifnqzozwg6yrcibx6ldxniamxk.py
# Topologically Sorted Source Nodes: [linear, layer_norm, transpose, multi_head_attention_forward], Original ATen: [aten.view, aten.native_layer_norm, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   layer_norm => add, add_1, mul, mul_1, rsqrt, sub, var_mean
#   linear => view_2
#   multi_head_attention_forward => clone
#   transpose => permute_1
# Graph fragment:
#   %addmm : Tensor "f32[64000, 128][128, 1]cuda:0" = PlaceHolder[target=addmm]
#   %buf2 : Tensor "f32[500, 128, 1][128, 1, 64000]cuda:0" = PlaceHolder[target=buf2]
#   %getitem_1 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=getitem_1]
#   %rsqrt : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %primals_4 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_4]
#   %primals_5 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_5]
#   %view_2 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [500, 128, 128]), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_2, [2]), kwargs = {correction: 0, keepdim: True})
#   %add : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %sub : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_2, %getitem_1), kwargs = {})
#   %mul : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %primals_4), kwargs = {})
#   %add_1 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %primals_5), kwargs = {})
#   %permute_1 : Tensor "f32[128, 500, 128][128, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_1, [1, 0, 2]), kwargs = {})
#   %clone : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_1,), kwargs = {memory_format: torch.contiguous_format})
#   return %getitem_1,%buf2,%rsqrt,%clone
triton_per_fused_clone_native_layer_norm_transpose_view_0 = async_compile.triton('triton_per_fused_clone_native_layer_norm_transpose_view_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clone_native_layer_norm_transpose_view_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 3, 'num_reduction': 4, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1024000, 'r0_': 98305024}}
)
@triton.jit
def triton_per_fused_clone_native_layer_norm_transpose_view_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, xmask)
    tl.store(out_ptr1 + (r0_1 + 128*x3 + 64000*x2), tmp27, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/l5/cl5j7vp4nqavqvxkj5xenqkcx46hc6rehraqup6ekokbixxotarf.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => add_2, clone_1, permute_3, squeeze, unsqueeze, view_4, view_5
# Graph fragment:
#   %mm : Tensor "f32[64000, 384][384, 1]cuda:0" = PlaceHolder[target=mm]
#   %primals_7 : Tensor "f32[384][1]cuda:0" = PlaceHolder[target=primals_7]
#   %view_4 : Tensor "f32[128, 500, 384][192000, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [128, 500, 384]), kwargs = {})
#   %add_2 : Tensor "f32[128, 500, 384][192000, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_4, %primals_7), kwargs = {})
#   %view_5 : Tensor "f32[128, 500, 3, 128][192000, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_2, [128, 500, 3, 128]), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 128, 500, 3, 128][24576000, 192000, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_5, 0), kwargs = {})
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
#   multi_head_attention_forward => add_2, clone_1, mul_2, permute_3, permute_4, select, squeeze, unsqueeze, view_4, view_5, view_6
# Graph fragment:
#   %clone_1 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0" = PlaceHolder[target=clone_1]
#   %view_4 : Tensor "f32[128, 500, 384][192000, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [128, 500, 384]), kwargs = {})
#   %add_2 : Tensor "f32[128, 500, 384][192000, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_4, %primals_7), kwargs = {})
#   %view_5 : Tensor "f32[128, 500, 3, 128][192000, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_2, [128, 500, 3, 128]), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 128, 500, 3, 128][24576000, 192000, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_5, 0), kwargs = {})
#   %permute_3 : Tensor "f32[3, 128, 500, 1, 128][128, 192000, 384, 24576000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze : Tensor "f32[3, 128, 500, 128][128, 192000, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_3, -2), kwargs = {})
#   %clone_1 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
#   %select : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_1, 0, 0), kwargs = {})
#   %view_6 : Tensor "f32[128, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select, [128, 4000, 16]), kwargs = {})
#   %permute_4 : Tensor "f32[4000, 128, 16][16, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_6, [1, 0, 2]), kwargs = {})
#   %mul_2 : Tensor "f32[4000, 128, 16][16, 64000, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_4, 0.25), kwargs = {})
#   return %mul_2
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


# kernel path: /traces/inductor_cache/4h/c4hwfapesysdx4gj5aycix6oedvrl3eflx6bbpn4xij74bfslr52.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.expand, aten.mul, aten.baddbmm]
# Source node to ATen node mapping:
#   multi_head_attention_forward => add_2, add_3, baddbmm, clone_1, clone_2, expand, mul_2, permute_3, permute_4, permute_5, permute_7, select, select_1, squeeze, unsqueeze, view_10, view_4, view_5, view_6, view_7, view_9
# Graph fragment:
#   %clone_1 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0" = PlaceHolder[target=clone_1]
#   %view_4 : Tensor "f32[128, 500, 384][192000, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [128, 500, 384]), kwargs = {})
#   %add_2 : Tensor "f32[128, 500, 384][192000, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_4, %primals_7), kwargs = {})
#   %view_5 : Tensor "f32[128, 500, 3, 128][192000, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_2, [128, 500, 3, 128]), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 128, 500, 3, 128][24576000, 192000, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_5, 0), kwargs = {})
#   %permute_3 : Tensor "f32[3, 128, 500, 1, 128][128, 192000, 384, 24576000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze : Tensor "f32[3, 128, 500, 128][128, 192000, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_3, -2), kwargs = {})
#   %clone_1 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
#   %select : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_1, 0, 0), kwargs = {})
#   %select_1 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_1, 0, 1), kwargs = {})
#   %view_6 : Tensor "f32[128, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select, [128, 4000, 16]), kwargs = {})
#   %permute_4 : Tensor "f32[4000, 128, 16][16, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_6, [1, 0, 2]), kwargs = {})
#   %view_7 : Tensor "f32[128, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_1, [128, 4000, 16]), kwargs = {})
#   %permute_5 : Tensor "f32[4000, 128, 16][16, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_7, [1, 0, 2]), kwargs = {})
#   %view_9 : Tensor "f32[500, 1, 1, 128][128, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_10, [500, 1, 1, 128]), kwargs = {})
#   %expand : Tensor "f32[500, 8, 1, 128][128, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%view_9, [-1, 8, -1, -1]), kwargs = {})
#   %clone_2 : Tensor "f32[500, 8, 1, 128][1024, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
#   %view_10 : Tensor "f32[4000, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_2, [4000, 1, 128]), kwargs = {})
#   %add_3 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_11, %view_10), kwargs = {})
#   %mul_2 : Tensor "f32[4000, 128, 16][16, 64000, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_4, 0.25), kwargs = {})
#   %permute_7 : Tensor "f32[4000, 16, 128][16, 1, 64000]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%permute_5, [0, 2, 1]), kwargs = {})
#   %baddbmm : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.baddbmm.default](args = (%add_3, %mul_2, %permute_7), kwargs = {})
#   return %buf9
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_baddbmm_clone_expand_mul_select_squeeze_transpose_unsqueeze_view_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 98304000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_baddbmm_clone_expand_mul_select_squeeze_transpose_unsqueeze_view_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (8192000 + x2), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/4o/c4owupynmtiw2f6y64gbaomfbmgbqz3ei45fchcgo7x2do5owbmr.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.bmm]
# Source node to ATen node mapping:
#   multi_head_attention_forward => add_2, bmm, clone_1, permute_3, permute_6, select_2, squeeze, unsqueeze, view_4, view_5, view_8
# Graph fragment:
#   %clone_1 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0" = PlaceHolder[target=clone_1]
#   %view_4 : Tensor "f32[128, 500, 384][192000, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [128, 500, 384]), kwargs = {})
#   %add_2 : Tensor "f32[128, 500, 384][192000, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_4, %primals_7), kwargs = {})
#   %view_5 : Tensor "f32[128, 500, 3, 128][192000, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_2, [128, 500, 3, 128]), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 128, 500, 3, 128][24576000, 192000, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_5, 0), kwargs = {})
#   %permute_3 : Tensor "f32[3, 128, 500, 1, 128][128, 192000, 384, 24576000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze : Tensor "f32[3, 128, 500, 128][128, 192000, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_3, -2), kwargs = {})
#   %clone_1 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
#   %select_2 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_1, 0, 2), kwargs = {})
#   %view_8 : Tensor "f32[128, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_2, [128, 4000, 16]), kwargs = {})
#   %permute_6 : Tensor "f32[4000, 128, 16][16, 64000, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_8, [1, 0, 2]), kwargs = {})
#   %bmm : Tensor "f32[4000, 128, 16][2048, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%mul_4, %permute_6), kwargs = {})
#   return %buf18
triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_4 = async_compile.triton('triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_4', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 98304000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (16384000 + x2), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/ge/cge6qepj6xb6xhkqft6bqsijns66t2b6fbflyq7zqougldnqr5sz.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view, aten.expand, aten.clone, aten._unsafe_view, aten.add]
# Source node to ATen node mapping:
#   multi_head_attention_forward => add_3, clone_2, expand, view_10, view_9
# Graph fragment:
#   %primals_11 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=primals_11]
#   %primals_10 : Tensor "f32[500, 128][128, 1]cuda:0" = PlaceHolder[target=primals_10]
#   %view_9 : Tensor "f32[500, 1, 1, 128][128, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_10, [500, 1, 1, 128]), kwargs = {})
#   %expand : Tensor "f32[500, 8, 1, 128][128, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%view_9, [-1, 8, -1, -1]), kwargs = {})
#   %clone_2 : Tensor "f32[500, 8, 1, 128][1024, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
#   %view_10 : Tensor "f32[4000, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_2, [4000, 1, 128]), kwargs = {})
#   %add_3 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_11, %view_10), kwargs = {})
#   return %add_3
triton_poi_fused__unsafe_view_add_clone_expand_view_5 = async_compile.triton('triton_poi_fused__unsafe_view_add_clone_expand_view_5', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_clone_expand_view_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 786688000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_clone_expand_view_5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /traces/inductor_cache/ci/ccimmbm3ahz7qkanbeg3ki5u7c5soesigyar5t4jyyymibb3yp6m.py
# Topologically Sorted Source Nodes: [, multi_head_attention_forward, inductor_random], Original ATen: [prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten.native_dropout]
# Source node to ATen node mapping:
#    => exp_default_6, inductor_lookup_seed_default, prepare_softmax_online_default_6, sub_tensor_6
#   inductor_random => inductor_random_default_26
#   multi_head_attention_forward => div, gt, mul_3, mul_4
# Graph fragment:
#   %baddbmm : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=baddbmm]
#   %inductor_seeds_default : Tensor "i64[27][1]cuda:0" = PlaceHolder[target=inductor_seeds_default]
#   %inductor_random_default_26 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=inductor_random_default_26]
#   %gt : Tensor "b8[4000, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=gt]
#   %getitem_68 : Tensor "f32[4000, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=getitem_68]
#   %getitem_69 : Tensor "f32[4000, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=getitem_69]
#   %prepare_softmax_online_default_6 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%baddbmm, -1), kwargs = {})
#   %sub_tensor_6 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%baddbmm, %getitem_68), kwargs = {})
#   %exp_default_6 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_6,), kwargs = {})
#   %div : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_6, %getitem_69), kwargs = {})
#   %inductor_lookup_seed_default : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 0), kwargs = {})
#   %inductor_random_default_26 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([4000, 128, 128], %inductor_lookup_seed_default, rand), kwargs = {})
#   %gt : Tensor "b8[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%inductor_random_default_26, 0.1), kwargs = {})
#   %mul_3 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt, %div), kwargs = {})
#   %mul_4 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, 1.1111111111111112), kwargs = {})
#   return %getitem_68,%getitem_69,%inductor_random_default_26,%gt,%mul_4
triton_per_fused__softmax_exp_native_dropout_prepare_softmax_online_sub_6 = async_compile.triton('triton_per_fused__softmax_exp_native_dropout_prepare_softmax_online_sub_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr3': '*i1', 'out_ptr4': '*fp32', 'load_seed_offset': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_exp_native_dropout_prepare_softmax_online_sub_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 4, 'num_reduction': 4, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 8192000, 'r0_': 917504000}}
)
@triton.jit
def triton_per_fused__softmax_exp_native_dropout_prepare_softmax_online_sub_6(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr3, out_ptr4, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
#   multi_head_attention_forward => view_12
#   transpose_1 => permute_10
# Graph fragment:
#   %addmm_1 : Tensor "f32[64000, 128][128, 1]cuda:0" = PlaceHolder[target=addmm_1]
#   %view_12 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_1, [128, 500, 128]), kwargs = {})
#   %permute_10 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_12, [1, 0, 2]), kwargs = {})
#   %clone_4 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_10,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_4, [2]), kwargs = {correction: 0, keepdim: True})
#   return %buf23
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


# kernel path: /traces/inductor_cache/c5/cc5tml4qbpj3kx3opp45rtxilj7de7wpuajfcsspyndjypry4cf6.py
# Topologically Sorted Source Nodes: [linear, multi_head_attention_forward, transpose_1, layer_norm_1, , inductor_random, dropout, iadd, layer_norm_2, div_32], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm, aten.native_dropout, aten.add, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#    => inductor_lookup_seed_default_1
#   div_32 => div_32
#   dropout => gt_1, mul_7, mul_8
#   iadd => add_6
#   inductor_random => inductor_random_default_25
#   layer_norm_1 => add_4, add_5, clone_4, mul_5, mul_6, rsqrt_1, sub_2, var_mean_1
#   layer_norm_2 => add_7, add_8, mul_10, mul_9, rsqrt_2, sub_3, var_mean_2
#   linear => view_2
#   multi_head_attention_forward => view_12
#   transpose_1 => permute_10
# Graph fragment:
#   %buf23 : Tensor "f32[500, 128, 1][1, 500, 64000]cuda:0" = PlaceHolder[target=buf23]
#   %addmm_1 : Tensor "f32[64000, 128][128, 1]cuda:0" = PlaceHolder[target=addmm_1]
#   %inductor_seeds_default : Tensor "i64[27][1]cuda:0" = PlaceHolder[target=inductor_seeds_default]
#   %inductor_random_default_25 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=inductor_random_default_25]
#   %gt_1 : Tensor "b8[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=gt_1]
#   %getitem_3 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=getitem_3]
#   %rsqrt_1 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_1]
#   %primals_12 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_12]
#   %primals_13 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_13]
#   %addmm : Tensor "f32[64000, 128][128, 1]cuda:0" = PlaceHolder[target=addmm]
#   %add_6 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=add_6]
#   %getitem_5 : Tensor "f32[500, 128, 1][128, 1, 64000]cuda:0" = PlaceHolder[target=getitem_5]
#   %buf30 : Tensor "f32[500, 128, 1][128, 1, 64000]cuda:0" = PlaceHolder[target=buf30]
#   %mul_9 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=mul_9]
#   %primals_14 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_14]
#   %primals_15 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_15]
#   %view_2 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [500, 128, 128]), kwargs = {})
#   %view_12 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_1, [128, 500, 128]), kwargs = {})
#   %permute_10 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_12, [1, 0, 2]), kwargs = {})
#   %clone_4 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_10,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_4, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_4 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %sub_2 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_4, %getitem_3), kwargs = {})
#   %mul_5 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_6 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %primals_12), kwargs = {})
#   %add_5 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %primals_13), kwargs = {})
#   %inductor_lookup_seed_default_1 : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 1), kwargs = {})
#   %inductor_random_default_25 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([500, 128, 128], %inductor_lookup_seed_default_1, rand), kwargs = {})
#   %gt_1 : Tensor "b8[500, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%inductor_random_default_25, 0.1), kwargs = {})
#   %mul_7 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt_1, %add_5), kwargs = {})
#   %mul_8 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, 1.1111111111111112), kwargs = {})
#   %add_6 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %view_2), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_6, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_7 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt_2 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_7,), kwargs = {})
#   %sub_3 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_6, %getitem_5), kwargs = {})
#   %mul_9 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %mul_10 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %primals_14), kwargs = {})
#   %add_8 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %primals_15), kwargs = {})
#   %div_32 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_2, 128), kwargs = {})
#   return %rsqrt_1,%getitem_3,%inductor_random_default_25,%gt_1,%add_6,%getitem_5,%buf30,%mul_9,%add_8,%div_32
triton_red_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_9 = async_compile.triton('triton_red_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr3': '*i1', 'out_ptr4': '*fp32', 'out_ptr7': '*fp32', 'out_ptr8': '*fp32', 'out_ptr9': '*fp32', 'load_seed_offset': 'constexpr', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'load_seed_offset': 1}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 10, 'num_store': 7, 'num_reduction': 3, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 512000, 'r0_': 278530048}}
)
@triton.jit
def triton_red_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr3, out_ptr4, out_ptr7, out_ptr8, out_ptr9, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp34_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp34_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp34_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
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
        tmp31 = tl.load(in_ptr5 + (r0_2 + 128*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tmp19.to(tl.float32)
        tmp22 = tmp21 - tmp8
        tmp23 = tmp22 * tmp5
        tmp25 = tmp23 * tmp24
        tmp27 = tmp25 + tmp26
        tmp28 = tmp20 * tmp27
        tmp29 = 1.1111111111111112
        tmp30 = tmp28 * tmp29
        tmp32 = tmp30 + tmp31
        tmp33 = tl.broadcast_to(tmp32, [XBLOCK, R0_BLOCK])
        tmp34_mean_next, tmp34_m2_next, tmp34_weight_next = triton_helpers.welford_reduce(
            tmp33, tmp34_mean, tmp34_m2, tmp34_weight, roffset == 0
        )
        tmp34_mean = tl.where(r0_mask & xmask, tmp34_mean_next, tmp34_mean)
        tmp34_m2 = tl.where(r0_mask & xmask, tmp34_m2_next, tmp34_m2)
        tmp34_weight = tl.where(r0_mask & xmask, tmp34_weight_next, tmp34_weight)
        tl.store(out_ptr4 + (r0_2 + 128*x3), tmp32, r0_mask & xmask)
    tmp35, tmp36, tmp37 = triton_helpers.welford(tmp34_mean, tmp34_m2, tmp34_weight, 1)
    tmp34 = tmp35[:, None]
    tmp38 = tmp36[:, None]
    tmp39 = tmp37[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp40 = tl.load(out_ptr4 + (r0_2 + 128*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp48 = tl.load(in_ptr6 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp50 = tl.load(in_ptr7 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp41 = tmp40 - tmp34
        tmp42 = 128.0
        tmp43 = (tmp38 / tmp42)
        tmp44 = 1e-05
        tmp45 = tmp43 + tmp44
        tmp46 = libdevice.rsqrt(tmp45)
        tmp47 = tmp41 * tmp46
        tmp49 = tmp47 * tmp48
        tmp51 = tmp49 + tmp50
        tl.store(out_ptr7 + (r0_2 + 128*x3), tmp47, r0_mask & xmask)
        tl.store(out_ptr8 + (r0_2 + 128*x3), tmp51, r0_mask & xmask)
    tmp52 = 128.0
    tmp53 = (tmp38 / tmp52)
    tmp54 = 1e-05
    tmp55 = tmp53 + tmp54
    tmp56 = libdevice.rsqrt(tmp55)
    tmp57 = 0.0078125
    tmp58 = tmp56 * tmp57
    tl.store(out_ptr9 + (x3), tmp58, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/wp/cwpossgbvevto4tujzvs4wbjf3ij3vvqp32ozisw47g6ariz7a3n.py
# Topologically Sorted Source Nodes: [linear_1, gelu, , inductor_random, dropout_1, layer_norm_3], Original ATen: [aten.view, aten.gelu, aten.native_dropout, aten.native_layer_norm]
# Source node to ATen node mapping:
#    => inductor_lookup_seed_default_2
#   dropout_1 => gt_2, mul_14, mul_15
#   gelu => add_9, erf, mul_11, mul_12, mul_13
#   inductor_random => inductor_random_default_24
#   layer_norm_3 => add_10, add_11, mul_16, mul_17, rsqrt_3, sub_4, var_mean_3
#   linear_1 => view_15
# Graph fragment:
#   %inductor_seeds_default : Tensor "i64[27][1]cuda:0" = PlaceHolder[target=inductor_seeds_default]
#   %inductor_random_default_24 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0" = PlaceHolder[target=inductor_random_default_24]
#   %gt_2 : Tensor "b8[500, 128, 512][65536, 512, 1]cuda:0" = PlaceHolder[target=gt_2]
#   %addmm_2 : Tensor "f32[64000, 512][512, 1]cuda:0" = PlaceHolder[target=addmm_2]
#   %buf38 : Tensor "f32[500, 128, 1][128, 1, 64000]cuda:0" = PlaceHolder[target=buf38]
#   %getitem_7 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=getitem_7]
#   %rsqrt_3 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_3]
#   %primals_18 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_18]
#   %primals_19 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_19]
#   %view_15 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_2, [500, 128, 512]), kwargs = {})
#   %mul_11 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_15, 0.5), kwargs = {})
#   %mul_12 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_15, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_12,), kwargs = {})
#   %add_9 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_13 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_11, %add_9), kwargs = {})
#   %inductor_lookup_seed_default_2 : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 2), kwargs = {})
#   %inductor_random_default_24 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([500, 128, 512], %inductor_lookup_seed_default_2, rand), kwargs = {})
#   %gt_2 : Tensor "b8[500, 128, 512][65536, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%inductor_random_default_24, 0.1), kwargs = {})
#   %mul_14 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt_2, %mul_13), kwargs = {})
#   %mul_15 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, 1.1111111111111112), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%mul_15, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_10 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_3 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_10,), kwargs = {})
#   %sub_4 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_15, %getitem_7), kwargs = {})
#   %mul_16 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_3), kwargs = {})
#   %mul_17 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %primals_18), kwargs = {})
#   %add_11 : Tensor "f32[500, 128, 512][65536, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %primals_19), kwargs = {})
#   return %inductor_random_default_24,%gt_2,%getitem_7,%buf38,%rsqrt_3,%add_11
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


# kernel path: /traces/inductor_cache/tc/ctcrtn4awrza7hoe5joxhbbzwpzig3fywi6qrodjvyv3mpuhtjw5.py
# Topologically Sorted Source Nodes: [, linear_2, inductor_random, dropout_2, iadd_1, layer_norm_4, transpose_2, multi_head_attention_forward_1, div_30], Original ATen: [aten.addmm, aten.view, aten.native_dropout, aten.add, aten.native_layer_norm, aten.transpose, aten.clone, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#    => add_tensor_5, inductor_lookup_seed_default_3
#   div_30 => div_30
#   dropout_2 => gt_3, mul_18, mul_19
#   iadd_1 => add_12
#   inductor_random => inductor_random_default_23
#   layer_norm_4 => add_13, add_14, mul_20, mul_21, rsqrt_4, sub_5, var_mean_4
#   linear_2 => view_17
#   multi_head_attention_forward_1 => clone_5
#   transpose_2 => permute_13
# Graph fragment:
#   %inductor_seeds_default : Tensor "i64[27][1]cuda:0" = PlaceHolder[target=inductor_seeds_default]
#   %inductor_random_default_23 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=inductor_random_default_23]
#   %gt_3 : Tensor "b8[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=gt_3]
#   %primals_21 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_21]
#   %mm_default_5 : Tensor "f32[64000, 128][128, 1]cuda:0" = PlaceHolder[target=mm_default_5]
#   %add_6 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=add_6]
#   %getitem_9 : Tensor "f32[500, 128, 1][128, 1, 64000]cuda:0" = PlaceHolder[target=getitem_9]
#   %buf46 : Tensor "f32[500, 128, 1][128, 1, 64000]cuda:0" = PlaceHolder[target=buf46]
#   %mul_20 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=mul_20]
#   %primals_22 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_22]
#   %primals_23 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_23]
#   %add_tensor_5 : Tensor "f32[64000, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_21, %mm_default_5), kwargs = {})
#   %view_17 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_5, [500, 128, 128]), kwargs = {})
#   %inductor_lookup_seed_default_3 : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 3), kwargs = {})
#   %inductor_random_default_23 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([500, 128, 128], %inductor_lookup_seed_default_3, rand), kwargs = {})
#   %gt_3 : Tensor "b8[500, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%inductor_random_default_23, 0.1), kwargs = {})
#   %mul_18 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt_3, %view_17), kwargs = {})
#   %mul_19 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, 1.1111111111111112), kwargs = {})
#   %add_12 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %add_6), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_12, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_13 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_4 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_13,), kwargs = {})
#   %sub_5 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_12, %getitem_9), kwargs = {})
#   %mul_20 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_4), kwargs = {})
#   %mul_21 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %primals_22), kwargs = {})
#   %add_14 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_21, %primals_23), kwargs = {})
#   %permute_13 : Tensor "f32[128, 500, 128][128, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_14, [1, 0, 2]), kwargs = {})
#   %clone_5 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_13,), kwargs = {memory_format: torch.contiguous_format})
#   %div_30 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_4, 128), kwargs = {})
#   return %inductor_random_default_23,%gt_3,%getitem_9,%buf46,%mul_20,%div_30,%clone_5
triton_per_fused_add_addmm_clone_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_11 = async_compile.triton('triton_per_fused_add_addmm_clone_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_11', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_clone_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 4, 'num_reduction': 4, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 512000, 'r0_': 212993536}}
)
@triton.jit
def triton_per_fused_add_addmm_clone_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr4, out_ptr5, out_ptr6, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /traces/inductor_cache/4n/c4nfxckctam3z4cpft7lfxai3d67jasg5sox4bb43nioylezcgj7.py
# Topologically Sorted Source Nodes: [, linear_2, dropout_2, iadd_1, multi_head_attention_forward_1, transpose_3, layer_norm_5, inductor_random, dropout_3, iadd_2, layer_norm_6, div_28], Original ATen: [aten.addmm, aten.view, aten.native_dropout, aten.add, aten.transpose, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#    => add_tensor_5, inductor_lookup_seed_default_5
#   div_28 => div_28
#   dropout_2 => mul_18, mul_19
#   dropout_3 => gt_5, mul_27, mul_28
#   iadd_1 => add_12
#   iadd_2 => add_19
#   inductor_random => inductor_random_default_21
#   layer_norm_5 => add_17, add_18, clone_9, mul_25, mul_26, rsqrt_5, sub_7, var_mean_5
#   layer_norm_6 => add_20, add_21, mul_29, mul_30, rsqrt_6, sub_8, var_mean_6
#   linear_2 => view_17
#   multi_head_attention_forward_1 => view_27
#   transpose_3 => permute_22
# Graph fragment:
#   %buf65 : Tensor "f32[500, 128, 1][1, 500, 64000]cuda:0" = PlaceHolder[target=buf65]
#   %addmm_4 : Tensor "f32[64000, 128][128, 1]cuda:0" = PlaceHolder[target=addmm_4]
#   %inductor_seeds_default : Tensor "i64[27][1]cuda:0" = PlaceHolder[target=inductor_seeds_default]
#   %inductor_random_default_21 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=inductor_random_default_21]
#   %gt_5 : Tensor "b8[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=gt_5]
#   %getitem_11 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=getitem_11]
#   %rsqrt_5 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_5]
#   %primals_28 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_28]
#   %primals_29 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_29]
#   %gt_3 : Tensor "b8[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=gt_3]
#   %primals_21 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_21]
#   %mm_default_5 : Tensor "f32[64000, 128][128, 1]cuda:0" = PlaceHolder[target=mm_default_5]
#   %add_6 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=add_6]
#   %add_19 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=add_19]
#   %getitem_13 : Tensor "f32[500, 128, 1][128, 1, 64000]cuda:0" = PlaceHolder[target=getitem_13]
#   %buf72 : Tensor "f32[500, 128, 1][128, 1, 64000]cuda:0" = PlaceHolder[target=buf72]
#   %mul_29 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=mul_29]
#   %primals_30 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_30]
#   %primals_31 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_31]
#   %add_tensor_5 : Tensor "f32[64000, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_21, %mm_default_5), kwargs = {})
#   %view_17 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_5, [500, 128, 128]), kwargs = {})
#   %mul_18 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt_3, %view_17), kwargs = {})
#   %mul_19 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, 1.1111111111111112), kwargs = {})
#   %add_12 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %add_6), kwargs = {})
#   %view_27 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_4, [128, 500, 128]), kwargs = {})
#   %permute_22 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_27, [1, 0, 2]), kwargs = {})
#   %clone_9 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_22,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_9, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_17 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_10, 1e-05), kwargs = {})
#   %rsqrt_5 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_17,), kwargs = {})
#   %sub_7 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_9, %getitem_11), kwargs = {})
#   %mul_25 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %rsqrt_5), kwargs = {})
#   %mul_26 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %primals_28), kwargs = {})
#   %add_18 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %primals_29), kwargs = {})
#   %inductor_lookup_seed_default_5 : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 5), kwargs = {})
#   %inductor_random_default_21 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([500, 128, 128], %inductor_lookup_seed_default_5, rand), kwargs = {})
#   %gt_5 : Tensor "b8[500, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%inductor_random_default_21, 0.1), kwargs = {})
#   %mul_27 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt_5, %add_18), kwargs = {})
#   %mul_28 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_27, 1.1111111111111112), kwargs = {})
#   %add_19 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_28, %add_12), kwargs = {})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_19, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_20 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-05), kwargs = {})
#   %rsqrt_6 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_20,), kwargs = {})
#   %sub_8 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_19, %getitem_13), kwargs = {})
#   %mul_29 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt_6), kwargs = {})
#   %mul_30 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_29, %primals_30), kwargs = {})
#   %add_21 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_30, %primals_31), kwargs = {})
#   %div_28 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_6, 128), kwargs = {})
#   return %rsqrt_5,%getitem_11,%inductor_random_default_21,%gt_5,%add_19,%getitem_13,%buf72,%mul_29,%add_21,%div_28
triton_red_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_12 = async_compile.triton('triton_red_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_12', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 13, 'num_store': 7, 'num_reduction': 3, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 512000, 'r0_': 319490560}}
)
@triton.jit
def triton_red_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, out_ptr3, out_ptr6, out_ptr7, out_ptr8, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /traces/inductor_cache/lp/clp5276salpe35xqglqmxo7aogpnmkf3xqdwipxcssflrhkg45zb.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward_5, permute_123], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.bmm]
# Source node to ATen node mapping:
#   multi_head_attention_forward_5 => add_67, bmm_5, clone_26, permute_63, permute_66, select_17, squeeze_5, unsqueeze_5, view_79, view_80, view_83
#   permute_123 => permute_123
# Graph fragment:
#   %clone_26 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0" = PlaceHolder[target=clone_26]
#   %view_79 : Tensor "f32[128, 500, 384][192000, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_5, [128, 500, 384]), kwargs = {})
#   %add_67 : Tensor "f32[128, 500, 384][192000, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_79, %primals_89), kwargs = {})
#   %view_80 : Tensor "f32[128, 500, 3, 128][192000, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_67, [128, 500, 3, 128]), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[1, 128, 500, 3, 128][24576000, 192000, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_80, 0), kwargs = {})
#   %permute_63 : Tensor "f32[3, 128, 500, 1, 128][128, 192000, 384, 24576000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_5, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze_5 : Tensor "f32[3, 128, 500, 128][128, 192000, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_63, -2), kwargs = {})
#   %clone_26 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_5,), kwargs = {memory_format: torch.contiguous_format})
#   %select_17 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_26, 0, 2), kwargs = {})
#   %view_83 : Tensor "f32[128, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_17, [128, 4000, 16]), kwargs = {})
#   %permute_66 : Tensor "f32[4000, 128, 16][16, 64000, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%view_83, [1, 0, 2]), kwargs = {})
#   %bmm_5 : Tensor "f32[4000, 128, 16][2048, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%mul_104, %permute_66), kwargs = {})
#   %permute_123 : Tensor "f32[4000, 16, 128][16, 1, 64000]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_66, [0, 2, 1]), kwargs = {})
#   return %buf228,%permute_123
triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_13 = async_compile.triton('triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_13', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 163840000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_13(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (16384000 + x2), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
    tl.store(out_ptr1 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/e3/ce3rratcius2bqgpeghszwj3vwyf3vjcm57a32hd5zfydwkptiyh.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward_6, permute_96], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.mul, aten.baddbmm]
# Source node to ATen node mapping:
#   multi_head_attention_forward_6 => add_80, baddbmm_6, clone_31, mul_122, permute_75, permute_76, permute_77, permute_79, select_18, select_19, squeeze_6, unsqueeze_6, view_94, view_95, view_96, view_97
#   permute_96 => permute_96
# Graph fragment:
#   %clone_31 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0" = PlaceHolder[target=clone_31]
#   %view_94 : Tensor "f32[128, 500, 384][192000, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_6, [128, 500, 384]), kwargs = {})
#   %add_80 : Tensor "f32[128, 500, 384][192000, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_94, %primals_105), kwargs = {})
#   %view_95 : Tensor "f32[128, 500, 3, 128][192000, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_80, [128, 500, 3, 128]), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[1, 128, 500, 3, 128][24576000, 192000, 384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_95, 0), kwargs = {})
#   %permute_75 : Tensor "f32[3, 128, 500, 1, 128][128, 192000, 384, 24576000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_6, [3, 1, 2, 0, 4]), kwargs = {})
#   %squeeze_6 : Tensor "f32[3, 128, 500, 128][128, 192000, 384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%permute_75, -2), kwargs = {})
#   %clone_31 : Tensor "f32[3, 128, 500, 128][8192000, 64000, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_6,), kwargs = {memory_format: torch.contiguous_format})
#   %select_18 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_31, 0, 0), kwargs = {})
#   %select_19 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%clone_31, 0, 1), kwargs = {})
#   %view_96 : Tensor "f32[128, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_18, [128, 4000, 16]), kwargs = {})
#   %permute_76 : Tensor "f32[4000, 128, 16][16, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_96, [1, 0, 2]), kwargs = {})
#   %view_97 : Tensor "f32[128, 4000, 16][64000, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%select_19, [128, 4000, 16]), kwargs = {})
#   %permute_77 : Tensor "f32[4000, 128, 16][16, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_97, [1, 0, 2]), kwargs = {})
#   %mul_122 : Tensor "f32[4000, 128, 16][16, 64000, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_76, 0.25), kwargs = {})
#   %permute_79 : Tensor "f32[4000, 16, 128][16, 1, 64000]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%permute_77, [0, 2, 1]), kwargs = {})
#   %baddbmm_6 : Tensor "f32[4000, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.baddbmm.default](args = (%add_3, %mul_122, %permute_79), kwargs = {})
#   %permute_96 : Tensor "f32[4000, 128, 16][16, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_79, [0, 2, 1]), kwargs = {})
#   return %buf263,%permute_96
triton_poi_fused__unsafe_view_add_baddbmm_clone_mul_select_squeeze_transpose_unsqueeze_view_14 = async_compile.triton('triton_poi_fused__unsafe_view_add_baddbmm_clone_mul_select_squeeze_transpose_unsqueeze_view_14', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_baddbmm_clone_mul_select_squeeze_transpose_unsqueeze_view_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 163840000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_baddbmm_clone_mul_select_squeeze_transpose_unsqueeze_view_14(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (8192000 + x2), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
    tl.store(out_ptr1 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/ws/cwslj3piw3lqnzex2t6goy4oq5i37fnpthcjzhfhdkdnehibxezh.py
# Topologically Sorted Source Nodes: [, linear_12, dropout_17, iadd_11, multi_head_attention_forward_6, transpose_13, layer_norm_25, inductor_random, dropout_18, iadd_12, layer_norm_26], Original ATen: [aten.addmm, aten.view, aten.native_dropout, aten.add, aten.transpose, aten.native_layer_norm]
# Source node to ATen node mapping:
#    => add_tensor, inductor_lookup_seed_default_25
#   dropout_17 => mul_118, mul_119
#   dropout_18 => gt_25, mul_127, mul_128
#   iadd_11 => add_77
#   iadd_12 => add_84
#   inductor_random => inductor_random_default_1
#   layer_norm_25 => add_82, add_83, clone_34, mul_125, mul_126, rsqrt_25, sub_32, var_mean_25
#   layer_norm_26 => add_85, add_86, mul_129, mul_130, rsqrt_26, sub_33, var_mean_26
#   linear_12 => view_92
#   multi_head_attention_forward_6 => view_102
#   transpose_13 => permute_82
# Graph fragment:
#   %buf275 : Tensor "f32[500, 128, 1][1, 500, 64000]cuda:0" = PlaceHolder[target=buf275]
#   %addmm_19 : Tensor "f32[64000, 128][128, 1]cuda:0" = PlaceHolder[target=addmm_19]
#   %inductor_seeds_default : Tensor "i64[27][1]cuda:0" = PlaceHolder[target=inductor_seeds_default]
#   %inductor_random_default_1 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=inductor_random_default_1]
#   %gt_25 : Tensor "b8[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=gt_25]
#   %getitem_51 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=getitem_51]
#   %rsqrt_25 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_25]
#   %primals_108 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_108]
#   %primals_109 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_109]
#   %gt_23 : Tensor "b8[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=gt_23]
#   %primals_101 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_101]
#   %mm_default : Tensor "f32[64000, 128][128, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %add_71 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=add_71]
#   %add_84 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=add_84]
#   %buf282 : Tensor "f32[500, 128, 1][128, 1, 64000]cuda:0" = PlaceHolder[target=buf282]
#   %getitem_53 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=getitem_53]
#   %rsqrt_26 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_26]
#   %primals_110 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_110]
#   %primals_111 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_111]
#   %add_tensor : Tensor "f32[64000, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_101, %mm_default), kwargs = {})
#   %view_92 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor, [500, 128, 128]), kwargs = {})
#   %mul_118 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt_23, %view_92), kwargs = {})
#   %mul_119 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_118, 1.1111111111111112), kwargs = {})
#   %add_77 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_119, %add_71), kwargs = {})
#   %view_102 : Tensor "f32[128, 500, 128][64000, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_19, [128, 500, 128]), kwargs = {})
#   %permute_82 : Tensor "f32[500, 128, 128][128, 64000, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_102, [1, 0, 2]), kwargs = {})
#   %clone_34 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_82,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_34, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_82 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_50, 1e-05), kwargs = {})
#   %rsqrt_25 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_82,), kwargs = {})
#   %sub_32 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_34, %getitem_51), kwargs = {})
#   %mul_125 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_32, %rsqrt_25), kwargs = {})
#   %mul_126 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_125, %primals_108), kwargs = {})
#   %add_83 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_126, %primals_109), kwargs = {})
#   %inductor_lookup_seed_default_25 : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.inductor_lookup_seed.default](args = (%inductor_seeds_default, 25), kwargs = {})
#   %inductor_random_default_1 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.inductor_random.default](args = ([500, 128, 128], %inductor_lookup_seed_default_25, rand), kwargs = {})
#   %gt_25 : Tensor "b8[500, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%inductor_random_default_1, 0.1), kwargs = {})
#   %mul_127 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%gt_25, %add_83), kwargs = {})
#   %mul_128 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_127, 1.1111111111111112), kwargs = {})
#   %add_84 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_128, %add_77), kwargs = {})
#   %var_mean_26 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_84, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_85 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_52, 1e-05), kwargs = {})
#   %rsqrt_26 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_85,), kwargs = {})
#   %sub_33 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_84, %getitem_53), kwargs = {})
#   %mul_129 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_33, %rsqrt_26), kwargs = {})
#   %mul_130 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_129, %primals_110), kwargs = {})
#   %add_86 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_130, %primals_111), kwargs = {})
#   return %rsqrt_25,%getitem_51,%inductor_random_default_1,%gt_25,%add_84,%getitem_53,%buf282,%rsqrt_26,%add_86
triton_red_fused_add_addmm_native_dropout_native_layer_norm_transpose_view_15 = async_compile.triton('triton_red_fused_add_addmm_native_dropout_native_layer_norm_transpose_view_15', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*i1', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr3': '*i1', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'load_seed_offset': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]], (19,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_addmm_native_dropout_native_layer_norm_transpose_view_15', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 13, 'num_store': 7, 'num_reduction': 3, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1024000, 'r0_': 253954560}}
)
@triton.jit
def triton_red_fused_add_addmm_native_dropout_native_layer_norm_transpose_view_15(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, out_ptr3, out_ptr4, out_ptr5, load_seed_offset, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tl.store(out_ptr4 + (x3), tmp42, xmask)
    tmp48 = 128.0
    tmp49 = (tmp46 / tmp48)
    tmp50 = 1e-05
    tmp51 = tmp49 + tmp50
    tmp52 = libdevice.rsqrt(tmp51)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp52, xmask)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp53 = tl.load(in_out_ptr0 + (r0_2 + 128*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp56 = tl.load(in_ptr8 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp58 = tl.load(in_ptr9 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp54 = tmp53 - tmp42
        tmp55 = tmp54 * tmp52
        tmp57 = tmp55 * tmp56
        tmp59 = tmp57 + tmp58
        tl.store(out_ptr5 + (r0_2 + 128*x3), tmp59, r0_mask & xmask)
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
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115 = args
        args.clear()
        assert_size_stride(primals_1, (500, 128, 1, 16), (2048, 16, 16, 1))
        assert_size_stride(primals_2, (128, 16), (16, 1))
        assert_size_stride(primals_3, (128, ), (1, ))
        assert_size_stride(primals_4, (128, ), (1, ))
        assert_size_stride(primals_5, (128, ), (1, ))
        assert_size_stride(primals_6, (384, 128), (128, 1))
        assert_size_stride(primals_7, (384, ), (1, ))
        assert_size_stride(primals_8, (128, 128), (128, 1))
        assert_size_stride(primals_9, (128, ), (1, ))
        assert_size_stride(primals_10, (500, 128), (128, 1))
        assert_size_stride(primals_11, (4000, 128, 128), (16384, 128, 1))
        assert_size_stride(primals_12, (128, ), (1, ))
        assert_size_stride(primals_13, (128, ), (1, ))
        assert_size_stride(primals_14, (128, ), (1, ))
        assert_size_stride(primals_15, (128, ), (1, ))
        assert_size_stride(primals_16, (512, 128), (128, 1))
        assert_size_stride(primals_17, (512, ), (1, ))
        assert_size_stride(primals_18, (512, ), (1, ))
        assert_size_stride(primals_19, (512, ), (1, ))
        assert_size_stride(primals_20, (128, 512), (512, 1))
        assert_size_stride(primals_21, (128, ), (1, ))
        assert_size_stride(primals_22, (128, ), (1, ))
        assert_size_stride(primals_23, (128, ), (1, ))
        assert_size_stride(primals_24, (384, 128), (128, 1))
        assert_size_stride(primals_25, (384, ), (1, ))
        assert_size_stride(primals_26, (128, 128), (128, 1))
        assert_size_stride(primals_27, (128, ), (1, ))
        assert_size_stride(primals_28, (128, ), (1, ))
        assert_size_stride(primals_29, (128, ), (1, ))
        assert_size_stride(primals_30, (128, ), (1, ))
        assert_size_stride(primals_31, (128, ), (1, ))
        assert_size_stride(primals_32, (512, 128), (128, 1))
        assert_size_stride(primals_33, (512, ), (1, ))
        assert_size_stride(primals_34, (512, ), (1, ))
        assert_size_stride(primals_35, (512, ), (1, ))
        assert_size_stride(primals_36, (128, 512), (512, 1))
        assert_size_stride(primals_37, (128, ), (1, ))
        assert_size_stride(primals_38, (128, ), (1, ))
        assert_size_stride(primals_39, (128, ), (1, ))
        assert_size_stride(primals_40, (384, 128), (128, 1))
        assert_size_stride(primals_41, (384, ), (1, ))
        assert_size_stride(primals_42, (128, 128), (128, 1))
        assert_size_stride(primals_43, (128, ), (1, ))
        assert_size_stride(primals_44, (128, ), (1, ))
        assert_size_stride(primals_45, (128, ), (1, ))
        assert_size_stride(primals_46, (128, ), (1, ))
        assert_size_stride(primals_47, (128, ), (1, ))
        assert_size_stride(primals_48, (512, 128), (128, 1))
        assert_size_stride(primals_49, (512, ), (1, ))
        assert_size_stride(primals_50, (512, ), (1, ))
        assert_size_stride(primals_51, (512, ), (1, ))
        assert_size_stride(primals_52, (128, 512), (512, 1))
        assert_size_stride(primals_53, (128, ), (1, ))
        assert_size_stride(primals_54, (128, ), (1, ))
        assert_size_stride(primals_55, (128, ), (1, ))
        assert_size_stride(primals_56, (384, 128), (128, 1))
        assert_size_stride(primals_57, (384, ), (1, ))
        assert_size_stride(primals_58, (128, 128), (128, 1))
        assert_size_stride(primals_59, (128, ), (1, ))
        assert_size_stride(primals_60, (128, ), (1, ))
        assert_size_stride(primals_61, (128, ), (1, ))
        assert_size_stride(primals_62, (128, ), (1, ))
        assert_size_stride(primals_63, (128, ), (1, ))
        assert_size_stride(primals_64, (512, 128), (128, 1))
        assert_size_stride(primals_65, (512, ), (1, ))
        assert_size_stride(primals_66, (512, ), (1, ))
        assert_size_stride(primals_67, (512, ), (1, ))
        assert_size_stride(primals_68, (128, 512), (512, 1))
        assert_size_stride(primals_69, (128, ), (1, ))
        assert_size_stride(primals_70, (128, ), (1, ))
        assert_size_stride(primals_71, (128, ), (1, ))
        assert_size_stride(primals_72, (384, 128), (128, 1))
        assert_size_stride(primals_73, (384, ), (1, ))
        assert_size_stride(primals_74, (128, 128), (128, 1))
        assert_size_stride(primals_75, (128, ), (1, ))
        assert_size_stride(primals_76, (128, ), (1, ))
        assert_size_stride(primals_77, (128, ), (1, ))
        assert_size_stride(primals_78, (128, ), (1, ))
        assert_size_stride(primals_79, (128, ), (1, ))
        assert_size_stride(primals_80, (512, 128), (128, 1))
        assert_size_stride(primals_81, (512, ), (1, ))
        assert_size_stride(primals_82, (512, ), (1, ))
        assert_size_stride(primals_83, (512, ), (1, ))
        assert_size_stride(primals_84, (128, 512), (512, 1))
        assert_size_stride(primals_85, (128, ), (1, ))
        assert_size_stride(primals_86, (128, ), (1, ))
        assert_size_stride(primals_87, (128, ), (1, ))
        assert_size_stride(primals_88, (384, 128), (128, 1))
        assert_size_stride(primals_89, (384, ), (1, ))
        assert_size_stride(primals_90, (128, 128), (128, 1))
        assert_size_stride(primals_91, (128, ), (1, ))
        assert_size_stride(primals_92, (128, ), (1, ))
        assert_size_stride(primals_93, (128, ), (1, ))
        assert_size_stride(primals_94, (128, ), (1, ))
        assert_size_stride(primals_95, (128, ), (1, ))
        assert_size_stride(primals_96, (512, 128), (128, 1))
        assert_size_stride(primals_97, (512, ), (1, ))
        assert_size_stride(primals_98, (512, ), (1, ))
        assert_size_stride(primals_99, (512, ), (1, ))
        assert_size_stride(primals_100, (128, 512), (512, 1))
        assert_size_stride(primals_101, (128, ), (1, ))
        assert_size_stride(primals_102, (128, ), (1, ))
        assert_size_stride(primals_103, (128, ), (1, ))
        assert_size_stride(primals_104, (384, 128), (128, 1))
        assert_size_stride(primals_105, (384, ), (1, ))
        assert_size_stride(primals_106, (128, 128), (128, 1))
        assert_size_stride(primals_107, (128, ), (1, ))
        assert_size_stride(primals_108, (128, ), (1, ))
        assert_size_stride(primals_109, (128, ), (1, ))
        assert_size_stride(primals_110, (128, ), (1, ))
        assert_size_stride(primals_111, (128, ), (1, ))
        assert_size_stride(primals_112, (512, 128), (128, 1))
        assert_size_stride(primals_113, (512, ), (1, ))
        assert_size_stride(primals_114, (512, ), (1, ))
        assert_size_stride(primals_115, (512, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf14 = empty_strided_cuda((27, ), (1, ), torch.int64)
            # Topologically Sorted Source Nodes: [], Original ATen: []
            # [Provenance debug handles] aten.randint.low_out:1
            aten.randint.low_out(-9223372036854775808, 9223372036854775807, [27], out=buf14)
            buf0 = empty_strided_cuda((64000, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view, linear], Original ATen: [aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:2
            extern_kernels.addmm(primals_3, reinterpret_tensor(primals_1, (64000, 16), (16, 1), 0), reinterpret_tensor(primals_2, (16, 128), (1, 16), 0), alpha=1, beta=1, out=buf0)
            del primals_3
            buf1 = empty_strided_cuda((500, 128, 1), (128, 1, 1), torch.float32)
            buf2 = empty_strided_cuda((500, 128, 1), (128, 1, 64000), torch.float32)
            buf4 = reinterpret_tensor(buf2, (500, 128, 1), (128, 1, 1), 0); del buf2  # reuse
            buf5 = empty_strided_cuda((128, 500, 128), (64000, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear, layer_norm, transpose, multi_head_attention_forward], Original ATen: [aten.view, aten.native_layer_norm, aten.transpose, aten.clone]
            # [Provenance debug handles] triton_per_fused_clone_native_layer_norm_transpose_view_0:3
            stream0 = get_raw_stream(0)
            triton_per_fused_clone_native_layer_norm_transpose_view_0.run(buf4, buf0, primals_4, primals_5, buf1, buf5, 64000, 128, stream=stream0)
            del primals_5
            buf6 = empty_strided_cuda((64000, 384), (384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear, layer_norm, transpose, multi_head_attention_forward], Original ATen: [aten.view, aten.native_layer_norm, aten.transpose, aten.t, aten.clone, aten._unsafe_view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:4
            extern_kernels.mm(reinterpret_tensor(buf5, (64000, 128), (128, 1), 0), reinterpret_tensor(primals_6, (128, 384), (1, 128), 0), out=buf6)
            buf7 = empty_strided_cuda((3, 128, 500, 128), (8192000, 64000, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_1:5
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_1.run(buf6, primals_7, buf7, 24576000, stream=stream0)
            del primals_7
            buf8 = empty_strided_cuda((4000, 128, 16), (16, 64000, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2:6
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2.run(buf7, buf8, 8192000, stream=stream0)
            buf9 = empty_strided_cuda((4000, 16, 128), (16, 1, 64000), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.expand, aten.mul, aten.baddbmm]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_baddbmm_clone_expand_mul_select_squeeze_transpose_unsqueeze_view_3:7
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_baddbmm_clone_expand_mul_select_squeeze_transpose_unsqueeze_view_3.run(buf7, buf9, 8192000, stream=stream0)
            buf18 = empty_strided_cuda((4000, 128, 16), (16, 64000, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.bmm]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_4:8
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_4.run(buf7, buf18, 8192000, stream=stream0)
            buf318 = empty_strided_cuda((4000, 16, 128), (16, 1, 64000), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward, permute_263], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_4:9
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_4.run(buf7, buf318, 8192000, stream=stream0)
            buf319 = empty_strided_cuda((4000, 128, 16), (16, 64000, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward, permute_264], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_baddbmm_clone_expand_mul_select_squeeze_transpose_unsqueeze_view_3:10
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_baddbmm_clone_expand_mul_select_squeeze_transpose_unsqueeze_view_3.run(buf7, buf319, 8192000, stream=stream0)
            buf10 = empty_strided_cuda((4000, 128, 128), (16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view, aten.expand, aten.clone, aten._unsafe_view, aten.add]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_clone_expand_view_5:11
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_expand_view_5.run(primals_11, primals_10, buf10, 65536000, stream=stream0)
            del primals_10
            del primals_11
            buf11 = empty_strided_cuda((4000, 128, 128), (16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.expand, aten.mul, aten.baddbmm]
            # [Provenance debug handles] extern_kernels.baddbmm:12
            extern_kernels.baddbmm(buf10, buf8, buf9, alpha=1, beta=1, out=buf11)
            buf12 = empty_strided_cuda((4000, 128, 1), (128, 1, 1), torch.float32)
            buf13 = empty_strided_cuda((4000, 128, 1), (128, 1, 1), torch.float32)
            buf16 = empty_strided_cuda((4000, 128, 128), (16384, 128, 1), torch.bool)
            buf17 = empty_strided_cuda((4000, 128, 128), (16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, multi_head_attention_forward, inductor_random], Original ATen: [prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten.native_dropout]
            # [Provenance debug handles] triton_per_fused__softmax_exp_native_dropout_prepare_softmax_online_sub_6:13
            stream0 = get_raw_stream(0)
            triton_per_fused__softmax_exp_native_dropout_prepare_softmax_online_sub_6.run(buf11, buf14, buf12, buf13, buf16, buf17, 0, 512000, 128, stream=stream0)
            buf19 = reinterpret_tensor(buf9, (4000, 128, 16), (2048, 16, 1), 0); del buf9  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:14
            extern_kernels.bmm(buf17, buf18, out=buf19)
            buf20 = reinterpret_tensor(buf18, (128, 4000, 16), (64000, 16, 1), 0); del buf18  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.transpose, aten.clone]
            # [Provenance debug handles] triton_poi_fused_clone_transpose_7:15
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_7.run(buf19, buf20, 8192000, stream=stream0)
            buf21 = reinterpret_tensor(buf19, (64000, 128), (128, 1), 0); del buf19  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:16
            extern_kernels.addmm(primals_9, reinterpret_tensor(buf20, (64000, 128), (128, 1), 0), reinterpret_tensor(primals_8, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf21)
            del primals_9
            buf23 = empty_strided_cuda((500, 128, 1), (1, 500, 64000), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward, transpose_1, layer_norm_1], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_transpose_view_8:17
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_transpose_view_8.run(buf21, buf23, 64000, 128, stream=stream0)
            buf25 = empty_strided_cuda((500, 128, 1), (128, 1, 1), torch.float32)
            buf22 = empty_strided_cuda((500, 128, 1), (128, 1, 1), torch.float32)
            buf27 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.bool)
            buf28 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.float32)
            buf32 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.float32)
            buf33 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.float32)
            buf317 = empty_strided_cuda((500, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear, multi_head_attention_forward, transpose_1, layer_norm_1, , inductor_random, dropout, iadd, layer_norm_2, div_32], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm, aten.native_dropout, aten.add, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_red_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_9:18
            stream0 = get_raw_stream(0)
            triton_red_fused_add_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_9.run(buf23, buf21, buf14, primals_12, primals_13, buf0, primals_14, primals_15, buf25, buf22, buf27, buf28, buf32, buf33, buf317, 1, 64000, 128, stream=stream0)
            del primals_13
            del primals_15
            buf34 = empty_strided_cuda((64000, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [layer_norm_2, linear_1], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:19
            extern_kernels.addmm(primals_17, reinterpret_tensor(buf33, (64000, 128), (128, 1), 0), reinterpret_tensor(primals_16, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf34)
            del primals_17
            buf36 = empty_strided_cuda((500, 128, 512), (65536, 512, 1), torch.bool)
            buf37 = reinterpret_tensor(buf23, (500, 128, 1), (128, 1, 1), 0); del buf23  # reuse
            buf38 = empty_strided_cuda((500, 128, 1), (128, 1, 64000), torch.float32)
            buf40 = reinterpret_tensor(buf38, (500, 128, 1), (128, 1, 1), 0); del buf38  # reuse
            buf41 = empty_strided_cuda((500, 128, 512), (65536, 512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_1, gelu, , inductor_random, dropout_1, layer_norm_3], Original ATen: [aten.view, aten.gelu, aten.native_dropout, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_gelu_native_dropout_native_layer_norm_view_10:20
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_native_dropout_native_layer_norm_view_10.run(buf40, buf14, buf34, primals_18, primals_19, buf36, buf37, buf41, 2, 64000, 512, stream=stream0)
            del primals_19
            buf42 = empty_strided_cuda((64000, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_1, gelu, dropout_1, layer_norm_3, linear_2, ], Original ATen: [aten.view, aten.gelu, aten.native_dropout, aten.native_layer_norm, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:21
            extern_kernels.mm(reinterpret_tensor(buf41, (64000, 512), (512, 1), 0), reinterpret_tensor(primals_20, (512, 128), (1, 512), 0), out=buf42)
            buf44 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.bool)
            buf48 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.float32)
            buf316 = empty_strided_cuda((500, 128, 1), (128, 1, 1), torch.float32)
            buf49 = empty_strided_cuda((128, 500, 128), (64000, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear_2, inductor_random, dropout_2, iadd_1, layer_norm_4, transpose_2, multi_head_attention_forward_1, div_30], Original ATen: [aten.addmm, aten.view, aten.native_dropout, aten.add, aten.native_layer_norm, aten.transpose, aten.clone, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_add_addmm_clone_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_11:22
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_clone_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_11.run(buf14, primals_21, buf42, buf28, primals_22, primals_23, buf44, buf48, buf316, buf49, 3, 64000, 128, stream=stream0)
            del primals_23
            buf50 = reinterpret_tensor(buf7, (64000, 384), (384, 1), 0); del buf7  # reuse
            # Topologically Sorted Source Nodes: [layer_norm_4, transpose_2, multi_head_attention_forward_1], Original ATen: [aten.native_layer_norm, aten.transpose, aten.t, aten.clone, aten._unsafe_view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:23
            extern_kernels.mm(reinterpret_tensor(buf49, (64000, 128), (128, 1), 0), reinterpret_tensor(primals_24, (128, 384), (1, 128), 0), out=buf50)
            buf51 = reinterpret_tensor(buf6, (3, 128, 500, 128), (8192000, 64000, 128, 1), 0); del buf6  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_1:24
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_1.run(buf50, primals_25, buf51, 24576000, stream=stream0)
            del primals_25
            buf52 = empty_strided_cuda((4000, 128, 16), (16, 64000, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2:25
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2.run(buf51, buf52, 8192000, stream=stream0)
            buf53 = empty_strided_cuda((4000, 16, 128), (16, 1, 64000), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.mul, aten.baddbmm]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_baddbmm_clone_expand_mul_select_squeeze_transpose_unsqueeze_view_3:26
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_baddbmm_clone_expand_mul_select_squeeze_transpose_unsqueeze_view_3.run(buf51, buf53, 8192000, stream=stream0)
            buf60 = empty_strided_cuda((4000, 128, 16), (16, 64000, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.bmm]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_4:27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_4.run(buf51, buf60, 8192000, stream=stream0)
            buf314 = empty_strided_cuda((4000, 16, 128), (16, 1, 64000), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_1, permute_235], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_4:28
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_4.run(buf51, buf314, 8192000, stream=stream0)
            buf315 = empty_strided_cuda((4000, 128, 16), (16, 64000, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_1, permute_236], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_baddbmm_clone_expand_mul_select_squeeze_transpose_unsqueeze_view_3:29
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_baddbmm_clone_expand_mul_select_squeeze_transpose_unsqueeze_view_3.run(buf51, buf315, 8192000, stream=stream0)
            buf54 = empty_strided_cuda((4000, 128, 128), (16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.mul, aten.baddbmm]
            # [Provenance debug handles] extern_kernels.baddbmm:30
            extern_kernels.baddbmm(buf10, buf52, buf53, alpha=1, beta=1, out=buf54)
            buf55 = empty_strided_cuda((4000, 128, 1), (128, 1, 1), torch.float32)
            buf56 = empty_strided_cuda((4000, 128, 1), (128, 1, 1), torch.float32)
            buf58 = empty_strided_cuda((4000, 128, 128), (16384, 128, 1), torch.bool)
            buf59 = empty_strided_cuda((4000, 128, 128), (16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, multi_head_attention_forward_1, inductor_random], Original ATen: [prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten.native_dropout]
            # [Provenance debug handles] triton_per_fused__softmax_exp_native_dropout_prepare_softmax_online_sub_6:31
            stream0 = get_raw_stream(0)
            triton_per_fused__softmax_exp_native_dropout_prepare_softmax_online_sub_6.run(buf54, buf14, buf55, buf56, buf58, buf59, 4, 512000, 128, stream=stream0)
            buf61 = reinterpret_tensor(buf53, (4000, 128, 16), (2048, 16, 1), 0); del buf53  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:32
            extern_kernels.bmm(buf59, buf60, out=buf61)
            buf62 = reinterpret_tensor(buf60, (128, 4000, 16), (64000, 16, 1), 0); del buf60  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.transpose, aten.clone]
            # [Provenance debug handles] triton_poi_fused_clone_transpose_7:33
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_7.run(buf61, buf62, 8192000, stream=stream0)
            buf63 = reinterpret_tensor(buf61, (64000, 128), (128, 1), 0); del buf61  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:34
            extern_kernels.addmm(primals_27, reinterpret_tensor(buf62, (64000, 128), (128, 1), 0), reinterpret_tensor(primals_26, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf63)
            del primals_27
            buf65 = empty_strided_cuda((500, 128, 1), (1, 500, 64000), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_1, transpose_3, layer_norm_5], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_transpose_view_8:35
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_transpose_view_8.run(buf63, buf65, 64000, 128, stream=stream0)
            buf67 = empty_strided_cuda((500, 128, 1), (128, 1, 1), torch.float32)
            buf64 = empty_strided_cuda((500, 128, 1), (128, 1, 1), torch.float32)
            buf69 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.bool)
            buf70 = reinterpret_tensor(buf42, (500, 128, 128), (16384, 128, 1), 0); del buf42  # reuse
            buf74 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.float32)
            buf75 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.float32)
            buf313 = empty_strided_cuda((500, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear_2, dropout_2, iadd_1, multi_head_attention_forward_1, transpose_3, layer_norm_5, inductor_random, dropout_3, iadd_2, layer_norm_6, div_28], Original ATen: [aten.addmm, aten.view, aten.native_dropout, aten.add, aten.transpose, aten.native_layer_norm, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_red_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_12:36
            stream0 = get_raw_stream(0)
            triton_red_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_12.run(buf70, buf65, buf63, buf14, primals_28, primals_29, buf44, primals_21, buf28, primals_30, primals_31, buf67, buf64, buf69, buf74, buf75, buf313, 5, 64000, 128, stream=stream0)
            del primals_21
            del primals_29
            del primals_31
            buf76 = empty_strided_cuda((64000, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [layer_norm_6, linear_3], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:37
            extern_kernels.addmm(primals_33, reinterpret_tensor(buf75, (64000, 128), (128, 1), 0), reinterpret_tensor(primals_32, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf76)
            del primals_33
            buf78 = empty_strided_cuda((500, 128, 512), (65536, 512, 1), torch.bool)
            buf79 = reinterpret_tensor(buf65, (500, 128, 1), (128, 1, 1), 0); del buf65  # reuse
            buf80 = empty_strided_cuda((500, 128, 1), (128, 1, 64000), torch.float32)
            buf82 = reinterpret_tensor(buf80, (500, 128, 1), (128, 1, 1), 0); del buf80  # reuse
            buf83 = empty_strided_cuda((500, 128, 512), (65536, 512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_3, gelu_1, , inductor_random, dropout_4, layer_norm_7], Original ATen: [aten.view, aten.gelu, aten.native_dropout, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_gelu_native_dropout_native_layer_norm_view_10:38
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_native_dropout_native_layer_norm_view_10.run(buf82, buf14, buf76, primals_34, primals_35, buf78, buf79, buf83, 6, 64000, 512, stream=stream0)
            del primals_35
            buf84 = reinterpret_tensor(buf28, (64000, 128), (128, 1), 0); del buf28  # reuse
            # Topologically Sorted Source Nodes: [linear_3, gelu_1, dropout_4, layer_norm_7, linear_4, ], Original ATen: [aten.view, aten.gelu, aten.native_dropout, aten.native_layer_norm, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:39
            extern_kernels.mm(reinterpret_tensor(buf83, (64000, 512), (512, 1), 0), reinterpret_tensor(primals_36, (512, 128), (1, 512), 0), out=buf84)
            buf86 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.bool)
            buf90 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.float32)
            buf312 = empty_strided_cuda((500, 128, 1), (128, 1, 1), torch.float32)
            buf91 = empty_strided_cuda((128, 500, 128), (64000, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear_4, inductor_random, dropout_5, iadd_3, layer_norm_8, transpose_4, multi_head_attention_forward_2, div_26], Original ATen: [aten.addmm, aten.view, aten.native_dropout, aten.add, aten.native_layer_norm, aten.transpose, aten.clone, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_add_addmm_clone_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_11:40
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_clone_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_11.run(buf14, primals_37, buf84, buf70, primals_38, primals_39, buf86, buf90, buf312, buf91, 7, 64000, 128, stream=stream0)
            del primals_39
            buf92 = reinterpret_tensor(buf51, (64000, 384), (384, 1), 0); del buf51  # reuse
            # Topologically Sorted Source Nodes: [layer_norm_8, transpose_4, multi_head_attention_forward_2], Original ATen: [aten.native_layer_norm, aten.transpose, aten.t, aten.clone, aten._unsafe_view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:41
            extern_kernels.mm(reinterpret_tensor(buf91, (64000, 128), (128, 1), 0), reinterpret_tensor(primals_40, (128, 384), (1, 128), 0), out=buf92)
            buf93 = reinterpret_tensor(buf50, (3, 128, 500, 128), (8192000, 64000, 128, 1), 0); del buf50  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_1:42
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_1.run(buf92, primals_41, buf93, 24576000, stream=stream0)
            del primals_41
            buf94 = empty_strided_cuda((4000, 128, 16), (16, 64000, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2:43
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2.run(buf93, buf94, 8192000, stream=stream0)
            buf95 = empty_strided_cuda((4000, 16, 128), (16, 1, 64000), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.mul, aten.baddbmm]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_baddbmm_clone_expand_mul_select_squeeze_transpose_unsqueeze_view_3:44
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_baddbmm_clone_expand_mul_select_squeeze_transpose_unsqueeze_view_3.run(buf93, buf95, 8192000, stream=stream0)
            buf102 = empty_strided_cuda((4000, 128, 16), (16, 64000, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.bmm]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_4:45
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_4.run(buf93, buf102, 8192000, stream=stream0)
            buf310 = empty_strided_cuda((4000, 16, 128), (16, 1, 64000), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_2, permute_207], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_4:46
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_4.run(buf93, buf310, 8192000, stream=stream0)
            buf311 = empty_strided_cuda((4000, 128, 16), (16, 64000, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_2, permute_208], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_baddbmm_clone_expand_mul_select_squeeze_transpose_unsqueeze_view_3:47
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_baddbmm_clone_expand_mul_select_squeeze_transpose_unsqueeze_view_3.run(buf93, buf311, 8192000, stream=stream0)
            buf96 = empty_strided_cuda((4000, 128, 128), (16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.mul, aten.baddbmm]
            # [Provenance debug handles] extern_kernels.baddbmm:48
            extern_kernels.baddbmm(buf10, buf94, buf95, alpha=1, beta=1, out=buf96)
            buf97 = empty_strided_cuda((4000, 128, 1), (128, 1, 1), torch.float32)
            buf98 = empty_strided_cuda((4000, 128, 1), (128, 1, 1), torch.float32)
            buf100 = empty_strided_cuda((4000, 128, 128), (16384, 128, 1), torch.bool)
            buf101 = empty_strided_cuda((4000, 128, 128), (16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, multi_head_attention_forward_2, inductor_random], Original ATen: [prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten.native_dropout]
            # [Provenance debug handles] triton_per_fused__softmax_exp_native_dropout_prepare_softmax_online_sub_6:49
            stream0 = get_raw_stream(0)
            triton_per_fused__softmax_exp_native_dropout_prepare_softmax_online_sub_6.run(buf96, buf14, buf97, buf98, buf100, buf101, 8, 512000, 128, stream=stream0)
            buf103 = reinterpret_tensor(buf95, (4000, 128, 16), (2048, 16, 1), 0); del buf95  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:50
            extern_kernels.bmm(buf101, buf102, out=buf103)
            buf104 = reinterpret_tensor(buf102, (128, 4000, 16), (64000, 16, 1), 0); del buf102  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.transpose, aten.clone]
            # [Provenance debug handles] triton_poi_fused_clone_transpose_7:51
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_7.run(buf103, buf104, 8192000, stream=stream0)
            buf105 = reinterpret_tensor(buf103, (64000, 128), (128, 1), 0); del buf103  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:52
            extern_kernels.addmm(primals_43, reinterpret_tensor(buf104, (64000, 128), (128, 1), 0), reinterpret_tensor(primals_42, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf105)
            del primals_43
            buf107 = empty_strided_cuda((500, 128, 1), (1, 500, 64000), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_2, transpose_5, layer_norm_9], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_transpose_view_8:53
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_transpose_view_8.run(buf105, buf107, 64000, 128, stream=stream0)
            buf109 = empty_strided_cuda((500, 128, 1), (128, 1, 1), torch.float32)
            buf106 = empty_strided_cuda((500, 128, 1), (128, 1, 1), torch.float32)
            buf111 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.bool)
            buf112 = reinterpret_tensor(buf84, (500, 128, 128), (16384, 128, 1), 0); del buf84  # reuse
            buf116 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.float32)
            buf117 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.float32)
            buf309 = empty_strided_cuda((500, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear_4, dropout_5, iadd_3, multi_head_attention_forward_2, transpose_5, layer_norm_9, inductor_random, dropout_6, iadd_4, layer_norm_10, div_24], Original ATen: [aten.addmm, aten.view, aten.native_dropout, aten.add, aten.transpose, aten.native_layer_norm, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_red_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_12:54
            stream0 = get_raw_stream(0)
            triton_red_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_12.run(buf112, buf107, buf105, buf14, primals_44, primals_45, buf86, primals_37, buf70, primals_46, primals_47, buf109, buf106, buf111, buf116, buf117, buf309, 9, 64000, 128, stream=stream0)
            del primals_37
            del primals_45
            del primals_47
            buf118 = empty_strided_cuda((64000, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [layer_norm_10, linear_5], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:55
            extern_kernels.addmm(primals_49, reinterpret_tensor(buf117, (64000, 128), (128, 1), 0), reinterpret_tensor(primals_48, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf118)
            del primals_49
            buf120 = empty_strided_cuda((500, 128, 512), (65536, 512, 1), torch.bool)
            buf121 = reinterpret_tensor(buf107, (500, 128, 1), (128, 1, 1), 0); del buf107  # reuse
            buf122 = empty_strided_cuda((500, 128, 1), (128, 1, 64000), torch.float32)
            buf124 = reinterpret_tensor(buf122, (500, 128, 1), (128, 1, 1), 0); del buf122  # reuse
            buf125 = empty_strided_cuda((500, 128, 512), (65536, 512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_5, gelu_2, , inductor_random, dropout_7, layer_norm_11], Original ATen: [aten.view, aten.gelu, aten.native_dropout, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_gelu_native_dropout_native_layer_norm_view_10:56
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_native_dropout_native_layer_norm_view_10.run(buf124, buf14, buf118, primals_50, primals_51, buf120, buf121, buf125, 10, 64000, 512, stream=stream0)
            del primals_51
            buf126 = reinterpret_tensor(buf70, (64000, 128), (128, 1), 0); del buf70  # reuse
            # Topologically Sorted Source Nodes: [linear_5, gelu_2, dropout_7, layer_norm_11, linear_6, ], Original ATen: [aten.view, aten.gelu, aten.native_dropout, aten.native_layer_norm, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:57
            extern_kernels.mm(reinterpret_tensor(buf125, (64000, 512), (512, 1), 0), reinterpret_tensor(primals_52, (512, 128), (1, 512), 0), out=buf126)
            buf128 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.bool)
            buf132 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.float32)
            buf308 = empty_strided_cuda((500, 128, 1), (128, 1, 1), torch.float32)
            buf133 = empty_strided_cuda((128, 500, 128), (64000, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear_6, inductor_random, dropout_8, iadd_5, layer_norm_12, transpose_6, multi_head_attention_forward_3, div_22], Original ATen: [aten.addmm, aten.view, aten.native_dropout, aten.add, aten.native_layer_norm, aten.transpose, aten.clone, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_add_addmm_clone_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_11:58
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_clone_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_11.run(buf14, primals_53, buf126, buf112, primals_54, primals_55, buf128, buf132, buf308, buf133, 11, 64000, 128, stream=stream0)
            del primals_55
            buf134 = reinterpret_tensor(buf93, (64000, 384), (384, 1), 0); del buf93  # reuse
            # Topologically Sorted Source Nodes: [layer_norm_12, transpose_6, multi_head_attention_forward_3], Original ATen: [aten.native_layer_norm, aten.transpose, aten.t, aten.clone, aten._unsafe_view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:59
            extern_kernels.mm(reinterpret_tensor(buf133, (64000, 128), (128, 1), 0), reinterpret_tensor(primals_56, (128, 384), (1, 128), 0), out=buf134)
            buf135 = reinterpret_tensor(buf92, (3, 128, 500, 128), (8192000, 64000, 128, 1), 0); del buf92  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_1:60
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_1.run(buf134, primals_57, buf135, 24576000, stream=stream0)
            del primals_57
            buf136 = empty_strided_cuda((4000, 128, 16), (16, 64000, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2:61
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2.run(buf135, buf136, 8192000, stream=stream0)
            buf137 = empty_strided_cuda((4000, 16, 128), (16, 1, 64000), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.mul, aten.baddbmm]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_baddbmm_clone_expand_mul_select_squeeze_transpose_unsqueeze_view_3:62
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_baddbmm_clone_expand_mul_select_squeeze_transpose_unsqueeze_view_3.run(buf135, buf137, 8192000, stream=stream0)
            buf144 = empty_strided_cuda((4000, 128, 16), (16, 64000, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.bmm]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_4:63
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_4.run(buf135, buf144, 8192000, stream=stream0)
            buf306 = empty_strided_cuda((4000, 16, 128), (16, 1, 64000), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_3, permute_179], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_4:64
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_4.run(buf135, buf306, 8192000, stream=stream0)
            buf307 = empty_strided_cuda((4000, 128, 16), (16, 64000, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_3, permute_180], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_baddbmm_clone_expand_mul_select_squeeze_transpose_unsqueeze_view_3:65
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_baddbmm_clone_expand_mul_select_squeeze_transpose_unsqueeze_view_3.run(buf135, buf307, 8192000, stream=stream0)
            buf138 = empty_strided_cuda((4000, 128, 128), (16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.mul, aten.baddbmm]
            # [Provenance debug handles] extern_kernels.baddbmm:66
            extern_kernels.baddbmm(buf10, buf136, buf137, alpha=1, beta=1, out=buf138)
            buf139 = empty_strided_cuda((4000, 128, 1), (128, 1, 1), torch.float32)
            buf140 = empty_strided_cuda((4000, 128, 1), (128, 1, 1), torch.float32)
            buf142 = empty_strided_cuda((4000, 128, 128), (16384, 128, 1), torch.bool)
            buf143 = empty_strided_cuda((4000, 128, 128), (16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, multi_head_attention_forward_3, inductor_random], Original ATen: [prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten.native_dropout]
            # [Provenance debug handles] triton_per_fused__softmax_exp_native_dropout_prepare_softmax_online_sub_6:67
            stream0 = get_raw_stream(0)
            triton_per_fused__softmax_exp_native_dropout_prepare_softmax_online_sub_6.run(buf138, buf14, buf139, buf140, buf142, buf143, 12, 512000, 128, stream=stream0)
            buf145 = reinterpret_tensor(buf137, (4000, 128, 16), (2048, 16, 1), 0); del buf137  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:68
            extern_kernels.bmm(buf143, buf144, out=buf145)
            buf146 = reinterpret_tensor(buf144, (128, 4000, 16), (64000, 16, 1), 0); del buf144  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.transpose, aten.clone]
            # [Provenance debug handles] triton_poi_fused_clone_transpose_7:69
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_7.run(buf145, buf146, 8192000, stream=stream0)
            buf147 = reinterpret_tensor(buf145, (64000, 128), (128, 1), 0); del buf145  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:70
            extern_kernels.addmm(primals_59, reinterpret_tensor(buf146, (64000, 128), (128, 1), 0), reinterpret_tensor(primals_58, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf147)
            del primals_59
            buf149 = empty_strided_cuda((500, 128, 1), (1, 500, 64000), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_3, transpose_7, layer_norm_13], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_transpose_view_8:71
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_transpose_view_8.run(buf147, buf149, 64000, 128, stream=stream0)
            buf151 = empty_strided_cuda((500, 128, 1), (128, 1, 1), torch.float32)
            buf148 = empty_strided_cuda((500, 128, 1), (128, 1, 1), torch.float32)
            buf153 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.bool)
            buf154 = reinterpret_tensor(buf126, (500, 128, 128), (16384, 128, 1), 0); del buf126  # reuse
            buf158 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.float32)
            buf159 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.float32)
            buf305 = empty_strided_cuda((500, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear_6, dropout_8, iadd_5, multi_head_attention_forward_3, transpose_7, layer_norm_13, inductor_random, dropout_9, iadd_6, layer_norm_14, div_20], Original ATen: [aten.addmm, aten.view, aten.native_dropout, aten.add, aten.transpose, aten.native_layer_norm, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_red_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_12:72
            stream0 = get_raw_stream(0)
            triton_red_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_12.run(buf154, buf149, buf147, buf14, primals_60, primals_61, buf128, primals_53, buf112, primals_62, primals_63, buf151, buf148, buf153, buf158, buf159, buf305, 13, 64000, 128, stream=stream0)
            del primals_53
            del primals_61
            del primals_63
            buf160 = empty_strided_cuda((64000, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [layer_norm_14, linear_7], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:73
            extern_kernels.addmm(primals_65, reinterpret_tensor(buf159, (64000, 128), (128, 1), 0), reinterpret_tensor(primals_64, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf160)
            del primals_65
            buf162 = empty_strided_cuda((500, 128, 512), (65536, 512, 1), torch.bool)
            buf163 = reinterpret_tensor(buf149, (500, 128, 1), (128, 1, 1), 0); del buf149  # reuse
            buf164 = empty_strided_cuda((500, 128, 1), (128, 1, 64000), torch.float32)
            buf166 = reinterpret_tensor(buf164, (500, 128, 1), (128, 1, 1), 0); del buf164  # reuse
            buf167 = empty_strided_cuda((500, 128, 512), (65536, 512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_7, gelu_3, , inductor_random, dropout_10, layer_norm_15], Original ATen: [aten.view, aten.gelu, aten.native_dropout, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_gelu_native_dropout_native_layer_norm_view_10:74
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_native_dropout_native_layer_norm_view_10.run(buf166, buf14, buf160, primals_66, primals_67, buf162, buf163, buf167, 14, 64000, 512, stream=stream0)
            del primals_67
            buf168 = reinterpret_tensor(buf112, (64000, 128), (128, 1), 0); del buf112  # reuse
            # Topologically Sorted Source Nodes: [linear_7, gelu_3, dropout_10, layer_norm_15, linear_8, ], Original ATen: [aten.view, aten.gelu, aten.native_dropout, aten.native_layer_norm, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:75
            extern_kernels.mm(reinterpret_tensor(buf167, (64000, 512), (512, 1), 0), reinterpret_tensor(primals_68, (512, 128), (1, 512), 0), out=buf168)
            buf170 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.bool)
            buf174 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.float32)
            buf304 = empty_strided_cuda((500, 128, 1), (128, 1, 1), torch.float32)
            buf175 = empty_strided_cuda((128, 500, 128), (64000, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear_8, inductor_random, dropout_11, iadd_7, layer_norm_16, transpose_8, multi_head_attention_forward_4, div_18], Original ATen: [aten.addmm, aten.view, aten.native_dropout, aten.add, aten.native_layer_norm, aten.transpose, aten.clone, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_add_addmm_clone_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_11:76
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_clone_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_11.run(buf14, primals_69, buf168, buf154, primals_70, primals_71, buf170, buf174, buf304, buf175, 15, 64000, 128, stream=stream0)
            del primals_71
            buf176 = reinterpret_tensor(buf135, (64000, 384), (384, 1), 0); del buf135  # reuse
            # Topologically Sorted Source Nodes: [layer_norm_16, transpose_8, multi_head_attention_forward_4], Original ATen: [aten.native_layer_norm, aten.transpose, aten.t, aten.clone, aten._unsafe_view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:77
            extern_kernels.mm(reinterpret_tensor(buf175, (64000, 128), (128, 1), 0), reinterpret_tensor(primals_72, (128, 384), (1, 128), 0), out=buf176)
            buf177 = reinterpret_tensor(buf134, (3, 128, 500, 128), (8192000, 64000, 128, 1), 0); del buf134  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_1:78
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_1.run(buf176, primals_73, buf177, 24576000, stream=stream0)
            del primals_73
            buf178 = empty_strided_cuda((4000, 128, 16), (16, 64000, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2:79
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2.run(buf177, buf178, 8192000, stream=stream0)
            buf179 = empty_strided_cuda((4000, 16, 128), (16, 1, 64000), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.mul, aten.baddbmm]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_baddbmm_clone_expand_mul_select_squeeze_transpose_unsqueeze_view_3:80
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_baddbmm_clone_expand_mul_select_squeeze_transpose_unsqueeze_view_3.run(buf177, buf179, 8192000, stream=stream0)
            buf186 = empty_strided_cuda((4000, 128, 16), (16, 64000, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.bmm]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_4:81
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_4.run(buf177, buf186, 8192000, stream=stream0)
            buf302 = empty_strided_cuda((4000, 16, 128), (16, 1, 64000), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_4, permute_151], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_4:82
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_4.run(buf177, buf302, 8192000, stream=stream0)
            buf303 = empty_strided_cuda((4000, 128, 16), (16, 64000, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_4, permute_152], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_baddbmm_clone_expand_mul_select_squeeze_transpose_unsqueeze_view_3:83
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_baddbmm_clone_expand_mul_select_squeeze_transpose_unsqueeze_view_3.run(buf177, buf303, 8192000, stream=stream0)
            buf180 = empty_strided_cuda((4000, 128, 128), (16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.mul, aten.baddbmm]
            # [Provenance debug handles] extern_kernels.baddbmm:84
            extern_kernels.baddbmm(buf10, buf178, buf179, alpha=1, beta=1, out=buf180)
            buf181 = empty_strided_cuda((4000, 128, 1), (128, 1, 1), torch.float32)
            buf182 = empty_strided_cuda((4000, 128, 1), (128, 1, 1), torch.float32)
            buf184 = empty_strided_cuda((4000, 128, 128), (16384, 128, 1), torch.bool)
            buf185 = empty_strided_cuda((4000, 128, 128), (16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, multi_head_attention_forward_4, inductor_random], Original ATen: [prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten.native_dropout]
            # [Provenance debug handles] triton_per_fused__softmax_exp_native_dropout_prepare_softmax_online_sub_6:85
            stream0 = get_raw_stream(0)
            triton_per_fused__softmax_exp_native_dropout_prepare_softmax_online_sub_6.run(buf180, buf14, buf181, buf182, buf184, buf185, 16, 512000, 128, stream=stream0)
            buf187 = reinterpret_tensor(buf179, (4000, 128, 16), (2048, 16, 1), 0); del buf179  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:86
            extern_kernels.bmm(buf185, buf186, out=buf187)
            buf188 = reinterpret_tensor(buf186, (128, 4000, 16), (64000, 16, 1), 0); del buf186  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.transpose, aten.clone]
            # [Provenance debug handles] triton_poi_fused_clone_transpose_7:87
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_7.run(buf187, buf188, 8192000, stream=stream0)
            buf189 = reinterpret_tensor(buf187, (64000, 128), (128, 1), 0); del buf187  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:88
            extern_kernels.addmm(primals_75, reinterpret_tensor(buf188, (64000, 128), (128, 1), 0), reinterpret_tensor(primals_74, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf189)
            del primals_75
            buf191 = empty_strided_cuda((500, 128, 1), (1, 500, 64000), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_4, transpose_9, layer_norm_17], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_transpose_view_8:89
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_transpose_view_8.run(buf189, buf191, 64000, 128, stream=stream0)
            buf193 = empty_strided_cuda((500, 128, 1), (128, 1, 1), torch.float32)
            buf190 = empty_strided_cuda((500, 128, 1), (128, 1, 1), torch.float32)
            buf195 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.bool)
            buf196 = reinterpret_tensor(buf168, (500, 128, 128), (16384, 128, 1), 0); del buf168  # reuse
            buf200 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.float32)
            buf201 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.float32)
            buf301 = empty_strided_cuda((500, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear_8, dropout_11, iadd_7, multi_head_attention_forward_4, transpose_9, layer_norm_17, inductor_random, dropout_12, iadd_8, layer_norm_18, div_16], Original ATen: [aten.addmm, aten.view, aten.native_dropout, aten.add, aten.transpose, aten.native_layer_norm, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_red_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_12:90
            stream0 = get_raw_stream(0)
            triton_red_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_12.run(buf196, buf191, buf189, buf14, primals_76, primals_77, buf170, primals_69, buf154, primals_78, primals_79, buf193, buf190, buf195, buf200, buf201, buf301, 17, 64000, 128, stream=stream0)
            del primals_69
            del primals_77
            del primals_79
            buf202 = empty_strided_cuda((64000, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [layer_norm_18, linear_9], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:91
            extern_kernels.addmm(primals_81, reinterpret_tensor(buf201, (64000, 128), (128, 1), 0), reinterpret_tensor(primals_80, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf202)
            del primals_81
            buf204 = empty_strided_cuda((500, 128, 512), (65536, 512, 1), torch.bool)
            buf205 = reinterpret_tensor(buf191, (500, 128, 1), (128, 1, 1), 0); del buf191  # reuse
            buf206 = empty_strided_cuda((500, 128, 1), (128, 1, 64000), torch.float32)
            buf208 = reinterpret_tensor(buf206, (500, 128, 1), (128, 1, 1), 0); del buf206  # reuse
            buf209 = empty_strided_cuda((500, 128, 512), (65536, 512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_9, gelu_4, , inductor_random, dropout_13, layer_norm_19], Original ATen: [aten.view, aten.gelu, aten.native_dropout, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_gelu_native_dropout_native_layer_norm_view_10:92
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_native_dropout_native_layer_norm_view_10.run(buf208, buf14, buf202, primals_82, primals_83, buf204, buf205, buf209, 18, 64000, 512, stream=stream0)
            del primals_83
            buf210 = reinterpret_tensor(buf154, (64000, 128), (128, 1), 0); del buf154  # reuse
            # Topologically Sorted Source Nodes: [linear_9, gelu_4, dropout_13, layer_norm_19, linear_10, ], Original ATen: [aten.view, aten.gelu, aten.native_dropout, aten.native_layer_norm, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:93
            extern_kernels.mm(reinterpret_tensor(buf209, (64000, 512), (512, 1), 0), reinterpret_tensor(primals_84, (512, 128), (1, 512), 0), out=buf210)
            buf212 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.bool)
            buf216 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.float32)
            buf300 = empty_strided_cuda((500, 128, 1), (128, 1, 1), torch.float32)
            buf217 = empty_strided_cuda((128, 500, 128), (64000, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear_10, inductor_random, dropout_14, iadd_9, layer_norm_20, transpose_10, multi_head_attention_forward_5, div_14], Original ATen: [aten.addmm, aten.view, aten.native_dropout, aten.add, aten.native_layer_norm, aten.transpose, aten.clone, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_add_addmm_clone_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_11:94
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_clone_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_11.run(buf14, primals_85, buf210, buf196, primals_86, primals_87, buf212, buf216, buf300, buf217, 19, 64000, 128, stream=stream0)
            del primals_87
            buf218 = reinterpret_tensor(buf177, (64000, 384), (384, 1), 0); del buf177  # reuse
            # Topologically Sorted Source Nodes: [layer_norm_20, transpose_10, multi_head_attention_forward_5], Original ATen: [aten.native_layer_norm, aten.transpose, aten.t, aten.clone, aten._unsafe_view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:95
            extern_kernels.mm(reinterpret_tensor(buf217, (64000, 128), (128, 1), 0), reinterpret_tensor(primals_88, (128, 384), (1, 128), 0), out=buf218)
            buf219 = reinterpret_tensor(buf176, (3, 128, 500, 128), (8192000, 64000, 128, 1), 0); del buf176  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_1:96
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_1.run(buf218, primals_89, buf219, 24576000, stream=stream0)
            del primals_89
            buf220 = empty_strided_cuda((4000, 128, 16), (16, 64000, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2:97
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2.run(buf219, buf220, 8192000, stream=stream0)
            buf221 = empty_strided_cuda((4000, 16, 128), (16, 1, 64000), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.mul, aten.baddbmm]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_baddbmm_clone_expand_mul_select_squeeze_transpose_unsqueeze_view_3:98
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_baddbmm_clone_expand_mul_select_squeeze_transpose_unsqueeze_view_3.run(buf219, buf221, 8192000, stream=stream0)
            buf299 = empty_strided_cuda((4000, 128, 16), (16, 64000, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_5, permute_124], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_baddbmm_clone_expand_mul_select_squeeze_transpose_unsqueeze_view_3:99
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_baddbmm_clone_expand_mul_select_squeeze_transpose_unsqueeze_view_3.run(buf219, buf299, 8192000, stream=stream0)
            buf228 = empty_strided_cuda((4000, 128, 16), (16, 64000, 1), torch.float32)
            buf298 = empty_strided_cuda((4000, 16, 128), (16, 1, 64000), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_5, permute_123], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.bmm]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_13:100
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_13.run(buf219, buf228, buf298, 8192000, stream=stream0)
            buf222 = empty_strided_cuda((4000, 128, 128), (16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.mul, aten.baddbmm]
            # [Provenance debug handles] extern_kernels.baddbmm:101
            extern_kernels.baddbmm(buf10, buf220, buf221, alpha=1, beta=1, out=buf222)
            buf223 = empty_strided_cuda((4000, 128, 1), (128, 1, 1), torch.float32)
            buf224 = empty_strided_cuda((4000, 128, 1), (128, 1, 1), torch.float32)
            buf226 = empty_strided_cuda((4000, 128, 128), (16384, 128, 1), torch.bool)
            buf227 = empty_strided_cuda((4000, 128, 128), (16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, multi_head_attention_forward_5, inductor_random], Original ATen: [prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten.native_dropout]
            # [Provenance debug handles] triton_per_fused__softmax_exp_native_dropout_prepare_softmax_online_sub_6:102
            stream0 = get_raw_stream(0)
            triton_per_fused__softmax_exp_native_dropout_prepare_softmax_online_sub_6.run(buf222, buf14, buf223, buf224, buf226, buf227, 20, 512000, 128, stream=stream0)
            buf229 = reinterpret_tensor(buf221, (4000, 128, 16), (2048, 16, 1), 0); del buf221  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:103
            extern_kernels.bmm(buf227, buf228, out=buf229)
            buf230 = reinterpret_tensor(buf228, (128, 4000, 16), (64000, 16, 1), 0); del buf228  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.transpose, aten.clone]
            # [Provenance debug handles] triton_poi_fused_clone_transpose_7:104
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_7.run(buf229, buf230, 8192000, stream=stream0)
            buf231 = reinterpret_tensor(buf229, (64000, 128), (128, 1), 0); del buf229  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:105
            extern_kernels.addmm(primals_91, reinterpret_tensor(buf230, (64000, 128), (128, 1), 0), reinterpret_tensor(primals_90, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf231)
            del primals_91
            buf233 = empty_strided_cuda((500, 128, 1), (1, 500, 64000), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_5, transpose_11, layer_norm_21], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_transpose_view_8:106
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_transpose_view_8.run(buf231, buf233, 64000, 128, stream=stream0)
            buf235 = empty_strided_cuda((500, 128, 1), (128, 1, 1), torch.float32)
            buf232 = empty_strided_cuda((500, 128, 1), (128, 1, 1), torch.float32)
            buf237 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.bool)
            buf238 = reinterpret_tensor(buf210, (500, 128, 128), (16384, 128, 1), 0); del buf210  # reuse
            buf242 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.float32)
            buf243 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.float32)
            buf297 = empty_strided_cuda((500, 128, 1), (128, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear_10, dropout_14, iadd_9, multi_head_attention_forward_5, transpose_11, layer_norm_21, inductor_random, dropout_15, iadd_10, layer_norm_22, div_12], Original ATen: [aten.addmm, aten.view, aten.native_dropout, aten.add, aten.transpose, aten.native_layer_norm, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_red_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_12:107
            stream0 = get_raw_stream(0)
            triton_red_fused_add_addmm_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_12.run(buf238, buf233, buf231, buf14, primals_92, primals_93, buf212, primals_85, buf196, primals_94, primals_95, buf235, buf232, buf237, buf242, buf243, buf297, 21, 64000, 128, stream=stream0)
            del primals_85
            del primals_93
            del primals_95
            buf244 = empty_strided_cuda((64000, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [layer_norm_22, linear_11], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:108
            extern_kernels.addmm(primals_97, reinterpret_tensor(buf243, (64000, 128), (128, 1), 0), reinterpret_tensor(primals_96, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf244)
            del primals_97
            buf246 = empty_strided_cuda((500, 128, 512), (65536, 512, 1), torch.bool)
            buf247 = reinterpret_tensor(buf233, (500, 128, 1), (128, 1, 1), 0); del buf233  # reuse
            buf248 = empty_strided_cuda((500, 128, 1), (128, 1, 64000), torch.float32)
            buf250 = reinterpret_tensor(buf248, (500, 128, 1), (128, 1, 1), 0); del buf248  # reuse
            buf251 = empty_strided_cuda((500, 128, 512), (65536, 512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_11, gelu_5, , inductor_random, dropout_16, layer_norm_23], Original ATen: [aten.view, aten.gelu, aten.native_dropout, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_gelu_native_dropout_native_layer_norm_view_10:109
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_native_dropout_native_layer_norm_view_10.run(buf250, buf14, buf244, primals_98, primals_99, buf246, buf247, buf251, 22, 64000, 512, stream=stream0)
            del primals_99
            buf252 = reinterpret_tensor(buf196, (64000, 128), (128, 1), 0); del buf196  # reuse
            # Topologically Sorted Source Nodes: [linear_11, gelu_5, dropout_16, layer_norm_23, linear_12, ], Original ATen: [aten.view, aten.gelu, aten.native_dropout, aten.native_layer_norm, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.mm:110
            extern_kernels.mm(reinterpret_tensor(buf251, (64000, 512), (512, 1), 0), reinterpret_tensor(primals_100, (512, 128), (1, 512), 0), out=buf252)
            buf254 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.bool)
            buf258 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.float32)
            buf296 = empty_strided_cuda((500, 128, 1), (128, 1, 1), torch.float32)
            buf259 = empty_strided_cuda((128, 500, 128), (64000, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear_12, inductor_random, dropout_17, iadd_11, layer_norm_24, transpose_12, multi_head_attention_forward_6, div_10], Original ATen: [aten.addmm, aten.view, aten.native_dropout, aten.add, aten.native_layer_norm, aten.transpose, aten.clone, aten.native_layer_norm_backward]
            # [Provenance debug handles] triton_per_fused_add_addmm_clone_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_11:111
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_clone_native_dropout_native_layer_norm_native_layer_norm_backward_transpose_view_11.run(buf14, primals_101, buf252, buf238, primals_102, primals_103, buf254, buf258, buf296, buf259, 23, 64000, 128, stream=stream0)
            del primals_103
            buf260 = reinterpret_tensor(buf219, (64000, 384), (384, 1), 0); del buf219  # reuse
            # Topologically Sorted Source Nodes: [layer_norm_24, transpose_12, multi_head_attention_forward_6], Original ATen: [aten.native_layer_norm, aten.transpose, aten.t, aten.clone, aten._unsafe_view, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:112
            extern_kernels.mm(reinterpret_tensor(buf259, (64000, 128), (128, 1), 0), reinterpret_tensor(primals_104, (128, 384), (1, 128), 0), out=buf260)
            buf261 = reinterpret_tensor(buf218, (3, 128, 500, 128), (8192000, 64000, 128, 1), 0); del buf218  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_1:113
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_squeeze_transpose_unsqueeze_view_1.run(buf260, primals_105, buf261, 24576000, stream=stream0)
            del buf260
            del primals_105
            buf262 = empty_strided_cuda((4000, 128, 16), (16, 64000, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2:114
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_clone_mul_select_squeeze_transpose_unsqueeze_view_2.run(buf261, buf262, 8192000, stream=stream0)
            buf263 = empty_strided_cuda((4000, 16, 128), (16, 1, 64000), torch.float32)
            buf295 = empty_strided_cuda((4000, 128, 16), (16, 64000, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_6, permute_96], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.mul, aten.baddbmm]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_baddbmm_clone_mul_select_squeeze_transpose_unsqueeze_view_14:115
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_baddbmm_clone_mul_select_squeeze_transpose_unsqueeze_view_14.run(buf261, buf263, buf295, 8192000, stream=stream0)
            buf264 = empty_strided_cuda((4000, 128, 128), (16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.mul, aten.baddbmm]
            # [Provenance debug handles] extern_kernels.baddbmm:116
            extern_kernels.baddbmm(buf10, buf262, buf263, alpha=1, beta=1, out=buf264)
            del buf10
            buf270 = reinterpret_tensor(buf263, (4000, 128, 16), (16, 64000, 1), 0); del buf263  # reuse
            buf294 = empty_strided_cuda((4000, 16, 128), (16, 1, 64000), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_6, permute_95], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.bmm]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_13:117
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_bmm_clone_select_squeeze_transpose_unsqueeze_view_13.run(buf261, buf270, buf294, 8192000, stream=stream0)
            del buf261
            buf265 = empty_strided_cuda((4000, 128, 1), (128, 1, 1), torch.float32)
            buf266 = empty_strided_cuda((4000, 128, 1), (128, 1, 1), torch.float32)
            buf268 = empty_strided_cuda((4000, 128, 128), (16384, 128, 1), torch.bool)
            buf269 = empty_strided_cuda((4000, 128, 128), (16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, multi_head_attention_forward_6, inductor_random], Original ATen: [prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax, aten.native_dropout]
            # [Provenance debug handles] triton_per_fused__softmax_exp_native_dropout_prepare_softmax_online_sub_6:118
            stream0 = get_raw_stream(0)
            triton_per_fused__softmax_exp_native_dropout_prepare_softmax_online_sub_6.run(buf264, buf14, buf265, buf266, buf268, buf269, 24, 512000, 128, stream=stream0)
            buf271 = empty_strided_cuda((4000, 128, 16), (2048, 16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten._unsafe_view, aten.add, aten.view, aten.unsqueeze, aten.transpose, aten.squeeze, aten.clone, aten.select, aten.bmm]
            # [Provenance debug handles] extern_kernels.bmm:119
            extern_kernels.bmm(buf269, buf270, out=buf271)
            buf272 = reinterpret_tensor(buf270, (128, 4000, 16), (64000, 16, 1), 0); del buf270  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten.transpose, aten.clone]
            # [Provenance debug handles] triton_poi_fused_clone_transpose_7:120
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_7.run(buf271, buf272, 8192000, stream=stream0)
            buf273 = reinterpret_tensor(buf271, (64000, 128), (128, 1), 0); del buf271  # reuse
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten.transpose, aten.clone, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:121
            extern_kernels.addmm(primals_107, reinterpret_tensor(buf272, (64000, 128), (128, 1), 0), reinterpret_tensor(primals_106, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf273)
            del primals_107
            buf275 = empty_strided_cuda((500, 128, 1), (1, 500, 64000), torch.float32)
            # Topologically Sorted Source Nodes: [multi_head_attention_forward_6, transpose_13, layer_norm_25], Original ATen: [aten.view, aten.transpose, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_native_layer_norm_transpose_view_8:122
            stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_transpose_view_8.run(buf273, buf275, 64000, 128, stream=stream0)
            buf277 = empty_strided_cuda((500, 128, 1), (128, 1, 1), torch.float32)
            buf274 = empty_strided_cuda((500, 128, 1), (128, 1, 1), torch.float32)
            buf279 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.bool)
            buf280 = reinterpret_tensor(buf252, (500, 128, 128), (16384, 128, 1), 0); del buf252  # reuse
            buf281 = empty_strided_cuda((500, 128, 1), (128, 1, 1), torch.float32)
            buf282 = empty_strided_cuda((500, 128, 1), (128, 1, 64000), torch.float32)
            buf284 = reinterpret_tensor(buf282, (500, 128, 1), (128, 1, 1), 0); del buf282  # reuse
            buf285 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [, linear_12, dropout_17, iadd_11, multi_head_attention_forward_6, transpose_13, layer_norm_25, inductor_random, dropout_18, iadd_12, layer_norm_26], Original ATen: [aten.addmm, aten.view, aten.native_dropout, aten.add, aten.transpose, aten.native_layer_norm]
            # [Provenance debug handles] triton_red_fused_add_addmm_native_dropout_native_layer_norm_transpose_view_15:123
            stream0 = get_raw_stream(0)
            triton_red_fused_add_addmm_native_dropout_native_layer_norm_transpose_view_15.run(buf280, buf284, buf275, buf273, buf14, primals_108, primals_109, buf254, primals_101, buf238, primals_110, primals_111, buf277, buf274, buf279, buf281, buf285, 25, 64000, 128, stream=stream0)
            del buf238
            del primals_101
            del primals_109
            del primals_111
            buf286 = empty_strided_cuda((64000, 512), (512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [layer_norm_26, linear_13], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:124
            extern_kernels.addmm(primals_113, reinterpret_tensor(buf285, (64000, 128), (128, 1), 0), reinterpret_tensor(primals_112, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf286)
            del primals_113
            buf288 = empty_strided_cuda((500, 128, 512), (65536, 512, 1), torch.bool)
            buf289 = reinterpret_tensor(buf275, (500, 128, 1), (128, 1, 1), 0); del buf275  # reuse
            buf290 = empty_strided_cuda((500, 128, 1), (128, 1, 64000), torch.float32)
            buf292 = reinterpret_tensor(buf290, (500, 128, 1), (128, 1, 1), 0); del buf290  # reuse
            buf293 = empty_strided_cuda((500, 128, 512), (65536, 512, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_13, gelu_6, , inductor_random, dropout_19, layer_norm_27], Original ATen: [aten.view, aten.gelu, aten.native_dropout, aten.native_layer_norm]
            # [Provenance debug handles] triton_per_fused_gelu_native_dropout_native_layer_norm_view_10:125
            stream0 = get_raw_stream(0)
            triton_per_fused_gelu_native_dropout_native_layer_norm_view_10.run(buf292, buf14, buf286, primals_114, primals_115, buf288, buf289, buf293, 26, 64000, 512, stream=stream0)
            del buf14
            del primals_115
        return (buf293, buf280, primals_2, primals_4, primals_6, primals_8, primals_12, primals_14, primals_16, primals_18, primals_20, primals_22, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_38, primals_40, primals_42, primals_44, primals_46, primals_48, primals_50, primals_52, primals_54, primals_56, primals_58, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_106, primals_108, primals_110, primals_112, primals_114, reinterpret_tensor(primals_1, (64000, 16), (16, 1), 0), buf0, buf1, buf4, reinterpret_tensor(buf5, (64000, 128), (128, 1), 0), buf11, buf12, buf13, buf16, reinterpret_tensor(buf20, (64000, 128), (128, 1), 0), buf21, buf22, buf25, buf27, buf32, reinterpret_tensor(buf33, (64000, 128), (128, 1), 0), buf34, buf36, buf37, buf40, reinterpret_tensor(buf41, (64000, 512), (512, 1), 0), buf44, buf48, reinterpret_tensor(buf49, (64000, 128), (128, 1), 0), buf54, buf55, buf56, buf58, reinterpret_tensor(buf62, (64000, 128), (128, 1), 0), buf63, buf64, buf67, buf69, buf74, reinterpret_tensor(buf75, (64000, 128), (128, 1), 0), buf76, buf78, buf79, buf82, reinterpret_tensor(buf83, (64000, 512), (512, 1), 0), buf86, buf90, reinterpret_tensor(buf91, (64000, 128), (128, 1), 0), buf96, buf97, buf98, buf100, reinterpret_tensor(buf104, (64000, 128), (128, 1), 0), buf105, buf106, buf109, buf111, buf116, reinterpret_tensor(buf117, (64000, 128), (128, 1), 0), buf118, buf120, buf121, buf124, reinterpret_tensor(buf125, (64000, 512), (512, 1), 0), buf128, buf132, reinterpret_tensor(buf133, (64000, 128), (128, 1), 0), buf138, buf139, buf140, buf142, reinterpret_tensor(buf146, (64000, 128), (128, 1), 0), buf147, buf148, buf151, buf153, buf158, reinterpret_tensor(buf159, (64000, 128), (128, 1), 0), buf160, buf162, buf163, buf166, reinterpret_tensor(buf167, (64000, 512), (512, 1), 0), buf170, buf174, reinterpret_tensor(buf175, (64000, 128), (128, 1), 0), buf180, buf181, buf182, buf184, reinterpret_tensor(buf188, (64000, 128), (128, 1), 0), buf189, buf190, buf193, buf195, buf200, reinterpret_tensor(buf201, (64000, 128), (128, 1), 0), buf202, buf204, buf205, buf208, reinterpret_tensor(buf209, (64000, 512), (512, 1), 0), buf212, buf216, reinterpret_tensor(buf217, (64000, 128), (128, 1), 0), buf222, buf223, buf224, buf226, reinterpret_tensor(buf230, (64000, 128), (128, 1), 0), buf231, buf232, buf235, buf237, buf242, reinterpret_tensor(buf243, (64000, 128), (128, 1), 0), buf244, buf246, buf247, buf250, reinterpret_tensor(buf251, (64000, 512), (512, 1), 0), buf254, buf258, reinterpret_tensor(buf259, (64000, 128), (128, 1), 0), buf264, buf265, buf266, buf268, reinterpret_tensor(buf272, (64000, 128), (128, 1), 0), buf273, buf274, buf277, buf279, buf280, buf281, buf284, reinterpret_tensor(buf285, (64000, 128), (128, 1), 0), buf286, buf288, buf289, buf292, reinterpret_tensor(buf269, (4000, 128, 128), (16384, 1, 128), 0), buf294, buf295, reinterpret_tensor(buf262, (4000, 16, 128), (16, 1, 64000), 0), buf296, buf297, reinterpret_tensor(buf227, (4000, 128, 128), (16384, 1, 128), 0), buf298, buf299, reinterpret_tensor(buf220, (4000, 16, 128), (16, 1, 64000), 0), buf300, buf301, reinterpret_tensor(buf185, (4000, 128, 128), (16384, 1, 128), 0), buf302, buf303, reinterpret_tensor(buf178, (4000, 16, 128), (16, 1, 64000), 0), buf304, buf305, reinterpret_tensor(buf143, (4000, 128, 128), (16384, 1, 128), 0), buf306, buf307, reinterpret_tensor(buf136, (4000, 16, 128), (16, 1, 64000), 0), buf308, buf309, reinterpret_tensor(buf101, (4000, 128, 128), (16384, 1, 128), 0), buf310, buf311, reinterpret_tensor(buf94, (4000, 16, 128), (16, 1, 64000), 0), buf312, buf313, reinterpret_tensor(buf59, (4000, 128, 128), (16384, 1, 128), 0), buf314, buf315, reinterpret_tensor(buf52, (4000, 16, 128), (16, 1, 64000), 0), buf316, buf317, reinterpret_tensor(buf17, (4000, 128, 128), (16384, 1, 128), 0), buf318, buf319, reinterpret_tensor(buf8, (4000, 16, 128), (16, 1, 64000), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((500, 128, 1, 16), (2048, 16, 16, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((128, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((500, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4000, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
