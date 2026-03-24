# AOT ID: ['12_forward']
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


# kernel path: /traces/inductor_cache/ej/cejycsizoc475r4s2wt4sdejsnwftbmmvganms3dn5vo5i43hnqm.py
# Topologically Sorted Source Nodes: [view, U, input_1], Original ATen: [aten.view, aten.transpose, aten.clone, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   U => permute
#   input_1 => clone, var_mean
#   view => view
# Graph fragment:
#   %primals_2 : Tensor "f32[500, 128, 128, 4][65536, 512, 4, 1]cuda:0" = PlaceHolder[target=primals_2]
#   %view : Tensor "f32[500, 16384, 4][65536, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_2, [500, 16384, 4]), kwargs = {})
#   %permute : Tensor "f32[500, 4, 16384][65536, 1, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view, [0, 2, 1]), kwargs = {})
#   %clone : Tensor "f32[500, 4, 16384][65536, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   return %buf0,%buf1,%buf2
triton_red_fused__native_batch_norm_legit_functional_clone_transpose_view_0 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_clone_transpose_view_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_clone_transpose_view_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 3, 'num_reduction': 3, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 133726208, 'r0_': 0}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_clone_transpose_view_0(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 110592
    r0_numel = 297
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x1 = ((xindex // 4) % 32)
    x2 = xindex // 128
    x0 = (xindex % 4)
    tmp23_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp23_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp23_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp0 = r0_3 + 297*x1
        tmp1 = tl.full([1, 1], 9482, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = r0_3 + 297*x1 + 9482*x2
        tmp4 = tl.full([1, 1], 8192000, tl.int32)
        tmp5 = tmp3 < tmp4
        tmp6 = tmp5 & tmp2
        tmp7 = tl.load(in_ptr0 + (x0 + 4*(((r0_3 + 297*x1 + 9482*x2) % 8192000))), r0_mask & tmp6, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = 0.0
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp6, tmp10, tmp11)
        tmp13 = tl.full(tmp12.shape, 0, tmp12.dtype)
        tmp14 = tl.where(tmp2, tmp12, tmp13)
        tmp15 = 1.0
        tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
        tmp17 = tl.where(tmp6, tmp15, tmp16)
        tmp18 = tl.full(tmp17.shape, 0, tmp17.dtype)
        tmp19 = tl.where(tmp2, tmp17, tmp18)
        tmp20 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp21 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
        tmp22 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
        tmp23_mean_next, tmp23_m2_next, tmp23_weight_next = triton_helpers.welford_combine(
            tmp23_mean, tmp23_m2, tmp23_weight,
            tmp20, tmp21, tmp22
        )
        tmp23_mean = tl.where(r0_mask, tmp23_mean_next, tmp23_mean)
        tmp23_m2 = tl.where(r0_mask, tmp23_m2_next, tmp23_m2)
        tmp23_weight = tl.where(r0_mask, tmp23_weight_next, tmp23_weight)
    tmp24, tmp25, tmp26 = triton_helpers.welford(tmp23_mean, tmp23_m2, tmp23_weight, 1)
    tmp23 = tmp24[:, None]
    tmp27 = tmp25[:, None]
    tmp28 = tmp26[:, None]
    tl.store(out_ptr0 + (x4), tmp23, None)
    tl.store(out_ptr1 + (x4), tmp27, None)
    tl.store(out_ptr2 + (x4), tmp28, None)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/5h/c5h7nszdtifgrl6dtfmk7rhjvvx7734qg3w7p63k5nwf5ew7wvyl.py
# Topologically Sorted Source Nodes: [view, U, input_1], Original ATen: [aten.view, aten.transpose, aten.clone, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   U => permute
#   input_1 => clone, var_mean
#   view => view
# Graph fragment:
#   %buf0 : Tensor "f32[1, 4, 1, 864, 32][110592, 1, 110592, 128, 4]cuda:0" = PlaceHolder[target=buf0]
#   %buf1 : Tensor "f32[1, 4, 1, 864, 32][110592, 1, 110592, 128, 4]cuda:0" = PlaceHolder[target=buf1]
#   %buf2 : Tensor "f32[1, 4, 1, 864, 32][110592, 1, 110592, 128, 4]cuda:0" = PlaceHolder[target=buf2]
#   %view : Tensor "f32[500, 16384, 4][65536, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_2, [500, 16384, 4]), kwargs = {})
#   %permute : Tensor "f32[500, 4, 16384][65536, 1, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view, [0, 2, 1]), kwargs = {})
#   %clone : Tensor "f32[500, 4, 16384][65536, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   return %buf3,%buf4,%buf5
triton_per_fused__native_batch_norm_legit_functional_clone_transpose_view_1 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_clone_transpose_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 32},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_clone_transpose_view_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 3, 'num_reduction': 3, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1382400, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_clone_transpose_view_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    r0_2 = r0_index
    x0 = (xindex % 4)
    x1 = xindex // 4
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4*r0_2 + 128*x1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 4*r0_2 + 128*x1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 4*r0_2 + 128*x1), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x1 + 864*x0), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
    tl.store(out_ptr2 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/ft/cftlsrkxj46ed6r6sazd47usjzoue7o34bdjbsy76nzvnafshori.py
# Topologically Sorted Source Nodes: [view, U, input_1], Original ATen: [aten.view, aten.transpose, aten.clone, aten._native_batch_norm_legit_functional, aten.copy_]
# Source node to ATen node mapping:
#   U => permute
#   input_1 => add_1, add_2, add_3, clone, mul_1, mul_2, mul_3, mul_4, mul_5, rsqrt, squeeze, squeeze_2, var_mean
#   view => view
# Graph fragment:
#   %buf3 : Tensor "f32[1, 4, 1, 864][3456, 864, 3456, 1]cuda:0" = PlaceHolder[target=buf3]
#   %buf4 : Tensor "f32[1, 4, 1, 864][3456, 1, 3456, 4]cuda:0" = PlaceHolder[target=buf4]
#   %buf5 : Tensor "f32[1, 4, 1, 864][3456, 1, 3456, 4]cuda:0" = PlaceHolder[target=buf5]
#   %buf7 : Tensor "f32[1, 4, 1][4, 1, 4]cuda:0" = PlaceHolder[target=buf7]
#   %copy__2 : Tensor "f32[4][1]cuda:0" = PlaceHolder[target=copy__2]
#   %add_3 : Tensor "f32[4][1]cuda:0" = PlaceHolder[target=add_3]
#   %getitem_1 : Tensor "f32[1, 4, 1][4, 1, 4]cuda:0" = PlaceHolder[target=getitem_1]
#   %copy__1 : Tensor "f32[4][1]cuda:0" = PlaceHolder[target=copy__1]
#   %add_2 : Tensor "f32[4][1]cuda:0" = PlaceHolder[target=add_2]
#   %view : Tensor "f32[500, 16384, 4][65536, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_2, [500, 16384, 4]), kwargs = {})
#   %permute : Tensor "f32[500, 4, 16384][65536, 1, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view, [0, 2, 1]), kwargs = {})
#   %clone : Tensor "f32[500, 4, 16384][65536, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %add_1 : Tensor "f32[1, 4, 1][4, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[1, 4, 1][4, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %squeeze : Tensor "f32[4][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_1, [0, 2]), kwargs = {})
#   %mul_1 : Tensor "f32[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze, 0.1), kwargs = {})
#   %mul_2 : Tensor "f32[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_4, 0.9), kwargs = {})
#   %add_2 : Tensor "f32[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %mul_2), kwargs = {})
#   %squeeze_2 : Tensor "f32[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem, [0, 2]), kwargs = {})
#   %mul_3 : Tensor "f32[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_2, 1.0000001220703274), kwargs = {})
#   %mul_4 : Tensor "f32[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, 0.1), kwargs = {})
#   %mul_5 : Tensor "f32[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_5, 0.9), kwargs = {})
#   %add_3 : Tensor "f32[4][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %mul_5), kwargs = {})
#   %copy__1 : Tensor "f32[4][1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_4, %add_2), kwargs = {})
#   %copy__2 : Tensor "f32[4][1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_5, %add_3), kwargs = {})
#   return %getitem_1,%buf7,%rsqrt,%add_3,%buf62,%add_2,%buf59
triton_red_fused__native_batch_norm_legit_functional_clone_copy__transpose_view_2 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_clone_copy__transpose_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4, 'r0_': 1024},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_clone_copy__transpose_view_2', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 5, 'num_reduction': 2, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 13920, 'r0_': 13824}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_clone_copy__transpose_view_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4
    r0_numel = 864
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp6_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 864*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 4*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + 4*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_combine(
            tmp6_mean, tmp6_m2, tmp6_weight,
            tmp3, tmp4, tmp5
        )
        tmp6_mean = tl.where(r0_mask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(r0_mask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(r0_mask & xmask, tmp6_weight_next, tmp6_weight)
    tmp7, tmp8, tmp9 = triton_helpers.welford(tmp6_mean, tmp6_m2, tmp6_weight, 1)
    tmp6 = tmp7[:, None]
    tmp10 = tmp8[:, None]
    tmp11 = tmp9[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
    tmp21 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = 8192000.0
    tmp13 = (tmp10 / tmp12)
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tmp17 = 1.0000001220703274
    tmp18 = tmp13 * tmp17
    tmp19 = 0.1
    tmp20 = tmp18 * tmp19
    tmp22 = 0.9
    tmp23 = tmp21 * tmp22
    tmp24 = tmp20 + tmp23
    tmp25 = tmp6 * tmp19
    tmp27 = tmp26 * tmp22
    tmp28 = tmp25 + tmp27
    tl.store(out_ptr2 + (x0), tmp16, xmask)
    tl.store(out_ptr4 + (x0), tmp24, xmask)
    tl.store(out_ptr6 + (x0), tmp28, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/62/c62xuv24kvervd2to2vb7scoyfrorz23q2wvkykujkusf2a7uh7z.py
# Topologically Sorted Source Nodes: [view, U, input_1], Original ATen: [aten.view, aten.transpose, aten.clone, aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   U => permute
#   input_1 => add_1, add_4, clone, mul, mul_6, rsqrt, sub, unsqueeze, unsqueeze_1, var_mean
#   view => view
# Graph fragment:
#   %primals_2 : Tensor "f32[500, 128, 128, 4][65536, 512, 4, 1]cuda:0" = PlaceHolder[target=primals_2]
#   %getitem_1 : Tensor "f32[1, 4, 1][4, 1, 4]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf7 : Tensor "f32[1, 4, 1][4, 1, 4]cuda:0" = PlaceHolder[target=buf7]
#   %primals_6 : Tensor "f32[4][1]cuda:0" = PlaceHolder[target=primals_6]
#   %primals_7 : Tensor "f32[4][1]cuda:0" = PlaceHolder[target=primals_7]
#   %view : Tensor "f32[500, 16384, 4][65536, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%primals_2, [500, 16384, 4]), kwargs = {})
#   %permute : Tensor "f32[500, 4, 16384][65536, 1, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view, [0, 2, 1]), kwargs = {})
#   %clone : Tensor "f32[500, 4, 16384][65536, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %add_1 : Tensor "f32[1, 4, 1][4, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[1, 4, 1][4, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %sub : Tensor "f32[500, 4, 16384][65536, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone, %getitem_1), kwargs = {})
#   %mul : Tensor "f32[500, 4, 16384][65536, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %unsqueeze : Tensor "f32[4, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_6, -1), kwargs = {})
#   %mul_6 : Tensor "f32[500, 4, 16384][65536, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %unsqueeze), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[4, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_7, -1), kwargs = {})
#   %add_4 : Tensor "f32[500, 4, 16384][65536, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %unsqueeze_1), kwargs = {})
#   return %add_4
triton_poi_fused__native_batch_norm_legit_functional_clone_transpose_view_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_clone_transpose_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 16384}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_clone_transpose_view_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 131072064, 'x': 262144000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_clone_transpose_view_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2000
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :]
    x2 = xindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4*x2 + 65536*y1), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 8192000.0
    tmp5 = (tmp3 / tmp4)
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2 + 16384*y3), tmp13, ymask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/uy/cuycqqgc65wnp4pg5kdca23wiu4cgv5rnnittbnrirmibey5jxo3.py
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_2 => convolution
# Graph fragment:
#   %buf11 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0" = PlaceHolder[target=buf11]
#   %primals_9 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=primals_9]
#   %convolution : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%add_4, %primals_8, %primals_9, [1], [0], [1], False, [0], 1), kwargs = {})
#   return %convolution
triton_poi_fused_convolution_4 = async_compile.triton('triton_poi_fused_convolution_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 6291456000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x1 = ((xindex // 16384) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/lj/cljto65qpix2752guatv5dnraffltgcd6thvph24f7vc6l2xwzod.py
# Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_3 => var_mean_1
# Graph fragment:
#   %convolution : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0" = PlaceHolder[target=convolution]
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convolution, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   return %buf13,%buf14,%buf15
triton_red_fused__native_batch_norm_legit_functional_5 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 3, 'num_reduction': 3, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 96768, 'r0_': 2097152000}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_5(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp13_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp13_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp13_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
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
        tmp4 = 0.0
        tmp5 = tl.full(tmp4.shape, 0, tmp4.dtype)
        tmp6 = tl.where(tmp2, tmp4, tmp5)
        tmp7 = 1.0
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp11 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp12 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp13_mean_next, tmp13_m2_next, tmp13_weight_next = triton_helpers.welford_combine(
            tmp13_mean, tmp13_m2, tmp13_weight,
            tmp10, tmp11, tmp12
        )
        tmp13_mean = tl.where(r0_mask & xmask, tmp13_mean_next, tmp13_mean)
        tmp13_m2 = tl.where(r0_mask & xmask, tmp13_m2_next, tmp13_m2)
        tmp13_weight = tl.where(r0_mask & xmask, tmp13_weight_next, tmp13_weight)
    tmp14, tmp15, tmp16 = triton_helpers.welford(tmp13_mean, tmp13_m2, tmp13_weight, 1)
    tmp13 = tmp14[:, None]
    tmp17 = tmp15[:, None]
    tmp18 = tmp16[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp17, xmask)
    tl.store(out_ptr2 + (x3), tmp18, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/lh/clhcsi7lviywvdfd75ks65yorct26vywxxkji5c5nggxbcfsggh6.py
# Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
# Source node to ATen node mapping:
#   input_3 => add_6, add_7, add_8, mul_10, mul_11, mul_12, mul_8, mul_9, rsqrt_1, squeeze_3, squeeze_5, var_mean_1
# Graph fragment:
#   %buf13 : Tensor "f32[1, 64, 1, 63][4032, 1, 4032, 64]cuda:0" = PlaceHolder[target=buf13]
#   %buf14 : Tensor "f32[1, 64, 1, 63][4032, 1, 4032, 64]cuda:0" = PlaceHolder[target=buf14]
#   %buf15 : Tensor "f32[1, 64, 1, 63][4032, 1, 4032, 64]cuda:0" = PlaceHolder[target=buf15]
#   %buf17 : Tensor "f32[1, 64, 1][64, 1, 64]cuda:0" = PlaceHolder[target=buf17]
#   %copy__5 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=copy__5]
#   %add_8 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=add_8]
#   %getitem_3 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=getitem_3]
#   %copy__4 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=copy__4]
#   %add_7 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=add_7]
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convolution, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %add_6 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %squeeze_3 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_3, [0, 2]), kwargs = {})
#   %mul_8 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_3, 0.1), kwargs = {})
#   %mul_9 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_11, 0.9), kwargs = {})
#   %add_7 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %mul_9), kwargs = {})
#   %squeeze_5 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_2, [0, 2]), kwargs = {})
#   %mul_10 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_5, 1.0000001220703274), kwargs = {})
#   %mul_11 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, 0.1), kwargs = {})
#   %mul_12 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_12, 0.9), kwargs = {})
#   %add_8 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %mul_12), kwargs = {})
#   %copy__4 : Tensor "f32[64][1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_11, %add_7), kwargs = {})
#   %copy__5 : Tensor "f32[64][1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_12, %add_8), kwargs = {})
#   return %getitem_3,%buf17,%rsqrt_1,%add_8,%buf70,%add_7,%buf67
triton_per_fused__native_batch_norm_legit_functional_copy__6 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_copy__6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_copy__6', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 4, 'num_reduction': 2, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 50944, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_copy__6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + 64*r0_1), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 64*r0_1), r0_mask & xmask, other=0.0)
    tmp25 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp3, 0)
    tmp8 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp9 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 8192000.0
    tmp17 = (tmp14 / tmp16)
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 1.0000001220703274
    tmp22 = tmp17 * tmp21
    tmp23 = 0.1
    tmp24 = tmp22 * tmp23
    tmp26 = 0.9
    tmp27 = tmp25 * tmp26
    tmp28 = tmp24 + tmp27
    tmp29 = tmp13 * tmp23
    tmp31 = tmp30 * tmp26
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp28, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/ii/ciisyxest65irzcuyi7hacz7qy4jimk4juciarbga5rlkiua4d4a.py
# Topologically Sorted Source Nodes: [input_3, input_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
# Source node to ATen node mapping:
#   input_3 => add_9, mul_13, mul_7, sub_1, unsqueeze_2, unsqueeze_3
#   input_4 => add_10, erf, mul_14, mul_15, mul_16
# Graph fragment:
#   %convolution : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0" = PlaceHolder[target=convolution]
#   %getitem_3 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=getitem_3]
#   %rsqrt_1 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_1]
#   %primals_13 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=primals_13]
#   %primals_14 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=primals_14]
#   %add_9 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0" = PlaceHolder[target=add_9]
#   %sub_1 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %getitem_3), kwargs = {})
#   %mul_7 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_13, -1), kwargs = {})
#   %mul_13 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_2), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_14, -1), kwargs = {})
#   %add_9 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %unsqueeze_3), kwargs = {})
#   %mul_14 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_9, 0.5), kwargs = {})
#   %mul_15 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_9, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_15,), kwargs = {})
#   %add_10 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_16 : Tensor "f32[500, 64, 16384][1048576, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, %add_10), kwargs = {})
#   return %add_9,%mul_16
triton_poi_fused__native_batch_norm_legit_functional_gelu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_gelu_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_gelu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 6291456000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_gelu_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = 0.7071067811865476
    tmp12 = tmp8 * tmp11
    tmp13 = libdevice.erf(tmp12)
    tmp14 = 1.0
    tmp15 = tmp13 + tmp14
    tmp16 = tmp10 * tmp15
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/x2/cx2f3osmjke5x6drfuigo4hemmg2i5mma4njqk675lyl4ea5slqs.py
# Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_11 => convolution_3
# Graph fragment:
#   %buf44 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0" = PlaceHolder[target=buf44]
#   %primals_30 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=primals_30]
#   %convolution_3 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_36, %primals_29, %primals_30, [1], [0], [1], False, [0], 1), kwargs = {})
#   return %convolution_3
triton_poi_fused_convolution_8 = async_compile.triton('triton_poi_fused_convolution_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 786432000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x1 = ((xindex // 16384) % 8)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/5k/c5k7ezv6kv6qpndqmy2rh3owxpyqshfhmqnx7jilecbclfqd2v6j.py
# Topologically Sorted Source Nodes: [input_12], Original ATen: [aten._native_batch_norm_legit_functional]
# Source node to ATen node mapping:
#   input_12 => var_mean_4
# Graph fragment:
#   %convolution_3 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0" = PlaceHolder[target=convolution_3]
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convolution_3, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   return %buf46,%buf47,%buf48
triton_red_fused__native_batch_norm_legit_functional_9 = async_compile.triton('triton_red_fused__native_batch_norm_legit_functional_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 3, 'num_reduction': 3, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 10368, 'r0_': 262144000}}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_functional_9(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp13_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp13_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp13_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
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
        tmp4 = 0.0
        tmp5 = tl.full(tmp4.shape, 0, tmp4.dtype)
        tmp6 = tl.where(tmp2, tmp4, tmp5)
        tmp7 = 1.0
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp11 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp12 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp13_mean_next, tmp13_m2_next, tmp13_weight_next = triton_helpers.welford_combine(
            tmp13_mean, tmp13_m2, tmp13_weight,
            tmp10, tmp11, tmp12
        )
        tmp13_mean = tl.where(r0_mask & xmask, tmp13_mean_next, tmp13_mean)
        tmp13_m2 = tl.where(r0_mask & xmask, tmp13_m2_next, tmp13_m2)
        tmp13_weight = tl.where(r0_mask & xmask, tmp13_weight_next, tmp13_weight)
    tmp14, tmp15, tmp16 = triton_helpers.welford(tmp13_mean, tmp13_m2, tmp13_weight, 1)
    tmp13 = tmp14[:, None]
    tmp17 = tmp15[:, None]
    tmp18 = tmp16[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp17, xmask)
    tl.store(out_ptr2 + (x3), tmp18, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/yx/cyxwds4ssgfczkqttxt3z6ex5ahouu27xuxsyovlaweqxnxnk3ub.py
# Topologically Sorted Source Nodes: [input_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
# Source node to ATen node mapping:
#   input_12 => add_24, add_25, add_26, mul_38, mul_39, mul_40, mul_41, mul_42, rsqrt_4, squeeze_12, squeeze_14, var_mean_4
# Graph fragment:
#   %buf46 : Tensor "f32[1, 8, 1, 54][432, 1, 432, 8]cuda:0" = PlaceHolder[target=buf46]
#   %buf47 : Tensor "f32[1, 8, 1, 54][432, 1, 432, 8]cuda:0" = PlaceHolder[target=buf47]
#   %buf48 : Tensor "f32[1, 8, 1, 54][432, 1, 432, 8]cuda:0" = PlaceHolder[target=buf48]
#   %buf50 : Tensor "f32[1, 8, 1][8, 1, 8]cuda:0" = PlaceHolder[target=buf50]
#   %copy__14 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=copy__14]
#   %add_26 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=add_26]
#   %getitem_9 : Tensor "f32[1, 8, 1][8, 1, 1]cuda:0" = PlaceHolder[target=getitem_9]
#   %copy__13 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=copy__13]
#   %add_25 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=add_25]
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convolution_3, [0, 2]), kwargs = {correction: 0, keepdim: True})
#   %add_24 : Tensor "f32[1, 8, 1][8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_4 : Tensor "f32[1, 8, 1][8, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_24,), kwargs = {})
#   %squeeze_12 : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_9, [0, 2]), kwargs = {})
#   %mul_38 : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_12, 0.1), kwargs = {})
#   %mul_39 : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_32, 0.9), kwargs = {})
#   %add_25 : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %mul_39), kwargs = {})
#   %squeeze_14 : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dims](args = (%getitem_8, [0, 2]), kwargs = {})
#   %mul_40 : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_14, 1.0000001220703274), kwargs = {})
#   %mul_41 : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_40, 0.1), kwargs = {})
#   %mul_42 : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_33, 0.9), kwargs = {})
#   %add_26 : Tensor "f32[8][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %mul_42), kwargs = {})
#   %copy__13 : Tensor "f32[8][1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_32, %add_25), kwargs = {})
#   %copy__14 : Tensor "f32[8][1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_33, %add_26), kwargs = {})
#   return %getitem_9,%buf50,%rsqrt_4,%add_26,%buf94,%add_25,%buf91
triton_per_fused__native_batch_norm_legit_functional_copy__10 = async_compile.triton('triton_per_fused__native_batch_norm_legit_functional_copy__10', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_copy__10', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 4, 'num_reduction': 2, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 5504, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_functional_copy__10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + 8*r0_1), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 8*r0_1), r0_mask & xmask, other=0.0)
    tmp25 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp3, 0)
    tmp8 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp9 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 8192000.0
    tmp17 = (tmp14 / tmp16)
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = 1.0000001220703274
    tmp22 = tmp17 * tmp21
    tmp23 = 0.1
    tmp24 = tmp22 * tmp23
    tmp26 = 0.9
    tmp27 = tmp25 * tmp26
    tmp28 = tmp24 + tmp27
    tmp29 = tmp13 * tmp23
    tmp31 = tmp30 * tmp26
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp28, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/gx/cgx5nfdhnlpcam7sjis4m5cyqxowuejzuqk5lmjsqiyfljcwwue2.py
# Topologically Sorted Source Nodes: [input_12, input_13], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
# Source node to ATen node mapping:
#   input_12 => add_27, mul_37, mul_43, sub_4, unsqueeze_8, unsqueeze_9
#   input_13 => add_28, erf_3, mul_44, mul_45, mul_46
# Graph fragment:
#   %convolution_3 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0" = PlaceHolder[target=convolution_3]
#   %getitem_9 : Tensor "f32[1, 8, 1][8, 1, 1]cuda:0" = PlaceHolder[target=getitem_9]
#   %rsqrt_4 : Tensor "f32[1, 8, 1][8, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_4]
#   %primals_34 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=primals_34]
#   %primals_35 : Tensor "f32[8][1]cuda:0" = PlaceHolder[target=primals_35]
#   %add_27 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0" = PlaceHolder[target=add_27]
#   %sub_4 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %getitem_9), kwargs = {})
#   %mul_37 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_4), kwargs = {})
#   %unsqueeze_8 : Tensor "f32[8, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_34, -1), kwargs = {})
#   %mul_43 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_8), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[8, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_35, -1), kwargs = {})
#   %add_27 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_43, %unsqueeze_9), kwargs = {})
#   %mul_44 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, 0.5), kwargs = {})
#   %mul_45 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, 0.7071067811865476), kwargs = {})
#   %erf_3 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_45,), kwargs = {})
#   %add_28 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_3, 1), kwargs = {})
#   %mul_46 : Tensor "f32[500, 8, 16384][131072, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_44, %add_28), kwargs = {})
#   return %add_27,%mul_46
triton_poi_fused__native_batch_norm_legit_functional_gelu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_functional_gelu_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_gelu_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 786432000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_functional_gelu_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = 0.7071067811865476
    tmp12 = tmp8 * tmp11
    tmp13 = libdevice.erf(tmp12)
    tmp14 = 1.0
    tmp15 = tmp13 + tmp14
    tmp16 = tmp10 * tmp15
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/lg/clguu7budcf5o65de5x7fsxf3qugco5nify7sszvlxipw6exhhs6.py
# Topologically Sorted Source Nodes: [add_], Original ATen: [aten.add, aten.copy_]
# Source node to ATen node mapping:
#   add_ => add
# Graph fragment:
#   %copy_ : Tensor "i64[][]cuda:0" = PlaceHolder[target=copy_]
#   %add : Tensor "i64[][]cuda:0" = PlaceHolder[target=add]
#   %add : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_3, 1), kwargs = {})
#   %copy_ : Tensor "i64[][]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_3, %add), kwargs = {})
#   return %add,%buf56
triton_poi_fused_add_copy__12 = async_compile.triton('triton_poi_fused_add_copy__12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr1': '*i64', 'xnumel': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_copy__12', 'mutated_arg_names': ['in_ptr0', 'out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_copy__12(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 + tmp2
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32).broadcast_to(XBLOCK)), tmp3, None)
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
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35 = args
        args.clear()
        assert_size_stride(primals_1, (500, 128, 16), (2048, 16, 1))
        assert_size_stride(primals_2, (500, 128, 128, 4), (65536, 512, 4, 1))
        assert_size_stride(primals_3, (), ())
        assert_size_stride(primals_4, (4, ), (1, ))
        assert_size_stride(primals_5, (4, ), (1, ))
        assert_size_stride(primals_6, (4, ), (1, ))
        assert_size_stride(primals_7, (4, ), (1, ))
        assert_size_stride(primals_8, (64, 4, 1), (4, 1, 1))
        assert_size_stride(primals_9, (64, ), (1, ))
        assert_size_stride(primals_10, (), ())
        assert_size_stride(primals_11, (64, ), (1, ))
        assert_size_stride(primals_12, (64, ), (1, ))
        assert_size_stride(primals_13, (64, ), (1, ))
        assert_size_stride(primals_14, (64, ), (1, ))
        assert_size_stride(primals_15, (64, 64, 1), (64, 1, 1))
        assert_size_stride(primals_16, (64, ), (1, ))
        assert_size_stride(primals_17, (), ())
        assert_size_stride(primals_18, (64, ), (1, ))
        assert_size_stride(primals_19, (64, ), (1, ))
        assert_size_stride(primals_20, (64, ), (1, ))
        assert_size_stride(primals_21, (64, ), (1, ))
        assert_size_stride(primals_22, (64, 64, 1), (64, 1, 1))
        assert_size_stride(primals_23, (64, ), (1, ))
        assert_size_stride(primals_24, (), ())
        assert_size_stride(primals_25, (64, ), (1, ))
        assert_size_stride(primals_26, (64, ), (1, ))
        assert_size_stride(primals_27, (64, ), (1, ))
        assert_size_stride(primals_28, (64, ), (1, ))
        assert_size_stride(primals_29, (8, 64, 1), (64, 1, 1))
        assert_size_stride(primals_30, (8, ), (1, ))
        assert_size_stride(primals_31, (), ())
        assert_size_stride(primals_32, (8, ), (1, ))
        assert_size_stride(primals_33, (8, ), (1, ))
        assert_size_stride(primals_34, (8, ), (1, ))
        assert_size_stride(primals_35, (8, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((1, 4, 1, 864, 32), (110592, 1, 110592, 128, 4), torch.float32)
            buf1 = empty_strided_cuda((1, 4, 1, 864, 32), (110592, 1, 110592, 128, 4), torch.float32)
            buf2 = empty_strided_cuda((1, 4, 1, 864, 32), (110592, 1, 110592, 128, 4), torch.float32)
            # Topologically Sorted Source Nodes: [view, U, input_1], Original ATen: [aten.view, aten.transpose, aten.clone, aten._native_batch_norm_legit_functional]
            # [Provenance debug handles] triton_red_fused__native_batch_norm_legit_functional_clone_transpose_view_0:1
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_clone_transpose_view_0.run(primals_2, buf0, buf1, buf2, 110592, 297, stream=stream0)
            buf3 = empty_strided_cuda((1, 4, 1, 864), (3456, 864, 3456, 1), torch.float32)
            buf4 = empty_strided_cuda((1, 4, 1, 864), (3456, 1, 3456, 4), torch.float32)
            buf5 = empty_strided_cuda((1, 4, 1, 864), (3456, 1, 3456, 4), torch.float32)
            # Topologically Sorted Source Nodes: [view, U, input_1], Original ATen: [aten.view, aten.transpose, aten.clone, aten._native_batch_norm_legit_functional]
            # [Provenance debug handles] triton_per_fused__native_batch_norm_legit_functional_clone_transpose_view_1:2
            stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_functional_clone_transpose_view_1.run(buf0, buf1, buf2, buf3, buf4, buf5, 3456, 32, stream=stream0)
            del buf0
            del buf1
            del buf2
            buf6 = empty_strided_cuda((1, 4, 1), (4, 1, 4), torch.float32)
            buf7 = empty_strided_cuda((1, 4, 1), (4, 1, 4), torch.float32)
            buf9 = empty_strided_cuda((1, 4, 1), (4, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view, U, input_1], Original ATen: [aten.view, aten.transpose, aten.clone, aten._native_batch_norm_legit_functional, aten.copy_]
            # [Provenance debug handles] triton_red_fused__native_batch_norm_legit_functional_clone_copy__transpose_view_2:3
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_clone_copy__transpose_view_2.run(buf3, buf4, buf5, primals_5, primals_4, buf6, buf7, buf9, primals_5, primals_4, 4, 864, stream=stream0)
            del buf3
            del buf4
            del buf5
            del primals_4
            del primals_5
            buf10 = empty_strided_cuda((500, 4, 16384), (65536, 16384, 1), torch.float32)
            # Topologically Sorted Source Nodes: [view, U, input_1], Original ATen: [aten.view, aten.transpose, aten.clone, aten._native_batch_norm_legit_functional]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_functional_clone_transpose_view_3:4
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_clone_transpose_view_3.run(primals_2, buf6, buf7, primals_6, primals_7, buf10, 2000, 16384, stream=stream0)
            del buf7
            del primals_6
            del primals_7
            # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:5
            buf11 = extern_kernels.convolution(buf10, primals_8, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
            assert_size_stride(buf11, (500, 64, 16384), (1048576, 16384, 1), 'torch.ops.aten.convolution.default')
            buf12 = buf11; del buf11  # reuse
            # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_4:6
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_4.run(buf12, primals_9, 524288000, stream=stream0)
            del primals_9
            buf13 = empty_strided_cuda((1, 64, 1, 63), (4032, 1, 4032, 64), torch.float32)
            buf14 = empty_strided_cuda((1, 64, 1, 63), (4032, 1, 4032, 64), torch.float32)
            buf15 = empty_strided_cuda((1, 64, 1, 63), (4032, 1, 4032, 64), torch.float32)
            # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._native_batch_norm_legit_functional]
            # [Provenance debug handles] triton_red_fused__native_batch_norm_legit_functional_5:7
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_5.run(buf12, buf13, buf14, buf15, 4032, 130032, stream=stream0)
            buf16 = empty_strided_cuda((1, 64, 1), (64, 1, 1), torch.float32)
            buf19 = empty_strided_cuda((1, 64, 1), (64, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
            # [Provenance debug handles] triton_per_fused__native_batch_norm_legit_functional_copy__6:8
            stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_functional_copy__6.run(buf13, buf14, buf15, primals_12, primals_11, buf16, buf19, primals_12, primals_11, 64, 63, stream=stream0)
            del primals_11
            del primals_12
            buf20 = empty_strided_cuda((500, 64, 16384), (1048576, 16384, 1), torch.float32)
            buf21 = buf20; del buf20  # reuse
            # Topologically Sorted Source Nodes: [input_3, input_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_functional_gelu_7:9
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_gelu_7.run(buf21, buf12, buf16, buf19, primals_13, primals_14, 524288000, stream=stream0)
            # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:10
            buf22 = extern_kernels.convolution(buf21, primals_15, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
            assert_size_stride(buf22, (500, 64, 16384), (1048576, 16384, 1), 'torch.ops.aten.convolution.default')
            buf23 = buf22; del buf22  # reuse
            # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_4:11
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_4.run(buf23, primals_16, 524288000, stream=stream0)
            del primals_16
            buf24 = buf15; del buf15  # reuse
            buf25 = buf14; del buf14  # reuse
            buf26 = buf13; del buf13  # reuse
            # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten._native_batch_norm_legit_functional]
            # [Provenance debug handles] triton_red_fused__native_batch_norm_legit_functional_5:12
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_5.run(buf23, buf24, buf25, buf26, 4032, 130032, stream=stream0)
            buf27 = empty_strided_cuda((1, 64, 1), (64, 1, 1), torch.float32)
            buf30 = empty_strided_cuda((1, 64, 1), (64, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
            # [Provenance debug handles] triton_per_fused__native_batch_norm_legit_functional_copy__6:13
            stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_functional_copy__6.run(buf24, buf25, buf26, primals_19, primals_18, buf27, buf30, primals_19, primals_18, 64, 63, stream=stream0)
            del primals_18
            del primals_19
            buf31 = empty_strided_cuda((500, 64, 16384), (1048576, 16384, 1), torch.float32)
            buf32 = buf31; del buf31  # reuse
            # Topologically Sorted Source Nodes: [input_6, input_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_functional_gelu_7:14
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_gelu_7.run(buf32, buf23, buf27, buf30, primals_20, primals_21, 524288000, stream=stream0)
            # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:15
            buf33 = extern_kernels.convolution(buf32, primals_22, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
            assert_size_stride(buf33, (500, 64, 16384), (1048576, 16384, 1), 'torch.ops.aten.convolution.default')
            buf34 = buf33; del buf33  # reuse
            # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_4:16
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_4.run(buf34, primals_23, 524288000, stream=stream0)
            del primals_23
            buf35 = buf26; del buf26  # reuse
            buf36 = buf25; del buf25  # reuse
            buf37 = buf24; del buf24  # reuse
            # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten._native_batch_norm_legit_functional]
            # [Provenance debug handles] triton_red_fused__native_batch_norm_legit_functional_5:17
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_5.run(buf34, buf35, buf36, buf37, 4032, 130032, stream=stream0)
            buf38 = empty_strided_cuda((1, 64, 1), (64, 1, 1), torch.float32)
            buf41 = empty_strided_cuda((1, 64, 1), (64, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
            # [Provenance debug handles] triton_per_fused__native_batch_norm_legit_functional_copy__6:18
            stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_functional_copy__6.run(buf35, buf36, buf37, primals_26, primals_25, buf38, buf41, primals_26, primals_25, 64, 63, stream=stream0)
            del buf35
            del buf36
            del buf37
            del primals_25
            del primals_26
            buf42 = empty_strided_cuda((500, 64, 16384), (1048576, 16384, 1), torch.float32)
            buf43 = buf42; del buf42  # reuse
            # Topologically Sorted Source Nodes: [input_9, input_10], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_functional_gelu_7:19
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_gelu_7.run(buf43, buf34, buf38, buf41, primals_27, primals_28, 524288000, stream=stream0)
            # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:20
            buf44 = extern_kernels.convolution(buf43, primals_29, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
            assert_size_stride(buf44, (500, 8, 16384), (131072, 16384, 1), 'torch.ops.aten.convolution.default')
            buf45 = buf44; del buf44  # reuse
            # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_8:21
            stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_8.run(buf45, primals_30, 65536000, stream=stream0)
            del primals_30
            buf46 = empty_strided_cuda((1, 8, 1, 54), (432, 1, 432, 8), torch.float32)
            buf47 = empty_strided_cuda((1, 8, 1, 54), (432, 1, 432, 8), torch.float32)
            buf48 = empty_strided_cuda((1, 8, 1, 54), (432, 1, 432, 8), torch.float32)
            # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten._native_batch_norm_legit_functional]
            # [Provenance debug handles] triton_red_fused__native_batch_norm_legit_functional_9:22
            stream0 = get_raw_stream(0)
            triton_red_fused__native_batch_norm_legit_functional_9.run(buf45, buf46, buf47, buf48, 432, 151704, stream=stream0)
            buf49 = empty_strided_cuda((1, 8, 1), (8, 1, 1), torch.float32)
            buf52 = empty_strided_cuda((1, 8, 1), (8, 1, 1), torch.float32)
            # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.copy_]
            # [Provenance debug handles] triton_per_fused__native_batch_norm_legit_functional_copy__10:23
            stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_functional_copy__10.run(buf46, buf47, buf48, primals_33, primals_32, buf49, buf52, primals_33, primals_32, 8, 54, stream=stream0)
            del buf46
            del buf47
            del buf48
            del primals_32
            del primals_33
            buf53 = empty_strided_cuda((500, 8, 16384), (131072, 16384, 1), torch.float32)
            buf54 = buf53; del buf53  # reuse
            # Topologically Sorted Source Nodes: [input_12, input_13], Original ATen: [aten._native_batch_norm_legit_functional, aten.gelu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_functional_gelu_11:24
            stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_functional_gelu_11.run(buf54, buf45, buf49, buf52, primals_34, primals_35, 65536000, stream=stream0)
            # Topologically Sorted Source Nodes: [add_], Original ATen: [aten.add, aten.copy_]
            # [Provenance debug handles] triton_poi_fused_add_copy__12:25
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_copy__12.run(primals_3, primals_3, 1, stream=stream0)
            del primals_3
            # Topologically Sorted Source Nodes: [add__1], Original ATen: [aten.add, aten.copy_]
            # [Provenance debug handles] triton_poi_fused_add_copy__12:26
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_copy__12.run(primals_10, primals_10, 1, stream=stream0)
            del primals_10
            # Topologically Sorted Source Nodes: [add__2], Original ATen: [aten.add, aten.copy_]
            # [Provenance debug handles] triton_poi_fused_add_copy__12:27
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_copy__12.run(primals_17, primals_17, 1, stream=stream0)
            del primals_17
            # Topologically Sorted Source Nodes: [add__3], Original ATen: [aten.add, aten.copy_]
            # [Provenance debug handles] triton_poi_fused_add_copy__12:28
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_copy__12.run(primals_24, primals_24, 1, stream=stream0)
            del primals_24
            # Topologically Sorted Source Nodes: [add__4], Original ATen: [aten.add, aten.copy_]
            # [Provenance debug handles] triton_poi_fused_add_copy__12:29
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_copy__12.run(primals_31, primals_31, 1, stream=stream0)
            del primals_31
        return (reinterpret_tensor(primals_1, (500, 128, 1, 16), (2048, 16, 16, 1), 0), reinterpret_tensor(buf54, (4000, 128, 128), (16384, 128, 1), 0), primals_2, primals_8, primals_13, primals_14, primals_15, primals_20, primals_21, primals_22, primals_27, primals_28, primals_29, primals_34, primals_35, reinterpret_tensor(buf9, (4, ), (1, ), 0), buf10, buf12, buf16, buf19, buf21, buf23, buf27, buf30, buf32, buf34, buf38, buf41, buf43, buf45, buf49, buf52, reinterpret_tensor(buf6, (1, 4, 1), (4, 1, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((500, 128, 16), (2048, 16, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((500, 128, 128, 4), (65536, 512, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_18 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((8, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_32 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
