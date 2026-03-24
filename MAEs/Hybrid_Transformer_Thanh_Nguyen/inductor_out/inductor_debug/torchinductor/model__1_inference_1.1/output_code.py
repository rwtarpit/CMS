# AOT ID: ['1_inference']
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


# kernel path: /traces/inductor_cache/cl/ccl65bsgahtw7ic2m2hygcyn6qrnp5m7sbtsn4yj6d5hqdkto5v3.py
# Topologically Sorted Source Nodes: [eta, unsqueeze, unsqueeze_1, eta_diff, pow_1, phi, unsqueeze_2, unsqueeze_3, sub_1, add, mod, phi_diff, pow_2, add_4, delta, pT, unsqueeze_4, unsqueeze_5, min_pT, kT, unsqueeze_6, unsqueeze_7, pT_sum, add_5, z], Original ATen: [aten.select, aten.unsqueeze, aten.sub, aten.pow, aten.add, aten.remainder, aten.sqrt, aten.minimum, aten.mul, aten.div]
# Source node to ATen node mapping:
#   add => add
#   add_4 => add_4
#   add_5 => add_5
#   delta => sqrt
#   eta => select_2
#   eta_diff => sub
#   kT => mul_3
#   min_pT => minimum
#   mod => remainder
#   pT => select_1
#   pT_sum => add_1
#   phi => select_3
#   phi_diff => sub_2
#   pow_1 => pow_1
#   pow_2 => pow_2
#   sub_1 => sub_1
#   unsqueeze => unsqueeze_3
#   unsqueeze_1 => unsqueeze_4
#   unsqueeze_2 => unsqueeze_5
#   unsqueeze_3 => unsqueeze_6
#   unsqueeze_4 => unsqueeze_7
#   unsqueeze_5 => unsqueeze_8
#   unsqueeze_6 => unsqueeze_9
#   unsqueeze_7 => unsqueeze_10
#   z => div
# Graph fragment:
#   %arg0_1 : Tensor "f32[500, 128, 4][512, 4, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %sqrt : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=sqrt]
#   %select_2 : Tensor "f32[500, 128][512, 4]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 2, 1), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[500, 128, 1][512, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_2, 2), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[500, 1, 128][512, 512, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_2, 1), kwargs = {})
#   %sub : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_3, %unsqueeze_4), kwargs = {})
#   %pow_1 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub, 2), kwargs = {})
#   %select_3 : Tensor "f32[500, 128][512, 4]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 2, 2), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[500, 128, 1][512, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_3, 2), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[500, 1, 128][512, 512, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_3, 1), kwargs = {})
#   %sub_1 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_5, %unsqueeze_6), kwargs = {})
#   %add : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_1, 3.141592653589793), kwargs = {})
#   %remainder : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%add, 6.283185307179586), kwargs = {})
#   %sub_2 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%remainder, 3.141592653589793), kwargs = {})
#   %pow_2 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_2, 2), kwargs = {})
#   %add_4 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_1, %pow_2), kwargs = {})
#   %sqrt : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_4,), kwargs = {})
#   %select_1 : Tensor "f32[500, 128][512, 4]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 2, 0), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[500, 128, 1][512, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_1, 2), kwargs = {})
#   %unsqueeze_8 : Tensor "f32[500, 1, 128][512, 512, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_1, 1), kwargs = {})
#   %minimum : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.minimum.default](args = (%unsqueeze_7, %unsqueeze_8), kwargs = {})
#   %mul_3 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%minimum, %sqrt), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[500, 128, 1][512, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_1, 2), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[500, 1, 128][512, 512, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_1, 1), kwargs = {})
#   %add_1 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze_9, %unsqueeze_10), kwargs = {})
#   %add_5 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, 1e-08), kwargs = {})
#   %div : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%minimum, %add_5), kwargs = {})
#   return %sqrt,%mul_3,%div
triton_poi_fused_add_div_minimum_mul_pow_remainder_select_sqrt_sub_unsqueeze_0 = async_compile.triton('triton_poi_fused_add_div_minimum_mul_pow_remainder_select_sqrt_sub_unsqueeze_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_minimum_mul_pow_remainder_select_sqrt_sub_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 3, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 196608000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_minimum_mul_pow_remainder_select_sqrt_sub_unsqueeze_0(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex // 128
    x0 = (xindex % 128)
    x2 = xindex // 16384
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (1 + 4*x3), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4*x0 + 512*x2), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 4*x3), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (2 + 4*x0 + 512*x2), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (4*x3), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr0 + (4*x0 + 512*x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tmp2 * tmp2
    tmp6 = tmp4 - tmp5
    tmp7 = 3.141592653589793
    tmp8 = tmp6 + tmp7
    tmp9 = 6.283185307179586
    tmp10 = (tmp8 % tmp9)
    tmp11 = tl.full([1], 0, tl.int32)
    tmp12 = tmp10 != tmp11
    tmp13 = (libdevice.signbit(tmp10) != 0) if (tmp10).dtype is tl.float32 else tmp10 < 0
    tmp14 = (libdevice.signbit(tmp9) != 0) if (tmp9).dtype is tl.float32 else tmp9 < 0
    tmp15 = tmp13 != tmp14
    tmp16 = tmp12 & tmp15
    tmp17 = tmp10 + tmp9
    tmp18 = tl.where(tmp16, tmp17, tmp10)
    tmp19 = tmp18 - tmp7
    tmp20 = tmp19 * tmp19
    tmp21 = tmp3 + tmp20
    tmp22 = tl.sqrt_rn(tmp21)
    tmp25 = triton_helpers.minimum(tmp23, tmp24)
    tmp26 = tmp25 * tmp22
    tmp27 = tmp23 + tmp24
    tmp28 = 1e-08
    tmp29 = tmp27 + tmp28
    tmp30 = (tmp25 / tmp29)
    tl.store(out_ptr0 + (x4), tmp22, None)
    tl.store(out_ptr1 + (x4), tmp26, None)
    tl.store(out_ptr2 + (x4), tmp30, None)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/6n/c6ntpufssdr4xp625hlrvfxdczygxl5imx22dg7cuvho2p5u35p2.py
# Topologically Sorted Source Nodes: [isnan, any_1], Original ATen: [aten.isnan, aten.any]
# Source node to ATen node mapping:
#   any_1 => any_1
#   isnan => isnan
# Graph fragment:
#   %sqrt : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=sqrt]
#   %isnan : Tensor "b8[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.isnan.default](args = (%sqrt,), kwargs = {})
#   %any_1 : Tensor "b8[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.any.default](args = (%isnan,), kwargs = {})
#   return %buf2
triton_red_fused_any_isnan_1 = async_compile.triton('triton_red_fused_any_isnan_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 32768},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_any_isnan_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 800, 'r0_': 32768000}}
)
@triton.jit
def triton_red_fused_any_isnan_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 400
    r0_numel = 20480
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], False, tl.int1)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 20480*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = libdevice.isnan(tmp0).to(tl.int1)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 | tmp2
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
    tmp5 = _tmp3.to(tl.int8)
    tmp3 = triton_helpers.any(tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/ou/couhnlcky6nvcm4jmbt6hfychm7dadu7vtia4mf4gitiqj6zq3gk.py
# Topologically Sorted Source Nodes: [isnan, any_1], Original ATen: [aten.isnan, aten.any]
# Source node to ATen node mapping:
#   any_1 => any_1
#   isnan => isnan
# Graph fragment:
#   %buf2 : Tensor "b8[400][1]cuda:0" = PlaceHolder[target=buf2]
#   %isnan : Tensor "b8[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.isnan.default](args = (%sqrt,), kwargs = {})
#   %any_1 : Tensor "b8[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.any.default](args = (%isnan,), kwargs = {})
#   return %any_1
triton_per_fused_any_isnan_2 = async_compile.triton('triton_per_fused_any_isnan_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'out_ptr0': '*i1', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_any_isnan_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'r0_': 400}}
)
@triton.jit
def triton_per_fused_any_isnan_2(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 400
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0).to(tl.int1)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask, tmp1, False)
    tmp4 = triton_helpers.any(tmp3, 1)[:, None].to(tl.int1)
    tl.store(out_ptr0 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp4, None)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/4e/c4exb3jrcvifzdczwp6okyvlmvotyd3davkgjprffu2zpnyncge5.py
# Topologically Sorted Source Nodes: [getitem, mask], Original ATen: [aten.select, aten.gt]
# Source node to ATen node mapping:
#   getitem => select
#   mask => gt
# Graph fragment:
#   %arg0_1 : Tensor "f32[500, 128, 4][512, 4, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %select : Tensor "f32[500, 128][512, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 2, 3), kwargs = {})
#   %gt : Tensor "b8[500, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%select, 0), kwargs = {})
#   return %gt
triton_poi_fused_gt_select_3 = async_compile.triton('triton_poi_fused_gt_select_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gt_select_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 128000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gt_select_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/3o/c3oz33u6k3au4eagoklqt2cx3aaxgwa3xavc5g6rlv5adt7p2tcw.py
# Topologically Sorted Source Nodes: [eta, phi, pT, cos, px, momentum, sin, py, sinh, pz, unsqueeze_10, unsqueeze_11, momentum_sum], Original ATen: [aten.select, aten.cos, aten.mul, aten.stack, aten.sin, aten.sinh, aten.unsqueeze, aten.add]
# Source node to ATen node mapping:
#   cos => cos
#   eta => select_2
#   momentum => cat, unsqueeze, unsqueeze_1, unsqueeze_2
#   momentum_sum => add_3
#   pT => select_1
#   phi => select_3
#   px => mul
#   py => mul_1
#   pz => mul_2
#   sin => sin
#   sinh => sinh
#   unsqueeze_10 => unsqueeze_13
#   unsqueeze_11 => unsqueeze_14
# Graph fragment:
#   %arg0_1 : Tensor "f32[500, 128, 4][512, 4, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %select_2 : Tensor "f32[500, 128][512, 4]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 2, 1), kwargs = {})
#   %select_3 : Tensor "f32[500, 128][512, 4]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 2, 2), kwargs = {})
#   %select_1 : Tensor "f32[500, 128][512, 4]cuda:0"[num_users=7] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 2, 0), kwargs = {})
#   %cos : Tensor "f32[500, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%select_3,), kwargs = {})
#   %mul : Tensor "f32[500, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_1, %cos), kwargs = {})
#   %unsqueeze : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul, 2), kwargs = {})
#   %sin : Tensor "f32[500, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%select_3,), kwargs = {})
#   %mul_1 : Tensor "f32[500, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_1, %sin), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_1, 2), kwargs = {})
#   %sinh : Tensor "f32[500, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sinh.default](args = (%select_2,), kwargs = {})
#   %mul_2 : Tensor "f32[500, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_1, %sinh), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[500, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_2, 2), kwargs = {})
#   %cat : Tensor "f32[500, 128, 3][384, 3, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze, %unsqueeze_1, %unsqueeze_2], -1), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[500, 128, 1, 3][384, 3, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cat, 2), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[500, 1, 128, 3][384, 384, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cat, 1), kwargs = {})
#   %add_3 : Tensor "f32[500, 128, 128, 3][49152, 384, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze_13, %unsqueeze_14), kwargs = {})
#   return %add_3
triton_poi_fused_add_cos_mul_select_sin_sinh_stack_unsqueeze_4 = async_compile.triton('triton_poi_fused_add_cos_mul_select_sin_sinh_stack_unsqueeze_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cos_mul_select_sin_sinh_stack_unsqueeze_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 12, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 196608000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cos_mul_select_sin_sinh_stack_unsqueeze_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 3)
    x5 = xindex // 384
    x1 = ((xindex // 3) % 128)
    x3 = xindex // 49152
    x6 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (4*x5), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr0 + (2 + 4*x5), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl_math.cos(tmp6)
    tmp8 = tmp5 * tmp7
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp11 = tmp0 >= tmp3
    tmp12 = tl.full([1], 2, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr0 + (4*x5), tmp14, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr0 + (2 + 4*x5), tmp14, eviction_policy='evict_last', other=0.0)
    tmp17 = tl_math.sin(tmp16)
    tmp18 = tmp15 * tmp17
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp14, tmp18, tmp19)
    tmp21 = tmp0 >= tmp12
    tmp22 = tl.full([1], 3, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tl.load(in_ptr0 + (4*x5), tmp21, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.load(in_ptr0 + (1 + 4*x5), tmp21, eviction_policy='evict_last', other=0.0)
    tmp26 = libdevice.sinh(tmp25)
    tmp27 = tmp24 * tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp21, tmp27, tmp28)
    tmp30 = tl.where(tmp14, tmp20, tmp29)
    tmp31 = tl.where(tmp4, tmp10, tmp30)
    tmp32 = tl.load(in_ptr0 + (4*x1 + 512*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.load(in_ptr0 + (2 + 4*x1 + 512*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp34 = tl_math.cos(tmp33)
    tmp35 = tmp32 * tmp34
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp4, tmp35, tmp36)
    tmp38 = tl.load(in_ptr0 + (4*x1 + 512*x3), tmp14, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.load(in_ptr0 + (2 + 4*x1 + 512*x3), tmp14, eviction_policy='evict_last', other=0.0)
    tmp40 = tl_math.sin(tmp39)
    tmp41 = tmp38 * tmp40
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp14, tmp41, tmp42)
    tmp44 = tl.load(in_ptr0 + (4*x1 + 512*x3), tmp21, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.load(in_ptr0 + (1 + 4*x1 + 512*x3), tmp21, eviction_policy='evict_last', other=0.0)
    tmp46 = libdevice.sinh(tmp45)
    tmp47 = tmp44 * tmp46
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp21, tmp47, tmp48)
    tmp50 = tl.where(tmp14, tmp43, tmp49)
    tmp51 = tl.where(tmp4, tmp37, tmp50)
    tmp52 = tmp31 + tmp51
    tl.store(out_ptr0 + (x6), tmp52, None)
''', device_str='cuda')


# kernel path: /traces/inductor_cache/xa/cxaoom5t5fvfucxyh344kspowaxmu3m4mz2ghbybh5iyzog5vcek.py
# Topologically Sorted Source Nodes: [energy, unsqueeze_8, unsqueeze_9, energy_sum, pow_3, norm, pow_4, m2], Original ATen: [aten.select, aten.unsqueeze, aten.add, aten.pow, aten.linalg_vector_norm, aten.sub]
# Source node to ATen node mapping:
#   energy => select_4
#   energy_sum => add_2
#   m2 => sub_3
#   norm => pow_4, pow_5, sum_1
#   pow_3 => pow_3
#   pow_4 => pow_6
#   unsqueeze_8 => unsqueeze_11
#   unsqueeze_9 => unsqueeze_12
# Graph fragment:
#   %arg0_1 : Tensor "f32[500, 128, 4][512, 4, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %add_3 : Tensor "f32[500, 128, 128, 3][49152, 384, 3, 1]cuda:0" = PlaceHolder[target=add_3]
#   %select_4 : Tensor "f32[500, 128][512, 4]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.select.int](args = (%arg0_1, 2, 3), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[500, 128, 1][512, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_4, 2), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[500, 1, 128][512, 512, 4]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_4, 1), kwargs = {})
#   %add_2 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze_11, %unsqueeze_12), kwargs = {})
#   %pow_3 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_2, 2), kwargs = {})
#   %pow_4 : Tensor "f32[500, 128, 128, 3][49152, 384, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_3, 2), kwargs = {})
#   %sum_1 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_4, [-1]), kwargs = {})
#   %pow_5 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %pow_6 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%pow_5, 2), kwargs = {})
#   %sub_3 : Tensor "f32[500, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%pow_3, %pow_6), kwargs = {})
#   return %sub_3
triton_poi_fused_add_linalg_vector_norm_pow_select_sub_unsqueeze_5 = async_compile.triton('triton_poi_fused_add_linalg_vector_norm_pow_select_sub_unsqueeze_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_linalg_vector_norm_pow_select_sub_unsqueeze_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'D221555120843152094FD7E43096F924AB4102579828DDF622EEF3389D4AEF7F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 65536000}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_linalg_vector_norm_pow_select_sub_unsqueeze_5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex // 128
    x0 = (xindex % 128)
    x2 = xindex // 16384
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (3 + 4*x3), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (3 + 4*x0 + 512*x2), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (3*x4), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (1 + 3*x4), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (2 + 3*x4), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2 * tmp2
    tmp5 = tmp4 * tmp4
    tmp7 = tmp6 * tmp6
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 + tmp10
    tmp12 = tl.sqrt_rn(tmp11)
    tmp13 = tmp12 * tmp12
    tmp14 = tmp3 - tmp13
    tl.store(out_ptr0 + (x4), tmp14, None)
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
        arg0_1, = args
        args.clear()
        assert_size_stride(arg0_1, (500, 128, 4), (512, 4, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.float32)
            buf5 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.float32)
            buf6 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [eta, unsqueeze, unsqueeze_1, eta_diff, pow_1, phi, unsqueeze_2, unsqueeze_3, sub_1, add, mod, phi_diff, pow_2, add_4, delta, pT, unsqueeze_4, unsqueeze_5, min_pT, kT, unsqueeze_6, unsqueeze_7, pT_sum, add_5, z], Original ATen: [aten.select, aten.unsqueeze, aten.sub, aten.pow, aten.add, aten.remainder, aten.sqrt, aten.minimum, aten.mul, aten.div]
            # [Provenance debug handles] triton_poi_fused_add_div_minimum_mul_pow_remainder_select_sqrt_sub_unsqueeze_0:1
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_div_minimum_mul_pow_remainder_select_sqrt_sub_unsqueeze_0.run(arg0_1, buf0, buf5, buf6, 8192000, stream=stream0)
            buf2 = empty_strided_cuda((400, ), (1, ), torch.bool)
            # Topologically Sorted Source Nodes: [isnan, any_1], Original ATen: [aten.isnan, aten.any]
            # [Provenance debug handles] triton_red_fused_any_isnan_1:2
            stream0 = get_raw_stream(0)
            triton_red_fused_any_isnan_1.run(buf0, buf2, 400, 20480, stream=stream0)
            buf3 = empty_strided_cuda((), (), torch.bool)
            # Topologically Sorted Source Nodes: [isnan, any_1], Original ATen: [aten.isnan, aten.any]
            # [Provenance debug handles] triton_per_fused_any_isnan_2:3
            stream0 = get_raw_stream(0)
            triton_per_fused_any_isnan_2.run(buf2, buf3, 1, 400, stream=stream0)
            del buf2
            buf4 = empty_strided_cuda((500, 128), (128, 1), torch.bool)
            # Topologically Sorted Source Nodes: [getitem, mask], Original ATen: [aten.select, aten.gt]
            # [Provenance debug handles] triton_poi_fused_gt_select_3:4
            stream0 = get_raw_stream(0)
            triton_poi_fused_gt_select_3.run(arg0_1, buf4, 64000, stream=stream0)
            buf7 = empty_strided_cuda((500, 128, 128, 3), (49152, 384, 3, 1), torch.float32)
            # Topologically Sorted Source Nodes: [eta, phi, pT, cos, px, momentum, sin, py, sinh, pz, unsqueeze_10, unsqueeze_11, momentum_sum], Original ATen: [aten.select, aten.cos, aten.mul, aten.stack, aten.sin, aten.sinh, aten.unsqueeze, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_cos_mul_select_sin_sinh_stack_unsqueeze_4:5
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_cos_mul_select_sin_sinh_stack_unsqueeze_4.run(arg0_1, buf7, 24576000, stream=stream0)
            buf8 = empty_strided_cuda((500, 128, 128), (16384, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [energy, unsqueeze_8, unsqueeze_9, energy_sum, pow_3, norm, pow_4, m2], Original ATen: [aten.select, aten.unsqueeze, aten.add, aten.pow, aten.linalg_vector_norm, aten.sub]
            # [Provenance debug handles] triton_poi_fused_add_linalg_vector_norm_pow_select_sub_unsqueeze_5:6
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_linalg_vector_norm_pow_select_sub_unsqueeze_5.run(arg0_1, buf7, buf8, 8192000, stream=stream0)
            del arg0_1
            del buf7
        return (buf3, buf4, buf0, buf5, buf6, buf8, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((500, 128, 4), (512, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
