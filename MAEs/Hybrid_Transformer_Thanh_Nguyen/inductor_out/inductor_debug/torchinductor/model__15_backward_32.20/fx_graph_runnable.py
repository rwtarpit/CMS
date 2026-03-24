
import os
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/traces/inductor_cache'
os.environ['TORCH_LOGS'] = 'graph_breaks,recompiles,graph'
os.environ['TORCHDYNAMO_VERBOSE'] = '1'
os.environ['TRITON_CACHE_DIR'] = '/traces/inductor_cache/triton/0'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.debug = True
torch._inductor.config.deterministic = False
torch._inductor.config.triton.store_cubin = False
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._inductor.config.trace.debug_dir = '/traces/inductor_debug'
torch._inductor.config.test_configs.runtime_triton_dtype_assert = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = False
torch._functorch.config.selective_decompose = False



isolate_fails_code_str = None





if "__compile_source__" in globals():
    import inspect as __after_aot_inspect
    import linecache as __after_aot_linecache
    __after_aot_filename = __after_aot_inspect.currentframe().f_code.co_filename
    __after_aot_linecache.cache[__after_aot_filename] = (
        len(__compile_source__),
        None,
        __compile_source__.splitlines(True),
        __after_aot_filename,
    )
# torch version: 2.10.0+cu128
# torch cuda version: 12.8
# torch git version: 449b1768410104d3ed79d3bcfe4ba1d65c7f22c0


# CUDA Info: 
# nvcc not found
# GPU Hardware Info: 
# NVIDIA A100-SXM4-40GB : 2 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, primals_2, primals_5, primals_7, primals_9, primals_13, primals_15, primals_17, primals_19, primals_21, view, gt, mul_2, view_2, baddbmm, amax, sum_1, gt_1, view_10, addmm_1, getitem_3, rsqrt_1, gt_2, mul_11, view_13, addmm_2, gt_3, getitem_7, rsqrt_3, view_15, gt_4, div_2, permute_27, permute_28, permute_29, permute_30, div_4, tangents_1):
        convert_element_type = torch.ops.prims.convert_element_type.default(gt_4, torch.float32);  gt_4 = None
        mul_22 = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
        mul_23 = torch.ops.aten.mul.Tensor(tangents_1, mul_22);  mul_22 = None
        view_17 = torch.ops.aten.view.default(mul_23, [64000, 128]);  mul_23 = None
        permute_12 = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
        permute_13 = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        mm_1 = torch.ops.aten.mm.default(view_17, permute_13);  permute_13 = None
        permute_14 = torch.ops.aten.permute.default(view_17, [1, 0])
        mm_2 = torch.ops.aten.mm.default(permute_14, view_15);  permute_14 = view_15 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(view_17, [0], True);  view_17 = None
        view_18 = torch.ops.aten.view.default(sum_2, [128]);  sum_2 = None
        view_19 = torch.ops.aten.view.default(mm_1, [500, 128, 512]);  mm_1 = None
        mul_25 = torch.ops.aten.mul.Tensor(view_19, primals_19);  primals_19 = None
        mul_26 = torch.ops.aten.mul.Tensor(mul_25, 512)
        sum_3 = torch.ops.aten.sum.dim_IntList(mul_25, [2], True)
        view_14 = torch.ops.aten.view.default(addmm_2, [500, 128, 512]);  addmm_2 = None
        mul_13 = torch.ops.aten.mul.Tensor(view_14, 0.5)
        mul_14 = torch.ops.aten.mul.Tensor(view_14, 0.7071067811865476)
        erf = torch.ops.aten.erf.default(mul_14);  mul_14 = None
        add_10 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_15 = torch.ops.aten.mul.Tensor(mul_13, add_10);  mul_13 = None
        mul_16 = torch.ops.aten.mul.Tensor(gt_3, mul_15);  mul_15 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_16, 1.1111111111111112);  mul_16 = None
        sub_4 = torch.ops.aten.sub.Tensor(mul_17, getitem_7);  mul_17 = getitem_7 = None
        mul_18 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = None
        mul_27 = torch.ops.aten.mul.Tensor(mul_25, mul_18);  mul_25 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(mul_27, [2], True);  mul_27 = None
        mul_28 = torch.ops.aten.mul.Tensor(mul_18, sum_4);  sum_4 = None
        sub_6 = torch.ops.aten.sub.Tensor(mul_26, sum_3);  mul_26 = sum_3 = None
        sub_7 = torch.ops.aten.sub.Tensor(sub_6, mul_28);  sub_6 = mul_28 = None
        div_1 = torch.ops.aten.div.Tensor(rsqrt_3, 512);  rsqrt_3 = None
        mul_29 = torch.ops.aten.mul.Tensor(div_1, sub_7);  div_1 = sub_7 = None
        mul_30 = torch.ops.aten.mul.Tensor(view_19, mul_18);  mul_18 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(mul_30, [0, 1]);  mul_30 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(view_19, [0, 1]);  view_19 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(gt_3, torch.float32);  gt_3 = None
        mul_31 = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_32 = torch.ops.aten.mul.Tensor(mul_29, mul_31);  mul_29 = mul_31 = None
        mul_34 = torch.ops.aten.mul.Tensor(add_10, 0.5);  add_10 = None
        mul_35 = torch.ops.aten.mul.Tensor(view_14, view_14)
        mul_36 = torch.ops.aten.mul.Tensor(mul_35, -0.5);  mul_35 = None
        exp_1 = torch.ops.aten.exp.default(mul_36);  mul_36 = None
        mul_37 = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
        mul_38 = torch.ops.aten.mul.Tensor(view_14, mul_37);  view_14 = mul_37 = None
        add_15 = torch.ops.aten.add.Tensor(mul_34, mul_38);  mul_34 = mul_38 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_32, add_15);  mul_32 = add_15 = None
        view_20 = torch.ops.aten.view.default(mul_39, [64000, 512]);  mul_39 = None
        permute_11 = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
        permute_17 = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        mm_3 = torch.ops.aten.mm.default(view_20, permute_17);  permute_17 = None
        permute_18 = torch.ops.aten.permute.default(view_20, [1, 0])
        mm_4 = torch.ops.aten.mm.default(permute_18, view_13);  permute_18 = view_13 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(view_20, [0], True);  view_20 = None
        view_21 = torch.ops.aten.view.default(sum_7, [512]);  sum_7 = None
        view_22 = torch.ops.aten.view.default(mm_3, [500, 128, 128]);  mm_3 = None
        mul_41 = torch.ops.aten.mul.Tensor(view_22, primals_15);  primals_15 = None
        mul_42 = torch.ops.aten.mul.Tensor(mul_41, 128)
        sum_8 = torch.ops.aten.sum.dim_IntList(mul_41, [2], True)
        mul_43 = torch.ops.aten.mul.Tensor(mul_41, mul_11);  mul_41 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(mul_43, [2], True);  mul_43 = None
        mul_44 = torch.ops.aten.mul.Tensor(mul_11, sum_9);  sum_9 = None
        sub_9 = torch.ops.aten.sub.Tensor(mul_42, sum_8);  mul_42 = sum_8 = None
        sub_10 = torch.ops.aten.sub.Tensor(sub_9, mul_44);  sub_9 = mul_44 = None
        mul_45 = torch.ops.aten.mul.Tensor(div_2, sub_10);  div_2 = sub_10 = None
        mul_46 = torch.ops.aten.mul.Tensor(view_22, mul_11);  mul_11 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(mul_46, [0, 1]);  mul_46 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(view_22, [0, 1]);  view_22 = None
        add_16 = torch.ops.aten.add.Tensor(tangents_1, mul_45);  tangents_1 = mul_45 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_47 = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
        mul_48 = torch.ops.aten.mul.Tensor(add_16, mul_47);  mul_47 = None
        view_11 = torch.ops.aten.view.default(addmm_1, [128, 500, 128]);  addmm_1 = None
        permute_10 = torch.ops.aten.permute.default(view_11, [1, 0, 2]);  view_11 = None
        sub_11 = torch.ops.aten.sub.Tensor(permute_10, getitem_3);  permute_10 = getitem_3 = None
        mul_49 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_1);  sub_11 = None
        mul_50 = torch.ops.aten.mul.Tensor(mul_48, primals_13);  primals_13 = None
        mul_51 = torch.ops.aten.mul.Tensor(mul_50, 128)
        sum_12 = torch.ops.aten.sum.dim_IntList(mul_50, [2], True)
        mul_52 = torch.ops.aten.mul.Tensor(mul_50, mul_49);  mul_50 = None
        sum_13 = torch.ops.aten.sum.dim_IntList(mul_52, [2], True);  mul_52 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_49, sum_13);  sum_13 = None
        sub_12 = torch.ops.aten.sub.Tensor(mul_51, sum_12);  mul_51 = sum_12 = None
        sub_13 = torch.ops.aten.sub.Tensor(sub_12, mul_53);  sub_12 = mul_53 = None
        div_3 = torch.ops.aten.div.Tensor(rsqrt_1, 128);  rsqrt_1 = None
        mul_54 = torch.ops.aten.mul.Tensor(div_3, sub_13);  div_3 = sub_13 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul_48, mul_49);  mul_49 = None
        sum_14 = torch.ops.aten.sum.dim_IntList(mul_55, [0, 1]);  mul_55 = None
        sum_15 = torch.ops.aten.sum.dim_IntList(mul_48, [0, 1]);  mul_48 = None
        permute_21 = torch.ops.aten.permute.default(mul_54, [1, 0, 2]);  mul_54 = None
        clone_8 = torch.ops.aten.clone.default(permute_21, memory_format = torch.contiguous_format);  permute_21 = None
        view_23 = torch.ops.aten.view.default(clone_8, [64000, 128]);  clone_8 = None
        permute_9 = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        permute_22 = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_5 = torch.ops.aten.mm.default(view_23, permute_22);  permute_22 = None
        permute_23 = torch.ops.aten.permute.default(view_23, [1, 0])
        mm_6 = torch.ops.aten.mm.default(permute_23, view_10);  permute_23 = view_10 = None
        sum_16 = torch.ops.aten.sum.dim_IntList(view_23, [0], True);  view_23 = None
        view_24 = torch.ops.aten.view.default(sum_16, [128]);  sum_16 = None
        view_25 = torch.ops.aten.view.default(mm_5, [128, 4000, 16]);  mm_5 = None
        permute_26 = torch.ops.aten.permute.default(view_25, [1, 0, 2]);  view_25 = None
        bmm_1 = torch.ops.aten.bmm.default(permute_27, permute_26);  permute_27 = None
        bmm_2 = torch.ops.aten.bmm.default(permute_26, permute_28);  permute_26 = permute_28 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_56 = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.1111111111111112);  convert_element_type_3 = None
        mul_57 = torch.ops.aten.mul.Tensor(bmm_2, mul_56);  bmm_2 = mul_56 = None
        sub_1 = torch.ops.aten.sub.Tensor(baddbmm, amax);  baddbmm = amax = None
        exp = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        div = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        mul_58 = torch.ops.aten.mul.Tensor(mul_57, div);  mul_57 = None
        sum_17 = torch.ops.aten.sum.dim_IntList(mul_58, [-1], True)
        neg = torch.ops.aten.neg.default(div);  div = None
        fma = torch.ops.prims.fma.default(neg, sum_17, mul_58);  neg = sum_17 = mul_58 = None
        bmm_3 = torch.ops.aten.bmm.default(fma, permute_29);  permute_29 = None
        bmm_4 = torch.ops.aten.bmm.default(permute_30, fma);  permute_30 = None
        permute_31 = torch.ops.aten.permute.default(bmm_4, [0, 2, 1]);  bmm_4 = None
        mul_59 = torch.ops.aten.mul.Tensor(bmm_3, 0.25);  bmm_3 = None
        permute_32 = torch.ops.aten.permute.default(bmm_1, [1, 0, 2]);  bmm_1 = None
        clone_10 = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
        view_26 = torch.ops.aten.view.default(clone_10, [128, 500, 128]);  clone_10 = None
        permute_33 = torch.ops.aten.permute.default(permute_31, [1, 0, 2]);  permute_31 = None
        view_27 = torch.ops.aten.view.default(permute_33, [128, 500, 128]);  permute_33 = None
        permute_34 = torch.ops.aten.permute.default(mul_59, [1, 0, 2]);  mul_59 = None
        clone_11 = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
        view_28 = torch.ops.aten.view.default(clone_11, [128, 500, 128]);  clone_11 = None
        full_default = torch.ops.aten.full.default([3, 128, 500, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter = torch.ops.aten.select_scatter.default(full_default, view_26, 0, 2);  view_26 = None
        select_scatter_1 = torch.ops.aten.select_scatter.default(full_default, view_27, 0, 1);  view_27 = None
        add_17 = torch.ops.aten.add.Tensor(select_scatter, select_scatter_1);  select_scatter = select_scatter_1 = None
        select_scatter_2 = torch.ops.aten.select_scatter.default(full_default, view_28, 0, 0);  full_default = view_28 = None
        add_18 = torch.ops.aten.add.Tensor(add_17, select_scatter_2);  add_17 = select_scatter_2 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(add_18, 3);  add_18 = None
        permute_35 = torch.ops.aten.permute.default(unsqueeze_1, [3, 1, 2, 0, 4]);  unsqueeze_1 = None
        squeeze_1 = torch.ops.aten.squeeze.dim(permute_35, 0);  permute_35 = None
        clone_12 = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
        view_29 = torch.ops.aten.view.default(clone_12, [128, 500, 384]);  clone_12 = None
        sum_18 = torch.ops.aten.sum.dim_IntList(view_29, [0, 1], True)
        view_30 = torch.ops.aten.view.default(sum_18, [384]);  sum_18 = None
        view_31 = torch.ops.aten.view.default(view_29, [64000, 384]);  view_29 = None
        permute_36 = torch.ops.aten.permute.default(view_31, [1, 0])
        mm_7 = torch.ops.aten.mm.default(permute_36, view_2);  permute_36 = view_2 = None
        permute_2 = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        permute_38 = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_8 = torch.ops.aten.mm.default(view_31, permute_38);  view_31 = permute_38 = None
        view_32 = torch.ops.aten.view.default(mm_8, [128, 500, 128]);  mm_8 = None
        permute_40 = torch.ops.aten.permute.default(view_32, [1, 0, 2]);  view_32 = None
        mul_61 = torch.ops.aten.mul.Tensor(permute_40, primals_5);  primals_5 = None
        mul_62 = torch.ops.aten.mul.Tensor(mul_61, 128)
        sum_19 = torch.ops.aten.sum.dim_IntList(mul_61, [2], True)
        mul_63 = torch.ops.aten.mul.Tensor(mul_61, mul_2);  mul_61 = None
        sum_20 = torch.ops.aten.sum.dim_IntList(mul_63, [2], True);  mul_63 = None
        mul_64 = torch.ops.aten.mul.Tensor(mul_2, sum_20);  sum_20 = None
        sub_15 = torch.ops.aten.sub.Tensor(mul_62, sum_19);  mul_62 = sum_19 = None
        sub_16 = torch.ops.aten.sub.Tensor(sub_15, mul_64);  sub_15 = mul_64 = None
        mul_65 = torch.ops.aten.mul.Tensor(div_4, sub_16);  div_4 = sub_16 = None
        mul_66 = torch.ops.aten.mul.Tensor(permute_40, mul_2);  mul_2 = None
        sum_21 = torch.ops.aten.sum.dim_IntList(mul_66, [0, 1]);  mul_66 = None
        sum_22 = torch.ops.aten.sum.dim_IntList(permute_40, [0, 1]);  permute_40 = None
        add_19 = torch.ops.aten.add.Tensor(add_16, mul_65);  add_16 = mul_65 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_67 = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
        mul_68 = torch.ops.aten.mul.Tensor(add_19, mul_67);  mul_67 = None
        view_33 = torch.ops.aten.view.default(mul_68, [64000, 128]);  mul_68 = None
        permute = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        permute_41 = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_9 = torch.ops.aten.mm.default(view_33, permute_41);  permute_41 = None
        permute_42 = torch.ops.aten.permute.default(view_33, [1, 0])
        mm_10 = torch.ops.aten.mm.default(permute_42, view);  permute_42 = view = None
        sum_23 = torch.ops.aten.sum.dim_IntList(view_33, [0], True);  view_33 = None
        view_34 = torch.ops.aten.view.default(sum_23, [128]);  sum_23 = None
        view_35 = torch.ops.aten.view.default(mm_9, [500, 128, 512]);  mm_9 = None
        return (view_35, mm_10, view_34, add_19, sum_21, sum_22, mm_7, view_30, mm_6, view_24, None, fma, sum_14, sum_15, sum_10, sum_11, mm_4, view_21, sum_5, sum_6, mm_2, view_18)
        
def load_args(reader):
    buf0 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf0, (128, 512), is_leaf=True)  # primals_2
    buf1 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1, (128,), is_leaf=True)  # primals_5
    buf2 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf2, (384, 128), is_leaf=True)  # primals_7
    buf3 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf3, (128, 128), is_leaf=True)  # primals_9
    buf4 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf4, (128,), is_leaf=True)  # primals_13
    buf5 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf5, (128,), is_leaf=True)  # primals_15
    buf6 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf6, (512, 128), is_leaf=True)  # primals_17
    buf7 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf7, (512,), is_leaf=True)  # primals_19
    buf8 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf8, (128, 512), is_leaf=True)  # primals_21
    buf9 = reader.storage(None, 131072000, device=device(type='cuda', index=0))
    reader.tensor(buf9, (64000, 512), is_leaf=True)  # view
    buf10 = reader.storage(None, 8192000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf10, (500, 128, 128), dtype=torch.bool, is_leaf=True)  # gt
    buf11 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf11, (500, 128, 128), is_leaf=True)  # mul_2
    buf12 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf12, (64000, 128), is_leaf=True)  # view_2
    buf13 = reader.storage(None, 262144000, device=device(type='cuda', index=0))
    reader.tensor(buf13, (4000, 128, 128), is_leaf=True)  # baddbmm
    buf14 = reader.storage(None, 2048000, device=device(type='cuda', index=0))
    reader.tensor(buf14, (4000, 128, 1), is_leaf=True)  # amax
    buf15 = reader.storage(None, 2048000, device=device(type='cuda', index=0))
    reader.tensor(buf15, (4000, 128, 1), is_leaf=True)  # sum_1
    buf16 = reader.storage(None, 65536000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf16, (4000, 128, 128), dtype=torch.bool, is_leaf=True)  # gt_1
    buf17 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf17, (64000, 128), is_leaf=True)  # view_10
    buf18 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf18, (64000, 128), is_leaf=True)  # addmm_1
    buf19 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf19, (500, 128, 1), is_leaf=True)  # getitem_3
    buf20 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf20, (500, 128, 1), is_leaf=True)  # rsqrt_1
    buf21 = reader.storage(None, 8192000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf21, (500, 128, 128), dtype=torch.bool, is_leaf=True)  # gt_2
    buf22 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf22, (500, 128, 128), is_leaf=True)  # mul_11
    buf23 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf23, (64000, 128), is_leaf=True)  # view_13
    buf24 = reader.storage(None, 131072000, device=device(type='cuda', index=0))
    reader.tensor(buf24, (64000, 512), is_leaf=True)  # addmm_2
    buf25 = reader.storage(None, 32768000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf25, (500, 128, 512), dtype=torch.bool, is_leaf=True)  # gt_3
    buf26 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf26, (500, 128, 1), is_leaf=True)  # getitem_7
    buf27 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf27, (500, 128, 1), is_leaf=True)  # rsqrt_3
    buf28 = reader.storage(None, 131072000, device=device(type='cuda', index=0))
    reader.tensor(buf28, (64000, 512), is_leaf=True)  # view_15
    buf29 = reader.storage(None, 8192000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf29, (500, 128, 128), dtype=torch.bool, is_leaf=True)  # gt_4
    buf30 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf30, (500, 128, 1), is_leaf=True)  # div_2
    buf31 = reader.storage(None, 262144000, device=device(type='cuda', index=0))
    reader.tensor(buf31, (4000, 128, 128), (16384, 1, 128), is_leaf=True)  # permute_27
    buf32 = reader.storage(None, 98304000, device=device(type='cuda', index=0))
    reader.tensor(buf32, (4000, 16, 128), (16, 1, 64000), storage_offset=16384000, is_leaf=True)  # permute_28
    reader.tensor(buf32, (4000, 128, 16), (16, 64000, 1), storage_offset=8192000, is_leaf=True)  # permute_29
    buf33 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf33, (4000, 16, 128), (16, 1, 64000), is_leaf=True)  # permute_30
    buf34 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf34, (500, 128, 1), is_leaf=True)  # div_4
    buf35 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf35, (500, 128, 128), is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)