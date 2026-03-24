
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

    
    
    def forward(self, primals_2, primals_7, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, view, cat_1, getitem_1, rsqrt, view_3, view_5, div, view_13, addmm_2, getitem_7, rsqrt_1, mul_5, view_16, addmm_3, getitem_11, rsqrt_3, view_18, mul_12, squeeze_1, div_1, div_3, permute_37, permute_38, permute_39, permute_47, permute_49, tangents_1):
        permute_17 = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
        permute_18 = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
        mm_1 = torch.ops.aten.mm.default(tangents_1, permute_18);  permute_18 = None
        permute_19 = torch.ops.aten.permute.default(tangents_1, [1, 0])
        constant_pad_nd_default_1 = torch.ops.aten.constant_pad_nd.default(permute_19, [0, 0, 0, 2]);  permute_19 = None
        mm_default = torch.ops.aten.mm.default(constant_pad_nd_default_1, squeeze_1);  constant_pad_nd_default_1 = squeeze_1 = None
        slice_tensor_1 = torch.ops.aten.slice.Tensor(mm_default, 0, 0, -2);  mm_default = None
        sum_2 = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view_20 = torch.ops.aten.view.default(sum_2, [10]);  sum_2 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(mm_1, 1);  mm_1 = None
        mul_15 = torch.ops.aten.mul.Tensor(unsqueeze_1, primals_23);  primals_23 = None
        mul_16 = torch.ops.aten.mul.Tensor(mul_15, 128)
        sum_3 = torch.ops.aten.sum.dim_IntList(mul_15, [2], True)
        mul_17 = torch.ops.aten.mul.Tensor(mul_15, mul_12);  mul_15 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(mul_17, [2], True);  mul_17 = None
        mul_18 = torch.ops.aten.mul.Tensor(mul_12, sum_4);  sum_4 = None
        sub_7 = torch.ops.aten.sub.Tensor(mul_16, sum_3);  mul_16 = sum_3 = None
        sub_8 = torch.ops.aten.sub.Tensor(sub_7, mul_18);  sub_7 = mul_18 = None
        mul_19 = torch.ops.aten.mul.Tensor(div_1, sub_8);  div_1 = sub_8 = None
        mul_20 = torch.ops.aten.mul.Tensor(unsqueeze_1, mul_12);  mul_12 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(mul_20, [0, 1]);  mul_20 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(unsqueeze_1, [0, 1]);  unsqueeze_1 = None
        view_21 = torch.ops.aten.view.default(mul_19, [500, 128])
        permute_16 = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
        permute_22 = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
        mm_3 = torch.ops.aten.mm.default(view_21, permute_22);  permute_22 = None
        permute_23 = torch.ops.aten.permute.default(view_21, [1, 0])
        mm_4 = torch.ops.aten.mm.default(permute_23, view_18);  permute_23 = view_18 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(view_21, [0], True);  view_21 = None
        view_22 = torch.ops.aten.view.default(sum_7, [128]);  sum_7 = None
        view_23 = torch.ops.aten.view.default(mm_3, [500, 1, 512]);  mm_3 = None
        mul_22 = torch.ops.aten.mul.Tensor(view_23, primals_19);  primals_19 = None
        mul_23 = torch.ops.aten.mul.Tensor(mul_22, 512)
        sum_8 = torch.ops.aten.sum.dim_IntList(mul_22, [2], True)
        view_17 = torch.ops.aten.view.default(addmm_3, [500, 1, 512]);  addmm_3 = None
        mul_7 = torch.ops.aten.mul.Tensor(view_17, 0.5)
        mul_8 = torch.ops.aten.mul.Tensor(view_17, 0.7071067811865476)
        erf = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add_9 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_7, add_9);  mul_7 = None
        sub_4 = torch.ops.aten.sub.Tensor(mul_9, getitem_11);  mul_9 = getitem_11 = None
        mul_10 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = None
        mul_24 = torch.ops.aten.mul.Tensor(mul_22, mul_10);  mul_22 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(mul_24, [2], True);  mul_24 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_10, sum_9);  sum_9 = None
        sub_10 = torch.ops.aten.sub.Tensor(mul_23, sum_8);  mul_23 = sum_8 = None
        sub_11 = torch.ops.aten.sub.Tensor(sub_10, mul_25);  sub_10 = mul_25 = None
        div_2 = torch.ops.aten.div.Tensor(rsqrt_3, 512);  rsqrt_3 = None
        mul_26 = torch.ops.aten.mul.Tensor(div_2, sub_11);  div_2 = sub_11 = None
        mul_27 = torch.ops.aten.mul.Tensor(view_23, mul_10);  mul_10 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(mul_27, [0, 1]);  mul_27 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(view_23, [0, 1]);  view_23 = None
        mul_29 = torch.ops.aten.mul.Tensor(add_9, 0.5);  add_9 = None
        mul_30 = torch.ops.aten.mul.Tensor(view_17, view_17)
        mul_31 = torch.ops.aten.mul.Tensor(mul_30, -0.5);  mul_30 = None
        exp_1 = torch.ops.aten.exp.default(mul_31);  mul_31 = None
        mul_32 = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
        mul_33 = torch.ops.aten.mul.Tensor(view_17, mul_32);  view_17 = mul_32 = None
        add_16 = torch.ops.aten.add.Tensor(mul_29, mul_33);  mul_29 = mul_33 = None
        mul_34 = torch.ops.aten.mul.Tensor(mul_26, add_16);  mul_26 = add_16 = None
        view_24 = torch.ops.aten.view.default(mul_34, [500, 512]);  mul_34 = None
        permute_15 = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
        permute_26 = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
        mm_5 = torch.ops.aten.mm.default(view_24, permute_26);  permute_26 = None
        permute_27 = torch.ops.aten.permute.default(view_24, [1, 0])
        mm_6 = torch.ops.aten.mm.default(permute_27, view_16);  permute_27 = view_16 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(view_24, [0], True);  view_24 = None
        view_25 = torch.ops.aten.view.default(sum_12, [512]);  sum_12 = None
        view_26 = torch.ops.aten.view.default(mm_5, [500, 1, 128]);  mm_5 = None
        mul_36 = torch.ops.aten.mul.Tensor(view_26, primals_15);  primals_15 = None
        mul_37 = torch.ops.aten.mul.Tensor(mul_36, 128)
        sum_13 = torch.ops.aten.sum.dim_IntList(mul_36, [2], True)
        mul_38 = torch.ops.aten.mul.Tensor(mul_36, mul_5);  mul_36 = None
        sum_14 = torch.ops.aten.sum.dim_IntList(mul_38, [2], True);  mul_38 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_5, sum_14);  sum_14 = None
        sub_13 = torch.ops.aten.sub.Tensor(mul_37, sum_13);  mul_37 = sum_13 = None
        sub_14 = torch.ops.aten.sub.Tensor(sub_13, mul_39);  sub_13 = mul_39 = None
        mul_40 = torch.ops.aten.mul.Tensor(div_3, sub_14);  div_3 = sub_14 = None
        mul_41 = torch.ops.aten.mul.Tensor(view_26, mul_5);  mul_5 = None
        sum_15 = torch.ops.aten.sum.dim_IntList(mul_41, [0, 1]);  mul_41 = None
        sum_16 = torch.ops.aten.sum.dim_IntList(view_26, [0, 1]);  view_26 = None
        add_17 = torch.ops.aten.add.Tensor(mul_19, mul_40);  mul_19 = mul_40 = None
        mul_43 = torch.ops.aten.mul.Tensor(add_17, primals_13);  primals_13 = None
        mul_44 = torch.ops.aten.mul.Tensor(mul_43, 128)
        sum_17 = torch.ops.aten.sum.dim_IntList(mul_43, [2], True)
        view_14 = torch.ops.aten.view.default(addmm_2, [1, 500, 128]);  addmm_2 = None
        permute_14 = torch.ops.aten.permute.default(view_14, [1, 0, 2]);  view_14 = None
        sub_2 = torch.ops.aten.sub.Tensor(permute_14, getitem_7);  permute_14 = getitem_7 = None
        mul_3 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_45 = torch.ops.aten.mul.Tensor(mul_43, mul_3);  mul_43 = None
        sum_18 = torch.ops.aten.sum.dim_IntList(mul_45, [2], True);  mul_45 = None
        mul_46 = torch.ops.aten.mul.Tensor(mul_3, sum_18);  sum_18 = None
        sub_16 = torch.ops.aten.sub.Tensor(mul_44, sum_17);  mul_44 = sum_17 = None
        sub_17 = torch.ops.aten.sub.Tensor(sub_16, mul_46);  sub_16 = mul_46 = None
        div_4 = torch.ops.aten.div.Tensor(rsqrt_1, 128);  rsqrt_1 = None
        mul_47 = torch.ops.aten.mul.Tensor(div_4, sub_17);  div_4 = sub_17 = None
        mul_48 = torch.ops.aten.mul.Tensor(add_17, mul_3);  mul_3 = None
        sum_19 = torch.ops.aten.sum.dim_IntList(mul_48, [0, 1]);  mul_48 = None
        sum_20 = torch.ops.aten.sum.dim_IntList(add_17, [0, 1])
        permute_30 = torch.ops.aten.permute.default(mul_47, [1, 0, 2]);  mul_47 = None
        view_27 = torch.ops.aten.view.default(permute_30, [500, 128]);  permute_30 = None
        permute_13 = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_31 = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        mm_7 = torch.ops.aten.mm.default(view_27, permute_31);  permute_31 = None
        permute_32 = torch.ops.aten.permute.default(view_27, [1, 0])
        mm_8 = torch.ops.aten.mm.default(permute_32, view_13);  permute_32 = view_13 = None
        sum_21 = torch.ops.aten.sum.dim_IntList(view_27, [0], True);  view_27 = None
        view_28 = torch.ops.aten.view.default(sum_21, [128]);  sum_21 = None
        view_29 = torch.ops.aten.view.default(mm_7, [1, 4000, 16]);  mm_7 = None
        permute_35 = torch.ops.aten.permute.default(view_29, [1, 0, 2]);  view_29 = None
        permute_36 = torch.ops.aten.permute.default(div, [0, 2, 1])
        bmm_1 = torch.ops.aten.bmm.default(permute_36, permute_35);  permute_36 = None
        bmm_2 = torch.ops.aten.bmm.default(permute_35, permute_37);  permute_35 = permute_37 = None
        mul_49 = torch.ops.aten.mul.Tensor(bmm_2, div);  bmm_2 = None
        sum_22 = torch.ops.aten.sum.dim_IntList(mul_49, [-1], True)
        neg = torch.ops.aten.neg.default(div);  div = None
        fma = torch.ops.prims.fma.default(neg, sum_22, mul_49);  neg = sum_22 = mul_49 = None
        bmm_3 = torch.ops.aten.bmm.default(fma, permute_38);  permute_38 = None
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(fma, [0, 3, 0, 0, 0, 0]);  fma = None
        bmm_default = torch.ops.aten.bmm.default(permute_39, constant_pad_nd_default);  permute_39 = constant_pad_nd_default = None
        slice_tensor = torch.ops.aten.slice.Tensor(bmm_default, 2, 0, -3);  bmm_default = None
        permute_40 = torch.ops.aten.permute.default(slice_tensor, [0, 2, 1]);  slice_tensor = None
        mul_50 = torch.ops.aten.mul.Tensor(bmm_3, 0.25);  bmm_3 = None
        permute_41 = torch.ops.aten.permute.default(bmm_1, [1, 0, 2]);  bmm_1 = None
        clone_7 = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None
        view_30 = torch.ops.aten.view.default(clone_7, [129, 500, 128]);  clone_7 = None
        permute_42 = torch.ops.aten.permute.default(permute_40, [1, 0, 2]);  permute_40 = None
        view_31 = torch.ops.aten.view.default(permute_42, [129, 500, 128]);  permute_42 = None
        permute_43 = torch.ops.aten.permute.default(mul_50, [1, 0, 2]);  mul_50 = None
        view_32 = torch.ops.aten.view.default(permute_43, [1, 500, 128]);  permute_43 = None
        full_default_1 = torch.ops.aten.full.default([2, 129, 500, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter = torch.ops.aten.select_scatter.default(full_default_1, view_30, 0, 1);  view_30 = None
        select_scatter_1 = torch.ops.aten.select_scatter.default(full_default_1, view_31, 0, 0);  full_default_1 = view_31 = None
        add_18 = torch.ops.aten.add.Tensor(select_scatter, select_scatter_1);  select_scatter = select_scatter_1 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(add_18, 3);  add_18 = None
        permute_44 = torch.ops.aten.permute.default(unsqueeze_2, [3, 1, 2, 0, 4]);  unsqueeze_2 = None
        squeeze_2 = torch.ops.aten.squeeze.dim(permute_44, 0);  permute_44 = None
        clone_8 = torch.ops.aten.clone.default(squeeze_2, memory_format = torch.contiguous_format);  squeeze_2 = None
        view_33 = torch.ops.aten.view.default(clone_8, [129, 500, 256]);  clone_8 = None
        sum_23 = torch.ops.aten.sum.dim_IntList(view_33, [0, 1], True)
        view_34 = torch.ops.aten.view.default(sum_23, [256]);  sum_23 = None
        view_35 = torch.ops.aten.view.default(view_33, [64500, 256]);  view_33 = None
        permute_45 = torch.ops.aten.permute.default(view_35, [1, 0])
        mm_9 = torch.ops.aten.mm.default(permute_45, view_5);  permute_45 = view_5 = None
        mm_10 = torch.ops.aten.mm.default(view_35, permute_47);  view_35 = permute_47 = None
        view_36 = torch.ops.aten.view.default(mm_10, [129, 500, 128]);  mm_10 = None
        view_37 = torch.ops.aten.view.default(view_32, [500, 128]);  view_32 = None
        mm_11 = torch.ops.aten.mm.default(view_37, permute_49);  permute_49 = None
        permute_50 = torch.ops.aten.permute.default(view_37, [1, 0])
        mm_12 = torch.ops.aten.mm.default(permute_50, view_3);  permute_50 = view_3 = None
        sum_24 = torch.ops.aten.sum.dim_IntList(view_37, [0], True);  view_37 = None
        view_38 = torch.ops.aten.view.default(sum_24, [128]);  sum_24 = None
        view_39 = torch.ops.aten.view.default(mm_11, [1, 500, 128]);  mm_11 = None
        cat_2 = torch.ops.aten.cat.default([view_38, view_34]);  view_38 = view_34 = None
        cat_3 = torch.ops.aten.cat.default([mm_12, mm_9]);  mm_12 = mm_9 = None
        permute_53 = torch.ops.aten.permute.default(view_36, [1, 0, 2]);  view_36 = None
        permute_54 = torch.ops.aten.permute.default(view_39, [1, 0, 2]);  view_39 = None
        add_19 = torch.ops.aten.add.Tensor(add_17, permute_54);  add_17 = permute_54 = None
        mul_52 = torch.ops.aten.mul.Tensor(permute_53, primals_7);  primals_7 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_52, 128)
        sum_25 = torch.ops.aten.sum.dim_IntList(mul_52, [2], True)
        sub = torch.ops.aten.sub.Tensor(cat_1, getitem_1);  cat_1 = getitem_1 = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_54 = torch.ops.aten.mul.Tensor(mul_52, mul);  mul_52 = None
        sum_26 = torch.ops.aten.sum.dim_IntList(mul_54, [2], True);  mul_54 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul, sum_26);  sum_26 = None
        sub_19 = torch.ops.aten.sub.Tensor(mul_53, sum_25);  mul_53 = sum_25 = None
        sub_20 = torch.ops.aten.sub.Tensor(sub_19, mul_55);  sub_19 = mul_55 = None
        div_5 = torch.ops.aten.div.Tensor(rsqrt, 128);  rsqrt = None
        mul_56 = torch.ops.aten.mul.Tensor(div_5, sub_20);  div_5 = sub_20 = None
        mul_57 = torch.ops.aten.mul.Tensor(permute_53, mul);  mul = None
        sum_27 = torch.ops.aten.sum.dim_IntList(mul_57, [0, 1]);  mul_57 = None
        sum_28 = torch.ops.aten.sum.dim_IntList(permute_53, [0, 1]);  permute_53 = None
        slice_2 = torch.ops.aten.slice.Tensor(mul_56, 1, 0, 1)
        slice_3 = torch.ops.aten.slice.Tensor(mul_56, 1, 1, 129);  mul_56 = None
        add_20 = torch.ops.aten.add.Tensor(add_19, slice_2);  add_19 = slice_2 = None
        view_40 = torch.ops.aten.view.default(add_20, [500, 128])
        permute = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        permute_55 = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_13 = torch.ops.aten.mm.default(view_40, permute_55);  permute_55 = None
        permute_56 = torch.ops.aten.permute.default(view_40, [1, 0])
        mm_14 = torch.ops.aten.mm.default(permute_56, view);  permute_56 = view = None
        sum_29 = torch.ops.aten.sum.dim_IntList(view_40, [0], True);  view_40 = None
        view_41 = torch.ops.aten.view.default(sum_29, [128]);  sum_29 = None
        view_42 = torch.ops.aten.view.default(mm_13, [500, 1, 512]);  mm_13 = None
        return (view_42, mm_14, view_41, add_20, None, slice_3, sum_27, sum_28, cat_3, cat_2, mm_8, view_28, sum_19, sum_20, sum_15, sum_16, mm_6, view_25, sum_10, sum_11, mm_4, view_22, sum_5, sum_6, slice_tensor_1, view_20)
        
def load_args(reader):
    buf0 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf0, (128, 512), is_leaf=True)  # primals_2
    buf1 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1, (128,), is_leaf=True)  # primals_7
    buf2 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf2, (128, 128), is_leaf=True)  # primals_11
    buf3 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf3, (128,), is_leaf=True)  # primals_13
    buf4 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf4, (128,), is_leaf=True)  # primals_15
    buf5 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf5, (512, 128), is_leaf=True)  # primals_17
    buf6 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf6, (512,), is_leaf=True)  # primals_19
    buf7 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf7, (128, 512), is_leaf=True)  # primals_21
    buf8 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf8, (128,), is_leaf=True)  # primals_23
    buf9 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf9, (10, 128), is_leaf=True)  # primals_25
    buf10 = reader.storage(None, 1024000, device=device(type='cuda', index=0))
    reader.tensor(buf10, (500, 512), is_leaf=True)  # view
    buf11 = reader.storage(None, 33024000, device=device(type='cuda', index=0))
    reader.tensor(buf11, (500, 129, 128), is_leaf=True)  # cat_1
    buf12 = reader.storage(None, 258000, device=device(type='cuda', index=0))
    reader.tensor(buf12, (500, 129, 1), is_leaf=True)  # getitem_1
    buf13 = reader.storage(None, 258000, device=device(type='cuda', index=0))
    reader.tensor(buf13, (500, 129, 1), is_leaf=True)  # rsqrt
    buf14 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf14, (500, 128), is_leaf=True)  # view_3
    buf15 = reader.storage(None, 33024000, device=device(type='cuda', index=0))
    reader.tensor(buf15, (64500, 128), is_leaf=True)  # view_5
    buf16 = reader.storage(None, 2064000, device=device(type='cuda', index=0))
    reader.tensor(buf16, (4000, 1, 129), is_leaf=True)  # div
    buf17 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf17, (500, 128), is_leaf=True)  # view_13
    buf18 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf18, (500, 128), is_leaf=True)  # addmm_2
    buf19 = reader.storage(None, 2000, device=device(type='cuda', index=0))
    reader.tensor(buf19, (500, 1, 1), is_leaf=True)  # getitem_7
    buf20 = reader.storage(None, 2000, device=device(type='cuda', index=0))
    reader.tensor(buf20, (500, 1, 1), is_leaf=True)  # rsqrt_1
    buf21 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf21, (500, 1, 128), is_leaf=True)  # mul_5
    buf22 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf22, (500, 128), is_leaf=True)  # view_16
    buf23 = reader.storage(None, 1024000, device=device(type='cuda', index=0))
    reader.tensor(buf23, (500, 512), is_leaf=True)  # addmm_3
    buf24 = reader.storage(None, 2000, device=device(type='cuda', index=0))
    reader.tensor(buf24, (500, 1, 1), is_leaf=True)  # getitem_11
    buf25 = reader.storage(None, 2000, device=device(type='cuda', index=0))
    reader.tensor(buf25, (500, 1, 1), is_leaf=True)  # rsqrt_3
    buf26 = reader.storage(None, 1024000, device=device(type='cuda', index=0))
    reader.tensor(buf26, (500, 512), is_leaf=True)  # view_18
    buf27 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf27, (500, 1, 128), is_leaf=True)  # mul_12
    buf28 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf28, (500, 128), is_leaf=True)  # squeeze_1
    buf29 = reader.storage(None, 2000, device=device(type='cuda', index=0))
    reader.tensor(buf29, (500, 1, 1), is_leaf=True)  # div_1
    buf30 = reader.storage(None, 2000, device=device(type='cuda', index=0))
    reader.tensor(buf30, (500, 1, 1), is_leaf=True)  # div_3
    buf31 = reader.storage(None, 66048000, device=device(type='cuda', index=0))
    reader.tensor(buf31, (4000, 16, 129), (16, 1, 64000), storage_offset=8256000, is_leaf=True)  # permute_37
    reader.tensor(buf31, (4000, 129, 16), (16, 64000, 1), is_leaf=True)  # permute_38
    buf32 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf32, (4000, 16, 1), (16, 1, 16), is_leaf=True)  # permute_39
    buf33 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf33, (256, 128), storage_offset=16384, is_leaf=True)  # permute_47
    reader.tensor(buf33, (128, 128), is_leaf=True)  # permute_49
    buf34 = reader.storage(None, 20000, device=device(type='cuda', index=0))
    reader.tensor(buf34, (500, 10), is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)