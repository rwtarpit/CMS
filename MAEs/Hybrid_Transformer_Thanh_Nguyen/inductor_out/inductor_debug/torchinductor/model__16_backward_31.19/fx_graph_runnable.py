
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

    
    
    def forward(self, primals_1, primals_4, primals_8, primals_10, primals_12, primals_14, primals_16, cat_1, getitem_1, rsqrt, view_2, div, view_10, addmm, getitem_7, rsqrt_1, add_6, getitem_9, rsqrt_2, view_13, addmm_1, getitem_11, rsqrt_3, permute_24, permute_25, permute_26, permute_34, permute_38, tangents_1, tangents_2):
        mul_13 = torch.ops.aten.mul.Tensor(tangents_1, primals_16);  primals_16 = None
        mul_14 = torch.ops.aten.mul.Tensor(mul_13, 512)
        sum_2 = torch.ops.aten.sum.dim_IntList(mul_13, [2], True)
        view_14 = torch.ops.aten.view.default(addmm_1, [500, 1, 512]);  addmm_1 = None
        mul_7 = torch.ops.aten.mul.Tensor(view_14, 0.5)
        mul_8 = torch.ops.aten.mul.Tensor(view_14, 0.7071067811865476)
        erf = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add_9 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_7, add_9);  mul_7 = None
        sub_4 = torch.ops.aten.sub.Tensor(mul_9, getitem_11);  mul_9 = getitem_11 = None
        mul_10 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = None
        mul_15 = torch.ops.aten.mul.Tensor(mul_13, mul_10);  mul_13 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(mul_15, [2], True);  mul_15 = None
        mul_16 = torch.ops.aten.mul.Tensor(mul_10, sum_3);  sum_3 = None
        sub_6 = torch.ops.aten.sub.Tensor(mul_14, sum_2);  mul_14 = sum_2 = None
        sub_7 = torch.ops.aten.sub.Tensor(sub_6, mul_16);  sub_6 = mul_16 = None
        div_1 = torch.ops.aten.div.Tensor(rsqrt_3, 512);  rsqrt_3 = None
        mul_17 = torch.ops.aten.mul.Tensor(div_1, sub_7);  div_1 = sub_7 = None
        mul_18 = torch.ops.aten.mul.Tensor(tangents_1, mul_10);  mul_10 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(mul_18, [0, 1]);  mul_18 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(tangents_1, [0, 1]);  tangents_1 = None
        mul_20 = torch.ops.aten.mul.Tensor(add_9, 0.5);  add_9 = None
        mul_21 = torch.ops.aten.mul.Tensor(view_14, view_14)
        mul_22 = torch.ops.aten.mul.Tensor(mul_21, -0.5);  mul_21 = None
        exp_1 = torch.ops.aten.exp.default(mul_22);  mul_22 = None
        mul_23 = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
        mul_24 = torch.ops.aten.mul.Tensor(view_14, mul_23);  view_14 = mul_23 = None
        add_13 = torch.ops.aten.add.Tensor(mul_20, mul_24);  mul_20 = mul_24 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_17, add_13);  mul_17 = add_13 = None
        view_15 = torch.ops.aten.view.default(mul_25, [500, 512]);  mul_25 = None
        permute_12 = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
        permute_13 = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        mm_2 = torch.ops.aten.mm.default(view_15, permute_13);  permute_13 = None
        permute_14 = torch.ops.aten.permute.default(view_15, [1, 0])
        mm_3 = torch.ops.aten.mm.default(permute_14, view_13);  permute_14 = view_13 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(view_15, [0], True);  view_15 = None
        view_16 = torch.ops.aten.view.default(sum_6, [512]);  sum_6 = None
        view_17 = torch.ops.aten.view.default(mm_2, [500, 1, 128]);  mm_2 = None
        mul_27 = torch.ops.aten.mul.Tensor(view_17, primals_12);  primals_12 = None
        mul_28 = torch.ops.aten.mul.Tensor(mul_27, 128)
        sum_7 = torch.ops.aten.sum.dim_IntList(mul_27, [2], True)
        sub_3 = torch.ops.aten.sub.Tensor(add_6, getitem_9);  add_6 = getitem_9 = None
        mul_5 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_27, mul_5);  mul_27 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(mul_29, [2], True);  mul_29 = None
        mul_30 = torch.ops.aten.mul.Tensor(mul_5, sum_8);  sum_8 = None
        sub_9 = torch.ops.aten.sub.Tensor(mul_28, sum_7);  mul_28 = sum_7 = None
        sub_10 = torch.ops.aten.sub.Tensor(sub_9, mul_30);  sub_9 = mul_30 = None
        div_2 = torch.ops.aten.div.Tensor(rsqrt_2, 128);  rsqrt_2 = None
        mul_31 = torch.ops.aten.mul.Tensor(div_2, sub_10);  div_2 = sub_10 = None
        mul_32 = torch.ops.aten.mul.Tensor(view_17, mul_5);  mul_5 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(mul_32, [0, 1]);  mul_32 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(view_17, [0, 1]);  view_17 = None
        add_14 = torch.ops.aten.add.Tensor(tangents_2, mul_31);  tangents_2 = mul_31 = None
        mul_34 = torch.ops.aten.mul.Tensor(add_14, primals_10);  primals_10 = None
        mul_35 = torch.ops.aten.mul.Tensor(mul_34, 128)
        sum_11 = torch.ops.aten.sum.dim_IntList(mul_34, [2], True)
        view_11 = torch.ops.aten.view.default(addmm, [1, 500, 128]);  addmm = None
        permute_11 = torch.ops.aten.permute.default(view_11, [1, 0, 2]);  view_11 = None
        sub_2 = torch.ops.aten.sub.Tensor(permute_11, getitem_7);  permute_11 = getitem_7 = None
        mul_3 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_36 = torch.ops.aten.mul.Tensor(mul_34, mul_3);  mul_34 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(mul_36, [2], True);  mul_36 = None
        mul_37 = torch.ops.aten.mul.Tensor(mul_3, sum_12);  sum_12 = None
        sub_12 = torch.ops.aten.sub.Tensor(mul_35, sum_11);  mul_35 = sum_11 = None
        sub_13 = torch.ops.aten.sub.Tensor(sub_12, mul_37);  sub_12 = mul_37 = None
        div_3 = torch.ops.aten.div.Tensor(rsqrt_1, 128);  rsqrt_1 = None
        mul_38 = torch.ops.aten.mul.Tensor(div_3, sub_13);  div_3 = sub_13 = None
        mul_39 = torch.ops.aten.mul.Tensor(add_14, mul_3);  mul_3 = None
        sum_13 = torch.ops.aten.sum.dim_IntList(mul_39, [0, 1]);  mul_39 = None
        sum_14 = torch.ops.aten.sum.dim_IntList(add_14, [0, 1])
        permute_17 = torch.ops.aten.permute.default(mul_38, [1, 0, 2]);  mul_38 = None
        view_18 = torch.ops.aten.view.default(permute_17, [500, 128]);  permute_17 = None
        permute_10 = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_18 = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm_4 = torch.ops.aten.mm.default(view_18, permute_18);  permute_18 = None
        permute_19 = torch.ops.aten.permute.default(view_18, [1, 0])
        mm_5 = torch.ops.aten.mm.default(permute_19, view_10);  permute_19 = view_10 = None
        sum_15 = torch.ops.aten.sum.dim_IntList(view_18, [0], True);  view_18 = None
        view_19 = torch.ops.aten.view.default(sum_15, [128]);  sum_15 = None
        view_20 = torch.ops.aten.view.default(mm_4, [1, 4000, 16]);  mm_4 = None
        permute_22 = torch.ops.aten.permute.default(view_20, [1, 0, 2]);  view_20 = None
        permute_23 = torch.ops.aten.permute.default(div, [0, 2, 1])
        bmm_1 = torch.ops.aten.bmm.default(permute_23, permute_22);  permute_23 = None
        bmm_2 = torch.ops.aten.bmm.default(permute_22, permute_24);  permute_22 = permute_24 = None
        mul_40 = torch.ops.aten.mul.Tensor(bmm_2, div);  bmm_2 = None
        sum_16 = torch.ops.aten.sum.dim_IntList(mul_40, [-1], True)
        neg = torch.ops.aten.neg.default(div);  div = None
        fma = torch.ops.prims.fma.default(neg, sum_16, mul_40);  neg = sum_16 = mul_40 = None
        bmm_3 = torch.ops.aten.bmm.default(fma, permute_25);  permute_25 = None
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(fma, [0, 3, 0, 0, 0, 0]);  fma = None
        bmm_default = torch.ops.aten.bmm.default(permute_26, constant_pad_nd_default);  permute_26 = constant_pad_nd_default = None
        slice_tensor = torch.ops.aten.slice.Tensor(bmm_default, 2, 0, -3);  bmm_default = None
        permute_27 = torch.ops.aten.permute.default(slice_tensor, [0, 2, 1]);  slice_tensor = None
        mul_41 = torch.ops.aten.mul.Tensor(bmm_3, 0.25);  bmm_3 = None
        permute_28 = torch.ops.aten.permute.default(bmm_1, [1, 0, 2]);  bmm_1 = None
        clone_5 = torch.ops.aten.clone.default(permute_28, memory_format = torch.contiguous_format);  permute_28 = None
        view_21 = torch.ops.aten.view.default(clone_5, [129, 500, 128]);  clone_5 = None
        permute_29 = torch.ops.aten.permute.default(permute_27, [1, 0, 2]);  permute_27 = None
        view_22 = torch.ops.aten.view.default(permute_29, [129, 500, 128]);  permute_29 = None
        permute_30 = torch.ops.aten.permute.default(mul_41, [1, 0, 2]);  mul_41 = None
        view_23 = torch.ops.aten.view.default(permute_30, [1, 500, 128]);  permute_30 = None
        full_default_1 = torch.ops.aten.full.default([2, 129, 500, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter = torch.ops.aten.select_scatter.default(full_default_1, view_21, 0, 1);  view_21 = None
        select_scatter_1 = torch.ops.aten.select_scatter.default(full_default_1, view_22, 0, 0);  full_default_1 = view_22 = None
        add_15 = torch.ops.aten.add.Tensor(select_scatter, select_scatter_1);  select_scatter = select_scatter_1 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(add_15, 3);  add_15 = None
        permute_31 = torch.ops.aten.permute.default(unsqueeze_1, [3, 1, 2, 0, 4]);  unsqueeze_1 = None
        squeeze_1 = torch.ops.aten.squeeze.dim(permute_31, 0);  permute_31 = None
        clone_6 = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
        view_24 = torch.ops.aten.view.default(clone_6, [129, 500, 256]);  clone_6 = None
        sum_17 = torch.ops.aten.sum.dim_IntList(view_24, [0, 1], True)
        view_25 = torch.ops.aten.view.default(sum_17, [256]);  sum_17 = None
        view_26 = torch.ops.aten.view.default(view_24, [64500, 256]);  view_24 = None
        permute_32 = torch.ops.aten.permute.default(view_26, [1, 0])
        mm_6 = torch.ops.aten.mm.default(permute_32, view_2);  permute_32 = view_2 = None
        mm_7 = torch.ops.aten.mm.default(view_26, permute_34);  view_26 = permute_34 = None
        view_27 = torch.ops.aten.view.default(mm_7, [129, 500, 128]);  mm_7 = None
        sum_18 = torch.ops.aten.sum.dim_IntList(view_23, [0, 1], True)
        view_28 = torch.ops.aten.view.default(sum_18, [128]);  sum_18 = None
        view_29 = torch.ops.aten.view.default(view_23, [500, 128]);  view_23 = None
        permute_36 = torch.ops.aten.permute.default(view_29, [1, 0])
        expand = torch.ops.aten.expand.default(primals_1, [500, -1, -1]);  primals_1 = None
        permute = torch.ops.aten.permute.default(expand, [1, 0, 2]);  expand = None
        view = torch.ops.aten.view.default(permute, [500, 128]);  permute = None
        mm_8 = torch.ops.aten.mm.default(permute_36, view);  permute_36 = view = None
        mm_9 = torch.ops.aten.mm.default(view_29, permute_38);  view_29 = permute_38 = None
        view_30 = torch.ops.aten.view.default(mm_9, [1, 500, 128]);  mm_9 = None
        cat_2 = torch.ops.aten.cat.default([view_28, view_25]);  view_28 = view_25 = None
        cat_3 = torch.ops.aten.cat.default([mm_8, mm_6]);  mm_8 = mm_6 = None
        permute_40 = torch.ops.aten.permute.default(view_27, [1, 0, 2]);  view_27 = None
        permute_41 = torch.ops.aten.permute.default(view_30, [1, 0, 2]);  view_30 = None
        add_16 = torch.ops.aten.add.Tensor(add_14, permute_41);  add_14 = permute_41 = None
        mul_43 = torch.ops.aten.mul.Tensor(permute_40, primals_4);  primals_4 = None
        mul_44 = torch.ops.aten.mul.Tensor(mul_43, 128)
        sum_19 = torch.ops.aten.sum.dim_IntList(mul_43, [2], True)
        sub = torch.ops.aten.sub.Tensor(cat_1, getitem_1);  cat_1 = getitem_1 = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_45 = torch.ops.aten.mul.Tensor(mul_43, mul);  mul_43 = None
        sum_20 = torch.ops.aten.sum.dim_IntList(mul_45, [2], True);  mul_45 = None
        mul_46 = torch.ops.aten.mul.Tensor(mul, sum_20);  sum_20 = None
        sub_15 = torch.ops.aten.sub.Tensor(mul_44, sum_19);  mul_44 = sum_19 = None
        sub_16 = torch.ops.aten.sub.Tensor(sub_15, mul_46);  sub_15 = mul_46 = None
        div_4 = torch.ops.aten.div.Tensor(rsqrt, 128);  rsqrt = None
        mul_47 = torch.ops.aten.mul.Tensor(div_4, sub_16);  div_4 = sub_16 = None
        mul_48 = torch.ops.aten.mul.Tensor(permute_40, mul);  mul = None
        sum_21 = torch.ops.aten.sum.dim_IntList(mul_48, [0, 1]);  mul_48 = None
        sum_22 = torch.ops.aten.sum.dim_IntList(permute_40, [0, 1]);  permute_40 = None
        slice_2 = torch.ops.aten.slice.Tensor(mul_47, 1, 0, 1)
        slice_3 = torch.ops.aten.slice.Tensor(mul_47, 1, 1, 129);  mul_47 = None
        add_17 = torch.ops.aten.add.Tensor(add_16, slice_2);  add_16 = slice_2 = None
        sum_23 = torch.ops.aten.sum.dim_IntList(add_17, [0], True);  add_17 = None
        return (sum_23, None, slice_3, sum_21, sum_22, cat_3, cat_2, mm_5, view_19, sum_13, sum_14, sum_9, sum_10, mm_3, view_16, sum_4, sum_5)
        
def load_args(reader):
    buf0 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf0, (1, 1, 128), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1, (128,), is_leaf=True)  # primals_4
    buf2 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf2, (128, 128), is_leaf=True)  # primals_8
    buf3 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf3, (128,), is_leaf=True)  # primals_10
    buf4 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf4, (128,), is_leaf=True)  # primals_12
    buf5 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf5, (512, 128), is_leaf=True)  # primals_14
    buf6 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf6, (512,), is_leaf=True)  # primals_16
    buf7 = reader.storage(None, 33024000, device=device(type='cuda', index=0))
    reader.tensor(buf7, (500, 129, 128), is_leaf=True)  # cat_1
    buf8 = reader.storage(None, 258000, device=device(type='cuda', index=0))
    reader.tensor(buf8, (500, 129, 1), is_leaf=True)  # getitem_1
    buf9 = reader.storage(None, 258000, device=device(type='cuda', index=0))
    reader.tensor(buf9, (500, 129, 1), is_leaf=True)  # rsqrt
    buf10 = reader.storage(None, 33024000, device=device(type='cuda', index=0))
    reader.tensor(buf10, (64500, 128), is_leaf=True)  # view_2
    buf11 = reader.storage(None, 2064000, device=device(type='cuda', index=0))
    reader.tensor(buf11, (4000, 1, 129), is_leaf=True)  # div
    buf12 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf12, (500, 128), is_leaf=True)  # view_10
    buf13 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf13, (500, 128), is_leaf=True)  # addmm
    buf14 = reader.storage(None, 2000, device=device(type='cuda', index=0))
    reader.tensor(buf14, (500, 1, 1), is_leaf=True)  # getitem_7
    buf15 = reader.storage(None, 2000, device=device(type='cuda', index=0))
    reader.tensor(buf15, (500, 1, 1), is_leaf=True)  # rsqrt_1
    buf16 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf16, (500, 1, 128), (128, 64000, 1), requires_grad=True)  # add_6
    buf17 = reader.storage(None, 2000, device=device(type='cuda', index=0))
    reader.tensor(buf17, (500, 1, 1), is_leaf=True)  # getitem_9
    buf18 = reader.storage(None, 2000, device=device(type='cuda', index=0))
    reader.tensor(buf18, (500, 1, 1), is_leaf=True)  # rsqrt_2
    buf19 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf19, (500, 128), is_leaf=True)  # view_13
    buf20 = reader.storage(None, 1024000, device=device(type='cuda', index=0))
    reader.tensor(buf20, (500, 512), is_leaf=True)  # addmm_1
    buf21 = reader.storage(None, 2000, device=device(type='cuda', index=0))
    reader.tensor(buf21, (500, 1, 1), is_leaf=True)  # getitem_11
    buf22 = reader.storage(None, 2000, device=device(type='cuda', index=0))
    reader.tensor(buf22, (500, 1, 1), is_leaf=True)  # rsqrt_3
    buf23 = reader.storage(None, 66048000, device=device(type='cuda', index=0))
    reader.tensor(buf23, (4000, 16, 129), (16, 1, 64000), storage_offset=8256000, is_leaf=True)  # permute_24
    reader.tensor(buf23, (4000, 129, 16), (16, 64000, 1), is_leaf=True)  # permute_25
    buf24 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf24, (4000, 16, 1), (16, 1, 16), is_leaf=True)  # permute_26
    buf25 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf25, (256, 128), storage_offset=16384, is_leaf=True)  # permute_34
    reader.tensor(buf25, (128, 128), is_leaf=True)  # permute_38
    buf26 = reader.storage(None, 1024000, device=device(type='cuda', index=0))
    reader.tensor(buf26, (500, 1, 512), is_leaf=True)  # tangents_1
    buf27 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf27, (500, 1, 128), (128, 64000, 1), is_leaf=True)  # tangents_2
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)