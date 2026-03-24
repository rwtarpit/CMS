
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
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17):
        expand = torch.ops.aten.expand.default(primals_1, [500, -1, -1])
        full_default = torch.ops.aten.full.default([500, 1], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat = torch.ops.aten.cat.default([full_default, primals_2], 1);  full_default = primals_2 = None
        cat_1 = torch.ops.aten.cat.default([expand, primals_3], 1);  primals_3 = None
        var_mean = torch.ops.aten.var_mean.correction(cat_1, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
        sub = torch.ops.aten.sub.Tensor(cat_1, getitem_1)
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, primals_4);  mul = None
        add_1 = torch.ops.aten.add.Tensor(mul_1, primals_5);  mul_1 = primals_5 = None
        permute = torch.ops.aten.permute.default(expand, [1, 0, 2])
        permute_1 = torch.ops.aten.permute.default(add_1, [1, 0, 2]);  add_1 = None
        split_with_sizes = torch.ops.aten.split_with_sizes.default(primals_6, [128, 256]);  primals_6 = None
        getitem_2 = split_with_sizes[0]
        getitem_3 = split_with_sizes[1];  split_with_sizes = None
        split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(primals_7, [128, 256]);  primals_7 = None
        getitem_4 = split_with_sizes_1[0]
        getitem_5 = split_with_sizes_1[1];  split_with_sizes_1 = None
        permute_2 = torch.ops.aten.permute.default(getitem_2, [1, 0]);  getitem_2 = None
        view = torch.ops.aten.view.default(permute, [500, 128]);  permute = None
        mm = torch.ops.aten.mm.default(view, permute_2);  view = None
        view_1 = torch.ops.aten.view.default(mm, [1, 500, 128]);  mm = None
        add_2 = torch.ops.aten.add.Tensor(view_1, getitem_4);  view_1 = getitem_4 = None
        permute_3 = torch.ops.aten.permute.default(getitem_3, [1, 0]);  getitem_3 = None
        clone = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_2 = torch.ops.aten.view.default(clone, [64500, 128]);  clone = None
        mm_1 = torch.ops.aten.mm.default(view_2, permute_3)
        view_3 = torch.ops.aten.view.default(mm_1, [129, 500, 256]);  mm_1 = None
        add_3 = torch.ops.aten.add.Tensor(view_3, getitem_5);  view_3 = getitem_5 = None
        view_4 = torch.ops.aten.view.default(add_3, [129, 500, 2, 128]);  add_3 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(view_4, 0);  view_4 = None
        permute_4 = torch.ops.aten.permute.default(unsqueeze, [3, 1, 2, 0, 4]);  unsqueeze = None
        squeeze = torch.ops.aten.squeeze.dim(permute_4, -2);  permute_4 = None
        clone_1 = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
        select = torch.ops.aten.select.int(clone_1, 0, 0)
        select_1 = torch.ops.aten.select.int(clone_1, 0, 1);  clone_1 = None
        view_5 = torch.ops.aten.view.default(add_2, [1, 4000, 16]);  add_2 = None
        permute_5 = torch.ops.aten.permute.default(view_5, [1, 0, 2]);  view_5 = None
        view_6 = torch.ops.aten.view.default(select, [129, 4000, 16]);  select = None
        permute_6 = torch.ops.aten.permute.default(view_6, [1, 0, 2]);  view_6 = None
        view_7 = torch.ops.aten.view.default(select_1, [129, 4000, 16]);  select_1 = None
        permute_7 = torch.ops.aten.permute.default(view_7, [1, 0, 2]);  view_7 = None
        view_8 = torch.ops.aten.view.default(cat, [500, 1, 1, 129]);  cat = None
        expand_1 = torch.ops.aten.expand.default(view_8, [-1, 8, -1, -1]);  view_8 = None
        clone_2 = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_9 = torch.ops.aten.view.default(clone_2, [4000, 1, 129]);  clone_2 = None
        mul_2 = torch.ops.aten.mul.Tensor(permute_5, 0.25);  permute_5 = None
        permute_8 = torch.ops.aten.permute.default(permute_6, [0, 2, 1]);  permute_6 = None
        baddbmm = torch.ops.aten.baddbmm.default(view_9, mul_2, permute_8);  view_9 = None
        amax = torch.ops.aten.amax.default(baddbmm, [-1], True)
        sub_1 = torch.ops.aten.sub.Tensor(baddbmm, amax);  baddbmm = amax = None
        exp = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        bmm = torch.ops.aten.bmm.default(div, permute_7)
        permute_9 = torch.ops.aten.permute.default(bmm, [1, 0, 2]);  bmm = None
        view_10 = torch.ops.aten.view.default(permute_9, [500, 128]);  permute_9 = None
        permute_10 = torch.ops.aten.permute.default(primals_8, [1, 0])
        addmm = torch.ops.aten.addmm.default(primals_9, view_10, permute_10);  primals_9 = permute_10 = None
        view_11 = torch.ops.aten.view.default(addmm, [1, 500, 128])
        permute_11 = torch.ops.aten.permute.default(view_11, [1, 0, 2]);  view_11 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(permute_11, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_1[0]
        getitem_7 = var_mean_1[1];  var_mean_1 = None
        add_4 = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_2 = torch.ops.aten.sub.Tensor(permute_11, getitem_7);  permute_11 = None
        mul_3 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_4 = torch.ops.aten.mul.Tensor(mul_3, primals_10);  mul_3 = None
        add_5 = torch.ops.aten.add.Tensor(mul_4, primals_11);  mul_4 = primals_11 = None
        add_6 = torch.ops.aten.add.Tensor(add_5, expand);  add_5 = expand = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_2[0]
        getitem_9 = var_mean_2[1];  var_mean_2 = None
        add_7 = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_6, getitem_9)
        mul_5 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_6 = torch.ops.aten.mul.Tensor(mul_5, primals_12);  mul_5 = None
        add_8 = torch.ops.aten.add.Tensor(mul_6, primals_13);  mul_6 = primals_13 = None
        view_13 = torch.ops.aten.view.default(add_8, [500, 128]);  add_8 = None
        permute_12 = torch.ops.aten.permute.default(primals_14, [1, 0])
        addmm_1 = torch.ops.aten.addmm.default(primals_15, view_13, permute_12);  primals_15 = permute_12 = None
        view_14 = torch.ops.aten.view.default(addmm_1, [500, 1, 512])
        mul_7 = torch.ops.aten.mul.Tensor(view_14, 0.5)
        mul_8 = torch.ops.aten.mul.Tensor(view_14, 0.7071067811865476);  view_14 = None
        erf = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add_9 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_7, add_9);  mul_7 = add_9 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(mul_9, [2], correction = 0, keepdim = True)
        getitem_10 = var_mean_3[0]
        getitem_11 = var_mean_3[1];  var_mean_3 = None
        add_10 = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_4 = torch.ops.aten.sub.Tensor(mul_9, getitem_11);  mul_9 = None
        mul_10 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = None
        mul_11 = torch.ops.aten.mul.Tensor(mul_10, primals_16);  mul_10 = None
        add_11 = torch.ops.aten.add.Tensor(mul_11, primals_17);  mul_11 = primals_17 = None
        permute_24 = torch.ops.aten.permute.default(permute_7, [0, 2, 1]);  permute_7 = None
        permute_25 = torch.ops.aten.permute.default(permute_8, [0, 2, 1]);  permute_8 = None
        permute_26 = torch.ops.aten.permute.default(mul_2, [0, 2, 1]);  mul_2 = None
        permute_34 = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        permute_38 = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        return (add_11, add_6, primals_1, primals_4, primals_8, primals_10, primals_12, primals_14, primals_16, cat_1, getitem_1, rsqrt, view_2, div, view_10, addmm, getitem_7, rsqrt_1, add_6, getitem_9, rsqrt_2, view_13, addmm_1, getitem_11, rsqrt_3, permute_24, permute_25, permute_26, permute_34, permute_38)
        
def load_args(reader):
    buf0 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf0, (1, 1, 128), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf1, (500, 128), is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf2, (500, 128, 128), is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf3, (128,), is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf4, (128,), is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf5, (384, 128), is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf6, (384,), is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf7, (128, 128), is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf8, (128,), is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf9, (128,), is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf10, (128,), is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf11, (128,), is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf12, (128,), is_leaf=True)  # primals_13
    buf13 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf13, (512, 128), is_leaf=True)  # primals_14
    buf14 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf14, (512,), is_leaf=True)  # primals_15
    buf15 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf15, (512,), is_leaf=True)  # primals_16
    buf16 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf16, (512,), is_leaf=True)  # primals_17
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)