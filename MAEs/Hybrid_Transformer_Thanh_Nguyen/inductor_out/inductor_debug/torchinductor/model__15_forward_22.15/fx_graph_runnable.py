
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22):
        view = torch.ops.aten.view.default(primals_1, [64000, 512]);  primals_1 = None
        permute = torch.ops.aten.permute.default(primals_2, [1, 0])
        addmm = torch.ops.aten.addmm.default(primals_3, view, permute);  primals_3 = permute = None
        view_1 = torch.ops.aten.view.default(addmm, [500, 128, 128]);  addmm = None
        inductor_seeds_default = torch.ops.prims.inductor_seeds.default(5, device(type='cuda', index=0))
        inductor_lookup_seed_default = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_4 = torch.ops.prims.inductor_random.default([500, 128, 128], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt = torch.ops.aten.gt.Scalar(inductor_random_default_4, 0.1);  inductor_random_default_4 = None
        mul = torch.ops.aten.mul.Tensor(gt, view_1);  view_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        add = torch.ops.aten.add.Tensor(mul_1, primals_4);  mul_1 = primals_4 = None
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_1 = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub = torch.ops.aten.sub.Tensor(add, getitem_1);  getitem_1 = None
        mul_2 = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_3 = torch.ops.aten.mul.Tensor(mul_2, primals_5)
        add_2 = torch.ops.aten.add.Tensor(mul_3, primals_6);  mul_3 = primals_6 = None
        permute_1 = torch.ops.aten.permute.default(add_2, [1, 0, 2]);  add_2 = None
        permute_2 = torch.ops.aten.permute.default(primals_7, [1, 0])
        clone = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_2 = torch.ops.aten.view.default(clone, [64000, 128]);  clone = None
        mm = torch.ops.aten.mm.default(view_2, permute_2);  permute_2 = None
        view_3 = torch.ops.aten.view.default(mm, [128, 500, 384]);  mm = None
        add_3 = torch.ops.aten.add.Tensor(view_3, primals_8);  view_3 = primals_8 = None
        view_4 = torch.ops.aten.view.default(add_3, [128, 500, 3, 128]);  add_3 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(view_4, 0);  view_4 = None
        permute_3 = torch.ops.aten.permute.default(unsqueeze, [3, 1, 2, 0, 4]);  unsqueeze = None
        squeeze = torch.ops.aten.squeeze.dim(permute_3, -2);  permute_3 = None
        clone_1 = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
        select = torch.ops.aten.select.int(clone_1, 0, 0)
        select_1 = torch.ops.aten.select.int(clone_1, 0, 1)
        select_2 = torch.ops.aten.select.int(clone_1, 0, 2);  clone_1 = None
        view_5 = torch.ops.aten.view.default(select, [128, 4000, 16]);  select = None
        permute_4 = torch.ops.aten.permute.default(view_5, [1, 0, 2]);  view_5 = None
        view_6 = torch.ops.aten.view.default(select_1, [128, 4000, 16]);  select_1 = None
        permute_5 = torch.ops.aten.permute.default(view_6, [1, 0, 2]);  view_6 = None
        view_7 = torch.ops.aten.view.default(select_2, [128, 4000, 16]);  select_2 = None
        permute_6 = torch.ops.aten.permute.default(view_7, [1, 0, 2]);  view_7 = None
        view_8 = torch.ops.aten.view.default(primals_11, [500, 1, 1, 128]);  primals_11 = None
        expand = torch.ops.aten.expand.default(view_8, [-1, 8, -1, -1]);  view_8 = None
        clone_2 = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_9 = torch.ops.aten.view.default(clone_2, [4000, 1, 128]);  clone_2 = None
        add_4 = torch.ops.aten.add.Tensor(primals_12, view_9);  primals_12 = view_9 = None
        mul_4 = torch.ops.aten.mul.Tensor(permute_4, 0.25);  permute_4 = None
        permute_7 = torch.ops.aten.permute.default(permute_5, [0, 2, 1]);  permute_5 = None
        baddbmm = torch.ops.aten.baddbmm.default(add_4, mul_4, permute_7);  add_4 = None
        amax = torch.ops.aten.amax.default(baddbmm, [-1], True)
        sub_1 = torch.ops.aten.sub.Tensor(baddbmm, amax)
        exp = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        inductor_lookup_seed_default_1 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_3 = torch.ops.prims.inductor_random.default([4000, 128, 128], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_1 = torch.ops.aten.gt.Scalar(inductor_random_default_3, 0.1);  inductor_random_default_3 = None
        mul_5 = torch.ops.aten.mul.Tensor(gt_1, div);  div = None
        mul_6 = torch.ops.aten.mul.Tensor(mul_5, 1.1111111111111112);  mul_5 = None
        bmm = torch.ops.aten.bmm.default(mul_6, permute_6)
        permute_8 = torch.ops.aten.permute.default(bmm, [1, 0, 2]);  bmm = None
        clone_3 = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        view_10 = torch.ops.aten.view.default(clone_3, [64000, 128]);  clone_3 = None
        permute_9 = torch.ops.aten.permute.default(primals_9, [1, 0])
        addmm_1 = torch.ops.aten.addmm.default(primals_10, view_10, permute_9);  primals_10 = permute_9 = None
        view_11 = torch.ops.aten.view.default(addmm_1, [128, 500, 128])
        permute_10 = torch.ops.aten.permute.default(view_11, [1, 0, 2]);  view_11 = None
        clone_4 = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(clone_4, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_5 = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_2 = torch.ops.aten.sub.Tensor(clone_4, getitem_3);  clone_4 = None
        mul_7 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_8 = torch.ops.aten.mul.Tensor(mul_7, primals_13);  mul_7 = None
        add_6 = torch.ops.aten.add.Tensor(mul_8, primals_14);  mul_8 = primals_14 = None
        inductor_lookup_seed_default_2 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_2 = torch.ops.prims.inductor_random.default([500, 128, 128], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        gt_2 = torch.ops.aten.gt.Scalar(inductor_random_default_2, 0.1);  inductor_random_default_2 = None
        mul_9 = torch.ops.aten.mul.Tensor(gt_2, add_6);  add_6 = None
        mul_10 = torch.ops.aten.mul.Tensor(mul_9, 1.1111111111111112);  mul_9 = None
        add_7 = torch.ops.aten.add.Tensor(mul_10, add);  mul_10 = add = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_2[0]
        getitem_5 = var_mean_2[1];  var_mean_2 = None
        add_8 = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_7, getitem_5);  getitem_5 = None
        mul_11 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_12 = torch.ops.aten.mul.Tensor(mul_11, primals_15)
        add_9 = torch.ops.aten.add.Tensor(mul_12, primals_16);  mul_12 = primals_16 = None
        view_13 = torch.ops.aten.view.default(add_9, [64000, 128]);  add_9 = None
        permute_11 = torch.ops.aten.permute.default(primals_17, [1, 0])
        addmm_2 = torch.ops.aten.addmm.default(primals_18, view_13, permute_11);  primals_18 = permute_11 = None
        view_14 = torch.ops.aten.view.default(addmm_2, [500, 128, 512])
        mul_13 = torch.ops.aten.mul.Tensor(view_14, 0.5)
        mul_14 = torch.ops.aten.mul.Tensor(view_14, 0.7071067811865476);  view_14 = None
        erf = torch.ops.aten.erf.default(mul_14);  mul_14 = None
        add_10 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_15 = torch.ops.aten.mul.Tensor(mul_13, add_10);  mul_13 = add_10 = None
        inductor_lookup_seed_default_3 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_1 = torch.ops.prims.inductor_random.default([500, 128, 512], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        gt_3 = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.1);  inductor_random_default_1 = None
        mul_16 = torch.ops.aten.mul.Tensor(gt_3, mul_15);  mul_15 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_16, 1.1111111111111112);  mul_16 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(mul_17, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_3[0]
        getitem_7 = var_mean_3[1];  var_mean_3 = None
        add_11 = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_4 = torch.ops.aten.sub.Tensor(mul_17, getitem_7);  mul_17 = None
        mul_18 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = None
        mul_19 = torch.ops.aten.mul.Tensor(mul_18, primals_19);  mul_18 = None
        add_12 = torch.ops.aten.add.Tensor(mul_19, primals_20);  mul_19 = primals_20 = None
        view_15 = torch.ops.aten.view.default(add_12, [64000, 512]);  add_12 = None
        permute_12 = torch.ops.aten.permute.default(primals_21, [1, 0])
        addmm_3 = torch.ops.aten.addmm.default(primals_22, view_15, permute_12);  primals_22 = permute_12 = None
        view_16 = torch.ops.aten.view.default(addmm_3, [500, 128, 128]);  addmm_3 = None
        inductor_lookup_seed_default_4 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4);  inductor_seeds_default = None
        inductor_random_default = torch.ops.prims.inductor_random.default([500, 128, 128], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        gt_4 = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_20 = torch.ops.aten.mul.Tensor(gt_4, view_16);  view_16 = None
        mul_21 = torch.ops.aten.mul.Tensor(mul_20, 1.1111111111111112);  mul_20 = None
        add_13 = torch.ops.aten.add.Tensor(mul_21, add_7);  mul_21 = add_7 = None
        div_2 = torch.ops.aten.div.Tensor(rsqrt_2, 128);  rsqrt_2 = None
        permute_27 = torch.ops.aten.permute.default(mul_6, [0, 2, 1]);  mul_6 = None
        permute_28 = torch.ops.aten.permute.default(permute_6, [0, 2, 1]);  permute_6 = None
        permute_29 = torch.ops.aten.permute.default(permute_7, [0, 2, 1]);  permute_7 = None
        permute_30 = torch.ops.aten.permute.default(mul_4, [0, 2, 1]);  mul_4 = None
        div_4 = torch.ops.aten.div.Tensor(rsqrt, 128);  rsqrt = None
        return (add_13, primals_2, primals_5, primals_7, primals_9, primals_13, primals_15, primals_17, primals_19, primals_21, view, gt, mul_2, view_2, baddbmm, amax, sum_1, gt_1, view_10, addmm_1, getitem_3, rsqrt_1, gt_2, mul_11, view_13, addmm_2, gt_3, getitem_7, rsqrt_3, view_15, gt_4, div_2, permute_27, permute_28, permute_29, permute_30, div_4)
        
def load_args(reader):
    buf0 = reader.storage(None, 131072000, device=device(type='cuda', index=0))
    reader.tensor(buf0, (500, 128, 512), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf1, (128, 512), is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf2, (128,), is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf3, (500, 128, 128), is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf4, (128,), is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf5, (128,), is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf6, (384, 128), is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf7, (384,), is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf8, (128, 128), is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf9, (128,), is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf10, (500, 128), is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 262144000, device=device(type='cuda', index=0))
    reader.tensor(buf11, (4000, 128, 128), is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf12, (128,), is_leaf=True)  # primals_13
    buf13 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf13, (128,), is_leaf=True)  # primals_14
    buf14 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf14, (128,), is_leaf=True)  # primals_15
    buf15 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf15, (128,), is_leaf=True)  # primals_16
    buf16 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf16, (512, 128), is_leaf=True)  # primals_17
    buf17 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf17, (512,), is_leaf=True)  # primals_18
    buf18 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf18, (512,), is_leaf=True)  # primals_19
    buf19 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf19, (512,), is_leaf=True)  # primals_20
    buf20 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf20, (128, 512), is_leaf=True)  # primals_21
    buf21 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf21, (128,), is_leaf=True)  # primals_22
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)