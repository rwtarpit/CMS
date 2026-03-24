
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

    
    
    def forward(self, arg0_1):
        select = torch.ops.aten.select.int(arg0_1, 2, 3)
        gt = torch.ops.aten.gt.Scalar(select, 0);  select = None
        select_1 = torch.ops.aten.select.int(arg0_1, 2, 0)
        select_2 = torch.ops.aten.select.int(arg0_1, 2, 1)
        select_3 = torch.ops.aten.select.int(arg0_1, 2, 2)
        select_4 = torch.ops.aten.select.int(arg0_1, 2, 3);  arg0_1 = None
        cos = torch.ops.aten.cos.default(select_3)
        mul = torch.ops.aten.mul.Tensor(select_1, cos);  cos = None
        sin = torch.ops.aten.sin.default(select_3)
        mul_1 = torch.ops.aten.mul.Tensor(select_1, sin);  sin = None
        sinh = torch.ops.aten.sinh.default(select_2)
        mul_2 = torch.ops.aten.mul.Tensor(select_1, sinh);  sinh = None
        unsqueeze = torch.ops.aten.unsqueeze.default(mul, 2);  mul = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(mul_1, 2);  mul_1 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(mul_2, 2);  mul_2 = None
        cat = torch.ops.aten.cat.default([unsqueeze, unsqueeze_1, unsqueeze_2], -1);  unsqueeze = unsqueeze_1 = unsqueeze_2 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(select_2, 2)
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(select_2, 1);  select_2 = None
        sub = torch.ops.aten.sub.Tensor(unsqueeze_3, unsqueeze_4);  unsqueeze_3 = unsqueeze_4 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(select_3, 2)
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(select_3, 1);  select_3 = None
        sub_1 = torch.ops.aten.sub.Tensor(unsqueeze_5, unsqueeze_6);  unsqueeze_5 = unsqueeze_6 = None
        add = torch.ops.aten.add.Tensor(sub_1, 3.141592653589793);  sub_1 = None
        remainder = torch.ops.aten.remainder.Scalar(add, 6.283185307179586);  add = None
        sub_2 = torch.ops.aten.sub.Tensor(remainder, 3.141592653589793);  remainder = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(select_1, 2)
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(select_1, 1)
        minimum = torch.ops.aten.minimum.default(unsqueeze_7, unsqueeze_8);  unsqueeze_7 = unsqueeze_8 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(select_1, 2)
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(select_1, 1);  select_1 = None
        add_1 = torch.ops.aten.add.Tensor(unsqueeze_9, unsqueeze_10);  unsqueeze_9 = unsqueeze_10 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(select_4, 2)
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(select_4, 1);  select_4 = None
        add_2 = torch.ops.aten.add.Tensor(unsqueeze_11, unsqueeze_12);  unsqueeze_11 = unsqueeze_12 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(cat, 2)
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(cat, 1);  cat = None
        add_3 = torch.ops.aten.add.Tensor(unsqueeze_13, unsqueeze_14);  unsqueeze_13 = unsqueeze_14 = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(sub, 2);  sub = None
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(sub_2, 2);  sub_2 = None
        add_4 = torch.ops.aten.add.Tensor(pow_1, pow_2);  pow_1 = pow_2 = None
        sqrt = torch.ops.aten.sqrt.default(add_4);  add_4 = None
        mul_3 = torch.ops.aten.mul.Tensor(minimum, sqrt)
        add_5 = torch.ops.aten.add.Tensor(add_1, 1e-08);  add_1 = None
        div = torch.ops.aten.div.Tensor(minimum, add_5);  minimum = add_5 = None
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(add_2, 2);  add_2 = None
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(add_3, 2);  add_3 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(pow_4, [-1]);  pow_4 = None
        pow_5 = torch.ops.aten.pow.Tensor_Scalar(sum_1, 0.5);  sum_1 = None
        pow_6 = torch.ops.aten.pow.Tensor_Scalar(pow_5, 2);  pow_5 = None
        sub_3 = torch.ops.aten.sub.Tensor(pow_3, pow_6);  pow_3 = pow_6 = None
        isnan = torch.ops.aten.isnan.default(sqrt)
        any_1 = torch.ops.aten.any.default(isnan);  isnan = None
        return (any_1, gt, sqrt, mul_3, div, sub_3)
        
def load_args(reader):
    buf0 = reader.storage(None, 1024000, device=device(type='cuda', index=0))
    reader.tensor(buf0, (500, 128, 4), is_leaf=True)  # arg0_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)