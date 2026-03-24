
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
        clamp_min = torch.ops.aten.clamp_min.default(arg0_1, 1e-08);  arg0_1 = None
        log = torch.ops.aten.log.default(clamp_min);  clamp_min = None
        clamp_min_1 = torch.ops.aten.clamp_min.default(arg1_1, 1e-08);  arg1_1 = None
        log_1 = torch.ops.aten.log.default(clamp_min_1);  clamp_min_1 = None
        clamp_min_2 = torch.ops.aten.clamp_min.default(arg2_1, 1e-08);  arg2_1 = None
        log_2 = torch.ops.aten.log.default(clamp_min_2);  clamp_min_2 = None
        clamp_min_3 = torch.ops.aten.clamp_min.default(arg3_1, 1e-08);  arg3_1 = None
        log_3 = torch.ops.aten.log.default(clamp_min_3);  clamp_min_3 = None
        isnan = torch.ops.aten.isnan.default(log)
        any_1 = torch.ops.aten.any.default(isnan);  isnan = None
        return (any_1, log, log_1, log_2, log_3)
        
def load_args(reader):
    buf0 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf0, (500, 128, 128), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf1, (500, 128, 128), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf2, (500, 128, 128), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf3, (500, 128, 128), is_leaf=True)  # arg3_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)