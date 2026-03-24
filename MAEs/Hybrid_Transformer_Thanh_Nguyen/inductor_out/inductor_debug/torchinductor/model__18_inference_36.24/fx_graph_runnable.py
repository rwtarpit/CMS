s38 = 893092


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
        self.register_buffer('_tensor_constant0', tensor(0.))

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
        index_put = torch.ops.aten.index_put.default(arg0_1, [arg1_1], arg3_1);  arg1_1 = arg3_1 = None
        iota = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        _tensor_constant0 = self._tensor_constant0;  _tensor_constant0 = None
        full_default = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        index_put_1 = torch.ops.aten.index_put.default(index_put, [None, iota, iota], full_default);  index_put = iota = full_default = None
        copy_ = torch.ops.aten.copy_.default(arg0_1, index_put_1);  arg0_1 = index_put_1 = None
        return (copy_,)
        
def load_args(reader):
    buf0 = reader.storage(None, 131072000, device=device(type='cuda', index=0))
    reader.tensor(buf0, (500, 128, 128, 4), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 8192000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf1, (500, 128, 128), dtype=torch.bool, is_leaf=True)  # arg1_1
    reader.symint(893092)  # arg2_1
    buf2 = reader.storage(None, 16*s38, device=device(type='cuda', index=0))
    reader.tensor(buf2, (s38, 4), is_leaf=True)  # arg3_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='symbolic', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='symbolic', check_str=None)
        # mod(*args)