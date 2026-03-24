
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35):
        view = torch.ops.aten.view.default(primals_2, [500, 16384, 4])
        permute = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
        add = torch.ops.aten.add.Tensor(primals_3, 1)
        clone = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        var_mean = torch.ops.aten.var_mean.correction(clone, [0, 2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_1 = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub = torch.ops.aten.sub.Tensor(clone, getitem_1);  clone = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze = torch.ops.aten.squeeze.dims(getitem_1, [0, 2]);  getitem_1 = None
        squeeze_1 = torch.ops.aten.squeeze.dims(rsqrt, [0, 2]);  rsqrt = None
        mul_1 = torch.ops.aten.mul.Tensor(squeeze, 0.1)
        mul_2 = torch.ops.aten.mul.Tensor(primals_4, 0.9)
        add_2 = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_2 = torch.ops.aten.squeeze.dims(getitem, [0, 2]);  getitem = None
        mul_3 = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000001220703274);  squeeze_2 = None
        mul_4 = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5 = torch.ops.aten.mul.Tensor(primals_5, 0.9)
        add_3 = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
        mul_6 = torch.ops.aten.mul.Tensor(mul, unsqueeze);  mul = unsqueeze = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(primals_7, -1);  primals_7 = None
        add_4 = torch.ops.aten.add.Tensor(mul_6, unsqueeze_1);  mul_6 = unsqueeze_1 = None
        convolution = torch.ops.aten.convolution.default(add_4, primals_8, primals_9, [1], [0], [1], False, [0], 1);  primals_9 = None
        add_5 = torch.ops.aten.add.Tensor(primals_10, 1)
        var_mean_1 = torch.ops.aten.var_mean.correction(convolution, [0, 2], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_6 = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_1 = torch.ops.aten.sub.Tensor(convolution, getitem_3)
        mul_7 = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        squeeze_3 = torch.ops.aten.squeeze.dims(getitem_3, [0, 2])
        mul_8 = torch.ops.aten.mul.Tensor(squeeze_3, 0.1);  squeeze_3 = None
        mul_9 = torch.ops.aten.mul.Tensor(primals_11, 0.9)
        add_7 = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        squeeze_5 = torch.ops.aten.squeeze.dims(getitem_2, [0, 2]);  getitem_2 = None
        mul_10 = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000001220703274);  squeeze_5 = None
        mul_11 = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
        mul_12 = torch.ops.aten.mul.Tensor(primals_12, 0.9)
        add_8 = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(primals_13, -1)
        mul_13 = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_2);  mul_7 = unsqueeze_2 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(primals_14, -1)
        add_9 = torch.ops.aten.add.Tensor(mul_13, unsqueeze_3);  mul_13 = unsqueeze_3 = None
        mul_14 = torch.ops.aten.mul.Tensor(add_9, 0.5)
        mul_15 = torch.ops.aten.mul.Tensor(add_9, 0.7071067811865476);  add_9 = None
        erf = torch.ops.aten.erf.default(mul_15);  mul_15 = None
        add_10 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_16 = torch.ops.aten.mul.Tensor(mul_14, add_10);  mul_14 = add_10 = None
        convolution_1 = torch.ops.aten.convolution.default(mul_16, primals_15, primals_16, [1], [0], [1], False, [0], 1);  primals_16 = None
        add_11 = torch.ops.aten.add.Tensor(primals_17, 1)
        var_mean_2 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2], correction = 0, keepdim = True)
        getitem_4 = var_mean_2[0]
        getitem_5 = var_mean_2[1];  var_mean_2 = None
        add_12 = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_2 = torch.ops.aten.sub.Tensor(convolution_1, getitem_5)
        mul_17 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        squeeze_6 = torch.ops.aten.squeeze.dims(getitem_5, [0, 2])
        mul_18 = torch.ops.aten.mul.Tensor(squeeze_6, 0.1);  squeeze_6 = None
        mul_19 = torch.ops.aten.mul.Tensor(primals_18, 0.9)
        add_13 = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        squeeze_8 = torch.ops.aten.squeeze.dims(getitem_4, [0, 2]);  getitem_4 = None
        mul_20 = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000001220703274);  squeeze_8 = None
        mul_21 = torch.ops.aten.mul.Tensor(mul_20, 0.1);  mul_20 = None
        mul_22 = torch.ops.aten.mul.Tensor(primals_19, 0.9)
        add_14 = torch.ops.aten.add.Tensor(mul_21, mul_22);  mul_21 = mul_22 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(primals_20, -1)
        mul_23 = torch.ops.aten.mul.Tensor(mul_17, unsqueeze_4);  mul_17 = unsqueeze_4 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(primals_21, -1)
        add_15 = torch.ops.aten.add.Tensor(mul_23, unsqueeze_5);  mul_23 = unsqueeze_5 = None
        mul_24 = torch.ops.aten.mul.Tensor(add_15, 0.5)
        mul_25 = torch.ops.aten.mul.Tensor(add_15, 0.7071067811865476);  add_15 = None
        erf_1 = torch.ops.aten.erf.default(mul_25);  mul_25 = None
        add_16 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_26 = torch.ops.aten.mul.Tensor(mul_24, add_16);  mul_24 = add_16 = None
        convolution_2 = torch.ops.aten.convolution.default(mul_26, primals_22, primals_23, [1], [0], [1], False, [0], 1);  primals_23 = None
        add_17 = torch.ops.aten.add.Tensor(primals_24, 1)
        var_mean_3 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2], correction = 0, keepdim = True)
        getitem_6 = var_mean_3[0]
        getitem_7 = var_mean_3[1];  var_mean_3 = None
        add_18 = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_3 = torch.ops.aten.sub.Tensor(convolution_2, getitem_7)
        mul_27 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        squeeze_9 = torch.ops.aten.squeeze.dims(getitem_7, [0, 2])
        mul_28 = torch.ops.aten.mul.Tensor(squeeze_9, 0.1);  squeeze_9 = None
        mul_29 = torch.ops.aten.mul.Tensor(primals_25, 0.9)
        add_19 = torch.ops.aten.add.Tensor(mul_28, mul_29);  mul_28 = mul_29 = None
        squeeze_11 = torch.ops.aten.squeeze.dims(getitem_6, [0, 2]);  getitem_6 = None
        mul_30 = torch.ops.aten.mul.Tensor(squeeze_11, 1.0000001220703274);  squeeze_11 = None
        mul_31 = torch.ops.aten.mul.Tensor(mul_30, 0.1);  mul_30 = None
        mul_32 = torch.ops.aten.mul.Tensor(primals_26, 0.9)
        add_20 = torch.ops.aten.add.Tensor(mul_31, mul_32);  mul_31 = mul_32 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(primals_27, -1)
        mul_33 = torch.ops.aten.mul.Tensor(mul_27, unsqueeze_6);  mul_27 = unsqueeze_6 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(primals_28, -1)
        add_21 = torch.ops.aten.add.Tensor(mul_33, unsqueeze_7);  mul_33 = unsqueeze_7 = None
        mul_34 = torch.ops.aten.mul.Tensor(add_21, 0.5)
        mul_35 = torch.ops.aten.mul.Tensor(add_21, 0.7071067811865476);  add_21 = None
        erf_2 = torch.ops.aten.erf.default(mul_35);  mul_35 = None
        add_22 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_36 = torch.ops.aten.mul.Tensor(mul_34, add_22);  mul_34 = add_22 = None
        convolution_3 = torch.ops.aten.convolution.default(mul_36, primals_29, primals_30, [1], [0], [1], False, [0], 1);  primals_30 = None
        add_23 = torch.ops.aten.add.Tensor(primals_31, 1)
        var_mean_4 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2], correction = 0, keepdim = True)
        getitem_8 = var_mean_4[0]
        getitem_9 = var_mean_4[1];  var_mean_4 = None
        add_24 = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        sub_4 = torch.ops.aten.sub.Tensor(convolution_3, getitem_9)
        mul_37 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        squeeze_12 = torch.ops.aten.squeeze.dims(getitem_9, [0, 2])
        mul_38 = torch.ops.aten.mul.Tensor(squeeze_12, 0.1);  squeeze_12 = None
        mul_39 = torch.ops.aten.mul.Tensor(primals_32, 0.9)
        add_25 = torch.ops.aten.add.Tensor(mul_38, mul_39);  mul_38 = mul_39 = None
        squeeze_14 = torch.ops.aten.squeeze.dims(getitem_8, [0, 2]);  getitem_8 = None
        mul_40 = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000001220703274);  squeeze_14 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_40, 0.1);  mul_40 = None
        mul_42 = torch.ops.aten.mul.Tensor(primals_33, 0.9)
        add_26 = torch.ops.aten.add.Tensor(mul_41, mul_42);  mul_41 = mul_42 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(primals_34, -1)
        mul_43 = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_8);  mul_37 = unsqueeze_8 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(primals_35, -1)
        add_27 = torch.ops.aten.add.Tensor(mul_43, unsqueeze_9);  mul_43 = unsqueeze_9 = None
        mul_44 = torch.ops.aten.mul.Tensor(add_27, 0.5)
        mul_45 = torch.ops.aten.mul.Tensor(add_27, 0.7071067811865476);  add_27 = None
        erf_3 = torch.ops.aten.erf.default(mul_45);  mul_45 = None
        add_28 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_46 = torch.ops.aten.mul.Tensor(mul_44, add_28);  mul_44 = add_28 = None
        view_1 = torch.ops.aten.view.default(mul_46, [4000, 128, 128]);  mul_46 = None
        view_2 = torch.ops.aten.view.default(primals_1, [500, 128, 1, 16]);  primals_1 = None
        unsqueeze_42 = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_43 = torch.ops.aten.unsqueeze.default(unsqueeze_42, 2);  unsqueeze_42 = None
        copy_ = torch.ops.aten.copy_.default(primals_3, add);  primals_3 = add = copy_ = None
        copy__1 = torch.ops.aten.copy_.default(primals_4, add_2);  primals_4 = add_2 = copy__1 = None
        copy__2 = torch.ops.aten.copy_.default(primals_5, add_3);  primals_5 = add_3 = copy__2 = None
        copy__3 = torch.ops.aten.copy_.default(primals_10, add_5);  primals_10 = add_5 = copy__3 = None
        copy__4 = torch.ops.aten.copy_.default(primals_11, add_7);  primals_11 = add_7 = copy__4 = None
        copy__5 = torch.ops.aten.copy_.default(primals_12, add_8);  primals_12 = add_8 = copy__5 = None
        copy__6 = torch.ops.aten.copy_.default(primals_17, add_11);  primals_17 = add_11 = copy__6 = None
        copy__7 = torch.ops.aten.copy_.default(primals_18, add_13);  primals_18 = add_13 = copy__7 = None
        copy__8 = torch.ops.aten.copy_.default(primals_19, add_14);  primals_19 = add_14 = copy__8 = None
        copy__9 = torch.ops.aten.copy_.default(primals_24, add_17);  primals_24 = add_17 = copy__9 = None
        copy__10 = torch.ops.aten.copy_.default(primals_25, add_19);  primals_25 = add_19 = copy__10 = None
        copy__11 = torch.ops.aten.copy_.default(primals_26, add_20);  primals_26 = add_20 = copy__11 = None
        copy__12 = torch.ops.aten.copy_.default(primals_31, add_23);  primals_31 = add_23 = copy__12 = None
        copy__13 = torch.ops.aten.copy_.default(primals_32, add_25);  primals_32 = add_25 = copy__13 = None
        copy__14 = torch.ops.aten.copy_.default(primals_33, add_26);  primals_33 = add_26 = copy__14 = None
        return (view_2, view_1, primals_2, primals_8, primals_13, primals_14, primals_15, primals_20, primals_21, primals_22, primals_27, primals_28, primals_29, primals_34, primals_35, squeeze_1, add_4, convolution, getitem_3, rsqrt_1, mul_16, convolution_1, getitem_5, rsqrt_2, mul_26, convolution_2, getitem_7, rsqrt_3, mul_36, convolution_3, getitem_9, rsqrt_4, unsqueeze_43)
        
def load_args(reader):
    buf0 = reader.storage(None, 4096000, device=device(type='cuda', index=0))
    reader.tensor(buf0, (500, 128, 16), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 131072000, device=device(type='cuda', index=0))
    reader.tensor(buf1, (500, 128, 128, 4), is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf2, (), dtype=torch.int64, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 16, device=device(type='cuda', index=0))
    reader.tensor(buf3, (4,), is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 16, device=device(type='cuda', index=0))
    reader.tensor(buf4, (4,), is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 16, device=device(type='cuda', index=0))
    reader.tensor(buf5, (4,), is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 16, device=device(type='cuda', index=0))
    reader.tensor(buf6, (4,), is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf7, (64, 4, 1), is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf8, (64,), is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf9, (), dtype=torch.int64, is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf10, (64,), is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf11, (64,), is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf12, (64,), is_leaf=True)  # primals_13
    buf13 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf13, (64,), is_leaf=True)  # primals_14
    buf14 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf14, (64, 64, 1), is_leaf=True)  # primals_15
    buf15 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf15, (64,), is_leaf=True)  # primals_16
    buf16 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf16, (), dtype=torch.int64, is_leaf=True)  # primals_17
    buf17 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf17, (64,), is_leaf=True)  # primals_18
    buf18 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf18, (64,), is_leaf=True)  # primals_19
    buf19 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf19, (64,), is_leaf=True)  # primals_20
    buf20 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf20, (64,), is_leaf=True)  # primals_21
    buf21 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf21, (64, 64, 1), is_leaf=True)  # primals_22
    buf22 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf22, (64,), is_leaf=True)  # primals_23
    buf23 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf23, (), dtype=torch.int64, is_leaf=True)  # primals_24
    buf24 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf24, (64,), is_leaf=True)  # primals_25
    buf25 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf25, (64,), is_leaf=True)  # primals_26
    buf26 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf26, (64,), is_leaf=True)  # primals_27
    buf27 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf27, (64,), is_leaf=True)  # primals_28
    buf28 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf28, (8, 64, 1), is_leaf=True)  # primals_29
    buf29 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf29, (8,), is_leaf=True)  # primals_30
    buf30 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf30, (), dtype=torch.int64, is_leaf=True)  # primals_31
    buf31 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf31, (8,), is_leaf=True)  # primals_32
    buf32 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf32, (8,), is_leaf=True)  # primals_33
    buf33 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf33, (8,), is_leaf=True)  # primals_34
    buf34 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf34, (8,), is_leaf=True)  # primals_35
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)