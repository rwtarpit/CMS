
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115):
        view = torch.ops.aten.view.default(primals_1, [500, 128, 16]);  primals_1 = None
        view_1 = torch.ops.aten.view.default(view, [64000, 16]);  view = None
        permute = torch.ops.aten.permute.default(primals_2, [1, 0])
        addmm = torch.ops.aten.addmm.default(primals_3, view_1, permute);  primals_3 = permute = None
        view_2 = torch.ops.aten.view.default(addmm, [500, 128, 128])
        var_mean = torch.ops.aten.var_mean.correction(view_2, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
        sub = torch.ops.aten.sub.Tensor(view_2, getitem_1)
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, primals_4);  mul = None
        add_1 = torch.ops.aten.add.Tensor(mul_1, primals_5);  mul_1 = primals_5 = None
        permute_1 = torch.ops.aten.permute.default(add_1, [1, 0, 2]);  add_1 = None
        permute_2 = torch.ops.aten.permute.default(primals_6, [1, 0])
        clone = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_3 = torch.ops.aten.view.default(clone, [64000, 128]);  clone = None
        mm = torch.ops.aten.mm.default(view_3, permute_2);  permute_2 = None
        view_4 = torch.ops.aten.view.default(mm, [128, 500, 384]);  mm = None
        add_2 = torch.ops.aten.add.Tensor(view_4, primals_7);  view_4 = primals_7 = None
        view_5 = torch.ops.aten.view.default(add_2, [128, 500, 3, 128]);  add_2 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(view_5, 0);  view_5 = None
        permute_3 = torch.ops.aten.permute.default(unsqueeze, [3, 1, 2, 0, 4]);  unsqueeze = None
        squeeze = torch.ops.aten.squeeze.dim(permute_3, -2);  permute_3 = None
        clone_1 = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
        select = torch.ops.aten.select.int(clone_1, 0, 0)
        select_1 = torch.ops.aten.select.int(clone_1, 0, 1)
        select_2 = torch.ops.aten.select.int(clone_1, 0, 2);  clone_1 = None
        view_6 = torch.ops.aten.view.default(select, [128, 4000, 16]);  select = None
        permute_4 = torch.ops.aten.permute.default(view_6, [1, 0, 2]);  view_6 = None
        view_7 = torch.ops.aten.view.default(select_1, [128, 4000, 16]);  select_1 = None
        permute_5 = torch.ops.aten.permute.default(view_7, [1, 0, 2]);  view_7 = None
        view_8 = torch.ops.aten.view.default(select_2, [128, 4000, 16]);  select_2 = None
        permute_6 = torch.ops.aten.permute.default(view_8, [1, 0, 2]);  view_8 = None
        view_9 = torch.ops.aten.view.default(primals_10, [500, 1, 1, 128]);  primals_10 = None
        expand = torch.ops.aten.expand.default(view_9, [-1, 8, -1, -1]);  view_9 = None
        clone_2 = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_10 = torch.ops.aten.view.default(clone_2, [4000, 1, 128]);  clone_2 = None
        add_3 = torch.ops.aten.add.Tensor(primals_11, view_10);  primals_11 = view_10 = None
        mul_2 = torch.ops.aten.mul.Tensor(permute_4, 0.25);  permute_4 = None
        permute_7 = torch.ops.aten.permute.default(permute_5, [0, 2, 1]);  permute_5 = None
        baddbmm = torch.ops.aten.baddbmm.default(add_3, mul_2, permute_7)
        amax = torch.ops.aten.amax.default(baddbmm, [-1], True)
        sub_1 = torch.ops.aten.sub.Tensor(baddbmm, amax)
        exp = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        inductor_seeds_default = torch.ops.prims.inductor_seeds.default(27, device(type='cuda', index=0))
        inductor_lookup_seed_default = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_26 = torch.ops.prims.inductor_random.default([4000, 128, 128], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt = torch.ops.aten.gt.Scalar(inductor_random_default_26, 0.1);  inductor_random_default_26 = None
        mul_3 = torch.ops.aten.mul.Tensor(gt, div);  div = None
        mul_4 = torch.ops.aten.mul.Tensor(mul_3, 1.1111111111111112);  mul_3 = None
        bmm = torch.ops.aten.bmm.default(mul_4, permute_6)
        permute_8 = torch.ops.aten.permute.default(bmm, [1, 0, 2]);  bmm = None
        clone_3 = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        view_11 = torch.ops.aten.view.default(clone_3, [64000, 128]);  clone_3 = None
        permute_9 = torch.ops.aten.permute.default(primals_8, [1, 0])
        addmm_1 = torch.ops.aten.addmm.default(primals_9, view_11, permute_9);  primals_9 = permute_9 = None
        view_12 = torch.ops.aten.view.default(addmm_1, [128, 500, 128])
        permute_10 = torch.ops.aten.permute.default(view_12, [1, 0, 2]);  view_12 = None
        clone_4 = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(clone_4, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_4 = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_2 = torch.ops.aten.sub.Tensor(clone_4, getitem_3);  clone_4 = None
        mul_5 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_6 = torch.ops.aten.mul.Tensor(mul_5, primals_12);  mul_5 = None
        add_5 = torch.ops.aten.add.Tensor(mul_6, primals_13);  mul_6 = primals_13 = None
        inductor_lookup_seed_default_1 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_25 = torch.ops.prims.inductor_random.default([500, 128, 128], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_1 = torch.ops.aten.gt.Scalar(inductor_random_default_25, 0.1);  inductor_random_default_25 = None
        mul_7 = torch.ops.aten.mul.Tensor(gt_1, add_5);  add_5 = None
        mul_8 = torch.ops.aten.mul.Tensor(mul_7, 1.1111111111111112);  mul_7 = None
        add_6 = torch.ops.aten.add.Tensor(mul_8, view_2);  mul_8 = view_2 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_2[0]
        getitem_5 = var_mean_2[1];  var_mean_2 = None
        add_7 = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_6, getitem_5);  getitem_5 = None
        mul_9 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_10 = torch.ops.aten.mul.Tensor(mul_9, primals_14)
        add_8 = torch.ops.aten.add.Tensor(mul_10, primals_15);  mul_10 = primals_15 = None
        view_14 = torch.ops.aten.view.default(add_8, [64000, 128]);  add_8 = None
        permute_11 = torch.ops.aten.permute.default(primals_16, [1, 0])
        addmm_2 = torch.ops.aten.addmm.default(primals_17, view_14, permute_11);  primals_17 = permute_11 = None
        view_15 = torch.ops.aten.view.default(addmm_2, [500, 128, 512])
        mul_11 = torch.ops.aten.mul.Tensor(view_15, 0.5)
        mul_12 = torch.ops.aten.mul.Tensor(view_15, 0.7071067811865476);  view_15 = None
        erf = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_9 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_11, add_9);  mul_11 = add_9 = None
        inductor_lookup_seed_default_2 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_24 = torch.ops.prims.inductor_random.default([500, 128, 512], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        gt_2 = torch.ops.aten.gt.Scalar(inductor_random_default_24, 0.1);  inductor_random_default_24 = None
        mul_14 = torch.ops.aten.mul.Tensor(gt_2, mul_13);  mul_13 = None
        mul_15 = torch.ops.aten.mul.Tensor(mul_14, 1.1111111111111112);  mul_14 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(mul_15, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_3[0]
        getitem_7 = var_mean_3[1];  var_mean_3 = None
        add_10 = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_4 = torch.ops.aten.sub.Tensor(mul_15, getitem_7);  mul_15 = None
        mul_16 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_16, primals_18);  mul_16 = None
        add_11 = torch.ops.aten.add.Tensor(mul_17, primals_19);  mul_17 = primals_19 = None
        view_16 = torch.ops.aten.view.default(add_11, [64000, 512]);  add_11 = None
        permute_12 = torch.ops.aten.permute.default(primals_20, [1, 0])
        addmm_3 = torch.ops.aten.addmm.default(primals_21, view_16, permute_12);  primals_21 = permute_12 = None
        view_17 = torch.ops.aten.view.default(addmm_3, [500, 128, 128]);  addmm_3 = None
        inductor_lookup_seed_default_3 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_23 = torch.ops.prims.inductor_random.default([500, 128, 128], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        gt_3 = torch.ops.aten.gt.Scalar(inductor_random_default_23, 0.1);  inductor_random_default_23 = None
        mul_18 = torch.ops.aten.mul.Tensor(gt_3, view_17);  view_17 = None
        mul_19 = torch.ops.aten.mul.Tensor(mul_18, 1.1111111111111112);  mul_18 = None
        add_12 = torch.ops.aten.add.Tensor(mul_19, add_6);  mul_19 = add_6 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_4[0]
        getitem_9 = var_mean_4[1];  var_mean_4 = None
        add_13 = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_5 = torch.ops.aten.sub.Tensor(add_12, getitem_9);  getitem_9 = None
        mul_20 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_4);  sub_5 = None
        mul_21 = torch.ops.aten.mul.Tensor(mul_20, primals_22)
        add_14 = torch.ops.aten.add.Tensor(mul_21, primals_23);  mul_21 = primals_23 = None
        permute_13 = torch.ops.aten.permute.default(add_14, [1, 0, 2]);  add_14 = None
        permute_14 = torch.ops.aten.permute.default(primals_24, [1, 0])
        clone_5 = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
        view_18 = torch.ops.aten.view.default(clone_5, [64000, 128]);  clone_5 = None
        mm_1 = torch.ops.aten.mm.default(view_18, permute_14);  permute_14 = None
        view_19 = torch.ops.aten.view.default(mm_1, [128, 500, 384]);  mm_1 = None
        add_15 = torch.ops.aten.add.Tensor(view_19, primals_25);  view_19 = primals_25 = None
        view_20 = torch.ops.aten.view.default(add_15, [128, 500, 3, 128]);  add_15 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(view_20, 0);  view_20 = None
        permute_15 = torch.ops.aten.permute.default(unsqueeze_1, [3, 1, 2, 0, 4]);  unsqueeze_1 = None
        squeeze_1 = torch.ops.aten.squeeze.dim(permute_15, -2);  permute_15 = None
        clone_6 = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
        select_3 = torch.ops.aten.select.int(clone_6, 0, 0)
        select_4 = torch.ops.aten.select.int(clone_6, 0, 1)
        select_5 = torch.ops.aten.select.int(clone_6, 0, 2);  clone_6 = None
        view_21 = torch.ops.aten.view.default(select_3, [128, 4000, 16]);  select_3 = None
        permute_16 = torch.ops.aten.permute.default(view_21, [1, 0, 2]);  view_21 = None
        view_22 = torch.ops.aten.view.default(select_4, [128, 4000, 16]);  select_4 = None
        permute_17 = torch.ops.aten.permute.default(view_22, [1, 0, 2]);  view_22 = None
        view_23 = torch.ops.aten.view.default(select_5, [128, 4000, 16]);  select_5 = None
        permute_18 = torch.ops.aten.permute.default(view_23, [1, 0, 2]);  view_23 = None
        mul_22 = torch.ops.aten.mul.Tensor(permute_16, 0.25);  permute_16 = None
        permute_19 = torch.ops.aten.permute.default(permute_17, [0, 2, 1]);  permute_17 = None
        baddbmm_1 = torch.ops.aten.baddbmm.default(add_3, mul_22, permute_19)
        amax_1 = torch.ops.aten.amax.default(baddbmm_1, [-1], True)
        sub_6 = torch.ops.aten.sub.Tensor(baddbmm_1, amax_1)
        exp_1 = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1 = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = None
        inductor_lookup_seed_default_4 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_22 = torch.ops.prims.inductor_random.default([4000, 128, 128], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        gt_4 = torch.ops.aten.gt.Scalar(inductor_random_default_22, 0.1);  inductor_random_default_22 = None
        mul_23 = torch.ops.aten.mul.Tensor(gt_4, div_1);  div_1 = None
        mul_24 = torch.ops.aten.mul.Tensor(mul_23, 1.1111111111111112);  mul_23 = None
        bmm_1 = torch.ops.aten.bmm.default(mul_24, permute_18)
        permute_20 = torch.ops.aten.permute.default(bmm_1, [1, 0, 2]);  bmm_1 = None
        clone_8 = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None
        view_26 = torch.ops.aten.view.default(clone_8, [64000, 128]);  clone_8 = None
        permute_21 = torch.ops.aten.permute.default(primals_26, [1, 0])
        addmm_4 = torch.ops.aten.addmm.default(primals_27, view_26, permute_21);  primals_27 = permute_21 = None
        view_27 = torch.ops.aten.view.default(addmm_4, [128, 500, 128])
        permute_22 = torch.ops.aten.permute.default(view_27, [1, 0, 2]);  view_27 = None
        clone_9 = torch.ops.aten.clone.default(permute_22, memory_format = torch.contiguous_format);  permute_22 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(clone_9, [2], correction = 0, keepdim = True)
        getitem_10 = var_mean_5[0]
        getitem_11 = var_mean_5[1];  var_mean_5 = None
        add_17 = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        sub_7 = torch.ops.aten.sub.Tensor(clone_9, getitem_11);  clone_9 = None
        mul_25 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_5);  sub_7 = None
        mul_26 = torch.ops.aten.mul.Tensor(mul_25, primals_28);  mul_25 = None
        add_18 = torch.ops.aten.add.Tensor(mul_26, primals_29);  mul_26 = primals_29 = None
        inductor_lookup_seed_default_5 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_21 = torch.ops.prims.inductor_random.default([500, 128, 128], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        gt_5 = torch.ops.aten.gt.Scalar(inductor_random_default_21, 0.1);  inductor_random_default_21 = None
        mul_27 = torch.ops.aten.mul.Tensor(gt_5, add_18);  add_18 = None
        mul_28 = torch.ops.aten.mul.Tensor(mul_27, 1.1111111111111112);  mul_27 = None
        add_19 = torch.ops.aten.add.Tensor(mul_28, add_12);  mul_28 = add_12 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_12 = var_mean_6[0]
        getitem_13 = var_mean_6[1];  var_mean_6 = None
        add_20 = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        sub_8 = torch.ops.aten.sub.Tensor(add_19, getitem_13);  getitem_13 = None
        mul_29 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_6);  sub_8 = None
        mul_30 = torch.ops.aten.mul.Tensor(mul_29, primals_30)
        add_21 = torch.ops.aten.add.Tensor(mul_30, primals_31);  mul_30 = primals_31 = None
        view_29 = torch.ops.aten.view.default(add_21, [64000, 128]);  add_21 = None
        permute_23 = torch.ops.aten.permute.default(primals_32, [1, 0])
        addmm_5 = torch.ops.aten.addmm.default(primals_33, view_29, permute_23);  primals_33 = permute_23 = None
        view_30 = torch.ops.aten.view.default(addmm_5, [500, 128, 512])
        mul_31 = torch.ops.aten.mul.Tensor(view_30, 0.5)
        mul_32 = torch.ops.aten.mul.Tensor(view_30, 0.7071067811865476);  view_30 = None
        erf_1 = torch.ops.aten.erf.default(mul_32);  mul_32 = None
        add_22 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_33 = torch.ops.aten.mul.Tensor(mul_31, add_22);  mul_31 = add_22 = None
        inductor_lookup_seed_default_6 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_20 = torch.ops.prims.inductor_random.default([500, 128, 512], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        gt_6 = torch.ops.aten.gt.Scalar(inductor_random_default_20, 0.1);  inductor_random_default_20 = None
        mul_34 = torch.ops.aten.mul.Tensor(gt_6, mul_33);  mul_33 = None
        mul_35 = torch.ops.aten.mul.Tensor(mul_34, 1.1111111111111112);  mul_34 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(mul_35, [2], correction = 0, keepdim = True)
        getitem_14 = var_mean_7[0]
        getitem_15 = var_mean_7[1];  var_mean_7 = None
        add_23 = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        sub_9 = torch.ops.aten.sub.Tensor(mul_35, getitem_15);  mul_35 = None
        mul_36 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_7);  sub_9 = None
        mul_37 = torch.ops.aten.mul.Tensor(mul_36, primals_34);  mul_36 = None
        add_24 = torch.ops.aten.add.Tensor(mul_37, primals_35);  mul_37 = primals_35 = None
        view_31 = torch.ops.aten.view.default(add_24, [64000, 512]);  add_24 = None
        permute_24 = torch.ops.aten.permute.default(primals_36, [1, 0])
        addmm_6 = torch.ops.aten.addmm.default(primals_37, view_31, permute_24);  primals_37 = permute_24 = None
        view_32 = torch.ops.aten.view.default(addmm_6, [500, 128, 128]);  addmm_6 = None
        inductor_lookup_seed_default_7 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_19 = torch.ops.prims.inductor_random.default([500, 128, 128], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        gt_7 = torch.ops.aten.gt.Scalar(inductor_random_default_19, 0.1);  inductor_random_default_19 = None
        mul_38 = torch.ops.aten.mul.Tensor(gt_7, view_32);  view_32 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_38, 1.1111111111111112);  mul_38 = None
        add_25 = torch.ops.aten.add.Tensor(mul_39, add_19);  mul_39 = add_19 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_16 = var_mean_8[0]
        getitem_17 = var_mean_8[1];  var_mean_8 = None
        add_26 = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_10 = torch.ops.aten.sub.Tensor(add_25, getitem_17);  getitem_17 = None
        mul_40 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_8);  sub_10 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_40, primals_38)
        add_27 = torch.ops.aten.add.Tensor(mul_41, primals_39);  mul_41 = primals_39 = None
        permute_25 = torch.ops.aten.permute.default(add_27, [1, 0, 2]);  add_27 = None
        permute_26 = torch.ops.aten.permute.default(primals_40, [1, 0])
        clone_10 = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None
        view_33 = torch.ops.aten.view.default(clone_10, [64000, 128]);  clone_10 = None
        mm_2 = torch.ops.aten.mm.default(view_33, permute_26);  permute_26 = None
        view_34 = torch.ops.aten.view.default(mm_2, [128, 500, 384]);  mm_2 = None
        add_28 = torch.ops.aten.add.Tensor(view_34, primals_41);  view_34 = primals_41 = None
        view_35 = torch.ops.aten.view.default(add_28, [128, 500, 3, 128]);  add_28 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(view_35, 0);  view_35 = None
        permute_27 = torch.ops.aten.permute.default(unsqueeze_2, [3, 1, 2, 0, 4]);  unsqueeze_2 = None
        squeeze_2 = torch.ops.aten.squeeze.dim(permute_27, -2);  permute_27 = None
        clone_11 = torch.ops.aten.clone.default(squeeze_2, memory_format = torch.contiguous_format);  squeeze_2 = None
        select_6 = torch.ops.aten.select.int(clone_11, 0, 0)
        select_7 = torch.ops.aten.select.int(clone_11, 0, 1)
        select_8 = torch.ops.aten.select.int(clone_11, 0, 2);  clone_11 = None
        view_36 = torch.ops.aten.view.default(select_6, [128, 4000, 16]);  select_6 = None
        permute_28 = torch.ops.aten.permute.default(view_36, [1, 0, 2]);  view_36 = None
        view_37 = torch.ops.aten.view.default(select_7, [128, 4000, 16]);  select_7 = None
        permute_29 = torch.ops.aten.permute.default(view_37, [1, 0, 2]);  view_37 = None
        view_38 = torch.ops.aten.view.default(select_8, [128, 4000, 16]);  select_8 = None
        permute_30 = torch.ops.aten.permute.default(view_38, [1, 0, 2]);  view_38 = None
        mul_42 = torch.ops.aten.mul.Tensor(permute_28, 0.25);  permute_28 = None
        permute_31 = torch.ops.aten.permute.default(permute_29, [0, 2, 1]);  permute_29 = None
        baddbmm_2 = torch.ops.aten.baddbmm.default(add_3, mul_42, permute_31)
        amax_2 = torch.ops.aten.amax.default(baddbmm_2, [-1], True)
        sub_11 = torch.ops.aten.sub.Tensor(baddbmm_2, amax_2)
        exp_2 = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2 = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = None
        inductor_lookup_seed_default_8 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_18 = torch.ops.prims.inductor_random.default([4000, 128, 128], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        gt_8 = torch.ops.aten.gt.Scalar(inductor_random_default_18, 0.1);  inductor_random_default_18 = None
        mul_43 = torch.ops.aten.mul.Tensor(gt_8, div_2);  div_2 = None
        mul_44 = torch.ops.aten.mul.Tensor(mul_43, 1.1111111111111112);  mul_43 = None
        bmm_2 = torch.ops.aten.bmm.default(mul_44, permute_30)
        permute_32 = torch.ops.aten.permute.default(bmm_2, [1, 0, 2]);  bmm_2 = None
        clone_13 = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
        view_41 = torch.ops.aten.view.default(clone_13, [64000, 128]);  clone_13 = None
        permute_33 = torch.ops.aten.permute.default(primals_42, [1, 0])
        addmm_7 = torch.ops.aten.addmm.default(primals_43, view_41, permute_33);  primals_43 = permute_33 = None
        view_42 = torch.ops.aten.view.default(addmm_7, [128, 500, 128])
        permute_34 = torch.ops.aten.permute.default(view_42, [1, 0, 2]);  view_42 = None
        clone_14 = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(clone_14, [2], correction = 0, keepdim = True)
        getitem_18 = var_mean_9[0]
        getitem_19 = var_mean_9[1];  var_mean_9 = None
        add_30 = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        sub_12 = torch.ops.aten.sub.Tensor(clone_14, getitem_19);  clone_14 = None
        mul_45 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_9);  sub_12 = None
        mul_46 = torch.ops.aten.mul.Tensor(mul_45, primals_44);  mul_45 = None
        add_31 = torch.ops.aten.add.Tensor(mul_46, primals_45);  mul_46 = primals_45 = None
        inductor_lookup_seed_default_9 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_17 = torch.ops.prims.inductor_random.default([500, 128, 128], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        gt_9 = torch.ops.aten.gt.Scalar(inductor_random_default_17, 0.1);  inductor_random_default_17 = None
        mul_47 = torch.ops.aten.mul.Tensor(gt_9, add_31);  add_31 = None
        mul_48 = torch.ops.aten.mul.Tensor(mul_47, 1.1111111111111112);  mul_47 = None
        add_32 = torch.ops.aten.add.Tensor(mul_48, add_25);  mul_48 = add_25 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
        getitem_20 = var_mean_10[0]
        getitem_21 = var_mean_10[1];  var_mean_10 = None
        add_33 = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        sub_13 = torch.ops.aten.sub.Tensor(add_32, getitem_21);  getitem_21 = None
        mul_49 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_10);  sub_13 = None
        mul_50 = torch.ops.aten.mul.Tensor(mul_49, primals_46)
        add_34 = torch.ops.aten.add.Tensor(mul_50, primals_47);  mul_50 = primals_47 = None
        view_44 = torch.ops.aten.view.default(add_34, [64000, 128]);  add_34 = None
        permute_35 = torch.ops.aten.permute.default(primals_48, [1, 0])
        addmm_8 = torch.ops.aten.addmm.default(primals_49, view_44, permute_35);  primals_49 = permute_35 = None
        view_45 = torch.ops.aten.view.default(addmm_8, [500, 128, 512])
        mul_51 = torch.ops.aten.mul.Tensor(view_45, 0.5)
        mul_52 = torch.ops.aten.mul.Tensor(view_45, 0.7071067811865476);  view_45 = None
        erf_2 = torch.ops.aten.erf.default(mul_52);  mul_52 = None
        add_35 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_51, add_35);  mul_51 = add_35 = None
        inductor_lookup_seed_default_10 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_16 = torch.ops.prims.inductor_random.default([500, 128, 512], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        gt_10 = torch.ops.aten.gt.Scalar(inductor_random_default_16, 0.1);  inductor_random_default_16 = None
        mul_54 = torch.ops.aten.mul.Tensor(gt_10, mul_53);  mul_53 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul_54, 1.1111111111111112);  mul_54 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(mul_55, [2], correction = 0, keepdim = True)
        getitem_22 = var_mean_11[0]
        getitem_23 = var_mean_11[1];  var_mean_11 = None
        add_36 = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_14 = torch.ops.aten.sub.Tensor(mul_55, getitem_23);  mul_55 = None
        mul_56 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_11);  sub_14 = None
        mul_57 = torch.ops.aten.mul.Tensor(mul_56, primals_50);  mul_56 = None
        add_37 = torch.ops.aten.add.Tensor(mul_57, primals_51);  mul_57 = primals_51 = None
        view_46 = torch.ops.aten.view.default(add_37, [64000, 512]);  add_37 = None
        permute_36 = torch.ops.aten.permute.default(primals_52, [1, 0])
        addmm_9 = torch.ops.aten.addmm.default(primals_53, view_46, permute_36);  primals_53 = permute_36 = None
        view_47 = torch.ops.aten.view.default(addmm_9, [500, 128, 128]);  addmm_9 = None
        inductor_lookup_seed_default_11 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default_15 = torch.ops.prims.inductor_random.default([500, 128, 128], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        gt_11 = torch.ops.aten.gt.Scalar(inductor_random_default_15, 0.1);  inductor_random_default_15 = None
        mul_58 = torch.ops.aten.mul.Tensor(gt_11, view_47);  view_47 = None
        mul_59 = torch.ops.aten.mul.Tensor(mul_58, 1.1111111111111112);  mul_58 = None
        add_38 = torch.ops.aten.add.Tensor(mul_59, add_32);  mul_59 = add_32 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
        getitem_24 = var_mean_12[0]
        getitem_25 = var_mean_12[1];  var_mean_12 = None
        add_39 = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        sub_15 = torch.ops.aten.sub.Tensor(add_38, getitem_25);  getitem_25 = None
        mul_60 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_12);  sub_15 = None
        mul_61 = torch.ops.aten.mul.Tensor(mul_60, primals_54)
        add_40 = torch.ops.aten.add.Tensor(mul_61, primals_55);  mul_61 = primals_55 = None
        permute_37 = torch.ops.aten.permute.default(add_40, [1, 0, 2]);  add_40 = None
        permute_38 = torch.ops.aten.permute.default(primals_56, [1, 0])
        clone_15 = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
        view_48 = torch.ops.aten.view.default(clone_15, [64000, 128]);  clone_15 = None
        mm_3 = torch.ops.aten.mm.default(view_48, permute_38);  permute_38 = None
        view_49 = torch.ops.aten.view.default(mm_3, [128, 500, 384]);  mm_3 = None
        add_41 = torch.ops.aten.add.Tensor(view_49, primals_57);  view_49 = primals_57 = None
        view_50 = torch.ops.aten.view.default(add_41, [128, 500, 3, 128]);  add_41 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(view_50, 0);  view_50 = None
        permute_39 = torch.ops.aten.permute.default(unsqueeze_3, [3, 1, 2, 0, 4]);  unsqueeze_3 = None
        squeeze_3 = torch.ops.aten.squeeze.dim(permute_39, -2);  permute_39 = None
        clone_16 = torch.ops.aten.clone.default(squeeze_3, memory_format = torch.contiguous_format);  squeeze_3 = None
        select_9 = torch.ops.aten.select.int(clone_16, 0, 0)
        select_10 = torch.ops.aten.select.int(clone_16, 0, 1)
        select_11 = torch.ops.aten.select.int(clone_16, 0, 2);  clone_16 = None
        view_51 = torch.ops.aten.view.default(select_9, [128, 4000, 16]);  select_9 = None
        permute_40 = torch.ops.aten.permute.default(view_51, [1, 0, 2]);  view_51 = None
        view_52 = torch.ops.aten.view.default(select_10, [128, 4000, 16]);  select_10 = None
        permute_41 = torch.ops.aten.permute.default(view_52, [1, 0, 2]);  view_52 = None
        view_53 = torch.ops.aten.view.default(select_11, [128, 4000, 16]);  select_11 = None
        permute_42 = torch.ops.aten.permute.default(view_53, [1, 0, 2]);  view_53 = None
        mul_62 = torch.ops.aten.mul.Tensor(permute_40, 0.25);  permute_40 = None
        permute_43 = torch.ops.aten.permute.default(permute_41, [0, 2, 1]);  permute_41 = None
        baddbmm_3 = torch.ops.aten.baddbmm.default(add_3, mul_62, permute_43)
        amax_3 = torch.ops.aten.amax.default(baddbmm_3, [-1], True)
        sub_16 = torch.ops.aten.sub.Tensor(baddbmm_3, amax_3)
        exp_3 = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3 = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = None
        inductor_lookup_seed_default_12 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 12)
        inductor_random_default_14 = torch.ops.prims.inductor_random.default([4000, 128, 128], inductor_lookup_seed_default_12, 'rand');  inductor_lookup_seed_default_12 = None
        gt_12 = torch.ops.aten.gt.Scalar(inductor_random_default_14, 0.1);  inductor_random_default_14 = None
        mul_63 = torch.ops.aten.mul.Tensor(gt_12, div_3);  div_3 = None
        mul_64 = torch.ops.aten.mul.Tensor(mul_63, 1.1111111111111112);  mul_63 = None
        bmm_3 = torch.ops.aten.bmm.default(mul_64, permute_42)
        permute_44 = torch.ops.aten.permute.default(bmm_3, [1, 0, 2]);  bmm_3 = None
        clone_18 = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
        view_56 = torch.ops.aten.view.default(clone_18, [64000, 128]);  clone_18 = None
        permute_45 = torch.ops.aten.permute.default(primals_58, [1, 0])
        addmm_10 = torch.ops.aten.addmm.default(primals_59, view_56, permute_45);  primals_59 = permute_45 = None
        view_57 = torch.ops.aten.view.default(addmm_10, [128, 500, 128])
        permute_46 = torch.ops.aten.permute.default(view_57, [1, 0, 2]);  view_57 = None
        clone_19 = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(clone_19, [2], correction = 0, keepdim = True)
        getitem_26 = var_mean_13[0]
        getitem_27 = var_mean_13[1];  var_mean_13 = None
        add_43 = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_17 = torch.ops.aten.sub.Tensor(clone_19, getitem_27);  clone_19 = None
        mul_65 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_13);  sub_17 = None
        mul_66 = torch.ops.aten.mul.Tensor(mul_65, primals_60);  mul_65 = None
        add_44 = torch.ops.aten.add.Tensor(mul_66, primals_61);  mul_66 = primals_61 = None
        inductor_lookup_seed_default_13 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 13)
        inductor_random_default_13 = torch.ops.prims.inductor_random.default([500, 128, 128], inductor_lookup_seed_default_13, 'rand');  inductor_lookup_seed_default_13 = None
        gt_13 = torch.ops.aten.gt.Scalar(inductor_random_default_13, 0.1);  inductor_random_default_13 = None
        mul_67 = torch.ops.aten.mul.Tensor(gt_13, add_44);  add_44 = None
        mul_68 = torch.ops.aten.mul.Tensor(mul_67, 1.1111111111111112);  mul_67 = None
        add_45 = torch.ops.aten.add.Tensor(mul_68, add_38);  mul_68 = add_38 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
        getitem_28 = var_mean_14[0]
        getitem_29 = var_mean_14[1];  var_mean_14 = None
        add_46 = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_18 = torch.ops.aten.sub.Tensor(add_45, getitem_29);  getitem_29 = None
        mul_69 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_14);  sub_18 = None
        mul_70 = torch.ops.aten.mul.Tensor(mul_69, primals_62)
        add_47 = torch.ops.aten.add.Tensor(mul_70, primals_63);  mul_70 = primals_63 = None
        view_59 = torch.ops.aten.view.default(add_47, [64000, 128]);  add_47 = None
        permute_47 = torch.ops.aten.permute.default(primals_64, [1, 0])
        addmm_11 = torch.ops.aten.addmm.default(primals_65, view_59, permute_47);  primals_65 = permute_47 = None
        view_60 = torch.ops.aten.view.default(addmm_11, [500, 128, 512])
        mul_71 = torch.ops.aten.mul.Tensor(view_60, 0.5)
        mul_72 = torch.ops.aten.mul.Tensor(view_60, 0.7071067811865476);  view_60 = None
        erf_3 = torch.ops.aten.erf.default(mul_72);  mul_72 = None
        add_48 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_73 = torch.ops.aten.mul.Tensor(mul_71, add_48);  mul_71 = add_48 = None
        inductor_lookup_seed_default_14 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 14)
        inductor_random_default_12 = torch.ops.prims.inductor_random.default([500, 128, 512], inductor_lookup_seed_default_14, 'rand');  inductor_lookup_seed_default_14 = None
        gt_14 = torch.ops.aten.gt.Scalar(inductor_random_default_12, 0.1);  inductor_random_default_12 = None
        mul_74 = torch.ops.aten.mul.Tensor(gt_14, mul_73);  mul_73 = None
        mul_75 = torch.ops.aten.mul.Tensor(mul_74, 1.1111111111111112);  mul_74 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(mul_75, [2], correction = 0, keepdim = True)
        getitem_30 = var_mean_15[0]
        getitem_31 = var_mean_15[1];  var_mean_15 = None
        add_49 = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        sub_19 = torch.ops.aten.sub.Tensor(mul_75, getitem_31);  mul_75 = None
        mul_76 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_15);  sub_19 = None
        mul_77 = torch.ops.aten.mul.Tensor(mul_76, primals_66);  mul_76 = None
        add_50 = torch.ops.aten.add.Tensor(mul_77, primals_67);  mul_77 = primals_67 = None
        view_61 = torch.ops.aten.view.default(add_50, [64000, 512]);  add_50 = None
        permute_48 = torch.ops.aten.permute.default(primals_68, [1, 0])
        addmm_12 = torch.ops.aten.addmm.default(primals_69, view_61, permute_48);  primals_69 = permute_48 = None
        view_62 = torch.ops.aten.view.default(addmm_12, [500, 128, 128]);  addmm_12 = None
        inductor_lookup_seed_default_15 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 15)
        inductor_random_default_11 = torch.ops.prims.inductor_random.default([500, 128, 128], inductor_lookup_seed_default_15, 'rand');  inductor_lookup_seed_default_15 = None
        gt_15 = torch.ops.aten.gt.Scalar(inductor_random_default_11, 0.1);  inductor_random_default_11 = None
        mul_78 = torch.ops.aten.mul.Tensor(gt_15, view_62);  view_62 = None
        mul_79 = torch.ops.aten.mul.Tensor(mul_78, 1.1111111111111112);  mul_78 = None
        add_51 = torch.ops.aten.add.Tensor(mul_79, add_45);  mul_79 = add_45 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
        getitem_32 = var_mean_16[0]
        getitem_33 = var_mean_16[1];  var_mean_16 = None
        add_52 = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        sub_20 = torch.ops.aten.sub.Tensor(add_51, getitem_33);  getitem_33 = None
        mul_80 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_16);  sub_20 = None
        mul_81 = torch.ops.aten.mul.Tensor(mul_80, primals_70)
        add_53 = torch.ops.aten.add.Tensor(mul_81, primals_71);  mul_81 = primals_71 = None
        permute_49 = torch.ops.aten.permute.default(add_53, [1, 0, 2]);  add_53 = None
        permute_50 = torch.ops.aten.permute.default(primals_72, [1, 0])
        clone_20 = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        view_63 = torch.ops.aten.view.default(clone_20, [64000, 128]);  clone_20 = None
        mm_4 = torch.ops.aten.mm.default(view_63, permute_50);  permute_50 = None
        view_64 = torch.ops.aten.view.default(mm_4, [128, 500, 384]);  mm_4 = None
        add_54 = torch.ops.aten.add.Tensor(view_64, primals_73);  view_64 = primals_73 = None
        view_65 = torch.ops.aten.view.default(add_54, [128, 500, 3, 128]);  add_54 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(view_65, 0);  view_65 = None
        permute_51 = torch.ops.aten.permute.default(unsqueeze_4, [3, 1, 2, 0, 4]);  unsqueeze_4 = None
        squeeze_4 = torch.ops.aten.squeeze.dim(permute_51, -2);  permute_51 = None
        clone_21 = torch.ops.aten.clone.default(squeeze_4, memory_format = torch.contiguous_format);  squeeze_4 = None
        select_12 = torch.ops.aten.select.int(clone_21, 0, 0)
        select_13 = torch.ops.aten.select.int(clone_21, 0, 1)
        select_14 = torch.ops.aten.select.int(clone_21, 0, 2);  clone_21 = None
        view_66 = torch.ops.aten.view.default(select_12, [128, 4000, 16]);  select_12 = None
        permute_52 = torch.ops.aten.permute.default(view_66, [1, 0, 2]);  view_66 = None
        view_67 = torch.ops.aten.view.default(select_13, [128, 4000, 16]);  select_13 = None
        permute_53 = torch.ops.aten.permute.default(view_67, [1, 0, 2]);  view_67 = None
        view_68 = torch.ops.aten.view.default(select_14, [128, 4000, 16]);  select_14 = None
        permute_54 = torch.ops.aten.permute.default(view_68, [1, 0, 2]);  view_68 = None
        mul_82 = torch.ops.aten.mul.Tensor(permute_52, 0.25);  permute_52 = None
        permute_55 = torch.ops.aten.permute.default(permute_53, [0, 2, 1]);  permute_53 = None
        baddbmm_4 = torch.ops.aten.baddbmm.default(add_3, mul_82, permute_55)
        amax_4 = torch.ops.aten.amax.default(baddbmm_4, [-1], True)
        sub_21 = torch.ops.aten.sub.Tensor(baddbmm_4, amax_4)
        exp_4 = torch.ops.aten.exp.default(sub_21);  sub_21 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4 = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = None
        inductor_lookup_seed_default_16 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 16)
        inductor_random_default_10 = torch.ops.prims.inductor_random.default([4000, 128, 128], inductor_lookup_seed_default_16, 'rand');  inductor_lookup_seed_default_16 = None
        gt_16 = torch.ops.aten.gt.Scalar(inductor_random_default_10, 0.1);  inductor_random_default_10 = None
        mul_83 = torch.ops.aten.mul.Tensor(gt_16, div_4);  div_4 = None
        mul_84 = torch.ops.aten.mul.Tensor(mul_83, 1.1111111111111112);  mul_83 = None
        bmm_4 = torch.ops.aten.bmm.default(mul_84, permute_54)
        permute_56 = torch.ops.aten.permute.default(bmm_4, [1, 0, 2]);  bmm_4 = None
        clone_23 = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
        view_71 = torch.ops.aten.view.default(clone_23, [64000, 128]);  clone_23 = None
        permute_57 = torch.ops.aten.permute.default(primals_74, [1, 0])
        addmm_13 = torch.ops.aten.addmm.default(primals_75, view_71, permute_57);  primals_75 = permute_57 = None
        view_72 = torch.ops.aten.view.default(addmm_13, [128, 500, 128])
        permute_58 = torch.ops.aten.permute.default(view_72, [1, 0, 2]);  view_72 = None
        clone_24 = torch.ops.aten.clone.default(permute_58, memory_format = torch.contiguous_format);  permute_58 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(clone_24, [2], correction = 0, keepdim = True)
        getitem_34 = var_mean_17[0]
        getitem_35 = var_mean_17[1];  var_mean_17 = None
        add_56 = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        sub_22 = torch.ops.aten.sub.Tensor(clone_24, getitem_35);  clone_24 = None
        mul_85 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_17);  sub_22 = None
        mul_86 = torch.ops.aten.mul.Tensor(mul_85, primals_76);  mul_85 = None
        add_57 = torch.ops.aten.add.Tensor(mul_86, primals_77);  mul_86 = primals_77 = None
        inductor_lookup_seed_default_17 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 17)
        inductor_random_default_9 = torch.ops.prims.inductor_random.default([500, 128, 128], inductor_lookup_seed_default_17, 'rand');  inductor_lookup_seed_default_17 = None
        gt_17 = torch.ops.aten.gt.Scalar(inductor_random_default_9, 0.1);  inductor_random_default_9 = None
        mul_87 = torch.ops.aten.mul.Tensor(gt_17, add_57);  add_57 = None
        mul_88 = torch.ops.aten.mul.Tensor(mul_87, 1.1111111111111112);  mul_87 = None
        add_58 = torch.ops.aten.add.Tensor(mul_88, add_51);  mul_88 = add_51 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_58, [2], correction = 0, keepdim = True)
        getitem_36 = var_mean_18[0]
        getitem_37 = var_mean_18[1];  var_mean_18 = None
        add_59 = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
        sub_23 = torch.ops.aten.sub.Tensor(add_58, getitem_37);  getitem_37 = None
        mul_89 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_18);  sub_23 = None
        mul_90 = torch.ops.aten.mul.Tensor(mul_89, primals_78)
        add_60 = torch.ops.aten.add.Tensor(mul_90, primals_79);  mul_90 = primals_79 = None
        view_74 = torch.ops.aten.view.default(add_60, [64000, 128]);  add_60 = None
        permute_59 = torch.ops.aten.permute.default(primals_80, [1, 0])
        addmm_14 = torch.ops.aten.addmm.default(primals_81, view_74, permute_59);  primals_81 = permute_59 = None
        view_75 = torch.ops.aten.view.default(addmm_14, [500, 128, 512])
        mul_91 = torch.ops.aten.mul.Tensor(view_75, 0.5)
        mul_92 = torch.ops.aten.mul.Tensor(view_75, 0.7071067811865476);  view_75 = None
        erf_4 = torch.ops.aten.erf.default(mul_92);  mul_92 = None
        add_61 = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_93 = torch.ops.aten.mul.Tensor(mul_91, add_61);  mul_91 = add_61 = None
        inductor_lookup_seed_default_18 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 18)
        inductor_random_default_8 = torch.ops.prims.inductor_random.default([500, 128, 512], inductor_lookup_seed_default_18, 'rand');  inductor_lookup_seed_default_18 = None
        gt_18 = torch.ops.aten.gt.Scalar(inductor_random_default_8, 0.1);  inductor_random_default_8 = None
        mul_94 = torch.ops.aten.mul.Tensor(gt_18, mul_93);  mul_93 = None
        mul_95 = torch.ops.aten.mul.Tensor(mul_94, 1.1111111111111112);  mul_94 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(mul_95, [2], correction = 0, keepdim = True)
        getitem_38 = var_mean_19[0]
        getitem_39 = var_mean_19[1];  var_mean_19 = None
        add_62 = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        sub_24 = torch.ops.aten.sub.Tensor(mul_95, getitem_39);  mul_95 = None
        mul_96 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_19);  sub_24 = None
        mul_97 = torch.ops.aten.mul.Tensor(mul_96, primals_82);  mul_96 = None
        add_63 = torch.ops.aten.add.Tensor(mul_97, primals_83);  mul_97 = primals_83 = None
        view_76 = torch.ops.aten.view.default(add_63, [64000, 512]);  add_63 = None
        permute_60 = torch.ops.aten.permute.default(primals_84, [1, 0])
        addmm_15 = torch.ops.aten.addmm.default(primals_85, view_76, permute_60);  primals_85 = permute_60 = None
        view_77 = torch.ops.aten.view.default(addmm_15, [500, 128, 128]);  addmm_15 = None
        inductor_lookup_seed_default_19 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 19)
        inductor_random_default_7 = torch.ops.prims.inductor_random.default([500, 128, 128], inductor_lookup_seed_default_19, 'rand');  inductor_lookup_seed_default_19 = None
        gt_19 = torch.ops.aten.gt.Scalar(inductor_random_default_7, 0.1);  inductor_random_default_7 = None
        mul_98 = torch.ops.aten.mul.Tensor(gt_19, view_77);  view_77 = None
        mul_99 = torch.ops.aten.mul.Tensor(mul_98, 1.1111111111111112);  mul_98 = None
        add_64 = torch.ops.aten.add.Tensor(mul_99, add_58);  mul_99 = add_58 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
        getitem_40 = var_mean_20[0]
        getitem_41 = var_mean_20[1];  var_mean_20 = None
        add_65 = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        sub_25 = torch.ops.aten.sub.Tensor(add_64, getitem_41);  getitem_41 = None
        mul_100 = torch.ops.aten.mul.Tensor(sub_25, rsqrt_20);  sub_25 = None
        mul_101 = torch.ops.aten.mul.Tensor(mul_100, primals_86)
        add_66 = torch.ops.aten.add.Tensor(mul_101, primals_87);  mul_101 = primals_87 = None
        permute_61 = torch.ops.aten.permute.default(add_66, [1, 0, 2]);  add_66 = None
        permute_62 = torch.ops.aten.permute.default(primals_88, [1, 0])
        clone_25 = torch.ops.aten.clone.default(permute_61, memory_format = torch.contiguous_format);  permute_61 = None
        view_78 = torch.ops.aten.view.default(clone_25, [64000, 128]);  clone_25 = None
        mm_5 = torch.ops.aten.mm.default(view_78, permute_62);  permute_62 = None
        view_79 = torch.ops.aten.view.default(mm_5, [128, 500, 384]);  mm_5 = None
        add_67 = torch.ops.aten.add.Tensor(view_79, primals_89);  view_79 = primals_89 = None
        view_80 = torch.ops.aten.view.default(add_67, [128, 500, 3, 128]);  add_67 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(view_80, 0);  view_80 = None
        permute_63 = torch.ops.aten.permute.default(unsqueeze_5, [3, 1, 2, 0, 4]);  unsqueeze_5 = None
        squeeze_5 = torch.ops.aten.squeeze.dim(permute_63, -2);  permute_63 = None
        clone_26 = torch.ops.aten.clone.default(squeeze_5, memory_format = torch.contiguous_format);  squeeze_5 = None
        select_15 = torch.ops.aten.select.int(clone_26, 0, 0)
        select_16 = torch.ops.aten.select.int(clone_26, 0, 1)
        select_17 = torch.ops.aten.select.int(clone_26, 0, 2);  clone_26 = None
        view_81 = torch.ops.aten.view.default(select_15, [128, 4000, 16]);  select_15 = None
        permute_64 = torch.ops.aten.permute.default(view_81, [1, 0, 2]);  view_81 = None
        view_82 = torch.ops.aten.view.default(select_16, [128, 4000, 16]);  select_16 = None
        permute_65 = torch.ops.aten.permute.default(view_82, [1, 0, 2]);  view_82 = None
        view_83 = torch.ops.aten.view.default(select_17, [128, 4000, 16]);  select_17 = None
        permute_66 = torch.ops.aten.permute.default(view_83, [1, 0, 2]);  view_83 = None
        mul_102 = torch.ops.aten.mul.Tensor(permute_64, 0.25);  permute_64 = None
        permute_67 = torch.ops.aten.permute.default(permute_65, [0, 2, 1]);  permute_65 = None
        baddbmm_5 = torch.ops.aten.baddbmm.default(add_3, mul_102, permute_67)
        amax_5 = torch.ops.aten.amax.default(baddbmm_5, [-1], True)
        sub_26 = torch.ops.aten.sub.Tensor(baddbmm_5, amax_5)
        exp_5 = torch.ops.aten.exp.default(sub_26);  sub_26 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5 = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = None
        inductor_lookup_seed_default_20 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 20)
        inductor_random_default_6 = torch.ops.prims.inductor_random.default([4000, 128, 128], inductor_lookup_seed_default_20, 'rand');  inductor_lookup_seed_default_20 = None
        gt_20 = torch.ops.aten.gt.Scalar(inductor_random_default_6, 0.1);  inductor_random_default_6 = None
        mul_103 = torch.ops.aten.mul.Tensor(gt_20, div_5);  div_5 = None
        mul_104 = torch.ops.aten.mul.Tensor(mul_103, 1.1111111111111112);  mul_103 = None
        bmm_5 = torch.ops.aten.bmm.default(mul_104, permute_66)
        permute_68 = torch.ops.aten.permute.default(bmm_5, [1, 0, 2]);  bmm_5 = None
        clone_28 = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
        view_86 = torch.ops.aten.view.default(clone_28, [64000, 128]);  clone_28 = None
        permute_69 = torch.ops.aten.permute.default(primals_90, [1, 0])
        addmm_16 = torch.ops.aten.addmm.default(primals_91, view_86, permute_69);  primals_91 = permute_69 = None
        view_87 = torch.ops.aten.view.default(addmm_16, [128, 500, 128])
        permute_70 = torch.ops.aten.permute.default(view_87, [1, 0, 2]);  view_87 = None
        clone_29 = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(clone_29, [2], correction = 0, keepdim = True)
        getitem_42 = var_mean_21[0]
        getitem_43 = var_mean_21[1];  var_mean_21 = None
        add_69 = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_27 = torch.ops.aten.sub.Tensor(clone_29, getitem_43);  clone_29 = None
        mul_105 = torch.ops.aten.mul.Tensor(sub_27, rsqrt_21);  sub_27 = None
        mul_106 = torch.ops.aten.mul.Tensor(mul_105, primals_92);  mul_105 = None
        add_70 = torch.ops.aten.add.Tensor(mul_106, primals_93);  mul_106 = primals_93 = None
        inductor_lookup_seed_default_21 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 21)
        inductor_random_default_5 = torch.ops.prims.inductor_random.default([500, 128, 128], inductor_lookup_seed_default_21, 'rand');  inductor_lookup_seed_default_21 = None
        gt_21 = torch.ops.aten.gt.Scalar(inductor_random_default_5, 0.1);  inductor_random_default_5 = None
        mul_107 = torch.ops.aten.mul.Tensor(gt_21, add_70);  add_70 = None
        mul_108 = torch.ops.aten.mul.Tensor(mul_107, 1.1111111111111112);  mul_107 = None
        add_71 = torch.ops.aten.add.Tensor(mul_108, add_64);  mul_108 = add_64 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
        getitem_44 = var_mean_22[0]
        getitem_45 = var_mean_22[1];  var_mean_22 = None
        add_72 = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        sub_28 = torch.ops.aten.sub.Tensor(add_71, getitem_45);  getitem_45 = None
        mul_109 = torch.ops.aten.mul.Tensor(sub_28, rsqrt_22);  sub_28 = None
        mul_110 = torch.ops.aten.mul.Tensor(mul_109, primals_94)
        add_73 = torch.ops.aten.add.Tensor(mul_110, primals_95);  mul_110 = primals_95 = None
        view_89 = torch.ops.aten.view.default(add_73, [64000, 128]);  add_73 = None
        permute_71 = torch.ops.aten.permute.default(primals_96, [1, 0])
        addmm_17 = torch.ops.aten.addmm.default(primals_97, view_89, permute_71);  primals_97 = permute_71 = None
        view_90 = torch.ops.aten.view.default(addmm_17, [500, 128, 512])
        mul_111 = torch.ops.aten.mul.Tensor(view_90, 0.5)
        mul_112 = torch.ops.aten.mul.Tensor(view_90, 0.7071067811865476);  view_90 = None
        erf_5 = torch.ops.aten.erf.default(mul_112);  mul_112 = None
        add_74 = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_113 = torch.ops.aten.mul.Tensor(mul_111, add_74);  mul_111 = add_74 = None
        inductor_lookup_seed_default_22 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 22)
        inductor_random_default_4 = torch.ops.prims.inductor_random.default([500, 128, 512], inductor_lookup_seed_default_22, 'rand');  inductor_lookup_seed_default_22 = None
        gt_22 = torch.ops.aten.gt.Scalar(inductor_random_default_4, 0.1);  inductor_random_default_4 = None
        mul_114 = torch.ops.aten.mul.Tensor(gt_22, mul_113);  mul_113 = None
        mul_115 = torch.ops.aten.mul.Tensor(mul_114, 1.1111111111111112);  mul_114 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(mul_115, [2], correction = 0, keepdim = True)
        getitem_46 = var_mean_23[0]
        getitem_47 = var_mean_23[1];  var_mean_23 = None
        add_75 = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        sub_29 = torch.ops.aten.sub.Tensor(mul_115, getitem_47);  mul_115 = None
        mul_116 = torch.ops.aten.mul.Tensor(sub_29, rsqrt_23);  sub_29 = None
        mul_117 = torch.ops.aten.mul.Tensor(mul_116, primals_98);  mul_116 = None
        add_76 = torch.ops.aten.add.Tensor(mul_117, primals_99);  mul_117 = primals_99 = None
        view_91 = torch.ops.aten.view.default(add_76, [64000, 512]);  add_76 = None
        permute_72 = torch.ops.aten.permute.default(primals_100, [1, 0])
        addmm_18 = torch.ops.aten.addmm.default(primals_101, view_91, permute_72);  primals_101 = permute_72 = None
        view_92 = torch.ops.aten.view.default(addmm_18, [500, 128, 128]);  addmm_18 = None
        inductor_lookup_seed_default_23 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 23)
        inductor_random_default_3 = torch.ops.prims.inductor_random.default([500, 128, 128], inductor_lookup_seed_default_23, 'rand');  inductor_lookup_seed_default_23 = None
        gt_23 = torch.ops.aten.gt.Scalar(inductor_random_default_3, 0.1);  inductor_random_default_3 = None
        mul_118 = torch.ops.aten.mul.Tensor(gt_23, view_92);  view_92 = None
        mul_119 = torch.ops.aten.mul.Tensor(mul_118, 1.1111111111111112);  mul_118 = None
        add_77 = torch.ops.aten.add.Tensor(mul_119, add_71);  mul_119 = add_71 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
        getitem_48 = var_mean_24[0]
        getitem_49 = var_mean_24[1];  var_mean_24 = None
        add_78 = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        sub_30 = torch.ops.aten.sub.Tensor(add_77, getitem_49);  getitem_49 = None
        mul_120 = torch.ops.aten.mul.Tensor(sub_30, rsqrt_24);  sub_30 = None
        mul_121 = torch.ops.aten.mul.Tensor(mul_120, primals_102)
        add_79 = torch.ops.aten.add.Tensor(mul_121, primals_103);  mul_121 = primals_103 = None
        permute_73 = torch.ops.aten.permute.default(add_79, [1, 0, 2]);  add_79 = None
        permute_74 = torch.ops.aten.permute.default(primals_104, [1, 0])
        clone_30 = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_93 = torch.ops.aten.view.default(clone_30, [64000, 128]);  clone_30 = None
        mm_6 = torch.ops.aten.mm.default(view_93, permute_74);  permute_74 = None
        view_94 = torch.ops.aten.view.default(mm_6, [128, 500, 384]);  mm_6 = None
        add_80 = torch.ops.aten.add.Tensor(view_94, primals_105);  view_94 = primals_105 = None
        view_95 = torch.ops.aten.view.default(add_80, [128, 500, 3, 128]);  add_80 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(view_95, 0);  view_95 = None
        permute_75 = torch.ops.aten.permute.default(unsqueeze_6, [3, 1, 2, 0, 4]);  unsqueeze_6 = None
        squeeze_6 = torch.ops.aten.squeeze.dim(permute_75, -2);  permute_75 = None
        clone_31 = torch.ops.aten.clone.default(squeeze_6, memory_format = torch.contiguous_format);  squeeze_6 = None
        select_18 = torch.ops.aten.select.int(clone_31, 0, 0)
        select_19 = torch.ops.aten.select.int(clone_31, 0, 1)
        select_20 = torch.ops.aten.select.int(clone_31, 0, 2);  clone_31 = None
        view_96 = torch.ops.aten.view.default(select_18, [128, 4000, 16]);  select_18 = None
        permute_76 = torch.ops.aten.permute.default(view_96, [1, 0, 2]);  view_96 = None
        view_97 = torch.ops.aten.view.default(select_19, [128, 4000, 16]);  select_19 = None
        permute_77 = torch.ops.aten.permute.default(view_97, [1, 0, 2]);  view_97 = None
        view_98 = torch.ops.aten.view.default(select_20, [128, 4000, 16]);  select_20 = None
        permute_78 = torch.ops.aten.permute.default(view_98, [1, 0, 2]);  view_98 = None
        mul_122 = torch.ops.aten.mul.Tensor(permute_76, 0.25);  permute_76 = None
        permute_79 = torch.ops.aten.permute.default(permute_77, [0, 2, 1]);  permute_77 = None
        baddbmm_6 = torch.ops.aten.baddbmm.default(add_3, mul_122, permute_79);  add_3 = None
        amax_6 = torch.ops.aten.amax.default(baddbmm_6, [-1], True)
        sub_31 = torch.ops.aten.sub.Tensor(baddbmm_6, amax_6)
        exp_6 = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6 = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = None
        inductor_lookup_seed_default_24 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 24)
        inductor_random_default_2 = torch.ops.prims.inductor_random.default([4000, 128, 128], inductor_lookup_seed_default_24, 'rand');  inductor_lookup_seed_default_24 = None
        gt_24 = torch.ops.aten.gt.Scalar(inductor_random_default_2, 0.1);  inductor_random_default_2 = None
        mul_123 = torch.ops.aten.mul.Tensor(gt_24, div_6);  div_6 = None
        mul_124 = torch.ops.aten.mul.Tensor(mul_123, 1.1111111111111112);  mul_123 = None
        bmm_6 = torch.ops.aten.bmm.default(mul_124, permute_78)
        permute_80 = torch.ops.aten.permute.default(bmm_6, [1, 0, 2]);  bmm_6 = None
        clone_33 = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
        view_101 = torch.ops.aten.view.default(clone_33, [64000, 128]);  clone_33 = None
        permute_81 = torch.ops.aten.permute.default(primals_106, [1, 0])
        addmm_19 = torch.ops.aten.addmm.default(primals_107, view_101, permute_81);  primals_107 = permute_81 = None
        view_102 = torch.ops.aten.view.default(addmm_19, [128, 500, 128])
        permute_82 = torch.ops.aten.permute.default(view_102, [1, 0, 2]);  view_102 = None
        clone_34 = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(clone_34, [2], correction = 0, keepdim = True)
        getitem_50 = var_mean_25[0]
        getitem_51 = var_mean_25[1];  var_mean_25 = None
        add_82 = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_32 = torch.ops.aten.sub.Tensor(clone_34, getitem_51);  clone_34 = None
        mul_125 = torch.ops.aten.mul.Tensor(sub_32, rsqrt_25);  sub_32 = None
        mul_126 = torch.ops.aten.mul.Tensor(mul_125, primals_108);  mul_125 = None
        add_83 = torch.ops.aten.add.Tensor(mul_126, primals_109);  mul_126 = primals_109 = None
        inductor_lookup_seed_default_25 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 25)
        inductor_random_default_1 = torch.ops.prims.inductor_random.default([500, 128, 128], inductor_lookup_seed_default_25, 'rand');  inductor_lookup_seed_default_25 = None
        gt_25 = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.1);  inductor_random_default_1 = None
        mul_127 = torch.ops.aten.mul.Tensor(gt_25, add_83);  add_83 = None
        mul_128 = torch.ops.aten.mul.Tensor(mul_127, 1.1111111111111112);  mul_127 = None
        add_84 = torch.ops.aten.add.Tensor(mul_128, add_77);  mul_128 = add_77 = None
        var_mean_26 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
        getitem_52 = var_mean_26[0]
        getitem_53 = var_mean_26[1];  var_mean_26 = None
        add_85 = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_26 = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        sub_33 = torch.ops.aten.sub.Tensor(add_84, getitem_53)
        mul_129 = torch.ops.aten.mul.Tensor(sub_33, rsqrt_26);  sub_33 = None
        mul_130 = torch.ops.aten.mul.Tensor(mul_129, primals_110);  mul_129 = None
        add_86 = torch.ops.aten.add.Tensor(mul_130, primals_111);  mul_130 = primals_111 = None
        view_104 = torch.ops.aten.view.default(add_86, [64000, 128]);  add_86 = None
        permute_83 = torch.ops.aten.permute.default(primals_112, [1, 0])
        addmm_20 = torch.ops.aten.addmm.default(primals_113, view_104, permute_83);  primals_113 = permute_83 = None
        view_105 = torch.ops.aten.view.default(addmm_20, [500, 128, 512])
        mul_131 = torch.ops.aten.mul.Tensor(view_105, 0.5)
        mul_132 = torch.ops.aten.mul.Tensor(view_105, 0.7071067811865476);  view_105 = None
        erf_6 = torch.ops.aten.erf.default(mul_132);  mul_132 = None
        add_87 = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_133 = torch.ops.aten.mul.Tensor(mul_131, add_87);  mul_131 = add_87 = None
        inductor_lookup_seed_default_26 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 26);  inductor_seeds_default = None
        inductor_random_default = torch.ops.prims.inductor_random.default([500, 128, 512], inductor_lookup_seed_default_26, 'rand');  inductor_lookup_seed_default_26 = None
        gt_26 = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_134 = torch.ops.aten.mul.Tensor(gt_26, mul_133);  mul_133 = None
        mul_135 = torch.ops.aten.mul.Tensor(mul_134, 1.1111111111111112);  mul_134 = None
        var_mean_27 = torch.ops.aten.var_mean.correction(mul_135, [2], correction = 0, keepdim = True)
        getitem_54 = var_mean_27[0]
        getitem_55 = var_mean_27[1];  var_mean_27 = None
        add_88 = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
        rsqrt_27 = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        sub_34 = torch.ops.aten.sub.Tensor(mul_135, getitem_55);  mul_135 = None
        mul_136 = torch.ops.aten.mul.Tensor(sub_34, rsqrt_27);  sub_34 = None
        mul_137 = torch.ops.aten.mul.Tensor(mul_136, primals_114);  mul_136 = None
        add_89 = torch.ops.aten.add.Tensor(mul_137, primals_115);  mul_137 = primals_115 = None
        permute_94 = torch.ops.aten.permute.default(mul_124, [0, 2, 1]);  mul_124 = None
        permute_95 = torch.ops.aten.permute.default(permute_78, [0, 2, 1]);  permute_78 = None
        permute_96 = torch.ops.aten.permute.default(permute_79, [0, 2, 1]);  permute_79 = None
        permute_97 = torch.ops.aten.permute.default(mul_122, [0, 2, 1]);  mul_122 = None
        div_10 = torch.ops.aten.div.Tensor(rsqrt_24, 128);  rsqrt_24 = None
        div_12 = torch.ops.aten.div.Tensor(rsqrt_22, 128);  rsqrt_22 = None
        permute_122 = torch.ops.aten.permute.default(mul_104, [0, 2, 1]);  mul_104 = None
        permute_123 = torch.ops.aten.permute.default(permute_66, [0, 2, 1]);  permute_66 = None
        permute_124 = torch.ops.aten.permute.default(permute_67, [0, 2, 1]);  permute_67 = None
        permute_125 = torch.ops.aten.permute.default(mul_102, [0, 2, 1]);  mul_102 = None
        div_14 = torch.ops.aten.div.Tensor(rsqrt_20, 128);  rsqrt_20 = None
        div_16 = torch.ops.aten.div.Tensor(rsqrt_18, 128);  rsqrt_18 = None
        permute_150 = torch.ops.aten.permute.default(mul_84, [0, 2, 1]);  mul_84 = None
        permute_151 = torch.ops.aten.permute.default(permute_54, [0, 2, 1]);  permute_54 = None
        permute_152 = torch.ops.aten.permute.default(permute_55, [0, 2, 1]);  permute_55 = None
        permute_153 = torch.ops.aten.permute.default(mul_82, [0, 2, 1]);  mul_82 = None
        div_18 = torch.ops.aten.div.Tensor(rsqrt_16, 128);  rsqrt_16 = None
        div_20 = torch.ops.aten.div.Tensor(rsqrt_14, 128);  rsqrt_14 = None
        permute_178 = torch.ops.aten.permute.default(mul_64, [0, 2, 1]);  mul_64 = None
        permute_179 = torch.ops.aten.permute.default(permute_42, [0, 2, 1]);  permute_42 = None
        permute_180 = torch.ops.aten.permute.default(permute_43, [0, 2, 1]);  permute_43 = None
        permute_181 = torch.ops.aten.permute.default(mul_62, [0, 2, 1]);  mul_62 = None
        div_22 = torch.ops.aten.div.Tensor(rsqrt_12, 128);  rsqrt_12 = None
        div_24 = torch.ops.aten.div.Tensor(rsqrt_10, 128);  rsqrt_10 = None
        permute_206 = torch.ops.aten.permute.default(mul_44, [0, 2, 1]);  mul_44 = None
        permute_207 = torch.ops.aten.permute.default(permute_30, [0, 2, 1]);  permute_30 = None
        permute_208 = torch.ops.aten.permute.default(permute_31, [0, 2, 1]);  permute_31 = None
        permute_209 = torch.ops.aten.permute.default(mul_42, [0, 2, 1]);  mul_42 = None
        div_26 = torch.ops.aten.div.Tensor(rsqrt_8, 128);  rsqrt_8 = None
        div_28 = torch.ops.aten.div.Tensor(rsqrt_6, 128);  rsqrt_6 = None
        permute_234 = torch.ops.aten.permute.default(mul_24, [0, 2, 1]);  mul_24 = None
        permute_235 = torch.ops.aten.permute.default(permute_18, [0, 2, 1]);  permute_18 = None
        permute_236 = torch.ops.aten.permute.default(permute_19, [0, 2, 1]);  permute_19 = None
        permute_237 = torch.ops.aten.permute.default(mul_22, [0, 2, 1]);  mul_22 = None
        div_30 = torch.ops.aten.div.Tensor(rsqrt_4, 128);  rsqrt_4 = None
        div_32 = torch.ops.aten.div.Tensor(rsqrt_2, 128);  rsqrt_2 = None
        permute_262 = torch.ops.aten.permute.default(mul_4, [0, 2, 1]);  mul_4 = None
        permute_263 = torch.ops.aten.permute.default(permute_6, [0, 2, 1]);  permute_6 = None
        permute_264 = torch.ops.aten.permute.default(permute_7, [0, 2, 1]);  permute_7 = None
        permute_265 = torch.ops.aten.permute.default(mul_2, [0, 2, 1]);  mul_2 = None
        return (add_89, add_84, primals_2, primals_4, primals_6, primals_8, primals_12, primals_14, primals_16, primals_18, primals_20, primals_22, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_38, primals_40, primals_42, primals_44, primals_46, primals_48, primals_50, primals_52, primals_54, primals_56, primals_58, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_106, primals_108, primals_110, primals_112, primals_114, view_1, addmm, getitem_1, rsqrt, view_3, baddbmm, amax, sum_1, gt, view_11, addmm_1, getitem_3, rsqrt_1, gt_1, mul_9, view_14, addmm_2, gt_2, getitem_7, rsqrt_3, view_16, gt_3, mul_20, view_18, baddbmm_1, amax_1, sum_2, gt_4, view_26, addmm_4, getitem_11, rsqrt_5, gt_5, mul_29, view_29, addmm_5, gt_6, getitem_15, rsqrt_7, view_31, gt_7, mul_40, view_33, baddbmm_2, amax_2, sum_3, gt_8, view_41, addmm_7, getitem_19, rsqrt_9, gt_9, mul_49, view_44, addmm_8, gt_10, getitem_23, rsqrt_11, view_46, gt_11, mul_60, view_48, baddbmm_3, amax_3, sum_4, gt_12, view_56, addmm_10, getitem_27, rsqrt_13, gt_13, mul_69, view_59, addmm_11, gt_14, getitem_31, rsqrt_15, view_61, gt_15, mul_80, view_63, baddbmm_4, amax_4, sum_5, gt_16, view_71, addmm_13, getitem_35, rsqrt_17, gt_17, mul_89, view_74, addmm_14, gt_18, getitem_39, rsqrt_19, view_76, gt_19, mul_100, view_78, baddbmm_5, amax_5, sum_6, gt_20, view_86, addmm_16, getitem_43, rsqrt_21, gt_21, mul_109, view_89, addmm_17, gt_22, getitem_47, rsqrt_23, view_91, gt_23, mul_120, view_93, baddbmm_6, amax_6, sum_7, gt_24, view_101, addmm_19, getitem_51, rsqrt_25, gt_25, add_84, getitem_53, rsqrt_26, view_104, addmm_20, gt_26, getitem_55, rsqrt_27, permute_94, permute_95, permute_96, permute_97, div_10, div_12, permute_122, permute_123, permute_124, permute_125, div_14, div_16, permute_150, permute_151, permute_152, permute_153, div_18, div_20, permute_178, permute_179, permute_180, permute_181, div_22, div_24, permute_206, permute_207, permute_208, permute_209, div_26, div_28, permute_234, permute_235, permute_236, permute_237, div_30, div_32, permute_262, permute_263, permute_264, permute_265)
        
def load_args(reader):
    buf0 = reader.storage(None, 4096000, device=device(type='cuda', index=0))
    reader.tensor(buf0, (500, 128, 1, 16), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf1, (128, 16), is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf2, (128,), is_leaf=True)  # primals_3
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
    buf9 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf9, (500, 128), is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 262144000, device=device(type='cuda', index=0))
    reader.tensor(buf10, (4000, 128, 128), is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf11, (128,), is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf12, (128,), is_leaf=True)  # primals_13
    buf13 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf13, (128,), is_leaf=True)  # primals_14
    buf14 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf14, (128,), is_leaf=True)  # primals_15
    buf15 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf15, (512, 128), is_leaf=True)  # primals_16
    buf16 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf16, (512,), is_leaf=True)  # primals_17
    buf17 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf17, (512,), is_leaf=True)  # primals_18
    buf18 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf18, (512,), is_leaf=True)  # primals_19
    buf19 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf19, (128, 512), is_leaf=True)  # primals_20
    buf20 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf20, (128,), is_leaf=True)  # primals_21
    buf21 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf21, (128,), is_leaf=True)  # primals_22
    buf22 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf22, (128,), is_leaf=True)  # primals_23
    buf23 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf23, (384, 128), is_leaf=True)  # primals_24
    buf24 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf24, (384,), is_leaf=True)  # primals_25
    buf25 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf25, (128, 128), is_leaf=True)  # primals_26
    buf26 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf26, (128,), is_leaf=True)  # primals_27
    buf27 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf27, (128,), is_leaf=True)  # primals_28
    buf28 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf28, (128,), is_leaf=True)  # primals_29
    buf29 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf29, (128,), is_leaf=True)  # primals_30
    buf30 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf30, (128,), is_leaf=True)  # primals_31
    buf31 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf31, (512, 128), is_leaf=True)  # primals_32
    buf32 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf32, (512,), is_leaf=True)  # primals_33
    buf33 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf33, (512,), is_leaf=True)  # primals_34
    buf34 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf34, (512,), is_leaf=True)  # primals_35
    buf35 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf35, (128, 512), is_leaf=True)  # primals_36
    buf36 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf36, (128,), is_leaf=True)  # primals_37
    buf37 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf37, (128,), is_leaf=True)  # primals_38
    buf38 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf38, (128,), is_leaf=True)  # primals_39
    buf39 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf39, (384, 128), is_leaf=True)  # primals_40
    buf40 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf40, (384,), is_leaf=True)  # primals_41
    buf41 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf41, (128, 128), is_leaf=True)  # primals_42
    buf42 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf42, (128,), is_leaf=True)  # primals_43
    buf43 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf43, (128,), is_leaf=True)  # primals_44
    buf44 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf44, (128,), is_leaf=True)  # primals_45
    buf45 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf45, (128,), is_leaf=True)  # primals_46
    buf46 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf46, (128,), is_leaf=True)  # primals_47
    buf47 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf47, (512, 128), is_leaf=True)  # primals_48
    buf48 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf48, (512,), is_leaf=True)  # primals_49
    buf49 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf49, (512,), is_leaf=True)  # primals_50
    buf50 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf50, (512,), is_leaf=True)  # primals_51
    buf51 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf51, (128, 512), is_leaf=True)  # primals_52
    buf52 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf52, (128,), is_leaf=True)  # primals_53
    buf53 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf53, (128,), is_leaf=True)  # primals_54
    buf54 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf54, (128,), is_leaf=True)  # primals_55
    buf55 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf55, (384, 128), is_leaf=True)  # primals_56
    buf56 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf56, (384,), is_leaf=True)  # primals_57
    buf57 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf57, (128, 128), is_leaf=True)  # primals_58
    buf58 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf58, (128,), is_leaf=True)  # primals_59
    buf59 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf59, (128,), is_leaf=True)  # primals_60
    buf60 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf60, (128,), is_leaf=True)  # primals_61
    buf61 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf61, (128,), is_leaf=True)  # primals_62
    buf62 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf62, (128,), is_leaf=True)  # primals_63
    buf63 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf63, (512, 128), is_leaf=True)  # primals_64
    buf64 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf64, (512,), is_leaf=True)  # primals_65
    buf65 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf65, (512,), is_leaf=True)  # primals_66
    buf66 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf66, (512,), is_leaf=True)  # primals_67
    buf67 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf67, (128, 512), is_leaf=True)  # primals_68
    buf68 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf68, (128,), is_leaf=True)  # primals_69
    buf69 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf69, (128,), is_leaf=True)  # primals_70
    buf70 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf70, (128,), is_leaf=True)  # primals_71
    buf71 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf71, (384, 128), is_leaf=True)  # primals_72
    buf72 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf72, (384,), is_leaf=True)  # primals_73
    buf73 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf73, (128, 128), is_leaf=True)  # primals_74
    buf74 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf74, (128,), is_leaf=True)  # primals_75
    buf75 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf75, (128,), is_leaf=True)  # primals_76
    buf76 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf76, (128,), is_leaf=True)  # primals_77
    buf77 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf77, (128,), is_leaf=True)  # primals_78
    buf78 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf78, (128,), is_leaf=True)  # primals_79
    buf79 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf79, (512, 128), is_leaf=True)  # primals_80
    buf80 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf80, (512,), is_leaf=True)  # primals_81
    buf81 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf81, (512,), is_leaf=True)  # primals_82
    buf82 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf82, (512,), is_leaf=True)  # primals_83
    buf83 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf83, (128, 512), is_leaf=True)  # primals_84
    buf84 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf84, (128,), is_leaf=True)  # primals_85
    buf85 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf85, (128,), is_leaf=True)  # primals_86
    buf86 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf86, (128,), is_leaf=True)  # primals_87
    buf87 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf87, (384, 128), is_leaf=True)  # primals_88
    buf88 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf88, (384,), is_leaf=True)  # primals_89
    buf89 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf89, (128, 128), is_leaf=True)  # primals_90
    buf90 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf90, (128,), is_leaf=True)  # primals_91
    buf91 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf91, (128,), is_leaf=True)  # primals_92
    buf92 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf92, (128,), is_leaf=True)  # primals_93
    buf93 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf93, (128,), is_leaf=True)  # primals_94
    buf94 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf94, (128,), is_leaf=True)  # primals_95
    buf95 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf95, (512, 128), is_leaf=True)  # primals_96
    buf96 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf96, (512,), is_leaf=True)  # primals_97
    buf97 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf97, (512,), is_leaf=True)  # primals_98
    buf98 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf98, (512,), is_leaf=True)  # primals_99
    buf99 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf99, (128, 512), is_leaf=True)  # primals_100
    buf100 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf100, (128,), is_leaf=True)  # primals_101
    buf101 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf101, (128,), is_leaf=True)  # primals_102
    buf102 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf102, (128,), is_leaf=True)  # primals_103
    buf103 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf103, (384, 128), is_leaf=True)  # primals_104
    buf104 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf104, (384,), is_leaf=True)  # primals_105
    buf105 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf105, (128, 128), is_leaf=True)  # primals_106
    buf106 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf106, (128,), is_leaf=True)  # primals_107
    buf107 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf107, (128,), is_leaf=True)  # primals_108
    buf108 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf108, (128,), is_leaf=True)  # primals_109
    buf109 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf109, (128,), is_leaf=True)  # primals_110
    buf110 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf110, (128,), is_leaf=True)  # primals_111
    buf111 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf111, (512, 128), is_leaf=True)  # primals_112
    buf112 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf112, (512,), is_leaf=True)  # primals_113
    buf113 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf113, (512,), is_leaf=True)  # primals_114
    buf114 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf114, (512,), is_leaf=True)  # primals_115
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)