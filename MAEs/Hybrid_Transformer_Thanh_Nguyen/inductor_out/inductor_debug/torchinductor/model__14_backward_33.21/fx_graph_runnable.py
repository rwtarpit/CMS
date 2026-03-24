
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

    
    
    def forward(self, primals_2, primals_4, primals_6, primals_8, primals_12, primals_14, primals_16, primals_18, primals_20, primals_22, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_38, primals_40, primals_42, primals_44, primals_46, primals_48, primals_50, primals_52, primals_54, primals_56, primals_58, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_106, primals_108, primals_110, primals_112, primals_114, view_1, addmm, getitem_1, rsqrt, view_3, baddbmm, amax, sum_1, gt, view_11, addmm_1, getitem_3, rsqrt_1, gt_1, mul_9, view_14, addmm_2, gt_2, getitem_7, rsqrt_3, view_16, gt_3, mul_20, view_18, baddbmm_1, amax_1, sum_2, gt_4, view_26, addmm_4, getitem_11, rsqrt_5, gt_5, mul_29, view_29, addmm_5, gt_6, getitem_15, rsqrt_7, view_31, gt_7, mul_40, view_33, baddbmm_2, amax_2, sum_3, gt_8, view_41, addmm_7, getitem_19, rsqrt_9, gt_9, mul_49, view_44, addmm_8, gt_10, getitem_23, rsqrt_11, view_46, gt_11, mul_60, view_48, baddbmm_3, amax_3, sum_4, gt_12, view_56, addmm_10, getitem_27, rsqrt_13, gt_13, mul_69, view_59, addmm_11, gt_14, getitem_31, rsqrt_15, view_61, gt_15, mul_80, view_63, baddbmm_4, amax_4, sum_5, gt_16, view_71, addmm_13, getitem_35, rsqrt_17, gt_17, mul_89, view_74, addmm_14, gt_18, getitem_39, rsqrt_19, view_76, gt_19, mul_100, view_78, baddbmm_5, amax_5, sum_6, gt_20, view_86, addmm_16, getitem_43, rsqrt_21, gt_21, mul_109, view_89, addmm_17, gt_22, getitem_47, rsqrt_23, view_91, gt_23, mul_120, view_93, baddbmm_6, amax_6, sum_7, gt_24, view_101, addmm_19, getitem_51, rsqrt_25, gt_25, add_84, getitem_53, rsqrt_26, view_104, addmm_20, gt_26, getitem_55, rsqrt_27, permute_94, permute_95, permute_96, permute_97, div_10, div_12, permute_122, permute_123, permute_124, permute_125, div_14, div_16, permute_150, permute_151, permute_152, permute_153, div_18, div_20, permute_178, permute_179, permute_180, permute_181, div_22, div_24, permute_206, permute_207, permute_208, permute_209, div_26, div_28, permute_234, permute_235, permute_236, permute_237, div_30, div_32, permute_262, permute_263, permute_264, permute_265, tangents_1, tangents_2):
        mul_139 = torch.ops.aten.mul.Tensor(tangents_1, primals_114);  primals_114 = None
        mul_140 = torch.ops.aten.mul.Tensor(mul_139, 512)
        sum_8 = torch.ops.aten.sum.dim_IntList(mul_139, [2], True)
        view_105 = torch.ops.aten.view.default(addmm_20, [500, 128, 512]);  addmm_20 = None
        mul_131 = torch.ops.aten.mul.Tensor(view_105, 0.5)
        mul_132 = torch.ops.aten.mul.Tensor(view_105, 0.7071067811865476)
        erf_6 = torch.ops.aten.erf.default(mul_132);  mul_132 = None
        add_87 = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_133 = torch.ops.aten.mul.Tensor(mul_131, add_87);  mul_131 = None
        mul_134 = torch.ops.aten.mul.Tensor(gt_26, mul_133);  mul_133 = None
        mul_135 = torch.ops.aten.mul.Tensor(mul_134, 1.1111111111111112);  mul_134 = None
        sub_34 = torch.ops.aten.sub.Tensor(mul_135, getitem_55);  mul_135 = getitem_55 = None
        mul_136 = torch.ops.aten.mul.Tensor(sub_34, rsqrt_27);  sub_34 = None
        mul_141 = torch.ops.aten.mul.Tensor(mul_139, mul_136);  mul_139 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(mul_141, [2], True);  mul_141 = None
        mul_142 = torch.ops.aten.mul.Tensor(mul_136, sum_9);  sum_9 = None
        sub_36 = torch.ops.aten.sub.Tensor(mul_140, sum_8);  mul_140 = sum_8 = None
        sub_37 = torch.ops.aten.sub.Tensor(sub_36, mul_142);  sub_36 = mul_142 = None
        div_7 = torch.ops.aten.div.Tensor(rsqrt_27, 512);  rsqrt_27 = None
        mul_143 = torch.ops.aten.mul.Tensor(div_7, sub_37);  div_7 = sub_37 = None
        mul_144 = torch.ops.aten.mul.Tensor(tangents_1, mul_136);  mul_136 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(mul_144, [0, 1]);  mul_144 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(tangents_1, [0, 1]);  tangents_1 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(gt_26, torch.float32);  gt_26 = None
        mul_145 = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
        mul_146 = torch.ops.aten.mul.Tensor(mul_143, mul_145);  mul_143 = mul_145 = None
        mul_148 = torch.ops.aten.mul.Tensor(add_87, 0.5);  add_87 = None
        mul_149 = torch.ops.aten.mul.Tensor(view_105, view_105)
        mul_150 = torch.ops.aten.mul.Tensor(mul_149, -0.5);  mul_149 = None
        exp_7 = torch.ops.aten.exp.default(mul_150);  mul_150 = None
        mul_151 = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
        mul_152 = torch.ops.aten.mul.Tensor(view_105, mul_151);  view_105 = mul_151 = None
        add_91 = torch.ops.aten.add.Tensor(mul_148, mul_152);  mul_148 = mul_152 = None
        mul_153 = torch.ops.aten.mul.Tensor(mul_146, add_91);  mul_146 = add_91 = None
        view_106 = torch.ops.aten.view.default(mul_153, [64000, 512]);  mul_153 = None
        permute_83 = torch.ops.aten.permute.default(primals_112, [1, 0]);  primals_112 = None
        permute_84 = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
        mm_7 = torch.ops.aten.mm.default(view_106, permute_84);  permute_84 = None
        permute_85 = torch.ops.aten.permute.default(view_106, [1, 0])
        mm_8 = torch.ops.aten.mm.default(permute_85, view_104);  permute_85 = view_104 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(view_106, [0], True);  view_106 = None
        view_107 = torch.ops.aten.view.default(sum_12, [512]);  sum_12 = None
        view_108 = torch.ops.aten.view.default(mm_7, [500, 128, 128]);  mm_7 = None
        mul_155 = torch.ops.aten.mul.Tensor(view_108, primals_110);  primals_110 = None
        mul_156 = torch.ops.aten.mul.Tensor(mul_155, 128)
        sum_13 = torch.ops.aten.sum.dim_IntList(mul_155, [2], True)
        sub_33 = torch.ops.aten.sub.Tensor(add_84, getitem_53);  add_84 = getitem_53 = None
        mul_129 = torch.ops.aten.mul.Tensor(sub_33, rsqrt_26);  sub_33 = None
        mul_157 = torch.ops.aten.mul.Tensor(mul_155, mul_129);  mul_155 = None
        sum_14 = torch.ops.aten.sum.dim_IntList(mul_157, [2], True);  mul_157 = None
        mul_158 = torch.ops.aten.mul.Tensor(mul_129, sum_14);  sum_14 = None
        sub_39 = torch.ops.aten.sub.Tensor(mul_156, sum_13);  mul_156 = sum_13 = None
        sub_40 = torch.ops.aten.sub.Tensor(sub_39, mul_158);  sub_39 = mul_158 = None
        div_8 = torch.ops.aten.div.Tensor(rsqrt_26, 128);  rsqrt_26 = None
        mul_159 = torch.ops.aten.mul.Tensor(div_8, sub_40);  div_8 = sub_40 = None
        mul_160 = torch.ops.aten.mul.Tensor(view_108, mul_129);  mul_129 = None
        sum_15 = torch.ops.aten.sum.dim_IntList(mul_160, [0, 1]);  mul_160 = None
        sum_16 = torch.ops.aten.sum.dim_IntList(view_108, [0, 1]);  view_108 = None
        add_92 = torch.ops.aten.add.Tensor(tangents_2, mul_159);  tangents_2 = mul_159 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(gt_25, torch.float32);  gt_25 = None
        mul_161 = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_162 = torch.ops.aten.mul.Tensor(add_92, mul_161);  mul_161 = None
        view_102 = torch.ops.aten.view.default(addmm_19, [128, 500, 128]);  addmm_19 = None
        permute_82 = torch.ops.aten.permute.default(view_102, [1, 0, 2]);  view_102 = None
        sub_41 = torch.ops.aten.sub.Tensor(permute_82, getitem_51);  permute_82 = getitem_51 = None
        mul_163 = torch.ops.aten.mul.Tensor(sub_41, rsqrt_25);  sub_41 = None
        mul_164 = torch.ops.aten.mul.Tensor(mul_162, primals_108);  primals_108 = None
        mul_165 = torch.ops.aten.mul.Tensor(mul_164, 128)
        sum_17 = torch.ops.aten.sum.dim_IntList(mul_164, [2], True)
        mul_166 = torch.ops.aten.mul.Tensor(mul_164, mul_163);  mul_164 = None
        sum_18 = torch.ops.aten.sum.dim_IntList(mul_166, [2], True);  mul_166 = None
        mul_167 = torch.ops.aten.mul.Tensor(mul_163, sum_18);  sum_18 = None
        sub_42 = torch.ops.aten.sub.Tensor(mul_165, sum_17);  mul_165 = sum_17 = None
        sub_43 = torch.ops.aten.sub.Tensor(sub_42, mul_167);  sub_42 = mul_167 = None
        div_9 = torch.ops.aten.div.Tensor(rsqrt_25, 128);  rsqrt_25 = None
        mul_168 = torch.ops.aten.mul.Tensor(div_9, sub_43);  div_9 = sub_43 = None
        mul_169 = torch.ops.aten.mul.Tensor(mul_162, mul_163);  mul_163 = None
        sum_19 = torch.ops.aten.sum.dim_IntList(mul_169, [0, 1]);  mul_169 = None
        sum_20 = torch.ops.aten.sum.dim_IntList(mul_162, [0, 1]);  mul_162 = None
        permute_88 = torch.ops.aten.permute.default(mul_168, [1, 0, 2]);  mul_168 = None
        clone_37 = torch.ops.aten.clone.default(permute_88, memory_format = torch.contiguous_format);  permute_88 = None
        view_109 = torch.ops.aten.view.default(clone_37, [64000, 128]);  clone_37 = None
        permute_81 = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
        permute_89 = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
        mm_9 = torch.ops.aten.mm.default(view_109, permute_89);  permute_89 = None
        permute_90 = torch.ops.aten.permute.default(view_109, [1, 0])
        mm_10 = torch.ops.aten.mm.default(permute_90, view_101);  permute_90 = view_101 = None
        sum_21 = torch.ops.aten.sum.dim_IntList(view_109, [0], True);  view_109 = None
        view_110 = torch.ops.aten.view.default(sum_21, [128]);  sum_21 = None
        view_111 = torch.ops.aten.view.default(mm_9, [128, 4000, 16]);  mm_9 = None
        permute_93 = torch.ops.aten.permute.default(view_111, [1, 0, 2]);  view_111 = None
        bmm_7 = torch.ops.aten.bmm.default(permute_94, permute_93);  permute_94 = None
        bmm_8 = torch.ops.aten.bmm.default(permute_93, permute_95);  permute_93 = permute_95 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(gt_24, torch.float32);  gt_24 = None
        mul_170 = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
        mul_171 = torch.ops.aten.mul.Tensor(bmm_8, mul_170);  bmm_8 = mul_170 = None
        sub_31 = torch.ops.aten.sub.Tensor(baddbmm_6, amax_6);  baddbmm_6 = amax_6 = None
        exp_6 = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        div_6 = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        mul_172 = torch.ops.aten.mul.Tensor(mul_171, div_6);  mul_171 = None
        sum_22 = torch.ops.aten.sum.dim_IntList(mul_172, [-1], True)
        neg = torch.ops.aten.neg.default(div_6);  div_6 = None
        fma = torch.ops.prims.fma.default(neg, sum_22, mul_172);  neg = sum_22 = mul_172 = None
        bmm_9 = torch.ops.aten.bmm.default(fma, permute_96);  permute_96 = None
        bmm_10 = torch.ops.aten.bmm.default(permute_97, fma);  permute_97 = None
        permute_98 = torch.ops.aten.permute.default(bmm_10, [0, 2, 1]);  bmm_10 = None
        mul_173 = torch.ops.aten.mul.Tensor(bmm_9, 0.25);  bmm_9 = None
        permute_99 = torch.ops.aten.permute.default(bmm_7, [1, 0, 2]);  bmm_7 = None
        clone_39 = torch.ops.aten.clone.default(permute_99, memory_format = torch.contiguous_format);  permute_99 = None
        view_112 = torch.ops.aten.view.default(clone_39, [128, 500, 128]);  clone_39 = None
        permute_100 = torch.ops.aten.permute.default(permute_98, [1, 0, 2]);  permute_98 = None
        view_113 = torch.ops.aten.view.default(permute_100, [128, 500, 128]);  permute_100 = None
        permute_101 = torch.ops.aten.permute.default(mul_173, [1, 0, 2]);  mul_173 = None
        clone_40 = torch.ops.aten.clone.default(permute_101, memory_format = torch.contiguous_format);  permute_101 = None
        view_114 = torch.ops.aten.view.default(clone_40, [128, 500, 128]);  clone_40 = None
        full_default = torch.ops.aten.full.default([3, 128, 500, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter = torch.ops.aten.select_scatter.default(full_default, view_112, 0, 2);  view_112 = None
        select_scatter_1 = torch.ops.aten.select_scatter.default(full_default, view_113, 0, 1);  view_113 = None
        add_93 = torch.ops.aten.add.Tensor(select_scatter, select_scatter_1);  select_scatter = select_scatter_1 = None
        select_scatter_2 = torch.ops.aten.select_scatter.default(full_default, view_114, 0, 0);  view_114 = None
        add_94 = torch.ops.aten.add.Tensor(add_93, select_scatter_2);  add_93 = select_scatter_2 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(add_94, 3);  add_94 = None
        permute_102 = torch.ops.aten.permute.default(unsqueeze_7, [3, 1, 2, 0, 4]);  unsqueeze_7 = None
        squeeze_7 = torch.ops.aten.squeeze.dim(permute_102, 0);  permute_102 = None
        clone_41 = torch.ops.aten.clone.default(squeeze_7, memory_format = torch.contiguous_format);  squeeze_7 = None
        view_115 = torch.ops.aten.view.default(clone_41, [128, 500, 384]);  clone_41 = None
        sum_23 = torch.ops.aten.sum.dim_IntList(view_115, [0, 1], True)
        view_116 = torch.ops.aten.view.default(sum_23, [384]);  sum_23 = None
        view_117 = torch.ops.aten.view.default(view_115, [64000, 384]);  view_115 = None
        permute_103 = torch.ops.aten.permute.default(view_117, [1, 0])
        mm_11 = torch.ops.aten.mm.default(permute_103, view_93);  permute_103 = view_93 = None
        permute_74 = torch.ops.aten.permute.default(primals_104, [1, 0]);  primals_104 = None
        permute_105 = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        mm_12 = torch.ops.aten.mm.default(view_117, permute_105);  view_117 = permute_105 = None
        view_118 = torch.ops.aten.view.default(mm_12, [128, 500, 128]);  mm_12 = None
        permute_107 = torch.ops.aten.permute.default(view_118, [1, 0, 2]);  view_118 = None
        mul_175 = torch.ops.aten.mul.Tensor(permute_107, primals_102);  primals_102 = None
        mul_176 = torch.ops.aten.mul.Tensor(mul_175, 128)
        sum_24 = torch.ops.aten.sum.dim_IntList(mul_175, [2], True)
        mul_177 = torch.ops.aten.mul.Tensor(mul_175, mul_120);  mul_175 = None
        sum_25 = torch.ops.aten.sum.dim_IntList(mul_177, [2], True);  mul_177 = None
        mul_178 = torch.ops.aten.mul.Tensor(mul_120, sum_25);  sum_25 = None
        sub_45 = torch.ops.aten.sub.Tensor(mul_176, sum_24);  mul_176 = sum_24 = None
        sub_46 = torch.ops.aten.sub.Tensor(sub_45, mul_178);  sub_45 = mul_178 = None
        mul_179 = torch.ops.aten.mul.Tensor(div_10, sub_46);  div_10 = sub_46 = None
        mul_180 = torch.ops.aten.mul.Tensor(permute_107, mul_120);  mul_120 = None
        sum_26 = torch.ops.aten.sum.dim_IntList(mul_180, [0, 1]);  mul_180 = None
        sum_27 = torch.ops.aten.sum.dim_IntList(permute_107, [0, 1]);  permute_107 = None
        add_95 = torch.ops.aten.add.Tensor(add_92, mul_179);  add_92 = mul_179 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(gt_23, torch.float32);  gt_23 = None
        mul_181 = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.1111111111111112);  convert_element_type_3 = None
        mul_182 = torch.ops.aten.mul.Tensor(add_95, mul_181);  mul_181 = None
        view_119 = torch.ops.aten.view.default(mul_182, [64000, 128]);  mul_182 = None
        permute_72 = torch.ops.aten.permute.default(primals_100, [1, 0]);  primals_100 = None
        permute_108 = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
        mm_13 = torch.ops.aten.mm.default(view_119, permute_108);  permute_108 = None
        permute_109 = torch.ops.aten.permute.default(view_119, [1, 0])
        mm_14 = torch.ops.aten.mm.default(permute_109, view_91);  permute_109 = view_91 = None
        sum_28 = torch.ops.aten.sum.dim_IntList(view_119, [0], True);  view_119 = None
        view_120 = torch.ops.aten.view.default(sum_28, [128]);  sum_28 = None
        view_121 = torch.ops.aten.view.default(mm_13, [500, 128, 512]);  mm_13 = None
        mul_184 = torch.ops.aten.mul.Tensor(view_121, primals_98);  primals_98 = None
        mul_185 = torch.ops.aten.mul.Tensor(mul_184, 512)
        sum_29 = torch.ops.aten.sum.dim_IntList(mul_184, [2], True)
        view_90 = torch.ops.aten.view.default(addmm_17, [500, 128, 512]);  addmm_17 = None
        mul_111 = torch.ops.aten.mul.Tensor(view_90, 0.5)
        mul_112 = torch.ops.aten.mul.Tensor(view_90, 0.7071067811865476)
        erf_5 = torch.ops.aten.erf.default(mul_112);  mul_112 = None
        add_74 = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_113 = torch.ops.aten.mul.Tensor(mul_111, add_74);  mul_111 = None
        mul_114 = torch.ops.aten.mul.Tensor(gt_22, mul_113);  mul_113 = None
        mul_115 = torch.ops.aten.mul.Tensor(mul_114, 1.1111111111111112);  mul_114 = None
        sub_29 = torch.ops.aten.sub.Tensor(mul_115, getitem_47);  mul_115 = getitem_47 = None
        mul_116 = torch.ops.aten.mul.Tensor(sub_29, rsqrt_23);  sub_29 = None
        mul_186 = torch.ops.aten.mul.Tensor(mul_184, mul_116);  mul_184 = None
        sum_30 = torch.ops.aten.sum.dim_IntList(mul_186, [2], True);  mul_186 = None
        mul_187 = torch.ops.aten.mul.Tensor(mul_116, sum_30);  sum_30 = None
        sub_48 = torch.ops.aten.sub.Tensor(mul_185, sum_29);  mul_185 = sum_29 = None
        sub_49 = torch.ops.aten.sub.Tensor(sub_48, mul_187);  sub_48 = mul_187 = None
        div_11 = torch.ops.aten.div.Tensor(rsqrt_23, 512);  rsqrt_23 = None
        mul_188 = torch.ops.aten.mul.Tensor(div_11, sub_49);  div_11 = sub_49 = None
        mul_189 = torch.ops.aten.mul.Tensor(view_121, mul_116);  mul_116 = None
        sum_31 = torch.ops.aten.sum.dim_IntList(mul_189, [0, 1]);  mul_189 = None
        sum_32 = torch.ops.aten.sum.dim_IntList(view_121, [0, 1]);  view_121 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(gt_22, torch.float32);  gt_22 = None
        mul_190 = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
        mul_191 = torch.ops.aten.mul.Tensor(mul_188, mul_190);  mul_188 = mul_190 = None
        mul_193 = torch.ops.aten.mul.Tensor(add_74, 0.5);  add_74 = None
        mul_194 = torch.ops.aten.mul.Tensor(view_90, view_90)
        mul_195 = torch.ops.aten.mul.Tensor(mul_194, -0.5);  mul_194 = None
        exp_8 = torch.ops.aten.exp.default(mul_195);  mul_195 = None
        mul_196 = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
        mul_197 = torch.ops.aten.mul.Tensor(view_90, mul_196);  view_90 = mul_196 = None
        add_97 = torch.ops.aten.add.Tensor(mul_193, mul_197);  mul_193 = mul_197 = None
        mul_198 = torch.ops.aten.mul.Tensor(mul_191, add_97);  mul_191 = add_97 = None
        view_122 = torch.ops.aten.view.default(mul_198, [64000, 512]);  mul_198 = None
        permute_71 = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
        permute_112 = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
        mm_15 = torch.ops.aten.mm.default(view_122, permute_112);  permute_112 = None
        permute_113 = torch.ops.aten.permute.default(view_122, [1, 0])
        mm_16 = torch.ops.aten.mm.default(permute_113, view_89);  permute_113 = view_89 = None
        sum_33 = torch.ops.aten.sum.dim_IntList(view_122, [0], True);  view_122 = None
        view_123 = torch.ops.aten.view.default(sum_33, [512]);  sum_33 = None
        view_124 = torch.ops.aten.view.default(mm_15, [500, 128, 128]);  mm_15 = None
        mul_200 = torch.ops.aten.mul.Tensor(view_124, primals_94);  primals_94 = None
        mul_201 = torch.ops.aten.mul.Tensor(mul_200, 128)
        sum_34 = torch.ops.aten.sum.dim_IntList(mul_200, [2], True)
        mul_202 = torch.ops.aten.mul.Tensor(mul_200, mul_109);  mul_200 = None
        sum_35 = torch.ops.aten.sum.dim_IntList(mul_202, [2], True);  mul_202 = None
        mul_203 = torch.ops.aten.mul.Tensor(mul_109, sum_35);  sum_35 = None
        sub_51 = torch.ops.aten.sub.Tensor(mul_201, sum_34);  mul_201 = sum_34 = None
        sub_52 = torch.ops.aten.sub.Tensor(sub_51, mul_203);  sub_51 = mul_203 = None
        mul_204 = torch.ops.aten.mul.Tensor(div_12, sub_52);  div_12 = sub_52 = None
        mul_205 = torch.ops.aten.mul.Tensor(view_124, mul_109);  mul_109 = None
        sum_36 = torch.ops.aten.sum.dim_IntList(mul_205, [0, 1]);  mul_205 = None
        sum_37 = torch.ops.aten.sum.dim_IntList(view_124, [0, 1]);  view_124 = None
        add_98 = torch.ops.aten.add.Tensor(add_95, mul_204);  add_95 = mul_204 = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(gt_21, torch.float32);  gt_21 = None
        mul_206 = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
        mul_207 = torch.ops.aten.mul.Tensor(add_98, mul_206);  mul_206 = None
        view_87 = torch.ops.aten.view.default(addmm_16, [128, 500, 128]);  addmm_16 = None
        permute_70 = torch.ops.aten.permute.default(view_87, [1, 0, 2]);  view_87 = None
        sub_53 = torch.ops.aten.sub.Tensor(permute_70, getitem_43);  permute_70 = getitem_43 = None
        mul_208 = torch.ops.aten.mul.Tensor(sub_53, rsqrt_21);  sub_53 = None
        mul_209 = torch.ops.aten.mul.Tensor(mul_207, primals_92);  primals_92 = None
        mul_210 = torch.ops.aten.mul.Tensor(mul_209, 128)
        sum_38 = torch.ops.aten.sum.dim_IntList(mul_209, [2], True)
        mul_211 = torch.ops.aten.mul.Tensor(mul_209, mul_208);  mul_209 = None
        sum_39 = torch.ops.aten.sum.dim_IntList(mul_211, [2], True);  mul_211 = None
        mul_212 = torch.ops.aten.mul.Tensor(mul_208, sum_39);  sum_39 = None
        sub_54 = torch.ops.aten.sub.Tensor(mul_210, sum_38);  mul_210 = sum_38 = None
        sub_55 = torch.ops.aten.sub.Tensor(sub_54, mul_212);  sub_54 = mul_212 = None
        div_13 = torch.ops.aten.div.Tensor(rsqrt_21, 128);  rsqrt_21 = None
        mul_213 = torch.ops.aten.mul.Tensor(div_13, sub_55);  div_13 = sub_55 = None
        mul_214 = torch.ops.aten.mul.Tensor(mul_207, mul_208);  mul_208 = None
        sum_40 = torch.ops.aten.sum.dim_IntList(mul_214, [0, 1]);  mul_214 = None
        sum_41 = torch.ops.aten.sum.dim_IntList(mul_207, [0, 1]);  mul_207 = None
        permute_116 = torch.ops.aten.permute.default(mul_213, [1, 0, 2]);  mul_213 = None
        clone_45 = torch.ops.aten.clone.default(permute_116, memory_format = torch.contiguous_format);  permute_116 = None
        view_125 = torch.ops.aten.view.default(clone_45, [64000, 128]);  clone_45 = None
        permute_69 = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
        permute_117 = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
        mm_17 = torch.ops.aten.mm.default(view_125, permute_117);  permute_117 = None
        permute_118 = torch.ops.aten.permute.default(view_125, [1, 0])
        mm_18 = torch.ops.aten.mm.default(permute_118, view_86);  permute_118 = view_86 = None
        sum_42 = torch.ops.aten.sum.dim_IntList(view_125, [0], True);  view_125 = None
        view_126 = torch.ops.aten.view.default(sum_42, [128]);  sum_42 = None
        view_127 = torch.ops.aten.view.default(mm_17, [128, 4000, 16]);  mm_17 = None
        permute_121 = torch.ops.aten.permute.default(view_127, [1, 0, 2]);  view_127 = None
        bmm_11 = torch.ops.aten.bmm.default(permute_122, permute_121);  permute_122 = None
        bmm_12 = torch.ops.aten.bmm.default(permute_121, permute_123);  permute_121 = permute_123 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(gt_20, torch.float32);  gt_20 = None
        mul_215 = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.1111111111111112);  convert_element_type_6 = None
        mul_216 = torch.ops.aten.mul.Tensor(bmm_12, mul_215);  bmm_12 = mul_215 = None
        sub_26 = torch.ops.aten.sub.Tensor(baddbmm_5, amax_5);  baddbmm_5 = amax_5 = None
        exp_5 = torch.ops.aten.exp.default(sub_26);  sub_26 = None
        div_5 = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        mul_217 = torch.ops.aten.mul.Tensor(mul_216, div_5);  mul_216 = None
        sum_43 = torch.ops.aten.sum.dim_IntList(mul_217, [-1], True)
        neg_1 = torch.ops.aten.neg.default(div_5);  div_5 = None
        fma_1 = torch.ops.prims.fma.default(neg_1, sum_43, mul_217);  neg_1 = sum_43 = mul_217 = None
        bmm_13 = torch.ops.aten.bmm.default(fma_1, permute_124);  permute_124 = None
        bmm_14 = torch.ops.aten.bmm.default(permute_125, fma_1);  permute_125 = None
        permute_126 = torch.ops.aten.permute.default(bmm_14, [0, 2, 1]);  bmm_14 = None
        mul_218 = torch.ops.aten.mul.Tensor(bmm_13, 0.25);  bmm_13 = None
        add_99 = torch.ops.aten.add.Tensor(fma, fma_1);  fma = fma_1 = None
        permute_127 = torch.ops.aten.permute.default(bmm_11, [1, 0, 2]);  bmm_11 = None
        clone_47 = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None
        view_128 = torch.ops.aten.view.default(clone_47, [128, 500, 128]);  clone_47 = None
        permute_128 = torch.ops.aten.permute.default(permute_126, [1, 0, 2]);  permute_126 = None
        view_129 = torch.ops.aten.view.default(permute_128, [128, 500, 128]);  permute_128 = None
        permute_129 = torch.ops.aten.permute.default(mul_218, [1, 0, 2]);  mul_218 = None
        clone_48 = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
        view_130 = torch.ops.aten.view.default(clone_48, [128, 500, 128]);  clone_48 = None
        select_scatter_3 = torch.ops.aten.select_scatter.default(full_default, view_128, 0, 2);  view_128 = None
        select_scatter_4 = torch.ops.aten.select_scatter.default(full_default, view_129, 0, 1);  view_129 = None
        add_100 = torch.ops.aten.add.Tensor(select_scatter_3, select_scatter_4);  select_scatter_3 = select_scatter_4 = None
        select_scatter_5 = torch.ops.aten.select_scatter.default(full_default, view_130, 0, 0);  view_130 = None
        add_101 = torch.ops.aten.add.Tensor(add_100, select_scatter_5);  add_100 = select_scatter_5 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(add_101, 3);  add_101 = None
        permute_130 = torch.ops.aten.permute.default(unsqueeze_8, [3, 1, 2, 0, 4]);  unsqueeze_8 = None
        squeeze_8 = torch.ops.aten.squeeze.dim(permute_130, 0);  permute_130 = None
        clone_49 = torch.ops.aten.clone.default(squeeze_8, memory_format = torch.contiguous_format);  squeeze_8 = None
        view_131 = torch.ops.aten.view.default(clone_49, [128, 500, 384]);  clone_49 = None
        sum_44 = torch.ops.aten.sum.dim_IntList(view_131, [0, 1], True)
        view_132 = torch.ops.aten.view.default(sum_44, [384]);  sum_44 = None
        view_133 = torch.ops.aten.view.default(view_131, [64000, 384]);  view_131 = None
        permute_131 = torch.ops.aten.permute.default(view_133, [1, 0])
        mm_19 = torch.ops.aten.mm.default(permute_131, view_78);  permute_131 = view_78 = None
        permute_62 = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
        permute_133 = torch.ops.aten.permute.default(permute_62, [1, 0]);  permute_62 = None
        mm_20 = torch.ops.aten.mm.default(view_133, permute_133);  view_133 = permute_133 = None
        view_134 = torch.ops.aten.view.default(mm_20, [128, 500, 128]);  mm_20 = None
        permute_135 = torch.ops.aten.permute.default(view_134, [1, 0, 2]);  view_134 = None
        mul_220 = torch.ops.aten.mul.Tensor(permute_135, primals_86);  primals_86 = None
        mul_221 = torch.ops.aten.mul.Tensor(mul_220, 128)
        sum_45 = torch.ops.aten.sum.dim_IntList(mul_220, [2], True)
        mul_222 = torch.ops.aten.mul.Tensor(mul_220, mul_100);  mul_220 = None
        sum_46 = torch.ops.aten.sum.dim_IntList(mul_222, [2], True);  mul_222 = None
        mul_223 = torch.ops.aten.mul.Tensor(mul_100, sum_46);  sum_46 = None
        sub_57 = torch.ops.aten.sub.Tensor(mul_221, sum_45);  mul_221 = sum_45 = None
        sub_58 = torch.ops.aten.sub.Tensor(sub_57, mul_223);  sub_57 = mul_223 = None
        mul_224 = torch.ops.aten.mul.Tensor(div_14, sub_58);  div_14 = sub_58 = None
        mul_225 = torch.ops.aten.mul.Tensor(permute_135, mul_100);  mul_100 = None
        sum_47 = torch.ops.aten.sum.dim_IntList(mul_225, [0, 1]);  mul_225 = None
        sum_48 = torch.ops.aten.sum.dim_IntList(permute_135, [0, 1]);  permute_135 = None
        add_102 = torch.ops.aten.add.Tensor(add_98, mul_224);  add_98 = mul_224 = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(gt_19, torch.float32);  gt_19 = None
        mul_226 = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
        mul_227 = torch.ops.aten.mul.Tensor(add_102, mul_226);  mul_226 = None
        view_135 = torch.ops.aten.view.default(mul_227, [64000, 128]);  mul_227 = None
        permute_60 = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
        permute_136 = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
        mm_21 = torch.ops.aten.mm.default(view_135, permute_136);  permute_136 = None
        permute_137 = torch.ops.aten.permute.default(view_135, [1, 0])
        mm_22 = torch.ops.aten.mm.default(permute_137, view_76);  permute_137 = view_76 = None
        sum_49 = torch.ops.aten.sum.dim_IntList(view_135, [0], True);  view_135 = None
        view_136 = torch.ops.aten.view.default(sum_49, [128]);  sum_49 = None
        view_137 = torch.ops.aten.view.default(mm_21, [500, 128, 512]);  mm_21 = None
        mul_229 = torch.ops.aten.mul.Tensor(view_137, primals_82);  primals_82 = None
        mul_230 = torch.ops.aten.mul.Tensor(mul_229, 512)
        sum_50 = torch.ops.aten.sum.dim_IntList(mul_229, [2], True)
        view_75 = torch.ops.aten.view.default(addmm_14, [500, 128, 512]);  addmm_14 = None
        mul_91 = torch.ops.aten.mul.Tensor(view_75, 0.5)
        mul_92 = torch.ops.aten.mul.Tensor(view_75, 0.7071067811865476)
        erf_4 = torch.ops.aten.erf.default(mul_92);  mul_92 = None
        add_61 = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_93 = torch.ops.aten.mul.Tensor(mul_91, add_61);  mul_91 = None
        mul_94 = torch.ops.aten.mul.Tensor(gt_18, mul_93);  mul_93 = None
        mul_95 = torch.ops.aten.mul.Tensor(mul_94, 1.1111111111111112);  mul_94 = None
        sub_24 = torch.ops.aten.sub.Tensor(mul_95, getitem_39);  mul_95 = getitem_39 = None
        mul_96 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_19);  sub_24 = None
        mul_231 = torch.ops.aten.mul.Tensor(mul_229, mul_96);  mul_229 = None
        sum_51 = torch.ops.aten.sum.dim_IntList(mul_231, [2], True);  mul_231 = None
        mul_232 = torch.ops.aten.mul.Tensor(mul_96, sum_51);  sum_51 = None
        sub_60 = torch.ops.aten.sub.Tensor(mul_230, sum_50);  mul_230 = sum_50 = None
        sub_61 = torch.ops.aten.sub.Tensor(sub_60, mul_232);  sub_60 = mul_232 = None
        div_15 = torch.ops.aten.div.Tensor(rsqrt_19, 512);  rsqrt_19 = None
        mul_233 = torch.ops.aten.mul.Tensor(div_15, sub_61);  div_15 = sub_61 = None
        mul_234 = torch.ops.aten.mul.Tensor(view_137, mul_96);  mul_96 = None
        sum_52 = torch.ops.aten.sum.dim_IntList(mul_234, [0, 1]);  mul_234 = None
        sum_53 = torch.ops.aten.sum.dim_IntList(view_137, [0, 1]);  view_137 = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(gt_18, torch.float32);  gt_18 = None
        mul_235 = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
        mul_236 = torch.ops.aten.mul.Tensor(mul_233, mul_235);  mul_233 = mul_235 = None
        mul_238 = torch.ops.aten.mul.Tensor(add_61, 0.5);  add_61 = None
        mul_239 = torch.ops.aten.mul.Tensor(view_75, view_75)
        mul_240 = torch.ops.aten.mul.Tensor(mul_239, -0.5);  mul_239 = None
        exp_9 = torch.ops.aten.exp.default(mul_240);  mul_240 = None
        mul_241 = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
        mul_242 = torch.ops.aten.mul.Tensor(view_75, mul_241);  view_75 = mul_241 = None
        add_104 = torch.ops.aten.add.Tensor(mul_238, mul_242);  mul_238 = mul_242 = None
        mul_243 = torch.ops.aten.mul.Tensor(mul_236, add_104);  mul_236 = add_104 = None
        view_138 = torch.ops.aten.view.default(mul_243, [64000, 512]);  mul_243 = None
        permute_59 = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
        permute_140 = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
        mm_23 = torch.ops.aten.mm.default(view_138, permute_140);  permute_140 = None
        permute_141 = torch.ops.aten.permute.default(view_138, [1, 0])
        mm_24 = torch.ops.aten.mm.default(permute_141, view_74);  permute_141 = view_74 = None
        sum_54 = torch.ops.aten.sum.dim_IntList(view_138, [0], True);  view_138 = None
        view_139 = torch.ops.aten.view.default(sum_54, [512]);  sum_54 = None
        view_140 = torch.ops.aten.view.default(mm_23, [500, 128, 128]);  mm_23 = None
        mul_245 = torch.ops.aten.mul.Tensor(view_140, primals_78);  primals_78 = None
        mul_246 = torch.ops.aten.mul.Tensor(mul_245, 128)
        sum_55 = torch.ops.aten.sum.dim_IntList(mul_245, [2], True)
        mul_247 = torch.ops.aten.mul.Tensor(mul_245, mul_89);  mul_245 = None
        sum_56 = torch.ops.aten.sum.dim_IntList(mul_247, [2], True);  mul_247 = None
        mul_248 = torch.ops.aten.mul.Tensor(mul_89, sum_56);  sum_56 = None
        sub_63 = torch.ops.aten.sub.Tensor(mul_246, sum_55);  mul_246 = sum_55 = None
        sub_64 = torch.ops.aten.sub.Tensor(sub_63, mul_248);  sub_63 = mul_248 = None
        mul_249 = torch.ops.aten.mul.Tensor(div_16, sub_64);  div_16 = sub_64 = None
        mul_250 = torch.ops.aten.mul.Tensor(view_140, mul_89);  mul_89 = None
        sum_57 = torch.ops.aten.sum.dim_IntList(mul_250, [0, 1]);  mul_250 = None
        sum_58 = torch.ops.aten.sum.dim_IntList(view_140, [0, 1]);  view_140 = None
        add_105 = torch.ops.aten.add.Tensor(add_102, mul_249);  add_102 = mul_249 = None
        convert_element_type_9 = torch.ops.prims.convert_element_type.default(gt_17, torch.float32);  gt_17 = None
        mul_251 = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
        mul_252 = torch.ops.aten.mul.Tensor(add_105, mul_251);  mul_251 = None
        view_72 = torch.ops.aten.view.default(addmm_13, [128, 500, 128]);  addmm_13 = None
        permute_58 = torch.ops.aten.permute.default(view_72, [1, 0, 2]);  view_72 = None
        sub_65 = torch.ops.aten.sub.Tensor(permute_58, getitem_35);  permute_58 = getitem_35 = None
        mul_253 = torch.ops.aten.mul.Tensor(sub_65, rsqrt_17);  sub_65 = None
        mul_254 = torch.ops.aten.mul.Tensor(mul_252, primals_76);  primals_76 = None
        mul_255 = torch.ops.aten.mul.Tensor(mul_254, 128)
        sum_59 = torch.ops.aten.sum.dim_IntList(mul_254, [2], True)
        mul_256 = torch.ops.aten.mul.Tensor(mul_254, mul_253);  mul_254 = None
        sum_60 = torch.ops.aten.sum.dim_IntList(mul_256, [2], True);  mul_256 = None
        mul_257 = torch.ops.aten.mul.Tensor(mul_253, sum_60);  sum_60 = None
        sub_66 = torch.ops.aten.sub.Tensor(mul_255, sum_59);  mul_255 = sum_59 = None
        sub_67 = torch.ops.aten.sub.Tensor(sub_66, mul_257);  sub_66 = mul_257 = None
        div_17 = torch.ops.aten.div.Tensor(rsqrt_17, 128);  rsqrt_17 = None
        mul_258 = torch.ops.aten.mul.Tensor(div_17, sub_67);  div_17 = sub_67 = None
        mul_259 = torch.ops.aten.mul.Tensor(mul_252, mul_253);  mul_253 = None
        sum_61 = torch.ops.aten.sum.dim_IntList(mul_259, [0, 1]);  mul_259 = None
        sum_62 = torch.ops.aten.sum.dim_IntList(mul_252, [0, 1]);  mul_252 = None
        permute_144 = torch.ops.aten.permute.default(mul_258, [1, 0, 2]);  mul_258 = None
        clone_53 = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
        view_141 = torch.ops.aten.view.default(clone_53, [64000, 128]);  clone_53 = None
        permute_57 = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
        permute_145 = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        mm_25 = torch.ops.aten.mm.default(view_141, permute_145);  permute_145 = None
        permute_146 = torch.ops.aten.permute.default(view_141, [1, 0])
        mm_26 = torch.ops.aten.mm.default(permute_146, view_71);  permute_146 = view_71 = None
        sum_63 = torch.ops.aten.sum.dim_IntList(view_141, [0], True);  view_141 = None
        view_142 = torch.ops.aten.view.default(sum_63, [128]);  sum_63 = None
        view_143 = torch.ops.aten.view.default(mm_25, [128, 4000, 16]);  mm_25 = None
        permute_149 = torch.ops.aten.permute.default(view_143, [1, 0, 2]);  view_143 = None
        bmm_15 = torch.ops.aten.bmm.default(permute_150, permute_149);  permute_150 = None
        bmm_16 = torch.ops.aten.bmm.default(permute_149, permute_151);  permute_149 = permute_151 = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(gt_16, torch.float32);  gt_16 = None
        mul_260 = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
        mul_261 = torch.ops.aten.mul.Tensor(bmm_16, mul_260);  bmm_16 = mul_260 = None
        sub_21 = torch.ops.aten.sub.Tensor(baddbmm_4, amax_4);  baddbmm_4 = amax_4 = None
        exp_4 = torch.ops.aten.exp.default(sub_21);  sub_21 = None
        div_4 = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        mul_262 = torch.ops.aten.mul.Tensor(mul_261, div_4);  mul_261 = None
        sum_64 = torch.ops.aten.sum.dim_IntList(mul_262, [-1], True)
        neg_2 = torch.ops.aten.neg.default(div_4);  div_4 = None
        fma_2 = torch.ops.prims.fma.default(neg_2, sum_64, mul_262);  neg_2 = sum_64 = mul_262 = None
        bmm_17 = torch.ops.aten.bmm.default(fma_2, permute_152);  permute_152 = None
        bmm_18 = torch.ops.aten.bmm.default(permute_153, fma_2);  permute_153 = None
        permute_154 = torch.ops.aten.permute.default(bmm_18, [0, 2, 1]);  bmm_18 = None
        mul_263 = torch.ops.aten.mul.Tensor(bmm_17, 0.25);  bmm_17 = None
        add_106 = torch.ops.aten.add.Tensor(add_99, fma_2);  add_99 = fma_2 = None
        permute_155 = torch.ops.aten.permute.default(bmm_15, [1, 0, 2]);  bmm_15 = None
        clone_55 = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
        view_144 = torch.ops.aten.view.default(clone_55, [128, 500, 128]);  clone_55 = None
        permute_156 = torch.ops.aten.permute.default(permute_154, [1, 0, 2]);  permute_154 = None
        view_145 = torch.ops.aten.view.default(permute_156, [128, 500, 128]);  permute_156 = None
        permute_157 = torch.ops.aten.permute.default(mul_263, [1, 0, 2]);  mul_263 = None
        clone_56 = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format);  permute_157 = None
        view_146 = torch.ops.aten.view.default(clone_56, [128, 500, 128]);  clone_56 = None
        select_scatter_6 = torch.ops.aten.select_scatter.default(full_default, view_144, 0, 2);  view_144 = None
        select_scatter_7 = torch.ops.aten.select_scatter.default(full_default, view_145, 0, 1);  view_145 = None
        add_107 = torch.ops.aten.add.Tensor(select_scatter_6, select_scatter_7);  select_scatter_6 = select_scatter_7 = None
        select_scatter_8 = torch.ops.aten.select_scatter.default(full_default, view_146, 0, 0);  view_146 = None
        add_108 = torch.ops.aten.add.Tensor(add_107, select_scatter_8);  add_107 = select_scatter_8 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(add_108, 3);  add_108 = None
        permute_158 = torch.ops.aten.permute.default(unsqueeze_9, [3, 1, 2, 0, 4]);  unsqueeze_9 = None
        squeeze_9 = torch.ops.aten.squeeze.dim(permute_158, 0);  permute_158 = None
        clone_57 = torch.ops.aten.clone.default(squeeze_9, memory_format = torch.contiguous_format);  squeeze_9 = None
        view_147 = torch.ops.aten.view.default(clone_57, [128, 500, 384]);  clone_57 = None
        sum_65 = torch.ops.aten.sum.dim_IntList(view_147, [0, 1], True)
        view_148 = torch.ops.aten.view.default(sum_65, [384]);  sum_65 = None
        view_149 = torch.ops.aten.view.default(view_147, [64000, 384]);  view_147 = None
        permute_159 = torch.ops.aten.permute.default(view_149, [1, 0])
        mm_27 = torch.ops.aten.mm.default(permute_159, view_63);  permute_159 = view_63 = None
        permute_50 = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
        permute_161 = torch.ops.aten.permute.default(permute_50, [1, 0]);  permute_50 = None
        mm_28 = torch.ops.aten.mm.default(view_149, permute_161);  view_149 = permute_161 = None
        view_150 = torch.ops.aten.view.default(mm_28, [128, 500, 128]);  mm_28 = None
        permute_163 = torch.ops.aten.permute.default(view_150, [1, 0, 2]);  view_150 = None
        mul_265 = torch.ops.aten.mul.Tensor(permute_163, primals_70);  primals_70 = None
        mul_266 = torch.ops.aten.mul.Tensor(mul_265, 128)
        sum_66 = torch.ops.aten.sum.dim_IntList(mul_265, [2], True)
        mul_267 = torch.ops.aten.mul.Tensor(mul_265, mul_80);  mul_265 = None
        sum_67 = torch.ops.aten.sum.dim_IntList(mul_267, [2], True);  mul_267 = None
        mul_268 = torch.ops.aten.mul.Tensor(mul_80, sum_67);  sum_67 = None
        sub_69 = torch.ops.aten.sub.Tensor(mul_266, sum_66);  mul_266 = sum_66 = None
        sub_70 = torch.ops.aten.sub.Tensor(sub_69, mul_268);  sub_69 = mul_268 = None
        mul_269 = torch.ops.aten.mul.Tensor(div_18, sub_70);  div_18 = sub_70 = None
        mul_270 = torch.ops.aten.mul.Tensor(permute_163, mul_80);  mul_80 = None
        sum_68 = torch.ops.aten.sum.dim_IntList(mul_270, [0, 1]);  mul_270 = None
        sum_69 = torch.ops.aten.sum.dim_IntList(permute_163, [0, 1]);  permute_163 = None
        add_109 = torch.ops.aten.add.Tensor(add_105, mul_269);  add_105 = mul_269 = None
        convert_element_type_11 = torch.ops.prims.convert_element_type.default(gt_15, torch.float32);  gt_15 = None
        mul_271 = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
        mul_272 = torch.ops.aten.mul.Tensor(add_109, mul_271);  mul_271 = None
        view_151 = torch.ops.aten.view.default(mul_272, [64000, 128]);  mul_272 = None
        permute_48 = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
        permute_164 = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
        mm_29 = torch.ops.aten.mm.default(view_151, permute_164);  permute_164 = None
        permute_165 = torch.ops.aten.permute.default(view_151, [1, 0])
        mm_30 = torch.ops.aten.mm.default(permute_165, view_61);  permute_165 = view_61 = None
        sum_70 = torch.ops.aten.sum.dim_IntList(view_151, [0], True);  view_151 = None
        view_152 = torch.ops.aten.view.default(sum_70, [128]);  sum_70 = None
        view_153 = torch.ops.aten.view.default(mm_29, [500, 128, 512]);  mm_29 = None
        mul_274 = torch.ops.aten.mul.Tensor(view_153, primals_66);  primals_66 = None
        mul_275 = torch.ops.aten.mul.Tensor(mul_274, 512)
        sum_71 = torch.ops.aten.sum.dim_IntList(mul_274, [2], True)
        view_60 = torch.ops.aten.view.default(addmm_11, [500, 128, 512]);  addmm_11 = None
        mul_71 = torch.ops.aten.mul.Tensor(view_60, 0.5)
        mul_72 = torch.ops.aten.mul.Tensor(view_60, 0.7071067811865476)
        erf_3 = torch.ops.aten.erf.default(mul_72);  mul_72 = None
        add_48 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_73 = torch.ops.aten.mul.Tensor(mul_71, add_48);  mul_71 = None
        mul_74 = torch.ops.aten.mul.Tensor(gt_14, mul_73);  mul_73 = None
        mul_75 = torch.ops.aten.mul.Tensor(mul_74, 1.1111111111111112);  mul_74 = None
        sub_19 = torch.ops.aten.sub.Tensor(mul_75, getitem_31);  mul_75 = getitem_31 = None
        mul_76 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_15);  sub_19 = None
        mul_276 = torch.ops.aten.mul.Tensor(mul_274, mul_76);  mul_274 = None
        sum_72 = torch.ops.aten.sum.dim_IntList(mul_276, [2], True);  mul_276 = None
        mul_277 = torch.ops.aten.mul.Tensor(mul_76, sum_72);  sum_72 = None
        sub_72 = torch.ops.aten.sub.Tensor(mul_275, sum_71);  mul_275 = sum_71 = None
        sub_73 = torch.ops.aten.sub.Tensor(sub_72, mul_277);  sub_72 = mul_277 = None
        div_19 = torch.ops.aten.div.Tensor(rsqrt_15, 512);  rsqrt_15 = None
        mul_278 = torch.ops.aten.mul.Tensor(div_19, sub_73);  div_19 = sub_73 = None
        mul_279 = torch.ops.aten.mul.Tensor(view_153, mul_76);  mul_76 = None
        sum_73 = torch.ops.aten.sum.dim_IntList(mul_279, [0, 1]);  mul_279 = None
        sum_74 = torch.ops.aten.sum.dim_IntList(view_153, [0, 1]);  view_153 = None
        convert_element_type_12 = torch.ops.prims.convert_element_type.default(gt_14, torch.float32);  gt_14 = None
        mul_280 = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
        mul_281 = torch.ops.aten.mul.Tensor(mul_278, mul_280);  mul_278 = mul_280 = None
        mul_283 = torch.ops.aten.mul.Tensor(add_48, 0.5);  add_48 = None
        mul_284 = torch.ops.aten.mul.Tensor(view_60, view_60)
        mul_285 = torch.ops.aten.mul.Tensor(mul_284, -0.5);  mul_284 = None
        exp_10 = torch.ops.aten.exp.default(mul_285);  mul_285 = None
        mul_286 = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
        mul_287 = torch.ops.aten.mul.Tensor(view_60, mul_286);  view_60 = mul_286 = None
        add_111 = torch.ops.aten.add.Tensor(mul_283, mul_287);  mul_283 = mul_287 = None
        mul_288 = torch.ops.aten.mul.Tensor(mul_281, add_111);  mul_281 = add_111 = None
        view_154 = torch.ops.aten.view.default(mul_288, [64000, 512]);  mul_288 = None
        permute_47 = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
        permute_168 = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
        mm_31 = torch.ops.aten.mm.default(view_154, permute_168);  permute_168 = None
        permute_169 = torch.ops.aten.permute.default(view_154, [1, 0])
        mm_32 = torch.ops.aten.mm.default(permute_169, view_59);  permute_169 = view_59 = None
        sum_75 = torch.ops.aten.sum.dim_IntList(view_154, [0], True);  view_154 = None
        view_155 = torch.ops.aten.view.default(sum_75, [512]);  sum_75 = None
        view_156 = torch.ops.aten.view.default(mm_31, [500, 128, 128]);  mm_31 = None
        mul_290 = torch.ops.aten.mul.Tensor(view_156, primals_62);  primals_62 = None
        mul_291 = torch.ops.aten.mul.Tensor(mul_290, 128)
        sum_76 = torch.ops.aten.sum.dim_IntList(mul_290, [2], True)
        mul_292 = torch.ops.aten.mul.Tensor(mul_290, mul_69);  mul_290 = None
        sum_77 = torch.ops.aten.sum.dim_IntList(mul_292, [2], True);  mul_292 = None
        mul_293 = torch.ops.aten.mul.Tensor(mul_69, sum_77);  sum_77 = None
        sub_75 = torch.ops.aten.sub.Tensor(mul_291, sum_76);  mul_291 = sum_76 = None
        sub_76 = torch.ops.aten.sub.Tensor(sub_75, mul_293);  sub_75 = mul_293 = None
        mul_294 = torch.ops.aten.mul.Tensor(div_20, sub_76);  div_20 = sub_76 = None
        mul_295 = torch.ops.aten.mul.Tensor(view_156, mul_69);  mul_69 = None
        sum_78 = torch.ops.aten.sum.dim_IntList(mul_295, [0, 1]);  mul_295 = None
        sum_79 = torch.ops.aten.sum.dim_IntList(view_156, [0, 1]);  view_156 = None
        add_112 = torch.ops.aten.add.Tensor(add_109, mul_294);  add_109 = mul_294 = None
        convert_element_type_13 = torch.ops.prims.convert_element_type.default(gt_13, torch.float32);  gt_13 = None
        mul_296 = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
        mul_297 = torch.ops.aten.mul.Tensor(add_112, mul_296);  mul_296 = None
        view_57 = torch.ops.aten.view.default(addmm_10, [128, 500, 128]);  addmm_10 = None
        permute_46 = torch.ops.aten.permute.default(view_57, [1, 0, 2]);  view_57 = None
        sub_77 = torch.ops.aten.sub.Tensor(permute_46, getitem_27);  permute_46 = getitem_27 = None
        mul_298 = torch.ops.aten.mul.Tensor(sub_77, rsqrt_13);  sub_77 = None
        mul_299 = torch.ops.aten.mul.Tensor(mul_297, primals_60);  primals_60 = None
        mul_300 = torch.ops.aten.mul.Tensor(mul_299, 128)
        sum_80 = torch.ops.aten.sum.dim_IntList(mul_299, [2], True)
        mul_301 = torch.ops.aten.mul.Tensor(mul_299, mul_298);  mul_299 = None
        sum_81 = torch.ops.aten.sum.dim_IntList(mul_301, [2], True);  mul_301 = None
        mul_302 = torch.ops.aten.mul.Tensor(mul_298, sum_81);  sum_81 = None
        sub_78 = torch.ops.aten.sub.Tensor(mul_300, sum_80);  mul_300 = sum_80 = None
        sub_79 = torch.ops.aten.sub.Tensor(sub_78, mul_302);  sub_78 = mul_302 = None
        div_21 = torch.ops.aten.div.Tensor(rsqrt_13, 128);  rsqrt_13 = None
        mul_303 = torch.ops.aten.mul.Tensor(div_21, sub_79);  div_21 = sub_79 = None
        mul_304 = torch.ops.aten.mul.Tensor(mul_297, mul_298);  mul_298 = None
        sum_82 = torch.ops.aten.sum.dim_IntList(mul_304, [0, 1]);  mul_304 = None
        sum_83 = torch.ops.aten.sum.dim_IntList(mul_297, [0, 1]);  mul_297 = None
        permute_172 = torch.ops.aten.permute.default(mul_303, [1, 0, 2]);  mul_303 = None
        clone_61 = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
        view_157 = torch.ops.aten.view.default(clone_61, [64000, 128]);  clone_61 = None
        permute_45 = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
        permute_173 = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        mm_33 = torch.ops.aten.mm.default(view_157, permute_173);  permute_173 = None
        permute_174 = torch.ops.aten.permute.default(view_157, [1, 0])
        mm_34 = torch.ops.aten.mm.default(permute_174, view_56);  permute_174 = view_56 = None
        sum_84 = torch.ops.aten.sum.dim_IntList(view_157, [0], True);  view_157 = None
        view_158 = torch.ops.aten.view.default(sum_84, [128]);  sum_84 = None
        view_159 = torch.ops.aten.view.default(mm_33, [128, 4000, 16]);  mm_33 = None
        permute_177 = torch.ops.aten.permute.default(view_159, [1, 0, 2]);  view_159 = None
        bmm_19 = torch.ops.aten.bmm.default(permute_178, permute_177);  permute_178 = None
        bmm_20 = torch.ops.aten.bmm.default(permute_177, permute_179);  permute_177 = permute_179 = None
        convert_element_type_14 = torch.ops.prims.convert_element_type.default(gt_12, torch.float32);  gt_12 = None
        mul_305 = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
        mul_306 = torch.ops.aten.mul.Tensor(bmm_20, mul_305);  bmm_20 = mul_305 = None
        sub_16 = torch.ops.aten.sub.Tensor(baddbmm_3, amax_3);  baddbmm_3 = amax_3 = None
        exp_3 = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        div_3 = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        mul_307 = torch.ops.aten.mul.Tensor(mul_306, div_3);  mul_306 = None
        sum_85 = torch.ops.aten.sum.dim_IntList(mul_307, [-1], True)
        neg_3 = torch.ops.aten.neg.default(div_3);  div_3 = None
        fma_3 = torch.ops.prims.fma.default(neg_3, sum_85, mul_307);  neg_3 = sum_85 = mul_307 = None
        bmm_21 = torch.ops.aten.bmm.default(fma_3, permute_180);  permute_180 = None
        bmm_22 = torch.ops.aten.bmm.default(permute_181, fma_3);  permute_181 = None
        permute_182 = torch.ops.aten.permute.default(bmm_22, [0, 2, 1]);  bmm_22 = None
        mul_308 = torch.ops.aten.mul.Tensor(bmm_21, 0.25);  bmm_21 = None
        add_113 = torch.ops.aten.add.Tensor(add_106, fma_3);  add_106 = fma_3 = None
        permute_183 = torch.ops.aten.permute.default(bmm_19, [1, 0, 2]);  bmm_19 = None
        clone_63 = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
        view_160 = torch.ops.aten.view.default(clone_63, [128, 500, 128]);  clone_63 = None
        permute_184 = torch.ops.aten.permute.default(permute_182, [1, 0, 2]);  permute_182 = None
        view_161 = torch.ops.aten.view.default(permute_184, [128, 500, 128]);  permute_184 = None
        permute_185 = torch.ops.aten.permute.default(mul_308, [1, 0, 2]);  mul_308 = None
        clone_64 = torch.ops.aten.clone.default(permute_185, memory_format = torch.contiguous_format);  permute_185 = None
        view_162 = torch.ops.aten.view.default(clone_64, [128, 500, 128]);  clone_64 = None
        select_scatter_9 = torch.ops.aten.select_scatter.default(full_default, view_160, 0, 2);  view_160 = None
        select_scatter_10 = torch.ops.aten.select_scatter.default(full_default, view_161, 0, 1);  view_161 = None
        add_114 = torch.ops.aten.add.Tensor(select_scatter_9, select_scatter_10);  select_scatter_9 = select_scatter_10 = None
        select_scatter_11 = torch.ops.aten.select_scatter.default(full_default, view_162, 0, 0);  view_162 = None
        add_115 = torch.ops.aten.add.Tensor(add_114, select_scatter_11);  add_114 = select_scatter_11 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(add_115, 3);  add_115 = None
        permute_186 = torch.ops.aten.permute.default(unsqueeze_10, [3, 1, 2, 0, 4]);  unsqueeze_10 = None
        squeeze_10 = torch.ops.aten.squeeze.dim(permute_186, 0);  permute_186 = None
        clone_65 = torch.ops.aten.clone.default(squeeze_10, memory_format = torch.contiguous_format);  squeeze_10 = None
        view_163 = torch.ops.aten.view.default(clone_65, [128, 500, 384]);  clone_65 = None
        sum_86 = torch.ops.aten.sum.dim_IntList(view_163, [0, 1], True)
        view_164 = torch.ops.aten.view.default(sum_86, [384]);  sum_86 = None
        view_165 = torch.ops.aten.view.default(view_163, [64000, 384]);  view_163 = None
        permute_187 = torch.ops.aten.permute.default(view_165, [1, 0])
        mm_35 = torch.ops.aten.mm.default(permute_187, view_48);  permute_187 = view_48 = None
        permute_38 = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
        permute_189 = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
        mm_36 = torch.ops.aten.mm.default(view_165, permute_189);  view_165 = permute_189 = None
        view_166 = torch.ops.aten.view.default(mm_36, [128, 500, 128]);  mm_36 = None
        permute_191 = torch.ops.aten.permute.default(view_166, [1, 0, 2]);  view_166 = None
        mul_310 = torch.ops.aten.mul.Tensor(permute_191, primals_54);  primals_54 = None
        mul_311 = torch.ops.aten.mul.Tensor(mul_310, 128)
        sum_87 = torch.ops.aten.sum.dim_IntList(mul_310, [2], True)
        mul_312 = torch.ops.aten.mul.Tensor(mul_310, mul_60);  mul_310 = None
        sum_88 = torch.ops.aten.sum.dim_IntList(mul_312, [2], True);  mul_312 = None
        mul_313 = torch.ops.aten.mul.Tensor(mul_60, sum_88);  sum_88 = None
        sub_81 = torch.ops.aten.sub.Tensor(mul_311, sum_87);  mul_311 = sum_87 = None
        sub_82 = torch.ops.aten.sub.Tensor(sub_81, mul_313);  sub_81 = mul_313 = None
        mul_314 = torch.ops.aten.mul.Tensor(div_22, sub_82);  div_22 = sub_82 = None
        mul_315 = torch.ops.aten.mul.Tensor(permute_191, mul_60);  mul_60 = None
        sum_89 = torch.ops.aten.sum.dim_IntList(mul_315, [0, 1]);  mul_315 = None
        sum_90 = torch.ops.aten.sum.dim_IntList(permute_191, [0, 1]);  permute_191 = None
        add_116 = torch.ops.aten.add.Tensor(add_112, mul_314);  add_112 = mul_314 = None
        convert_element_type_15 = torch.ops.prims.convert_element_type.default(gt_11, torch.float32);  gt_11 = None
        mul_316 = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
        mul_317 = torch.ops.aten.mul.Tensor(add_116, mul_316);  mul_316 = None
        view_167 = torch.ops.aten.view.default(mul_317, [64000, 128]);  mul_317 = None
        permute_36 = torch.ops.aten.permute.default(primals_52, [1, 0]);  primals_52 = None
        permute_192 = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
        mm_37 = torch.ops.aten.mm.default(view_167, permute_192);  permute_192 = None
        permute_193 = torch.ops.aten.permute.default(view_167, [1, 0])
        mm_38 = torch.ops.aten.mm.default(permute_193, view_46);  permute_193 = view_46 = None
        sum_91 = torch.ops.aten.sum.dim_IntList(view_167, [0], True);  view_167 = None
        view_168 = torch.ops.aten.view.default(sum_91, [128]);  sum_91 = None
        view_169 = torch.ops.aten.view.default(mm_37, [500, 128, 512]);  mm_37 = None
        mul_319 = torch.ops.aten.mul.Tensor(view_169, primals_50);  primals_50 = None
        mul_320 = torch.ops.aten.mul.Tensor(mul_319, 512)
        sum_92 = torch.ops.aten.sum.dim_IntList(mul_319, [2], True)
        view_45 = torch.ops.aten.view.default(addmm_8, [500, 128, 512]);  addmm_8 = None
        mul_51 = torch.ops.aten.mul.Tensor(view_45, 0.5)
        mul_52 = torch.ops.aten.mul.Tensor(view_45, 0.7071067811865476)
        erf_2 = torch.ops.aten.erf.default(mul_52);  mul_52 = None
        add_35 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_51, add_35);  mul_51 = None
        mul_54 = torch.ops.aten.mul.Tensor(gt_10, mul_53);  mul_53 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul_54, 1.1111111111111112);  mul_54 = None
        sub_14 = torch.ops.aten.sub.Tensor(mul_55, getitem_23);  mul_55 = getitem_23 = None
        mul_56 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_11);  sub_14 = None
        mul_321 = torch.ops.aten.mul.Tensor(mul_319, mul_56);  mul_319 = None
        sum_93 = torch.ops.aten.sum.dim_IntList(mul_321, [2], True);  mul_321 = None
        mul_322 = torch.ops.aten.mul.Tensor(mul_56, sum_93);  sum_93 = None
        sub_84 = torch.ops.aten.sub.Tensor(mul_320, sum_92);  mul_320 = sum_92 = None
        sub_85 = torch.ops.aten.sub.Tensor(sub_84, mul_322);  sub_84 = mul_322 = None
        div_23 = torch.ops.aten.div.Tensor(rsqrt_11, 512);  rsqrt_11 = None
        mul_323 = torch.ops.aten.mul.Tensor(div_23, sub_85);  div_23 = sub_85 = None
        mul_324 = torch.ops.aten.mul.Tensor(view_169, mul_56);  mul_56 = None
        sum_94 = torch.ops.aten.sum.dim_IntList(mul_324, [0, 1]);  mul_324 = None
        sum_95 = torch.ops.aten.sum.dim_IntList(view_169, [0, 1]);  view_169 = None
        convert_element_type_16 = torch.ops.prims.convert_element_type.default(gt_10, torch.float32);  gt_10 = None
        mul_325 = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
        mul_326 = torch.ops.aten.mul.Tensor(mul_323, mul_325);  mul_323 = mul_325 = None
        mul_328 = torch.ops.aten.mul.Tensor(add_35, 0.5);  add_35 = None
        mul_329 = torch.ops.aten.mul.Tensor(view_45, view_45)
        mul_330 = torch.ops.aten.mul.Tensor(mul_329, -0.5);  mul_329 = None
        exp_11 = torch.ops.aten.exp.default(mul_330);  mul_330 = None
        mul_331 = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
        mul_332 = torch.ops.aten.mul.Tensor(view_45, mul_331);  view_45 = mul_331 = None
        add_118 = torch.ops.aten.add.Tensor(mul_328, mul_332);  mul_328 = mul_332 = None
        mul_333 = torch.ops.aten.mul.Tensor(mul_326, add_118);  mul_326 = add_118 = None
        view_170 = torch.ops.aten.view.default(mul_333, [64000, 512]);  mul_333 = None
        permute_35 = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
        permute_196 = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        mm_39 = torch.ops.aten.mm.default(view_170, permute_196);  permute_196 = None
        permute_197 = torch.ops.aten.permute.default(view_170, [1, 0])
        mm_40 = torch.ops.aten.mm.default(permute_197, view_44);  permute_197 = view_44 = None
        sum_96 = torch.ops.aten.sum.dim_IntList(view_170, [0], True);  view_170 = None
        view_171 = torch.ops.aten.view.default(sum_96, [512]);  sum_96 = None
        view_172 = torch.ops.aten.view.default(mm_39, [500, 128, 128]);  mm_39 = None
        mul_335 = torch.ops.aten.mul.Tensor(view_172, primals_46);  primals_46 = None
        mul_336 = torch.ops.aten.mul.Tensor(mul_335, 128)
        sum_97 = torch.ops.aten.sum.dim_IntList(mul_335, [2], True)
        mul_337 = torch.ops.aten.mul.Tensor(mul_335, mul_49);  mul_335 = None
        sum_98 = torch.ops.aten.sum.dim_IntList(mul_337, [2], True);  mul_337 = None
        mul_338 = torch.ops.aten.mul.Tensor(mul_49, sum_98);  sum_98 = None
        sub_87 = torch.ops.aten.sub.Tensor(mul_336, sum_97);  mul_336 = sum_97 = None
        sub_88 = torch.ops.aten.sub.Tensor(sub_87, mul_338);  sub_87 = mul_338 = None
        mul_339 = torch.ops.aten.mul.Tensor(div_24, sub_88);  div_24 = sub_88 = None
        mul_340 = torch.ops.aten.mul.Tensor(view_172, mul_49);  mul_49 = None
        sum_99 = torch.ops.aten.sum.dim_IntList(mul_340, [0, 1]);  mul_340 = None
        sum_100 = torch.ops.aten.sum.dim_IntList(view_172, [0, 1]);  view_172 = None
        add_119 = torch.ops.aten.add.Tensor(add_116, mul_339);  add_116 = mul_339 = None
        convert_element_type_17 = torch.ops.prims.convert_element_type.default(gt_9, torch.float32);  gt_9 = None
        mul_341 = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
        mul_342 = torch.ops.aten.mul.Tensor(add_119, mul_341);  mul_341 = None
        view_42 = torch.ops.aten.view.default(addmm_7, [128, 500, 128]);  addmm_7 = None
        permute_34 = torch.ops.aten.permute.default(view_42, [1, 0, 2]);  view_42 = None
        sub_89 = torch.ops.aten.sub.Tensor(permute_34, getitem_19);  permute_34 = getitem_19 = None
        mul_343 = torch.ops.aten.mul.Tensor(sub_89, rsqrt_9);  sub_89 = None
        mul_344 = torch.ops.aten.mul.Tensor(mul_342, primals_44);  primals_44 = None
        mul_345 = torch.ops.aten.mul.Tensor(mul_344, 128)
        sum_101 = torch.ops.aten.sum.dim_IntList(mul_344, [2], True)
        mul_346 = torch.ops.aten.mul.Tensor(mul_344, mul_343);  mul_344 = None
        sum_102 = torch.ops.aten.sum.dim_IntList(mul_346, [2], True);  mul_346 = None
        mul_347 = torch.ops.aten.mul.Tensor(mul_343, sum_102);  sum_102 = None
        sub_90 = torch.ops.aten.sub.Tensor(mul_345, sum_101);  mul_345 = sum_101 = None
        sub_91 = torch.ops.aten.sub.Tensor(sub_90, mul_347);  sub_90 = mul_347 = None
        div_25 = torch.ops.aten.div.Tensor(rsqrt_9, 128);  rsqrt_9 = None
        mul_348 = torch.ops.aten.mul.Tensor(div_25, sub_91);  div_25 = sub_91 = None
        mul_349 = torch.ops.aten.mul.Tensor(mul_342, mul_343);  mul_343 = None
        sum_103 = torch.ops.aten.sum.dim_IntList(mul_349, [0, 1]);  mul_349 = None
        sum_104 = torch.ops.aten.sum.dim_IntList(mul_342, [0, 1]);  mul_342 = None
        permute_200 = torch.ops.aten.permute.default(mul_348, [1, 0, 2]);  mul_348 = None
        clone_69 = torch.ops.aten.clone.default(permute_200, memory_format = torch.contiguous_format);  permute_200 = None
        view_173 = torch.ops.aten.view.default(clone_69, [64000, 128]);  clone_69 = None
        permute_33 = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
        permute_201 = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        mm_41 = torch.ops.aten.mm.default(view_173, permute_201);  permute_201 = None
        permute_202 = torch.ops.aten.permute.default(view_173, [1, 0])
        mm_42 = torch.ops.aten.mm.default(permute_202, view_41);  permute_202 = view_41 = None
        sum_105 = torch.ops.aten.sum.dim_IntList(view_173, [0], True);  view_173 = None
        view_174 = torch.ops.aten.view.default(sum_105, [128]);  sum_105 = None
        view_175 = torch.ops.aten.view.default(mm_41, [128, 4000, 16]);  mm_41 = None
        permute_205 = torch.ops.aten.permute.default(view_175, [1, 0, 2]);  view_175 = None
        bmm_23 = torch.ops.aten.bmm.default(permute_206, permute_205);  permute_206 = None
        bmm_24 = torch.ops.aten.bmm.default(permute_205, permute_207);  permute_205 = permute_207 = None
        convert_element_type_18 = torch.ops.prims.convert_element_type.default(gt_8, torch.float32);  gt_8 = None
        mul_350 = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
        mul_351 = torch.ops.aten.mul.Tensor(bmm_24, mul_350);  bmm_24 = mul_350 = None
        sub_11 = torch.ops.aten.sub.Tensor(baddbmm_2, amax_2);  baddbmm_2 = amax_2 = None
        exp_2 = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        div_2 = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        mul_352 = torch.ops.aten.mul.Tensor(mul_351, div_2);  mul_351 = None
        sum_106 = torch.ops.aten.sum.dim_IntList(mul_352, [-1], True)
        neg_4 = torch.ops.aten.neg.default(div_2);  div_2 = None
        fma_4 = torch.ops.prims.fma.default(neg_4, sum_106, mul_352);  neg_4 = sum_106 = mul_352 = None
        bmm_25 = torch.ops.aten.bmm.default(fma_4, permute_208);  permute_208 = None
        bmm_26 = torch.ops.aten.bmm.default(permute_209, fma_4);  permute_209 = None
        permute_210 = torch.ops.aten.permute.default(bmm_26, [0, 2, 1]);  bmm_26 = None
        mul_353 = torch.ops.aten.mul.Tensor(bmm_25, 0.25);  bmm_25 = None
        add_120 = torch.ops.aten.add.Tensor(add_113, fma_4);  add_113 = fma_4 = None
        permute_211 = torch.ops.aten.permute.default(bmm_23, [1, 0, 2]);  bmm_23 = None
        clone_71 = torch.ops.aten.clone.default(permute_211, memory_format = torch.contiguous_format);  permute_211 = None
        view_176 = torch.ops.aten.view.default(clone_71, [128, 500, 128]);  clone_71 = None
        permute_212 = torch.ops.aten.permute.default(permute_210, [1, 0, 2]);  permute_210 = None
        view_177 = torch.ops.aten.view.default(permute_212, [128, 500, 128]);  permute_212 = None
        permute_213 = torch.ops.aten.permute.default(mul_353, [1, 0, 2]);  mul_353 = None
        clone_72 = torch.ops.aten.clone.default(permute_213, memory_format = torch.contiguous_format);  permute_213 = None
        view_178 = torch.ops.aten.view.default(clone_72, [128, 500, 128]);  clone_72 = None
        select_scatter_12 = torch.ops.aten.select_scatter.default(full_default, view_176, 0, 2);  view_176 = None
        select_scatter_13 = torch.ops.aten.select_scatter.default(full_default, view_177, 0, 1);  view_177 = None
        add_121 = torch.ops.aten.add.Tensor(select_scatter_12, select_scatter_13);  select_scatter_12 = select_scatter_13 = None
        select_scatter_14 = torch.ops.aten.select_scatter.default(full_default, view_178, 0, 0);  view_178 = None
        add_122 = torch.ops.aten.add.Tensor(add_121, select_scatter_14);  add_121 = select_scatter_14 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(add_122, 3);  add_122 = None
        permute_214 = torch.ops.aten.permute.default(unsqueeze_11, [3, 1, 2, 0, 4]);  unsqueeze_11 = None
        squeeze_11 = torch.ops.aten.squeeze.dim(permute_214, 0);  permute_214 = None
        clone_73 = torch.ops.aten.clone.default(squeeze_11, memory_format = torch.contiguous_format);  squeeze_11 = None
        view_179 = torch.ops.aten.view.default(clone_73, [128, 500, 384]);  clone_73 = None
        sum_107 = torch.ops.aten.sum.dim_IntList(view_179, [0, 1], True)
        view_180 = torch.ops.aten.view.default(sum_107, [384]);  sum_107 = None
        view_181 = torch.ops.aten.view.default(view_179, [64000, 384]);  view_179 = None
        permute_215 = torch.ops.aten.permute.default(view_181, [1, 0])
        mm_43 = torch.ops.aten.mm.default(permute_215, view_33);  permute_215 = view_33 = None
        permute_26 = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
        permute_217 = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
        mm_44 = torch.ops.aten.mm.default(view_181, permute_217);  view_181 = permute_217 = None
        view_182 = torch.ops.aten.view.default(mm_44, [128, 500, 128]);  mm_44 = None
        permute_219 = torch.ops.aten.permute.default(view_182, [1, 0, 2]);  view_182 = None
        mul_355 = torch.ops.aten.mul.Tensor(permute_219, primals_38);  primals_38 = None
        mul_356 = torch.ops.aten.mul.Tensor(mul_355, 128)
        sum_108 = torch.ops.aten.sum.dim_IntList(mul_355, [2], True)
        mul_357 = torch.ops.aten.mul.Tensor(mul_355, mul_40);  mul_355 = None
        sum_109 = torch.ops.aten.sum.dim_IntList(mul_357, [2], True);  mul_357 = None
        mul_358 = torch.ops.aten.mul.Tensor(mul_40, sum_109);  sum_109 = None
        sub_93 = torch.ops.aten.sub.Tensor(mul_356, sum_108);  mul_356 = sum_108 = None
        sub_94 = torch.ops.aten.sub.Tensor(sub_93, mul_358);  sub_93 = mul_358 = None
        mul_359 = torch.ops.aten.mul.Tensor(div_26, sub_94);  div_26 = sub_94 = None
        mul_360 = torch.ops.aten.mul.Tensor(permute_219, mul_40);  mul_40 = None
        sum_110 = torch.ops.aten.sum.dim_IntList(mul_360, [0, 1]);  mul_360 = None
        sum_111 = torch.ops.aten.sum.dim_IntList(permute_219, [0, 1]);  permute_219 = None
        add_123 = torch.ops.aten.add.Tensor(add_119, mul_359);  add_119 = mul_359 = None
        convert_element_type_19 = torch.ops.prims.convert_element_type.default(gt_7, torch.float32);  gt_7 = None
        mul_361 = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
        mul_362 = torch.ops.aten.mul.Tensor(add_123, mul_361);  mul_361 = None
        view_183 = torch.ops.aten.view.default(mul_362, [64000, 128]);  mul_362 = None
        permute_24 = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
        permute_220 = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        mm_45 = torch.ops.aten.mm.default(view_183, permute_220);  permute_220 = None
        permute_221 = torch.ops.aten.permute.default(view_183, [1, 0])
        mm_46 = torch.ops.aten.mm.default(permute_221, view_31);  permute_221 = view_31 = None
        sum_112 = torch.ops.aten.sum.dim_IntList(view_183, [0], True);  view_183 = None
        view_184 = torch.ops.aten.view.default(sum_112, [128]);  sum_112 = None
        view_185 = torch.ops.aten.view.default(mm_45, [500, 128, 512]);  mm_45 = None
        mul_364 = torch.ops.aten.mul.Tensor(view_185, primals_34);  primals_34 = None
        mul_365 = torch.ops.aten.mul.Tensor(mul_364, 512)
        sum_113 = torch.ops.aten.sum.dim_IntList(mul_364, [2], True)
        view_30 = torch.ops.aten.view.default(addmm_5, [500, 128, 512]);  addmm_5 = None
        mul_31 = torch.ops.aten.mul.Tensor(view_30, 0.5)
        mul_32 = torch.ops.aten.mul.Tensor(view_30, 0.7071067811865476)
        erf_1 = torch.ops.aten.erf.default(mul_32);  mul_32 = None
        add_22 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_33 = torch.ops.aten.mul.Tensor(mul_31, add_22);  mul_31 = None
        mul_34 = torch.ops.aten.mul.Tensor(gt_6, mul_33);  mul_33 = None
        mul_35 = torch.ops.aten.mul.Tensor(mul_34, 1.1111111111111112);  mul_34 = None
        sub_9 = torch.ops.aten.sub.Tensor(mul_35, getitem_15);  mul_35 = getitem_15 = None
        mul_36 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_7);  sub_9 = None
        mul_366 = torch.ops.aten.mul.Tensor(mul_364, mul_36);  mul_364 = None
        sum_114 = torch.ops.aten.sum.dim_IntList(mul_366, [2], True);  mul_366 = None
        mul_367 = torch.ops.aten.mul.Tensor(mul_36, sum_114);  sum_114 = None
        sub_96 = torch.ops.aten.sub.Tensor(mul_365, sum_113);  mul_365 = sum_113 = None
        sub_97 = torch.ops.aten.sub.Tensor(sub_96, mul_367);  sub_96 = mul_367 = None
        div_27 = torch.ops.aten.div.Tensor(rsqrt_7, 512);  rsqrt_7 = None
        mul_368 = torch.ops.aten.mul.Tensor(div_27, sub_97);  div_27 = sub_97 = None
        mul_369 = torch.ops.aten.mul.Tensor(view_185, mul_36);  mul_36 = None
        sum_115 = torch.ops.aten.sum.dim_IntList(mul_369, [0, 1]);  mul_369 = None
        sum_116 = torch.ops.aten.sum.dim_IntList(view_185, [0, 1]);  view_185 = None
        convert_element_type_20 = torch.ops.prims.convert_element_type.default(gt_6, torch.float32);  gt_6 = None
        mul_370 = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
        mul_371 = torch.ops.aten.mul.Tensor(mul_368, mul_370);  mul_368 = mul_370 = None
        mul_373 = torch.ops.aten.mul.Tensor(add_22, 0.5);  add_22 = None
        mul_374 = torch.ops.aten.mul.Tensor(view_30, view_30)
        mul_375 = torch.ops.aten.mul.Tensor(mul_374, -0.5);  mul_374 = None
        exp_12 = torch.ops.aten.exp.default(mul_375);  mul_375 = None
        mul_376 = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
        mul_377 = torch.ops.aten.mul.Tensor(view_30, mul_376);  view_30 = mul_376 = None
        add_125 = torch.ops.aten.add.Tensor(mul_373, mul_377);  mul_373 = mul_377 = None
        mul_378 = torch.ops.aten.mul.Tensor(mul_371, add_125);  mul_371 = add_125 = None
        view_186 = torch.ops.aten.view.default(mul_378, [64000, 512]);  mul_378 = None
        permute_23 = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
        permute_224 = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        mm_47 = torch.ops.aten.mm.default(view_186, permute_224);  permute_224 = None
        permute_225 = torch.ops.aten.permute.default(view_186, [1, 0])
        mm_48 = torch.ops.aten.mm.default(permute_225, view_29);  permute_225 = view_29 = None
        sum_117 = torch.ops.aten.sum.dim_IntList(view_186, [0], True);  view_186 = None
        view_187 = torch.ops.aten.view.default(sum_117, [512]);  sum_117 = None
        view_188 = torch.ops.aten.view.default(mm_47, [500, 128, 128]);  mm_47 = None
        mul_380 = torch.ops.aten.mul.Tensor(view_188, primals_30);  primals_30 = None
        mul_381 = torch.ops.aten.mul.Tensor(mul_380, 128)
        sum_118 = torch.ops.aten.sum.dim_IntList(mul_380, [2], True)
        mul_382 = torch.ops.aten.mul.Tensor(mul_380, mul_29);  mul_380 = None
        sum_119 = torch.ops.aten.sum.dim_IntList(mul_382, [2], True);  mul_382 = None
        mul_383 = torch.ops.aten.mul.Tensor(mul_29, sum_119);  sum_119 = None
        sub_99 = torch.ops.aten.sub.Tensor(mul_381, sum_118);  mul_381 = sum_118 = None
        sub_100 = torch.ops.aten.sub.Tensor(sub_99, mul_383);  sub_99 = mul_383 = None
        mul_384 = torch.ops.aten.mul.Tensor(div_28, sub_100);  div_28 = sub_100 = None
        mul_385 = torch.ops.aten.mul.Tensor(view_188, mul_29);  mul_29 = None
        sum_120 = torch.ops.aten.sum.dim_IntList(mul_385, [0, 1]);  mul_385 = None
        sum_121 = torch.ops.aten.sum.dim_IntList(view_188, [0, 1]);  view_188 = None
        add_126 = torch.ops.aten.add.Tensor(add_123, mul_384);  add_123 = mul_384 = None
        convert_element_type_21 = torch.ops.prims.convert_element_type.default(gt_5, torch.float32);  gt_5 = None
        mul_386 = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
        mul_387 = torch.ops.aten.mul.Tensor(add_126, mul_386);  mul_386 = None
        view_27 = torch.ops.aten.view.default(addmm_4, [128, 500, 128]);  addmm_4 = None
        permute_22 = torch.ops.aten.permute.default(view_27, [1, 0, 2]);  view_27 = None
        sub_101 = torch.ops.aten.sub.Tensor(permute_22, getitem_11);  permute_22 = getitem_11 = None
        mul_388 = torch.ops.aten.mul.Tensor(sub_101, rsqrt_5);  sub_101 = None
        mul_389 = torch.ops.aten.mul.Tensor(mul_387, primals_28);  primals_28 = None
        mul_390 = torch.ops.aten.mul.Tensor(mul_389, 128)
        sum_122 = torch.ops.aten.sum.dim_IntList(mul_389, [2], True)
        mul_391 = torch.ops.aten.mul.Tensor(mul_389, mul_388);  mul_389 = None
        sum_123 = torch.ops.aten.sum.dim_IntList(mul_391, [2], True);  mul_391 = None
        mul_392 = torch.ops.aten.mul.Tensor(mul_388, sum_123);  sum_123 = None
        sub_102 = torch.ops.aten.sub.Tensor(mul_390, sum_122);  mul_390 = sum_122 = None
        sub_103 = torch.ops.aten.sub.Tensor(sub_102, mul_392);  sub_102 = mul_392 = None
        div_29 = torch.ops.aten.div.Tensor(rsqrt_5, 128);  rsqrt_5 = None
        mul_393 = torch.ops.aten.mul.Tensor(div_29, sub_103);  div_29 = sub_103 = None
        mul_394 = torch.ops.aten.mul.Tensor(mul_387, mul_388);  mul_388 = None
        sum_124 = torch.ops.aten.sum.dim_IntList(mul_394, [0, 1]);  mul_394 = None
        sum_125 = torch.ops.aten.sum.dim_IntList(mul_387, [0, 1]);  mul_387 = None
        permute_228 = torch.ops.aten.permute.default(mul_393, [1, 0, 2]);  mul_393 = None
        clone_77 = torch.ops.aten.clone.default(permute_228, memory_format = torch.contiguous_format);  permute_228 = None
        view_189 = torch.ops.aten.view.default(clone_77, [64000, 128]);  clone_77 = None
        permute_21 = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
        permute_229 = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        mm_49 = torch.ops.aten.mm.default(view_189, permute_229);  permute_229 = None
        permute_230 = torch.ops.aten.permute.default(view_189, [1, 0])
        mm_50 = torch.ops.aten.mm.default(permute_230, view_26);  permute_230 = view_26 = None
        sum_126 = torch.ops.aten.sum.dim_IntList(view_189, [0], True);  view_189 = None
        view_190 = torch.ops.aten.view.default(sum_126, [128]);  sum_126 = None
        view_191 = torch.ops.aten.view.default(mm_49, [128, 4000, 16]);  mm_49 = None
        permute_233 = torch.ops.aten.permute.default(view_191, [1, 0, 2]);  view_191 = None
        bmm_27 = torch.ops.aten.bmm.default(permute_234, permute_233);  permute_234 = None
        bmm_28 = torch.ops.aten.bmm.default(permute_233, permute_235);  permute_233 = permute_235 = None
        convert_element_type_22 = torch.ops.prims.convert_element_type.default(gt_4, torch.float32);  gt_4 = None
        mul_395 = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
        mul_396 = torch.ops.aten.mul.Tensor(bmm_28, mul_395);  bmm_28 = mul_395 = None
        sub_6 = torch.ops.aten.sub.Tensor(baddbmm_1, amax_1);  baddbmm_1 = amax_1 = None
        exp_1 = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        div_1 = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        mul_397 = torch.ops.aten.mul.Tensor(mul_396, div_1);  mul_396 = None
        sum_127 = torch.ops.aten.sum.dim_IntList(mul_397, [-1], True)
        neg_5 = torch.ops.aten.neg.default(div_1);  div_1 = None
        fma_5 = torch.ops.prims.fma.default(neg_5, sum_127, mul_397);  neg_5 = sum_127 = mul_397 = None
        bmm_29 = torch.ops.aten.bmm.default(fma_5, permute_236);  permute_236 = None
        bmm_30 = torch.ops.aten.bmm.default(permute_237, fma_5);  permute_237 = None
        permute_238 = torch.ops.aten.permute.default(bmm_30, [0, 2, 1]);  bmm_30 = None
        mul_398 = torch.ops.aten.mul.Tensor(bmm_29, 0.25);  bmm_29 = None
        add_127 = torch.ops.aten.add.Tensor(add_120, fma_5);  add_120 = fma_5 = None
        permute_239 = torch.ops.aten.permute.default(bmm_27, [1, 0, 2]);  bmm_27 = None
        clone_79 = torch.ops.aten.clone.default(permute_239, memory_format = torch.contiguous_format);  permute_239 = None
        view_192 = torch.ops.aten.view.default(clone_79, [128, 500, 128]);  clone_79 = None
        permute_240 = torch.ops.aten.permute.default(permute_238, [1, 0, 2]);  permute_238 = None
        view_193 = torch.ops.aten.view.default(permute_240, [128, 500, 128]);  permute_240 = None
        permute_241 = torch.ops.aten.permute.default(mul_398, [1, 0, 2]);  mul_398 = None
        clone_80 = torch.ops.aten.clone.default(permute_241, memory_format = torch.contiguous_format);  permute_241 = None
        view_194 = torch.ops.aten.view.default(clone_80, [128, 500, 128]);  clone_80 = None
        select_scatter_15 = torch.ops.aten.select_scatter.default(full_default, view_192, 0, 2);  view_192 = None
        select_scatter_16 = torch.ops.aten.select_scatter.default(full_default, view_193, 0, 1);  view_193 = None
        add_128 = torch.ops.aten.add.Tensor(select_scatter_15, select_scatter_16);  select_scatter_15 = select_scatter_16 = None
        select_scatter_17 = torch.ops.aten.select_scatter.default(full_default, view_194, 0, 0);  view_194 = None
        add_129 = torch.ops.aten.add.Tensor(add_128, select_scatter_17);  add_128 = select_scatter_17 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(add_129, 3);  add_129 = None
        permute_242 = torch.ops.aten.permute.default(unsqueeze_12, [3, 1, 2, 0, 4]);  unsqueeze_12 = None
        squeeze_12 = torch.ops.aten.squeeze.dim(permute_242, 0);  permute_242 = None
        clone_81 = torch.ops.aten.clone.default(squeeze_12, memory_format = torch.contiguous_format);  squeeze_12 = None
        view_195 = torch.ops.aten.view.default(clone_81, [128, 500, 384]);  clone_81 = None
        sum_128 = torch.ops.aten.sum.dim_IntList(view_195, [0, 1], True)
        view_196 = torch.ops.aten.view.default(sum_128, [384]);  sum_128 = None
        view_197 = torch.ops.aten.view.default(view_195, [64000, 384]);  view_195 = None
        permute_243 = torch.ops.aten.permute.default(view_197, [1, 0])
        mm_51 = torch.ops.aten.mm.default(permute_243, view_18);  permute_243 = view_18 = None
        permute_14 = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
        permute_245 = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
        mm_52 = torch.ops.aten.mm.default(view_197, permute_245);  view_197 = permute_245 = None
        view_198 = torch.ops.aten.view.default(mm_52, [128, 500, 128]);  mm_52 = None
        permute_247 = torch.ops.aten.permute.default(view_198, [1, 0, 2]);  view_198 = None
        mul_400 = torch.ops.aten.mul.Tensor(permute_247, primals_22);  primals_22 = None
        mul_401 = torch.ops.aten.mul.Tensor(mul_400, 128)
        sum_129 = torch.ops.aten.sum.dim_IntList(mul_400, [2], True)
        mul_402 = torch.ops.aten.mul.Tensor(mul_400, mul_20);  mul_400 = None
        sum_130 = torch.ops.aten.sum.dim_IntList(mul_402, [2], True);  mul_402 = None
        mul_403 = torch.ops.aten.mul.Tensor(mul_20, sum_130);  sum_130 = None
        sub_105 = torch.ops.aten.sub.Tensor(mul_401, sum_129);  mul_401 = sum_129 = None
        sub_106 = torch.ops.aten.sub.Tensor(sub_105, mul_403);  sub_105 = mul_403 = None
        mul_404 = torch.ops.aten.mul.Tensor(div_30, sub_106);  div_30 = sub_106 = None
        mul_405 = torch.ops.aten.mul.Tensor(permute_247, mul_20);  mul_20 = None
        sum_131 = torch.ops.aten.sum.dim_IntList(mul_405, [0, 1]);  mul_405 = None
        sum_132 = torch.ops.aten.sum.dim_IntList(permute_247, [0, 1]);  permute_247 = None
        add_130 = torch.ops.aten.add.Tensor(add_126, mul_404);  add_126 = mul_404 = None
        convert_element_type_23 = torch.ops.prims.convert_element_type.default(gt_3, torch.float32);  gt_3 = None
        mul_406 = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
        mul_407 = torch.ops.aten.mul.Tensor(add_130, mul_406);  mul_406 = None
        view_199 = torch.ops.aten.view.default(mul_407, [64000, 128]);  mul_407 = None
        permute_12 = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
        permute_248 = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        mm_53 = torch.ops.aten.mm.default(view_199, permute_248);  permute_248 = None
        permute_249 = torch.ops.aten.permute.default(view_199, [1, 0])
        mm_54 = torch.ops.aten.mm.default(permute_249, view_16);  permute_249 = view_16 = None
        sum_133 = torch.ops.aten.sum.dim_IntList(view_199, [0], True);  view_199 = None
        view_200 = torch.ops.aten.view.default(sum_133, [128]);  sum_133 = None
        view_201 = torch.ops.aten.view.default(mm_53, [500, 128, 512]);  mm_53 = None
        mul_409 = torch.ops.aten.mul.Tensor(view_201, primals_18);  primals_18 = None
        mul_410 = torch.ops.aten.mul.Tensor(mul_409, 512)
        sum_134 = torch.ops.aten.sum.dim_IntList(mul_409, [2], True)
        view_15 = torch.ops.aten.view.default(addmm_2, [500, 128, 512]);  addmm_2 = None
        mul_11 = torch.ops.aten.mul.Tensor(view_15, 0.5)
        mul_12 = torch.ops.aten.mul.Tensor(view_15, 0.7071067811865476)
        erf = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_9 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_11, add_9);  mul_11 = None
        mul_14 = torch.ops.aten.mul.Tensor(gt_2, mul_13);  mul_13 = None
        mul_15 = torch.ops.aten.mul.Tensor(mul_14, 1.1111111111111112);  mul_14 = None
        sub_4 = torch.ops.aten.sub.Tensor(mul_15, getitem_7);  mul_15 = getitem_7 = None
        mul_16 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = None
        mul_411 = torch.ops.aten.mul.Tensor(mul_409, mul_16);  mul_409 = None
        sum_135 = torch.ops.aten.sum.dim_IntList(mul_411, [2], True);  mul_411 = None
        mul_412 = torch.ops.aten.mul.Tensor(mul_16, sum_135);  sum_135 = None
        sub_108 = torch.ops.aten.sub.Tensor(mul_410, sum_134);  mul_410 = sum_134 = None
        sub_109 = torch.ops.aten.sub.Tensor(sub_108, mul_412);  sub_108 = mul_412 = None
        div_31 = torch.ops.aten.div.Tensor(rsqrt_3, 512);  rsqrt_3 = None
        mul_413 = torch.ops.aten.mul.Tensor(div_31, sub_109);  div_31 = sub_109 = None
        mul_414 = torch.ops.aten.mul.Tensor(view_201, mul_16);  mul_16 = None
        sum_136 = torch.ops.aten.sum.dim_IntList(mul_414, [0, 1]);  mul_414 = None
        sum_137 = torch.ops.aten.sum.dim_IntList(view_201, [0, 1]);  view_201 = None
        convert_element_type_24 = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_415 = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
        mul_416 = torch.ops.aten.mul.Tensor(mul_413, mul_415);  mul_413 = mul_415 = None
        mul_418 = torch.ops.aten.mul.Tensor(add_9, 0.5);  add_9 = None
        mul_419 = torch.ops.aten.mul.Tensor(view_15, view_15)
        mul_420 = torch.ops.aten.mul.Tensor(mul_419, -0.5);  mul_419 = None
        exp_13 = torch.ops.aten.exp.default(mul_420);  mul_420 = None
        mul_421 = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
        mul_422 = torch.ops.aten.mul.Tensor(view_15, mul_421);  view_15 = mul_421 = None
        add_132 = torch.ops.aten.add.Tensor(mul_418, mul_422);  mul_418 = mul_422 = None
        mul_423 = torch.ops.aten.mul.Tensor(mul_416, add_132);  mul_416 = add_132 = None
        view_202 = torch.ops.aten.view.default(mul_423, [64000, 512]);  mul_423 = None
        permute_11 = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        permute_252 = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        mm_55 = torch.ops.aten.mm.default(view_202, permute_252);  permute_252 = None
        permute_253 = torch.ops.aten.permute.default(view_202, [1, 0])
        mm_56 = torch.ops.aten.mm.default(permute_253, view_14);  permute_253 = view_14 = None
        sum_138 = torch.ops.aten.sum.dim_IntList(view_202, [0], True);  view_202 = None
        view_203 = torch.ops.aten.view.default(sum_138, [512]);  sum_138 = None
        view_204 = torch.ops.aten.view.default(mm_55, [500, 128, 128]);  mm_55 = None
        mul_425 = torch.ops.aten.mul.Tensor(view_204, primals_14);  primals_14 = None
        mul_426 = torch.ops.aten.mul.Tensor(mul_425, 128)
        sum_139 = torch.ops.aten.sum.dim_IntList(mul_425, [2], True)
        mul_427 = torch.ops.aten.mul.Tensor(mul_425, mul_9);  mul_425 = None
        sum_140 = torch.ops.aten.sum.dim_IntList(mul_427, [2], True);  mul_427 = None
        mul_428 = torch.ops.aten.mul.Tensor(mul_9, sum_140);  sum_140 = None
        sub_111 = torch.ops.aten.sub.Tensor(mul_426, sum_139);  mul_426 = sum_139 = None
        sub_112 = torch.ops.aten.sub.Tensor(sub_111, mul_428);  sub_111 = mul_428 = None
        mul_429 = torch.ops.aten.mul.Tensor(div_32, sub_112);  div_32 = sub_112 = None
        mul_430 = torch.ops.aten.mul.Tensor(view_204, mul_9);  mul_9 = None
        sum_141 = torch.ops.aten.sum.dim_IntList(mul_430, [0, 1]);  mul_430 = None
        sum_142 = torch.ops.aten.sum.dim_IntList(view_204, [0, 1]);  view_204 = None
        add_133 = torch.ops.aten.add.Tensor(add_130, mul_429);  add_130 = mul_429 = None
        convert_element_type_25 = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_431 = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
        mul_432 = torch.ops.aten.mul.Tensor(add_133, mul_431);  mul_431 = None
        view_12 = torch.ops.aten.view.default(addmm_1, [128, 500, 128]);  addmm_1 = None
        permute_10 = torch.ops.aten.permute.default(view_12, [1, 0, 2]);  view_12 = None
        sub_113 = torch.ops.aten.sub.Tensor(permute_10, getitem_3);  permute_10 = getitem_3 = None
        mul_433 = torch.ops.aten.mul.Tensor(sub_113, rsqrt_1);  sub_113 = None
        mul_434 = torch.ops.aten.mul.Tensor(mul_432, primals_12);  primals_12 = None
        mul_435 = torch.ops.aten.mul.Tensor(mul_434, 128)
        sum_143 = torch.ops.aten.sum.dim_IntList(mul_434, [2], True)
        mul_436 = torch.ops.aten.mul.Tensor(mul_434, mul_433);  mul_434 = None
        sum_144 = torch.ops.aten.sum.dim_IntList(mul_436, [2], True);  mul_436 = None
        mul_437 = torch.ops.aten.mul.Tensor(mul_433, sum_144);  sum_144 = None
        sub_114 = torch.ops.aten.sub.Tensor(mul_435, sum_143);  mul_435 = sum_143 = None
        sub_115 = torch.ops.aten.sub.Tensor(sub_114, mul_437);  sub_114 = mul_437 = None
        div_33 = torch.ops.aten.div.Tensor(rsqrt_1, 128);  rsqrt_1 = None
        mul_438 = torch.ops.aten.mul.Tensor(div_33, sub_115);  div_33 = sub_115 = None
        mul_439 = torch.ops.aten.mul.Tensor(mul_432, mul_433);  mul_433 = None
        sum_145 = torch.ops.aten.sum.dim_IntList(mul_439, [0, 1]);  mul_439 = None
        sum_146 = torch.ops.aten.sum.dim_IntList(mul_432, [0, 1]);  mul_432 = None
        permute_256 = torch.ops.aten.permute.default(mul_438, [1, 0, 2]);  mul_438 = None
        clone_85 = torch.ops.aten.clone.default(permute_256, memory_format = torch.contiguous_format);  permute_256 = None
        view_205 = torch.ops.aten.view.default(clone_85, [64000, 128]);  clone_85 = None
        permute_9 = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_257 = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_57 = torch.ops.aten.mm.default(view_205, permute_257);  permute_257 = None
        permute_258 = torch.ops.aten.permute.default(view_205, [1, 0])
        mm_58 = torch.ops.aten.mm.default(permute_258, view_11);  permute_258 = view_11 = None
        sum_147 = torch.ops.aten.sum.dim_IntList(view_205, [0], True);  view_205 = None
        view_206 = torch.ops.aten.view.default(sum_147, [128]);  sum_147 = None
        view_207 = torch.ops.aten.view.default(mm_57, [128, 4000, 16]);  mm_57 = None
        permute_261 = torch.ops.aten.permute.default(view_207, [1, 0, 2]);  view_207 = None
        bmm_31 = torch.ops.aten.bmm.default(permute_262, permute_261);  permute_262 = None
        bmm_32 = torch.ops.aten.bmm.default(permute_261, permute_263);  permute_261 = permute_263 = None
        convert_element_type_26 = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_440 = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
        mul_441 = torch.ops.aten.mul.Tensor(bmm_32, mul_440);  bmm_32 = mul_440 = None
        sub_1 = torch.ops.aten.sub.Tensor(baddbmm, amax);  baddbmm = amax = None
        exp = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        div = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        mul_442 = torch.ops.aten.mul.Tensor(mul_441, div);  mul_441 = None
        sum_148 = torch.ops.aten.sum.dim_IntList(mul_442, [-1], True)
        neg_6 = torch.ops.aten.neg.default(div);  div = None
        fma_6 = torch.ops.prims.fma.default(neg_6, sum_148, mul_442);  neg_6 = sum_148 = mul_442 = None
        bmm_33 = torch.ops.aten.bmm.default(fma_6, permute_264);  permute_264 = None
        bmm_34 = torch.ops.aten.bmm.default(permute_265, fma_6);  permute_265 = None
        permute_266 = torch.ops.aten.permute.default(bmm_34, [0, 2, 1]);  bmm_34 = None
        mul_443 = torch.ops.aten.mul.Tensor(bmm_33, 0.25);  bmm_33 = None
        add_134 = torch.ops.aten.add.Tensor(add_127, fma_6);  add_127 = fma_6 = None
        permute_267 = torch.ops.aten.permute.default(bmm_31, [1, 0, 2]);  bmm_31 = None
        clone_87 = torch.ops.aten.clone.default(permute_267, memory_format = torch.contiguous_format);  permute_267 = None
        view_208 = torch.ops.aten.view.default(clone_87, [128, 500, 128]);  clone_87 = None
        permute_268 = torch.ops.aten.permute.default(permute_266, [1, 0, 2]);  permute_266 = None
        view_209 = torch.ops.aten.view.default(permute_268, [128, 500, 128]);  permute_268 = None
        permute_269 = torch.ops.aten.permute.default(mul_443, [1, 0, 2]);  mul_443 = None
        clone_88 = torch.ops.aten.clone.default(permute_269, memory_format = torch.contiguous_format);  permute_269 = None
        view_210 = torch.ops.aten.view.default(clone_88, [128, 500, 128]);  clone_88 = None
        select_scatter_18 = torch.ops.aten.select_scatter.default(full_default, view_208, 0, 2);  view_208 = None
        select_scatter_19 = torch.ops.aten.select_scatter.default(full_default, view_209, 0, 1);  view_209 = None
        add_135 = torch.ops.aten.add.Tensor(select_scatter_18, select_scatter_19);  select_scatter_18 = select_scatter_19 = None
        select_scatter_20 = torch.ops.aten.select_scatter.default(full_default, view_210, 0, 0);  full_default = view_210 = None
        add_136 = torch.ops.aten.add.Tensor(add_135, select_scatter_20);  add_135 = select_scatter_20 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(add_136, 3);  add_136 = None
        permute_270 = torch.ops.aten.permute.default(unsqueeze_13, [3, 1, 2, 0, 4]);  unsqueeze_13 = None
        squeeze_13 = torch.ops.aten.squeeze.dim(permute_270, 0);  permute_270 = None
        clone_89 = torch.ops.aten.clone.default(squeeze_13, memory_format = torch.contiguous_format);  squeeze_13 = None
        view_211 = torch.ops.aten.view.default(clone_89, [128, 500, 384]);  clone_89 = None
        sum_149 = torch.ops.aten.sum.dim_IntList(view_211, [0, 1], True)
        view_212 = torch.ops.aten.view.default(sum_149, [384]);  sum_149 = None
        view_213 = torch.ops.aten.view.default(view_211, [64000, 384]);  view_211 = None
        permute_271 = torch.ops.aten.permute.default(view_213, [1, 0])
        mm_59 = torch.ops.aten.mm.default(permute_271, view_3);  permute_271 = view_3 = None
        permute_2 = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_273 = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_60 = torch.ops.aten.mm.default(view_213, permute_273);  view_213 = permute_273 = None
        view_214 = torch.ops.aten.view.default(mm_60, [128, 500, 128]);  mm_60 = None
        permute_275 = torch.ops.aten.permute.default(view_214, [1, 0, 2]);  view_214 = None
        mul_445 = torch.ops.aten.mul.Tensor(permute_275, primals_4);  primals_4 = None
        mul_446 = torch.ops.aten.mul.Tensor(mul_445, 128)
        sum_150 = torch.ops.aten.sum.dim_IntList(mul_445, [2], True)
        view_2 = torch.ops.aten.view.default(addmm, [500, 128, 128]);  addmm = None
        sub = torch.ops.aten.sub.Tensor(view_2, getitem_1);  view_2 = getitem_1 = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_447 = torch.ops.aten.mul.Tensor(mul_445, mul);  mul_445 = None
        sum_151 = torch.ops.aten.sum.dim_IntList(mul_447, [2], True);  mul_447 = None
        mul_448 = torch.ops.aten.mul.Tensor(mul, sum_151);  sum_151 = None
        sub_117 = torch.ops.aten.sub.Tensor(mul_446, sum_150);  mul_446 = sum_150 = None
        sub_118 = torch.ops.aten.sub.Tensor(sub_117, mul_448);  sub_117 = mul_448 = None
        div_34 = torch.ops.aten.div.Tensor(rsqrt, 128);  rsqrt = None
        mul_449 = torch.ops.aten.mul.Tensor(div_34, sub_118);  div_34 = sub_118 = None
        mul_450 = torch.ops.aten.mul.Tensor(permute_275, mul);  mul = None
        sum_152 = torch.ops.aten.sum.dim_IntList(mul_450, [0, 1]);  mul_450 = None
        sum_153 = torch.ops.aten.sum.dim_IntList(permute_275, [0, 1]);  permute_275 = None
        add_137 = torch.ops.aten.add.Tensor(add_133, mul_449);  add_133 = mul_449 = None
        view_215 = torch.ops.aten.view.default(add_137, [64000, 128]);  add_137 = None
        permute = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        permute_276 = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_61 = torch.ops.aten.mm.default(view_215, permute_276);  permute_276 = None
        permute_277 = torch.ops.aten.permute.default(view_215, [1, 0])
        mm_62 = torch.ops.aten.mm.default(permute_277, view_1);  permute_277 = view_1 = None
        sum_154 = torch.ops.aten.sum.dim_IntList(view_215, [0], True);  view_215 = None
        view_216 = torch.ops.aten.view.default(sum_154, [128]);  sum_154 = None
        view_217 = torch.ops.aten.view.default(mm_61, [500, 128, 16]);  mm_61 = None
        view_218 = torch.ops.aten.view.default(view_217, [500, 128, 1, 16]);  view_217 = None
        return (view_218, mm_62, view_216, sum_152, sum_153, mm_59, view_212, mm_58, view_206, None, add_134, sum_145, sum_146, sum_141, sum_142, mm_56, view_203, sum_136, sum_137, mm_54, view_200, sum_131, sum_132, mm_51, view_196, mm_50, view_190, sum_124, sum_125, sum_120, sum_121, mm_48, view_187, sum_115, sum_116, mm_46, view_184, sum_110, sum_111, mm_43, view_180, mm_42, view_174, sum_103, sum_104, sum_99, sum_100, mm_40, view_171, sum_94, sum_95, mm_38, view_168, sum_89, sum_90, mm_35, view_164, mm_34, view_158, sum_82, sum_83, sum_78, sum_79, mm_32, view_155, sum_73, sum_74, mm_30, view_152, sum_68, sum_69, mm_27, view_148, mm_26, view_142, sum_61, sum_62, sum_57, sum_58, mm_24, view_139, sum_52, sum_53, mm_22, view_136, sum_47, sum_48, mm_19, view_132, mm_18, view_126, sum_40, sum_41, sum_36, sum_37, mm_16, view_123, sum_31, sum_32, mm_14, view_120, sum_26, sum_27, mm_11, view_116, mm_10, view_110, sum_19, sum_20, sum_15, sum_16, mm_8, view_107, sum_10, sum_11)
        
def load_args(reader):
    buf0 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf0, (128, 16), is_leaf=True)  # primals_2
    buf1 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf1, (128,), is_leaf=True)  # primals_4
    buf2 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf2, (384, 128), is_leaf=True)  # primals_6
    buf3 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf3, (128, 128), is_leaf=True)  # primals_8
    buf4 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf4, (128,), is_leaf=True)  # primals_12
    buf5 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf5, (128,), is_leaf=True)  # primals_14
    buf6 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf6, (512, 128), is_leaf=True)  # primals_16
    buf7 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf7, (512,), is_leaf=True)  # primals_18
    buf8 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf8, (128, 512), is_leaf=True)  # primals_20
    buf9 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf9, (128,), is_leaf=True)  # primals_22
    buf10 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf10, (384, 128), is_leaf=True)  # primals_24
    buf11 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf11, (128, 128), is_leaf=True)  # primals_26
    buf12 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf12, (128,), is_leaf=True)  # primals_28
    buf13 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf13, (128,), is_leaf=True)  # primals_30
    buf14 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf14, (512, 128), is_leaf=True)  # primals_32
    buf15 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf15, (512,), is_leaf=True)  # primals_34
    buf16 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf16, (128, 512), is_leaf=True)  # primals_36
    buf17 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf17, (128,), is_leaf=True)  # primals_38
    buf18 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf18, (384, 128), is_leaf=True)  # primals_40
    buf19 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf19, (128, 128), is_leaf=True)  # primals_42
    buf20 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf20, (128,), is_leaf=True)  # primals_44
    buf21 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf21, (128,), is_leaf=True)  # primals_46
    buf22 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf22, (512, 128), is_leaf=True)  # primals_48
    buf23 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf23, (512,), is_leaf=True)  # primals_50
    buf24 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf24, (128, 512), is_leaf=True)  # primals_52
    buf25 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf25, (128,), is_leaf=True)  # primals_54
    buf26 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf26, (384, 128), is_leaf=True)  # primals_56
    buf27 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf27, (128, 128), is_leaf=True)  # primals_58
    buf28 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf28, (128,), is_leaf=True)  # primals_60
    buf29 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf29, (128,), is_leaf=True)  # primals_62
    buf30 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf30, (512, 128), is_leaf=True)  # primals_64
    buf31 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf31, (512,), is_leaf=True)  # primals_66
    buf32 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf32, (128, 512), is_leaf=True)  # primals_68
    buf33 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf33, (128,), is_leaf=True)  # primals_70
    buf34 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf34, (384, 128), is_leaf=True)  # primals_72
    buf35 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf35, (128, 128), is_leaf=True)  # primals_74
    buf36 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf36, (128,), is_leaf=True)  # primals_76
    buf37 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf37, (128,), is_leaf=True)  # primals_78
    buf38 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf38, (512, 128), is_leaf=True)  # primals_80
    buf39 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf39, (512,), is_leaf=True)  # primals_82
    buf40 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf40, (128, 512), is_leaf=True)  # primals_84
    buf41 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf41, (128,), is_leaf=True)  # primals_86
    buf42 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf42, (384, 128), is_leaf=True)  # primals_88
    buf43 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf43, (128, 128), is_leaf=True)  # primals_90
    buf44 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf44, (128,), is_leaf=True)  # primals_92
    buf45 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf45, (128,), is_leaf=True)  # primals_94
    buf46 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf46, (512, 128), is_leaf=True)  # primals_96
    buf47 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf47, (512,), is_leaf=True)  # primals_98
    buf48 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf48, (128, 512), is_leaf=True)  # primals_100
    buf49 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf49, (128,), is_leaf=True)  # primals_102
    buf50 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf50, (384, 128), is_leaf=True)  # primals_104
    buf51 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf51, (128, 128), is_leaf=True)  # primals_106
    buf52 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf52, (128,), is_leaf=True)  # primals_108
    buf53 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf53, (128,), is_leaf=True)  # primals_110
    buf54 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf54, (512, 128), is_leaf=True)  # primals_112
    buf55 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf55, (512,), is_leaf=True)  # primals_114
    buf56 = reader.storage(None, 4096000, device=device(type='cuda', index=0))
    reader.tensor(buf56, (64000, 16), is_leaf=True)  # view_1
    buf57 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf57, (64000, 128), is_leaf=True)  # addmm
    buf58 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf58, (500, 128, 1), is_leaf=True)  # getitem_1
    buf59 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf59, (500, 128, 1), is_leaf=True)  # rsqrt
    buf60 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf60, (64000, 128), is_leaf=True)  # view_3
    buf61 = reader.storage(None, 262144000, device=device(type='cuda', index=0))
    reader.tensor(buf61, (4000, 128, 128), is_leaf=True)  # baddbmm
    buf62 = reader.storage(None, 2048000, device=device(type='cuda', index=0))
    reader.tensor(buf62, (4000, 128, 1), is_leaf=True)  # amax
    buf63 = reader.storage(None, 2048000, device=device(type='cuda', index=0))
    reader.tensor(buf63, (4000, 128, 1), is_leaf=True)  # sum_1
    buf64 = reader.storage(None, 65536000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf64, (4000, 128, 128), dtype=torch.bool, is_leaf=True)  # gt
    buf65 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf65, (64000, 128), is_leaf=True)  # view_11
    buf66 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf66, (64000, 128), is_leaf=True)  # addmm_1
    buf67 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf67, (500, 128, 1), is_leaf=True)  # getitem_3
    buf68 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf68, (500, 128, 1), is_leaf=True)  # rsqrt_1
    buf69 = reader.storage(None, 8192000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf69, (500, 128, 128), dtype=torch.bool, is_leaf=True)  # gt_1
    buf70 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf70, (500, 128, 128), is_leaf=True)  # mul_9
    buf71 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf71, (64000, 128), is_leaf=True)  # view_14
    buf72 = reader.storage(None, 131072000, device=device(type='cuda', index=0))
    reader.tensor(buf72, (64000, 512), is_leaf=True)  # addmm_2
    buf73 = reader.storage(None, 32768000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf73, (500, 128, 512), dtype=torch.bool, is_leaf=True)  # gt_2
    buf74 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf74, (500, 128, 1), is_leaf=True)  # getitem_7
    buf75 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf75, (500, 128, 1), is_leaf=True)  # rsqrt_3
    buf76 = reader.storage(None, 131072000, device=device(type='cuda', index=0))
    reader.tensor(buf76, (64000, 512), is_leaf=True)  # view_16
    buf77 = reader.storage(None, 8192000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf77, (500, 128, 128), dtype=torch.bool, is_leaf=True)  # gt_3
    buf78 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf78, (500, 128, 128), is_leaf=True)  # mul_20
    buf79 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf79, (64000, 128), is_leaf=True)  # view_18
    buf80 = reader.storage(None, 262144000, device=device(type='cuda', index=0))
    reader.tensor(buf80, (4000, 128, 128), is_leaf=True)  # baddbmm_1
    buf81 = reader.storage(None, 2048000, device=device(type='cuda', index=0))
    reader.tensor(buf81, (4000, 128, 1), is_leaf=True)  # amax_1
    buf82 = reader.storage(None, 2048000, device=device(type='cuda', index=0))
    reader.tensor(buf82, (4000, 128, 1), is_leaf=True)  # sum_2
    buf83 = reader.storage(None, 65536000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf83, (4000, 128, 128), dtype=torch.bool, is_leaf=True)  # gt_4
    buf84 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf84, (64000, 128), is_leaf=True)  # view_26
    buf85 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf85, (64000, 128), is_leaf=True)  # addmm_4
    buf86 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf86, (500, 128, 1), is_leaf=True)  # getitem_11
    buf87 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf87, (500, 128, 1), is_leaf=True)  # rsqrt_5
    buf88 = reader.storage(None, 8192000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf88, (500, 128, 128), dtype=torch.bool, is_leaf=True)  # gt_5
    buf89 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf89, (500, 128, 128), is_leaf=True)  # mul_29
    buf90 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf90, (64000, 128), is_leaf=True)  # view_29
    buf91 = reader.storage(None, 131072000, device=device(type='cuda', index=0))
    reader.tensor(buf91, (64000, 512), is_leaf=True)  # addmm_5
    buf92 = reader.storage(None, 32768000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf92, (500, 128, 512), dtype=torch.bool, is_leaf=True)  # gt_6
    buf93 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf93, (500, 128, 1), is_leaf=True)  # getitem_15
    buf94 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf94, (500, 128, 1), is_leaf=True)  # rsqrt_7
    buf95 = reader.storage(None, 131072000, device=device(type='cuda', index=0))
    reader.tensor(buf95, (64000, 512), is_leaf=True)  # view_31
    buf96 = reader.storage(None, 8192000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf96, (500, 128, 128), dtype=torch.bool, is_leaf=True)  # gt_7
    buf97 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf97, (500, 128, 128), is_leaf=True)  # mul_40
    buf98 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf98, (64000, 128), is_leaf=True)  # view_33
    buf99 = reader.storage(None, 262144000, device=device(type='cuda', index=0))
    reader.tensor(buf99, (4000, 128, 128), is_leaf=True)  # baddbmm_2
    buf100 = reader.storage(None, 2048000, device=device(type='cuda', index=0))
    reader.tensor(buf100, (4000, 128, 1), is_leaf=True)  # amax_2
    buf101 = reader.storage(None, 2048000, device=device(type='cuda', index=0))
    reader.tensor(buf101, (4000, 128, 1), is_leaf=True)  # sum_3
    buf102 = reader.storage(None, 65536000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf102, (4000, 128, 128), dtype=torch.bool, is_leaf=True)  # gt_8
    buf103 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf103, (64000, 128), is_leaf=True)  # view_41
    buf104 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf104, (64000, 128), is_leaf=True)  # addmm_7
    buf105 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf105, (500, 128, 1), is_leaf=True)  # getitem_19
    buf106 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf106, (500, 128, 1), is_leaf=True)  # rsqrt_9
    buf107 = reader.storage(None, 8192000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf107, (500, 128, 128), dtype=torch.bool, is_leaf=True)  # gt_9
    buf108 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf108, (500, 128, 128), is_leaf=True)  # mul_49
    buf109 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf109, (64000, 128), is_leaf=True)  # view_44
    buf110 = reader.storage(None, 131072000, device=device(type='cuda', index=0))
    reader.tensor(buf110, (64000, 512), is_leaf=True)  # addmm_8
    buf111 = reader.storage(None, 32768000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf111, (500, 128, 512), dtype=torch.bool, is_leaf=True)  # gt_10
    buf112 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf112, (500, 128, 1), is_leaf=True)  # getitem_23
    buf113 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf113, (500, 128, 1), is_leaf=True)  # rsqrt_11
    buf114 = reader.storage(None, 131072000, device=device(type='cuda', index=0))
    reader.tensor(buf114, (64000, 512), is_leaf=True)  # view_46
    buf115 = reader.storage(None, 8192000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf115, (500, 128, 128), dtype=torch.bool, is_leaf=True)  # gt_11
    buf116 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf116, (500, 128, 128), is_leaf=True)  # mul_60
    buf117 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf117, (64000, 128), is_leaf=True)  # view_48
    buf118 = reader.storage(None, 262144000, device=device(type='cuda', index=0))
    reader.tensor(buf118, (4000, 128, 128), is_leaf=True)  # baddbmm_3
    buf119 = reader.storage(None, 2048000, device=device(type='cuda', index=0))
    reader.tensor(buf119, (4000, 128, 1), is_leaf=True)  # amax_3
    buf120 = reader.storage(None, 2048000, device=device(type='cuda', index=0))
    reader.tensor(buf120, (4000, 128, 1), is_leaf=True)  # sum_4
    buf121 = reader.storage(None, 65536000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf121, (4000, 128, 128), dtype=torch.bool, is_leaf=True)  # gt_12
    buf122 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf122, (64000, 128), is_leaf=True)  # view_56
    buf123 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf123, (64000, 128), is_leaf=True)  # addmm_10
    buf124 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf124, (500, 128, 1), is_leaf=True)  # getitem_27
    buf125 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf125, (500, 128, 1), is_leaf=True)  # rsqrt_13
    buf126 = reader.storage(None, 8192000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf126, (500, 128, 128), dtype=torch.bool, is_leaf=True)  # gt_13
    buf127 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf127, (500, 128, 128), is_leaf=True)  # mul_69
    buf128 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf128, (64000, 128), is_leaf=True)  # view_59
    buf129 = reader.storage(None, 131072000, device=device(type='cuda', index=0))
    reader.tensor(buf129, (64000, 512), is_leaf=True)  # addmm_11
    buf130 = reader.storage(None, 32768000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf130, (500, 128, 512), dtype=torch.bool, is_leaf=True)  # gt_14
    buf131 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf131, (500, 128, 1), is_leaf=True)  # getitem_31
    buf132 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf132, (500, 128, 1), is_leaf=True)  # rsqrt_15
    buf133 = reader.storage(None, 131072000, device=device(type='cuda', index=0))
    reader.tensor(buf133, (64000, 512), is_leaf=True)  # view_61
    buf134 = reader.storage(None, 8192000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf134, (500, 128, 128), dtype=torch.bool, is_leaf=True)  # gt_15
    buf135 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf135, (500, 128, 128), is_leaf=True)  # mul_80
    buf136 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf136, (64000, 128), is_leaf=True)  # view_63
    buf137 = reader.storage(None, 262144000, device=device(type='cuda', index=0))
    reader.tensor(buf137, (4000, 128, 128), is_leaf=True)  # baddbmm_4
    buf138 = reader.storage(None, 2048000, device=device(type='cuda', index=0))
    reader.tensor(buf138, (4000, 128, 1), is_leaf=True)  # amax_4
    buf139 = reader.storage(None, 2048000, device=device(type='cuda', index=0))
    reader.tensor(buf139, (4000, 128, 1), is_leaf=True)  # sum_5
    buf140 = reader.storage(None, 65536000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf140, (4000, 128, 128), dtype=torch.bool, is_leaf=True)  # gt_16
    buf141 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf141, (64000, 128), is_leaf=True)  # view_71
    buf142 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf142, (64000, 128), is_leaf=True)  # addmm_13
    buf143 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf143, (500, 128, 1), is_leaf=True)  # getitem_35
    buf144 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf144, (500, 128, 1), is_leaf=True)  # rsqrt_17
    buf145 = reader.storage(None, 8192000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf145, (500, 128, 128), dtype=torch.bool, is_leaf=True)  # gt_17
    buf146 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf146, (500, 128, 128), is_leaf=True)  # mul_89
    buf147 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf147, (64000, 128), is_leaf=True)  # view_74
    buf148 = reader.storage(None, 131072000, device=device(type='cuda', index=0))
    reader.tensor(buf148, (64000, 512), is_leaf=True)  # addmm_14
    buf149 = reader.storage(None, 32768000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf149, (500, 128, 512), dtype=torch.bool, is_leaf=True)  # gt_18
    buf150 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf150, (500, 128, 1), is_leaf=True)  # getitem_39
    buf151 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf151, (500, 128, 1), is_leaf=True)  # rsqrt_19
    buf152 = reader.storage(None, 131072000, device=device(type='cuda', index=0))
    reader.tensor(buf152, (64000, 512), is_leaf=True)  # view_76
    buf153 = reader.storage(None, 8192000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf153, (500, 128, 128), dtype=torch.bool, is_leaf=True)  # gt_19
    buf154 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf154, (500, 128, 128), is_leaf=True)  # mul_100
    buf155 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf155, (64000, 128), is_leaf=True)  # view_78
    buf156 = reader.storage(None, 262144000, device=device(type='cuda', index=0))
    reader.tensor(buf156, (4000, 128, 128), is_leaf=True)  # baddbmm_5
    buf157 = reader.storage(None, 2048000, device=device(type='cuda', index=0))
    reader.tensor(buf157, (4000, 128, 1), is_leaf=True)  # amax_5
    buf158 = reader.storage(None, 2048000, device=device(type='cuda', index=0))
    reader.tensor(buf158, (4000, 128, 1), is_leaf=True)  # sum_6
    buf159 = reader.storage(None, 65536000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf159, (4000, 128, 128), dtype=torch.bool, is_leaf=True)  # gt_20
    buf160 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf160, (64000, 128), is_leaf=True)  # view_86
    buf161 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf161, (64000, 128), is_leaf=True)  # addmm_16
    buf162 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf162, (500, 128, 1), is_leaf=True)  # getitem_43
    buf163 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf163, (500, 128, 1), is_leaf=True)  # rsqrt_21
    buf164 = reader.storage(None, 8192000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf164, (500, 128, 128), dtype=torch.bool, is_leaf=True)  # gt_21
    buf165 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf165, (500, 128, 128), is_leaf=True)  # mul_109
    buf166 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf166, (64000, 128), is_leaf=True)  # view_89
    buf167 = reader.storage(None, 131072000, device=device(type='cuda', index=0))
    reader.tensor(buf167, (64000, 512), is_leaf=True)  # addmm_17
    buf168 = reader.storage(None, 32768000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf168, (500, 128, 512), dtype=torch.bool, is_leaf=True)  # gt_22
    buf169 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf169, (500, 128, 1), is_leaf=True)  # getitem_47
    buf170 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf170, (500, 128, 1), is_leaf=True)  # rsqrt_23
    buf171 = reader.storage(None, 131072000, device=device(type='cuda', index=0))
    reader.tensor(buf171, (64000, 512), is_leaf=True)  # view_91
    buf172 = reader.storage(None, 8192000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf172, (500, 128, 128), dtype=torch.bool, is_leaf=True)  # gt_23
    buf173 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf173, (500, 128, 128), is_leaf=True)  # mul_120
    buf174 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf174, (64000, 128), is_leaf=True)  # view_93
    buf175 = reader.storage(None, 262144000, device=device(type='cuda', index=0))
    reader.tensor(buf175, (4000, 128, 128), is_leaf=True)  # baddbmm_6
    buf176 = reader.storage(None, 2048000, device=device(type='cuda', index=0))
    reader.tensor(buf176, (4000, 128, 1), is_leaf=True)  # amax_6
    buf177 = reader.storage(None, 2048000, device=device(type='cuda', index=0))
    reader.tensor(buf177, (4000, 128, 1), is_leaf=True)  # sum_7
    buf178 = reader.storage(None, 65536000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf178, (4000, 128, 128), dtype=torch.bool, is_leaf=True)  # gt_24
    buf179 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf179, (64000, 128), is_leaf=True)  # view_101
    buf180 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf180, (64000, 128), is_leaf=True)  # addmm_19
    buf181 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf181, (500, 128, 1), is_leaf=True)  # getitem_51
    buf182 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf182, (500, 128, 1), is_leaf=True)  # rsqrt_25
    buf183 = reader.storage(None, 8192000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf183, (500, 128, 128), dtype=torch.bool, is_leaf=True)  # gt_25
    buf184 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf184, (500, 128, 128), requires_grad=True)  # add_84
    buf185 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf185, (500, 128, 1), is_leaf=True)  # getitem_53
    buf186 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf186, (500, 128, 1), is_leaf=True)  # rsqrt_26
    buf187 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf187, (64000, 128), is_leaf=True)  # view_104
    buf188 = reader.storage(None, 131072000, device=device(type='cuda', index=0))
    reader.tensor(buf188, (64000, 512), is_leaf=True)  # addmm_20
    buf189 = reader.storage(None, 32768000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf189, (500, 128, 512), dtype=torch.bool, is_leaf=True)  # gt_26
    buf190 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf190, (500, 128, 1), is_leaf=True)  # getitem_55
    buf191 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf191, (500, 128, 1), is_leaf=True)  # rsqrt_27
    buf192 = reader.storage(None, 262144000, device=device(type='cuda', index=0))
    reader.tensor(buf192, (4000, 128, 128), (16384, 1, 128), is_leaf=True)  # permute_94
    buf193 = reader.storage(None, 98304000, device=device(type='cuda', index=0))
    reader.tensor(buf193, (4000, 16, 128), (16, 1, 64000), storage_offset=16384000, is_leaf=True)  # permute_95
    reader.tensor(buf193, (4000, 128, 16), (16, 64000, 1), storage_offset=8192000, is_leaf=True)  # permute_96
    buf194 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf194, (4000, 16, 128), (16, 1, 64000), is_leaf=True)  # permute_97
    buf195 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf195, (500, 128, 1), is_leaf=True)  # div_10
    buf196 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf196, (500, 128, 1), is_leaf=True)  # div_12
    buf197 = reader.storage(None, 262144000, device=device(type='cuda', index=0))
    reader.tensor(buf197, (4000, 128, 128), (16384, 1, 128), is_leaf=True)  # permute_122
    buf198 = reader.storage(None, 98304000, device=device(type='cuda', index=0))
    reader.tensor(buf198, (4000, 16, 128), (16, 1, 64000), storage_offset=16384000, is_leaf=True)  # permute_123
    reader.tensor(buf198, (4000, 128, 16), (16, 64000, 1), storage_offset=8192000, is_leaf=True)  # permute_124
    buf199 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf199, (4000, 16, 128), (16, 1, 64000), is_leaf=True)  # permute_125
    buf200 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf200, (500, 128, 1), is_leaf=True)  # div_14
    buf201 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf201, (500, 128, 1), is_leaf=True)  # div_16
    buf202 = reader.storage(None, 262144000, device=device(type='cuda', index=0))
    reader.tensor(buf202, (4000, 128, 128), (16384, 1, 128), is_leaf=True)  # permute_150
    buf203 = reader.storage(None, 98304000, device=device(type='cuda', index=0))
    reader.tensor(buf203, (4000, 16, 128), (16, 1, 64000), storage_offset=16384000, is_leaf=True)  # permute_151
    reader.tensor(buf203, (4000, 128, 16), (16, 64000, 1), storage_offset=8192000, is_leaf=True)  # permute_152
    buf204 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf204, (4000, 16, 128), (16, 1, 64000), is_leaf=True)  # permute_153
    buf205 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf205, (500, 128, 1), is_leaf=True)  # div_18
    buf206 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf206, (500, 128, 1), is_leaf=True)  # div_20
    buf207 = reader.storage(None, 262144000, device=device(type='cuda', index=0))
    reader.tensor(buf207, (4000, 128, 128), (16384, 1, 128), is_leaf=True)  # permute_178
    buf208 = reader.storage(None, 98304000, device=device(type='cuda', index=0))
    reader.tensor(buf208, (4000, 16, 128), (16, 1, 64000), storage_offset=16384000, is_leaf=True)  # permute_179
    reader.tensor(buf208, (4000, 128, 16), (16, 64000, 1), storage_offset=8192000, is_leaf=True)  # permute_180
    buf209 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf209, (4000, 16, 128), (16, 1, 64000), is_leaf=True)  # permute_181
    buf210 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf210, (500, 128, 1), is_leaf=True)  # div_22
    buf211 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf211, (500, 128, 1), is_leaf=True)  # div_24
    buf212 = reader.storage(None, 262144000, device=device(type='cuda', index=0))
    reader.tensor(buf212, (4000, 128, 128), (16384, 1, 128), is_leaf=True)  # permute_206
    buf213 = reader.storage(None, 98304000, device=device(type='cuda', index=0))
    reader.tensor(buf213, (4000, 16, 128), (16, 1, 64000), storage_offset=16384000, is_leaf=True)  # permute_207
    reader.tensor(buf213, (4000, 128, 16), (16, 64000, 1), storage_offset=8192000, is_leaf=True)  # permute_208
    buf214 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf214, (4000, 16, 128), (16, 1, 64000), is_leaf=True)  # permute_209
    buf215 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf215, (500, 128, 1), is_leaf=True)  # div_26
    buf216 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf216, (500, 128, 1), is_leaf=True)  # div_28
    buf217 = reader.storage(None, 262144000, device=device(type='cuda', index=0))
    reader.tensor(buf217, (4000, 128, 128), (16384, 1, 128), is_leaf=True)  # permute_234
    buf218 = reader.storage(None, 98304000, device=device(type='cuda', index=0))
    reader.tensor(buf218, (4000, 16, 128), (16, 1, 64000), storage_offset=16384000, is_leaf=True)  # permute_235
    reader.tensor(buf218, (4000, 128, 16), (16, 64000, 1), storage_offset=8192000, is_leaf=True)  # permute_236
    buf219 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf219, (4000, 16, 128), (16, 1, 64000), is_leaf=True)  # permute_237
    buf220 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf220, (500, 128, 1), is_leaf=True)  # div_30
    buf221 = reader.storage(None, 256000, device=device(type='cuda', index=0))
    reader.tensor(buf221, (500, 128, 1), is_leaf=True)  # div_32
    buf222 = reader.storage(None, 262144000, device=device(type='cuda', index=0))
    reader.tensor(buf222, (4000, 128, 128), (16384, 1, 128), is_leaf=True)  # permute_262
    buf223 = reader.storage(None, 98304000, device=device(type='cuda', index=0))
    reader.tensor(buf223, (4000, 16, 128), (16, 1, 64000), storage_offset=16384000, is_leaf=True)  # permute_263
    reader.tensor(buf223, (4000, 128, 16), (16, 64000, 1), storage_offset=8192000, is_leaf=True)  # permute_264
    buf224 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf224, (4000, 16, 128), (16, 1, 64000), is_leaf=True)  # permute_265
    buf225 = reader.storage(None, 131072000, device=device(type='cuda', index=0))
    reader.tensor(buf225, (500, 128, 512), is_leaf=True)  # tangents_1
    buf226 = reader.storage(None, 32768000, device=device(type='cuda', index=0))
    reader.tensor(buf226, (500, 128, 128), is_leaf=True)  # tangents_2
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)