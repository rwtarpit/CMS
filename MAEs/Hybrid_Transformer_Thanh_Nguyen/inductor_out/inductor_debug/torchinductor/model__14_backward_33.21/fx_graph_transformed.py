class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "f32[128, 16]", primals_4: "f32[128]", primals_6: "f32[384, 128]", primals_8: "f32[128, 128]", primals_12: "f32[128]", primals_14: "f32[128]", primals_16: "f32[512, 128]", primals_18: "f32[512]", primals_20: "f32[128, 512]", primals_22: "f32[128]", primals_24: "f32[384, 128]", primals_26: "f32[128, 128]", primals_28: "f32[128]", primals_30: "f32[128]", primals_32: "f32[512, 128]", primals_34: "f32[512]", primals_36: "f32[128, 512]", primals_38: "f32[128]", primals_40: "f32[384, 128]", primals_42: "f32[128, 128]", primals_44: "f32[128]", primals_46: "f32[128]", primals_48: "f32[512, 128]", primals_50: "f32[512]", primals_52: "f32[128, 512]", primals_54: "f32[128]", primals_56: "f32[384, 128]", primals_58: "f32[128, 128]", primals_60: "f32[128]", primals_62: "f32[128]", primals_64: "f32[512, 128]", primals_66: "f32[512]", primals_68: "f32[128, 512]", primals_70: "f32[128]", primals_72: "f32[384, 128]", primals_74: "f32[128, 128]", primals_76: "f32[128]", primals_78: "f32[128]", primals_80: "f32[512, 128]", primals_82: "f32[512]", primals_84: "f32[128, 512]", primals_86: "f32[128]", primals_88: "f32[384, 128]", primals_90: "f32[128, 128]", primals_92: "f32[128]", primals_94: "f32[128]", primals_96: "f32[512, 128]", primals_98: "f32[512]", primals_100: "f32[128, 512]", primals_102: "f32[128]", primals_104: "f32[384, 128]", primals_106: "f32[128, 128]", primals_108: "f32[128]", primals_110: "f32[128]", primals_112: "f32[512, 128]", primals_114: "f32[512]", view_1: "f32[64000, 16]", addmm: "f32[64000, 128]", getitem_1: "f32[500, 128, 1]", rsqrt: "f32[500, 128, 1]", view_3: "f32[64000, 128]", baddbmm: "f32[4000, 128, 128]", amax: "f32[4000, 128, 1]", sum_1: "f32[4000, 128, 1]", gt: "b8[4000, 128, 128]", view_11: "f32[64000, 128]", addmm_1: "f32[64000, 128]", getitem_3: "f32[500, 128, 1]", rsqrt_1: "f32[500, 128, 1]", gt_1: "b8[500, 128, 128]", mul_9: "f32[500, 128, 128]", view_14: "f32[64000, 128]", addmm_2: "f32[64000, 512]", gt_2: "b8[500, 128, 512]", getitem_7: "f32[500, 128, 1]", rsqrt_3: "f32[500, 128, 1]", view_16: "f32[64000, 512]", gt_3: "b8[500, 128, 128]", mul_20: "f32[500, 128, 128]", view_18: "f32[64000, 128]", baddbmm_1: "f32[4000, 128, 128]", amax_1: "f32[4000, 128, 1]", sum_2: "f32[4000, 128, 1]", gt_4: "b8[4000, 128, 128]", view_26: "f32[64000, 128]", addmm_4: "f32[64000, 128]", getitem_11: "f32[500, 128, 1]", rsqrt_5: "f32[500, 128, 1]", gt_5: "b8[500, 128, 128]", mul_29: "f32[500, 128, 128]", view_29: "f32[64000, 128]", addmm_5: "f32[64000, 512]", gt_6: "b8[500, 128, 512]", getitem_15: "f32[500, 128, 1]", rsqrt_7: "f32[500, 128, 1]", view_31: "f32[64000, 512]", gt_7: "b8[500, 128, 128]", mul_40: "f32[500, 128, 128]", view_33: "f32[64000, 128]", baddbmm_2: "f32[4000, 128, 128]", amax_2: "f32[4000, 128, 1]", sum_3: "f32[4000, 128, 1]", gt_8: "b8[4000, 128, 128]", view_41: "f32[64000, 128]", addmm_7: "f32[64000, 128]", getitem_19: "f32[500, 128, 1]", rsqrt_9: "f32[500, 128, 1]", gt_9: "b8[500, 128, 128]", mul_49: "f32[500, 128, 128]", view_44: "f32[64000, 128]", addmm_8: "f32[64000, 512]", gt_10: "b8[500, 128, 512]", getitem_23: "f32[500, 128, 1]", rsqrt_11: "f32[500, 128, 1]", view_46: "f32[64000, 512]", gt_11: "b8[500, 128, 128]", mul_60: "f32[500, 128, 128]", view_48: "f32[64000, 128]", baddbmm_3: "f32[4000, 128, 128]", amax_3: "f32[4000, 128, 1]", sum_4: "f32[4000, 128, 1]", gt_12: "b8[4000, 128, 128]", view_56: "f32[64000, 128]", addmm_10: "f32[64000, 128]", getitem_27: "f32[500, 128, 1]", rsqrt_13: "f32[500, 128, 1]", gt_13: "b8[500, 128, 128]", mul_69: "f32[500, 128, 128]", view_59: "f32[64000, 128]", addmm_11: "f32[64000, 512]", gt_14: "b8[500, 128, 512]", getitem_31: "f32[500, 128, 1]", rsqrt_15: "f32[500, 128, 1]", view_61: "f32[64000, 512]", gt_15: "b8[500, 128, 128]", mul_80: "f32[500, 128, 128]", view_63: "f32[64000, 128]", baddbmm_4: "f32[4000, 128, 128]", amax_4: "f32[4000, 128, 1]", sum_5: "f32[4000, 128, 1]", gt_16: "b8[4000, 128, 128]", view_71: "f32[64000, 128]", addmm_13: "f32[64000, 128]", getitem_35: "f32[500, 128, 1]", rsqrt_17: "f32[500, 128, 1]", gt_17: "b8[500, 128, 128]", mul_89: "f32[500, 128, 128]", view_74: "f32[64000, 128]", addmm_14: "f32[64000, 512]", gt_18: "b8[500, 128, 512]", getitem_39: "f32[500, 128, 1]", rsqrt_19: "f32[500, 128, 1]", view_76: "f32[64000, 512]", gt_19: "b8[500, 128, 128]", mul_100: "f32[500, 128, 128]", view_78: "f32[64000, 128]", baddbmm_5: "f32[4000, 128, 128]", amax_5: "f32[4000, 128, 1]", sum_6: "f32[4000, 128, 1]", gt_20: "b8[4000, 128, 128]", view_86: "f32[64000, 128]", addmm_16: "f32[64000, 128]", getitem_43: "f32[500, 128, 1]", rsqrt_21: "f32[500, 128, 1]", gt_21: "b8[500, 128, 128]", mul_109: "f32[500, 128, 128]", view_89: "f32[64000, 128]", addmm_17: "f32[64000, 512]", gt_22: "b8[500, 128, 512]", getitem_47: "f32[500, 128, 1]", rsqrt_23: "f32[500, 128, 1]", view_91: "f32[64000, 512]", gt_23: "b8[500, 128, 128]", mul_120: "f32[500, 128, 128]", view_93: "f32[64000, 128]", baddbmm_6: "f32[4000, 128, 128]", amax_6: "f32[4000, 128, 1]", sum_7: "f32[4000, 128, 1]", gt_24: "b8[4000, 128, 128]", view_101: "f32[64000, 128]", addmm_19: "f32[64000, 128]", getitem_51: "f32[500, 128, 1]", rsqrt_25: "f32[500, 128, 1]", gt_25: "b8[500, 128, 128]", add_84: "f32[500, 128, 128]", getitem_53: "f32[500, 128, 1]", rsqrt_26: "f32[500, 128, 1]", view_104: "f32[64000, 128]", addmm_20: "f32[64000, 512]", gt_26: "b8[500, 128, 512]", getitem_55: "f32[500, 128, 1]", rsqrt_27: "f32[500, 128, 1]", permute_94: "f32[4000, 128, 128]", permute_95: "f32[4000, 16, 128]", permute_96: "f32[4000, 128, 16]", permute_97: "f32[4000, 16, 128]", div_10: "f32[500, 128, 1]", div_12: "f32[500, 128, 1]", permute_122: "f32[4000, 128, 128]", permute_123: "f32[4000, 16, 128]", permute_124: "f32[4000, 128, 16]", permute_125: "f32[4000, 16, 128]", div_14: "f32[500, 128, 1]", div_16: "f32[500, 128, 1]", permute_150: "f32[4000, 128, 128]", permute_151: "f32[4000, 16, 128]", permute_152: "f32[4000, 128, 16]", permute_153: "f32[4000, 16, 128]", div_18: "f32[500, 128, 1]", div_20: "f32[500, 128, 1]", permute_178: "f32[4000, 128, 128]", permute_179: "f32[4000, 16, 128]", permute_180: "f32[4000, 128, 16]", permute_181: "f32[4000, 16, 128]", div_22: "f32[500, 128, 1]", div_24: "f32[500, 128, 1]", permute_206: "f32[4000, 128, 128]", permute_207: "f32[4000, 16, 128]", permute_208: "f32[4000, 128, 16]", permute_209: "f32[4000, 16, 128]", div_26: "f32[500, 128, 1]", div_28: "f32[500, 128, 1]", permute_234: "f32[4000, 128, 128]", permute_235: "f32[4000, 16, 128]", permute_236: "f32[4000, 128, 16]", permute_237: "f32[4000, 16, 128]", div_30: "f32[500, 128, 1]", div_32: "f32[500, 128, 1]", permute_262: "f32[4000, 128, 128]", permute_263: "f32[4000, 16, 128]", permute_264: "f32[4000, 128, 16]", permute_265: "f32[4000, 16, 128]", tangents_1: "f32[500, 128, 512]", tangents_2: "f32[500, 128, 128]"):
        # File: /app/src/models/feedforward.py:31 in forward, code: x = self.layernorm2(x)
        mul_139: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(tangents_1, primals_114);  primals_114 = None
        mul_140: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_139, 512)
        sum_8: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_139, [2], True)
        
        # File: /app/src/models/feedforward.py:27 in forward, code: x = self.linear1(x)
        view_105: "f32[500, 128, 512]" = torch.ops.aten.reshape.default(addmm_20, [500, 128, 512]);  addmm_20 = None
        
        # File: /app/src/models/feedforward.py:28 in forward, code: x = self.act(x)
        mul_131: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_105, 0.5)
        mul_132: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_105, 0.7071067811865476)
        erf_6: "f32[500, 128, 512]" = torch.ops.aten.erf.default(mul_132);  mul_132 = None
        add_87: "f32[500, 128, 512]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_133: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_131, add_87);  mul_131 = None
        
        # File: /app/src/models/feedforward.py:29 in forward, code: x = self.dropout1(x)
        mul_134: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(gt_26, mul_133);  mul_133 = None
        mul_135: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_134, 1.1111111111111112);  mul_134 = None
        
        # File: /app/src/models/feedforward.py:31 in forward, code: x = self.layernorm2(x)
        sub_34: "f32[500, 128, 512]" = torch.ops.aten.sub.Tensor(mul_135, getitem_55);  mul_135 = getitem_55 = None
        mul_136: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_27);  sub_34 = None
        mul_141: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_139, mul_136);  mul_139 = None
        sum_9: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_141, [2], True);  mul_141 = None
        mul_142: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_136, sum_9);  sum_9 = None
        sub_36: "f32[500, 128, 512]" = torch.ops.aten.sub.Tensor(mul_140, sum_8);  mul_140 = sum_8 = None
        sub_37: "f32[500, 128, 512]" = torch.ops.aten.sub.Tensor(sub_36, mul_142);  sub_36 = mul_142 = None
        div_7: "f32[500, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 512);  rsqrt_27 = None
        mul_143: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(div_7, sub_37);  div_7 = sub_37 = None
        mul_144: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(tangents_1, mul_136);  mul_136 = None
        sum_10: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_144, [0, 1]);  mul_144 = None
        sum_11: "f32[512]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0, 1]);  tangents_1 = None
        
        # File: /app/src/models/feedforward.py:29 in forward, code: x = self.dropout1(x)
        convert_element_type: "f32[500, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_26, torch.float32);  gt_26 = None
        mul_145: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
        mul_146: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_143, mul_145);  mul_143 = mul_145 = None
        
        # File: /app/src/models/feedforward.py:28 in forward, code: x = self.act(x)
        mul_148: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(add_87, 0.5);  add_87 = None
        mul_149: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_105, view_105)
        mul_150: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_149, -0.5);  mul_149 = None
        exp_7: "f32[500, 128, 512]" = torch.ops.aten.exp.default(mul_150);  mul_150 = None
        mul_151: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
        mul_152: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_105, mul_151);  view_105 = mul_151 = None
        add_91: "f32[500, 128, 512]" = torch.ops.aten.add.Tensor(mul_148, mul_152);  mul_148 = mul_152 = None
        mul_153: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_146, add_91);  mul_146 = add_91 = None
        
        # File: /app/src/models/feedforward.py:27 in forward, code: x = self.linear1(x)
        view_106: "f32[64000, 512]" = torch.ops.aten.reshape.default(mul_153, [64000, 512]);  mul_153 = None
        permute_83: "f32[128, 512]" = torch.ops.aten.permute.default(primals_112, [1, 0]);  primals_112 = None
        permute_84: "f32[512, 128]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
        mm_7: "f32[64000, 128]" = torch.ops.aten.mm.default(view_106, permute_84);  permute_84 = None
        permute_85: "f32[512, 64000]" = torch.ops.aten.permute.default(view_106, [1, 0])
        mm_8: "f32[512, 128]" = torch.ops.aten.mm.default(permute_85, view_104);  permute_85 = view_104 = None
        sum_12: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_106, [0], True);  view_106 = None
        view_107: "f32[512]" = torch.ops.aten.reshape.default(sum_12, [512]);  sum_12 = None
        view_108: "f32[500, 128, 128]" = torch.ops.aten.reshape.default(mm_7, [500, 128, 128]);  mm_7 = None
        
        # File: /app/src/models/feedforward.py:26 in forward, code: x = self.layernorm1(x)
        mul_155: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(view_108, primals_110);  primals_110 = None
        mul_156: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_155, 128)
        sum_13: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_155, [2], True)
        sub_33: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(add_84, getitem_53);  add_84 = getitem_53 = None
        mul_129: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_26);  sub_33 = None
        mul_157: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_155, mul_129);  mul_155 = None
        sum_14: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_157, [2], True);  mul_157 = None
        mul_158: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_129, sum_14);  sum_14 = None
        sub_39: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(mul_156, sum_13);  mul_156 = sum_13 = None
        sub_40: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(sub_39, mul_158);  sub_39 = mul_158 = None
        div_8: "f32[500, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 128);  rsqrt_26 = None
        mul_159: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(div_8, sub_40);  div_8 = sub_40 = None
        mul_160: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(view_108, mul_129);  mul_129 = None
        sum_15: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_160, [0, 1]);  mul_160 = None
        sum_16: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_108, [0, 1]);  view_108 = None
        add_92: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(tangents_2, mul_159);  tangents_2 = mul_159 = None
        
        # File: /app/src/models/particle_transformer.py:46 in forward, code: x = self.dropout(x)
        convert_element_type_1: "f32[500, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_25, torch.float32);  gt_25 = None
        mul_161: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_162: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(add_92, mul_161);  mul_161 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        view_102: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(addmm_19, [128, 500, 128]);  addmm_19 = None
        permute_82: "f32[500, 128, 128]" = torch.ops.aten.permute.default(view_102, [1, 0, 2]);  view_102 = None
        
        # File: /app/src/models/particle_transformer.py:45 in forward, code: x = self.layernorm2(x)
        sub_41: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(permute_82, getitem_51);  permute_82 = getitem_51 = None
        mul_163: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_25);  sub_41 = None
        mul_164: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_162, primals_108);  primals_108 = None
        mul_165: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_164, 128)
        sum_17: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_164, [2], True)
        mul_166: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_164, mul_163);  mul_164 = None
        sum_18: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_166, [2], True);  mul_166 = None
        mul_167: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_163, sum_18);  sum_18 = None
        sub_42: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(mul_165, sum_17);  mul_165 = sum_17 = None
        sub_43: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(sub_42, mul_167);  sub_42 = mul_167 = None
        div_9: "f32[500, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 128);  rsqrt_25 = None
        mul_168: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(div_9, sub_43);  div_9 = sub_43 = None
        mul_169: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_162, mul_163);  mul_163 = None
        sum_19: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_169, [0, 1]);  mul_169 = None
        sum_20: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_162, [0, 1]);  mul_162 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        permute_88: "f32[128, 500, 128]" = torch.ops.aten.permute.default(mul_168, [1, 0, 2]);  mul_168 = None
        clone_37: "f32[128, 500, 128]" = torch.ops.aten.clone.default(permute_88, memory_format = torch.contiguous_format);  permute_88 = None
        view_109: "f32[64000, 128]" = torch.ops.aten.reshape.default(clone_37, [64000, 128]);  clone_37 = None
        permute_81: "f32[128, 128]" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
        permute_89: "f32[128, 128]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
        mm_9: "f32[64000, 128]" = torch.ops.aten.mm.default(view_109, permute_89);  permute_89 = None
        permute_90: "f32[128, 64000]" = torch.ops.aten.permute.default(view_109, [1, 0])
        mm_10: "f32[128, 128]" = torch.ops.aten.mm.default(permute_90, view_101);  permute_90 = view_101 = None
        sum_21: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_109, [0], True);  view_109 = None
        view_110: "f32[128]" = torch.ops.aten.reshape.default(sum_21, [128]);  sum_21 = None
        view_111: "f32[128, 4000, 16]" = torch.ops.aten.reshape.default(mm_9, [128, 4000, 16]);  mm_9 = None
        permute_93: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(view_111, [1, 0, 2]);  view_111 = None
        bmm_7: "f32[4000, 128, 16]" = torch.ops.aten.bmm.default(permute_94, permute_93);  permute_94 = None
        bmm_8: "f32[4000, 128, 128]" = torch.ops.aten.bmm.default(permute_93, permute_95);  permute_93 = permute_95 = None
        convert_element_type_2: "f32[4000, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_24, torch.float32);  gt_24 = None
        mul_170: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
        mul_171: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_8, mul_170);  bmm_8 = mul_170 = None
        sub_31: "f32[4000, 128, 128]" = torch.ops.aten.sub.Tensor(baddbmm_6, amax_6);  baddbmm_6 = amax_6 = None
        exp_6: "f32[4000, 128, 128]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        div_6: "f32[4000, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        mul_172: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(mul_171, div_6);  mul_171 = None
        sum_22: "f32[4000, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_172, [-1], True)
        neg: "f32[4000, 128, 128]" = torch.ops.aten.neg.default(div_6);  div_6 = None
        fma: "f32[4000, 128, 128]" = torch.ops.prims.fma.default(neg, sum_22, mul_172);  neg = sum_22 = mul_172 = None
        bmm_9: "f32[4000, 128, 16]" = torch.ops.aten.bmm.default(fma, permute_96);  permute_96 = None
        bmm_10: "f32[4000, 16, 128]" = torch.ops.aten.bmm.default(permute_97, fma);  permute_97 = None
        permute_98: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(bmm_10, [0, 2, 1]);  bmm_10 = None
        mul_173: "f32[4000, 128, 16]" = torch.ops.aten.mul.Tensor(bmm_9, 0.25);  bmm_9 = None
        permute_99: "f32[128, 4000, 16]" = torch.ops.aten.permute.default(bmm_7, [1, 0, 2]);  bmm_7 = None
        clone_39: "f32[128, 4000, 16]" = torch.ops.aten.clone.default(permute_99, memory_format = torch.contiguous_format);  permute_99 = None
        view_112: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(clone_39, [128, 500, 128]);  clone_39 = None
        permute_100: "f32[128, 4000, 16]" = torch.ops.aten.permute.default(permute_98, [1, 0, 2]);  permute_98 = None
        view_113: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(permute_100, [128, 500, 128]);  permute_100 = None
        permute_101: "f32[128, 4000, 16]" = torch.ops.aten.permute.default(mul_173, [1, 0, 2]);  mul_173 = None
        clone_40: "f32[128, 4000, 16]" = torch.ops.aten.clone.default(permute_101, memory_format = torch.contiguous_format);  permute_101 = None
        view_114: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(clone_40, [128, 500, 128]);  clone_40 = None
        full_default: "f32[3, 128, 500, 128]" = torch.ops.aten.full.default([3, 128, 500, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # No stacktrace found for following nodes
        select_scatter_default: "f32[3, 128, 500, 128]" = torch.ops.aten.select_scatter.default(full_default, view_112, 0, 2);  view_112 = None
        select_scatter_default_1: "f32[3, 128, 500, 128]" = torch.ops.aten.select_scatter.default(full_default, view_113, 0, 1);  view_113 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        add_93: "f32[3, 128, 500, 128]" = torch.ops.aten.add.Tensor(select_scatter_default, select_scatter_default_1);  select_scatter_default = select_scatter_default_1 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_2: "f32[3, 128, 500, 128]" = torch.ops.aten.select_scatter.default(full_default, view_114, 0, 0);  view_114 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        add_94: "f32[3, 128, 500, 128]" = torch.ops.aten.add.Tensor(add_93, select_scatter_default_2);  add_93 = select_scatter_default_2 = None
        unsqueeze_7: "f32[3, 128, 500, 1, 128]" = torch.ops.aten.unsqueeze.default(add_94, 3);  add_94 = None
        permute_102: "f32[1, 128, 500, 3, 128]" = torch.ops.aten.permute.default(unsqueeze_7, [3, 1, 2, 0, 4]);  unsqueeze_7 = None
        squeeze_7: "f32[128, 500, 3, 128]" = torch.ops.aten.squeeze.dim(permute_102, 0);  permute_102 = None
        clone_41: "f32[128, 500, 3, 128]" = torch.ops.aten.clone.default(squeeze_7, memory_format = torch.contiguous_format);  squeeze_7 = None
        view_115: "f32[128, 500, 384]" = torch.ops.aten.reshape.default(clone_41, [128, 500, 384]);  clone_41 = None
        sum_23: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(view_115, [0, 1], True)
        view_116: "f32[384]" = torch.ops.aten.reshape.default(sum_23, [384]);  sum_23 = None
        view_117: "f32[64000, 384]" = torch.ops.aten.reshape.default(view_115, [64000, 384]);  view_115 = None
        permute_103: "f32[384, 64000]" = torch.ops.aten.permute.default(view_117, [1, 0])
        mm_11: "f32[384, 128]" = torch.ops.aten.mm.default(permute_103, view_93);  permute_103 = view_93 = None
        permute_74: "f32[128, 384]" = torch.ops.aten.permute.default(primals_104, [1, 0]);  primals_104 = None
        permute_105: "f32[384, 128]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        mm_12: "f32[64000, 128]" = torch.ops.aten.mm.default(view_117, permute_105);  view_117 = permute_105 = None
        view_118: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(mm_12, [128, 500, 128]);  mm_12 = None
        permute_107: "f32[500, 128, 128]" = torch.ops.aten.permute.default(view_118, [1, 0, 2]);  view_118 = None
        
        # File: /app/src/models/particle_transformer.py:43 in forward, code: x = self.layernorm1(x)
        mul_175: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(permute_107, primals_102);  primals_102 = None
        mul_176: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_175, 128)
        sum_24: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_175, [2], True)
        mul_177: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_175, mul_120);  mul_175 = None
        sum_25: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_177, [2], True);  mul_177 = None
        mul_178: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_120, sum_25);  sum_25 = None
        sub_45: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(mul_176, sum_24);  mul_176 = sum_24 = None
        sub_46: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(sub_45, mul_178);  sub_45 = mul_178 = None
        mul_179: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(div_10, sub_46);  div_10 = sub_46 = None
        mul_180: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(permute_107, mul_120);  mul_120 = None
        sum_26: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_180, [0, 1]);  mul_180 = None
        sum_27: "f32[128]" = torch.ops.aten.sum.dim_IntList(permute_107, [0, 1]);  permute_107 = None
        add_95: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(add_92, mul_179);  add_92 = mul_179 = None
        
        # File: /app/src/models/feedforward.py:33 in forward, code: x = self.dropout2(x)
        convert_element_type_3: "f32[500, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_23, torch.float32);  gt_23 = None
        mul_181: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.1111111111111112);  convert_element_type_3 = None
        mul_182: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(add_95, mul_181);  mul_181 = None
        
        # File: /app/src/models/feedforward.py:32 in forward, code: x = self.linear2(x)
        view_119: "f32[64000, 128]" = torch.ops.aten.reshape.default(mul_182, [64000, 128]);  mul_182 = None
        permute_72: "f32[512, 128]" = torch.ops.aten.permute.default(primals_100, [1, 0]);  primals_100 = None
        permute_108: "f32[128, 512]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
        mm_13: "f32[64000, 512]" = torch.ops.aten.mm.default(view_119, permute_108);  permute_108 = None
        permute_109: "f32[128, 64000]" = torch.ops.aten.permute.default(view_119, [1, 0])
        mm_14: "f32[128, 512]" = torch.ops.aten.mm.default(permute_109, view_91);  permute_109 = view_91 = None
        sum_28: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_119, [0], True);  view_119 = None
        view_120: "f32[128]" = torch.ops.aten.reshape.default(sum_28, [128]);  sum_28 = None
        view_121: "f32[500, 128, 512]" = torch.ops.aten.reshape.default(mm_13, [500, 128, 512]);  mm_13 = None
        
        # File: /app/src/models/feedforward.py:31 in forward, code: x = self.layernorm2(x)
        mul_184: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_121, primals_98);  primals_98 = None
        mul_185: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_184, 512)
        sum_29: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_184, [2], True)
        
        # File: /app/src/models/feedforward.py:27 in forward, code: x = self.linear1(x)
        view_90: "f32[500, 128, 512]" = torch.ops.aten.reshape.default(addmm_17, [500, 128, 512]);  addmm_17 = None
        
        # File: /app/src/models/feedforward.py:28 in forward, code: x = self.act(x)
        mul_111: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_90, 0.5)
        mul_112: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_90, 0.7071067811865476)
        erf_5: "f32[500, 128, 512]" = torch.ops.aten.erf.default(mul_112);  mul_112 = None
        add_74: "f32[500, 128, 512]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_113: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_111, add_74);  mul_111 = None
        
        # File: /app/src/models/feedforward.py:29 in forward, code: x = self.dropout1(x)
        mul_114: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(gt_22, mul_113);  mul_113 = None
        mul_115: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_114, 1.1111111111111112);  mul_114 = None
        
        # File: /app/src/models/feedforward.py:31 in forward, code: x = self.layernorm2(x)
        sub_29: "f32[500, 128, 512]" = torch.ops.aten.sub.Tensor(mul_115, getitem_47);  mul_115 = getitem_47 = None
        mul_116: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_23);  sub_29 = None
        mul_186: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_184, mul_116);  mul_184 = None
        sum_30: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_186, [2], True);  mul_186 = None
        mul_187: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_116, sum_30);  sum_30 = None
        sub_48: "f32[500, 128, 512]" = torch.ops.aten.sub.Tensor(mul_185, sum_29);  mul_185 = sum_29 = None
        sub_49: "f32[500, 128, 512]" = torch.ops.aten.sub.Tensor(sub_48, mul_187);  sub_48 = mul_187 = None
        div_11: "f32[500, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 512);  rsqrt_23 = None
        mul_188: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(div_11, sub_49);  div_11 = sub_49 = None
        mul_189: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_121, mul_116);  mul_116 = None
        sum_31: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_189, [0, 1]);  mul_189 = None
        sum_32: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_121, [0, 1]);  view_121 = None
        
        # File: /app/src/models/feedforward.py:29 in forward, code: x = self.dropout1(x)
        convert_element_type_4: "f32[500, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_22, torch.float32);  gt_22 = None
        mul_190: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
        mul_191: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_188, mul_190);  mul_188 = mul_190 = None
        
        # File: /app/src/models/feedforward.py:28 in forward, code: x = self.act(x)
        mul_193: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(add_74, 0.5);  add_74 = None
        mul_194: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_90, view_90)
        mul_195: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_194, -0.5);  mul_194 = None
        exp_8: "f32[500, 128, 512]" = torch.ops.aten.exp.default(mul_195);  mul_195 = None
        mul_196: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
        mul_197: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_90, mul_196);  view_90 = mul_196 = None
        add_97: "f32[500, 128, 512]" = torch.ops.aten.add.Tensor(mul_193, mul_197);  mul_193 = mul_197 = None
        mul_198: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_191, add_97);  mul_191 = add_97 = None
        
        # File: /app/src/models/feedforward.py:27 in forward, code: x = self.linear1(x)
        view_122: "f32[64000, 512]" = torch.ops.aten.reshape.default(mul_198, [64000, 512]);  mul_198 = None
        permute_71: "f32[128, 512]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
        permute_112: "f32[512, 128]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
        mm_15: "f32[64000, 128]" = torch.ops.aten.mm.default(view_122, permute_112);  permute_112 = None
        permute_113: "f32[512, 64000]" = torch.ops.aten.permute.default(view_122, [1, 0])
        mm_16: "f32[512, 128]" = torch.ops.aten.mm.default(permute_113, view_89);  permute_113 = view_89 = None
        sum_33: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_122, [0], True);  view_122 = None
        view_123: "f32[512]" = torch.ops.aten.reshape.default(sum_33, [512]);  sum_33 = None
        view_124: "f32[500, 128, 128]" = torch.ops.aten.reshape.default(mm_15, [500, 128, 128]);  mm_15 = None
        
        # File: /app/src/models/feedforward.py:26 in forward, code: x = self.layernorm1(x)
        mul_200: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(view_124, primals_94);  primals_94 = None
        mul_201: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_200, 128)
        sum_34: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_200, [2], True)
        mul_202: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_200, mul_109);  mul_200 = None
        sum_35: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_202, [2], True);  mul_202 = None
        mul_203: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_109, sum_35);  sum_35 = None
        sub_51: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(mul_201, sum_34);  mul_201 = sum_34 = None
        sub_52: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(sub_51, mul_203);  sub_51 = mul_203 = None
        mul_204: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(div_12, sub_52);  div_12 = sub_52 = None
        mul_205: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(view_124, mul_109);  mul_109 = None
        sum_36: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_205, [0, 1]);  mul_205 = None
        sum_37: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_124, [0, 1]);  view_124 = None
        add_98: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(add_95, mul_204);  add_95 = mul_204 = None
        
        # File: /app/src/models/particle_transformer.py:46 in forward, code: x = self.dropout(x)
        convert_element_type_5: "f32[500, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_21, torch.float32);  gt_21 = None
        mul_206: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
        mul_207: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(add_98, mul_206);  mul_206 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        view_87: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(addmm_16, [128, 500, 128]);  addmm_16 = None
        permute_70: "f32[500, 128, 128]" = torch.ops.aten.permute.default(view_87, [1, 0, 2]);  view_87 = None
        
        # File: /app/src/models/particle_transformer.py:45 in forward, code: x = self.layernorm2(x)
        sub_53: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(permute_70, getitem_43);  permute_70 = getitem_43 = None
        mul_208: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_21);  sub_53 = None
        mul_209: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_207, primals_92);  primals_92 = None
        mul_210: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_209, 128)
        sum_38: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_209, [2], True)
        mul_211: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_209, mul_208);  mul_209 = None
        sum_39: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_211, [2], True);  mul_211 = None
        mul_212: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_208, sum_39);  sum_39 = None
        sub_54: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(mul_210, sum_38);  mul_210 = sum_38 = None
        sub_55: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(sub_54, mul_212);  sub_54 = mul_212 = None
        div_13: "f32[500, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 128);  rsqrt_21 = None
        mul_213: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(div_13, sub_55);  div_13 = sub_55 = None
        mul_214: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_207, mul_208);  mul_208 = None
        sum_40: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_214, [0, 1]);  mul_214 = None
        sum_41: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_207, [0, 1]);  mul_207 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        permute_116: "f32[128, 500, 128]" = torch.ops.aten.permute.default(mul_213, [1, 0, 2]);  mul_213 = None
        clone_45: "f32[128, 500, 128]" = torch.ops.aten.clone.default(permute_116, memory_format = torch.contiguous_format);  permute_116 = None
        view_125: "f32[64000, 128]" = torch.ops.aten.reshape.default(clone_45, [64000, 128]);  clone_45 = None
        permute_69: "f32[128, 128]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
        permute_117: "f32[128, 128]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
        mm_17: "f32[64000, 128]" = torch.ops.aten.mm.default(view_125, permute_117);  permute_117 = None
        permute_118: "f32[128, 64000]" = torch.ops.aten.permute.default(view_125, [1, 0])
        mm_18: "f32[128, 128]" = torch.ops.aten.mm.default(permute_118, view_86);  permute_118 = view_86 = None
        sum_42: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_125, [0], True);  view_125 = None
        view_126: "f32[128]" = torch.ops.aten.reshape.default(sum_42, [128]);  sum_42 = None
        view_127: "f32[128, 4000, 16]" = torch.ops.aten.reshape.default(mm_17, [128, 4000, 16]);  mm_17 = None
        permute_121: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(view_127, [1, 0, 2]);  view_127 = None
        bmm_11: "f32[4000, 128, 16]" = torch.ops.aten.bmm.default(permute_122, permute_121);  permute_122 = None
        bmm_12: "f32[4000, 128, 128]" = torch.ops.aten.bmm.default(permute_121, permute_123);  permute_121 = permute_123 = None
        convert_element_type_6: "f32[4000, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_20, torch.float32);  gt_20 = None
        mul_215: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.1111111111111112);  convert_element_type_6 = None
        mul_216: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_12, mul_215);  bmm_12 = mul_215 = None
        sub_26: "f32[4000, 128, 128]" = torch.ops.aten.sub.Tensor(baddbmm_5, amax_5);  baddbmm_5 = amax_5 = None
        exp_5: "f32[4000, 128, 128]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
        div_5: "f32[4000, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        mul_217: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(mul_216, div_5);  mul_216 = None
        sum_43: "f32[4000, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_217, [-1], True)
        neg_1: "f32[4000, 128, 128]" = torch.ops.aten.neg.default(div_5);  div_5 = None
        fma_1: "f32[4000, 128, 128]" = torch.ops.prims.fma.default(neg_1, sum_43, mul_217);  neg_1 = sum_43 = mul_217 = None
        bmm_13: "f32[4000, 128, 16]" = torch.ops.aten.bmm.default(fma_1, permute_124);  permute_124 = None
        bmm_14: "f32[4000, 16, 128]" = torch.ops.aten.bmm.default(permute_125, fma_1);  permute_125 = None
        permute_126: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(bmm_14, [0, 2, 1]);  bmm_14 = None
        mul_218: "f32[4000, 128, 16]" = torch.ops.aten.mul.Tensor(bmm_13, 0.25);  bmm_13 = None
        add_99: "f32[4000, 128, 128]" = torch.ops.aten.add.Tensor(fma, fma_1);  fma = fma_1 = None
        permute_127: "f32[128, 4000, 16]" = torch.ops.aten.permute.default(bmm_11, [1, 0, 2]);  bmm_11 = None
        clone_47: "f32[128, 4000, 16]" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None
        view_128: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(clone_47, [128, 500, 128]);  clone_47 = None
        permute_128: "f32[128, 4000, 16]" = torch.ops.aten.permute.default(permute_126, [1, 0, 2]);  permute_126 = None
        view_129: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(permute_128, [128, 500, 128]);  permute_128 = None
        permute_129: "f32[128, 4000, 16]" = torch.ops.aten.permute.default(mul_218, [1, 0, 2]);  mul_218 = None
        clone_48: "f32[128, 4000, 16]" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
        view_130: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(clone_48, [128, 500, 128]);  clone_48 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_3: "f32[3, 128, 500, 128]" = torch.ops.aten.select_scatter.default(full_default, view_128, 0, 2);  view_128 = None
        select_scatter_default_4: "f32[3, 128, 500, 128]" = torch.ops.aten.select_scatter.default(full_default, view_129, 0, 1);  view_129 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        add_100: "f32[3, 128, 500, 128]" = torch.ops.aten.add.Tensor(select_scatter_default_3, select_scatter_default_4);  select_scatter_default_3 = select_scatter_default_4 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_5: "f32[3, 128, 500, 128]" = torch.ops.aten.select_scatter.default(full_default, view_130, 0, 0);  view_130 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        add_101: "f32[3, 128, 500, 128]" = torch.ops.aten.add.Tensor(add_100, select_scatter_default_5);  add_100 = select_scatter_default_5 = None
        unsqueeze_8: "f32[3, 128, 500, 1, 128]" = torch.ops.aten.unsqueeze.default(add_101, 3);  add_101 = None
        permute_130: "f32[1, 128, 500, 3, 128]" = torch.ops.aten.permute.default(unsqueeze_8, [3, 1, 2, 0, 4]);  unsqueeze_8 = None
        squeeze_8: "f32[128, 500, 3, 128]" = torch.ops.aten.squeeze.dim(permute_130, 0);  permute_130 = None
        clone_49: "f32[128, 500, 3, 128]" = torch.ops.aten.clone.default(squeeze_8, memory_format = torch.contiguous_format);  squeeze_8 = None
        view_131: "f32[128, 500, 384]" = torch.ops.aten.reshape.default(clone_49, [128, 500, 384]);  clone_49 = None
        sum_44: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(view_131, [0, 1], True)
        view_132: "f32[384]" = torch.ops.aten.reshape.default(sum_44, [384]);  sum_44 = None
        view_133: "f32[64000, 384]" = torch.ops.aten.reshape.default(view_131, [64000, 384]);  view_131 = None
        permute_131: "f32[384, 64000]" = torch.ops.aten.permute.default(view_133, [1, 0])
        mm_19: "f32[384, 128]" = torch.ops.aten.mm.default(permute_131, view_78);  permute_131 = view_78 = None
        permute_62: "f32[128, 384]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
        permute_133: "f32[384, 128]" = torch.ops.aten.permute.default(permute_62, [1, 0]);  permute_62 = None
        mm_20: "f32[64000, 128]" = torch.ops.aten.mm.default(view_133, permute_133);  view_133 = permute_133 = None
        view_134: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(mm_20, [128, 500, 128]);  mm_20 = None
        permute_135: "f32[500, 128, 128]" = torch.ops.aten.permute.default(view_134, [1, 0, 2]);  view_134 = None
        
        # File: /app/src/models/particle_transformer.py:43 in forward, code: x = self.layernorm1(x)
        mul_220: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(permute_135, primals_86);  primals_86 = None
        mul_221: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_220, 128)
        sum_45: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_220, [2], True)
        mul_222: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_220, mul_100);  mul_220 = None
        sum_46: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_222, [2], True);  mul_222 = None
        mul_223: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_100, sum_46);  sum_46 = None
        sub_57: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(mul_221, sum_45);  mul_221 = sum_45 = None
        sub_58: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(sub_57, mul_223);  sub_57 = mul_223 = None
        mul_224: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(div_14, sub_58);  div_14 = sub_58 = None
        mul_225: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(permute_135, mul_100);  mul_100 = None
        sum_47: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_225, [0, 1]);  mul_225 = None
        sum_48: "f32[128]" = torch.ops.aten.sum.dim_IntList(permute_135, [0, 1]);  permute_135 = None
        add_102: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(add_98, mul_224);  add_98 = mul_224 = None
        
        # File: /app/src/models/feedforward.py:33 in forward, code: x = self.dropout2(x)
        convert_element_type_7: "f32[500, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_19, torch.float32);  gt_19 = None
        mul_226: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
        mul_227: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(add_102, mul_226);  mul_226 = None
        
        # File: /app/src/models/feedforward.py:32 in forward, code: x = self.linear2(x)
        view_135: "f32[64000, 128]" = torch.ops.aten.reshape.default(mul_227, [64000, 128]);  mul_227 = None
        permute_60: "f32[512, 128]" = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
        permute_136: "f32[128, 512]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
        mm_21: "f32[64000, 512]" = torch.ops.aten.mm.default(view_135, permute_136);  permute_136 = None
        permute_137: "f32[128, 64000]" = torch.ops.aten.permute.default(view_135, [1, 0])
        mm_22: "f32[128, 512]" = torch.ops.aten.mm.default(permute_137, view_76);  permute_137 = view_76 = None
        sum_49: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_135, [0], True);  view_135 = None
        view_136: "f32[128]" = torch.ops.aten.reshape.default(sum_49, [128]);  sum_49 = None
        view_137: "f32[500, 128, 512]" = torch.ops.aten.reshape.default(mm_21, [500, 128, 512]);  mm_21 = None
        
        # File: /app/src/models/feedforward.py:31 in forward, code: x = self.layernorm2(x)
        mul_229: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_137, primals_82);  primals_82 = None
        mul_230: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_229, 512)
        sum_50: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_229, [2], True)
        
        # File: /app/src/models/feedforward.py:27 in forward, code: x = self.linear1(x)
        view_75: "f32[500, 128, 512]" = torch.ops.aten.reshape.default(addmm_14, [500, 128, 512]);  addmm_14 = None
        
        # File: /app/src/models/feedforward.py:28 in forward, code: x = self.act(x)
        mul_91: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_75, 0.5)
        mul_92: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_75, 0.7071067811865476)
        erf_4: "f32[500, 128, 512]" = torch.ops.aten.erf.default(mul_92);  mul_92 = None
        add_61: "f32[500, 128, 512]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_93: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_91, add_61);  mul_91 = None
        
        # File: /app/src/models/feedforward.py:29 in forward, code: x = self.dropout1(x)
        mul_94: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(gt_18, mul_93);  mul_93 = None
        mul_95: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_94, 1.1111111111111112);  mul_94 = None
        
        # File: /app/src/models/feedforward.py:31 in forward, code: x = self.layernorm2(x)
        sub_24: "f32[500, 128, 512]" = torch.ops.aten.sub.Tensor(mul_95, getitem_39);  mul_95 = getitem_39 = None
        mul_96: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_19);  sub_24 = None
        mul_231: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_229, mul_96);  mul_229 = None
        sum_51: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_231, [2], True);  mul_231 = None
        mul_232: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_96, sum_51);  sum_51 = None
        sub_60: "f32[500, 128, 512]" = torch.ops.aten.sub.Tensor(mul_230, sum_50);  mul_230 = sum_50 = None
        sub_61: "f32[500, 128, 512]" = torch.ops.aten.sub.Tensor(sub_60, mul_232);  sub_60 = mul_232 = None
        div_15: "f32[500, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 512);  rsqrt_19 = None
        mul_233: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(div_15, sub_61);  div_15 = sub_61 = None
        mul_234: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_137, mul_96);  mul_96 = None
        sum_52: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_234, [0, 1]);  mul_234 = None
        sum_53: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_137, [0, 1]);  view_137 = None
        
        # File: /app/src/models/feedforward.py:29 in forward, code: x = self.dropout1(x)
        convert_element_type_8: "f32[500, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_18, torch.float32);  gt_18 = None
        mul_235: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
        mul_236: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_233, mul_235);  mul_233 = mul_235 = None
        
        # File: /app/src/models/feedforward.py:28 in forward, code: x = self.act(x)
        mul_238: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(add_61, 0.5);  add_61 = None
        mul_239: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_75, view_75)
        mul_240: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_239, -0.5);  mul_239 = None
        exp_9: "f32[500, 128, 512]" = torch.ops.aten.exp.default(mul_240);  mul_240 = None
        mul_241: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
        mul_242: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_75, mul_241);  view_75 = mul_241 = None
        add_104: "f32[500, 128, 512]" = torch.ops.aten.add.Tensor(mul_238, mul_242);  mul_238 = mul_242 = None
        mul_243: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_236, add_104);  mul_236 = add_104 = None
        
        # File: /app/src/models/feedforward.py:27 in forward, code: x = self.linear1(x)
        view_138: "f32[64000, 512]" = torch.ops.aten.reshape.default(mul_243, [64000, 512]);  mul_243 = None
        permute_59: "f32[128, 512]" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
        permute_140: "f32[512, 128]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
        mm_23: "f32[64000, 128]" = torch.ops.aten.mm.default(view_138, permute_140);  permute_140 = None
        permute_141: "f32[512, 64000]" = torch.ops.aten.permute.default(view_138, [1, 0])
        mm_24: "f32[512, 128]" = torch.ops.aten.mm.default(permute_141, view_74);  permute_141 = view_74 = None
        sum_54: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_138, [0], True);  view_138 = None
        view_139: "f32[512]" = torch.ops.aten.reshape.default(sum_54, [512]);  sum_54 = None
        view_140: "f32[500, 128, 128]" = torch.ops.aten.reshape.default(mm_23, [500, 128, 128]);  mm_23 = None
        
        # File: /app/src/models/feedforward.py:26 in forward, code: x = self.layernorm1(x)
        mul_245: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(view_140, primals_78);  primals_78 = None
        mul_246: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_245, 128)
        sum_55: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_245, [2], True)
        mul_247: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_245, mul_89);  mul_245 = None
        sum_56: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_247, [2], True);  mul_247 = None
        mul_248: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_89, sum_56);  sum_56 = None
        sub_63: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(mul_246, sum_55);  mul_246 = sum_55 = None
        sub_64: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(sub_63, mul_248);  sub_63 = mul_248 = None
        mul_249: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(div_16, sub_64);  div_16 = sub_64 = None
        mul_250: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(view_140, mul_89);  mul_89 = None
        sum_57: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_250, [0, 1]);  mul_250 = None
        sum_58: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_140, [0, 1]);  view_140 = None
        add_105: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(add_102, mul_249);  add_102 = mul_249 = None
        
        # File: /app/src/models/particle_transformer.py:46 in forward, code: x = self.dropout(x)
        convert_element_type_9: "f32[500, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_17, torch.float32);  gt_17 = None
        mul_251: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
        mul_252: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(add_105, mul_251);  mul_251 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        view_72: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(addmm_13, [128, 500, 128]);  addmm_13 = None
        permute_58: "f32[500, 128, 128]" = torch.ops.aten.permute.default(view_72, [1, 0, 2]);  view_72 = None
        
        # File: /app/src/models/particle_transformer.py:45 in forward, code: x = self.layernorm2(x)
        sub_65: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(permute_58, getitem_35);  permute_58 = getitem_35 = None
        mul_253: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_17);  sub_65 = None
        mul_254: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_252, primals_76);  primals_76 = None
        mul_255: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_254, 128)
        sum_59: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_254, [2], True)
        mul_256: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_254, mul_253);  mul_254 = None
        sum_60: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_256, [2], True);  mul_256 = None
        mul_257: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_253, sum_60);  sum_60 = None
        sub_66: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(mul_255, sum_59);  mul_255 = sum_59 = None
        sub_67: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(sub_66, mul_257);  sub_66 = mul_257 = None
        div_17: "f32[500, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 128);  rsqrt_17 = None
        mul_258: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(div_17, sub_67);  div_17 = sub_67 = None
        mul_259: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_252, mul_253);  mul_253 = None
        sum_61: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_259, [0, 1]);  mul_259 = None
        sum_62: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_252, [0, 1]);  mul_252 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        permute_144: "f32[128, 500, 128]" = torch.ops.aten.permute.default(mul_258, [1, 0, 2]);  mul_258 = None
        clone_53: "f32[128, 500, 128]" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
        view_141: "f32[64000, 128]" = torch.ops.aten.reshape.default(clone_53, [64000, 128]);  clone_53 = None
        permute_57: "f32[128, 128]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
        permute_145: "f32[128, 128]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        mm_25: "f32[64000, 128]" = torch.ops.aten.mm.default(view_141, permute_145);  permute_145 = None
        permute_146: "f32[128, 64000]" = torch.ops.aten.permute.default(view_141, [1, 0])
        mm_26: "f32[128, 128]" = torch.ops.aten.mm.default(permute_146, view_71);  permute_146 = view_71 = None
        sum_63: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_141, [0], True);  view_141 = None
        view_142: "f32[128]" = torch.ops.aten.reshape.default(sum_63, [128]);  sum_63 = None
        view_143: "f32[128, 4000, 16]" = torch.ops.aten.reshape.default(mm_25, [128, 4000, 16]);  mm_25 = None
        permute_149: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(view_143, [1, 0, 2]);  view_143 = None
        bmm_15: "f32[4000, 128, 16]" = torch.ops.aten.bmm.default(permute_150, permute_149);  permute_150 = None
        bmm_16: "f32[4000, 128, 128]" = torch.ops.aten.bmm.default(permute_149, permute_151);  permute_149 = permute_151 = None
        convert_element_type_10: "f32[4000, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_16, torch.float32);  gt_16 = None
        mul_260: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
        mul_261: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_16, mul_260);  bmm_16 = mul_260 = None
        sub_21: "f32[4000, 128, 128]" = torch.ops.aten.sub.Tensor(baddbmm_4, amax_4);  baddbmm_4 = amax_4 = None
        exp_4: "f32[4000, 128, 128]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
        div_4: "f32[4000, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        mul_262: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(mul_261, div_4);  mul_261 = None
        sum_64: "f32[4000, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_262, [-1], True)
        neg_2: "f32[4000, 128, 128]" = torch.ops.aten.neg.default(div_4);  div_4 = None
        fma_2: "f32[4000, 128, 128]" = torch.ops.prims.fma.default(neg_2, sum_64, mul_262);  neg_2 = sum_64 = mul_262 = None
        bmm_17: "f32[4000, 128, 16]" = torch.ops.aten.bmm.default(fma_2, permute_152);  permute_152 = None
        bmm_18: "f32[4000, 16, 128]" = torch.ops.aten.bmm.default(permute_153, fma_2);  permute_153 = None
        permute_154: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(bmm_18, [0, 2, 1]);  bmm_18 = None
        mul_263: "f32[4000, 128, 16]" = torch.ops.aten.mul.Tensor(bmm_17, 0.25);  bmm_17 = None
        add_106: "f32[4000, 128, 128]" = torch.ops.aten.add.Tensor(add_99, fma_2);  add_99 = fma_2 = None
        permute_155: "f32[128, 4000, 16]" = torch.ops.aten.permute.default(bmm_15, [1, 0, 2]);  bmm_15 = None
        clone_55: "f32[128, 4000, 16]" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
        view_144: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(clone_55, [128, 500, 128]);  clone_55 = None
        permute_156: "f32[128, 4000, 16]" = torch.ops.aten.permute.default(permute_154, [1, 0, 2]);  permute_154 = None
        view_145: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(permute_156, [128, 500, 128]);  permute_156 = None
        permute_157: "f32[128, 4000, 16]" = torch.ops.aten.permute.default(mul_263, [1, 0, 2]);  mul_263 = None
        clone_56: "f32[128, 4000, 16]" = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format);  permute_157 = None
        view_146: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(clone_56, [128, 500, 128]);  clone_56 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_6: "f32[3, 128, 500, 128]" = torch.ops.aten.select_scatter.default(full_default, view_144, 0, 2);  view_144 = None
        select_scatter_default_7: "f32[3, 128, 500, 128]" = torch.ops.aten.select_scatter.default(full_default, view_145, 0, 1);  view_145 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        add_107: "f32[3, 128, 500, 128]" = torch.ops.aten.add.Tensor(select_scatter_default_6, select_scatter_default_7);  select_scatter_default_6 = select_scatter_default_7 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_8: "f32[3, 128, 500, 128]" = torch.ops.aten.select_scatter.default(full_default, view_146, 0, 0);  view_146 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        add_108: "f32[3, 128, 500, 128]" = torch.ops.aten.add.Tensor(add_107, select_scatter_default_8);  add_107 = select_scatter_default_8 = None
        unsqueeze_9: "f32[3, 128, 500, 1, 128]" = torch.ops.aten.unsqueeze.default(add_108, 3);  add_108 = None
        permute_158: "f32[1, 128, 500, 3, 128]" = torch.ops.aten.permute.default(unsqueeze_9, [3, 1, 2, 0, 4]);  unsqueeze_9 = None
        squeeze_9: "f32[128, 500, 3, 128]" = torch.ops.aten.squeeze.dim(permute_158, 0);  permute_158 = None
        clone_57: "f32[128, 500, 3, 128]" = torch.ops.aten.clone.default(squeeze_9, memory_format = torch.contiguous_format);  squeeze_9 = None
        view_147: "f32[128, 500, 384]" = torch.ops.aten.reshape.default(clone_57, [128, 500, 384]);  clone_57 = None
        sum_65: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(view_147, [0, 1], True)
        view_148: "f32[384]" = torch.ops.aten.reshape.default(sum_65, [384]);  sum_65 = None
        view_149: "f32[64000, 384]" = torch.ops.aten.reshape.default(view_147, [64000, 384]);  view_147 = None
        permute_159: "f32[384, 64000]" = torch.ops.aten.permute.default(view_149, [1, 0])
        mm_27: "f32[384, 128]" = torch.ops.aten.mm.default(permute_159, view_63);  permute_159 = view_63 = None
        permute_50: "f32[128, 384]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
        permute_161: "f32[384, 128]" = torch.ops.aten.permute.default(permute_50, [1, 0]);  permute_50 = None
        mm_28: "f32[64000, 128]" = torch.ops.aten.mm.default(view_149, permute_161);  view_149 = permute_161 = None
        view_150: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(mm_28, [128, 500, 128]);  mm_28 = None
        permute_163: "f32[500, 128, 128]" = torch.ops.aten.permute.default(view_150, [1, 0, 2]);  view_150 = None
        
        # File: /app/src/models/particle_transformer.py:43 in forward, code: x = self.layernorm1(x)
        mul_265: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(permute_163, primals_70);  primals_70 = None
        mul_266: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_265, 128)
        sum_66: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_265, [2], True)
        mul_267: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_265, mul_80);  mul_265 = None
        sum_67: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_267, [2], True);  mul_267 = None
        mul_268: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_80, sum_67);  sum_67 = None
        sub_69: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(mul_266, sum_66);  mul_266 = sum_66 = None
        sub_70: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(sub_69, mul_268);  sub_69 = mul_268 = None
        mul_269: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(div_18, sub_70);  div_18 = sub_70 = None
        mul_270: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(permute_163, mul_80);  mul_80 = None
        sum_68: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_270, [0, 1]);  mul_270 = None
        sum_69: "f32[128]" = torch.ops.aten.sum.dim_IntList(permute_163, [0, 1]);  permute_163 = None
        add_109: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(add_105, mul_269);  add_105 = mul_269 = None
        
        # File: /app/src/models/feedforward.py:33 in forward, code: x = self.dropout2(x)
        convert_element_type_11: "f32[500, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_15, torch.float32);  gt_15 = None
        mul_271: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
        mul_272: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(add_109, mul_271);  mul_271 = None
        
        # File: /app/src/models/feedforward.py:32 in forward, code: x = self.linear2(x)
        view_151: "f32[64000, 128]" = torch.ops.aten.reshape.default(mul_272, [64000, 128]);  mul_272 = None
        permute_48: "f32[512, 128]" = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
        permute_164: "f32[128, 512]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
        mm_29: "f32[64000, 512]" = torch.ops.aten.mm.default(view_151, permute_164);  permute_164 = None
        permute_165: "f32[128, 64000]" = torch.ops.aten.permute.default(view_151, [1, 0])
        mm_30: "f32[128, 512]" = torch.ops.aten.mm.default(permute_165, view_61);  permute_165 = view_61 = None
        sum_70: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_151, [0], True);  view_151 = None
        view_152: "f32[128]" = torch.ops.aten.reshape.default(sum_70, [128]);  sum_70 = None
        view_153: "f32[500, 128, 512]" = torch.ops.aten.reshape.default(mm_29, [500, 128, 512]);  mm_29 = None
        
        # File: /app/src/models/feedforward.py:31 in forward, code: x = self.layernorm2(x)
        mul_274: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_153, primals_66);  primals_66 = None
        mul_275: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_274, 512)
        sum_71: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_274, [2], True)
        
        # File: /app/src/models/feedforward.py:27 in forward, code: x = self.linear1(x)
        view_60: "f32[500, 128, 512]" = torch.ops.aten.reshape.default(addmm_11, [500, 128, 512]);  addmm_11 = None
        
        # File: /app/src/models/feedforward.py:28 in forward, code: x = self.act(x)
        mul_71: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_60, 0.5)
        mul_72: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_60, 0.7071067811865476)
        erf_3: "f32[500, 128, 512]" = torch.ops.aten.erf.default(mul_72);  mul_72 = None
        add_48: "f32[500, 128, 512]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_73: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_71, add_48);  mul_71 = None
        
        # File: /app/src/models/feedforward.py:29 in forward, code: x = self.dropout1(x)
        mul_74: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(gt_14, mul_73);  mul_73 = None
        mul_75: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_74, 1.1111111111111112);  mul_74 = None
        
        # File: /app/src/models/feedforward.py:31 in forward, code: x = self.layernorm2(x)
        sub_19: "f32[500, 128, 512]" = torch.ops.aten.sub.Tensor(mul_75, getitem_31);  mul_75 = getitem_31 = None
        mul_76: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_15);  sub_19 = None
        mul_276: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_274, mul_76);  mul_274 = None
        sum_72: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_276, [2], True);  mul_276 = None
        mul_277: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_76, sum_72);  sum_72 = None
        sub_72: "f32[500, 128, 512]" = torch.ops.aten.sub.Tensor(mul_275, sum_71);  mul_275 = sum_71 = None
        sub_73: "f32[500, 128, 512]" = torch.ops.aten.sub.Tensor(sub_72, mul_277);  sub_72 = mul_277 = None
        div_19: "f32[500, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 512);  rsqrt_15 = None
        mul_278: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(div_19, sub_73);  div_19 = sub_73 = None
        mul_279: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_153, mul_76);  mul_76 = None
        sum_73: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_279, [0, 1]);  mul_279 = None
        sum_74: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_153, [0, 1]);  view_153 = None
        
        # File: /app/src/models/feedforward.py:29 in forward, code: x = self.dropout1(x)
        convert_element_type_12: "f32[500, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_14, torch.float32);  gt_14 = None
        mul_280: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
        mul_281: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_278, mul_280);  mul_278 = mul_280 = None
        
        # File: /app/src/models/feedforward.py:28 in forward, code: x = self.act(x)
        mul_283: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(add_48, 0.5);  add_48 = None
        mul_284: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_60, view_60)
        mul_285: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_284, -0.5);  mul_284 = None
        exp_10: "f32[500, 128, 512]" = torch.ops.aten.exp.default(mul_285);  mul_285 = None
        mul_286: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
        mul_287: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_60, mul_286);  view_60 = mul_286 = None
        add_111: "f32[500, 128, 512]" = torch.ops.aten.add.Tensor(mul_283, mul_287);  mul_283 = mul_287 = None
        mul_288: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_281, add_111);  mul_281 = add_111 = None
        
        # File: /app/src/models/feedforward.py:27 in forward, code: x = self.linear1(x)
        view_154: "f32[64000, 512]" = torch.ops.aten.reshape.default(mul_288, [64000, 512]);  mul_288 = None
        permute_47: "f32[128, 512]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
        permute_168: "f32[512, 128]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
        mm_31: "f32[64000, 128]" = torch.ops.aten.mm.default(view_154, permute_168);  permute_168 = None
        permute_169: "f32[512, 64000]" = torch.ops.aten.permute.default(view_154, [1, 0])
        mm_32: "f32[512, 128]" = torch.ops.aten.mm.default(permute_169, view_59);  permute_169 = view_59 = None
        sum_75: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_154, [0], True);  view_154 = None
        view_155: "f32[512]" = torch.ops.aten.reshape.default(sum_75, [512]);  sum_75 = None
        view_156: "f32[500, 128, 128]" = torch.ops.aten.reshape.default(mm_31, [500, 128, 128]);  mm_31 = None
        
        # File: /app/src/models/feedforward.py:26 in forward, code: x = self.layernorm1(x)
        mul_290: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(view_156, primals_62);  primals_62 = None
        mul_291: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_290, 128)
        sum_76: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_290, [2], True)
        mul_292: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_290, mul_69);  mul_290 = None
        sum_77: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_292, [2], True);  mul_292 = None
        mul_293: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_69, sum_77);  sum_77 = None
        sub_75: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(mul_291, sum_76);  mul_291 = sum_76 = None
        sub_76: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(sub_75, mul_293);  sub_75 = mul_293 = None
        mul_294: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(div_20, sub_76);  div_20 = sub_76 = None
        mul_295: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(view_156, mul_69);  mul_69 = None
        sum_78: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_295, [0, 1]);  mul_295 = None
        sum_79: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_156, [0, 1]);  view_156 = None
        add_112: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(add_109, mul_294);  add_109 = mul_294 = None
        
        # File: /app/src/models/particle_transformer.py:46 in forward, code: x = self.dropout(x)
        convert_element_type_13: "f32[500, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_13, torch.float32);  gt_13 = None
        mul_296: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
        mul_297: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(add_112, mul_296);  mul_296 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        view_57: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(addmm_10, [128, 500, 128]);  addmm_10 = None
        permute_46: "f32[500, 128, 128]" = torch.ops.aten.permute.default(view_57, [1, 0, 2]);  view_57 = None
        
        # File: /app/src/models/particle_transformer.py:45 in forward, code: x = self.layernorm2(x)
        sub_77: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(permute_46, getitem_27);  permute_46 = getitem_27 = None
        mul_298: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_13);  sub_77 = None
        mul_299: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_297, primals_60);  primals_60 = None
        mul_300: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_299, 128)
        sum_80: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_299, [2], True)
        mul_301: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_299, mul_298);  mul_299 = None
        sum_81: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_301, [2], True);  mul_301 = None
        mul_302: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_298, sum_81);  sum_81 = None
        sub_78: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(mul_300, sum_80);  mul_300 = sum_80 = None
        sub_79: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(sub_78, mul_302);  sub_78 = mul_302 = None
        div_21: "f32[500, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 128);  rsqrt_13 = None
        mul_303: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(div_21, sub_79);  div_21 = sub_79 = None
        mul_304: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_297, mul_298);  mul_298 = None
        sum_82: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_304, [0, 1]);  mul_304 = None
        sum_83: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_297, [0, 1]);  mul_297 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        permute_172: "f32[128, 500, 128]" = torch.ops.aten.permute.default(mul_303, [1, 0, 2]);  mul_303 = None
        clone_61: "f32[128, 500, 128]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
        view_157: "f32[64000, 128]" = torch.ops.aten.reshape.default(clone_61, [64000, 128]);  clone_61 = None
        permute_45: "f32[128, 128]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
        permute_173: "f32[128, 128]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        mm_33: "f32[64000, 128]" = torch.ops.aten.mm.default(view_157, permute_173);  permute_173 = None
        permute_174: "f32[128, 64000]" = torch.ops.aten.permute.default(view_157, [1, 0])
        mm_34: "f32[128, 128]" = torch.ops.aten.mm.default(permute_174, view_56);  permute_174 = view_56 = None
        sum_84: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_157, [0], True);  view_157 = None
        view_158: "f32[128]" = torch.ops.aten.reshape.default(sum_84, [128]);  sum_84 = None
        view_159: "f32[128, 4000, 16]" = torch.ops.aten.reshape.default(mm_33, [128, 4000, 16]);  mm_33 = None
        permute_177: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(view_159, [1, 0, 2]);  view_159 = None
        bmm_19: "f32[4000, 128, 16]" = torch.ops.aten.bmm.default(permute_178, permute_177);  permute_178 = None
        bmm_20: "f32[4000, 128, 128]" = torch.ops.aten.bmm.default(permute_177, permute_179);  permute_177 = permute_179 = None
        convert_element_type_14: "f32[4000, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_12, torch.float32);  gt_12 = None
        mul_305: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
        mul_306: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_20, mul_305);  bmm_20 = mul_305 = None
        sub_16: "f32[4000, 128, 128]" = torch.ops.aten.sub.Tensor(baddbmm_3, amax_3);  baddbmm_3 = amax_3 = None
        exp_3: "f32[4000, 128, 128]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        div_3: "f32[4000, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        mul_307: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(mul_306, div_3);  mul_306 = None
        sum_85: "f32[4000, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_307, [-1], True)
        neg_3: "f32[4000, 128, 128]" = torch.ops.aten.neg.default(div_3);  div_3 = None
        fma_3: "f32[4000, 128, 128]" = torch.ops.prims.fma.default(neg_3, sum_85, mul_307);  neg_3 = sum_85 = mul_307 = None
        bmm_21: "f32[4000, 128, 16]" = torch.ops.aten.bmm.default(fma_3, permute_180);  permute_180 = None
        bmm_22: "f32[4000, 16, 128]" = torch.ops.aten.bmm.default(permute_181, fma_3);  permute_181 = None
        permute_182: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(bmm_22, [0, 2, 1]);  bmm_22 = None
        mul_308: "f32[4000, 128, 16]" = torch.ops.aten.mul.Tensor(bmm_21, 0.25);  bmm_21 = None
        add_113: "f32[4000, 128, 128]" = torch.ops.aten.add.Tensor(add_106, fma_3);  add_106 = fma_3 = None
        permute_183: "f32[128, 4000, 16]" = torch.ops.aten.permute.default(bmm_19, [1, 0, 2]);  bmm_19 = None
        clone_63: "f32[128, 4000, 16]" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
        view_160: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(clone_63, [128, 500, 128]);  clone_63 = None
        permute_184: "f32[128, 4000, 16]" = torch.ops.aten.permute.default(permute_182, [1, 0, 2]);  permute_182 = None
        view_161: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(permute_184, [128, 500, 128]);  permute_184 = None
        permute_185: "f32[128, 4000, 16]" = torch.ops.aten.permute.default(mul_308, [1, 0, 2]);  mul_308 = None
        clone_64: "f32[128, 4000, 16]" = torch.ops.aten.clone.default(permute_185, memory_format = torch.contiguous_format);  permute_185 = None
        view_162: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(clone_64, [128, 500, 128]);  clone_64 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_9: "f32[3, 128, 500, 128]" = torch.ops.aten.select_scatter.default(full_default, view_160, 0, 2);  view_160 = None
        select_scatter_default_10: "f32[3, 128, 500, 128]" = torch.ops.aten.select_scatter.default(full_default, view_161, 0, 1);  view_161 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        add_114: "f32[3, 128, 500, 128]" = torch.ops.aten.add.Tensor(select_scatter_default_9, select_scatter_default_10);  select_scatter_default_9 = select_scatter_default_10 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_11: "f32[3, 128, 500, 128]" = torch.ops.aten.select_scatter.default(full_default, view_162, 0, 0);  view_162 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        add_115: "f32[3, 128, 500, 128]" = torch.ops.aten.add.Tensor(add_114, select_scatter_default_11);  add_114 = select_scatter_default_11 = None
        unsqueeze_10: "f32[3, 128, 500, 1, 128]" = torch.ops.aten.unsqueeze.default(add_115, 3);  add_115 = None
        permute_186: "f32[1, 128, 500, 3, 128]" = torch.ops.aten.permute.default(unsqueeze_10, [3, 1, 2, 0, 4]);  unsqueeze_10 = None
        squeeze_10: "f32[128, 500, 3, 128]" = torch.ops.aten.squeeze.dim(permute_186, 0);  permute_186 = None
        clone_65: "f32[128, 500, 3, 128]" = torch.ops.aten.clone.default(squeeze_10, memory_format = torch.contiguous_format);  squeeze_10 = None
        view_163: "f32[128, 500, 384]" = torch.ops.aten.reshape.default(clone_65, [128, 500, 384]);  clone_65 = None
        sum_86: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(view_163, [0, 1], True)
        view_164: "f32[384]" = torch.ops.aten.reshape.default(sum_86, [384]);  sum_86 = None
        view_165: "f32[64000, 384]" = torch.ops.aten.reshape.default(view_163, [64000, 384]);  view_163 = None
        permute_187: "f32[384, 64000]" = torch.ops.aten.permute.default(view_165, [1, 0])
        mm_35: "f32[384, 128]" = torch.ops.aten.mm.default(permute_187, view_48);  permute_187 = view_48 = None
        permute_38: "f32[128, 384]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
        permute_189: "f32[384, 128]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
        mm_36: "f32[64000, 128]" = torch.ops.aten.mm.default(view_165, permute_189);  view_165 = permute_189 = None
        view_166: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(mm_36, [128, 500, 128]);  mm_36 = None
        permute_191: "f32[500, 128, 128]" = torch.ops.aten.permute.default(view_166, [1, 0, 2]);  view_166 = None
        
        # File: /app/src/models/particle_transformer.py:43 in forward, code: x = self.layernorm1(x)
        mul_310: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(permute_191, primals_54);  primals_54 = None
        mul_311: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_310, 128)
        sum_87: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_310, [2], True)
        mul_312: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_310, mul_60);  mul_310 = None
        sum_88: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_312, [2], True);  mul_312 = None
        mul_313: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_60, sum_88);  sum_88 = None
        sub_81: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(mul_311, sum_87);  mul_311 = sum_87 = None
        sub_82: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(sub_81, mul_313);  sub_81 = mul_313 = None
        mul_314: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(div_22, sub_82);  div_22 = sub_82 = None
        mul_315: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(permute_191, mul_60);  mul_60 = None
        sum_89: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_315, [0, 1]);  mul_315 = None
        sum_90: "f32[128]" = torch.ops.aten.sum.dim_IntList(permute_191, [0, 1]);  permute_191 = None
        add_116: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(add_112, mul_314);  add_112 = mul_314 = None
        
        # File: /app/src/models/feedforward.py:33 in forward, code: x = self.dropout2(x)
        convert_element_type_15: "f32[500, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_11, torch.float32);  gt_11 = None
        mul_316: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
        mul_317: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(add_116, mul_316);  mul_316 = None
        
        # File: /app/src/models/feedforward.py:32 in forward, code: x = self.linear2(x)
        view_167: "f32[64000, 128]" = torch.ops.aten.reshape.default(mul_317, [64000, 128]);  mul_317 = None
        permute_36: "f32[512, 128]" = torch.ops.aten.permute.default(primals_52, [1, 0]);  primals_52 = None
        permute_192: "f32[128, 512]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
        mm_37: "f32[64000, 512]" = torch.ops.aten.mm.default(view_167, permute_192);  permute_192 = None
        permute_193: "f32[128, 64000]" = torch.ops.aten.permute.default(view_167, [1, 0])
        mm_38: "f32[128, 512]" = torch.ops.aten.mm.default(permute_193, view_46);  permute_193 = view_46 = None
        sum_91: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_167, [0], True);  view_167 = None
        view_168: "f32[128]" = torch.ops.aten.reshape.default(sum_91, [128]);  sum_91 = None
        view_169: "f32[500, 128, 512]" = torch.ops.aten.reshape.default(mm_37, [500, 128, 512]);  mm_37 = None
        
        # File: /app/src/models/feedforward.py:31 in forward, code: x = self.layernorm2(x)
        mul_319: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_169, primals_50);  primals_50 = None
        mul_320: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_319, 512)
        sum_92: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_319, [2], True)
        
        # File: /app/src/models/feedforward.py:27 in forward, code: x = self.linear1(x)
        view_45: "f32[500, 128, 512]" = torch.ops.aten.reshape.default(addmm_8, [500, 128, 512]);  addmm_8 = None
        
        # File: /app/src/models/feedforward.py:28 in forward, code: x = self.act(x)
        mul_51: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_45, 0.5)
        mul_52: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_45, 0.7071067811865476)
        erf_2: "f32[500, 128, 512]" = torch.ops.aten.erf.default(mul_52);  mul_52 = None
        add_35: "f32[500, 128, 512]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_53: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_51, add_35);  mul_51 = None
        
        # File: /app/src/models/feedforward.py:29 in forward, code: x = self.dropout1(x)
        mul_54: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(gt_10, mul_53);  mul_53 = None
        mul_55: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_54, 1.1111111111111112);  mul_54 = None
        
        # File: /app/src/models/feedforward.py:31 in forward, code: x = self.layernorm2(x)
        sub_14: "f32[500, 128, 512]" = torch.ops.aten.sub.Tensor(mul_55, getitem_23);  mul_55 = getitem_23 = None
        mul_56: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_11);  sub_14 = None
        mul_321: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_319, mul_56);  mul_319 = None
        sum_93: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_321, [2], True);  mul_321 = None
        mul_322: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_56, sum_93);  sum_93 = None
        sub_84: "f32[500, 128, 512]" = torch.ops.aten.sub.Tensor(mul_320, sum_92);  mul_320 = sum_92 = None
        sub_85: "f32[500, 128, 512]" = torch.ops.aten.sub.Tensor(sub_84, mul_322);  sub_84 = mul_322 = None
        div_23: "f32[500, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 512);  rsqrt_11 = None
        mul_323: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(div_23, sub_85);  div_23 = sub_85 = None
        mul_324: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_169, mul_56);  mul_56 = None
        sum_94: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_324, [0, 1]);  mul_324 = None
        sum_95: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_169, [0, 1]);  view_169 = None
        
        # File: /app/src/models/feedforward.py:29 in forward, code: x = self.dropout1(x)
        convert_element_type_16: "f32[500, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_10, torch.float32);  gt_10 = None
        mul_325: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
        mul_326: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_323, mul_325);  mul_323 = mul_325 = None
        
        # File: /app/src/models/feedforward.py:28 in forward, code: x = self.act(x)
        mul_328: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(add_35, 0.5);  add_35 = None
        mul_329: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_45, view_45)
        mul_330: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_329, -0.5);  mul_329 = None
        exp_11: "f32[500, 128, 512]" = torch.ops.aten.exp.default(mul_330);  mul_330 = None
        mul_331: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
        mul_332: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_45, mul_331);  view_45 = mul_331 = None
        add_118: "f32[500, 128, 512]" = torch.ops.aten.add.Tensor(mul_328, mul_332);  mul_328 = mul_332 = None
        mul_333: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_326, add_118);  mul_326 = add_118 = None
        
        # File: /app/src/models/feedforward.py:27 in forward, code: x = self.linear1(x)
        view_170: "f32[64000, 512]" = torch.ops.aten.reshape.default(mul_333, [64000, 512]);  mul_333 = None
        permute_35: "f32[128, 512]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
        permute_196: "f32[512, 128]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        mm_39: "f32[64000, 128]" = torch.ops.aten.mm.default(view_170, permute_196);  permute_196 = None
        permute_197: "f32[512, 64000]" = torch.ops.aten.permute.default(view_170, [1, 0])
        mm_40: "f32[512, 128]" = torch.ops.aten.mm.default(permute_197, view_44);  permute_197 = view_44 = None
        sum_96: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_170, [0], True);  view_170 = None
        view_171: "f32[512]" = torch.ops.aten.reshape.default(sum_96, [512]);  sum_96 = None
        view_172: "f32[500, 128, 128]" = torch.ops.aten.reshape.default(mm_39, [500, 128, 128]);  mm_39 = None
        
        # File: /app/src/models/feedforward.py:26 in forward, code: x = self.layernorm1(x)
        mul_335: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(view_172, primals_46);  primals_46 = None
        mul_336: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_335, 128)
        sum_97: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_335, [2], True)
        mul_337: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_335, mul_49);  mul_335 = None
        sum_98: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_337, [2], True);  mul_337 = None
        mul_338: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_49, sum_98);  sum_98 = None
        sub_87: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(mul_336, sum_97);  mul_336 = sum_97 = None
        sub_88: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(sub_87, mul_338);  sub_87 = mul_338 = None
        mul_339: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(div_24, sub_88);  div_24 = sub_88 = None
        mul_340: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(view_172, mul_49);  mul_49 = None
        sum_99: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_340, [0, 1]);  mul_340 = None
        sum_100: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_172, [0, 1]);  view_172 = None
        add_119: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(add_116, mul_339);  add_116 = mul_339 = None
        
        # File: /app/src/models/particle_transformer.py:46 in forward, code: x = self.dropout(x)
        convert_element_type_17: "f32[500, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_9, torch.float32);  gt_9 = None
        mul_341: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
        mul_342: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(add_119, mul_341);  mul_341 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        view_42: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(addmm_7, [128, 500, 128]);  addmm_7 = None
        permute_34: "f32[500, 128, 128]" = torch.ops.aten.permute.default(view_42, [1, 0, 2]);  view_42 = None
        
        # File: /app/src/models/particle_transformer.py:45 in forward, code: x = self.layernorm2(x)
        sub_89: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(permute_34, getitem_19);  permute_34 = getitem_19 = None
        mul_343: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_9);  sub_89 = None
        mul_344: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_342, primals_44);  primals_44 = None
        mul_345: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_344, 128)
        sum_101: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_344, [2], True)
        mul_346: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_344, mul_343);  mul_344 = None
        sum_102: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_346, [2], True);  mul_346 = None
        mul_347: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_343, sum_102);  sum_102 = None
        sub_90: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(mul_345, sum_101);  mul_345 = sum_101 = None
        sub_91: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(sub_90, mul_347);  sub_90 = mul_347 = None
        div_25: "f32[500, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 128);  rsqrt_9 = None
        mul_348: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(div_25, sub_91);  div_25 = sub_91 = None
        mul_349: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_342, mul_343);  mul_343 = None
        sum_103: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_349, [0, 1]);  mul_349 = None
        sum_104: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_342, [0, 1]);  mul_342 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        permute_200: "f32[128, 500, 128]" = torch.ops.aten.permute.default(mul_348, [1, 0, 2]);  mul_348 = None
        clone_69: "f32[128, 500, 128]" = torch.ops.aten.clone.default(permute_200, memory_format = torch.contiguous_format);  permute_200 = None
        view_173: "f32[64000, 128]" = torch.ops.aten.reshape.default(clone_69, [64000, 128]);  clone_69 = None
        permute_33: "f32[128, 128]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
        permute_201: "f32[128, 128]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        mm_41: "f32[64000, 128]" = torch.ops.aten.mm.default(view_173, permute_201);  permute_201 = None
        permute_202: "f32[128, 64000]" = torch.ops.aten.permute.default(view_173, [1, 0])
        mm_42: "f32[128, 128]" = torch.ops.aten.mm.default(permute_202, view_41);  permute_202 = view_41 = None
        sum_105: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_173, [0], True);  view_173 = None
        view_174: "f32[128]" = torch.ops.aten.reshape.default(sum_105, [128]);  sum_105 = None
        view_175: "f32[128, 4000, 16]" = torch.ops.aten.reshape.default(mm_41, [128, 4000, 16]);  mm_41 = None
        permute_205: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(view_175, [1, 0, 2]);  view_175 = None
        bmm_23: "f32[4000, 128, 16]" = torch.ops.aten.bmm.default(permute_206, permute_205);  permute_206 = None
        bmm_24: "f32[4000, 128, 128]" = torch.ops.aten.bmm.default(permute_205, permute_207);  permute_205 = permute_207 = None
        convert_element_type_18: "f32[4000, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_8, torch.float32);  gt_8 = None
        mul_350: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
        mul_351: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_24, mul_350);  bmm_24 = mul_350 = None
        sub_11: "f32[4000, 128, 128]" = torch.ops.aten.sub.Tensor(baddbmm_2, amax_2);  baddbmm_2 = amax_2 = None
        exp_2: "f32[4000, 128, 128]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        div_2: "f32[4000, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        mul_352: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(mul_351, div_2);  mul_351 = None
        sum_106: "f32[4000, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_352, [-1], True)
        neg_4: "f32[4000, 128, 128]" = torch.ops.aten.neg.default(div_2);  div_2 = None
        fma_4: "f32[4000, 128, 128]" = torch.ops.prims.fma.default(neg_4, sum_106, mul_352);  neg_4 = sum_106 = mul_352 = None
        bmm_25: "f32[4000, 128, 16]" = torch.ops.aten.bmm.default(fma_4, permute_208);  permute_208 = None
        bmm_26: "f32[4000, 16, 128]" = torch.ops.aten.bmm.default(permute_209, fma_4);  permute_209 = None
        permute_210: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(bmm_26, [0, 2, 1]);  bmm_26 = None
        mul_353: "f32[4000, 128, 16]" = torch.ops.aten.mul.Tensor(bmm_25, 0.25);  bmm_25 = None
        add_120: "f32[4000, 128, 128]" = torch.ops.aten.add.Tensor(add_113, fma_4);  add_113 = fma_4 = None
        permute_211: "f32[128, 4000, 16]" = torch.ops.aten.permute.default(bmm_23, [1, 0, 2]);  bmm_23 = None
        clone_71: "f32[128, 4000, 16]" = torch.ops.aten.clone.default(permute_211, memory_format = torch.contiguous_format);  permute_211 = None
        view_176: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(clone_71, [128, 500, 128]);  clone_71 = None
        permute_212: "f32[128, 4000, 16]" = torch.ops.aten.permute.default(permute_210, [1, 0, 2]);  permute_210 = None
        view_177: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(permute_212, [128, 500, 128]);  permute_212 = None
        permute_213: "f32[128, 4000, 16]" = torch.ops.aten.permute.default(mul_353, [1, 0, 2]);  mul_353 = None
        clone_72: "f32[128, 4000, 16]" = torch.ops.aten.clone.default(permute_213, memory_format = torch.contiguous_format);  permute_213 = None
        view_178: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(clone_72, [128, 500, 128]);  clone_72 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_12: "f32[3, 128, 500, 128]" = torch.ops.aten.select_scatter.default(full_default, view_176, 0, 2);  view_176 = None
        select_scatter_default_13: "f32[3, 128, 500, 128]" = torch.ops.aten.select_scatter.default(full_default, view_177, 0, 1);  view_177 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        add_121: "f32[3, 128, 500, 128]" = torch.ops.aten.add.Tensor(select_scatter_default_12, select_scatter_default_13);  select_scatter_default_12 = select_scatter_default_13 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_14: "f32[3, 128, 500, 128]" = torch.ops.aten.select_scatter.default(full_default, view_178, 0, 0);  view_178 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        add_122: "f32[3, 128, 500, 128]" = torch.ops.aten.add.Tensor(add_121, select_scatter_default_14);  add_121 = select_scatter_default_14 = None
        unsqueeze_11: "f32[3, 128, 500, 1, 128]" = torch.ops.aten.unsqueeze.default(add_122, 3);  add_122 = None
        permute_214: "f32[1, 128, 500, 3, 128]" = torch.ops.aten.permute.default(unsqueeze_11, [3, 1, 2, 0, 4]);  unsqueeze_11 = None
        squeeze_11: "f32[128, 500, 3, 128]" = torch.ops.aten.squeeze.dim(permute_214, 0);  permute_214 = None
        clone_73: "f32[128, 500, 3, 128]" = torch.ops.aten.clone.default(squeeze_11, memory_format = torch.contiguous_format);  squeeze_11 = None
        view_179: "f32[128, 500, 384]" = torch.ops.aten.reshape.default(clone_73, [128, 500, 384]);  clone_73 = None
        sum_107: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(view_179, [0, 1], True)
        view_180: "f32[384]" = torch.ops.aten.reshape.default(sum_107, [384]);  sum_107 = None
        view_181: "f32[64000, 384]" = torch.ops.aten.reshape.default(view_179, [64000, 384]);  view_179 = None
        permute_215: "f32[384, 64000]" = torch.ops.aten.permute.default(view_181, [1, 0])
        mm_43: "f32[384, 128]" = torch.ops.aten.mm.default(permute_215, view_33);  permute_215 = view_33 = None
        permute_26: "f32[128, 384]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
        permute_217: "f32[384, 128]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
        mm_44: "f32[64000, 128]" = torch.ops.aten.mm.default(view_181, permute_217);  view_181 = permute_217 = None
        view_182: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(mm_44, [128, 500, 128]);  mm_44 = None
        permute_219: "f32[500, 128, 128]" = torch.ops.aten.permute.default(view_182, [1, 0, 2]);  view_182 = None
        
        # File: /app/src/models/particle_transformer.py:43 in forward, code: x = self.layernorm1(x)
        mul_355: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(permute_219, primals_38);  primals_38 = None
        mul_356: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_355, 128)
        sum_108: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_355, [2], True)
        mul_357: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_355, mul_40);  mul_355 = None
        sum_109: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_357, [2], True);  mul_357 = None
        mul_358: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_40, sum_109);  sum_109 = None
        sub_93: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(mul_356, sum_108);  mul_356 = sum_108 = None
        sub_94: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(sub_93, mul_358);  sub_93 = mul_358 = None
        mul_359: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(div_26, sub_94);  div_26 = sub_94 = None
        mul_360: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(permute_219, mul_40);  mul_40 = None
        sum_110: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_360, [0, 1]);  mul_360 = None
        sum_111: "f32[128]" = torch.ops.aten.sum.dim_IntList(permute_219, [0, 1]);  permute_219 = None
        add_123: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(add_119, mul_359);  add_119 = mul_359 = None
        
        # File: /app/src/models/feedforward.py:33 in forward, code: x = self.dropout2(x)
        convert_element_type_19: "f32[500, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_7, torch.float32);  gt_7 = None
        mul_361: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
        mul_362: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(add_123, mul_361);  mul_361 = None
        
        # File: /app/src/models/feedforward.py:32 in forward, code: x = self.linear2(x)
        view_183: "f32[64000, 128]" = torch.ops.aten.reshape.default(mul_362, [64000, 128]);  mul_362 = None
        permute_24: "f32[512, 128]" = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
        permute_220: "f32[128, 512]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        mm_45: "f32[64000, 512]" = torch.ops.aten.mm.default(view_183, permute_220);  permute_220 = None
        permute_221: "f32[128, 64000]" = torch.ops.aten.permute.default(view_183, [1, 0])
        mm_46: "f32[128, 512]" = torch.ops.aten.mm.default(permute_221, view_31);  permute_221 = view_31 = None
        sum_112: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_183, [0], True);  view_183 = None
        view_184: "f32[128]" = torch.ops.aten.reshape.default(sum_112, [128]);  sum_112 = None
        view_185: "f32[500, 128, 512]" = torch.ops.aten.reshape.default(mm_45, [500, 128, 512]);  mm_45 = None
        
        # File: /app/src/models/feedforward.py:31 in forward, code: x = self.layernorm2(x)
        mul_364: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_185, primals_34);  primals_34 = None
        mul_365: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_364, 512)
        sum_113: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_364, [2], True)
        
        # File: /app/src/models/feedforward.py:27 in forward, code: x = self.linear1(x)
        view_30: "f32[500, 128, 512]" = torch.ops.aten.reshape.default(addmm_5, [500, 128, 512]);  addmm_5 = None
        
        # File: /app/src/models/feedforward.py:28 in forward, code: x = self.act(x)
        mul_31: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_30, 0.5)
        mul_32: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_30, 0.7071067811865476)
        erf_1: "f32[500, 128, 512]" = torch.ops.aten.erf.default(mul_32);  mul_32 = None
        add_22: "f32[500, 128, 512]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_33: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_31, add_22);  mul_31 = None
        
        # File: /app/src/models/feedforward.py:29 in forward, code: x = self.dropout1(x)
        mul_34: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(gt_6, mul_33);  mul_33 = None
        mul_35: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_34, 1.1111111111111112);  mul_34 = None
        
        # File: /app/src/models/feedforward.py:31 in forward, code: x = self.layernorm2(x)
        sub_9: "f32[500, 128, 512]" = torch.ops.aten.sub.Tensor(mul_35, getitem_15);  mul_35 = getitem_15 = None
        mul_36: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_7);  sub_9 = None
        mul_366: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_364, mul_36);  mul_364 = None
        sum_114: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_366, [2], True);  mul_366 = None
        mul_367: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_36, sum_114);  sum_114 = None
        sub_96: "f32[500, 128, 512]" = torch.ops.aten.sub.Tensor(mul_365, sum_113);  mul_365 = sum_113 = None
        sub_97: "f32[500, 128, 512]" = torch.ops.aten.sub.Tensor(sub_96, mul_367);  sub_96 = mul_367 = None
        div_27: "f32[500, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 512);  rsqrt_7 = None
        mul_368: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(div_27, sub_97);  div_27 = sub_97 = None
        mul_369: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_185, mul_36);  mul_36 = None
        sum_115: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_369, [0, 1]);  mul_369 = None
        sum_116: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_185, [0, 1]);  view_185 = None
        
        # File: /app/src/models/feedforward.py:29 in forward, code: x = self.dropout1(x)
        convert_element_type_20: "f32[500, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_6, torch.float32);  gt_6 = None
        mul_370: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
        mul_371: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_368, mul_370);  mul_368 = mul_370 = None
        
        # File: /app/src/models/feedforward.py:28 in forward, code: x = self.act(x)
        mul_373: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(add_22, 0.5);  add_22 = None
        mul_374: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_30, view_30)
        mul_375: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_374, -0.5);  mul_374 = None
        exp_12: "f32[500, 128, 512]" = torch.ops.aten.exp.default(mul_375);  mul_375 = None
        mul_376: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
        mul_377: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_30, mul_376);  view_30 = mul_376 = None
        add_125: "f32[500, 128, 512]" = torch.ops.aten.add.Tensor(mul_373, mul_377);  mul_373 = mul_377 = None
        mul_378: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_371, add_125);  mul_371 = add_125 = None
        
        # File: /app/src/models/feedforward.py:27 in forward, code: x = self.linear1(x)
        view_186: "f32[64000, 512]" = torch.ops.aten.reshape.default(mul_378, [64000, 512]);  mul_378 = None
        permute_23: "f32[128, 512]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
        permute_224: "f32[512, 128]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        mm_47: "f32[64000, 128]" = torch.ops.aten.mm.default(view_186, permute_224);  permute_224 = None
        permute_225: "f32[512, 64000]" = torch.ops.aten.permute.default(view_186, [1, 0])
        mm_48: "f32[512, 128]" = torch.ops.aten.mm.default(permute_225, view_29);  permute_225 = view_29 = None
        sum_117: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_186, [0], True);  view_186 = None
        view_187: "f32[512]" = torch.ops.aten.reshape.default(sum_117, [512]);  sum_117 = None
        view_188: "f32[500, 128, 128]" = torch.ops.aten.reshape.default(mm_47, [500, 128, 128]);  mm_47 = None
        
        # File: /app/src/models/feedforward.py:26 in forward, code: x = self.layernorm1(x)
        mul_380: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(view_188, primals_30);  primals_30 = None
        mul_381: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_380, 128)
        sum_118: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_380, [2], True)
        mul_382: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_380, mul_29);  mul_380 = None
        sum_119: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_382, [2], True);  mul_382 = None
        mul_383: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_29, sum_119);  sum_119 = None
        sub_99: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(mul_381, sum_118);  mul_381 = sum_118 = None
        sub_100: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(sub_99, mul_383);  sub_99 = mul_383 = None
        mul_384: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(div_28, sub_100);  div_28 = sub_100 = None
        mul_385: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(view_188, mul_29);  mul_29 = None
        sum_120: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_385, [0, 1]);  mul_385 = None
        sum_121: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_188, [0, 1]);  view_188 = None
        add_126: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(add_123, mul_384);  add_123 = mul_384 = None
        
        # File: /app/src/models/particle_transformer.py:46 in forward, code: x = self.dropout(x)
        convert_element_type_21: "f32[500, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_5, torch.float32);  gt_5 = None
        mul_386: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
        mul_387: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(add_126, mul_386);  mul_386 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        view_27: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(addmm_4, [128, 500, 128]);  addmm_4 = None
        permute_22: "f32[500, 128, 128]" = torch.ops.aten.permute.default(view_27, [1, 0, 2]);  view_27 = None
        
        # File: /app/src/models/particle_transformer.py:45 in forward, code: x = self.layernorm2(x)
        sub_101: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(permute_22, getitem_11);  permute_22 = getitem_11 = None
        mul_388: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub_101, rsqrt_5);  sub_101 = None
        mul_389: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_387, primals_28);  primals_28 = None
        mul_390: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_389, 128)
        sum_122: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_389, [2], True)
        mul_391: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_389, mul_388);  mul_389 = None
        sum_123: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_391, [2], True);  mul_391 = None
        mul_392: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_388, sum_123);  sum_123 = None
        sub_102: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(mul_390, sum_122);  mul_390 = sum_122 = None
        sub_103: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(sub_102, mul_392);  sub_102 = mul_392 = None
        div_29: "f32[500, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 128);  rsqrt_5 = None
        mul_393: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(div_29, sub_103);  div_29 = sub_103 = None
        mul_394: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_387, mul_388);  mul_388 = None
        sum_124: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_394, [0, 1]);  mul_394 = None
        sum_125: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_387, [0, 1]);  mul_387 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        permute_228: "f32[128, 500, 128]" = torch.ops.aten.permute.default(mul_393, [1, 0, 2]);  mul_393 = None
        clone_77: "f32[128, 500, 128]" = torch.ops.aten.clone.default(permute_228, memory_format = torch.contiguous_format);  permute_228 = None
        view_189: "f32[64000, 128]" = torch.ops.aten.reshape.default(clone_77, [64000, 128]);  clone_77 = None
        permute_21: "f32[128, 128]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
        permute_229: "f32[128, 128]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        mm_49: "f32[64000, 128]" = torch.ops.aten.mm.default(view_189, permute_229);  permute_229 = None
        permute_230: "f32[128, 64000]" = torch.ops.aten.permute.default(view_189, [1, 0])
        mm_50: "f32[128, 128]" = torch.ops.aten.mm.default(permute_230, view_26);  permute_230 = view_26 = None
        sum_126: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_189, [0], True);  view_189 = None
        view_190: "f32[128]" = torch.ops.aten.reshape.default(sum_126, [128]);  sum_126 = None
        view_191: "f32[128, 4000, 16]" = torch.ops.aten.reshape.default(mm_49, [128, 4000, 16]);  mm_49 = None
        permute_233: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(view_191, [1, 0, 2]);  view_191 = None
        bmm_27: "f32[4000, 128, 16]" = torch.ops.aten.bmm.default(permute_234, permute_233);  permute_234 = None
        bmm_28: "f32[4000, 128, 128]" = torch.ops.aten.bmm.default(permute_233, permute_235);  permute_233 = permute_235 = None
        convert_element_type_22: "f32[4000, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_4, torch.float32);  gt_4 = None
        mul_395: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
        mul_396: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_28, mul_395);  bmm_28 = mul_395 = None
        sub_6: "f32[4000, 128, 128]" = torch.ops.aten.sub.Tensor(baddbmm_1, amax_1);  baddbmm_1 = amax_1 = None
        exp_1: "f32[4000, 128, 128]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        div_1: "f32[4000, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        mul_397: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(mul_396, div_1);  mul_396 = None
        sum_127: "f32[4000, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_397, [-1], True)
        neg_5: "f32[4000, 128, 128]" = torch.ops.aten.neg.default(div_1);  div_1 = None
        fma_5: "f32[4000, 128, 128]" = torch.ops.prims.fma.default(neg_5, sum_127, mul_397);  neg_5 = sum_127 = mul_397 = None
        bmm_29: "f32[4000, 128, 16]" = torch.ops.aten.bmm.default(fma_5, permute_236);  permute_236 = None
        bmm_30: "f32[4000, 16, 128]" = torch.ops.aten.bmm.default(permute_237, fma_5);  permute_237 = None
        permute_238: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(bmm_30, [0, 2, 1]);  bmm_30 = None
        mul_398: "f32[4000, 128, 16]" = torch.ops.aten.mul.Tensor(bmm_29, 0.25);  bmm_29 = None
        add_127: "f32[4000, 128, 128]" = torch.ops.aten.add.Tensor(add_120, fma_5);  add_120 = fma_5 = None
        permute_239: "f32[128, 4000, 16]" = torch.ops.aten.permute.default(bmm_27, [1, 0, 2]);  bmm_27 = None
        clone_79: "f32[128, 4000, 16]" = torch.ops.aten.clone.default(permute_239, memory_format = torch.contiguous_format);  permute_239 = None
        view_192: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(clone_79, [128, 500, 128]);  clone_79 = None
        permute_240: "f32[128, 4000, 16]" = torch.ops.aten.permute.default(permute_238, [1, 0, 2]);  permute_238 = None
        view_193: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(permute_240, [128, 500, 128]);  permute_240 = None
        permute_241: "f32[128, 4000, 16]" = torch.ops.aten.permute.default(mul_398, [1, 0, 2]);  mul_398 = None
        clone_80: "f32[128, 4000, 16]" = torch.ops.aten.clone.default(permute_241, memory_format = torch.contiguous_format);  permute_241 = None
        view_194: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(clone_80, [128, 500, 128]);  clone_80 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_15: "f32[3, 128, 500, 128]" = torch.ops.aten.select_scatter.default(full_default, view_192, 0, 2);  view_192 = None
        select_scatter_default_16: "f32[3, 128, 500, 128]" = torch.ops.aten.select_scatter.default(full_default, view_193, 0, 1);  view_193 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        add_128: "f32[3, 128, 500, 128]" = torch.ops.aten.add.Tensor(select_scatter_default_15, select_scatter_default_16);  select_scatter_default_15 = select_scatter_default_16 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_17: "f32[3, 128, 500, 128]" = torch.ops.aten.select_scatter.default(full_default, view_194, 0, 0);  view_194 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        add_129: "f32[3, 128, 500, 128]" = torch.ops.aten.add.Tensor(add_128, select_scatter_default_17);  add_128 = select_scatter_default_17 = None
        unsqueeze_12: "f32[3, 128, 500, 1, 128]" = torch.ops.aten.unsqueeze.default(add_129, 3);  add_129 = None
        permute_242: "f32[1, 128, 500, 3, 128]" = torch.ops.aten.permute.default(unsqueeze_12, [3, 1, 2, 0, 4]);  unsqueeze_12 = None
        squeeze_12: "f32[128, 500, 3, 128]" = torch.ops.aten.squeeze.dim(permute_242, 0);  permute_242 = None
        clone_81: "f32[128, 500, 3, 128]" = torch.ops.aten.clone.default(squeeze_12, memory_format = torch.contiguous_format);  squeeze_12 = None
        view_195: "f32[128, 500, 384]" = torch.ops.aten.reshape.default(clone_81, [128, 500, 384]);  clone_81 = None
        sum_128: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(view_195, [0, 1], True)
        view_196: "f32[384]" = torch.ops.aten.reshape.default(sum_128, [384]);  sum_128 = None
        view_197: "f32[64000, 384]" = torch.ops.aten.reshape.default(view_195, [64000, 384]);  view_195 = None
        permute_243: "f32[384, 64000]" = torch.ops.aten.permute.default(view_197, [1, 0])
        mm_51: "f32[384, 128]" = torch.ops.aten.mm.default(permute_243, view_18);  permute_243 = view_18 = None
        permute_14: "f32[128, 384]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
        permute_245: "f32[384, 128]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
        mm_52: "f32[64000, 128]" = torch.ops.aten.mm.default(view_197, permute_245);  view_197 = permute_245 = None
        view_198: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(mm_52, [128, 500, 128]);  mm_52 = None
        permute_247: "f32[500, 128, 128]" = torch.ops.aten.permute.default(view_198, [1, 0, 2]);  view_198 = None
        
        # File: /app/src/models/particle_transformer.py:43 in forward, code: x = self.layernorm1(x)
        mul_400: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(permute_247, primals_22);  primals_22 = None
        mul_401: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_400, 128)
        sum_129: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_400, [2], True)
        mul_402: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_400, mul_20);  mul_400 = None
        sum_130: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_402, [2], True);  mul_402 = None
        mul_403: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_20, sum_130);  sum_130 = None
        sub_105: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(mul_401, sum_129);  mul_401 = sum_129 = None
        sub_106: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(sub_105, mul_403);  sub_105 = mul_403 = None
        mul_404: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(div_30, sub_106);  div_30 = sub_106 = None
        mul_405: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(permute_247, mul_20);  mul_20 = None
        sum_131: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_405, [0, 1]);  mul_405 = None
        sum_132: "f32[128]" = torch.ops.aten.sum.dim_IntList(permute_247, [0, 1]);  permute_247 = None
        add_130: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(add_126, mul_404);  add_126 = mul_404 = None
        
        # File: /app/src/models/feedforward.py:33 in forward, code: x = self.dropout2(x)
        convert_element_type_23: "f32[500, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_3, torch.float32);  gt_3 = None
        mul_406: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
        mul_407: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(add_130, mul_406);  mul_406 = None
        
        # File: /app/src/models/feedforward.py:32 in forward, code: x = self.linear2(x)
        view_199: "f32[64000, 128]" = torch.ops.aten.reshape.default(mul_407, [64000, 128]);  mul_407 = None
        permute_12: "f32[512, 128]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
        permute_248: "f32[128, 512]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        mm_53: "f32[64000, 512]" = torch.ops.aten.mm.default(view_199, permute_248);  permute_248 = None
        permute_249: "f32[128, 64000]" = torch.ops.aten.permute.default(view_199, [1, 0])
        mm_54: "f32[128, 512]" = torch.ops.aten.mm.default(permute_249, view_16);  permute_249 = view_16 = None
        sum_133: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_199, [0], True);  view_199 = None
        view_200: "f32[128]" = torch.ops.aten.reshape.default(sum_133, [128]);  sum_133 = None
        view_201: "f32[500, 128, 512]" = torch.ops.aten.reshape.default(mm_53, [500, 128, 512]);  mm_53 = None
        
        # File: /app/src/models/feedforward.py:31 in forward, code: x = self.layernorm2(x)
        mul_409: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_201, primals_18);  primals_18 = None
        mul_410: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_409, 512)
        sum_134: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_409, [2], True)
        
        # File: /app/src/models/feedforward.py:27 in forward, code: x = self.linear1(x)
        view_15: "f32[500, 128, 512]" = torch.ops.aten.reshape.default(addmm_2, [500, 128, 512]);  addmm_2 = None
        
        # File: /app/src/models/feedforward.py:28 in forward, code: x = self.act(x)
        mul_11: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_15, 0.5)
        mul_12: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_15, 0.7071067811865476)
        erf: "f32[500, 128, 512]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_9: "f32[500, 128, 512]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_13: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_11, add_9);  mul_11 = None
        
        # File: /app/src/models/feedforward.py:29 in forward, code: x = self.dropout1(x)
        mul_14: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(gt_2, mul_13);  mul_13 = None
        mul_15: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_14, 1.1111111111111112);  mul_14 = None
        
        # File: /app/src/models/feedforward.py:31 in forward, code: x = self.layernorm2(x)
        sub_4: "f32[500, 128, 512]" = torch.ops.aten.sub.Tensor(mul_15, getitem_7);  mul_15 = getitem_7 = None
        mul_16: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = None
        mul_411: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_409, mul_16);  mul_409 = None
        sum_135: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_411, [2], True);  mul_411 = None
        mul_412: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_16, sum_135);  sum_135 = None
        sub_108: "f32[500, 128, 512]" = torch.ops.aten.sub.Tensor(mul_410, sum_134);  mul_410 = sum_134 = None
        sub_109: "f32[500, 128, 512]" = torch.ops.aten.sub.Tensor(sub_108, mul_412);  sub_108 = mul_412 = None
        div_31: "f32[500, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 512);  rsqrt_3 = None
        mul_413: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(div_31, sub_109);  div_31 = sub_109 = None
        mul_414: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_201, mul_16);  mul_16 = None
        sum_136: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_414, [0, 1]);  mul_414 = None
        sum_137: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_201, [0, 1]);  view_201 = None
        
        # File: /app/src/models/feedforward.py:29 in forward, code: x = self.dropout1(x)
        convert_element_type_24: "f32[500, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_415: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
        mul_416: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_413, mul_415);  mul_413 = mul_415 = None
        
        # File: /app/src/models/feedforward.py:28 in forward, code: x = self.act(x)
        mul_418: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(add_9, 0.5);  add_9 = None
        mul_419: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_15, view_15)
        mul_420: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_419, -0.5);  mul_419 = None
        exp_13: "f32[500, 128, 512]" = torch.ops.aten.exp.default(mul_420);  mul_420 = None
        mul_421: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
        mul_422: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_15, mul_421);  view_15 = mul_421 = None
        add_132: "f32[500, 128, 512]" = torch.ops.aten.add.Tensor(mul_418, mul_422);  mul_418 = mul_422 = None
        mul_423: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_416, add_132);  mul_416 = add_132 = None
        
        # File: /app/src/models/feedforward.py:27 in forward, code: x = self.linear1(x)
        view_202: "f32[64000, 512]" = torch.ops.aten.reshape.default(mul_423, [64000, 512]);  mul_423 = None
        permute_11: "f32[128, 512]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        permute_252: "f32[512, 128]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        mm_55: "f32[64000, 128]" = torch.ops.aten.mm.default(view_202, permute_252);  permute_252 = None
        permute_253: "f32[512, 64000]" = torch.ops.aten.permute.default(view_202, [1, 0])
        mm_56: "f32[512, 128]" = torch.ops.aten.mm.default(permute_253, view_14);  permute_253 = view_14 = None
        sum_138: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_202, [0], True);  view_202 = None
        view_203: "f32[512]" = torch.ops.aten.reshape.default(sum_138, [512]);  sum_138 = None
        view_204: "f32[500, 128, 128]" = torch.ops.aten.reshape.default(mm_55, [500, 128, 128]);  mm_55 = None
        
        # File: /app/src/models/feedforward.py:26 in forward, code: x = self.layernorm1(x)
        mul_425: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(view_204, primals_14);  primals_14 = None
        mul_426: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_425, 128)
        sum_139: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_425, [2], True)
        mul_427: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_425, mul_9);  mul_425 = None
        sum_140: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_427, [2], True);  mul_427 = None
        mul_428: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_9, sum_140);  sum_140 = None
        sub_111: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(mul_426, sum_139);  mul_426 = sum_139 = None
        sub_112: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(sub_111, mul_428);  sub_111 = mul_428 = None
        mul_429: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(div_32, sub_112);  div_32 = sub_112 = None
        mul_430: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(view_204, mul_9);  mul_9 = None
        sum_141: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_430, [0, 1]);  mul_430 = None
        sum_142: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_204, [0, 1]);  view_204 = None
        add_133: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(add_130, mul_429);  add_130 = mul_429 = None
        
        # File: /app/src/models/particle_transformer.py:46 in forward, code: x = self.dropout(x)
        convert_element_type_25: "f32[500, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_431: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
        mul_432: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(add_133, mul_431);  mul_431 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        view_12: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(addmm_1, [128, 500, 128]);  addmm_1 = None
        permute_10: "f32[500, 128, 128]" = torch.ops.aten.permute.default(view_12, [1, 0, 2]);  view_12 = None
        
        # File: /app/src/models/particle_transformer.py:45 in forward, code: x = self.layernorm2(x)
        sub_113: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(permute_10, getitem_3);  permute_10 = getitem_3 = None
        mul_433: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub_113, rsqrt_1);  sub_113 = None
        mul_434: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_432, primals_12);  primals_12 = None
        mul_435: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_434, 128)
        sum_143: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_434, [2], True)
        mul_436: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_434, mul_433);  mul_434 = None
        sum_144: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_436, [2], True);  mul_436 = None
        mul_437: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_433, sum_144);  sum_144 = None
        sub_114: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(mul_435, sum_143);  mul_435 = sum_143 = None
        sub_115: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(sub_114, mul_437);  sub_114 = mul_437 = None
        div_33: "f32[500, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 128);  rsqrt_1 = None
        mul_438: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(div_33, sub_115);  div_33 = sub_115 = None
        mul_439: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_432, mul_433);  mul_433 = None
        sum_145: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_439, [0, 1]);  mul_439 = None
        sum_146: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_432, [0, 1]);  mul_432 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        permute_256: "f32[128, 500, 128]" = torch.ops.aten.permute.default(mul_438, [1, 0, 2]);  mul_438 = None
        clone_85: "f32[128, 500, 128]" = torch.ops.aten.clone.default(permute_256, memory_format = torch.contiguous_format);  permute_256 = None
        view_205: "f32[64000, 128]" = torch.ops.aten.reshape.default(clone_85, [64000, 128]);  clone_85 = None
        permute_9: "f32[128, 128]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_257: "f32[128, 128]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_57: "f32[64000, 128]" = torch.ops.aten.mm.default(view_205, permute_257);  permute_257 = None
        permute_258: "f32[128, 64000]" = torch.ops.aten.permute.default(view_205, [1, 0])
        mm_58: "f32[128, 128]" = torch.ops.aten.mm.default(permute_258, view_11);  permute_258 = view_11 = None
        sum_147: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_205, [0], True);  view_205 = None
        view_206: "f32[128]" = torch.ops.aten.reshape.default(sum_147, [128]);  sum_147 = None
        view_207: "f32[128, 4000, 16]" = torch.ops.aten.reshape.default(mm_57, [128, 4000, 16]);  mm_57 = None
        permute_261: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(view_207, [1, 0, 2]);  view_207 = None
        bmm_31: "f32[4000, 128, 16]" = torch.ops.aten.bmm.default(permute_262, permute_261);  permute_262 = None
        bmm_32: "f32[4000, 128, 128]" = torch.ops.aten.bmm.default(permute_261, permute_263);  permute_261 = permute_263 = None
        convert_element_type_26: "f32[4000, 128, 128]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_440: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
        mul_441: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_32, mul_440);  bmm_32 = mul_440 = None
        sub_1: "f32[4000, 128, 128]" = torch.ops.aten.sub.Tensor(baddbmm, amax);  baddbmm = amax = None
        exp: "f32[4000, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        div: "f32[4000, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        mul_442: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(mul_441, div);  mul_441 = None
        sum_148: "f32[4000, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_442, [-1], True)
        neg_6: "f32[4000, 128, 128]" = torch.ops.aten.neg.default(div);  div = None
        fma_6: "f32[4000, 128, 128]" = torch.ops.prims.fma.default(neg_6, sum_148, mul_442);  neg_6 = sum_148 = mul_442 = None
        bmm_33: "f32[4000, 128, 16]" = torch.ops.aten.bmm.default(fma_6, permute_264);  permute_264 = None
        bmm_34: "f32[4000, 16, 128]" = torch.ops.aten.bmm.default(permute_265, fma_6);  permute_265 = None
        permute_266: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(bmm_34, [0, 2, 1]);  bmm_34 = None
        mul_443: "f32[4000, 128, 16]" = torch.ops.aten.mul.Tensor(bmm_33, 0.25);  bmm_33 = None
        add_134: "f32[4000, 128, 128]" = torch.ops.aten.add.Tensor(add_127, fma_6);  add_127 = fma_6 = None
        permute_267: "f32[128, 4000, 16]" = torch.ops.aten.permute.default(bmm_31, [1, 0, 2]);  bmm_31 = None
        clone_87: "f32[128, 4000, 16]" = torch.ops.aten.clone.default(permute_267, memory_format = torch.contiguous_format);  permute_267 = None
        view_208: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(clone_87, [128, 500, 128]);  clone_87 = None
        permute_268: "f32[128, 4000, 16]" = torch.ops.aten.permute.default(permute_266, [1, 0, 2]);  permute_266 = None
        view_209: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(permute_268, [128, 500, 128]);  permute_268 = None
        permute_269: "f32[128, 4000, 16]" = torch.ops.aten.permute.default(mul_443, [1, 0, 2]);  mul_443 = None
        clone_88: "f32[128, 4000, 16]" = torch.ops.aten.clone.default(permute_269, memory_format = torch.contiguous_format);  permute_269 = None
        view_210: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(clone_88, [128, 500, 128]);  clone_88 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_18: "f32[3, 128, 500, 128]" = torch.ops.aten.select_scatter.default(full_default, view_208, 0, 2);  view_208 = None
        select_scatter_default_19: "f32[3, 128, 500, 128]" = torch.ops.aten.select_scatter.default(full_default, view_209, 0, 1);  view_209 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        add_135: "f32[3, 128, 500, 128]" = torch.ops.aten.add.Tensor(select_scatter_default_18, select_scatter_default_19);  select_scatter_default_18 = select_scatter_default_19 = None
        
        # No stacktrace found for following nodes
        select_scatter_default_20: "f32[3, 128, 500, 128]" = torch.ops.aten.select_scatter.default(full_default, view_210, 0, 0);  full_default = view_210 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        add_136: "f32[3, 128, 500, 128]" = torch.ops.aten.add.Tensor(add_135, select_scatter_default_20);  add_135 = select_scatter_default_20 = None
        unsqueeze_13: "f32[3, 128, 500, 1, 128]" = torch.ops.aten.unsqueeze.default(add_136, 3);  add_136 = None
        permute_270: "f32[1, 128, 500, 3, 128]" = torch.ops.aten.permute.default(unsqueeze_13, [3, 1, 2, 0, 4]);  unsqueeze_13 = None
        squeeze_13: "f32[128, 500, 3, 128]" = torch.ops.aten.squeeze.dim(permute_270, 0);  permute_270 = None
        clone_89: "f32[128, 500, 3, 128]" = torch.ops.aten.clone.default(squeeze_13, memory_format = torch.contiguous_format);  squeeze_13 = None
        view_211: "f32[128, 500, 384]" = torch.ops.aten.reshape.default(clone_89, [128, 500, 384]);  clone_89 = None
        sum_149: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(view_211, [0, 1], True)
        view_212: "f32[384]" = torch.ops.aten.reshape.default(sum_149, [384]);  sum_149 = None
        view_213: "f32[64000, 384]" = torch.ops.aten.reshape.default(view_211, [64000, 384]);  view_211 = None
        permute_271: "f32[384, 64000]" = torch.ops.aten.permute.default(view_213, [1, 0])
        mm_59: "f32[384, 128]" = torch.ops.aten.mm.default(permute_271, view_3);  permute_271 = view_3 = None
        permute_2: "f32[128, 384]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_273: "f32[384, 128]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_60: "f32[64000, 128]" = torch.ops.aten.mm.default(view_213, permute_273);  view_213 = permute_273 = None
        view_214: "f32[128, 500, 128]" = torch.ops.aten.reshape.default(mm_60, [128, 500, 128]);  mm_60 = None
        permute_275: "f32[500, 128, 128]" = torch.ops.aten.permute.default(view_214, [1, 0, 2]);  view_214 = None
        
        # File: /app/src/models/particle_transformer.py:43 in forward, code: x = self.layernorm1(x)
        mul_445: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(permute_275, primals_4);  primals_4 = None
        mul_446: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_445, 128)
        sum_150: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_445, [2], True)
        
        # File: /app/src/models/lorentz_part.py:59 in torch_dynamo_resume_in_forward_at_55, code: x = self.proj(x)  # (B, N, embed_dim)
        view_2: "f32[500, 128, 128]" = torch.ops.aten.reshape.default(addmm, [500, 128, 128]);  addmm = None
        
        # File: /app/src/models/particle_transformer.py:43 in forward, code: x = self.layernorm1(x)
        sub: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(view_2, getitem_1);  view_2 = getitem_1 = None
        mul: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_447: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_445, mul);  mul_445 = None
        sum_151: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_447, [2], True);  mul_447 = None
        mul_448: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul, sum_151);  sum_151 = None
        sub_117: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(mul_446, sum_150);  mul_446 = sum_150 = None
        sub_118: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(sub_117, mul_448);  sub_117 = mul_448 = None
        div_34: "f32[500, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 128);  rsqrt = None
        mul_449: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(div_34, sub_118);  div_34 = sub_118 = None
        mul_450: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(permute_275, mul);  mul = None
        sum_152: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_450, [0, 1]);  mul_450 = None
        sum_153: "f32[128]" = torch.ops.aten.sum.dim_IntList(permute_275, [0, 1]);  permute_275 = None
        add_137: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(add_133, mul_449);  add_133 = mul_449 = None
        
        # File: /app/src/models/lorentz_part.py:59 in torch_dynamo_resume_in_forward_at_55, code: x = self.proj(x)  # (B, N, embed_dim)
        view_215: "f32[64000, 128]" = torch.ops.aten.reshape.default(add_137, [64000, 128]);  add_137 = None
        permute: "f32[16, 128]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        permute_276: "f32[128, 16]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_61: "f32[64000, 16]" = torch.ops.aten.mm.default(view_215, permute_276);  permute_276 = None
        permute_277: "f32[128, 64000]" = torch.ops.aten.permute.default(view_215, [1, 0])
        mm_62: "f32[128, 16]" = torch.ops.aten.mm.default(permute_277, view_1);  permute_277 = view_1 = None
        sum_154: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_215, [0], True);  view_215 = None
        view_216: "f32[128]" = torch.ops.aten.reshape.default(sum_154, [128]);  sum_154 = None
        view_217: "f32[500, 128, 16]" = torch.ops.aten.reshape.default(mm_61, [500, 128, 16]);  mm_61 = None
        
        # File: /app/src/models/lorentz_part.py:56 in torch_dynamo_resume_in_forward_at_55, code: x = x.view(B, N, 16)
        view_218: "f32[500, 128, 1, 16]" = torch.ops.aten.reshape.default(view_217, [500, 128, 1, 16]);  view_217 = None
        return (view_218, mm_62, view_216, sum_152, sum_153, mm_59, view_212, mm_58, view_206, None, add_134, sum_145, sum_146, sum_141, sum_142, mm_56, view_203, sum_136, sum_137, mm_54, view_200, sum_131, sum_132, mm_51, view_196, mm_50, view_190, sum_124, sum_125, sum_120, sum_121, mm_48, view_187, sum_115, sum_116, mm_46, view_184, sum_110, sum_111, mm_43, view_180, mm_42, view_174, sum_103, sum_104, sum_99, sum_100, mm_40, view_171, sum_94, sum_95, mm_38, view_168, sum_89, sum_90, mm_35, view_164, mm_34, view_158, sum_82, sum_83, sum_78, sum_79, mm_32, view_155, sum_73, sum_74, mm_30, view_152, sum_68, sum_69, mm_27, view_148, mm_26, view_142, sum_61, sum_62, sum_57, sum_58, mm_24, view_139, sum_52, sum_53, mm_22, view_136, sum_47, sum_48, mm_19, view_132, mm_18, view_126, sum_40, sum_41, sum_36, sum_37, mm_16, view_123, sum_31, sum_32, mm_14, view_120, sum_26, sum_27, mm_11, view_116, mm_10, view_110, sum_19, sum_20, sum_15, sum_16, mm_8, view_107, sum_10, sum_11)
        