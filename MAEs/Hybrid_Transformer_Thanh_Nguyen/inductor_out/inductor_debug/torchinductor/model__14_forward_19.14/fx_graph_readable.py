class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[500, 128, 1, 16]", primals_2: "f32[128, 16]", primals_3: "f32[128]", primals_4: "f32[128]", primals_5: "f32[128]", primals_6: "f32[384, 128]", primals_7: "f32[384]", primals_8: "f32[128, 128]", primals_9: "f32[128]", primals_10: "f32[500, 128]", primals_11: "f32[4000, 128, 128]", primals_12: "f32[128]", primals_13: "f32[128]", primals_14: "f32[128]", primals_15: "f32[128]", primals_16: "f32[512, 128]", primals_17: "f32[512]", primals_18: "f32[512]", primals_19: "f32[512]", primals_20: "f32[128, 512]", primals_21: "f32[128]", primals_22: "f32[128]", primals_23: "f32[128]", primals_24: "f32[384, 128]", primals_25: "f32[384]", primals_26: "f32[128, 128]", primals_27: "f32[128]", primals_28: "f32[128]", primals_29: "f32[128]", primals_30: "f32[128]", primals_31: "f32[128]", primals_32: "f32[512, 128]", primals_33: "f32[512]", primals_34: "f32[512]", primals_35: "f32[512]", primals_36: "f32[128, 512]", primals_37: "f32[128]", primals_38: "f32[128]", primals_39: "f32[128]", primals_40: "f32[384, 128]", primals_41: "f32[384]", primals_42: "f32[128, 128]", primals_43: "f32[128]", primals_44: "f32[128]", primals_45: "f32[128]", primals_46: "f32[128]", primals_47: "f32[128]", primals_48: "f32[512, 128]", primals_49: "f32[512]", primals_50: "f32[512]", primals_51: "f32[512]", primals_52: "f32[128, 512]", primals_53: "f32[128]", primals_54: "f32[128]", primals_55: "f32[128]", primals_56: "f32[384, 128]", primals_57: "f32[384]", primals_58: "f32[128, 128]", primals_59: "f32[128]", primals_60: "f32[128]", primals_61: "f32[128]", primals_62: "f32[128]", primals_63: "f32[128]", primals_64: "f32[512, 128]", primals_65: "f32[512]", primals_66: "f32[512]", primals_67: "f32[512]", primals_68: "f32[128, 512]", primals_69: "f32[128]", primals_70: "f32[128]", primals_71: "f32[128]", primals_72: "f32[384, 128]", primals_73: "f32[384]", primals_74: "f32[128, 128]", primals_75: "f32[128]", primals_76: "f32[128]", primals_77: "f32[128]", primals_78: "f32[128]", primals_79: "f32[128]", primals_80: "f32[512, 128]", primals_81: "f32[512]", primals_82: "f32[512]", primals_83: "f32[512]", primals_84: "f32[128, 512]", primals_85: "f32[128]", primals_86: "f32[128]", primals_87: "f32[128]", primals_88: "f32[384, 128]", primals_89: "f32[384]", primals_90: "f32[128, 128]", primals_91: "f32[128]", primals_92: "f32[128]", primals_93: "f32[128]", primals_94: "f32[128]", primals_95: "f32[128]", primals_96: "f32[512, 128]", primals_97: "f32[512]", primals_98: "f32[512]", primals_99: "f32[512]", primals_100: "f32[128, 512]", primals_101: "f32[128]", primals_102: "f32[128]", primals_103: "f32[128]", primals_104: "f32[384, 128]", primals_105: "f32[384]", primals_106: "f32[128, 128]", primals_107: "f32[128]", primals_108: "f32[128]", primals_109: "f32[128]", primals_110: "f32[128]", primals_111: "f32[128]", primals_112: "f32[512, 128]", primals_113: "f32[512]", primals_114: "f32[512]", primals_115: "f32[512]"):
        # File: /app/src/models/lorentz_part.py:56 in torch_dynamo_resume_in_forward_at_55, code: x = x.view(B, N, 16)
        view: "f32[500, 128, 16]" = torch.ops.aten.view.default(primals_1, [500, 128, 16]);  primals_1 = None
        
        # File: /app/src/models/lorentz_part.py:59 in torch_dynamo_resume_in_forward_at_55, code: x = self.proj(x)  # (B, N, embed_dim)
        view_1: "f32[64000, 16]" = torch.ops.aten.view.default(view, [64000, 16]);  view = None
        permute: "f32[16, 128]" = torch.ops.aten.permute.default(primals_2, [1, 0])
        addmm: "f32[64000, 128]" = torch.ops.aten.addmm.default(primals_3, view_1, permute);  primals_3 = permute = None
        view_2: "f32[500, 128, 128]" = torch.ops.aten.view.default(addmm, [500, 128, 128])
        
        # File: /app/src/models/particle_transformer.py:43 in forward, code: x = self.layernorm1(x)
        var_mean = torch.ops.aten.var_mean.correction(view_2, [2], correction = 0, keepdim = True)
        getitem: "f32[500, 128, 1]" = var_mean[0]
        getitem_1: "f32[500, 128, 1]" = var_mean[1];  var_mean = None
        add: "f32[500, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[500, 128, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(view_2, getitem_1)
        mul: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul, primals_4);  mul = None
        add_1: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_1, primals_5);  mul_1 = primals_5 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        permute_1: "f32[128, 500, 128]" = torch.ops.aten.permute.default(add_1, [1, 0, 2]);  add_1 = None
        permute_2: "f32[128, 384]" = torch.ops.aten.permute.default(primals_6, [1, 0])
        clone: "f32[128, 500, 128]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_3: "f32[64000, 128]" = torch.ops.aten.view.default(clone, [64000, 128]);  clone = None
        mm: "f32[64000, 384]" = torch.ops.aten.mm.default(view_3, permute_2);  permute_2 = None
        view_4: "f32[128, 500, 384]" = torch.ops.aten.view.default(mm, [128, 500, 384]);  mm = None
        add_2: "f32[128, 500, 384]" = torch.ops.aten.add.Tensor(view_4, primals_7);  view_4 = primals_7 = None
        view_5: "f32[128, 500, 3, 128]" = torch.ops.aten.view.default(add_2, [128, 500, 3, 128]);  add_2 = None
        unsqueeze: "f32[1, 128, 500, 3, 128]" = torch.ops.aten.unsqueeze.default(view_5, 0);  view_5 = None
        permute_3: "f32[3, 128, 500, 1, 128]" = torch.ops.aten.permute.default(unsqueeze, [3, 1, 2, 0, 4]);  unsqueeze = None
        squeeze: "f32[3, 128, 500, 128]" = torch.ops.aten.squeeze.dim(permute_3, -2);  permute_3 = None
        clone_1: "f32[3, 128, 500, 128]" = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
        select: "f32[128, 500, 128]" = torch.ops.aten.select.int(clone_1, 0, 0)
        select_1: "f32[128, 500, 128]" = torch.ops.aten.select.int(clone_1, 0, 1)
        select_2: "f32[128, 500, 128]" = torch.ops.aten.select.int(clone_1, 0, 2);  clone_1 = None
        view_6: "f32[128, 4000, 16]" = torch.ops.aten.view.default(select, [128, 4000, 16]);  select = None
        permute_4: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(view_6, [1, 0, 2]);  view_6 = None
        view_7: "f32[128, 4000, 16]" = torch.ops.aten.view.default(select_1, [128, 4000, 16]);  select_1 = None
        permute_5: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(view_7, [1, 0, 2]);  view_7 = None
        view_8: "f32[128, 4000, 16]" = torch.ops.aten.view.default(select_2, [128, 4000, 16]);  select_2 = None
        permute_6: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(view_8, [1, 0, 2]);  view_8 = None
        view_9: "f32[500, 1, 1, 128]" = torch.ops.aten.view.default(primals_10, [500, 1, 1, 128]);  primals_10 = None
        expand: "f32[500, 8, 1, 128]" = torch.ops.aten.expand.default(view_9, [-1, 8, -1, -1]);  view_9 = None
        clone_2: "f32[500, 8, 1, 128]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_10: "f32[4000, 1, 128]" = torch.ops.aten.view.default(clone_2, [4000, 1, 128]);  clone_2 = None
        add_3: "f32[4000, 128, 128]" = torch.ops.aten.add.Tensor(primals_11, view_10);  primals_11 = view_10 = None
        mul_2: "f32[4000, 128, 16]" = torch.ops.aten.mul.Tensor(permute_4, 0.25);  permute_4 = None
        permute_7: "f32[4000, 16, 128]" = torch.ops.aten.permute.default(permute_5, [0, 2, 1]);  permute_5 = None
        baddbmm: "f32[4000, 128, 128]" = torch.ops.aten.baddbmm.default(add_3, mul_2, permute_7)
        amax: "f32[4000, 128, 1]" = torch.ops.aten.amax.default(baddbmm, [-1], True)
        sub_1: "f32[4000, 128, 128]" = torch.ops.aten.sub.Tensor(baddbmm, amax)
        exp: "f32[4000, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[4000, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[4000, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        
        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[27]" = torch.ops.prims.inductor_seeds.default(27, device(type='cuda', index=0))
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_26: "f32[4000, 128, 128]" = torch.ops.prims.inductor_random.default([4000, 128, 128], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[4000, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_26, 0.1);  inductor_random_default_26 = None
        mul_3: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(gt, div);  div = None
        mul_4: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(mul_3, 1.1111111111111112);  mul_3 = None
        bmm: "f32[4000, 128, 16]" = torch.ops.aten.bmm.default(mul_4, permute_6)
        permute_8: "f32[128, 4000, 16]" = torch.ops.aten.permute.default(bmm, [1, 0, 2]);  bmm = None
        clone_3: "f32[128, 4000, 16]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        view_11: "f32[64000, 128]" = torch.ops.aten.view.default(clone_3, [64000, 128]);  clone_3 = None
        permute_9: "f32[128, 128]" = torch.ops.aten.permute.default(primals_8, [1, 0])
        addmm_1: "f32[64000, 128]" = torch.ops.aten.addmm.default(primals_9, view_11, permute_9);  primals_9 = permute_9 = None
        view_12: "f32[128, 500, 128]" = torch.ops.aten.view.default(addmm_1, [128, 500, 128])
        permute_10: "f32[500, 128, 128]" = torch.ops.aten.permute.default(view_12, [1, 0, 2]);  view_12 = None
        
        # File: /app/src/models/particle_transformer.py:45 in forward, code: x = self.layernorm2(x)
        clone_4: "f32[500, 128, 128]" = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(clone_4, [2], correction = 0, keepdim = True)
        getitem_2: "f32[500, 128, 1]" = var_mean_1[0]
        getitem_3: "f32[500, 128, 1]" = var_mean_1[1];  var_mean_1 = None
        add_4: "f32[500, 128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[500, 128, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_2: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(clone_4, getitem_3);  clone_4 = None
        mul_5: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_6: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_5, primals_12);  mul_5 = None
        add_5: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_6, primals_13);  mul_6 = primals_13 = None
        
        # File: /app/src/models/particle_transformer.py:46 in forward, code: x = self.dropout(x)
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_25: "f32[500, 128, 128]" = torch.ops.prims.inductor_random.default([500, 128, 128], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_1: "b8[500, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_25, 0.1);  inductor_random_default_25 = None
        mul_7: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(gt_1, add_5);  add_5 = None
        mul_8: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_7, 1.1111111111111112);  mul_7 = None
        
        # File: /app/src/models/particle_transformer.py:48 in forward, code: x += residual
        add_6: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_8, view_2);  mul_8 = view_2 = None
        
        # File: /app/src/models/feedforward.py:26 in forward, code: x = self.layernorm1(x)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
        getitem_4: "f32[500, 128, 1]" = var_mean_2[0]
        getitem_5: "f32[500, 128, 1]" = var_mean_2[1];  var_mean_2 = None
        add_7: "f32[500, 128, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2: "f32[500, 128, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        sub_3: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(add_6, getitem_5);  getitem_5 = None
        mul_9: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_10: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_9, primals_14)
        add_8: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_10, primals_15);  mul_10 = primals_15 = None
        
        # File: /app/src/models/feedforward.py:27 in forward, code: x = self.linear1(x)
        view_14: "f32[64000, 128]" = torch.ops.aten.view.default(add_8, [64000, 128]);  add_8 = None
        permute_11: "f32[128, 512]" = torch.ops.aten.permute.default(primals_16, [1, 0])
        addmm_2: "f32[64000, 512]" = torch.ops.aten.addmm.default(primals_17, view_14, permute_11);  primals_17 = permute_11 = None
        view_15: "f32[500, 128, 512]" = torch.ops.aten.view.default(addmm_2, [500, 128, 512])
        
        # File: /app/src/models/feedforward.py:28 in forward, code: x = self.act(x)
        mul_11: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_15, 0.5)
        mul_12: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_15, 0.7071067811865476);  view_15 = None
        erf: "f32[500, 128, 512]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_9: "f32[500, 128, 512]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_13: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_11, add_9);  mul_11 = add_9 = None
        
        # File: /app/src/models/feedforward.py:29 in forward, code: x = self.dropout1(x)
        inductor_lookup_seed_default_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_24: "f32[500, 128, 512]" = torch.ops.prims.inductor_random.default([500, 128, 512], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        gt_2: "b8[500, 128, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_24, 0.1);  inductor_random_default_24 = None
        mul_14: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(gt_2, mul_13);  mul_13 = None
        mul_15: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_14, 1.1111111111111112);  mul_14 = None
        
        # File: /app/src/models/feedforward.py:31 in forward, code: x = self.layernorm2(x)
        var_mean_3 = torch.ops.aten.var_mean.correction(mul_15, [2], correction = 0, keepdim = True)
        getitem_6: "f32[500, 128, 1]" = var_mean_3[0]
        getitem_7: "f32[500, 128, 1]" = var_mean_3[1];  var_mean_3 = None
        add_10: "f32[500, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3: "f32[500, 128, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_4: "f32[500, 128, 512]" = torch.ops.aten.sub.Tensor(mul_15, getitem_7);  mul_15 = None
        mul_16: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = None
        mul_17: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_16, primals_18);  mul_16 = None
        add_11: "f32[500, 128, 512]" = torch.ops.aten.add.Tensor(mul_17, primals_19);  mul_17 = primals_19 = None
        
        # File: /app/src/models/feedforward.py:32 in forward, code: x = self.linear2(x)
        view_16: "f32[64000, 512]" = torch.ops.aten.view.default(add_11, [64000, 512]);  add_11 = None
        permute_12: "f32[512, 128]" = torch.ops.aten.permute.default(primals_20, [1, 0])
        addmm_3: "f32[64000, 128]" = torch.ops.aten.addmm.default(primals_21, view_16, permute_12);  primals_21 = permute_12 = None
        view_17: "f32[500, 128, 128]" = torch.ops.aten.view.default(addmm_3, [500, 128, 128]);  addmm_3 = None
        
        # File: /app/src/models/feedforward.py:33 in forward, code: x = self.dropout2(x)
        inductor_lookup_seed_default_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_23: "f32[500, 128, 128]" = torch.ops.prims.inductor_random.default([500, 128, 128], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        gt_3: "b8[500, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_23, 0.1);  inductor_random_default_23 = None
        mul_18: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(gt_3, view_17);  view_17 = None
        mul_19: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_18, 1.1111111111111112);  mul_18 = None
        
        # File: /app/src/models/feedforward.py:34 in forward, code: x += residual
        add_12: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_19, add_6);  mul_19 = add_6 = None
        
        # File: /app/src/models/particle_transformer.py:43 in forward, code: x = self.layernorm1(x)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
        getitem_8: "f32[500, 128, 1]" = var_mean_4[0]
        getitem_9: "f32[500, 128, 1]" = var_mean_4[1];  var_mean_4 = None
        add_13: "f32[500, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_4: "f32[500, 128, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_5: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(add_12, getitem_9);  getitem_9 = None
        mul_20: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_4);  sub_5 = None
        mul_21: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_20, primals_22)
        add_14: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_21, primals_23);  mul_21 = primals_23 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        permute_13: "f32[128, 500, 128]" = torch.ops.aten.permute.default(add_14, [1, 0, 2]);  add_14 = None
        permute_14: "f32[128, 384]" = torch.ops.aten.permute.default(primals_24, [1, 0])
        clone_5: "f32[128, 500, 128]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
        view_18: "f32[64000, 128]" = torch.ops.aten.view.default(clone_5, [64000, 128]);  clone_5 = None
        mm_1: "f32[64000, 384]" = torch.ops.aten.mm.default(view_18, permute_14);  permute_14 = None
        view_19: "f32[128, 500, 384]" = torch.ops.aten.view.default(mm_1, [128, 500, 384]);  mm_1 = None
        add_15: "f32[128, 500, 384]" = torch.ops.aten.add.Tensor(view_19, primals_25);  view_19 = primals_25 = None
        view_20: "f32[128, 500, 3, 128]" = torch.ops.aten.view.default(add_15, [128, 500, 3, 128]);  add_15 = None
        unsqueeze_1: "f32[1, 128, 500, 3, 128]" = torch.ops.aten.unsqueeze.default(view_20, 0);  view_20 = None
        permute_15: "f32[3, 128, 500, 1, 128]" = torch.ops.aten.permute.default(unsqueeze_1, [3, 1, 2, 0, 4]);  unsqueeze_1 = None
        squeeze_1: "f32[3, 128, 500, 128]" = torch.ops.aten.squeeze.dim(permute_15, -2);  permute_15 = None
        clone_6: "f32[3, 128, 500, 128]" = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
        select_3: "f32[128, 500, 128]" = torch.ops.aten.select.int(clone_6, 0, 0)
        select_4: "f32[128, 500, 128]" = torch.ops.aten.select.int(clone_6, 0, 1)
        select_5: "f32[128, 500, 128]" = torch.ops.aten.select.int(clone_6, 0, 2);  clone_6 = None
        view_21: "f32[128, 4000, 16]" = torch.ops.aten.view.default(select_3, [128, 4000, 16]);  select_3 = None
        permute_16: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(view_21, [1, 0, 2]);  view_21 = None
        view_22: "f32[128, 4000, 16]" = torch.ops.aten.view.default(select_4, [128, 4000, 16]);  select_4 = None
        permute_17: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(view_22, [1, 0, 2]);  view_22 = None
        view_23: "f32[128, 4000, 16]" = torch.ops.aten.view.default(select_5, [128, 4000, 16]);  select_5 = None
        permute_18: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(view_23, [1, 0, 2]);  view_23 = None
        mul_22: "f32[4000, 128, 16]" = torch.ops.aten.mul.Tensor(permute_16, 0.25);  permute_16 = None
        permute_19: "f32[4000, 16, 128]" = torch.ops.aten.permute.default(permute_17, [0, 2, 1]);  permute_17 = None
        baddbmm_1: "f32[4000, 128, 128]" = torch.ops.aten.baddbmm.default(add_3, mul_22, permute_19)
        amax_1: "f32[4000, 128, 1]" = torch.ops.aten.amax.default(baddbmm_1, [-1], True)
        sub_6: "f32[4000, 128, 128]" = torch.ops.aten.sub.Tensor(baddbmm_1, amax_1)
        exp_1: "f32[4000, 128, 128]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_2: "f32[4000, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[4000, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = None
        inductor_lookup_seed_default_4: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_22: "f32[4000, 128, 128]" = torch.ops.prims.inductor_random.default([4000, 128, 128], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        gt_4: "b8[4000, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_22, 0.1);  inductor_random_default_22 = None
        mul_23: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(gt_4, div_1);  div_1 = None
        mul_24: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(mul_23, 1.1111111111111112);  mul_23 = None
        bmm_1: "f32[4000, 128, 16]" = torch.ops.aten.bmm.default(mul_24, permute_18)
        permute_20: "f32[128, 4000, 16]" = torch.ops.aten.permute.default(bmm_1, [1, 0, 2]);  bmm_1 = None
        clone_8: "f32[128, 4000, 16]" = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None
        view_26: "f32[64000, 128]" = torch.ops.aten.view.default(clone_8, [64000, 128]);  clone_8 = None
        permute_21: "f32[128, 128]" = torch.ops.aten.permute.default(primals_26, [1, 0])
        addmm_4: "f32[64000, 128]" = torch.ops.aten.addmm.default(primals_27, view_26, permute_21);  primals_27 = permute_21 = None
        view_27: "f32[128, 500, 128]" = torch.ops.aten.view.default(addmm_4, [128, 500, 128])
        permute_22: "f32[500, 128, 128]" = torch.ops.aten.permute.default(view_27, [1, 0, 2]);  view_27 = None
        
        # File: /app/src/models/particle_transformer.py:45 in forward, code: x = self.layernorm2(x)
        clone_9: "f32[500, 128, 128]" = torch.ops.aten.clone.default(permute_22, memory_format = torch.contiguous_format);  permute_22 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(clone_9, [2], correction = 0, keepdim = True)
        getitem_10: "f32[500, 128, 1]" = var_mean_5[0]
        getitem_11: "f32[500, 128, 1]" = var_mean_5[1];  var_mean_5 = None
        add_17: "f32[500, 128, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_5: "f32[500, 128, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        sub_7: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(clone_9, getitem_11);  clone_9 = None
        mul_25: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_5);  sub_7 = None
        mul_26: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_25, primals_28);  mul_25 = None
        add_18: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_26, primals_29);  mul_26 = primals_29 = None
        
        # File: /app/src/models/particle_transformer.py:46 in forward, code: x = self.dropout(x)
        inductor_lookup_seed_default_5: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_21: "f32[500, 128, 128]" = torch.ops.prims.inductor_random.default([500, 128, 128], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        gt_5: "b8[500, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_21, 0.1);  inductor_random_default_21 = None
        mul_27: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(gt_5, add_18);  add_18 = None
        mul_28: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_27, 1.1111111111111112);  mul_27 = None
        
        # File: /app/src/models/particle_transformer.py:48 in forward, code: x += residual
        add_19: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_28, add_12);  mul_28 = add_12 = None
        
        # File: /app/src/models/feedforward.py:26 in forward, code: x = self.layernorm1(x)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_12: "f32[500, 128, 1]" = var_mean_6[0]
        getitem_13: "f32[500, 128, 1]" = var_mean_6[1];  var_mean_6 = None
        add_20: "f32[500, 128, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_6: "f32[500, 128, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        sub_8: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(add_19, getitem_13);  getitem_13 = None
        mul_29: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_6);  sub_8 = None
        mul_30: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_29, primals_30)
        add_21: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_30, primals_31);  mul_30 = primals_31 = None
        
        # File: /app/src/models/feedforward.py:27 in forward, code: x = self.linear1(x)
        view_29: "f32[64000, 128]" = torch.ops.aten.view.default(add_21, [64000, 128]);  add_21 = None
        permute_23: "f32[128, 512]" = torch.ops.aten.permute.default(primals_32, [1, 0])
        addmm_5: "f32[64000, 512]" = torch.ops.aten.addmm.default(primals_33, view_29, permute_23);  primals_33 = permute_23 = None
        view_30: "f32[500, 128, 512]" = torch.ops.aten.view.default(addmm_5, [500, 128, 512])
        
        # File: /app/src/models/feedforward.py:28 in forward, code: x = self.act(x)
        mul_31: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_30, 0.5)
        mul_32: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_30, 0.7071067811865476);  view_30 = None
        erf_1: "f32[500, 128, 512]" = torch.ops.aten.erf.default(mul_32);  mul_32 = None
        add_22: "f32[500, 128, 512]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_33: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_31, add_22);  mul_31 = add_22 = None
        
        # File: /app/src/models/feedforward.py:29 in forward, code: x = self.dropout1(x)
        inductor_lookup_seed_default_6: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_20: "f32[500, 128, 512]" = torch.ops.prims.inductor_random.default([500, 128, 512], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        gt_6: "b8[500, 128, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_20, 0.1);  inductor_random_default_20 = None
        mul_34: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(gt_6, mul_33);  mul_33 = None
        mul_35: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_34, 1.1111111111111112);  mul_34 = None
        
        # File: /app/src/models/feedforward.py:31 in forward, code: x = self.layernorm2(x)
        var_mean_7 = torch.ops.aten.var_mean.correction(mul_35, [2], correction = 0, keepdim = True)
        getitem_14: "f32[500, 128, 1]" = var_mean_7[0]
        getitem_15: "f32[500, 128, 1]" = var_mean_7[1];  var_mean_7 = None
        add_23: "f32[500, 128, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_7: "f32[500, 128, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        sub_9: "f32[500, 128, 512]" = torch.ops.aten.sub.Tensor(mul_35, getitem_15);  mul_35 = None
        mul_36: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_7);  sub_9 = None
        mul_37: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_36, primals_34);  mul_36 = None
        add_24: "f32[500, 128, 512]" = torch.ops.aten.add.Tensor(mul_37, primals_35);  mul_37 = primals_35 = None
        
        # File: /app/src/models/feedforward.py:32 in forward, code: x = self.linear2(x)
        view_31: "f32[64000, 512]" = torch.ops.aten.view.default(add_24, [64000, 512]);  add_24 = None
        permute_24: "f32[512, 128]" = torch.ops.aten.permute.default(primals_36, [1, 0])
        addmm_6: "f32[64000, 128]" = torch.ops.aten.addmm.default(primals_37, view_31, permute_24);  primals_37 = permute_24 = None
        view_32: "f32[500, 128, 128]" = torch.ops.aten.view.default(addmm_6, [500, 128, 128]);  addmm_6 = None
        
        # File: /app/src/models/feedforward.py:33 in forward, code: x = self.dropout2(x)
        inductor_lookup_seed_default_7: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_19: "f32[500, 128, 128]" = torch.ops.prims.inductor_random.default([500, 128, 128], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        gt_7: "b8[500, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_19, 0.1);  inductor_random_default_19 = None
        mul_38: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(gt_7, view_32);  view_32 = None
        mul_39: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_38, 1.1111111111111112);  mul_38 = None
        
        # File: /app/src/models/feedforward.py:34 in forward, code: x += residual
        add_25: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_39, add_19);  mul_39 = add_19 = None
        
        # File: /app/src/models/particle_transformer.py:43 in forward, code: x = self.layernorm1(x)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_16: "f32[500, 128, 1]" = var_mean_8[0]
        getitem_17: "f32[500, 128, 1]" = var_mean_8[1];  var_mean_8 = None
        add_26: "f32[500, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_8: "f32[500, 128, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_10: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(add_25, getitem_17);  getitem_17 = None
        mul_40: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_8);  sub_10 = None
        mul_41: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_40, primals_38)
        add_27: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_41, primals_39);  mul_41 = primals_39 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        permute_25: "f32[128, 500, 128]" = torch.ops.aten.permute.default(add_27, [1, 0, 2]);  add_27 = None
        permute_26: "f32[128, 384]" = torch.ops.aten.permute.default(primals_40, [1, 0])
        clone_10: "f32[128, 500, 128]" = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None
        view_33: "f32[64000, 128]" = torch.ops.aten.view.default(clone_10, [64000, 128]);  clone_10 = None
        mm_2: "f32[64000, 384]" = torch.ops.aten.mm.default(view_33, permute_26);  permute_26 = None
        view_34: "f32[128, 500, 384]" = torch.ops.aten.view.default(mm_2, [128, 500, 384]);  mm_2 = None
        add_28: "f32[128, 500, 384]" = torch.ops.aten.add.Tensor(view_34, primals_41);  view_34 = primals_41 = None
        view_35: "f32[128, 500, 3, 128]" = torch.ops.aten.view.default(add_28, [128, 500, 3, 128]);  add_28 = None
        unsqueeze_2: "f32[1, 128, 500, 3, 128]" = torch.ops.aten.unsqueeze.default(view_35, 0);  view_35 = None
        permute_27: "f32[3, 128, 500, 1, 128]" = torch.ops.aten.permute.default(unsqueeze_2, [3, 1, 2, 0, 4]);  unsqueeze_2 = None
        squeeze_2: "f32[3, 128, 500, 128]" = torch.ops.aten.squeeze.dim(permute_27, -2);  permute_27 = None
        clone_11: "f32[3, 128, 500, 128]" = torch.ops.aten.clone.default(squeeze_2, memory_format = torch.contiguous_format);  squeeze_2 = None
        select_6: "f32[128, 500, 128]" = torch.ops.aten.select.int(clone_11, 0, 0)
        select_7: "f32[128, 500, 128]" = torch.ops.aten.select.int(clone_11, 0, 1)
        select_8: "f32[128, 500, 128]" = torch.ops.aten.select.int(clone_11, 0, 2);  clone_11 = None
        view_36: "f32[128, 4000, 16]" = torch.ops.aten.view.default(select_6, [128, 4000, 16]);  select_6 = None
        permute_28: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(view_36, [1, 0, 2]);  view_36 = None
        view_37: "f32[128, 4000, 16]" = torch.ops.aten.view.default(select_7, [128, 4000, 16]);  select_7 = None
        permute_29: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(view_37, [1, 0, 2]);  view_37 = None
        view_38: "f32[128, 4000, 16]" = torch.ops.aten.view.default(select_8, [128, 4000, 16]);  select_8 = None
        permute_30: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(view_38, [1, 0, 2]);  view_38 = None
        mul_42: "f32[4000, 128, 16]" = torch.ops.aten.mul.Tensor(permute_28, 0.25);  permute_28 = None
        permute_31: "f32[4000, 16, 128]" = torch.ops.aten.permute.default(permute_29, [0, 2, 1]);  permute_29 = None
        baddbmm_2: "f32[4000, 128, 128]" = torch.ops.aten.baddbmm.default(add_3, mul_42, permute_31)
        amax_2: "f32[4000, 128, 1]" = torch.ops.aten.amax.default(baddbmm_2, [-1], True)
        sub_11: "f32[4000, 128, 128]" = torch.ops.aten.sub.Tensor(baddbmm_2, amax_2)
        exp_2: "f32[4000, 128, 128]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        sum_3: "f32[4000, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[4000, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = None
        inductor_lookup_seed_default_8: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_18: "f32[4000, 128, 128]" = torch.ops.prims.inductor_random.default([4000, 128, 128], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        gt_8: "b8[4000, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_18, 0.1);  inductor_random_default_18 = None
        mul_43: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(gt_8, div_2);  div_2 = None
        mul_44: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(mul_43, 1.1111111111111112);  mul_43 = None
        bmm_2: "f32[4000, 128, 16]" = torch.ops.aten.bmm.default(mul_44, permute_30)
        permute_32: "f32[128, 4000, 16]" = torch.ops.aten.permute.default(bmm_2, [1, 0, 2]);  bmm_2 = None
        clone_13: "f32[128, 4000, 16]" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
        view_41: "f32[64000, 128]" = torch.ops.aten.view.default(clone_13, [64000, 128]);  clone_13 = None
        permute_33: "f32[128, 128]" = torch.ops.aten.permute.default(primals_42, [1, 0])
        addmm_7: "f32[64000, 128]" = torch.ops.aten.addmm.default(primals_43, view_41, permute_33);  primals_43 = permute_33 = None
        view_42: "f32[128, 500, 128]" = torch.ops.aten.view.default(addmm_7, [128, 500, 128])
        permute_34: "f32[500, 128, 128]" = torch.ops.aten.permute.default(view_42, [1, 0, 2]);  view_42 = None
        
        # File: /app/src/models/particle_transformer.py:45 in forward, code: x = self.layernorm2(x)
        clone_14: "f32[500, 128, 128]" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(clone_14, [2], correction = 0, keepdim = True)
        getitem_18: "f32[500, 128, 1]" = var_mean_9[0]
        getitem_19: "f32[500, 128, 1]" = var_mean_9[1];  var_mean_9 = None
        add_30: "f32[500, 128, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_9: "f32[500, 128, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        sub_12: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(clone_14, getitem_19);  clone_14 = None
        mul_45: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_9);  sub_12 = None
        mul_46: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_45, primals_44);  mul_45 = None
        add_31: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_46, primals_45);  mul_46 = primals_45 = None
        
        # File: /app/src/models/particle_transformer.py:46 in forward, code: x = self.dropout(x)
        inductor_lookup_seed_default_9: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_17: "f32[500, 128, 128]" = torch.ops.prims.inductor_random.default([500, 128, 128], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        gt_9: "b8[500, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_17, 0.1);  inductor_random_default_17 = None
        mul_47: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(gt_9, add_31);  add_31 = None
        mul_48: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_47, 1.1111111111111112);  mul_47 = None
        
        # File: /app/src/models/particle_transformer.py:48 in forward, code: x += residual
        add_32: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_48, add_25);  mul_48 = add_25 = None
        
        # File: /app/src/models/feedforward.py:26 in forward, code: x = self.layernorm1(x)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
        getitem_20: "f32[500, 128, 1]" = var_mean_10[0]
        getitem_21: "f32[500, 128, 1]" = var_mean_10[1];  var_mean_10 = None
        add_33: "f32[500, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_10: "f32[500, 128, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        sub_13: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(add_32, getitem_21);  getitem_21 = None
        mul_49: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_10);  sub_13 = None
        mul_50: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_49, primals_46)
        add_34: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_50, primals_47);  mul_50 = primals_47 = None
        
        # File: /app/src/models/feedforward.py:27 in forward, code: x = self.linear1(x)
        view_44: "f32[64000, 128]" = torch.ops.aten.view.default(add_34, [64000, 128]);  add_34 = None
        permute_35: "f32[128, 512]" = torch.ops.aten.permute.default(primals_48, [1, 0])
        addmm_8: "f32[64000, 512]" = torch.ops.aten.addmm.default(primals_49, view_44, permute_35);  primals_49 = permute_35 = None
        view_45: "f32[500, 128, 512]" = torch.ops.aten.view.default(addmm_8, [500, 128, 512])
        
        # File: /app/src/models/feedforward.py:28 in forward, code: x = self.act(x)
        mul_51: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_45, 0.5)
        mul_52: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_45, 0.7071067811865476);  view_45 = None
        erf_2: "f32[500, 128, 512]" = torch.ops.aten.erf.default(mul_52);  mul_52 = None
        add_35: "f32[500, 128, 512]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_53: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_51, add_35);  mul_51 = add_35 = None
        
        # File: /app/src/models/feedforward.py:29 in forward, code: x = self.dropout1(x)
        inductor_lookup_seed_default_10: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_16: "f32[500, 128, 512]" = torch.ops.prims.inductor_random.default([500, 128, 512], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        gt_10: "b8[500, 128, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_16, 0.1);  inductor_random_default_16 = None
        mul_54: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(gt_10, mul_53);  mul_53 = None
        mul_55: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_54, 1.1111111111111112);  mul_54 = None
        
        # File: /app/src/models/feedforward.py:31 in forward, code: x = self.layernorm2(x)
        var_mean_11 = torch.ops.aten.var_mean.correction(mul_55, [2], correction = 0, keepdim = True)
        getitem_22: "f32[500, 128, 1]" = var_mean_11[0]
        getitem_23: "f32[500, 128, 1]" = var_mean_11[1];  var_mean_11 = None
        add_36: "f32[500, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_11: "f32[500, 128, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_14: "f32[500, 128, 512]" = torch.ops.aten.sub.Tensor(mul_55, getitem_23);  mul_55 = None
        mul_56: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_11);  sub_14 = None
        mul_57: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_56, primals_50);  mul_56 = None
        add_37: "f32[500, 128, 512]" = torch.ops.aten.add.Tensor(mul_57, primals_51);  mul_57 = primals_51 = None
        
        # File: /app/src/models/feedforward.py:32 in forward, code: x = self.linear2(x)
        view_46: "f32[64000, 512]" = torch.ops.aten.view.default(add_37, [64000, 512]);  add_37 = None
        permute_36: "f32[512, 128]" = torch.ops.aten.permute.default(primals_52, [1, 0])
        addmm_9: "f32[64000, 128]" = torch.ops.aten.addmm.default(primals_53, view_46, permute_36);  primals_53 = permute_36 = None
        view_47: "f32[500, 128, 128]" = torch.ops.aten.view.default(addmm_9, [500, 128, 128]);  addmm_9 = None
        
        # File: /app/src/models/feedforward.py:33 in forward, code: x = self.dropout2(x)
        inductor_lookup_seed_default_11: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default_15: "f32[500, 128, 128]" = torch.ops.prims.inductor_random.default([500, 128, 128], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        gt_11: "b8[500, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_15, 0.1);  inductor_random_default_15 = None
        mul_58: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(gt_11, view_47);  view_47 = None
        mul_59: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_58, 1.1111111111111112);  mul_58 = None
        
        # File: /app/src/models/feedforward.py:34 in forward, code: x += residual
        add_38: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_59, add_32);  mul_59 = add_32 = None
        
        # File: /app/src/models/particle_transformer.py:43 in forward, code: x = self.layernorm1(x)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
        getitem_24: "f32[500, 128, 1]" = var_mean_12[0]
        getitem_25: "f32[500, 128, 1]" = var_mean_12[1];  var_mean_12 = None
        add_39: "f32[500, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_12: "f32[500, 128, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        sub_15: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(add_38, getitem_25);  getitem_25 = None
        mul_60: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_12);  sub_15 = None
        mul_61: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_60, primals_54)
        add_40: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_61, primals_55);  mul_61 = primals_55 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        permute_37: "f32[128, 500, 128]" = torch.ops.aten.permute.default(add_40, [1, 0, 2]);  add_40 = None
        permute_38: "f32[128, 384]" = torch.ops.aten.permute.default(primals_56, [1, 0])
        clone_15: "f32[128, 500, 128]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
        view_48: "f32[64000, 128]" = torch.ops.aten.view.default(clone_15, [64000, 128]);  clone_15 = None
        mm_3: "f32[64000, 384]" = torch.ops.aten.mm.default(view_48, permute_38);  permute_38 = None
        view_49: "f32[128, 500, 384]" = torch.ops.aten.view.default(mm_3, [128, 500, 384]);  mm_3 = None
        add_41: "f32[128, 500, 384]" = torch.ops.aten.add.Tensor(view_49, primals_57);  view_49 = primals_57 = None
        view_50: "f32[128, 500, 3, 128]" = torch.ops.aten.view.default(add_41, [128, 500, 3, 128]);  add_41 = None
        unsqueeze_3: "f32[1, 128, 500, 3, 128]" = torch.ops.aten.unsqueeze.default(view_50, 0);  view_50 = None
        permute_39: "f32[3, 128, 500, 1, 128]" = torch.ops.aten.permute.default(unsqueeze_3, [3, 1, 2, 0, 4]);  unsqueeze_3 = None
        squeeze_3: "f32[3, 128, 500, 128]" = torch.ops.aten.squeeze.dim(permute_39, -2);  permute_39 = None
        clone_16: "f32[3, 128, 500, 128]" = torch.ops.aten.clone.default(squeeze_3, memory_format = torch.contiguous_format);  squeeze_3 = None
        select_9: "f32[128, 500, 128]" = torch.ops.aten.select.int(clone_16, 0, 0)
        select_10: "f32[128, 500, 128]" = torch.ops.aten.select.int(clone_16, 0, 1)
        select_11: "f32[128, 500, 128]" = torch.ops.aten.select.int(clone_16, 0, 2);  clone_16 = None
        view_51: "f32[128, 4000, 16]" = torch.ops.aten.view.default(select_9, [128, 4000, 16]);  select_9 = None
        permute_40: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(view_51, [1, 0, 2]);  view_51 = None
        view_52: "f32[128, 4000, 16]" = torch.ops.aten.view.default(select_10, [128, 4000, 16]);  select_10 = None
        permute_41: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(view_52, [1, 0, 2]);  view_52 = None
        view_53: "f32[128, 4000, 16]" = torch.ops.aten.view.default(select_11, [128, 4000, 16]);  select_11 = None
        permute_42: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(view_53, [1, 0, 2]);  view_53 = None
        mul_62: "f32[4000, 128, 16]" = torch.ops.aten.mul.Tensor(permute_40, 0.25);  permute_40 = None
        permute_43: "f32[4000, 16, 128]" = torch.ops.aten.permute.default(permute_41, [0, 2, 1]);  permute_41 = None
        baddbmm_3: "f32[4000, 128, 128]" = torch.ops.aten.baddbmm.default(add_3, mul_62, permute_43)
        amax_3: "f32[4000, 128, 1]" = torch.ops.aten.amax.default(baddbmm_3, [-1], True)
        sub_16: "f32[4000, 128, 128]" = torch.ops.aten.sub.Tensor(baddbmm_3, amax_3)
        exp_3: "f32[4000, 128, 128]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_4: "f32[4000, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[4000, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = None
        inductor_lookup_seed_default_12: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 12)
        inductor_random_default_14: "f32[4000, 128, 128]" = torch.ops.prims.inductor_random.default([4000, 128, 128], inductor_lookup_seed_default_12, 'rand');  inductor_lookup_seed_default_12 = None
        gt_12: "b8[4000, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_14, 0.1);  inductor_random_default_14 = None
        mul_63: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(gt_12, div_3);  div_3 = None
        mul_64: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(mul_63, 1.1111111111111112);  mul_63 = None
        bmm_3: "f32[4000, 128, 16]" = torch.ops.aten.bmm.default(mul_64, permute_42)
        permute_44: "f32[128, 4000, 16]" = torch.ops.aten.permute.default(bmm_3, [1, 0, 2]);  bmm_3 = None
        clone_18: "f32[128, 4000, 16]" = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
        view_56: "f32[64000, 128]" = torch.ops.aten.view.default(clone_18, [64000, 128]);  clone_18 = None
        permute_45: "f32[128, 128]" = torch.ops.aten.permute.default(primals_58, [1, 0])
        addmm_10: "f32[64000, 128]" = torch.ops.aten.addmm.default(primals_59, view_56, permute_45);  primals_59 = permute_45 = None
        view_57: "f32[128, 500, 128]" = torch.ops.aten.view.default(addmm_10, [128, 500, 128])
        permute_46: "f32[500, 128, 128]" = torch.ops.aten.permute.default(view_57, [1, 0, 2]);  view_57 = None
        
        # File: /app/src/models/particle_transformer.py:45 in forward, code: x = self.layernorm2(x)
        clone_19: "f32[500, 128, 128]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(clone_19, [2], correction = 0, keepdim = True)
        getitem_26: "f32[500, 128, 1]" = var_mean_13[0]
        getitem_27: "f32[500, 128, 1]" = var_mean_13[1];  var_mean_13 = None
        add_43: "f32[500, 128, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_13: "f32[500, 128, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_17: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(clone_19, getitem_27);  clone_19 = None
        mul_65: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_13);  sub_17 = None
        mul_66: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_65, primals_60);  mul_65 = None
        add_44: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_66, primals_61);  mul_66 = primals_61 = None
        
        # File: /app/src/models/particle_transformer.py:46 in forward, code: x = self.dropout(x)
        inductor_lookup_seed_default_13: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 13)
        inductor_random_default_13: "f32[500, 128, 128]" = torch.ops.prims.inductor_random.default([500, 128, 128], inductor_lookup_seed_default_13, 'rand');  inductor_lookup_seed_default_13 = None
        gt_13: "b8[500, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_13, 0.1);  inductor_random_default_13 = None
        mul_67: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(gt_13, add_44);  add_44 = None
        mul_68: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_67, 1.1111111111111112);  mul_67 = None
        
        # File: /app/src/models/particle_transformer.py:48 in forward, code: x += residual
        add_45: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_68, add_38);  mul_68 = add_38 = None
        
        # File: /app/src/models/feedforward.py:26 in forward, code: x = self.layernorm1(x)
        var_mean_14 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
        getitem_28: "f32[500, 128, 1]" = var_mean_14[0]
        getitem_29: "f32[500, 128, 1]" = var_mean_14[1];  var_mean_14 = None
        add_46: "f32[500, 128, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_14: "f32[500, 128, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_18: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(add_45, getitem_29);  getitem_29 = None
        mul_69: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_14);  sub_18 = None
        mul_70: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_69, primals_62)
        add_47: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_70, primals_63);  mul_70 = primals_63 = None
        
        # File: /app/src/models/feedforward.py:27 in forward, code: x = self.linear1(x)
        view_59: "f32[64000, 128]" = torch.ops.aten.view.default(add_47, [64000, 128]);  add_47 = None
        permute_47: "f32[128, 512]" = torch.ops.aten.permute.default(primals_64, [1, 0])
        addmm_11: "f32[64000, 512]" = torch.ops.aten.addmm.default(primals_65, view_59, permute_47);  primals_65 = permute_47 = None
        view_60: "f32[500, 128, 512]" = torch.ops.aten.view.default(addmm_11, [500, 128, 512])
        
        # File: /app/src/models/feedforward.py:28 in forward, code: x = self.act(x)
        mul_71: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_60, 0.5)
        mul_72: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_60, 0.7071067811865476);  view_60 = None
        erf_3: "f32[500, 128, 512]" = torch.ops.aten.erf.default(mul_72);  mul_72 = None
        add_48: "f32[500, 128, 512]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_73: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_71, add_48);  mul_71 = add_48 = None
        
        # File: /app/src/models/feedforward.py:29 in forward, code: x = self.dropout1(x)
        inductor_lookup_seed_default_14: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 14)
        inductor_random_default_12: "f32[500, 128, 512]" = torch.ops.prims.inductor_random.default([500, 128, 512], inductor_lookup_seed_default_14, 'rand');  inductor_lookup_seed_default_14 = None
        gt_14: "b8[500, 128, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_12, 0.1);  inductor_random_default_12 = None
        mul_74: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(gt_14, mul_73);  mul_73 = None
        mul_75: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_74, 1.1111111111111112);  mul_74 = None
        
        # File: /app/src/models/feedforward.py:31 in forward, code: x = self.layernorm2(x)
        var_mean_15 = torch.ops.aten.var_mean.correction(mul_75, [2], correction = 0, keepdim = True)
        getitem_30: "f32[500, 128, 1]" = var_mean_15[0]
        getitem_31: "f32[500, 128, 1]" = var_mean_15[1];  var_mean_15 = None
        add_49: "f32[500, 128, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_15: "f32[500, 128, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        sub_19: "f32[500, 128, 512]" = torch.ops.aten.sub.Tensor(mul_75, getitem_31);  mul_75 = None
        mul_76: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_15);  sub_19 = None
        mul_77: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_76, primals_66);  mul_76 = None
        add_50: "f32[500, 128, 512]" = torch.ops.aten.add.Tensor(mul_77, primals_67);  mul_77 = primals_67 = None
        
        # File: /app/src/models/feedforward.py:32 in forward, code: x = self.linear2(x)
        view_61: "f32[64000, 512]" = torch.ops.aten.view.default(add_50, [64000, 512]);  add_50 = None
        permute_48: "f32[512, 128]" = torch.ops.aten.permute.default(primals_68, [1, 0])
        addmm_12: "f32[64000, 128]" = torch.ops.aten.addmm.default(primals_69, view_61, permute_48);  primals_69 = permute_48 = None
        view_62: "f32[500, 128, 128]" = torch.ops.aten.view.default(addmm_12, [500, 128, 128]);  addmm_12 = None
        
        # File: /app/src/models/feedforward.py:33 in forward, code: x = self.dropout2(x)
        inductor_lookup_seed_default_15: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 15)
        inductor_random_default_11: "f32[500, 128, 128]" = torch.ops.prims.inductor_random.default([500, 128, 128], inductor_lookup_seed_default_15, 'rand');  inductor_lookup_seed_default_15 = None
        gt_15: "b8[500, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_11, 0.1);  inductor_random_default_11 = None
        mul_78: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(gt_15, view_62);  view_62 = None
        mul_79: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_78, 1.1111111111111112);  mul_78 = None
        
        # File: /app/src/models/feedforward.py:34 in forward, code: x += residual
        add_51: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_79, add_45);  mul_79 = add_45 = None
        
        # File: /app/src/models/particle_transformer.py:43 in forward, code: x = self.layernorm1(x)
        var_mean_16 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
        getitem_32: "f32[500, 128, 1]" = var_mean_16[0]
        getitem_33: "f32[500, 128, 1]" = var_mean_16[1];  var_mean_16 = None
        add_52: "f32[500, 128, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_16: "f32[500, 128, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        sub_20: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(add_51, getitem_33);  getitem_33 = None
        mul_80: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_16);  sub_20 = None
        mul_81: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_80, primals_70)
        add_53: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_81, primals_71);  mul_81 = primals_71 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        permute_49: "f32[128, 500, 128]" = torch.ops.aten.permute.default(add_53, [1, 0, 2]);  add_53 = None
        permute_50: "f32[128, 384]" = torch.ops.aten.permute.default(primals_72, [1, 0])
        clone_20: "f32[128, 500, 128]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        view_63: "f32[64000, 128]" = torch.ops.aten.view.default(clone_20, [64000, 128]);  clone_20 = None
        mm_4: "f32[64000, 384]" = torch.ops.aten.mm.default(view_63, permute_50);  permute_50 = None
        view_64: "f32[128, 500, 384]" = torch.ops.aten.view.default(mm_4, [128, 500, 384]);  mm_4 = None
        add_54: "f32[128, 500, 384]" = torch.ops.aten.add.Tensor(view_64, primals_73);  view_64 = primals_73 = None
        view_65: "f32[128, 500, 3, 128]" = torch.ops.aten.view.default(add_54, [128, 500, 3, 128]);  add_54 = None
        unsqueeze_4: "f32[1, 128, 500, 3, 128]" = torch.ops.aten.unsqueeze.default(view_65, 0);  view_65 = None
        permute_51: "f32[3, 128, 500, 1, 128]" = torch.ops.aten.permute.default(unsqueeze_4, [3, 1, 2, 0, 4]);  unsqueeze_4 = None
        squeeze_4: "f32[3, 128, 500, 128]" = torch.ops.aten.squeeze.dim(permute_51, -2);  permute_51 = None
        clone_21: "f32[3, 128, 500, 128]" = torch.ops.aten.clone.default(squeeze_4, memory_format = torch.contiguous_format);  squeeze_4 = None
        select_12: "f32[128, 500, 128]" = torch.ops.aten.select.int(clone_21, 0, 0)
        select_13: "f32[128, 500, 128]" = torch.ops.aten.select.int(clone_21, 0, 1)
        select_14: "f32[128, 500, 128]" = torch.ops.aten.select.int(clone_21, 0, 2);  clone_21 = None
        view_66: "f32[128, 4000, 16]" = torch.ops.aten.view.default(select_12, [128, 4000, 16]);  select_12 = None
        permute_52: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(view_66, [1, 0, 2]);  view_66 = None
        view_67: "f32[128, 4000, 16]" = torch.ops.aten.view.default(select_13, [128, 4000, 16]);  select_13 = None
        permute_53: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(view_67, [1, 0, 2]);  view_67 = None
        view_68: "f32[128, 4000, 16]" = torch.ops.aten.view.default(select_14, [128, 4000, 16]);  select_14 = None
        permute_54: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(view_68, [1, 0, 2]);  view_68 = None
        mul_82: "f32[4000, 128, 16]" = torch.ops.aten.mul.Tensor(permute_52, 0.25);  permute_52 = None
        permute_55: "f32[4000, 16, 128]" = torch.ops.aten.permute.default(permute_53, [0, 2, 1]);  permute_53 = None
        baddbmm_4: "f32[4000, 128, 128]" = torch.ops.aten.baddbmm.default(add_3, mul_82, permute_55)
        amax_4: "f32[4000, 128, 1]" = torch.ops.aten.amax.default(baddbmm_4, [-1], True)
        sub_21: "f32[4000, 128, 128]" = torch.ops.aten.sub.Tensor(baddbmm_4, amax_4)
        exp_4: "f32[4000, 128, 128]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
        sum_5: "f32[4000, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[4000, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = None
        inductor_lookup_seed_default_16: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 16)
        inductor_random_default_10: "f32[4000, 128, 128]" = torch.ops.prims.inductor_random.default([4000, 128, 128], inductor_lookup_seed_default_16, 'rand');  inductor_lookup_seed_default_16 = None
        gt_16: "b8[4000, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_10, 0.1);  inductor_random_default_10 = None
        mul_83: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(gt_16, div_4);  div_4 = None
        mul_84: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(mul_83, 1.1111111111111112);  mul_83 = None
        bmm_4: "f32[4000, 128, 16]" = torch.ops.aten.bmm.default(mul_84, permute_54)
        permute_56: "f32[128, 4000, 16]" = torch.ops.aten.permute.default(bmm_4, [1, 0, 2]);  bmm_4 = None
        clone_23: "f32[128, 4000, 16]" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
        view_71: "f32[64000, 128]" = torch.ops.aten.view.default(clone_23, [64000, 128]);  clone_23 = None
        permute_57: "f32[128, 128]" = torch.ops.aten.permute.default(primals_74, [1, 0])
        addmm_13: "f32[64000, 128]" = torch.ops.aten.addmm.default(primals_75, view_71, permute_57);  primals_75 = permute_57 = None
        view_72: "f32[128, 500, 128]" = torch.ops.aten.view.default(addmm_13, [128, 500, 128])
        permute_58: "f32[500, 128, 128]" = torch.ops.aten.permute.default(view_72, [1, 0, 2]);  view_72 = None
        
        # File: /app/src/models/particle_transformer.py:45 in forward, code: x = self.layernorm2(x)
        clone_24: "f32[500, 128, 128]" = torch.ops.aten.clone.default(permute_58, memory_format = torch.contiguous_format);  permute_58 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(clone_24, [2], correction = 0, keepdim = True)
        getitem_34: "f32[500, 128, 1]" = var_mean_17[0]
        getitem_35: "f32[500, 128, 1]" = var_mean_17[1];  var_mean_17 = None
        add_56: "f32[500, 128, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_17: "f32[500, 128, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        sub_22: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(clone_24, getitem_35);  clone_24 = None
        mul_85: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_17);  sub_22 = None
        mul_86: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_85, primals_76);  mul_85 = None
        add_57: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_86, primals_77);  mul_86 = primals_77 = None
        
        # File: /app/src/models/particle_transformer.py:46 in forward, code: x = self.dropout(x)
        inductor_lookup_seed_default_17: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 17)
        inductor_random_default_9: "f32[500, 128, 128]" = torch.ops.prims.inductor_random.default([500, 128, 128], inductor_lookup_seed_default_17, 'rand');  inductor_lookup_seed_default_17 = None
        gt_17: "b8[500, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_9, 0.1);  inductor_random_default_9 = None
        mul_87: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(gt_17, add_57);  add_57 = None
        mul_88: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_87, 1.1111111111111112);  mul_87 = None
        
        # File: /app/src/models/particle_transformer.py:48 in forward, code: x += residual
        add_58: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_88, add_51);  mul_88 = add_51 = None
        
        # File: /app/src/models/feedforward.py:26 in forward, code: x = self.layernorm1(x)
        var_mean_18 = torch.ops.aten.var_mean.correction(add_58, [2], correction = 0, keepdim = True)
        getitem_36: "f32[500, 128, 1]" = var_mean_18[0]
        getitem_37: "f32[500, 128, 1]" = var_mean_18[1];  var_mean_18 = None
        add_59: "f32[500, 128, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_18: "f32[500, 128, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
        sub_23: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(add_58, getitem_37);  getitem_37 = None
        mul_89: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_18);  sub_23 = None
        mul_90: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_89, primals_78)
        add_60: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_90, primals_79);  mul_90 = primals_79 = None
        
        # File: /app/src/models/feedforward.py:27 in forward, code: x = self.linear1(x)
        view_74: "f32[64000, 128]" = torch.ops.aten.view.default(add_60, [64000, 128]);  add_60 = None
        permute_59: "f32[128, 512]" = torch.ops.aten.permute.default(primals_80, [1, 0])
        addmm_14: "f32[64000, 512]" = torch.ops.aten.addmm.default(primals_81, view_74, permute_59);  primals_81 = permute_59 = None
        view_75: "f32[500, 128, 512]" = torch.ops.aten.view.default(addmm_14, [500, 128, 512])
        
        # File: /app/src/models/feedforward.py:28 in forward, code: x = self.act(x)
        mul_91: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_75, 0.5)
        mul_92: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_75, 0.7071067811865476);  view_75 = None
        erf_4: "f32[500, 128, 512]" = torch.ops.aten.erf.default(mul_92);  mul_92 = None
        add_61: "f32[500, 128, 512]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_93: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_91, add_61);  mul_91 = add_61 = None
        
        # File: /app/src/models/feedforward.py:29 in forward, code: x = self.dropout1(x)
        inductor_lookup_seed_default_18: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 18)
        inductor_random_default_8: "f32[500, 128, 512]" = torch.ops.prims.inductor_random.default([500, 128, 512], inductor_lookup_seed_default_18, 'rand');  inductor_lookup_seed_default_18 = None
        gt_18: "b8[500, 128, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_8, 0.1);  inductor_random_default_8 = None
        mul_94: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(gt_18, mul_93);  mul_93 = None
        mul_95: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_94, 1.1111111111111112);  mul_94 = None
        
        # File: /app/src/models/feedforward.py:31 in forward, code: x = self.layernorm2(x)
        var_mean_19 = torch.ops.aten.var_mean.correction(mul_95, [2], correction = 0, keepdim = True)
        getitem_38: "f32[500, 128, 1]" = var_mean_19[0]
        getitem_39: "f32[500, 128, 1]" = var_mean_19[1];  var_mean_19 = None
        add_62: "f32[500, 128, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_19: "f32[500, 128, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        sub_24: "f32[500, 128, 512]" = torch.ops.aten.sub.Tensor(mul_95, getitem_39);  mul_95 = None
        mul_96: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_19);  sub_24 = None
        mul_97: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_96, primals_82);  mul_96 = None
        add_63: "f32[500, 128, 512]" = torch.ops.aten.add.Tensor(mul_97, primals_83);  mul_97 = primals_83 = None
        
        # File: /app/src/models/feedforward.py:32 in forward, code: x = self.linear2(x)
        view_76: "f32[64000, 512]" = torch.ops.aten.view.default(add_63, [64000, 512]);  add_63 = None
        permute_60: "f32[512, 128]" = torch.ops.aten.permute.default(primals_84, [1, 0])
        addmm_15: "f32[64000, 128]" = torch.ops.aten.addmm.default(primals_85, view_76, permute_60);  primals_85 = permute_60 = None
        view_77: "f32[500, 128, 128]" = torch.ops.aten.view.default(addmm_15, [500, 128, 128]);  addmm_15 = None
        
        # File: /app/src/models/feedforward.py:33 in forward, code: x = self.dropout2(x)
        inductor_lookup_seed_default_19: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 19)
        inductor_random_default_7: "f32[500, 128, 128]" = torch.ops.prims.inductor_random.default([500, 128, 128], inductor_lookup_seed_default_19, 'rand');  inductor_lookup_seed_default_19 = None
        gt_19: "b8[500, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_7, 0.1);  inductor_random_default_7 = None
        mul_98: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(gt_19, view_77);  view_77 = None
        mul_99: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_98, 1.1111111111111112);  mul_98 = None
        
        # File: /app/src/models/feedforward.py:34 in forward, code: x += residual
        add_64: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_99, add_58);  mul_99 = add_58 = None
        
        # File: /app/src/models/particle_transformer.py:43 in forward, code: x = self.layernorm1(x)
        var_mean_20 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
        getitem_40: "f32[500, 128, 1]" = var_mean_20[0]
        getitem_41: "f32[500, 128, 1]" = var_mean_20[1];  var_mean_20 = None
        add_65: "f32[500, 128, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_20: "f32[500, 128, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        sub_25: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(add_64, getitem_41);  getitem_41 = None
        mul_100: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_20);  sub_25 = None
        mul_101: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_100, primals_86)
        add_66: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_101, primals_87);  mul_101 = primals_87 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        permute_61: "f32[128, 500, 128]" = torch.ops.aten.permute.default(add_66, [1, 0, 2]);  add_66 = None
        permute_62: "f32[128, 384]" = torch.ops.aten.permute.default(primals_88, [1, 0])
        clone_25: "f32[128, 500, 128]" = torch.ops.aten.clone.default(permute_61, memory_format = torch.contiguous_format);  permute_61 = None
        view_78: "f32[64000, 128]" = torch.ops.aten.view.default(clone_25, [64000, 128]);  clone_25 = None
        mm_5: "f32[64000, 384]" = torch.ops.aten.mm.default(view_78, permute_62);  permute_62 = None
        view_79: "f32[128, 500, 384]" = torch.ops.aten.view.default(mm_5, [128, 500, 384]);  mm_5 = None
        add_67: "f32[128, 500, 384]" = torch.ops.aten.add.Tensor(view_79, primals_89);  view_79 = primals_89 = None
        view_80: "f32[128, 500, 3, 128]" = torch.ops.aten.view.default(add_67, [128, 500, 3, 128]);  add_67 = None
        unsqueeze_5: "f32[1, 128, 500, 3, 128]" = torch.ops.aten.unsqueeze.default(view_80, 0);  view_80 = None
        permute_63: "f32[3, 128, 500, 1, 128]" = torch.ops.aten.permute.default(unsqueeze_5, [3, 1, 2, 0, 4]);  unsqueeze_5 = None
        squeeze_5: "f32[3, 128, 500, 128]" = torch.ops.aten.squeeze.dim(permute_63, -2);  permute_63 = None
        clone_26: "f32[3, 128, 500, 128]" = torch.ops.aten.clone.default(squeeze_5, memory_format = torch.contiguous_format);  squeeze_5 = None
        select_15: "f32[128, 500, 128]" = torch.ops.aten.select.int(clone_26, 0, 0)
        select_16: "f32[128, 500, 128]" = torch.ops.aten.select.int(clone_26, 0, 1)
        select_17: "f32[128, 500, 128]" = torch.ops.aten.select.int(clone_26, 0, 2);  clone_26 = None
        view_81: "f32[128, 4000, 16]" = torch.ops.aten.view.default(select_15, [128, 4000, 16]);  select_15 = None
        permute_64: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(view_81, [1, 0, 2]);  view_81 = None
        view_82: "f32[128, 4000, 16]" = torch.ops.aten.view.default(select_16, [128, 4000, 16]);  select_16 = None
        permute_65: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(view_82, [1, 0, 2]);  view_82 = None
        view_83: "f32[128, 4000, 16]" = torch.ops.aten.view.default(select_17, [128, 4000, 16]);  select_17 = None
        permute_66: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(view_83, [1, 0, 2]);  view_83 = None
        mul_102: "f32[4000, 128, 16]" = torch.ops.aten.mul.Tensor(permute_64, 0.25);  permute_64 = None
        permute_67: "f32[4000, 16, 128]" = torch.ops.aten.permute.default(permute_65, [0, 2, 1]);  permute_65 = None
        baddbmm_5: "f32[4000, 128, 128]" = torch.ops.aten.baddbmm.default(add_3, mul_102, permute_67)
        amax_5: "f32[4000, 128, 1]" = torch.ops.aten.amax.default(baddbmm_5, [-1], True)
        sub_26: "f32[4000, 128, 128]" = torch.ops.aten.sub.Tensor(baddbmm_5, amax_5)
        exp_5: "f32[4000, 128, 128]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
        sum_6: "f32[4000, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[4000, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = None
        inductor_lookup_seed_default_20: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 20)
        inductor_random_default_6: "f32[4000, 128, 128]" = torch.ops.prims.inductor_random.default([4000, 128, 128], inductor_lookup_seed_default_20, 'rand');  inductor_lookup_seed_default_20 = None
        gt_20: "b8[4000, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_6, 0.1);  inductor_random_default_6 = None
        mul_103: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(gt_20, div_5);  div_5 = None
        mul_104: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(mul_103, 1.1111111111111112);  mul_103 = None
        bmm_5: "f32[4000, 128, 16]" = torch.ops.aten.bmm.default(mul_104, permute_66)
        permute_68: "f32[128, 4000, 16]" = torch.ops.aten.permute.default(bmm_5, [1, 0, 2]);  bmm_5 = None
        clone_28: "f32[128, 4000, 16]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
        view_86: "f32[64000, 128]" = torch.ops.aten.view.default(clone_28, [64000, 128]);  clone_28 = None
        permute_69: "f32[128, 128]" = torch.ops.aten.permute.default(primals_90, [1, 0])
        addmm_16: "f32[64000, 128]" = torch.ops.aten.addmm.default(primals_91, view_86, permute_69);  primals_91 = permute_69 = None
        view_87: "f32[128, 500, 128]" = torch.ops.aten.view.default(addmm_16, [128, 500, 128])
        permute_70: "f32[500, 128, 128]" = torch.ops.aten.permute.default(view_87, [1, 0, 2]);  view_87 = None
        
        # File: /app/src/models/particle_transformer.py:45 in forward, code: x = self.layernorm2(x)
        clone_29: "f32[500, 128, 128]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(clone_29, [2], correction = 0, keepdim = True)
        getitem_42: "f32[500, 128, 1]" = var_mean_21[0]
        getitem_43: "f32[500, 128, 1]" = var_mean_21[1];  var_mean_21 = None
        add_69: "f32[500, 128, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_21: "f32[500, 128, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_27: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(clone_29, getitem_43);  clone_29 = None
        mul_105: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_21);  sub_27 = None
        mul_106: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_105, primals_92);  mul_105 = None
        add_70: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_106, primals_93);  mul_106 = primals_93 = None
        
        # File: /app/src/models/particle_transformer.py:46 in forward, code: x = self.dropout(x)
        inductor_lookup_seed_default_21: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 21)
        inductor_random_default_5: "f32[500, 128, 128]" = torch.ops.prims.inductor_random.default([500, 128, 128], inductor_lookup_seed_default_21, 'rand');  inductor_lookup_seed_default_21 = None
        gt_21: "b8[500, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_5, 0.1);  inductor_random_default_5 = None
        mul_107: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(gt_21, add_70);  add_70 = None
        mul_108: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_107, 1.1111111111111112);  mul_107 = None
        
        # File: /app/src/models/particle_transformer.py:48 in forward, code: x += residual
        add_71: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_108, add_64);  mul_108 = add_64 = None
        
        # File: /app/src/models/feedforward.py:26 in forward, code: x = self.layernorm1(x)
        var_mean_22 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
        getitem_44: "f32[500, 128, 1]" = var_mean_22[0]
        getitem_45: "f32[500, 128, 1]" = var_mean_22[1];  var_mean_22 = None
        add_72: "f32[500, 128, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_22: "f32[500, 128, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        sub_28: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(add_71, getitem_45);  getitem_45 = None
        mul_109: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_22);  sub_28 = None
        mul_110: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_109, primals_94)
        add_73: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_110, primals_95);  mul_110 = primals_95 = None
        
        # File: /app/src/models/feedforward.py:27 in forward, code: x = self.linear1(x)
        view_89: "f32[64000, 128]" = torch.ops.aten.view.default(add_73, [64000, 128]);  add_73 = None
        permute_71: "f32[128, 512]" = torch.ops.aten.permute.default(primals_96, [1, 0])
        addmm_17: "f32[64000, 512]" = torch.ops.aten.addmm.default(primals_97, view_89, permute_71);  primals_97 = permute_71 = None
        view_90: "f32[500, 128, 512]" = torch.ops.aten.view.default(addmm_17, [500, 128, 512])
        
        # File: /app/src/models/feedforward.py:28 in forward, code: x = self.act(x)
        mul_111: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_90, 0.5)
        mul_112: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_90, 0.7071067811865476);  view_90 = None
        erf_5: "f32[500, 128, 512]" = torch.ops.aten.erf.default(mul_112);  mul_112 = None
        add_74: "f32[500, 128, 512]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_113: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_111, add_74);  mul_111 = add_74 = None
        
        # File: /app/src/models/feedforward.py:29 in forward, code: x = self.dropout1(x)
        inductor_lookup_seed_default_22: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 22)
        inductor_random_default_4: "f32[500, 128, 512]" = torch.ops.prims.inductor_random.default([500, 128, 512], inductor_lookup_seed_default_22, 'rand');  inductor_lookup_seed_default_22 = None
        gt_22: "b8[500, 128, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_4, 0.1);  inductor_random_default_4 = None
        mul_114: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(gt_22, mul_113);  mul_113 = None
        mul_115: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_114, 1.1111111111111112);  mul_114 = None
        
        # File: /app/src/models/feedforward.py:31 in forward, code: x = self.layernorm2(x)
        var_mean_23 = torch.ops.aten.var_mean.correction(mul_115, [2], correction = 0, keepdim = True)
        getitem_46: "f32[500, 128, 1]" = var_mean_23[0]
        getitem_47: "f32[500, 128, 1]" = var_mean_23[1];  var_mean_23 = None
        add_75: "f32[500, 128, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_23: "f32[500, 128, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        sub_29: "f32[500, 128, 512]" = torch.ops.aten.sub.Tensor(mul_115, getitem_47);  mul_115 = None
        mul_116: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_23);  sub_29 = None
        mul_117: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_116, primals_98);  mul_116 = None
        add_76: "f32[500, 128, 512]" = torch.ops.aten.add.Tensor(mul_117, primals_99);  mul_117 = primals_99 = None
        
        # File: /app/src/models/feedforward.py:32 in forward, code: x = self.linear2(x)
        view_91: "f32[64000, 512]" = torch.ops.aten.view.default(add_76, [64000, 512]);  add_76 = None
        permute_72: "f32[512, 128]" = torch.ops.aten.permute.default(primals_100, [1, 0])
        addmm_18: "f32[64000, 128]" = torch.ops.aten.addmm.default(primals_101, view_91, permute_72);  primals_101 = permute_72 = None
        view_92: "f32[500, 128, 128]" = torch.ops.aten.view.default(addmm_18, [500, 128, 128]);  addmm_18 = None
        
        # File: /app/src/models/feedforward.py:33 in forward, code: x = self.dropout2(x)
        inductor_lookup_seed_default_23: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 23)
        inductor_random_default_3: "f32[500, 128, 128]" = torch.ops.prims.inductor_random.default([500, 128, 128], inductor_lookup_seed_default_23, 'rand');  inductor_lookup_seed_default_23 = None
        gt_23: "b8[500, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_3, 0.1);  inductor_random_default_3 = None
        mul_118: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(gt_23, view_92);  view_92 = None
        mul_119: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_118, 1.1111111111111112);  mul_118 = None
        
        # File: /app/src/models/feedforward.py:34 in forward, code: x += residual
        add_77: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_119, add_71);  mul_119 = add_71 = None
        
        # File: /app/src/models/particle_transformer.py:43 in forward, code: x = self.layernorm1(x)
        var_mean_24 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
        getitem_48: "f32[500, 128, 1]" = var_mean_24[0]
        getitem_49: "f32[500, 128, 1]" = var_mean_24[1];  var_mean_24 = None
        add_78: "f32[500, 128, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_24: "f32[500, 128, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        sub_30: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(add_77, getitem_49);  getitem_49 = None
        mul_120: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_24);  sub_30 = None
        mul_121: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_120, primals_102)
        add_79: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_121, primals_103);  mul_121 = primals_103 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        permute_73: "f32[128, 500, 128]" = torch.ops.aten.permute.default(add_79, [1, 0, 2]);  add_79 = None
        permute_74: "f32[128, 384]" = torch.ops.aten.permute.default(primals_104, [1, 0])
        clone_30: "f32[128, 500, 128]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_93: "f32[64000, 128]" = torch.ops.aten.view.default(clone_30, [64000, 128]);  clone_30 = None
        mm_6: "f32[64000, 384]" = torch.ops.aten.mm.default(view_93, permute_74);  permute_74 = None
        view_94: "f32[128, 500, 384]" = torch.ops.aten.view.default(mm_6, [128, 500, 384]);  mm_6 = None
        add_80: "f32[128, 500, 384]" = torch.ops.aten.add.Tensor(view_94, primals_105);  view_94 = primals_105 = None
        view_95: "f32[128, 500, 3, 128]" = torch.ops.aten.view.default(add_80, [128, 500, 3, 128]);  add_80 = None
        unsqueeze_6: "f32[1, 128, 500, 3, 128]" = torch.ops.aten.unsqueeze.default(view_95, 0);  view_95 = None
        permute_75: "f32[3, 128, 500, 1, 128]" = torch.ops.aten.permute.default(unsqueeze_6, [3, 1, 2, 0, 4]);  unsqueeze_6 = None
        squeeze_6: "f32[3, 128, 500, 128]" = torch.ops.aten.squeeze.dim(permute_75, -2);  permute_75 = None
        clone_31: "f32[3, 128, 500, 128]" = torch.ops.aten.clone.default(squeeze_6, memory_format = torch.contiguous_format);  squeeze_6 = None
        select_18: "f32[128, 500, 128]" = torch.ops.aten.select.int(clone_31, 0, 0)
        select_19: "f32[128, 500, 128]" = torch.ops.aten.select.int(clone_31, 0, 1)
        select_20: "f32[128, 500, 128]" = torch.ops.aten.select.int(clone_31, 0, 2);  clone_31 = None
        view_96: "f32[128, 4000, 16]" = torch.ops.aten.view.default(select_18, [128, 4000, 16]);  select_18 = None
        permute_76: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(view_96, [1, 0, 2]);  view_96 = None
        view_97: "f32[128, 4000, 16]" = torch.ops.aten.view.default(select_19, [128, 4000, 16]);  select_19 = None
        permute_77: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(view_97, [1, 0, 2]);  view_97 = None
        view_98: "f32[128, 4000, 16]" = torch.ops.aten.view.default(select_20, [128, 4000, 16]);  select_20 = None
        permute_78: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(view_98, [1, 0, 2]);  view_98 = None
        mul_122: "f32[4000, 128, 16]" = torch.ops.aten.mul.Tensor(permute_76, 0.25);  permute_76 = None
        permute_79: "f32[4000, 16, 128]" = torch.ops.aten.permute.default(permute_77, [0, 2, 1]);  permute_77 = None
        baddbmm_6: "f32[4000, 128, 128]" = torch.ops.aten.baddbmm.default(add_3, mul_122, permute_79);  add_3 = None
        amax_6: "f32[4000, 128, 1]" = torch.ops.aten.amax.default(baddbmm_6, [-1], True)
        sub_31: "f32[4000, 128, 128]" = torch.ops.aten.sub.Tensor(baddbmm_6, amax_6)
        exp_6: "f32[4000, 128, 128]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_7: "f32[4000, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[4000, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = None
        inductor_lookup_seed_default_24: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 24)
        inductor_random_default_2: "f32[4000, 128, 128]" = torch.ops.prims.inductor_random.default([4000, 128, 128], inductor_lookup_seed_default_24, 'rand');  inductor_lookup_seed_default_24 = None
        gt_24: "b8[4000, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_2, 0.1);  inductor_random_default_2 = None
        mul_123: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(gt_24, div_6);  div_6 = None
        mul_124: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(mul_123, 1.1111111111111112);  mul_123 = None
        bmm_6: "f32[4000, 128, 16]" = torch.ops.aten.bmm.default(mul_124, permute_78)
        permute_80: "f32[128, 4000, 16]" = torch.ops.aten.permute.default(bmm_6, [1, 0, 2]);  bmm_6 = None
        clone_33: "f32[128, 4000, 16]" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
        view_101: "f32[64000, 128]" = torch.ops.aten.view.default(clone_33, [64000, 128]);  clone_33 = None
        permute_81: "f32[128, 128]" = torch.ops.aten.permute.default(primals_106, [1, 0])
        addmm_19: "f32[64000, 128]" = torch.ops.aten.addmm.default(primals_107, view_101, permute_81);  primals_107 = permute_81 = None
        view_102: "f32[128, 500, 128]" = torch.ops.aten.view.default(addmm_19, [128, 500, 128])
        permute_82: "f32[500, 128, 128]" = torch.ops.aten.permute.default(view_102, [1, 0, 2]);  view_102 = None
        
        # File: /app/src/models/particle_transformer.py:45 in forward, code: x = self.layernorm2(x)
        clone_34: "f32[500, 128, 128]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(clone_34, [2], correction = 0, keepdim = True)
        getitem_50: "f32[500, 128, 1]" = var_mean_25[0]
        getitem_51: "f32[500, 128, 1]" = var_mean_25[1];  var_mean_25 = None
        add_82: "f32[500, 128, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_25: "f32[500, 128, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_32: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(clone_34, getitem_51);  clone_34 = None
        mul_125: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_25);  sub_32 = None
        mul_126: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_125, primals_108);  mul_125 = None
        add_83: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_126, primals_109);  mul_126 = primals_109 = None
        
        # File: /app/src/models/particle_transformer.py:46 in forward, code: x = self.dropout(x)
        inductor_lookup_seed_default_25: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 25)
        inductor_random_default_1: "f32[500, 128, 128]" = torch.ops.prims.inductor_random.default([500, 128, 128], inductor_lookup_seed_default_25, 'rand');  inductor_lookup_seed_default_25 = None
        gt_25: "b8[500, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.1);  inductor_random_default_1 = None
        mul_127: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(gt_25, add_83);  add_83 = None
        mul_128: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_127, 1.1111111111111112);  mul_127 = None
        
        # File: /app/src/models/particle_transformer.py:48 in forward, code: x += residual
        add_84: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_128, add_77);  mul_128 = add_77 = None
        
        # File: /app/src/models/feedforward.py:26 in forward, code: x = self.layernorm1(x)
        var_mean_26 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
        getitem_52: "f32[500, 128, 1]" = var_mean_26[0]
        getitem_53: "f32[500, 128, 1]" = var_mean_26[1];  var_mean_26 = None
        add_85: "f32[500, 128, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_26: "f32[500, 128, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        sub_33: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(add_84, getitem_53)
        mul_129: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_26);  sub_33 = None
        mul_130: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_129, primals_110);  mul_129 = None
        add_86: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_130, primals_111);  mul_130 = primals_111 = None
        
        # File: /app/src/models/feedforward.py:27 in forward, code: x = self.linear1(x)
        view_104: "f32[64000, 128]" = torch.ops.aten.view.default(add_86, [64000, 128]);  add_86 = None
        permute_83: "f32[128, 512]" = torch.ops.aten.permute.default(primals_112, [1, 0])
        addmm_20: "f32[64000, 512]" = torch.ops.aten.addmm.default(primals_113, view_104, permute_83);  primals_113 = permute_83 = None
        view_105: "f32[500, 128, 512]" = torch.ops.aten.view.default(addmm_20, [500, 128, 512])
        
        # File: /app/src/models/feedforward.py:28 in forward, code: x = self.act(x)
        mul_131: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_105, 0.5)
        mul_132: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_105, 0.7071067811865476);  view_105 = None
        erf_6: "f32[500, 128, 512]" = torch.ops.aten.erf.default(mul_132);  mul_132 = None
        add_87: "f32[500, 128, 512]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_133: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_131, add_87);  mul_131 = add_87 = None
        
        # File: /app/src/models/feedforward.py:29 in forward, code: x = self.dropout1(x)
        inductor_lookup_seed_default_26: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 26);  inductor_seeds_default = None
        inductor_random_default: "f32[500, 128, 512]" = torch.ops.prims.inductor_random.default([500, 128, 512], inductor_lookup_seed_default_26, 'rand');  inductor_lookup_seed_default_26 = None
        gt_26: "b8[500, 128, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_134: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(gt_26, mul_133);  mul_133 = None
        mul_135: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_134, 1.1111111111111112);  mul_134 = None
        
        # File: /app/src/models/feedforward.py:31 in forward, code: x = self.layernorm2(x)
        var_mean_27 = torch.ops.aten.var_mean.correction(mul_135, [2], correction = 0, keepdim = True)
        getitem_54: "f32[500, 128, 1]" = var_mean_27[0]
        getitem_55: "f32[500, 128, 1]" = var_mean_27[1];  var_mean_27 = None
        add_88: "f32[500, 128, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
        rsqrt_27: "f32[500, 128, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        sub_34: "f32[500, 128, 512]" = torch.ops.aten.sub.Tensor(mul_135, getitem_55);  mul_135 = None
        mul_136: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_27);  sub_34 = None
        mul_137: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_136, primals_114);  mul_136 = None
        add_89: "f32[500, 128, 512]" = torch.ops.aten.add.Tensor(mul_137, primals_115);  mul_137 = primals_115 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        permute_94: "f32[4000, 128, 128]" = torch.ops.aten.permute.default(mul_124, [0, 2, 1]);  mul_124 = None
        permute_95: "f32[4000, 16, 128]" = torch.ops.aten.permute.default(permute_78, [0, 2, 1]);  permute_78 = None
        permute_96: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(permute_79, [0, 2, 1]);  permute_79 = None
        permute_97: "f32[4000, 16, 128]" = torch.ops.aten.permute.default(mul_122, [0, 2, 1]);  mul_122 = None
        
        # File: /app/src/models/particle_transformer.py:43 in forward, code: x = self.layernorm1(x)
        div_10: "f32[500, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 128);  rsqrt_24 = None
        
        # File: /app/src/models/feedforward.py:26 in forward, code: x = self.layernorm1(x)
        div_12: "f32[500, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 128);  rsqrt_22 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        permute_122: "f32[4000, 128, 128]" = torch.ops.aten.permute.default(mul_104, [0, 2, 1]);  mul_104 = None
        permute_123: "f32[4000, 16, 128]" = torch.ops.aten.permute.default(permute_66, [0, 2, 1]);  permute_66 = None
        permute_124: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(permute_67, [0, 2, 1]);  permute_67 = None
        permute_125: "f32[4000, 16, 128]" = torch.ops.aten.permute.default(mul_102, [0, 2, 1]);  mul_102 = None
        
        # File: /app/src/models/particle_transformer.py:43 in forward, code: x = self.layernorm1(x)
        div_14: "f32[500, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 128);  rsqrt_20 = None
        
        # File: /app/src/models/feedforward.py:26 in forward, code: x = self.layernorm1(x)
        div_16: "f32[500, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 128);  rsqrt_18 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        permute_150: "f32[4000, 128, 128]" = torch.ops.aten.permute.default(mul_84, [0, 2, 1]);  mul_84 = None
        permute_151: "f32[4000, 16, 128]" = torch.ops.aten.permute.default(permute_54, [0, 2, 1]);  permute_54 = None
        permute_152: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(permute_55, [0, 2, 1]);  permute_55 = None
        permute_153: "f32[4000, 16, 128]" = torch.ops.aten.permute.default(mul_82, [0, 2, 1]);  mul_82 = None
        
        # File: /app/src/models/particle_transformer.py:43 in forward, code: x = self.layernorm1(x)
        div_18: "f32[500, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 128);  rsqrt_16 = None
        
        # File: /app/src/models/feedforward.py:26 in forward, code: x = self.layernorm1(x)
        div_20: "f32[500, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 128);  rsqrt_14 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        permute_178: "f32[4000, 128, 128]" = torch.ops.aten.permute.default(mul_64, [0, 2, 1]);  mul_64 = None
        permute_179: "f32[4000, 16, 128]" = torch.ops.aten.permute.default(permute_42, [0, 2, 1]);  permute_42 = None
        permute_180: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(permute_43, [0, 2, 1]);  permute_43 = None
        permute_181: "f32[4000, 16, 128]" = torch.ops.aten.permute.default(mul_62, [0, 2, 1]);  mul_62 = None
        
        # File: /app/src/models/particle_transformer.py:43 in forward, code: x = self.layernorm1(x)
        div_22: "f32[500, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 128);  rsqrt_12 = None
        
        # File: /app/src/models/feedforward.py:26 in forward, code: x = self.layernorm1(x)
        div_24: "f32[500, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 128);  rsqrt_10 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        permute_206: "f32[4000, 128, 128]" = torch.ops.aten.permute.default(mul_44, [0, 2, 1]);  mul_44 = None
        permute_207: "f32[4000, 16, 128]" = torch.ops.aten.permute.default(permute_30, [0, 2, 1]);  permute_30 = None
        permute_208: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(permute_31, [0, 2, 1]);  permute_31 = None
        permute_209: "f32[4000, 16, 128]" = torch.ops.aten.permute.default(mul_42, [0, 2, 1]);  mul_42 = None
        
        # File: /app/src/models/particle_transformer.py:43 in forward, code: x = self.layernorm1(x)
        div_26: "f32[500, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 128);  rsqrt_8 = None
        
        # File: /app/src/models/feedforward.py:26 in forward, code: x = self.layernorm1(x)
        div_28: "f32[500, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 128);  rsqrt_6 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        permute_234: "f32[4000, 128, 128]" = torch.ops.aten.permute.default(mul_24, [0, 2, 1]);  mul_24 = None
        permute_235: "f32[4000, 16, 128]" = torch.ops.aten.permute.default(permute_18, [0, 2, 1]);  permute_18 = None
        permute_236: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(permute_19, [0, 2, 1]);  permute_19 = None
        permute_237: "f32[4000, 16, 128]" = torch.ops.aten.permute.default(mul_22, [0, 2, 1]);  mul_22 = None
        
        # File: /app/src/models/particle_transformer.py:43 in forward, code: x = self.layernorm1(x)
        div_30: "f32[500, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 128);  rsqrt_4 = None
        
        # File: /app/src/models/feedforward.py:26 in forward, code: x = self.layernorm1(x)
        div_32: "f32[500, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 128);  rsqrt_2 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        permute_262: "f32[4000, 128, 128]" = torch.ops.aten.permute.default(mul_4, [0, 2, 1]);  mul_4 = None
        permute_263: "f32[4000, 16, 128]" = torch.ops.aten.permute.default(permute_6, [0, 2, 1]);  permute_6 = None
        permute_264: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(permute_7, [0, 2, 1]);  permute_7 = None
        permute_265: "f32[4000, 16, 128]" = torch.ops.aten.permute.default(mul_2, [0, 2, 1]);  mul_2 = None
        return (add_89, add_84, primals_2, primals_4, primals_6, primals_8, primals_12, primals_14, primals_16, primals_18, primals_20, primals_22, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_38, primals_40, primals_42, primals_44, primals_46, primals_48, primals_50, primals_52, primals_54, primals_56, primals_58, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_106, primals_108, primals_110, primals_112, primals_114, view_1, addmm, getitem_1, rsqrt, view_3, baddbmm, amax, sum_1, gt, view_11, addmm_1, getitem_3, rsqrt_1, gt_1, mul_9, view_14, addmm_2, gt_2, getitem_7, rsqrt_3, view_16, gt_3, mul_20, view_18, baddbmm_1, amax_1, sum_2, gt_4, view_26, addmm_4, getitem_11, rsqrt_5, gt_5, mul_29, view_29, addmm_5, gt_6, getitem_15, rsqrt_7, view_31, gt_7, mul_40, view_33, baddbmm_2, amax_2, sum_3, gt_8, view_41, addmm_7, getitem_19, rsqrt_9, gt_9, mul_49, view_44, addmm_8, gt_10, getitem_23, rsqrt_11, view_46, gt_11, mul_60, view_48, baddbmm_3, amax_3, sum_4, gt_12, view_56, addmm_10, getitem_27, rsqrt_13, gt_13, mul_69, view_59, addmm_11, gt_14, getitem_31, rsqrt_15, view_61, gt_15, mul_80, view_63, baddbmm_4, amax_4, sum_5, gt_16, view_71, addmm_13, getitem_35, rsqrt_17, gt_17, mul_89, view_74, addmm_14, gt_18, getitem_39, rsqrt_19, view_76, gt_19, mul_100, view_78, baddbmm_5, amax_5, sum_6, gt_20, view_86, addmm_16, getitem_43, rsqrt_21, gt_21, mul_109, view_89, addmm_17, gt_22, getitem_47, rsqrt_23, view_91, gt_23, mul_120, view_93, baddbmm_6, amax_6, sum_7, gt_24, view_101, addmm_19, getitem_51, rsqrt_25, gt_25, add_84, getitem_53, rsqrt_26, view_104, addmm_20, gt_26, getitem_55, rsqrt_27, permute_94, permute_95, permute_96, permute_97, div_10, div_12, permute_122, permute_123, permute_124, permute_125, div_14, div_16, permute_150, permute_151, permute_152, permute_153, div_18, div_20, permute_178, permute_179, permute_180, permute_181, div_22, div_24, permute_206, permute_207, permute_208, permute_209, div_26, div_28, permute_234, permute_235, permute_236, permute_237, div_30, div_32, permute_262, permute_263, permute_264, permute_265)
        