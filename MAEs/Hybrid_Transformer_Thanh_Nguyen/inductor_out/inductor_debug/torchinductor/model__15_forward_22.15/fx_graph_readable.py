class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[500, 128, 512]", primals_2: "f32[128, 512]", primals_3: "f32[128]", primals_4: "f32[500, 128, 128]", primals_5: "f32[128]", primals_6: "f32[128]", primals_7: "f32[384, 128]", primals_8: "f32[384]", primals_9: "f32[128, 128]", primals_10: "f32[128]", primals_11: "f32[500, 128]", primals_12: "f32[4000, 128, 128]", primals_13: "f32[128]", primals_14: "f32[128]", primals_15: "f32[128]", primals_16: "f32[128]", primals_17: "f32[512, 128]", primals_18: "f32[512]", primals_19: "f32[512]", primals_20: "f32[512]", primals_21: "f32[128, 512]", primals_22: "f32[128]"):
        # File: /app/src/models/feedforward.py:32 in forward, code: x = self.linear2(x)
        view: "f32[64000, 512]" = torch.ops.aten.view.default(primals_1, [64000, 512]);  primals_1 = None
        permute: "f32[512, 128]" = torch.ops.aten.permute.default(primals_2, [1, 0])
        addmm: "f32[64000, 128]" = torch.ops.aten.addmm.default(primals_3, view, permute);  primals_3 = permute = None
        view_1: "f32[500, 128, 128]" = torch.ops.aten.view.default(addmm, [500, 128, 128]);  addmm = None
        
        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[5]" = torch.ops.prims.inductor_seeds.default(5, device(type='cuda', index=0))
        
        # File: /app/src/models/feedforward.py:33 in forward, code: x = self.dropout2(x)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_4: "f32[500, 128, 128]" = torch.ops.prims.inductor_random.default([500, 128, 128], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[500, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_4, 0.1);  inductor_random_default_4 = None
        mul: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(gt, view_1);  view_1 = None
        mul_1: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        
        # File: /app/src/models/feedforward.py:34 in forward, code: x += residual
        add: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_1, primals_4);  mul_1 = primals_4 = None
        
        # File: /app/src/models/particle_transformer.py:43 in forward, code: x = self.layernorm1(x)
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[500, 128, 1]" = var_mean[0]
        getitem_1: "f32[500, 128, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[500, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[500, 128, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(add, getitem_1);  getitem_1 = None
        mul_2: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_3: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_2, primals_5)
        add_2: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_3, primals_6);  mul_3 = primals_6 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        permute_1: "f32[128, 500, 128]" = torch.ops.aten.permute.default(add_2, [1, 0, 2]);  add_2 = None
        permute_2: "f32[128, 384]" = torch.ops.aten.permute.default(primals_7, [1, 0])
        clone: "f32[128, 500, 128]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_2: "f32[64000, 128]" = torch.ops.aten.view.default(clone, [64000, 128]);  clone = None
        mm: "f32[64000, 384]" = torch.ops.aten.mm.default(view_2, permute_2);  permute_2 = None
        view_3: "f32[128, 500, 384]" = torch.ops.aten.view.default(mm, [128, 500, 384]);  mm = None
        add_3: "f32[128, 500, 384]" = torch.ops.aten.add.Tensor(view_3, primals_8);  view_3 = primals_8 = None
        view_4: "f32[128, 500, 3, 128]" = torch.ops.aten.view.default(add_3, [128, 500, 3, 128]);  add_3 = None
        unsqueeze: "f32[1, 128, 500, 3, 128]" = torch.ops.aten.unsqueeze.default(view_4, 0);  view_4 = None
        permute_3: "f32[3, 128, 500, 1, 128]" = torch.ops.aten.permute.default(unsqueeze, [3, 1, 2, 0, 4]);  unsqueeze = None
        squeeze: "f32[3, 128, 500, 128]" = torch.ops.aten.squeeze.dim(permute_3, -2);  permute_3 = None
        clone_1: "f32[3, 128, 500, 128]" = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
        select: "f32[128, 500, 128]" = torch.ops.aten.select.int(clone_1, 0, 0)
        select_1: "f32[128, 500, 128]" = torch.ops.aten.select.int(clone_1, 0, 1)
        select_2: "f32[128, 500, 128]" = torch.ops.aten.select.int(clone_1, 0, 2);  clone_1 = None
        view_5: "f32[128, 4000, 16]" = torch.ops.aten.view.default(select, [128, 4000, 16]);  select = None
        permute_4: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(view_5, [1, 0, 2]);  view_5 = None
        view_6: "f32[128, 4000, 16]" = torch.ops.aten.view.default(select_1, [128, 4000, 16]);  select_1 = None
        permute_5: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(view_6, [1, 0, 2]);  view_6 = None
        view_7: "f32[128, 4000, 16]" = torch.ops.aten.view.default(select_2, [128, 4000, 16]);  select_2 = None
        permute_6: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(view_7, [1, 0, 2]);  view_7 = None
        view_8: "f32[500, 1, 1, 128]" = torch.ops.aten.view.default(primals_11, [500, 1, 1, 128]);  primals_11 = None
        expand: "f32[500, 8, 1, 128]" = torch.ops.aten.expand.default(view_8, [-1, 8, -1, -1]);  view_8 = None
        clone_2: "f32[500, 8, 1, 128]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_9: "f32[4000, 1, 128]" = torch.ops.aten.view.default(clone_2, [4000, 1, 128]);  clone_2 = None
        add_4: "f32[4000, 128, 128]" = torch.ops.aten.add.Tensor(primals_12, view_9);  primals_12 = view_9 = None
        mul_4: "f32[4000, 128, 16]" = torch.ops.aten.mul.Tensor(permute_4, 0.25);  permute_4 = None
        permute_7: "f32[4000, 16, 128]" = torch.ops.aten.permute.default(permute_5, [0, 2, 1]);  permute_5 = None
        baddbmm: "f32[4000, 128, 128]" = torch.ops.aten.baddbmm.default(add_4, mul_4, permute_7);  add_4 = None
        amax: "f32[4000, 128, 1]" = torch.ops.aten.amax.default(baddbmm, [-1], True)
        sub_1: "f32[4000, 128, 128]" = torch.ops.aten.sub.Tensor(baddbmm, amax)
        exp: "f32[4000, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[4000, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[4000, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_3: "f32[4000, 128, 128]" = torch.ops.prims.inductor_random.default([4000, 128, 128], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_1: "b8[4000, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_3, 0.1);  inductor_random_default_3 = None
        mul_5: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(gt_1, div);  div = None
        mul_6: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(mul_5, 1.1111111111111112);  mul_5 = None
        bmm: "f32[4000, 128, 16]" = torch.ops.aten.bmm.default(mul_6, permute_6)
        permute_8: "f32[128, 4000, 16]" = torch.ops.aten.permute.default(bmm, [1, 0, 2]);  bmm = None
        clone_3: "f32[128, 4000, 16]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        view_10: "f32[64000, 128]" = torch.ops.aten.view.default(clone_3, [64000, 128]);  clone_3 = None
        permute_9: "f32[128, 128]" = torch.ops.aten.permute.default(primals_9, [1, 0])
        addmm_1: "f32[64000, 128]" = torch.ops.aten.addmm.default(primals_10, view_10, permute_9);  primals_10 = permute_9 = None
        view_11: "f32[128, 500, 128]" = torch.ops.aten.view.default(addmm_1, [128, 500, 128])
        permute_10: "f32[500, 128, 128]" = torch.ops.aten.permute.default(view_11, [1, 0, 2]);  view_11 = None
        
        # File: /app/src/models/particle_transformer.py:45 in forward, code: x = self.layernorm2(x)
        clone_4: "f32[500, 128, 128]" = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(clone_4, [2], correction = 0, keepdim = True)
        getitem_2: "f32[500, 128, 1]" = var_mean_1[0]
        getitem_3: "f32[500, 128, 1]" = var_mean_1[1];  var_mean_1 = None
        add_5: "f32[500, 128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[500, 128, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_2: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(clone_4, getitem_3);  clone_4 = None
        mul_7: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_8: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_7, primals_13);  mul_7 = None
        add_6: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_8, primals_14);  mul_8 = primals_14 = None
        
        # File: /app/src/models/particle_transformer.py:46 in forward, code: x = self.dropout(x)
        inductor_lookup_seed_default_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_2: "f32[500, 128, 128]" = torch.ops.prims.inductor_random.default([500, 128, 128], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        gt_2: "b8[500, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_2, 0.1);  inductor_random_default_2 = None
        mul_9: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(gt_2, add_6);  add_6 = None
        mul_10: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_9, 1.1111111111111112);  mul_9 = None
        
        # File: /app/src/models/particle_transformer.py:48 in forward, code: x += residual
        add_7: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_10, add);  mul_10 = add = None
        
        # File: /app/src/models/feedforward.py:26 in forward, code: x = self.layernorm1(x)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
        getitem_4: "f32[500, 128, 1]" = var_mean_2[0]
        getitem_5: "f32[500, 128, 1]" = var_mean_2[1];  var_mean_2 = None
        add_8: "f32[500, 128, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2: "f32[500, 128, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_3: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(add_7, getitem_5);  getitem_5 = None
        mul_11: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_12: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_11, primals_15)
        add_9: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_12, primals_16);  mul_12 = primals_16 = None
        
        # File: /app/src/models/feedforward.py:27 in forward, code: x = self.linear1(x)
        view_13: "f32[64000, 128]" = torch.ops.aten.view.default(add_9, [64000, 128]);  add_9 = None
        permute_11: "f32[128, 512]" = torch.ops.aten.permute.default(primals_17, [1, 0])
        addmm_2: "f32[64000, 512]" = torch.ops.aten.addmm.default(primals_18, view_13, permute_11);  primals_18 = permute_11 = None
        view_14: "f32[500, 128, 512]" = torch.ops.aten.view.default(addmm_2, [500, 128, 512])
        
        # File: /app/src/models/feedforward.py:28 in forward, code: x = self.act(x)
        mul_13: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_14, 0.5)
        mul_14: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_14, 0.7071067811865476);  view_14 = None
        erf: "f32[500, 128, 512]" = torch.ops.aten.erf.default(mul_14);  mul_14 = None
        add_10: "f32[500, 128, 512]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_15: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_13, add_10);  mul_13 = add_10 = None
        
        # File: /app/src/models/feedforward.py:29 in forward, code: x = self.dropout1(x)
        inductor_lookup_seed_default_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_1: "f32[500, 128, 512]" = torch.ops.prims.inductor_random.default([500, 128, 512], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        gt_3: "b8[500, 128, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.1);  inductor_random_default_1 = None
        mul_16: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(gt_3, mul_15);  mul_15 = None
        mul_17: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_16, 1.1111111111111112);  mul_16 = None
        
        # File: /app/src/models/feedforward.py:31 in forward, code: x = self.layernorm2(x)
        var_mean_3 = torch.ops.aten.var_mean.correction(mul_17, [2], correction = 0, keepdim = True)
        getitem_6: "f32[500, 128, 1]" = var_mean_3[0]
        getitem_7: "f32[500, 128, 1]" = var_mean_3[1];  var_mean_3 = None
        add_11: "f32[500, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3: "f32[500, 128, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_4: "f32[500, 128, 512]" = torch.ops.aten.sub.Tensor(mul_17, getitem_7);  mul_17 = None
        mul_18: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = None
        mul_19: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_18, primals_19);  mul_18 = None
        add_12: "f32[500, 128, 512]" = torch.ops.aten.add.Tensor(mul_19, primals_20);  mul_19 = primals_20 = None
        
        # File: /app/src/models/feedforward.py:32 in forward, code: x = self.linear2(x)
        view_15: "f32[64000, 512]" = torch.ops.aten.view.default(add_12, [64000, 512]);  add_12 = None
        permute_12: "f32[512, 128]" = torch.ops.aten.permute.default(primals_21, [1, 0])
        addmm_3: "f32[64000, 128]" = torch.ops.aten.addmm.default(primals_22, view_15, permute_12);  primals_22 = permute_12 = None
        view_16: "f32[500, 128, 128]" = torch.ops.aten.view.default(addmm_3, [500, 128, 128]);  addmm_3 = None
        
        # File: /app/src/models/feedforward.py:33 in forward, code: x = self.dropout2(x)
        inductor_lookup_seed_default_4: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4);  inductor_seeds_default = None
        inductor_random_default: "f32[500, 128, 128]" = torch.ops.prims.inductor_random.default([500, 128, 128], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        gt_4: "b8[500, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_20: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(gt_4, view_16);  view_16 = None
        mul_21: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_20, 1.1111111111111112);  mul_20 = None
        
        # File: /app/src/models/feedforward.py:34 in forward, code: x += residual
        add_13: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(mul_21, add_7);  mul_21 = add_7 = None
        
        # File: /app/src/models/feedforward.py:26 in forward, code: x = self.layernorm1(x)
        div_2: "f32[500, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 128);  rsqrt_2 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        permute_27: "f32[4000, 128, 128]" = torch.ops.aten.permute.default(mul_6, [0, 2, 1]);  mul_6 = None
        permute_28: "f32[4000, 16, 128]" = torch.ops.aten.permute.default(permute_6, [0, 2, 1]);  permute_6 = None
        permute_29: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(permute_7, [0, 2, 1]);  permute_7 = None
        permute_30: "f32[4000, 16, 128]" = torch.ops.aten.permute.default(mul_4, [0, 2, 1]);  mul_4 = None
        
        # File: /app/src/models/particle_transformer.py:43 in forward, code: x = self.layernorm1(x)
        div_4: "f32[500, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 128);  rsqrt = None
        return (add_13, primals_2, primals_5, primals_7, primals_9, primals_13, primals_15, primals_17, primals_19, primals_21, view, gt, mul_2, view_2, baddbmm, amax, sum_1, gt_1, view_10, addmm_1, getitem_3, rsqrt_1, gt_2, mul_11, view_13, addmm_2, gt_3, getitem_7, rsqrt_3, view_15, gt_4, div_2, permute_27, permute_28, permute_29, permute_30, div_4)
        