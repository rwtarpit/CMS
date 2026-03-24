class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[500, 1, 512]", primals_2: "f32[128, 512]", primals_3: "f32[128]", primals_4: "f32[500, 1, 128]", primals_5: "f32[500, 128]", primals_6: "f32[500, 128, 128]", primals_7: "f32[128]", primals_8: "f32[128]", primals_9: "f32[384, 128]", primals_10: "f32[384]", primals_11: "f32[128, 128]", primals_12: "f32[128]", primals_13: "f32[128]", primals_14: "f32[128]", primals_15: "f32[128]", primals_16: "f32[128]", primals_17: "f32[512, 128]", primals_18: "f32[512]", primals_19: "f32[512]", primals_20: "f32[512]", primals_21: "f32[128, 512]", primals_22: "f32[128]", primals_23: "f32[128]", primals_24: "f32[128]", primals_25: "f32[10, 128]", primals_26: "f32[10]"):
        # File: /app/src/models/feedforward.py:32 in forward, code: x = self.linear2(x)
        view: "f32[500, 512]" = torch.ops.aten.view.default(primals_1, [500, 512]);  primals_1 = None
        permute: "f32[512, 128]" = torch.ops.aten.permute.default(primals_2, [1, 0])
        addmm: "f32[500, 128]" = torch.ops.aten.addmm.default(primals_3, view, permute);  primals_3 = permute = None
        view_1: "f32[500, 1, 128]" = torch.ops.aten.view.default(addmm, [500, 1, 128]);  addmm = None
        
        # File: /app/src/models/feedforward.py:34 in forward, code: x += residual
        add: "f32[500, 1, 128]" = torch.ops.aten.add.Tensor(view_1, primals_4);  view_1 = primals_4 = None
        
        # File: /app/src/models/classifier.py:41 in forward, code: padding_mask = torch.cat((torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1)
        full_default: "f32[500, 1]" = torch.ops.aten.full.default([500, 1], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat: "f32[500, 129]" = torch.ops.aten.cat.default([full_default, primals_5], 1);  full_default = primals_5 = None
        
        # File: /app/src/models/classifier.py:44 in forward, code: x = torch.cat((x_cls, x), dim=1)  # (B, N + 1, D)
        cat_1: "f32[500, 129, 128]" = torch.ops.aten.cat.default([add, primals_6], 1);  primals_6 = None
        
        # File: /app/src/models/classifier.py:45 in forward, code: x = self.layernorm1(x)
        var_mean = torch.ops.aten.var_mean.correction(cat_1, [2], correction = 0, keepdim = True)
        getitem: "f32[500, 129, 1]" = var_mean[0]
        getitem_1: "f32[500, 129, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[500, 129, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[500, 129, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[500, 129, 128]" = torch.ops.aten.sub.Tensor(cat_1, getitem_1)
        mul: "f32[500, 129, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[500, 129, 128]" = torch.ops.aten.mul.Tensor(mul, primals_7);  mul = None
        add_2: "f32[500, 129, 128]" = torch.ops.aten.add.Tensor(mul_1, primals_8);  mul_1 = primals_8 = None
        
        # File: /app/src/models/classifier.py:46 in forward, code: x, _ = self.mha(x_cls, x, x, key_padding_mask=padding_mask)
        permute_2: "f32[129, 500, 128]" = torch.ops.aten.permute.default(add_2, [1, 0, 2]);  add_2 = None
        split_with_sizes = torch.ops.aten.split_with_sizes.default(primals_9, [128, 256]);  primals_9 = None
        getitem_2: "f32[128, 128]" = split_with_sizes[0]
        getitem_3: "f32[256, 128]" = split_with_sizes[1];  split_with_sizes = None
        split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(primals_10, [128, 256]);  primals_10 = None
        getitem_4: "f32[128]" = split_with_sizes_1[0]
        getitem_5: "f32[256]" = split_with_sizes_1[1];  split_with_sizes_1 = None
        permute_4: "f32[128, 128]" = torch.ops.aten.permute.default(getitem_2, [1, 0]);  getitem_2 = None
        permute_5: "f32[1, 500, 128]" = torch.ops.aten.permute.default(add, [1, 0, 2])
        view_3: "f32[500, 128]" = torch.ops.aten.view.default(permute_5, [500, 128]);  permute_5 = None
        addmm_1: "f32[500, 128]" = torch.ops.aten.addmm.default(getitem_4, view_3, permute_4);  getitem_4 = None
        view_4: "f32[1, 500, 128]" = torch.ops.aten.view.default(addmm_1, [1, 500, 128]);  addmm_1 = None
        permute_6: "f32[128, 256]" = torch.ops.aten.permute.default(getitem_3, [1, 0]);  getitem_3 = None
        clone_1: "f32[129, 500, 128]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_5: "f32[64500, 128]" = torch.ops.aten.view.default(clone_1, [64500, 128]);  clone_1 = None
        mm: "f32[64500, 256]" = torch.ops.aten.mm.default(view_5, permute_6)
        view_6: "f32[129, 500, 256]" = torch.ops.aten.view.default(mm, [129, 500, 256]);  mm = None
        add_3: "f32[129, 500, 256]" = torch.ops.aten.add.Tensor(view_6, getitem_5);  view_6 = getitem_5 = None
        view_7: "f32[129, 500, 2, 128]" = torch.ops.aten.view.default(add_3, [129, 500, 2, 128]);  add_3 = None
        unsqueeze: "f32[1, 129, 500, 2, 128]" = torch.ops.aten.unsqueeze.default(view_7, 0);  view_7 = None
        permute_7: "f32[2, 129, 500, 1, 128]" = torch.ops.aten.permute.default(unsqueeze, [3, 1, 2, 0, 4]);  unsqueeze = None
        squeeze: "f32[2, 129, 500, 128]" = torch.ops.aten.squeeze.dim(permute_7, -2);  permute_7 = None
        clone_2: "f32[2, 129, 500, 128]" = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
        select: "f32[129, 500, 128]" = torch.ops.aten.select.int(clone_2, 0, 0)
        select_1: "f32[129, 500, 128]" = torch.ops.aten.select.int(clone_2, 0, 1);  clone_2 = None
        view_8: "f32[1, 4000, 16]" = torch.ops.aten.view.default(view_4, [1, 4000, 16]);  view_4 = None
        permute_8: "f32[4000, 1, 16]" = torch.ops.aten.permute.default(view_8, [1, 0, 2]);  view_8 = None
        view_9: "f32[129, 4000, 16]" = torch.ops.aten.view.default(select, [129, 4000, 16]);  select = None
        permute_9: "f32[4000, 129, 16]" = torch.ops.aten.permute.default(view_9, [1, 0, 2]);  view_9 = None
        view_10: "f32[129, 4000, 16]" = torch.ops.aten.view.default(select_1, [129, 4000, 16]);  select_1 = None
        permute_10: "f32[4000, 129, 16]" = torch.ops.aten.permute.default(view_10, [1, 0, 2]);  view_10 = None
        view_11: "f32[500, 1, 1, 129]" = torch.ops.aten.view.default(cat, [500, 1, 1, 129]);  cat = None
        expand: "f32[500, 8, 1, 129]" = torch.ops.aten.expand.default(view_11, [-1, 8, -1, -1]);  view_11 = None
        clone_3: "f32[500, 8, 1, 129]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_12: "f32[4000, 1, 129]" = torch.ops.aten.view.default(clone_3, [4000, 1, 129]);  clone_3 = None
        mul_2: "f32[4000, 1, 16]" = torch.ops.aten.mul.Tensor(permute_8, 0.25);  permute_8 = None
        permute_11: "f32[4000, 16, 129]" = torch.ops.aten.permute.default(permute_9, [0, 2, 1]);  permute_9 = None
        baddbmm: "f32[4000, 1, 129]" = torch.ops.aten.baddbmm.default(view_12, mul_2, permute_11);  view_12 = None
        amax: "f32[4000, 1, 1]" = torch.ops.aten.amax.default(baddbmm, [-1], True)
        sub_1: "f32[4000, 1, 129]" = torch.ops.aten.sub.Tensor(baddbmm, amax);  baddbmm = amax = None
        exp: "f32[4000, 1, 129]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[4000, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[4000, 1, 129]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        bmm: "f32[4000, 1, 16]" = torch.ops.aten.bmm.default(div, permute_10)
        permute_12: "f32[1, 4000, 16]" = torch.ops.aten.permute.default(bmm, [1, 0, 2]);  bmm = None
        view_13: "f32[500, 128]" = torch.ops.aten.view.default(permute_12, [500, 128]);  permute_12 = None
        permute_13: "f32[128, 128]" = torch.ops.aten.permute.default(primals_11, [1, 0])
        addmm_2: "f32[500, 128]" = torch.ops.aten.addmm.default(primals_12, view_13, permute_13);  primals_12 = permute_13 = None
        view_14: "f32[1, 500, 128]" = torch.ops.aten.view.default(addmm_2, [1, 500, 128])
        permute_14: "f32[500, 1, 128]" = torch.ops.aten.permute.default(view_14, [1, 0, 2]);  view_14 = None
        
        # File: /app/src/models/classifier.py:47 in forward, code: x = self.layernorm2(x)
        var_mean_1 = torch.ops.aten.var_mean.correction(permute_14, [2], correction = 0, keepdim = True)
        getitem_6: "f32[500, 1, 1]" = var_mean_1[0]
        getitem_7: "f32[500, 1, 1]" = var_mean_1[1];  var_mean_1 = None
        add_4: "f32[500, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_1: "f32[500, 1, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_2: "f32[500, 1, 128]" = torch.ops.aten.sub.Tensor(permute_14, getitem_7);  permute_14 = None
        mul_3: "f32[500, 1, 128]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_4: "f32[500, 1, 128]" = torch.ops.aten.mul.Tensor(mul_3, primals_13);  mul_3 = None
        add_5: "f32[500, 1, 128]" = torch.ops.aten.add.Tensor(mul_4, primals_14);  mul_4 = primals_14 = None
        
        # File: /app/src/models/classifier.py:50 in forward, code: x += residual
        add_6: "f32[500, 1, 128]" = torch.ops.aten.add.Tensor(add_5, add);  add_5 = add = None
        
        # File: /app/src/models/feedforward.py:26 in forward, code: x = self.layernorm1(x)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
        getitem_8: "f32[500, 1, 1]" = var_mean_2[0]
        getitem_9: "f32[500, 1, 1]" = var_mean_2[1];  var_mean_2 = None
        add_7: "f32[500, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_2: "f32[500, 1, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        sub_3: "f32[500, 1, 128]" = torch.ops.aten.sub.Tensor(add_6, getitem_9);  getitem_9 = None
        mul_5: "f32[500, 1, 128]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_6: "f32[500, 1, 128]" = torch.ops.aten.mul.Tensor(mul_5, primals_15)
        add_8: "f32[500, 1, 128]" = torch.ops.aten.add.Tensor(mul_6, primals_16);  mul_6 = primals_16 = None
        
        # File: /app/src/models/feedforward.py:27 in forward, code: x = self.linear1(x)
        view_16: "f32[500, 128]" = torch.ops.aten.view.default(add_8, [500, 128]);  add_8 = None
        permute_15: "f32[128, 512]" = torch.ops.aten.permute.default(primals_17, [1, 0])
        addmm_3: "f32[500, 512]" = torch.ops.aten.addmm.default(primals_18, view_16, permute_15);  primals_18 = permute_15 = None
        view_17: "f32[500, 1, 512]" = torch.ops.aten.view.default(addmm_3, [500, 1, 512])
        
        # File: /app/src/models/feedforward.py:28 in forward, code: x = self.act(x)
        mul_7: "f32[500, 1, 512]" = torch.ops.aten.mul.Tensor(view_17, 0.5)
        mul_8: "f32[500, 1, 512]" = torch.ops.aten.mul.Tensor(view_17, 0.7071067811865476);  view_17 = None
        erf: "f32[500, 1, 512]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add_9: "f32[500, 1, 512]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_9: "f32[500, 1, 512]" = torch.ops.aten.mul.Tensor(mul_7, add_9);  mul_7 = add_9 = None
        
        # File: /app/src/models/feedforward.py:31 in forward, code: x = self.layernorm2(x)
        var_mean_3 = torch.ops.aten.var_mean.correction(mul_9, [2], correction = 0, keepdim = True)
        getitem_10: "f32[500, 1, 1]" = var_mean_3[0]
        getitem_11: "f32[500, 1, 1]" = var_mean_3[1];  var_mean_3 = None
        add_10: "f32[500, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_3: "f32[500, 1, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_4: "f32[500, 1, 512]" = torch.ops.aten.sub.Tensor(mul_9, getitem_11);  mul_9 = None
        mul_10: "f32[500, 1, 512]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = None
        mul_11: "f32[500, 1, 512]" = torch.ops.aten.mul.Tensor(mul_10, primals_19);  mul_10 = None
        add_11: "f32[500, 1, 512]" = torch.ops.aten.add.Tensor(mul_11, primals_20);  mul_11 = primals_20 = None
        
        # File: /app/src/models/feedforward.py:32 in forward, code: x = self.linear2(x)
        view_18: "f32[500, 512]" = torch.ops.aten.view.default(add_11, [500, 512]);  add_11 = None
        permute_16: "f32[512, 128]" = torch.ops.aten.permute.default(primals_21, [1, 0])
        addmm_4: "f32[500, 128]" = torch.ops.aten.addmm.default(primals_22, view_18, permute_16);  primals_22 = permute_16 = None
        view_19: "f32[500, 1, 128]" = torch.ops.aten.view.default(addmm_4, [500, 1, 128]);  addmm_4 = None
        
        # File: /app/src/models/feedforward.py:34 in forward, code: x += residual
        add_12: "f32[500, 1, 128]" = torch.ops.aten.add.Tensor(view_19, add_6);  view_19 = add_6 = None
        
        # File: /app/src/models/lorentz_part.py:286 in torch_dynamo_resume_in_forward_at_275, code: x_cls = self.layernorm(x_cls).squeeze(1)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
        getitem_12: "f32[500, 1, 1]" = var_mean_4[0]
        getitem_13: "f32[500, 1, 1]" = var_mean_4[1];  var_mean_4 = None
        add_13: "f32[500, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_4: "f32[500, 1, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_5: "f32[500, 1, 128]" = torch.ops.aten.sub.Tensor(add_12, getitem_13);  add_12 = getitem_13 = None
        mul_12: "f32[500, 1, 128]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_4);  sub_5 = None
        mul_13: "f32[500, 1, 128]" = torch.ops.aten.mul.Tensor(mul_12, primals_23)
        add_14: "f32[500, 1, 128]" = torch.ops.aten.add.Tensor(mul_13, primals_24);  mul_13 = primals_24 = None
        squeeze_1: "f32[500, 128]" = torch.ops.aten.squeeze.dim(add_14, 1);  add_14 = None
        
        # File: /app/src/models/classifier.py:76 in forward, code: return self.layers(x)
        permute_17: "f32[128, 10]" = torch.ops.aten.permute.default(primals_25, [1, 0])
        addmm_5: "f32[500, 10]" = torch.ops.aten.addmm.default(primals_26, squeeze_1, permute_17);  primals_26 = permute_17 = None
        
        # File: /app/src/models/lorentz_part.py:286 in torch_dynamo_resume_in_forward_at_275, code: x_cls = self.layernorm(x_cls).squeeze(1)
        div_1: "f32[500, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 128);  rsqrt_4 = None
        
        # File: /app/src/models/feedforward.py:26 in forward, code: x = self.layernorm1(x)
        div_3: "f32[500, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 128);  rsqrt_2 = None
        
        # File: /app/src/models/classifier.py:46 in forward, code: x, _ = self.mha(x_cls, x, x, key_padding_mask=padding_mask)
        permute_37: "f32[4000, 16, 129]" = torch.ops.aten.permute.default(permute_10, [0, 2, 1]);  permute_10 = None
        permute_38: "f32[4000, 129, 16]" = torch.ops.aten.permute.default(permute_11, [0, 2, 1]);  permute_11 = None
        permute_39: "f32[4000, 16, 1]" = torch.ops.aten.permute.default(mul_2, [0, 2, 1]);  mul_2 = None
        permute_47: "f32[256, 128]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
        permute_49: "f32[128, 128]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        return (addmm_5, primals_2, primals_7, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, view, cat_1, getitem_1, rsqrt, view_3, view_5, div, view_13, addmm_2, getitem_7, rsqrt_1, mul_5, view_16, addmm_3, getitem_11, rsqrt_3, view_18, mul_12, squeeze_1, div_1, div_3, permute_37, permute_38, permute_39, permute_47, permute_49)
        