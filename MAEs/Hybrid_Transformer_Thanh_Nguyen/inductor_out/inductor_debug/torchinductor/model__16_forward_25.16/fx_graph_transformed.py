class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[1, 1, 128]", primals_2: "f32[500, 128]", primals_3: "f32[500, 128, 128]", primals_4: "f32[128]", primals_5: "f32[128]", primals_6: "f32[384, 128]", primals_7: "f32[384]", primals_8: "f32[128, 128]", primals_9: "f32[128]", primals_10: "f32[128]", primals_11: "f32[128]", primals_12: "f32[128]", primals_13: "f32[128]", primals_14: "f32[512, 128]", primals_15: "f32[512]", primals_16: "f32[512]", primals_17: "f32[512]"):
        # File: /app/src/models/lorentz_part.py:279 in torch_dynamo_resume_in_forward_at_275, code: x_cls = self.cls_token.expand(B, -1, -1)
        expand: "f32[500, 1, 128]" = torch.ops.aten.expand.default(primals_1, [500, -1, -1])
        
        # File: /app/src/models/classifier.py:41 in forward, code: padding_mask = torch.cat((torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1)
        full_default: "f32[500, 1]" = torch.ops.aten.full.default([500, 1], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat: "f32[500, 129]" = torch.ops.aten.cat.default([full_default, primals_2], 1);  full_default = primals_2 = None
        
        # File: /app/src/models/classifier.py:44 in forward, code: x = torch.cat((x_cls, x), dim=1)  # (B, N + 1, D)
        cat_1: "f32[500, 129, 128]" = torch.ops.aten.cat.default([expand, primals_3], 1);  primals_3 = None
        
        # File: /app/src/models/classifier.py:45 in forward, code: x = self.layernorm1(x)
        var_mean = torch.ops.aten.var_mean.correction(cat_1, [2], correction = 0, keepdim = True)
        getitem: "f32[500, 129, 1]" = var_mean[0]
        getitem_1: "f32[500, 129, 1]" = var_mean[1];  var_mean = None
        add: "f32[500, 129, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[500, 129, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[500, 129, 128]" = torch.ops.aten.sub.Tensor(cat_1, getitem_1)
        mul: "f32[500, 129, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[500, 129, 128]" = torch.ops.aten.mul.Tensor(mul, primals_4);  mul = None
        add_1: "f32[500, 129, 128]" = torch.ops.aten.add.Tensor(mul_1, primals_5);  mul_1 = primals_5 = None
        
        # File: /app/src/models/classifier.py:46 in forward, code: x, _ = self.mha(x_cls, x, x, key_padding_mask=padding_mask)
        permute: "f32[1, 500, 128]" = torch.ops.aten.permute.default(expand, [1, 0, 2])
        permute_1: "f32[129, 500, 128]" = torch.ops.aten.permute.default(add_1, [1, 0, 2]);  add_1 = None
        split_with_sizes = torch.ops.aten.split_with_sizes.default(primals_6, [128, 256]);  primals_6 = None
        getitem_2: "f32[128, 128]" = split_with_sizes[0]
        getitem_3: "f32[256, 128]" = split_with_sizes[1];  split_with_sizes = None
        split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(primals_7, [128, 256]);  primals_7 = None
        getitem_4: "f32[128]" = split_with_sizes_1[0]
        getitem_5: "f32[256]" = split_with_sizes_1[1];  split_with_sizes_1 = None
        permute_2: "f32[128, 128]" = torch.ops.aten.permute.default(getitem_2, [1, 0]);  getitem_2 = None
        view: "f32[500, 128]" = torch.ops.aten.reshape.default(permute, [500, 128]);  permute = None
        mm: "f32[500, 128]" = torch.ops.aten.mm.default(view, permute_2);  view = None
        view_1: "f32[1, 500, 128]" = torch.ops.aten.reshape.default(mm, [1, 500, 128]);  mm = None
        add_2: "f32[1, 500, 128]" = torch.ops.aten.add.Tensor(view_1, getitem_4);  view_1 = getitem_4 = None
        permute_3: "f32[128, 256]" = torch.ops.aten.permute.default(getitem_3, [1, 0]);  getitem_3 = None
        clone: "f32[129, 500, 128]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_2: "f32[64500, 128]" = torch.ops.aten.reshape.default(clone, [64500, 128]);  clone = None
        mm_1: "f32[64500, 256]" = torch.ops.aten.mm.default(view_2, permute_3)
        view_3: "f32[129, 500, 256]" = torch.ops.aten.reshape.default(mm_1, [129, 500, 256]);  mm_1 = None
        add_3: "f32[129, 500, 256]" = torch.ops.aten.add.Tensor(view_3, getitem_5);  view_3 = getitem_5 = None
        view_4: "f32[129, 500, 2, 128]" = torch.ops.aten.reshape.default(add_3, [129, 500, 2, 128]);  add_3 = None
        unsqueeze: "f32[1, 129, 500, 2, 128]" = torch.ops.aten.unsqueeze.default(view_4, 0);  view_4 = None
        permute_4: "f32[2, 129, 500, 1, 128]" = torch.ops.aten.permute.default(unsqueeze, [3, 1, 2, 0, 4]);  unsqueeze = None
        squeeze: "f32[2, 129, 500, 128]" = torch.ops.aten.squeeze.dim(permute_4, -2);  permute_4 = None
        clone_1: "f32[2, 129, 500, 128]" = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
        select: "f32[129, 500, 128]" = torch.ops.aten.select.int(clone_1, 0, 0)
        select_1: "f32[129, 500, 128]" = torch.ops.aten.select.int(clone_1, 0, 1);  clone_1 = None
        view_5: "f32[1, 4000, 16]" = torch.ops.aten.reshape.default(add_2, [1, 4000, 16]);  add_2 = None
        permute_5: "f32[4000, 1, 16]" = torch.ops.aten.permute.default(view_5, [1, 0, 2]);  view_5 = None
        view_6: "f32[129, 4000, 16]" = torch.ops.aten.reshape.default(select, [129, 4000, 16]);  select = None
        permute_6: "f32[4000, 129, 16]" = torch.ops.aten.permute.default(view_6, [1, 0, 2]);  view_6 = None
        view_7: "f32[129, 4000, 16]" = torch.ops.aten.reshape.default(select_1, [129, 4000, 16]);  select_1 = None
        permute_7: "f32[4000, 129, 16]" = torch.ops.aten.permute.default(view_7, [1, 0, 2]);  view_7 = None
        view_8: "f32[500, 1, 1, 129]" = torch.ops.aten.reshape.default(cat, [500, 1, 1, 129]);  cat = None
        expand_1: "f32[500, 8, 1, 129]" = torch.ops.aten.expand.default(view_8, [-1, 8, -1, -1]);  view_8 = None
        clone_2: "f32[500, 8, 1, 129]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_9: "f32[4000, 1, 129]" = torch.ops.aten.reshape.default(clone_2, [4000, 1, 129]);  clone_2 = None
        mul_2: "f32[4000, 1, 16]" = torch.ops.aten.mul.Tensor(permute_5, 0.25);  permute_5 = None
        permute_8: "f32[4000, 16, 129]" = torch.ops.aten.permute.default(permute_6, [0, 2, 1]);  permute_6 = None
        baddbmm: "f32[4000, 1, 129]" = torch.ops.aten.baddbmm.default(view_9, mul_2, permute_8);  view_9 = None
        
        # File: /app/src/models/lorentz_part.py:272 in forward, code: x, U = self.processor(x)
        prepare_softmax_online_default = torch.ops.prims.prepare_softmax_online.default(baddbmm, -1)
        getitem_12: "f32[4000, 1, 1]" = prepare_softmax_online_default[0]
        getitem_13: "f32[4000, 1, 1]" = prepare_softmax_online_default[1];  prepare_softmax_online_default = None
        sub_tensor: "f32[4000, 1, 129]" = torch.ops.aten.sub.Tensor(baddbmm, getitem_12);  baddbmm = getitem_12 = None
        exp_default: "f32[4000, 1, 129]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        
        # File: /app/src/models/classifier.py:46 in forward, code: x, _ = self.mha(x_cls, x, x, key_padding_mask=padding_mask)
        div: "f32[4000, 1, 129]" = torch.ops.aten.div.Tensor(exp_default, getitem_13);  exp_default = getitem_13 = None
        bmm: "f32[4000, 1, 16]" = torch.ops.aten.bmm.default(div, permute_7)
        permute_9: "f32[1, 4000, 16]" = torch.ops.aten.permute.default(bmm, [1, 0, 2]);  bmm = None
        view_10: "f32[500, 128]" = torch.ops.aten.reshape.default(permute_9, [500, 128]);  permute_9 = None
        permute_10: "f32[128, 128]" = torch.ops.aten.permute.default(primals_8, [1, 0])
        addmm: "f32[500, 128]" = torch.ops.aten.addmm.default(primals_9, view_10, permute_10);  primals_9 = permute_10 = None
        view_11: "f32[1, 500, 128]" = torch.ops.aten.reshape.default(addmm, [1, 500, 128])
        permute_11: "f32[500, 1, 128]" = torch.ops.aten.permute.default(view_11, [1, 0, 2]);  view_11 = None
        
        # File: /app/src/models/classifier.py:47 in forward, code: x = self.layernorm2(x)
        var_mean_1 = torch.ops.aten.var_mean.correction(permute_11, [2], correction = 0, keepdim = True)
        getitem_6: "f32[500, 1, 1]" = var_mean_1[0]
        getitem_7: "f32[500, 1, 1]" = var_mean_1[1];  var_mean_1 = None
        add_4: "f32[500, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_1: "f32[500, 1, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_2: "f32[500, 1, 128]" = torch.ops.aten.sub.Tensor(permute_11, getitem_7);  permute_11 = None
        mul_3: "f32[500, 1, 128]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_4: "f32[500, 1, 128]" = torch.ops.aten.mul.Tensor(mul_3, primals_10);  mul_3 = None
        add_5: "f32[500, 1, 128]" = torch.ops.aten.add.Tensor(mul_4, primals_11);  mul_4 = primals_11 = None
        
        # File: /app/src/models/classifier.py:50 in forward, code: x += residual
        add_6: "f32[500, 1, 128]" = torch.ops.aten.add.Tensor(add_5, expand);  add_5 = expand = None
        
        # File: /app/src/models/feedforward.py:26 in forward, code: x = self.layernorm1(x)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
        getitem_8: "f32[500, 1, 1]" = var_mean_2[0]
        getitem_9: "f32[500, 1, 1]" = var_mean_2[1];  var_mean_2 = None
        add_7: "f32[500, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_2: "f32[500, 1, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        sub_3: "f32[500, 1, 128]" = torch.ops.aten.sub.Tensor(add_6, getitem_9)
        mul_5: "f32[500, 1, 128]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_6: "f32[500, 1, 128]" = torch.ops.aten.mul.Tensor(mul_5, primals_12);  mul_5 = None
        add_8: "f32[500, 1, 128]" = torch.ops.aten.add.Tensor(mul_6, primals_13);  mul_6 = primals_13 = None
        
        # File: /app/src/models/feedforward.py:27 in forward, code: x = self.linear1(x)
        view_13: "f32[500, 128]" = torch.ops.aten.reshape.default(add_8, [500, 128]);  add_8 = None
        permute_12: "f32[128, 512]" = torch.ops.aten.permute.default(primals_14, [1, 0])
        addmm_1: "f32[500, 512]" = torch.ops.aten.addmm.default(primals_15, view_13, permute_12);  primals_15 = permute_12 = None
        view_14: "f32[500, 1, 512]" = torch.ops.aten.reshape.default(addmm_1, [500, 1, 512])
        
        # File: /app/src/models/feedforward.py:28 in forward, code: x = self.act(x)
        mul_7: "f32[500, 1, 512]" = torch.ops.aten.mul.Tensor(view_14, 0.5)
        mul_8: "f32[500, 1, 512]" = torch.ops.aten.mul.Tensor(view_14, 0.7071067811865476);  view_14 = None
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
        mul_11: "f32[500, 1, 512]" = torch.ops.aten.mul.Tensor(mul_10, primals_16);  mul_10 = None
        add_11: "f32[500, 1, 512]" = torch.ops.aten.add.Tensor(mul_11, primals_17);  mul_11 = primals_17 = None
        
        # File: /app/src/models/classifier.py:46 in forward, code: x, _ = self.mha(x_cls, x, x, key_padding_mask=padding_mask)
        permute_24: "f32[4000, 16, 129]" = torch.ops.aten.permute.default(permute_7, [0, 2, 1]);  permute_7 = None
        permute_25: "f32[4000, 129, 16]" = torch.ops.aten.permute.default(permute_8, [0, 2, 1]);  permute_8 = None
        permute_26: "f32[4000, 16, 1]" = torch.ops.aten.permute.default(mul_2, [0, 2, 1]);  mul_2 = None
        permute_34: "f32[256, 128]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        permute_38: "f32[128, 128]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        return (add_11, add_6, primals_1, primals_4, primals_8, primals_10, primals_12, primals_14, primals_16, cat_1, getitem_1, rsqrt, view_2, div, view_10, addmm, getitem_7, rsqrt_1, add_6, getitem_9, rsqrt_2, view_13, addmm_1, getitem_11, rsqrt_3, permute_24, permute_25, permute_26, permute_34, permute_38)
        