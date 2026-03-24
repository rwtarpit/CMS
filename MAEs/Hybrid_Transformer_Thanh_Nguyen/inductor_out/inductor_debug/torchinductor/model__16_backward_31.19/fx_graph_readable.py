class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[1, 1, 128]", primals_4: "f32[128]", primals_8: "f32[128, 128]", primals_10: "f32[128]", primals_12: "f32[128]", primals_14: "f32[512, 128]", primals_16: "f32[512]", cat_1: "f32[500, 129, 128]", getitem_1: "f32[500, 129, 1]", rsqrt: "f32[500, 129, 1]", view_2: "f32[64500, 128]", div: "f32[4000, 1, 129]", view_10: "f32[500, 128]", addmm: "f32[500, 128]", getitem_7: "f32[500, 1, 1]", rsqrt_1: "f32[500, 1, 1]", add_6: "f32[500, 1, 128]", getitem_9: "f32[500, 1, 1]", rsqrt_2: "f32[500, 1, 1]", view_13: "f32[500, 128]", addmm_1: "f32[500, 512]", getitem_11: "f32[500, 1, 1]", rsqrt_3: "f32[500, 1, 1]", permute_24: "f32[4000, 16, 129]", permute_25: "f32[4000, 129, 16]", permute_26: "f32[4000, 16, 1]", permute_34: "f32[256, 128]", permute_38: "f32[128, 128]", tangents_1: "f32[500, 1, 512]", tangents_2: "f32[500, 1, 128]"):
        # File: /app/src/models/feedforward.py:31 in forward, code: x = self.layernorm2(x)
        mul_13: "f32[500, 1, 512]" = torch.ops.aten.mul.Tensor(tangents_1, primals_16);  primals_16 = None
        mul_14: "f32[500, 1, 512]" = torch.ops.aten.mul.Tensor(mul_13, 512)
        sum_2: "f32[500, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_13, [2], True)
        
        # File: /app/src/models/feedforward.py:27 in forward, code: x = self.linear1(x)
        view_14: "f32[500, 1, 512]" = torch.ops.aten.view.default(addmm_1, [500, 1, 512]);  addmm_1 = None
        
        # File: /app/src/models/feedforward.py:28 in forward, code: x = self.act(x)
        mul_7: "f32[500, 1, 512]" = torch.ops.aten.mul.Tensor(view_14, 0.5)
        mul_8: "f32[500, 1, 512]" = torch.ops.aten.mul.Tensor(view_14, 0.7071067811865476)
        erf: "f32[500, 1, 512]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add_9: "f32[500, 1, 512]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_9: "f32[500, 1, 512]" = torch.ops.aten.mul.Tensor(mul_7, add_9);  mul_7 = None
        
        # File: /app/src/models/feedforward.py:31 in forward, code: x = self.layernorm2(x)
        sub_4: "f32[500, 1, 512]" = torch.ops.aten.sub.Tensor(mul_9, getitem_11);  mul_9 = getitem_11 = None
        mul_10: "f32[500, 1, 512]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = None
        mul_15: "f32[500, 1, 512]" = torch.ops.aten.mul.Tensor(mul_13, mul_10);  mul_13 = None
        sum_3: "f32[500, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_15, [2], True);  mul_15 = None
        mul_16: "f32[500, 1, 512]" = torch.ops.aten.mul.Tensor(mul_10, sum_3);  sum_3 = None
        sub_6: "f32[500, 1, 512]" = torch.ops.aten.sub.Tensor(mul_14, sum_2);  mul_14 = sum_2 = None
        sub_7: "f32[500, 1, 512]" = torch.ops.aten.sub.Tensor(sub_6, mul_16);  sub_6 = mul_16 = None
        div_1: "f32[500, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 512);  rsqrt_3 = None
        mul_17: "f32[500, 1, 512]" = torch.ops.aten.mul.Tensor(div_1, sub_7);  div_1 = sub_7 = None
        mul_18: "f32[500, 1, 512]" = torch.ops.aten.mul.Tensor(tangents_1, mul_10);  mul_10 = None
        sum_4: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_18, [0, 1]);  mul_18 = None
        sum_5: "f32[512]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0, 1]);  tangents_1 = None
        
        # File: /app/src/models/feedforward.py:28 in forward, code: x = self.act(x)
        mul_20: "f32[500, 1, 512]" = torch.ops.aten.mul.Tensor(add_9, 0.5);  add_9 = None
        mul_21: "f32[500, 1, 512]" = torch.ops.aten.mul.Tensor(view_14, view_14)
        mul_22: "f32[500, 1, 512]" = torch.ops.aten.mul.Tensor(mul_21, -0.5);  mul_21 = None
        exp_1: "f32[500, 1, 512]" = torch.ops.aten.exp.default(mul_22);  mul_22 = None
        mul_23: "f32[500, 1, 512]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
        mul_24: "f32[500, 1, 512]" = torch.ops.aten.mul.Tensor(view_14, mul_23);  view_14 = mul_23 = None
        add_13: "f32[500, 1, 512]" = torch.ops.aten.add.Tensor(mul_20, mul_24);  mul_20 = mul_24 = None
        mul_25: "f32[500, 1, 512]" = torch.ops.aten.mul.Tensor(mul_17, add_13);  mul_17 = add_13 = None
        
        # File: /app/src/models/feedforward.py:27 in forward, code: x = self.linear1(x)
        view_15: "f32[500, 512]" = torch.ops.aten.view.default(mul_25, [500, 512]);  mul_25 = None
        permute_12: "f32[128, 512]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
        permute_13: "f32[512, 128]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        mm_2: "f32[500, 128]" = torch.ops.aten.mm.default(view_15, permute_13);  permute_13 = None
        permute_14: "f32[512, 500]" = torch.ops.aten.permute.default(view_15, [1, 0])
        mm_3: "f32[512, 128]" = torch.ops.aten.mm.default(permute_14, view_13);  permute_14 = view_13 = None
        sum_6: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_15, [0], True);  view_15 = None
        view_16: "f32[512]" = torch.ops.aten.view.default(sum_6, [512]);  sum_6 = None
        view_17: "f32[500, 1, 128]" = torch.ops.aten.view.default(mm_2, [500, 1, 128]);  mm_2 = None
        
        # File: /app/src/models/feedforward.py:26 in forward, code: x = self.layernorm1(x)
        mul_27: "f32[500, 1, 128]" = torch.ops.aten.mul.Tensor(view_17, primals_12);  primals_12 = None
        mul_28: "f32[500, 1, 128]" = torch.ops.aten.mul.Tensor(mul_27, 128)
        sum_7: "f32[500, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_27, [2], True)
        sub_3: "f32[500, 1, 128]" = torch.ops.aten.sub.Tensor(add_6, getitem_9);  add_6 = getitem_9 = None
        mul_5: "f32[500, 1, 128]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_29: "f32[500, 1, 128]" = torch.ops.aten.mul.Tensor(mul_27, mul_5);  mul_27 = None
        sum_8: "f32[500, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_29, [2], True);  mul_29 = None
        mul_30: "f32[500, 1, 128]" = torch.ops.aten.mul.Tensor(mul_5, sum_8);  sum_8 = None
        sub_9: "f32[500, 1, 128]" = torch.ops.aten.sub.Tensor(mul_28, sum_7);  mul_28 = sum_7 = None
        sub_10: "f32[500, 1, 128]" = torch.ops.aten.sub.Tensor(sub_9, mul_30);  sub_9 = mul_30 = None
        div_2: "f32[500, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 128);  rsqrt_2 = None
        mul_31: "f32[500, 1, 128]" = torch.ops.aten.mul.Tensor(div_2, sub_10);  div_2 = sub_10 = None
        mul_32: "f32[500, 1, 128]" = torch.ops.aten.mul.Tensor(view_17, mul_5);  mul_5 = None
        sum_9: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_32, [0, 1]);  mul_32 = None
        sum_10: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_17, [0, 1]);  view_17 = None
        add_14: "f32[500, 1, 128]" = torch.ops.aten.add.Tensor(tangents_2, mul_31);  tangents_2 = mul_31 = None
        
        # File: /app/src/models/classifier.py:47 in forward, code: x = self.layernorm2(x)
        mul_34: "f32[500, 1, 128]" = torch.ops.aten.mul.Tensor(add_14, primals_10);  primals_10 = None
        mul_35: "f32[500, 1, 128]" = torch.ops.aten.mul.Tensor(mul_34, 128)
        sum_11: "f32[500, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_34, [2], True)
        
        # File: /app/src/models/classifier.py:46 in forward, code: x, _ = self.mha(x_cls, x, x, key_padding_mask=padding_mask)
        view_11: "f32[1, 500, 128]" = torch.ops.aten.view.default(addmm, [1, 500, 128]);  addmm = None
        permute_11: "f32[500, 1, 128]" = torch.ops.aten.permute.default(view_11, [1, 0, 2]);  view_11 = None
        
        # File: /app/src/models/classifier.py:47 in forward, code: x = self.layernorm2(x)
        sub_2: "f32[500, 1, 128]" = torch.ops.aten.sub.Tensor(permute_11, getitem_7);  permute_11 = getitem_7 = None
        mul_3: "f32[500, 1, 128]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_36: "f32[500, 1, 128]" = torch.ops.aten.mul.Tensor(mul_34, mul_3);  mul_34 = None
        sum_12: "f32[500, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_36, [2], True);  mul_36 = None
        mul_37: "f32[500, 1, 128]" = torch.ops.aten.mul.Tensor(mul_3, sum_12);  sum_12 = None
        sub_12: "f32[500, 1, 128]" = torch.ops.aten.sub.Tensor(mul_35, sum_11);  mul_35 = sum_11 = None
        sub_13: "f32[500, 1, 128]" = torch.ops.aten.sub.Tensor(sub_12, mul_37);  sub_12 = mul_37 = None
        div_3: "f32[500, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 128);  rsqrt_1 = None
        mul_38: "f32[500, 1, 128]" = torch.ops.aten.mul.Tensor(div_3, sub_13);  div_3 = sub_13 = None
        mul_39: "f32[500, 1, 128]" = torch.ops.aten.mul.Tensor(add_14, mul_3);  mul_3 = None
        sum_13: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_39, [0, 1]);  mul_39 = None
        sum_14: "f32[128]" = torch.ops.aten.sum.dim_IntList(add_14, [0, 1])
        
        # File: /app/src/models/classifier.py:46 in forward, code: x, _ = self.mha(x_cls, x, x, key_padding_mask=padding_mask)
        permute_17: "f32[1, 500, 128]" = torch.ops.aten.permute.default(mul_38, [1, 0, 2]);  mul_38 = None
        view_18: "f32[500, 128]" = torch.ops.aten.view.default(permute_17, [500, 128]);  permute_17 = None
        permute_10: "f32[128, 128]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_18: "f32[128, 128]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm_4: "f32[500, 128]" = torch.ops.aten.mm.default(view_18, permute_18);  permute_18 = None
        permute_19: "f32[128, 500]" = torch.ops.aten.permute.default(view_18, [1, 0])
        mm_5: "f32[128, 128]" = torch.ops.aten.mm.default(permute_19, view_10);  permute_19 = view_10 = None
        sum_15: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_18, [0], True);  view_18 = None
        view_19: "f32[128]" = torch.ops.aten.view.default(sum_15, [128]);  sum_15 = None
        view_20: "f32[1, 4000, 16]" = torch.ops.aten.view.default(mm_4, [1, 4000, 16]);  mm_4 = None
        permute_22: "f32[4000, 1, 16]" = torch.ops.aten.permute.default(view_20, [1, 0, 2]);  view_20 = None
        permute_23: "f32[4000, 129, 1]" = torch.ops.aten.permute.default(div, [0, 2, 1])
        bmm_1: "f32[4000, 129, 16]" = torch.ops.aten.bmm.default(permute_23, permute_22);  permute_23 = None
        bmm_2: "f32[4000, 1, 129]" = torch.ops.aten.bmm.default(permute_22, permute_24);  permute_22 = permute_24 = None
        mul_40: "f32[4000, 1, 129]" = torch.ops.aten.mul.Tensor(bmm_2, div);  bmm_2 = None
        sum_16: "f32[4000, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_40, [-1], True)
        neg: "f32[4000, 1, 129]" = torch.ops.aten.neg.default(div);  div = None
        fma: "f32[4000, 1, 129]" = torch.ops.prims.fma.default(neg, sum_16, mul_40);  neg = sum_16 = mul_40 = None
        bmm_3: "f32[4000, 1, 16]" = torch.ops.aten.bmm.default(fma, permute_25);  permute_25 = None
        constant_pad_nd_default: "f32[4000, 1, 132]" = torch.ops.aten.constant_pad_nd.default(fma, [0, 3, 0, 0, 0, 0]);  fma = None
        bmm_default: "f32[4000, 16, 132]" = torch.ops.aten.bmm.default(permute_26, constant_pad_nd_default);  permute_26 = constant_pad_nd_default = None
        slice_tensor: "f32[4000, 16, 129]" = torch.ops.aten.slice.Tensor(bmm_default, 2, 0, -3);  bmm_default = None
        permute_27: "f32[4000, 129, 16]" = torch.ops.aten.permute.default(slice_tensor, [0, 2, 1]);  slice_tensor = None
        mul_41: "f32[4000, 1, 16]" = torch.ops.aten.mul.Tensor(bmm_3, 0.25);  bmm_3 = None
        permute_28: "f32[129, 4000, 16]" = torch.ops.aten.permute.default(bmm_1, [1, 0, 2]);  bmm_1 = None
        clone_5: "f32[129, 4000, 16]" = torch.ops.aten.clone.default(permute_28, memory_format = torch.contiguous_format);  permute_28 = None
        view_21: "f32[129, 500, 128]" = torch.ops.aten.view.default(clone_5, [129, 500, 128]);  clone_5 = None
        permute_29: "f32[129, 4000, 16]" = torch.ops.aten.permute.default(permute_27, [1, 0, 2]);  permute_27 = None
        view_22: "f32[129, 500, 128]" = torch.ops.aten.view.default(permute_29, [129, 500, 128]);  permute_29 = None
        permute_30: "f32[1, 4000, 16]" = torch.ops.aten.permute.default(mul_41, [1, 0, 2]);  mul_41 = None
        view_23: "f32[1, 500, 128]" = torch.ops.aten.view.default(permute_30, [1, 500, 128]);  permute_30 = None
        full_default_1: "f32[2, 129, 500, 128]" = torch.ops.aten.full.default([2, 129, 500, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter: "f32[2, 129, 500, 128]" = torch.ops.aten.select_scatter.default(full_default_1, view_21, 0, 1);  view_21 = None
        select_scatter_1: "f32[2, 129, 500, 128]" = torch.ops.aten.select_scatter.default(full_default_1, view_22, 0, 0);  full_default_1 = view_22 = None
        add_15: "f32[2, 129, 500, 128]" = torch.ops.aten.add.Tensor(select_scatter, select_scatter_1);  select_scatter = select_scatter_1 = None
        unsqueeze_1: "f32[2, 129, 500, 1, 128]" = torch.ops.aten.unsqueeze.default(add_15, 3);  add_15 = None
        permute_31: "f32[1, 129, 500, 2, 128]" = torch.ops.aten.permute.default(unsqueeze_1, [3, 1, 2, 0, 4]);  unsqueeze_1 = None
        squeeze_1: "f32[129, 500, 2, 128]" = torch.ops.aten.squeeze.dim(permute_31, 0);  permute_31 = None
        clone_6: "f32[129, 500, 2, 128]" = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
        view_24: "f32[129, 500, 256]" = torch.ops.aten.view.default(clone_6, [129, 500, 256]);  clone_6 = None
        sum_17: "f32[1, 1, 256]" = torch.ops.aten.sum.dim_IntList(view_24, [0, 1], True)
        view_25: "f32[256]" = torch.ops.aten.view.default(sum_17, [256]);  sum_17 = None
        view_26: "f32[64500, 256]" = torch.ops.aten.view.default(view_24, [64500, 256]);  view_24 = None
        permute_32: "f32[256, 64500]" = torch.ops.aten.permute.default(view_26, [1, 0])
        mm_6: "f32[256, 128]" = torch.ops.aten.mm.default(permute_32, view_2);  permute_32 = view_2 = None
        mm_7: "f32[64500, 128]" = torch.ops.aten.mm.default(view_26, permute_34);  view_26 = permute_34 = None
        view_27: "f32[129, 500, 128]" = torch.ops.aten.view.default(mm_7, [129, 500, 128]);  mm_7 = None
        sum_18: "f32[1, 1, 128]" = torch.ops.aten.sum.dim_IntList(view_23, [0, 1], True)
        view_28: "f32[128]" = torch.ops.aten.view.default(sum_18, [128]);  sum_18 = None
        view_29: "f32[500, 128]" = torch.ops.aten.view.default(view_23, [500, 128]);  view_23 = None
        permute_36: "f32[128, 500]" = torch.ops.aten.permute.default(view_29, [1, 0])
        
        # File: /app/src/models/lorentz_part.py:279 in torch_dynamo_resume_in_forward_at_275, code: x_cls = self.cls_token.expand(B, -1, -1)
        expand: "f32[500, 1, 128]" = torch.ops.aten.expand.default(primals_1, [500, -1, -1]);  primals_1 = None
        
        # File: /app/src/models/classifier.py:46 in forward, code: x, _ = self.mha(x_cls, x, x, key_padding_mask=padding_mask)
        permute: "f32[1, 500, 128]" = torch.ops.aten.permute.default(expand, [1, 0, 2]);  expand = None
        view: "f32[500, 128]" = torch.ops.aten.view.default(permute, [500, 128]);  permute = None
        mm_8: "f32[128, 128]" = torch.ops.aten.mm.default(permute_36, view);  permute_36 = view = None
        mm_9: "f32[500, 128]" = torch.ops.aten.mm.default(view_29, permute_38);  view_29 = permute_38 = None
        view_30: "f32[1, 500, 128]" = torch.ops.aten.view.default(mm_9, [1, 500, 128]);  mm_9 = None
        cat_2: "f32[384]" = torch.ops.aten.cat.default([view_28, view_25]);  view_28 = view_25 = None
        cat_3: "f32[384, 128]" = torch.ops.aten.cat.default([mm_8, mm_6]);  mm_8 = mm_6 = None
        permute_40: "f32[500, 129, 128]" = torch.ops.aten.permute.default(view_27, [1, 0, 2]);  view_27 = None
        permute_41: "f32[500, 1, 128]" = torch.ops.aten.permute.default(view_30, [1, 0, 2]);  view_30 = None
        add_16: "f32[500, 1, 128]" = torch.ops.aten.add.Tensor(add_14, permute_41);  add_14 = permute_41 = None
        
        # File: /app/src/models/classifier.py:45 in forward, code: x = self.layernorm1(x)
        mul_43: "f32[500, 129, 128]" = torch.ops.aten.mul.Tensor(permute_40, primals_4);  primals_4 = None
        mul_44: "f32[500, 129, 128]" = torch.ops.aten.mul.Tensor(mul_43, 128)
        sum_19: "f32[500, 129, 1]" = torch.ops.aten.sum.dim_IntList(mul_43, [2], True)
        sub: "f32[500, 129, 128]" = torch.ops.aten.sub.Tensor(cat_1, getitem_1);  cat_1 = getitem_1 = None
        mul: "f32[500, 129, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_45: "f32[500, 129, 128]" = torch.ops.aten.mul.Tensor(mul_43, mul);  mul_43 = None
        sum_20: "f32[500, 129, 1]" = torch.ops.aten.sum.dim_IntList(mul_45, [2], True);  mul_45 = None
        mul_46: "f32[500, 129, 128]" = torch.ops.aten.mul.Tensor(mul, sum_20);  sum_20 = None
        sub_15: "f32[500, 129, 128]" = torch.ops.aten.sub.Tensor(mul_44, sum_19);  mul_44 = sum_19 = None
        sub_16: "f32[500, 129, 128]" = torch.ops.aten.sub.Tensor(sub_15, mul_46);  sub_15 = mul_46 = None
        div_4: "f32[500, 129, 1]" = torch.ops.aten.div.Tensor(rsqrt, 128);  rsqrt = None
        mul_47: "f32[500, 129, 128]" = torch.ops.aten.mul.Tensor(div_4, sub_16);  div_4 = sub_16 = None
        mul_48: "f32[500, 129, 128]" = torch.ops.aten.mul.Tensor(permute_40, mul);  mul = None
        sum_21: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_48, [0, 1]);  mul_48 = None
        sum_22: "f32[128]" = torch.ops.aten.sum.dim_IntList(permute_40, [0, 1]);  permute_40 = None
        
        # File: /app/src/models/classifier.py:44 in forward, code: x = torch.cat((x_cls, x), dim=1)  # (B, N + 1, D)
        slice_2: "f32[500, 1, 128]" = torch.ops.aten.slice.Tensor(mul_47, 1, 0, 1)
        slice_3: "f32[500, 128, 128]" = torch.ops.aten.slice.Tensor(mul_47, 1, 1, 129);  mul_47 = None
        add_17: "f32[500, 1, 128]" = torch.ops.aten.add.Tensor(add_16, slice_2);  add_16 = slice_2 = None
        
        # File: /app/src/models/lorentz_part.py:279 in torch_dynamo_resume_in_forward_at_275, code: x_cls = self.cls_token.expand(B, -1, -1)
        sum_23: "f32[1, 1, 128]" = torch.ops.aten.sum.dim_IntList(add_17, [0], True);  add_17 = None
        return (sum_23, None, slice_3, sum_21, sum_22, cat_3, cat_2, mm_5, view_19, sum_13, sum_14, sum_9, sum_10, mm_3, view_16, sum_4, sum_5)
        