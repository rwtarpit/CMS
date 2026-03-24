class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "f32[128, 512]", primals_5: "f32[128]", primals_7: "f32[384, 128]", primals_9: "f32[128, 128]", primals_13: "f32[128]", primals_15: "f32[128]", primals_17: "f32[512, 128]", primals_19: "f32[512]", primals_21: "f32[128, 512]", view: "f32[64000, 512]", gt: "b8[500, 128, 128]", mul_2: "f32[500, 128, 128]", view_2: "f32[64000, 128]", baddbmm: "f32[4000, 128, 128]", amax: "f32[4000, 128, 1]", sum_1: "f32[4000, 128, 1]", gt_1: "b8[4000, 128, 128]", view_10: "f32[64000, 128]", addmm_1: "f32[64000, 128]", getitem_3: "f32[500, 128, 1]", rsqrt_1: "f32[500, 128, 1]", gt_2: "b8[500, 128, 128]", mul_11: "f32[500, 128, 128]", view_13: "f32[64000, 128]", addmm_2: "f32[64000, 512]", gt_3: "b8[500, 128, 512]", getitem_7: "f32[500, 128, 1]", rsqrt_3: "f32[500, 128, 1]", view_15: "f32[64000, 512]", gt_4: "b8[500, 128, 128]", div_2: "f32[500, 128, 1]", permute_27: "f32[4000, 128, 128]", permute_28: "f32[4000, 16, 128]", permute_29: "f32[4000, 128, 16]", permute_30: "f32[4000, 16, 128]", div_4: "f32[500, 128, 1]", tangents_1: "f32[500, 128, 128]"):
        # File: /app/src/models/feedforward.py:33 in forward, code: x = self.dropout2(x)
        convert_element_type: "f32[500, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_4, torch.float32);  gt_4 = None
        mul_22: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
        mul_23: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(tangents_1, mul_22);  mul_22 = None
        
        # File: /app/src/models/feedforward.py:32 in forward, code: x = self.linear2(x)
        view_17: "f32[64000, 128]" = torch.ops.aten.view.default(mul_23, [64000, 128]);  mul_23 = None
        permute_12: "f32[512, 128]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
        permute_13: "f32[128, 512]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        mm_1: "f32[64000, 512]" = torch.ops.aten.mm.default(view_17, permute_13);  permute_13 = None
        permute_14: "f32[128, 64000]" = torch.ops.aten.permute.default(view_17, [1, 0])
        mm_2: "f32[128, 512]" = torch.ops.aten.mm.default(permute_14, view_15);  permute_14 = view_15 = None
        sum_2: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_17, [0], True);  view_17 = None
        view_18: "f32[128]" = torch.ops.aten.view.default(sum_2, [128]);  sum_2 = None
        view_19: "f32[500, 128, 512]" = torch.ops.aten.view.default(mm_1, [500, 128, 512]);  mm_1 = None
        
        # File: /app/src/models/feedforward.py:31 in forward, code: x = self.layernorm2(x)
        mul_25: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_19, primals_19);  primals_19 = None
        mul_26: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_25, 512)
        sum_3: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_25, [2], True)
        
        # File: /app/src/models/feedforward.py:27 in forward, code: x = self.linear1(x)
        view_14: "f32[500, 128, 512]" = torch.ops.aten.view.default(addmm_2, [500, 128, 512]);  addmm_2 = None
        
        # File: /app/src/models/feedforward.py:28 in forward, code: x = self.act(x)
        mul_13: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_14, 0.5)
        mul_14: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_14, 0.7071067811865476)
        erf: "f32[500, 128, 512]" = torch.ops.aten.erf.default(mul_14);  mul_14 = None
        add_10: "f32[500, 128, 512]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_15: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_13, add_10);  mul_13 = None
        
        # File: /app/src/models/feedforward.py:29 in forward, code: x = self.dropout1(x)
        mul_16: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(gt_3, mul_15);  mul_15 = None
        mul_17: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_16, 1.1111111111111112);  mul_16 = None
        
        # File: /app/src/models/feedforward.py:31 in forward, code: x = self.layernorm2(x)
        sub_4: "f32[500, 128, 512]" = torch.ops.aten.sub.Tensor(mul_17, getitem_7);  mul_17 = getitem_7 = None
        mul_18: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = None
        mul_27: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_25, mul_18);  mul_25 = None
        sum_4: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_27, [2], True);  mul_27 = None
        mul_28: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_18, sum_4);  sum_4 = None
        sub_6: "f32[500, 128, 512]" = torch.ops.aten.sub.Tensor(mul_26, sum_3);  mul_26 = sum_3 = None
        sub_7: "f32[500, 128, 512]" = torch.ops.aten.sub.Tensor(sub_6, mul_28);  sub_6 = mul_28 = None
        div_1: "f32[500, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 512);  rsqrt_3 = None
        mul_29: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(div_1, sub_7);  div_1 = sub_7 = None
        mul_30: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_19, mul_18);  mul_18 = None
        sum_5: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_30, [0, 1]);  mul_30 = None
        sum_6: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_19, [0, 1]);  view_19 = None
        
        # File: /app/src/models/feedforward.py:29 in forward, code: x = self.dropout1(x)
        convert_element_type_1: "f32[500, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_3, torch.float32);  gt_3 = None
        mul_31: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_32: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_29, mul_31);  mul_29 = mul_31 = None
        
        # File: /app/src/models/feedforward.py:28 in forward, code: x = self.act(x)
        mul_34: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(add_10, 0.5);  add_10 = None
        mul_35: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_14, view_14)
        mul_36: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_35, -0.5);  mul_35 = None
        exp_1: "f32[500, 128, 512]" = torch.ops.aten.exp.default(mul_36);  mul_36 = None
        mul_37: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
        mul_38: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(view_14, mul_37);  view_14 = mul_37 = None
        add_15: "f32[500, 128, 512]" = torch.ops.aten.add.Tensor(mul_34, mul_38);  mul_34 = mul_38 = None
        mul_39: "f32[500, 128, 512]" = torch.ops.aten.mul.Tensor(mul_32, add_15);  mul_32 = add_15 = None
        
        # File: /app/src/models/feedforward.py:27 in forward, code: x = self.linear1(x)
        view_20: "f32[64000, 512]" = torch.ops.aten.view.default(mul_39, [64000, 512]);  mul_39 = None
        permute_11: "f32[128, 512]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
        permute_17: "f32[512, 128]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        mm_3: "f32[64000, 128]" = torch.ops.aten.mm.default(view_20, permute_17);  permute_17 = None
        permute_18: "f32[512, 64000]" = torch.ops.aten.permute.default(view_20, [1, 0])
        mm_4: "f32[512, 128]" = torch.ops.aten.mm.default(permute_18, view_13);  permute_18 = view_13 = None
        sum_7: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_20, [0], True);  view_20 = None
        view_21: "f32[512]" = torch.ops.aten.view.default(sum_7, [512]);  sum_7 = None
        view_22: "f32[500, 128, 128]" = torch.ops.aten.view.default(mm_3, [500, 128, 128]);  mm_3 = None
        
        # File: /app/src/models/feedforward.py:26 in forward, code: x = self.layernorm1(x)
        mul_41: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(view_22, primals_15);  primals_15 = None
        mul_42: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_41, 128)
        sum_8: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_41, [2], True)
        mul_43: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_41, mul_11);  mul_41 = None
        sum_9: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_43, [2], True);  mul_43 = None
        mul_44: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_11, sum_9);  sum_9 = None
        sub_9: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(mul_42, sum_8);  mul_42 = sum_8 = None
        sub_10: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(sub_9, mul_44);  sub_9 = mul_44 = None
        mul_45: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(div_2, sub_10);  div_2 = sub_10 = None
        mul_46: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(view_22, mul_11);  mul_11 = None
        sum_10: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_46, [0, 1]);  mul_46 = None
        sum_11: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_22, [0, 1]);  view_22 = None
        add_16: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(tangents_1, mul_45);  tangents_1 = mul_45 = None
        
        # File: /app/src/models/particle_transformer.py:46 in forward, code: x = self.dropout(x)
        convert_element_type_2: "f32[500, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_47: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
        mul_48: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(add_16, mul_47);  mul_47 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        view_11: "f32[128, 500, 128]" = torch.ops.aten.view.default(addmm_1, [128, 500, 128]);  addmm_1 = None
        permute_10: "f32[500, 128, 128]" = torch.ops.aten.permute.default(view_11, [1, 0, 2]);  view_11 = None
        
        # File: /app/src/models/particle_transformer.py:45 in forward, code: x = self.layernorm2(x)
        sub_11: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(permute_10, getitem_3);  permute_10 = getitem_3 = None
        mul_49: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_1);  sub_11 = None
        mul_50: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_48, primals_13);  primals_13 = None
        mul_51: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_50, 128)
        sum_12: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_50, [2], True)
        mul_52: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_50, mul_49);  mul_50 = None
        sum_13: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_52, [2], True);  mul_52 = None
        mul_53: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_49, sum_13);  sum_13 = None
        sub_12: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(mul_51, sum_12);  mul_51 = sum_12 = None
        sub_13: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(sub_12, mul_53);  sub_12 = mul_53 = None
        div_3: "f32[500, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 128);  rsqrt_1 = None
        mul_54: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(div_3, sub_13);  div_3 = sub_13 = None
        mul_55: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_48, mul_49);  mul_49 = None
        sum_14: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_55, [0, 1]);  mul_55 = None
        sum_15: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_48, [0, 1]);  mul_48 = None
        
        # File: /app/src/models/particle_transformer.py:44 in forward, code: x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        permute_21: "f32[128, 500, 128]" = torch.ops.aten.permute.default(mul_54, [1, 0, 2]);  mul_54 = None
        clone_8: "f32[128, 500, 128]" = torch.ops.aten.clone.default(permute_21, memory_format = torch.contiguous_format);  permute_21 = None
        view_23: "f32[64000, 128]" = torch.ops.aten.view.default(clone_8, [64000, 128]);  clone_8 = None
        permute_9: "f32[128, 128]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        permute_22: "f32[128, 128]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_5: "f32[64000, 128]" = torch.ops.aten.mm.default(view_23, permute_22);  permute_22 = None
        permute_23: "f32[128, 64000]" = torch.ops.aten.permute.default(view_23, [1, 0])
        mm_6: "f32[128, 128]" = torch.ops.aten.mm.default(permute_23, view_10);  permute_23 = view_10 = None
        sum_16: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_23, [0], True);  view_23 = None
        view_24: "f32[128]" = torch.ops.aten.view.default(sum_16, [128]);  sum_16 = None
        view_25: "f32[128, 4000, 16]" = torch.ops.aten.view.default(mm_5, [128, 4000, 16]);  mm_5 = None
        permute_26: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(view_25, [1, 0, 2]);  view_25 = None
        bmm_1: "f32[4000, 128, 16]" = torch.ops.aten.bmm.default(permute_27, permute_26);  permute_27 = None
        bmm_2: "f32[4000, 128, 128]" = torch.ops.aten.bmm.default(permute_26, permute_28);  permute_26 = permute_28 = None
        convert_element_type_3: "f32[4000, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_56: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.1111111111111112);  convert_element_type_3 = None
        mul_57: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(bmm_2, mul_56);  bmm_2 = mul_56 = None
        sub_1: "f32[4000, 128, 128]" = torch.ops.aten.sub.Tensor(baddbmm, amax);  baddbmm = amax = None
        exp: "f32[4000, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        div: "f32[4000, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        mul_58: "f32[4000, 128, 128]" = torch.ops.aten.mul.Tensor(mul_57, div);  mul_57 = None
        sum_17: "f32[4000, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_58, [-1], True)
        neg: "f32[4000, 128, 128]" = torch.ops.aten.neg.default(div);  div = None
        fma: "f32[4000, 128, 128]" = torch.ops.prims.fma.default(neg, sum_17, mul_58);  neg = sum_17 = mul_58 = None
        bmm_3: "f32[4000, 128, 16]" = torch.ops.aten.bmm.default(fma, permute_29);  permute_29 = None
        bmm_4: "f32[4000, 16, 128]" = torch.ops.aten.bmm.default(permute_30, fma);  permute_30 = None
        permute_31: "f32[4000, 128, 16]" = torch.ops.aten.permute.default(bmm_4, [0, 2, 1]);  bmm_4 = None
        mul_59: "f32[4000, 128, 16]" = torch.ops.aten.mul.Tensor(bmm_3, 0.25);  bmm_3 = None
        permute_32: "f32[128, 4000, 16]" = torch.ops.aten.permute.default(bmm_1, [1, 0, 2]);  bmm_1 = None
        clone_10: "f32[128, 4000, 16]" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
        view_26: "f32[128, 500, 128]" = torch.ops.aten.view.default(clone_10, [128, 500, 128]);  clone_10 = None
        permute_33: "f32[128, 4000, 16]" = torch.ops.aten.permute.default(permute_31, [1, 0, 2]);  permute_31 = None
        view_27: "f32[128, 500, 128]" = torch.ops.aten.view.default(permute_33, [128, 500, 128]);  permute_33 = None
        permute_34: "f32[128, 4000, 16]" = torch.ops.aten.permute.default(mul_59, [1, 0, 2]);  mul_59 = None
        clone_11: "f32[128, 4000, 16]" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
        view_28: "f32[128, 500, 128]" = torch.ops.aten.view.default(clone_11, [128, 500, 128]);  clone_11 = None
        full_default: "f32[3, 128, 500, 128]" = torch.ops.aten.full.default([3, 128, 500, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter: "f32[3, 128, 500, 128]" = torch.ops.aten.select_scatter.default(full_default, view_26, 0, 2);  view_26 = None
        select_scatter_1: "f32[3, 128, 500, 128]" = torch.ops.aten.select_scatter.default(full_default, view_27, 0, 1);  view_27 = None
        add_17: "f32[3, 128, 500, 128]" = torch.ops.aten.add.Tensor(select_scatter, select_scatter_1);  select_scatter = select_scatter_1 = None
        select_scatter_2: "f32[3, 128, 500, 128]" = torch.ops.aten.select_scatter.default(full_default, view_28, 0, 0);  full_default = view_28 = None
        add_18: "f32[3, 128, 500, 128]" = torch.ops.aten.add.Tensor(add_17, select_scatter_2);  add_17 = select_scatter_2 = None
        unsqueeze_1: "f32[3, 128, 500, 1, 128]" = torch.ops.aten.unsqueeze.default(add_18, 3);  add_18 = None
        permute_35: "f32[1, 128, 500, 3, 128]" = torch.ops.aten.permute.default(unsqueeze_1, [3, 1, 2, 0, 4]);  unsqueeze_1 = None
        squeeze_1: "f32[128, 500, 3, 128]" = torch.ops.aten.squeeze.dim(permute_35, 0);  permute_35 = None
        clone_12: "f32[128, 500, 3, 128]" = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
        view_29: "f32[128, 500, 384]" = torch.ops.aten.view.default(clone_12, [128, 500, 384]);  clone_12 = None
        sum_18: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(view_29, [0, 1], True)
        view_30: "f32[384]" = torch.ops.aten.view.default(sum_18, [384]);  sum_18 = None
        view_31: "f32[64000, 384]" = torch.ops.aten.view.default(view_29, [64000, 384]);  view_29 = None
        permute_36: "f32[384, 64000]" = torch.ops.aten.permute.default(view_31, [1, 0])
        mm_7: "f32[384, 128]" = torch.ops.aten.mm.default(permute_36, view_2);  permute_36 = view_2 = None
        permute_2: "f32[128, 384]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        permute_38: "f32[384, 128]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_8: "f32[64000, 128]" = torch.ops.aten.mm.default(view_31, permute_38);  view_31 = permute_38 = None
        view_32: "f32[128, 500, 128]" = torch.ops.aten.view.default(mm_8, [128, 500, 128]);  mm_8 = None
        permute_40: "f32[500, 128, 128]" = torch.ops.aten.permute.default(view_32, [1, 0, 2]);  view_32 = None
        
        # File: /app/src/models/particle_transformer.py:43 in forward, code: x = self.layernorm1(x)
        mul_61: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(permute_40, primals_5);  primals_5 = None
        mul_62: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_61, 128)
        sum_19: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_61, [2], True)
        mul_63: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_61, mul_2);  mul_61 = None
        sum_20: "f32[500, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_63, [2], True);  mul_63 = None
        mul_64: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(mul_2, sum_20);  sum_20 = None
        sub_15: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(mul_62, sum_19);  mul_62 = sum_19 = None
        sub_16: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(sub_15, mul_64);  sub_15 = mul_64 = None
        mul_65: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(div_4, sub_16);  div_4 = sub_16 = None
        mul_66: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(permute_40, mul_2);  mul_2 = None
        sum_21: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_66, [0, 1]);  mul_66 = None
        sum_22: "f32[128]" = torch.ops.aten.sum.dim_IntList(permute_40, [0, 1]);  permute_40 = None
        add_19: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(add_16, mul_65);  add_16 = mul_65 = None
        
        # File: /app/src/models/feedforward.py:33 in forward, code: x = self.dropout2(x)
        convert_element_type_4: "f32[500, 128, 128]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_67: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
        mul_68: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(add_19, mul_67);  mul_67 = None
        
        # File: /app/src/models/feedforward.py:32 in forward, code: x = self.linear2(x)
        view_33: "f32[64000, 128]" = torch.ops.aten.view.default(mul_68, [64000, 128]);  mul_68 = None
        permute: "f32[512, 128]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        permute_41: "f32[128, 512]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_9: "f32[64000, 512]" = torch.ops.aten.mm.default(view_33, permute_41);  permute_41 = None
        permute_42: "f32[128, 64000]" = torch.ops.aten.permute.default(view_33, [1, 0])
        mm_10: "f32[128, 512]" = torch.ops.aten.mm.default(permute_42, view);  permute_42 = view = None
        sum_23: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_33, [0], True);  view_33 = None
        view_34: "f32[128]" = torch.ops.aten.view.default(sum_23, [128]);  sum_23 = None
        view_35: "f32[500, 128, 512]" = torch.ops.aten.view.default(mm_9, [500, 128, 512]);  mm_9 = None
        return (view_35, mm_10, view_34, add_19, sum_21, sum_22, mm_7, view_30, mm_6, view_24, None, fma, sum_14, sum_15, sum_10, sum_11, mm_4, view_21, sum_5, sum_6, mm_2, view_18)
        