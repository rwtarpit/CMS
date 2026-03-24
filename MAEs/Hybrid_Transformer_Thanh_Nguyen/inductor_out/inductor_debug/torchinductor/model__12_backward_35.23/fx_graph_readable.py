class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "f32[500, 128, 128, 4]", primals_8: "f32[64, 4, 1]", primals_13: "f32[64]", primals_14: "f32[64]", primals_15: "f32[64, 64, 1]", primals_20: "f32[64]", primals_21: "f32[64]", primals_22: "f32[64, 64, 1]", primals_27: "f32[64]", primals_28: "f32[64]", primals_29: "f32[8, 64, 1]", primals_34: "f32[8]", primals_35: "f32[8]", squeeze_1: "f32[4]", add_4: "f32[500, 4, 16384]", convolution: "f32[500, 64, 16384]", getitem_3: "f32[1, 64, 1]", rsqrt_1: "f32[1, 64, 1]", mul_16: "f32[500, 64, 16384]", convolution_1: "f32[500, 64, 16384]", getitem_5: "f32[1, 64, 1]", rsqrt_2: "f32[1, 64, 1]", mul_26: "f32[500, 64, 16384]", convolution_2: "f32[500, 64, 16384]", getitem_7: "f32[1, 64, 1]", rsqrt_3: "f32[1, 64, 1]", mul_36: "f32[500, 64, 16384]", convolution_3: "f32[500, 8, 16384]", getitem_9: "f32[1, 8, 1]", rsqrt_4: "f32[1, 8, 1]", unsqueeze_43: "f32[1, 4, 1]", tangents_1: "f32[4000, 128, 128]"):
        # File: /app/src/models/processor.py:130 in forward, code: U = U.view(B * U.shape[1], N, N)  # (B * num_heads, N, N)
        view_3: "f32[500, 8, 16384]" = torch.ops.aten.view.default(tangents_1, [500, 8, 16384]);  tangents_1 = None
        
        # File: /app/src/models/processor.py:129 in forward, code: U = self.embed(U)  # (B, num_heads, N * N)
        sub_4: "f32[500, 8, 16384]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_9)
        mul_37: "f32[500, 8, 16384]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        unsqueeze_8: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1)
        mul_43: "f32[500, 8, 16384]" = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_8);  mul_37 = unsqueeze_8 = None
        unsqueeze_9: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1);  primals_35 = None
        add_27: "f32[500, 8, 16384]" = torch.ops.aten.add.Tensor(mul_43, unsqueeze_9);  mul_43 = unsqueeze_9 = None
        mul_45: "f32[500, 8, 16384]" = torch.ops.aten.mul.Tensor(add_27, 0.7071067811865476)
        erf_3: "f32[500, 8, 16384]" = torch.ops.aten.erf.default(mul_45);  mul_45 = None
        add_28: "f32[500, 8, 16384]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_48: "f32[500, 8, 16384]" = torch.ops.aten.mul.Tensor(add_28, 0.5);  add_28 = None
        mul_49: "f32[500, 8, 16384]" = torch.ops.aten.mul.Tensor(add_27, add_27)
        mul_50: "f32[500, 8, 16384]" = torch.ops.aten.mul.Tensor(mul_49, -0.5);  mul_49 = None
        exp: "f32[500, 8, 16384]" = torch.ops.aten.exp.default(mul_50);  mul_50 = None
        mul_51: "f32[500, 8, 16384]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_52: "f32[500, 8, 16384]" = torch.ops.aten.mul.Tensor(add_27, mul_51);  add_27 = mul_51 = None
        add_30: "f32[500, 8, 16384]" = torch.ops.aten.add.Tensor(mul_48, mul_52);  mul_48 = mul_52 = None
        mul_53: "f32[500, 8, 16384]" = torch.ops.aten.mul.Tensor(view_3, add_30);  view_3 = add_30 = None
        squeeze_12: "f32[8]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2]);  getitem_9 = None
        unsqueeze_10: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
        unsqueeze_11: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None
        sum_1: "f32[8]" = torch.ops.aten.sum.dim_IntList(mul_53, [0, 2])
        sub_5: "f32[500, 8, 16384]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_11);  convolution_3 = unsqueeze_11 = None
        mul_54: "f32[500, 8, 16384]" = torch.ops.aten.mul.Tensor(mul_53, sub_5)
        sum_2: "f32[8]" = torch.ops.aten.sum.dim_IntList(mul_54, [0, 2]);  mul_54 = None
        mul_55: "f32[8]" = torch.ops.aten.mul.Tensor(sum_1, 1.220703125e-07)
        unsqueeze_12: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_55, 0);  mul_55 = None
        unsqueeze_13: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, 2);  unsqueeze_12 = None
        mul_56: "f32[8]" = torch.ops.aten.mul.Tensor(sum_2, 1.220703125e-07)
        squeeze_13: "f32[8]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2]);  rsqrt_4 = None
        mul_57: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_58: "f32[8]" = torch.ops.aten.mul.Tensor(mul_56, mul_57);  mul_56 = mul_57 = None
        unsqueeze_14: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_58, 0);  mul_58 = None
        unsqueeze_15: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 2);  unsqueeze_14 = None
        mul_59: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_34);  primals_34 = None
        unsqueeze_16: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_59, 0);  mul_59 = None
        unsqueeze_17: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, 2);  unsqueeze_16 = None
        mul_60: "f32[500, 8, 16384]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_15);  sub_5 = unsqueeze_15 = None
        sub_7: "f32[500, 8, 16384]" = torch.ops.aten.sub.Tensor(mul_53, mul_60);  mul_53 = mul_60 = None
        sub_8: "f32[500, 8, 16384]" = torch.ops.aten.sub.Tensor(sub_7, unsqueeze_13);  sub_7 = unsqueeze_13 = None
        mul_61: "f32[500, 8, 16384]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_17);  sub_8 = unsqueeze_17 = None
        mul_62: "f32[8]" = torch.ops.aten.mul.Tensor(sum_2, squeeze_13);  sum_2 = squeeze_13 = None
        sum_3: "f32[8]" = torch.ops.aten.sum.dim_IntList(mul_61, [0, 2])
        convolution_backward = torch.ops.aten.convolution_backward.default(mul_61, mul_36, primals_29, [8], [1], [0], [1], False, [0], 1, [True, True, False]);  mul_61 = mul_36 = primals_29 = None
        getitem_10: "f32[500, 64, 16384]" = convolution_backward[0]
        getitem_11: "f32[8, 64, 1]" = convolution_backward[1];  convolution_backward = None
        sub_3: "f32[500, 64, 16384]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_7)
        mul_27: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1)
        mul_33: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(mul_27, unsqueeze_6);  mul_27 = unsqueeze_6 = None
        unsqueeze_7: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1);  primals_28 = None
        add_21: "f32[500, 64, 16384]" = torch.ops.aten.add.Tensor(mul_33, unsqueeze_7);  mul_33 = unsqueeze_7 = None
        mul_35: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(add_21, 0.7071067811865476)
        erf_2: "f32[500, 64, 16384]" = torch.ops.aten.erf.default(mul_35);  mul_35 = None
        add_22: "f32[500, 64, 16384]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_64: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(add_22, 0.5);  add_22 = None
        mul_65: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(add_21, add_21)
        mul_66: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(mul_65, -0.5);  mul_65 = None
        exp_1: "f32[500, 64, 16384]" = torch.ops.aten.exp.default(mul_66);  mul_66 = None
        mul_67: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
        mul_68: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(add_21, mul_67);  add_21 = mul_67 = None
        add_32: "f32[500, 64, 16384]" = torch.ops.aten.add.Tensor(mul_64, mul_68);  mul_64 = mul_68 = None
        mul_69: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(getitem_10, add_32);  getitem_10 = add_32 = None
        squeeze_9: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2]);  getitem_7 = None
        unsqueeze_18: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
        unsqueeze_19: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, 2);  unsqueeze_18 = None
        sum_4: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_69, [0, 2])
        sub_9: "f32[500, 64, 16384]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_19);  convolution_2 = unsqueeze_19 = None
        mul_70: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(mul_69, sub_9)
        sum_5: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_70, [0, 2]);  mul_70 = None
        mul_71: "f32[64]" = torch.ops.aten.mul.Tensor(sum_4, 1.220703125e-07)
        unsqueeze_20: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_71, 0);  mul_71 = None
        unsqueeze_21: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, 2);  unsqueeze_20 = None
        mul_72: "f32[64]" = torch.ops.aten.mul.Tensor(sum_5, 1.220703125e-07)
        squeeze_10: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2]);  rsqrt_3 = None
        mul_73: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_74: "f32[64]" = torch.ops.aten.mul.Tensor(mul_72, mul_73);  mul_72 = mul_73 = None
        unsqueeze_22: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_74, 0);  mul_74 = None
        unsqueeze_23: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, 2);  unsqueeze_22 = None
        mul_75: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_27);  primals_27 = None
        unsqueeze_24: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_75, 0);  mul_75 = None
        unsqueeze_25: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, 2);  unsqueeze_24 = None
        mul_76: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_23);  sub_9 = unsqueeze_23 = None
        sub_11: "f32[500, 64, 16384]" = torch.ops.aten.sub.Tensor(mul_69, mul_76);  mul_69 = mul_76 = None
        sub_12: "f32[500, 64, 16384]" = torch.ops.aten.sub.Tensor(sub_11, unsqueeze_21);  sub_11 = unsqueeze_21 = None
        mul_77: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_25);  sub_12 = unsqueeze_25 = None
        mul_78: "f32[64]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_10);  sum_5 = squeeze_10 = None
        sum_6: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_77, [0, 2])
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_77, mul_26, primals_22, [64], [1], [0], [1], False, [0], 1, [True, True, False]);  mul_77 = mul_26 = primals_22 = None
        getitem_13: "f32[500, 64, 16384]" = convolution_backward_1[0]
        getitem_14: "f32[64, 64, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
        sub_2: "f32[500, 64, 16384]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_5)
        mul_17: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        unsqueeze_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1)
        mul_23: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(mul_17, unsqueeze_4);  mul_17 = unsqueeze_4 = None
        unsqueeze_5: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1);  primals_21 = None
        add_15: "f32[500, 64, 16384]" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_5);  mul_23 = unsqueeze_5 = None
        mul_25: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(add_15, 0.7071067811865476)
        erf_1: "f32[500, 64, 16384]" = torch.ops.aten.erf.default(mul_25);  mul_25 = None
        add_16: "f32[500, 64, 16384]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_80: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(add_16, 0.5);  add_16 = None
        mul_81: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(add_15, add_15)
        mul_82: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(mul_81, -0.5);  mul_81 = None
        exp_2: "f32[500, 64, 16384]" = torch.ops.aten.exp.default(mul_82);  mul_82 = None
        mul_83: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
        mul_84: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(add_15, mul_83);  add_15 = mul_83 = None
        add_34: "f32[500, 64, 16384]" = torch.ops.aten.add.Tensor(mul_80, mul_84);  mul_80 = mul_84 = None
        mul_85: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(getitem_13, add_34);  getitem_13 = add_34 = None
        squeeze_6: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2]);  getitem_5 = None
        unsqueeze_26: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
        unsqueeze_27: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, 2);  unsqueeze_26 = None
        sum_7: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_85, [0, 2])
        sub_13: "f32[500, 64, 16384]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_27);  convolution_1 = unsqueeze_27 = None
        mul_86: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(mul_85, sub_13)
        sum_8: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_86, [0, 2]);  mul_86 = None
        mul_87: "f32[64]" = torch.ops.aten.mul.Tensor(sum_7, 1.220703125e-07)
        unsqueeze_28: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_87, 0);  mul_87 = None
        unsqueeze_29: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, 2);  unsqueeze_28 = None
        mul_88: "f32[64]" = torch.ops.aten.mul.Tensor(sum_8, 1.220703125e-07)
        squeeze_7: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2]);  rsqrt_2 = None
        mul_89: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_90: "f32[64]" = torch.ops.aten.mul.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
        unsqueeze_30: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_90, 0);  mul_90 = None
        unsqueeze_31: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 2);  unsqueeze_30 = None
        mul_91: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_20);  primals_20 = None
        unsqueeze_32: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_91, 0);  mul_91 = None
        unsqueeze_33: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, 2);  unsqueeze_32 = None
        mul_92: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_31);  sub_13 = unsqueeze_31 = None
        sub_15: "f32[500, 64, 16384]" = torch.ops.aten.sub.Tensor(mul_85, mul_92);  mul_85 = mul_92 = None
        sub_16: "f32[500, 64, 16384]" = torch.ops.aten.sub.Tensor(sub_15, unsqueeze_29);  sub_15 = unsqueeze_29 = None
        mul_93: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_33);  sub_16 = unsqueeze_33 = None
        mul_94: "f32[64]" = torch.ops.aten.mul.Tensor(sum_8, squeeze_7);  sum_8 = squeeze_7 = None
        sum_9: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_93, [0, 2])
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_93, mul_16, primals_15, [64], [1], [0], [1], False, [0], 1, [True, True, False]);  mul_93 = mul_16 = primals_15 = None
        getitem_16: "f32[500, 64, 16384]" = convolution_backward_2[0]
        getitem_17: "f32[64, 64, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
        sub_1: "f32[500, 64, 16384]" = torch.ops.aten.sub.Tensor(convolution, getitem_3)
        mul_7: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
        mul_13: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_2);  mul_7 = unsqueeze_2 = None
        unsqueeze_3: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1);  primals_14 = None
        add_9: "f32[500, 64, 16384]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_3);  mul_13 = unsqueeze_3 = None
        mul_15: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(add_9, 0.7071067811865476)
        erf: "f32[500, 64, 16384]" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
        add_10: "f32[500, 64, 16384]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_96: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(add_10, 0.5);  add_10 = None
        mul_97: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(add_9, add_9)
        mul_98: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(mul_97, -0.5);  mul_97 = None
        exp_3: "f32[500, 64, 16384]" = torch.ops.aten.exp.default(mul_98);  mul_98 = None
        mul_99: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(exp_3, 0.3989422804014327);  exp_3 = None
        mul_100: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(add_9, mul_99);  add_9 = mul_99 = None
        add_36: "f32[500, 64, 16384]" = torch.ops.aten.add.Tensor(mul_96, mul_100);  mul_96 = mul_100 = None
        mul_101: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(getitem_16, add_36);  getitem_16 = add_36 = None
        squeeze_3: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2]);  getitem_3 = None
        unsqueeze_34: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
        unsqueeze_35: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, 2);  unsqueeze_34 = None
        sum_10: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_101, [0, 2])
        sub_17: "f32[500, 64, 16384]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_35);  convolution = unsqueeze_35 = None
        mul_102: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(mul_101, sub_17)
        sum_11: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_102, [0, 2]);  mul_102 = None
        mul_103: "f32[64]" = torch.ops.aten.mul.Tensor(sum_10, 1.220703125e-07)
        unsqueeze_36: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_103, 0);  mul_103 = None
        unsqueeze_37: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, 2);  unsqueeze_36 = None
        mul_104: "f32[64]" = torch.ops.aten.mul.Tensor(sum_11, 1.220703125e-07)
        squeeze_4: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2]);  rsqrt_1 = None
        mul_105: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_106: "f32[64]" = torch.ops.aten.mul.Tensor(mul_104, mul_105);  mul_104 = mul_105 = None
        unsqueeze_38: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_106, 0);  mul_106 = None
        unsqueeze_39: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, 2);  unsqueeze_38 = None
        mul_107: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_13);  primals_13 = None
        unsqueeze_40: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_107, 0);  mul_107 = None
        unsqueeze_41: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, 2);  unsqueeze_40 = None
        mul_108: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_39);  sub_17 = unsqueeze_39 = None
        sub_19: "f32[500, 64, 16384]" = torch.ops.aten.sub.Tensor(mul_101, mul_108);  mul_101 = mul_108 = None
        sub_20: "f32[500, 64, 16384]" = torch.ops.aten.sub.Tensor(sub_19, unsqueeze_37);  sub_19 = unsqueeze_37 = None
        mul_109: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_41);  sub_20 = unsqueeze_41 = None
        mul_110: "f32[64]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_4);  sum_11 = squeeze_4 = None
        sum_12: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_109, [0, 2])
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_109, add_4, primals_8, [64], [1], [0], [1], False, [0], 1, [True, True, False]);  mul_109 = add_4 = primals_8 = None
        getitem_19: "f32[500, 4, 16384]" = convolution_backward_3[0]
        getitem_20: "f32[64, 4, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
        sum_13: "f32[4]" = torch.ops.aten.sum.dim_IntList(getitem_19, [0, 2])
        
        # File: /app/src/models/processor.py:128 in forward, code: U = U.view(B, N * N, F).transpose(1, 2)  # (B, F, N * N)
        view: "f32[500, 16384, 4]" = torch.ops.aten.view.default(primals_2, [500, 16384, 4]);  primals_2 = None
        permute: "f32[500, 4, 16384]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
        
        # File: /app/src/models/processor.py:129 in forward, code: U = self.embed(U)  # (B, num_heads, N * N)
        clone: "f32[500, 4, 16384]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        sub_21: "f32[500, 4, 16384]" = torch.ops.aten.sub.Tensor(clone, unsqueeze_43);  clone = unsqueeze_43 = None
        mul_111: "f32[500, 4, 16384]" = torch.ops.aten.mul.Tensor(getitem_19, sub_21);  getitem_19 = sub_21 = None
        sum_14: "f32[4]" = torch.ops.aten.sum.dim_IntList(mul_111, [0, 2]);  mul_111 = None
        mul_119: "f32[4]" = torch.ops.aten.mul.Tensor(sum_14, squeeze_1);  sum_14 = squeeze_1 = None
        return (None, None, None, None, None, mul_119, sum_13, getitem_20, sum_12, None, None, None, mul_110, sum_10, getitem_17, sum_9, None, None, None, mul_94, sum_7, getitem_14, sum_6, None, None, None, mul_78, sum_4, getitem_11, sum_3, None, None, None, mul_62, sum_1)
        