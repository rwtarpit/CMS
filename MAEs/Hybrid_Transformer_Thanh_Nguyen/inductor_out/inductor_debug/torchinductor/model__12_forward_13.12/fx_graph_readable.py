class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[500, 128, 16]", primals_2: "f32[500, 128, 128, 4]", primals_3: "i64[]", primals_4: "f32[4]", primals_5: "f32[4]", primals_6: "f32[4]", primals_7: "f32[4]", primals_8: "f32[64, 4, 1]", primals_9: "f32[64]", primals_10: "i64[]", primals_11: "f32[64]", primals_12: "f32[64]", primals_13: "f32[64]", primals_14: "f32[64]", primals_15: "f32[64, 64, 1]", primals_16: "f32[64]", primals_17: "i64[]", primals_18: "f32[64]", primals_19: "f32[64]", primals_20: "f32[64]", primals_21: "f32[64]", primals_22: "f32[64, 64, 1]", primals_23: "f32[64]", primals_24: "i64[]", primals_25: "f32[64]", primals_26: "f32[64]", primals_27: "f32[64]", primals_28: "f32[64]", primals_29: "f32[8, 64, 1]", primals_30: "f32[8]", primals_31: "i64[]", primals_32: "f32[8]", primals_33: "f32[8]", primals_34: "f32[8]", primals_35: "f32[8]"):
        # File: /app/src/models/processor.py:128 in forward, code: U = U.view(B, N * N, F).transpose(1, 2)  # (B, F, N * N)
        view: "f32[500, 16384, 4]" = torch.ops.aten.view.default(primals_2, [500, 16384, 4])
        permute: "f32[500, 4, 16384]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
        
        # File: /app/src/models/processor.py:129 in forward, code: U = self.embed(U)  # (B, num_heads, N * N)
        add: "i64[]" = torch.ops.aten.add.Tensor(primals_3, 1)
        clone: "f32[500, 4, 16384]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        var_mean = torch.ops.aten.var_mean.correction(clone, [0, 2], correction = 0, keepdim = True)
        getitem: "f32[1, 4, 1]" = var_mean[0]
        getitem_1: "f32[1, 4, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[1, 4, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt: "f32[1, 4, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[500, 4, 16384]" = torch.ops.aten.sub.Tensor(clone, getitem_1);  clone = None
        mul: "f32[500, 4, 16384]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[4]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2]);  getitem_1 = None
        squeeze_1: "f32[4]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2]);  rsqrt = None
        mul_1: "f32[4]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
        mul_2: "f32[4]" = torch.ops.aten.mul.Tensor(primals_4, 0.9)
        add_2: "f32[4]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_2: "f32[4]" = torch.ops.aten.squeeze.dims(getitem, [0, 2]);  getitem = None
        mul_3: "f32[4]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000001220703274);  squeeze_2 = None
        mul_4: "f32[4]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5: "f32[4]" = torch.ops.aten.mul.Tensor(primals_5, 0.9)
        add_3: "f32[4]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze: "f32[4, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
        mul_6: "f32[500, 4, 16384]" = torch.ops.aten.mul.Tensor(mul, unsqueeze);  mul = unsqueeze = None
        unsqueeze_1: "f32[4, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1);  primals_7 = None
        add_4: "f32[500, 4, 16384]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_1);  mul_6 = unsqueeze_1 = None
        convolution: "f32[500, 64, 16384]" = torch.ops.aten.convolution.default(add_4, primals_8, primals_9, [1], [0], [1], False, [0], 1);  primals_9 = None
        add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_10, 1)
        var_mean_1 = torch.ops.aten.var_mean.correction(convolution, [0, 2], correction = 0, keepdim = True)
        getitem_2: "f32[1, 64, 1]" = var_mean_1[0]
        getitem_3: "f32[1, 64, 1]" = var_mean_1[1];  var_mean_1 = None
        add_6: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
        rsqrt_1: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_1: "f32[500, 64, 16384]" = torch.ops.aten.sub.Tensor(convolution, getitem_3)
        mul_7: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        squeeze_3: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2])
        mul_8: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1);  squeeze_3 = None
        mul_9: "f32[64]" = torch.ops.aten.mul.Tensor(primals_11, 0.9)
        add_7: "f32[64]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        squeeze_5: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2]);  getitem_2 = None
        mul_10: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000001220703274);  squeeze_5 = None
        mul_11: "f32[64]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
        mul_12: "f32[64]" = torch.ops.aten.mul.Tensor(primals_12, 0.9)
        add_8: "f32[64]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
        mul_13: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_2);  mul_7 = unsqueeze_2 = None
        unsqueeze_3: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1)
        add_9: "f32[500, 64, 16384]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_3);  mul_13 = unsqueeze_3 = None
        mul_14: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(add_9, 0.5)
        mul_15: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(add_9, 0.7071067811865476);  add_9 = None
        erf: "f32[500, 64, 16384]" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
        add_10: "f32[500, 64, 16384]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_16: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(mul_14, add_10);  mul_14 = add_10 = None
        convolution_1: "f32[500, 64, 16384]" = torch.ops.aten.convolution.default(mul_16, primals_15, primals_16, [1], [0], [1], False, [0], 1);  primals_16 = None
        add_11: "i64[]" = torch.ops.aten.add.Tensor(primals_17, 1)
        var_mean_2 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2], correction = 0, keepdim = True)
        getitem_4: "f32[1, 64, 1]" = var_mean_2[0]
        getitem_5: "f32[1, 64, 1]" = var_mean_2[1];  var_mean_2 = None
        add_12: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
        rsqrt_2: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_2: "f32[500, 64, 16384]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_5)
        mul_17: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        squeeze_6: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2])
        mul_18: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1);  squeeze_6 = None
        mul_19: "f32[64]" = torch.ops.aten.mul.Tensor(primals_18, 0.9)
        add_13: "f32[64]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        squeeze_8: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2]);  getitem_4 = None
        mul_20: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000001220703274);  squeeze_8 = None
        mul_21: "f32[64]" = torch.ops.aten.mul.Tensor(mul_20, 0.1);  mul_20 = None
        mul_22: "f32[64]" = torch.ops.aten.mul.Tensor(primals_19, 0.9)
        add_14: "f32[64]" = torch.ops.aten.add.Tensor(mul_21, mul_22);  mul_21 = mul_22 = None
        unsqueeze_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1)
        mul_23: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(mul_17, unsqueeze_4);  mul_17 = unsqueeze_4 = None
        unsqueeze_5: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1)
        add_15: "f32[500, 64, 16384]" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_5);  mul_23 = unsqueeze_5 = None
        mul_24: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(add_15, 0.5)
        mul_25: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(add_15, 0.7071067811865476);  add_15 = None
        erf_1: "f32[500, 64, 16384]" = torch.ops.aten.erf.default(mul_25);  mul_25 = None
        add_16: "f32[500, 64, 16384]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_26: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(mul_24, add_16);  mul_24 = add_16 = None
        convolution_2: "f32[500, 64, 16384]" = torch.ops.aten.convolution.default(mul_26, primals_22, primals_23, [1], [0], [1], False, [0], 1);  primals_23 = None
        add_17: "i64[]" = torch.ops.aten.add.Tensor(primals_24, 1)
        var_mean_3 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2], correction = 0, keepdim = True)
        getitem_6: "f32[1, 64, 1]" = var_mean_3[0]
        getitem_7: "f32[1, 64, 1]" = var_mean_3[1];  var_mean_3 = None
        add_18: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
        rsqrt_3: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_3: "f32[500, 64, 16384]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_7)
        mul_27: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        squeeze_9: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2])
        mul_28: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1);  squeeze_9 = None
        mul_29: "f32[64]" = torch.ops.aten.mul.Tensor(primals_25, 0.9)
        add_19: "f32[64]" = torch.ops.aten.add.Tensor(mul_28, mul_29);  mul_28 = mul_29 = None
        squeeze_11: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2]);  getitem_6 = None
        mul_30: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0000001220703274);  squeeze_11 = None
        mul_31: "f32[64]" = torch.ops.aten.mul.Tensor(mul_30, 0.1);  mul_30 = None
        mul_32: "f32[64]" = torch.ops.aten.mul.Tensor(primals_26, 0.9)
        add_20: "f32[64]" = torch.ops.aten.add.Tensor(mul_31, mul_32);  mul_31 = mul_32 = None
        unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1)
        mul_33: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(mul_27, unsqueeze_6);  mul_27 = unsqueeze_6 = None
        unsqueeze_7: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1)
        add_21: "f32[500, 64, 16384]" = torch.ops.aten.add.Tensor(mul_33, unsqueeze_7);  mul_33 = unsqueeze_7 = None
        mul_34: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(add_21, 0.5)
        mul_35: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(add_21, 0.7071067811865476);  add_21 = None
        erf_2: "f32[500, 64, 16384]" = torch.ops.aten.erf.default(mul_35);  mul_35 = None
        add_22: "f32[500, 64, 16384]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_36: "f32[500, 64, 16384]" = torch.ops.aten.mul.Tensor(mul_34, add_22);  mul_34 = add_22 = None
        convolution_3: "f32[500, 8, 16384]" = torch.ops.aten.convolution.default(mul_36, primals_29, primals_30, [1], [0], [1], False, [0], 1);  primals_30 = None
        add_23: "i64[]" = torch.ops.aten.add.Tensor(primals_31, 1)
        var_mean_4 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2], correction = 0, keepdim = True)
        getitem_8: "f32[1, 8, 1]" = var_mean_4[0]
        getitem_9: "f32[1, 8, 1]" = var_mean_4[1];  var_mean_4 = None
        add_24: "f32[1, 8, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
        rsqrt_4: "f32[1, 8, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        sub_4: "f32[500, 8, 16384]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_9)
        mul_37: "f32[500, 8, 16384]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        squeeze_12: "f32[8]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2])
        mul_38: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1);  squeeze_12 = None
        mul_39: "f32[8]" = torch.ops.aten.mul.Tensor(primals_32, 0.9)
        add_25: "f32[8]" = torch.ops.aten.add.Tensor(mul_38, mul_39);  mul_38 = mul_39 = None
        squeeze_14: "f32[8]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2]);  getitem_8 = None
        mul_40: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000001220703274);  squeeze_14 = None
        mul_41: "f32[8]" = torch.ops.aten.mul.Tensor(mul_40, 0.1);  mul_40 = None
        mul_42: "f32[8]" = torch.ops.aten.mul.Tensor(primals_33, 0.9)
        add_26: "f32[8]" = torch.ops.aten.add.Tensor(mul_41, mul_42);  mul_41 = mul_42 = None
        unsqueeze_8: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1)
        mul_43: "f32[500, 8, 16384]" = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_8);  mul_37 = unsqueeze_8 = None
        unsqueeze_9: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
        add_27: "f32[500, 8, 16384]" = torch.ops.aten.add.Tensor(mul_43, unsqueeze_9);  mul_43 = unsqueeze_9 = None
        mul_44: "f32[500, 8, 16384]" = torch.ops.aten.mul.Tensor(add_27, 0.5)
        mul_45: "f32[500, 8, 16384]" = torch.ops.aten.mul.Tensor(add_27, 0.7071067811865476);  add_27 = None
        erf_3: "f32[500, 8, 16384]" = torch.ops.aten.erf.default(mul_45);  mul_45 = None
        add_28: "f32[500, 8, 16384]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_46: "f32[500, 8, 16384]" = torch.ops.aten.mul.Tensor(mul_44, add_28);  mul_44 = add_28 = None
        
        # File: /app/src/models/processor.py:130 in forward, code: U = U.view(B * U.shape[1], N, N)  # (B * num_heads, N, N)
        view_1: "f32[4000, 128, 128]" = torch.ops.aten.view.default(mul_46, [4000, 128, 128]);  mul_46 = None
        
        # File: /app/src/models/lorentz_part.py:54 in forward, code: x = x.view(B, N, 1, F)
        view_2: "f32[500, 128, 1, 16]" = torch.ops.aten.view.default(primals_1, [500, 128, 1, 16]);  primals_1 = None
        
        # File: /app/src/models/processor.py:129 in forward, code: U = self.embed(U)  # (B, num_heads, N * N)
        unsqueeze_42: "f32[1, 4]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_43: "f32[1, 4, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, 2);  unsqueeze_42 = None
        
        # File: /app/src/models/lorentz_part.py:272 in forward, code: x, U = self.processor(x)
        copy_: "i64[]" = torch.ops.aten.copy_.default(primals_3, add);  primals_3 = add = copy_ = None
        copy__1: "f32[4]" = torch.ops.aten.copy_.default(primals_4, add_2);  primals_4 = add_2 = copy__1 = None
        copy__2: "f32[4]" = torch.ops.aten.copy_.default(primals_5, add_3);  primals_5 = add_3 = copy__2 = None
        copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_10, add_5);  primals_10 = add_5 = copy__3 = None
        copy__4: "f32[64]" = torch.ops.aten.copy_.default(primals_11, add_7);  primals_11 = add_7 = copy__4 = None
        copy__5: "f32[64]" = torch.ops.aten.copy_.default(primals_12, add_8);  primals_12 = add_8 = copy__5 = None
        copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_17, add_11);  primals_17 = add_11 = copy__6 = None
        copy__7: "f32[64]" = torch.ops.aten.copy_.default(primals_18, add_13);  primals_18 = add_13 = copy__7 = None
        copy__8: "f32[64]" = torch.ops.aten.copy_.default(primals_19, add_14);  primals_19 = add_14 = copy__8 = None
        copy__9: "i64[]" = torch.ops.aten.copy_.default(primals_24, add_17);  primals_24 = add_17 = copy__9 = None
        copy__10: "f32[64]" = torch.ops.aten.copy_.default(primals_25, add_19);  primals_25 = add_19 = copy__10 = None
        copy__11: "f32[64]" = torch.ops.aten.copy_.default(primals_26, add_20);  primals_26 = add_20 = copy__11 = None
        copy__12: "i64[]" = torch.ops.aten.copy_.default(primals_31, add_23);  primals_31 = add_23 = copy__12 = None
        copy__13: "f32[8]" = torch.ops.aten.copy_.default(primals_32, add_25);  primals_32 = add_25 = copy__13 = None
        copy__14: "f32[8]" = torch.ops.aten.copy_.default(primals_33, add_26);  primals_33 = add_26 = copy__14 = None
        return (view_2, view_1, primals_2, primals_8, primals_13, primals_14, primals_15, primals_20, primals_21, primals_22, primals_27, primals_28, primals_29, primals_34, primals_35, squeeze_1, add_4, convolution, getitem_3, rsqrt_1, mul_16, convolution_1, getitem_5, rsqrt_2, mul_26, convolution_2, getitem_7, rsqrt_3, mul_36, convolution_3, getitem_9, rsqrt_4, unsqueeze_43)
        