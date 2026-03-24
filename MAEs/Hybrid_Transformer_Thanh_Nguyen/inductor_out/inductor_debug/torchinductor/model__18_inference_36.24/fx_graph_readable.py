class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[500, 128, 128, 4]", arg1_1: "b8[500, 128, 128]", arg2_1: "Sym(s38)", arg3_1: "f32[s38, 4]"):
        # File: /app/src/models/processor.py:76 in torch_dynamo_resume_in__get_interaction_at_76, code: U[valid_pairs] = U_vals[valid_pairs]
        index_put: "f32[500, 128, 128, 4]" = torch.ops.aten.index_put.default(arg0_1, [arg1_1], arg3_1);  arg1_1 = arg3_1 = None
        
        # File: /app/src/models/processor.py:79 in torch_dynamo_resume_in__get_interaction_at_76, code: idx = torch.arange(U.size(1), device=U.device)
        iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        
        # File: /app/src/models/processor.py:80 in torch_dynamo_resume_in__get_interaction_at_76, code: U[:, idx, idx, :] = 0
        _tensor_constant0: "f32[]" = self._tensor_constant0;  _tensor_constant0 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        index_put_1: "f32[500, 128, 128, 4]" = torch.ops.aten.index_put.default(index_put, [None, iota, iota], full_default);  index_put = iota = full_default = None
        copy_: "f32[500, 128, 128, 4]" = torch.ops.aten.copy_.default(arg0_1, index_put_1);  arg0_1 = index_put_1 = None
        return (copy_,)
        