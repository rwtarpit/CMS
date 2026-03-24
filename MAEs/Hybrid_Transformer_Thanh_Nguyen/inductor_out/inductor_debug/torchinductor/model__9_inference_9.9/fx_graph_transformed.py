class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[500, 128, 128]", arg1_1: "f32[500, 128, 128]", arg2_1: "f32[500, 128, 128]", arg3_1: "f32[500, 128, 128]", arg4_1: "b8[500, 128]"):
        # File: /app/src/models/processor.py:67 in torch_dynamo_resume_in__get_interaction_at_63, code: U_vals = torch.stack((ln_delta, ln_kT, ln_z, ln_m2), dim=-1)  # (B, N, N, 4)
        unsqueeze: "f32[500, 128, 128, 1]" = torch.ops.aten.unsqueeze.default(arg0_1, 3);  arg0_1 = None
        unsqueeze_1: "f32[500, 128, 128, 1]" = torch.ops.aten.unsqueeze.default(arg1_1, 3);  arg1_1 = None
        unsqueeze_2: "f32[500, 128, 128, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, 3);  arg2_1 = None
        unsqueeze_3: "f32[500, 128, 128, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, 3);  arg3_1 = None
        cat: "f32[500, 128, 128, 4]" = torch.ops.aten.cat.default([unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3], -1);  unsqueeze = unsqueeze_1 = unsqueeze_2 = unsqueeze_3 = None
        
        # File: /app/src/models/processor.py:73 in torch_dynamo_resume_in__get_interaction_at_63, code: valid_pairs = mask.unsqueeze(2) & mask.unsqueeze(1)
        unsqueeze_4: "b8[500, 128, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, 2)
        unsqueeze_5: "b8[500, 1, 128]" = torch.ops.aten.unsqueeze.default(arg4_1, 1);  arg4_1 = None
        bitwise_and: "b8[500, 128, 128]" = torch.ops.aten.bitwise_and.Tensor(unsqueeze_4, unsqueeze_5);  unsqueeze_4 = unsqueeze_5 = None
        
        # File: /app/src/models/processor.py:70 in torch_dynamo_resume_in__get_interaction_at_63, code: U = torch.full_like(U_vals, fill_value=-1e9)
        full_default: "f32[500, 128, 128, 4]" = torch.ops.aten.full.default([500, 128, 128, 4], -1000000000.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        return (cat, bitwise_and, full_default)
        