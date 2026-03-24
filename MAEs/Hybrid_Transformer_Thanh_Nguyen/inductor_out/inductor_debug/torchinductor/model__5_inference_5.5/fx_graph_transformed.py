class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[500, 128, 128]", arg1_1: "f32[500, 128, 128]", arg2_1: "f32[500, 128, 128]", arg3_1: "f32[500, 128, 128]"):
        # File: /app/src/models/processor.py:52 in torch_dynamo_resume_in__get_interaction_at_48, code: ln_delta = torch.log(torch.clamp(delta, min=eps))
        clamp_min: "f32[500, 128, 128]" = torch.ops.aten.clamp_min.default(arg0_1, 1e-08);  arg0_1 = None
        log: "f32[500, 128, 128]" = torch.ops.aten.log.default(clamp_min);  clamp_min = None
        
        # File: /app/src/models/processor.py:57 in torch_dynamo_resume_in__get_interaction_at_48, code: if torch.isnan(ln_delta).any():
        isnan: "b8[500, 128, 128]" = torch.ops.aten.isnan.default(log)
        any_1: "b8[]" = torch.ops.aten.any.default(isnan);  isnan = None
        
        # File: /app/src/models/processor.py:53 in torch_dynamo_resume_in__get_interaction_at_48, code: ln_kT = torch.log(torch.clamp(kT, min=eps))
        clamp_min_1: "f32[500, 128, 128]" = torch.ops.aten.clamp_min.default(arg1_1, 1e-08);  arg1_1 = None
        log_1: "f32[500, 128, 128]" = torch.ops.aten.log.default(clamp_min_1);  clamp_min_1 = None
        
        # File: /app/src/models/processor.py:54 in torch_dynamo_resume_in__get_interaction_at_48, code: ln_z = torch.log(torch.clamp(z, min=eps))
        clamp_min_2: "f32[500, 128, 128]" = torch.ops.aten.clamp_min.default(arg2_1, 1e-08);  arg2_1 = None
        log_2: "f32[500, 128, 128]" = torch.ops.aten.log.default(clamp_min_2);  clamp_min_2 = None
        
        # File: /app/src/models/processor.py:55 in torch_dynamo_resume_in__get_interaction_at_48, code: ln_m2 = torch.log(torch.clamp(m2, min=eps))
        clamp_min_3: "f32[500, 128, 128]" = torch.ops.aten.clamp_min.default(arg3_1, 1e-08);  arg3_1 = None
        log_3: "f32[500, 128, 128]" = torch.ops.aten.log.default(clamp_min_3);  clamp_min_3 = None
        return (any_1, log, log_1, log_2, log_3)
        