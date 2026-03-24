class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[500, 128, 128]"):
        # File: /app/src/models/processor.py:44 in torch_dynamo_resume_in__get_interaction_at_42, code: if torch.isnan(kT).any():
        isnan: "b8[500, 128, 128]" = torch.ops.aten.isnan.default(arg0_1);  arg0_1 = None
        any_1: "b8[]" = torch.ops.aten.any.default(isnan);  isnan = None
        return (any_1,)
        