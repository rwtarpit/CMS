class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[500, 128, 4]"):
        # File: /app/src/models/lorentz_part.py:264 in forward, code: padding_mask = (x[..., 3] == 0).float()  # (B, N)
        select: "f32[500, 128]" = torch.ops.aten.select.int(arg0_1, 2, 3);  arg0_1 = None
        eq: "b8[500, 128]" = torch.ops.aten.eq.Scalar(select, 0);  select = None
        convert_element_type: "f32[500, 128]" = torch.ops.prims.convert_element_type.default(eq, torch.float32);  eq = None
        return (convert_element_type,)
        