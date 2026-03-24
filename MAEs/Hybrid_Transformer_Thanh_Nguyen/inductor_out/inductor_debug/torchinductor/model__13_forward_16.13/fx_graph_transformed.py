class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[1, 1]", primals_2: "f32[500, 128, 1, 16]"):
        # File: /usr/local/lib/python3.12/site-packages/lgatr/interface/scalar.py:21 in embed_scalar, code: non_scalar_components = torch.zeros(
        full_default: "f32[1, 15]" = torch.ops.aten.full.default([1, 15], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # File: /usr/local/lib/python3.12/site-packages/lgatr/interface/scalar.py:24 in embed_scalar, code: embedding = torch.cat((scalars, non_scalar_components), dim=-1)
        cat: "f32[1, 16]" = torch.ops.aten.cat.default([primals_1, full_default], -1);  primals_1 = full_default = None
        
        # File: /usr/local/lib/python3.12/site-packages/lgatr/layers/linear.py:159 in torch_dynamo_resume_in_forward_at_155, code: outputs_mv = outputs_mv + bias
        add: "f32[500, 128, 1, 16]" = torch.ops.aten.add.Tensor(primals_2, cat);  primals_2 = cat = None
        return (add,)
        