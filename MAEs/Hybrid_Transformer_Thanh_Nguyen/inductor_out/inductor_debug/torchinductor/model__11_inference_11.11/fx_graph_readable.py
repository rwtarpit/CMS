class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[500, 128, 4]"):
        # File: /app/src/models/processor.py:100 in torch_dynamo_resume_in_forward_at_88, code: x = x.view(B, N, 1, F)  # for compatibility with the EquiLinear layer
        view: "f32[500, 128, 1, 4]" = torch.ops.aten.view.default(arg0_1, [500, 128, 1, 4]);  arg0_1 = None
        
        # File: /usr/local/lib/python3.12/site-packages/lgatr/interface/vector.py:21 in embed_vector, code: multivector = torch.zeros(
        full: "f32[500, 128, 1, 16]" = torch.ops.aten.full.default([500, 128, 1, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # File: /usr/local/lib/python3.12/site-packages/lgatr/interface/vector.py:26 in embed_vector, code: multivector[..., 1:5] = vector
        slice_1: "f32[500, 128, 1, 4]" = torch.ops.aten.slice.Tensor(full, 3, 1, 5)
        copy: "f32[500, 128, 1, 4]" = torch.ops.aten.copy.default(slice_1, view);  slice_1 = view = None
        slice_scatter: "f32[500, 128, 1, 16]" = torch.ops.aten.slice_scatter.default(full, copy, 3, 1, 5);  full = copy = None
        
        # File: /app/src/models/processor.py:88 in forward, code: U = self._get_interaction(x)  # (B, N, N, 4)
        view_2: "f32[500, 128, 16]" = torch.ops.aten.view.default(slice_scatter, [500, 128, 16]);  slice_scatter = None
        return (view_2,)
        