class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[500, 128, 1, 16]"):
        # File: /usr/local/lib/python3.12/site-packages/lgatr/layers/linear.py:159 in torch_dynamo_resume_in_forward_at_155, code: outputs_mv = outputs_mv + bias
        sum_1: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0, 1], True)
        view: "f32[1, 16]" = torch.ops.aten.reshape.default(sum_1, [1, 16]);  sum_1 = None
        
        # File: /usr/local/lib/python3.12/site-packages/lgatr/interface/scalar.py:24 in embed_scalar, code: embedding = torch.cat((scalars, non_scalar_components), dim=-1)
        slice_1: "f32[1, 1]" = torch.ops.aten.slice.Tensor(view, 1, 0, 1);  view = None
        return (slice_1, tangents_1)
        