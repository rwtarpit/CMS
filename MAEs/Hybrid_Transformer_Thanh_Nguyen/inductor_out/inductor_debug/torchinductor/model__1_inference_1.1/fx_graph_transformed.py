class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[500, 128, 4]"):
        # File: /app/src/models/processor.py:18 in _get_interaction, code: eta = x[..., 1]
        select_2: "f32[500, 128]" = torch.ops.aten.select.int(arg0_1, 2, 1)
        
        # File: /app/src/models/processor.py:30 in _get_interaction, code: eta_diff = eta.unsqueeze(2) - eta.unsqueeze(1)
        unsqueeze_3: "f32[500, 128, 1]" = torch.ops.aten.unsqueeze.default(select_2, 2)
        unsqueeze_4: "f32[500, 1, 128]" = torch.ops.aten.unsqueeze.default(select_2, 1)
        sub: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(unsqueeze_3, unsqueeze_4);  unsqueeze_3 = unsqueeze_4 = None
        
        # File: /app/src/models/processor.py:37 in _get_interaction, code: delta = torch.sqrt(eta_diff**2 + phi_diff**2)
        pow_1: "f32[500, 128, 128]" = torch.ops.aten.pow.Tensor_Scalar(sub, 2);  sub = None
        
        # File: /app/src/models/processor.py:19 in _get_interaction, code: phi = x[..., 2]
        select_3: "f32[500, 128]" = torch.ops.aten.select.int(arg0_1, 2, 2)
        
        # File: /app/src/models/processor.py:31 in _get_interaction, code: phi_diff = ((phi.unsqueeze(2) - phi.unsqueeze(1)) + torch.pi) % (2 * torch.pi) - torch.pi
        unsqueeze_5: "f32[500, 128, 1]" = torch.ops.aten.unsqueeze.default(select_3, 2)
        unsqueeze_6: "f32[500, 1, 128]" = torch.ops.aten.unsqueeze.default(select_3, 1)
        sub_1: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(unsqueeze_5, unsqueeze_6);  unsqueeze_5 = unsqueeze_6 = None
        add: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(sub_1, 3.141592653589793);  sub_1 = None
        remainder: "f32[500, 128, 128]" = torch.ops.aten.remainder.Scalar(add, 6.283185307179586);  add = None
        sub_2: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(remainder, 3.141592653589793);  remainder = None
        
        # File: /app/src/models/processor.py:37 in _get_interaction, code: delta = torch.sqrt(eta_diff**2 + phi_diff**2)
        pow_2: "f32[500, 128, 128]" = torch.ops.aten.pow.Tensor_Scalar(sub_2, 2);  sub_2 = None
        add_4: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(pow_1, pow_2);  pow_1 = pow_2 = None
        sqrt: "f32[500, 128, 128]" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
        
        # File: /app/src/models/processor.py:42 in _get_interaction, code: if torch.isnan(delta).any():
        isnan: "b8[500, 128, 128]" = torch.ops.aten.isnan.default(sqrt)
        any_1: "b8[]" = torch.ops.aten.any.default(isnan);  isnan = None
        
        # File: /app/src/models/processor.py:14 in _get_interaction, code: mask = x[..., 3] > 0  # (B, N)
        select: "f32[500, 128]" = torch.ops.aten.select.int(arg0_1, 2, 3)
        gt: "b8[500, 128]" = torch.ops.aten.gt.Scalar(select, 0);  select = None
        
        # File: /app/src/models/processor.py:17 in _get_interaction, code: pT = x[..., 0]
        select_1: "f32[500, 128]" = torch.ops.aten.select.int(arg0_1, 2, 0)
        
        # File: /app/src/models/processor.py:32 in _get_interaction, code: min_pT = torch.minimum(pT.unsqueeze(2), pT.unsqueeze(1))
        unsqueeze_7: "f32[500, 128, 1]" = torch.ops.aten.unsqueeze.default(select_1, 2)
        unsqueeze_8: "f32[500, 1, 128]" = torch.ops.aten.unsqueeze.default(select_1, 1)
        minimum: "f32[500, 128, 128]" = torch.ops.aten.minimum.default(unsqueeze_7, unsqueeze_8);  unsqueeze_7 = unsqueeze_8 = None
        
        # File: /app/src/models/processor.py:38 in _get_interaction, code: kT = min_pT * delta
        mul_3: "f32[500, 128, 128]" = torch.ops.aten.mul.Tensor(minimum, sqrt)
        
        # File: /app/src/models/processor.py:33 in _get_interaction, code: pT_sum = pT.unsqueeze(2) + pT.unsqueeze(1)
        unsqueeze_9: "f32[500, 128, 1]" = torch.ops.aten.unsqueeze.default(select_1, 2)
        unsqueeze_10: "f32[500, 1, 128]" = torch.ops.aten.unsqueeze.default(select_1, 1)
        add_1: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(unsqueeze_9, unsqueeze_10);  unsqueeze_9 = unsqueeze_10 = None
        
        # File: /app/src/models/processor.py:39 in _get_interaction, code: z = min_pT / (pT_sum + eps)
        add_5: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(add_1, 1e-08);  add_1 = None
        div: "f32[500, 128, 128]" = torch.ops.aten.div.Tensor(minimum, add_5);  minimum = add_5 = None
        
        # File: /app/src/models/processor.py:20 in _get_interaction, code: energy = x[..., 3]
        select_4: "f32[500, 128]" = torch.ops.aten.select.int(arg0_1, 2, 3);  arg0_1 = None
        
        # File: /app/src/models/processor.py:34 in _get_interaction, code: energy_sum = energy.unsqueeze(2) + energy.unsqueeze(1)
        unsqueeze_11: "f32[500, 128, 1]" = torch.ops.aten.unsqueeze.default(select_4, 2)
        unsqueeze_12: "f32[500, 1, 128]" = torch.ops.aten.unsqueeze.default(select_4, 1);  select_4 = None
        add_2: "f32[500, 128, 128]" = torch.ops.aten.add.Tensor(unsqueeze_11, unsqueeze_12);  unsqueeze_11 = unsqueeze_12 = None
        
        # File: /app/src/models/processor.py:40 in _get_interaction, code: m2 = energy_sum**2 - momentum_sum.norm(dim=-1)**2
        pow_3: "f32[500, 128, 128]" = torch.ops.aten.pow.Tensor_Scalar(add_2, 2);  add_2 = None
        
        # File: /app/src/models/processor.py:23 in _get_interaction, code: px = pT * torch.cos(phi)
        cos: "f32[500, 128]" = torch.ops.aten.cos.default(select_3)
        mul: "f32[500, 128]" = torch.ops.aten.mul.Tensor(select_1, cos);  cos = None
        
        # File: /app/src/models/processor.py:26 in _get_interaction, code: momentum = torch.stack((px, py, pz), dim=-1)  # (B, N, 3)
        unsqueeze: "f32[500, 128, 1]" = torch.ops.aten.unsqueeze.default(mul, 2);  mul = None
        
        # File: /app/src/models/processor.py:24 in _get_interaction, code: py = pT * torch.sin(phi)
        sin: "f32[500, 128]" = torch.ops.aten.sin.default(select_3);  select_3 = None
        mul_1: "f32[500, 128]" = torch.ops.aten.mul.Tensor(select_1, sin);  sin = None
        
        # File: /app/src/models/processor.py:26 in _get_interaction, code: momentum = torch.stack((px, py, pz), dim=-1)  # (B, N, 3)
        unsqueeze_1: "f32[500, 128, 1]" = torch.ops.aten.unsqueeze.default(mul_1, 2);  mul_1 = None
        
        # File: /app/src/models/processor.py:25 in _get_interaction, code: pz = pT * torch.sinh(eta)
        sinh: "f32[500, 128]" = torch.ops.aten.sinh.default(select_2);  select_2 = None
        mul_2: "f32[500, 128]" = torch.ops.aten.mul.Tensor(select_1, sinh);  select_1 = sinh = None
        
        # File: /app/src/models/processor.py:26 in _get_interaction, code: momentum = torch.stack((px, py, pz), dim=-1)  # (B, N, 3)
        unsqueeze_2: "f32[500, 128, 1]" = torch.ops.aten.unsqueeze.default(mul_2, 2);  mul_2 = None
        cat: "f32[500, 128, 3]" = torch.ops.aten.cat.default([unsqueeze, unsqueeze_1, unsqueeze_2], -1);  unsqueeze = unsqueeze_1 = unsqueeze_2 = None
        
        # File: /app/src/models/processor.py:35 in _get_interaction, code: momentum_sum = momentum.unsqueeze(2) + momentum.unsqueeze(1)
        unsqueeze_13: "f32[500, 128, 1, 3]" = torch.ops.aten.unsqueeze.default(cat, 2)
        unsqueeze_14: "f32[500, 1, 128, 3]" = torch.ops.aten.unsqueeze.default(cat, 1);  cat = None
        add_3: "f32[500, 128, 128, 3]" = torch.ops.aten.add.Tensor(unsqueeze_13, unsqueeze_14);  unsqueeze_13 = unsqueeze_14 = None
        
        # File: /app/src/models/processor.py:40 in _get_interaction, code: m2 = energy_sum**2 - momentum_sum.norm(dim=-1)**2
        pow_4: "f32[500, 128, 128, 3]" = torch.ops.aten.pow.Tensor_Scalar(add_3, 2);  add_3 = None
        sum_1: "f32[500, 128, 128]" = torch.ops.aten.sum.dim_IntList(pow_4, [-1]);  pow_4 = None
        pow_5: "f32[500, 128, 128]" = torch.ops.aten.pow.Tensor_Scalar(sum_1, 0.5);  sum_1 = None
        pow_6: "f32[500, 128, 128]" = torch.ops.aten.pow.Tensor_Scalar(pow_5, 2);  pow_5 = None
        sub_3: "f32[500, 128, 128]" = torch.ops.aten.sub.Tensor(pow_3, pow_6);  pow_3 = pow_6 = None
        return (any_1, gt, sqrt, mul_3, div, sub_3)
        