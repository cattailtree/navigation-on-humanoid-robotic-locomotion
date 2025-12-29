import torch
model = torch.jit.load("policy.pt")
print(model)