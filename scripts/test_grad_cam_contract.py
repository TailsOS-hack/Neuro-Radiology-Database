#!/usr/bin/env python3
"""Offline analytical Grad-CAM contract test (not a clinical validation)."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from src.grad_cam import GradCAM

model=nn.Sequential(nn.Conv2d(1,1,1,bias=False),nn.AdaptiveAvgPool2d(1),nn.Flatten())
with torch.no_grad(): model[0].weight.fill_(1)
model.train()
x=torch.arange(16,dtype=torch.float32).reshape(1,1,4,4)
expected=x[0,0].numpy()/15
for _ in range(3):
    with torch.no_grad(): result=GradCAM(model,model[0]).generate(x,0)
    np.testing.assert_allclose(result,expected,atol=1e-7)
    assert model.training
    assert not model[0]._forward_hooks and not model[0]._backward_hooks
assert GradCAM(model,model[0]).generate(x,3) is None
assert GradCAM(model,model[0]).generate(torch.zeros_like(x),0) is None
assert not model[0]._forward_hooks and not model[0]._backward_hooks
assert model.training
print('PASS: analytical CAM, repeatability, no_grad override, mode restoration, invalid target, flat map and hook cleanup')
