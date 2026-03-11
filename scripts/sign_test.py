import signatory
import torch

batch, stream, channels = 1, 10, 2
depth = 4
path = torch.rand(batch, stream, channels)
signature = signatory.signature(path, depth)
# signature is a PyTorch tensor
print(signature)
print(signature.shape)
