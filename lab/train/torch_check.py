import torch
print('TORCH-OK', torch.__version__, 'cuda:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    x = torch.randn(4096, 4096, device='cuda')
    y = x @ x
    torch.cuda.synchronize()
    print('matmul OK', y.sum().item())
