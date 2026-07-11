import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
x = torch.randn(2048, 2048)
y = x @ x
print('Matmul test OK, norm:', y.norm().item())
print('CPU threads:', torch.get_num_threads())
