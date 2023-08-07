import torch

def get_torch_device(device_id):
    if device_id >= 0 and torch.cuda.is_available():
        return 'cuda:{}'.format(device_id)
    return 'cpu'
