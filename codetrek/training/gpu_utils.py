import torch

def get_torch_device(device_id):
    if device_id >= 0:
        if torch.cuda.is_available():
            return 'cuda:{}'.format(device_id)
        if torch.backends.mps.is_available():
            return 'mps:{}'.format(device_id)
    return 'cpu'
