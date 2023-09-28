import os, json

import torch
import torch.optim as optim

from tqdm import tqdm

from .dataset import AnchorNotFoundError
from ..training.classifier import BinaryNet

def save_model(model_dir, model):
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  torch.save(model.state_dict(), os.path.join(model_dir, 'model.ckpt'))
  print("Saved the model to", model_dir)

def load_model(model_dir, prog_dict, hparams, device):
  assert os.path.exists(model_dir)
  model = BinaryNet(prog_dict, hparams['embed_dim'], hparams['transformer_layers'],
                    hparams['dim_feedforward'], hparams['nhead'], hparams['dropout'])
  if not os.path.exists(os.path.join(model_dir, 'model.ckpt')):
    print("Loaded a fresh model.")
    return model
  model = model.to(device)
  model.load_state_dict(torch.load(os.path.join(model_dir, 'model.ckpt'), map_location=device))
  print("Loaded an existing model from", model_dir)
  return model

def load_config(model_dir):
  with open(os.path.join(model_dir, 'hparams.json')) as f:
    hparams = json.load(f)
  with open(os.path.join(model_dir, 'biases.json')) as f:
    biases = json.load(f)
  return hparams, biases

def train_loop(device, model, model_dir, fn_db_train, fn_db_dev, fn_eval, nn_arg_constructor, hparams):
  model = model.to(device)
  train_loader = fn_db_train.get_train_loader()
  dev_loader = fn_db_dev.get_test_loader()

  optimizer = optim.Adam(model.parameters(), lr=hparams['learning_rate'])

  train_iter = iter(train_loader)
  best_metric = -1
  print("Training in progress...")
  for epoch in range(hparams['num_epochs']):
    print("Epoch", epoch)
    model.train()
    for _ in tqdm(range(hparams['iter_per_epoch'])):
      try:
        nn_args = next(train_iter)
      except StopIteration:
        train_iter = iter(train_loader)
        try:
          nn_args = next(train_iter)
        except AnchorNotFoundError:
          continue
      except AnchorNotFoundError:
        continue

      optimizer.zero_grad()
      nn_args = nn_arg_constructor(nn_args, device)
      loss = model(**nn_args)
      loss.backward()

      if hparams['grad_clip'] > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=hparams['grad_clip'])

      optimizer.step()

    if fn_eval is not None:
      auc = fn_eval(model, 'dev', dev_loader, device)
      if auc > best_metric:
        best_metric = auc
        save_model(model_dir, model)
