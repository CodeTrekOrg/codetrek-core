import os

import torch
import torch.optim as optim

from tqdm import tqdm

from codetrek.training.dataset import AnchorNotFoundError

def train_loop(args, device, prog_dict, model, fn_db_train, fn_db_dev, fn_eval, nn_arg_constructor):
  model = model.to(device)
  train_loader = fn_db_train.get_train_loader(args)
  dev_loader = fn_db_dev.get_test_loader(args)

  optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

  train_iter = iter(train_loader)
  best_metric = -1
  print("Training in progress...")
  for epoch in range(args.num_epochs):
    print("Epoch", epoch)
    model.train()
    for _ in tqdm(range(args.iter_per_epoch)):
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

      if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

      optimizer.step()

    if fn_eval is not None:
      auc = fn_eval(model, 'dev', dev_loader, device)
      if auc > best_metric:
        best_metric = auc
        torch.save(model.state_dict(), os.path.join(args.data_dir, args.model_dump))
        print("Saved a better model.")
