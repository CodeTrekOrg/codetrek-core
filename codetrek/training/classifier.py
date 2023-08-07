import torch
import torch.nn as nn

from codetrek.training.encoder import WalkSetEmbed

class BinaryNet(WalkSetEmbed):
  def __init__(self, args, prog_dict):
    super(BinaryNet, self).__init__(args, prog_dict)
    self.out_classifier = nn.Linear(args.embed_dim, 1)
    
  def forward(self, node_idx, edge_idx, *, node_val_mat=None, label=None):
    prog_repr = super(BinaryNet, self).forward(node_idx, edge_idx, node_val_mat)
    logits = self.out_classifier(prog_repr)
    prob = torch.sigmoid(logits)
    if label is not None:
      label = label.to(prob).view(prob.shape)
      loss = -label * torch.log(prob + 1e-18) - (1 - label) * torch.log(1 - prob + 1e-18)
      return torch.mean(loss)
    return prob
