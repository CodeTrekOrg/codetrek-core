import pickle
import torch
import torch.nn as nn

from .encoder import WalkSetEmbed
from ..configs import cmd_args

class BinaryNet(WalkSetEmbed):
  def __init__(self, prog_dict):
    super(BinaryNet, self).__init__(prog_dict)
    self.out_classifier = nn.Linear(cmd_args.embed_dim, 1)
    
  def forward(self, node_idx, edge_idx, *, node_val_mat=None, label=None):
    prog_repr = super(BinaryNet, self).forward(node_idx, edge_idx, node_val_mat)
    if cmd_args.desc_gen:
      prog_repr, sorted_walks = prog_repr
      with open(f'{cmd_args.output_dir}/.walks/sorted_walks.pkl', 'ab') as f:
        pickle.dump(sorted_walks, f)
    logits = self.out_classifier(prog_repr)
    prob = torch.sigmoid(logits)
    if label is not None:
      label = label.to(prob).view(prob.shape)
      loss = -label * torch.log(prob + 1e-18) - (1 - label) * torch.log(1 - prob + 1e-18)
      return torch.mean(loss)
    return prob
