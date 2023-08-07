import torch
import torch.nn as nn

from .utils import InitUtils, gnn_spmm

class LearnedPositionalEncoder(nn.Module):
  def __init__(self, d_model, dropout, max_len=50):
    super(LearnedPositionalEncoder, self).__init__()
    self.dropout = nn.Dropout(p=dropout)
    self.pos_embed = nn.Parameter(torch.zeros(max_len, d_model))
    InitUtils._param_init(self.pos_embed)
    
  def forward(self, x):
    pe = self.pos_embed[:x.size(0), :]
    for _ in range(pe.dim(), x.dim()):
      pe = pe.unsqueeze(1)
    x = x + pe
    return self.dropout(x)


class TokenEncoder(nn.Module):
  def __init__(self, prog_dict, embed_dim, dropout=0.0):
    super(TokenEncoder, self).__init__()
    self.pg_dict = prog_dict

    self.node_pos_encoding = LearnedPositionalEncoder(d_model=embed_dim, dropout=dropout)
    self.edge_pos_encoding = LearnedPositionalEncoder(d_model=embed_dim, dropout=dropout)
    self.val_pos_encoding = LearnedPositionalEncoder(d_model=embed_dim, dropout=dropout)

    self.node_embed = nn.Embedding(len(self.pg_dict.node_types), embed_dim)
    self.edge_embed = nn.Embedding(len(self.pg_dict.edge_types), embed_dim)
    self.val_embed = nn.Parameter(torch.Tensor(len(self.pg_dict.token_vocab), embed_dim))

  def forward(self, node_idx, edge_idx, node_val_mat):
    node_embed = self.node_embed(node_idx)
    node_embed = self.node_pos_encoding(node_embed)

    edge_embed = self.edge_embed(edge_idx)
    edge_embed = self.edge_pos_encoding(edge_embed)
    node_val_embed = gnn_spmm(node_val_mat, self.val_embed).view(node_embed.shape)
    val_embed = self.val_pos_encoding(node_val_embed)

    return torch.cat((node_embed, edge_embed, val_embed), dim=0)

class WalkEncoder(nn.Module):
  def __init__(self, d_model=256, nhead=4, num_encoder_layers=3,
               dim_feedforward=512, dropout=0.0, activation='relu', walk_repr='mean'):
    super(WalkEncoder, self).__init__()
    self.d_model = d_model
    encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
    encoder_norm = nn.LayerNorm(d_model)
    self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
    self.walk_repr = walk_repr
    
  def forward(self, walk_token_embed):
      assert walk_token_embed.dim() == 4 # L x N x B x d_model
      L, N, B, _ = walk_token_embed.shape
      walk_token_embed = walk_token_embed.view(L, -1, self.d_model)
      
      memory = self.encoder(walk_token_embed)
      memory = memory.view(L, N, B, -1)
      walk_repr = torch.mean(memory, dim=0)
      
      return walk_repr

NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "sigmoid": nn.Sigmoid(),
    "elu": nn.ELU(),
    "square": lambda x: x**2,
    "identity": lambda x: x,
}

class MLP(nn.Module):
  def __init__(self, input_dim, hidden_dims, nonlinearity='elu', act_last=None, bn=False, dropout=-1):
    super(MLP, self).__init__()
    self.act_last = act_last
    self.nonlinearity = nonlinearity
    self.input_dim = input_dim
    self.bn = bn
    if isinstance(hidden_dims, str):
      hidden_dims = list(map(int, hidden_dims.split("-")))
    assert len(hidden_dims)
    hidden_dims = [input_dim] + hidden_dims
    self.output_size = hidden_dims[-1]
    
    list_layers = []
    for i in range(1, len(hidden_dims)):
      list_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
      if i + 1 < len(hidden_dims):  # not the last layer
        if self.bn:
          bnorm_layer = nn.BatchNorm1d(hidden_dims[i])
          list_layers.append(bnorm_layer)
        list_layers.append(NONLINEARITIES[self.nonlinearity])
        if dropout > 0:
          list_layers.append(nn.Dropout(dropout))
      else:
        if act_last is not None:
          list_layers.append(NONLINEARITIES[act_last])
    
    self.main = nn.Sequential(*list_layers)
    
  def forward(self, x):
    return self.main(x)


class ProgramDeepset(nn.Module):
  def __init__(self, embed_dim, dropout=0.0):
    super(ProgramDeepset, self).__init__()
    self.mlp = MLP(embed_dim, [2 * embed_dim, embed_dim], dropout=dropout)
  
  def forward(self, walk_repr, get_before_agg=False):
    walk_hidden = self.mlp(walk_repr)
    prog_repr, _ = torch.max(walk_hidden, dim=0)
    if get_before_agg:
      return prog_repr, walk_hidden
    return prog_repr

class WalkSetEmbed(nn.Module):
  def __init__(self, args, prog_dict):
    super(WalkSetEmbed, self).__init__()
    self.tok_encoding = TokenEncoder(prog_dict, args.embed_dim, args.dropout)
    self.walk_encoding = WalkEncoder(args.embed_dim, args.nhead, args.transformer_layers,  args.dim_feedforward, args.dropout)
    self.prob_encoding = ProgramDeepset(args.embed_dim, args.dropout)
    
  def forward(self, node_idx, edge_idx, node_val_mat, get_before_agg=False):
    seq_tok_embed = self.tok_encoding(node_idx, edge_idx, node_val_mat)
    walk_repr = self.walk_encoding(seq_tok_embed)
    prog_repr = self.prob_encoding(walk_repr, get_before_agg)
    return prog_repr
