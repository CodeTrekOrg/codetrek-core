import torch
import torch.nn as nn

from .utils import InitUtils, gnn_spmm
from .transformer import TransformerEncoderLayer, TransformerEncoder
from .mlp import MultilayerPerceptron
from ..configs import cmd_args

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
    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
    encoder_norm = nn.LayerNorm(d_model)
    self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
    self.walk_repr = walk_repr
    
  def forward(self, walk_token_embed):
      assert walk_token_embed.dim() == 4 # L x N x B x d_model
      L, N, B, _ = walk_token_embed.shape
      walk_token_embed = walk_token_embed.view(L, -1, self.d_model)

      memory, attn_weights = self.encoder(walk_token_embed)
      memory = memory.view(L, N, B, -1)
      walk_repr = torch.mean(memory, dim=0)
      
      return walk_repr, attn_weights

class ProgramDeepset(nn.Module):
  def __init__(self, embed_dim, dropout=0.0):
    super(ProgramDeepset, self).__init__()
    self.mlp = MultilayerPerceptron(embed_dim, [2 * embed_dim, embed_dim], dropout=dropout)

  def forward(self, walk_repr):
    walk_hidden = self.mlp(walk_repr)
    prog_repr, max_idx = torch.max(walk_hidden, dim=0)

    if cmd_args.desc_gen:
      counts_per_row = torch.zeros(walk_hidden.shape[1], walk_hidden.shape[0])
      for item in torch.arange(walk_hidden.shape[0]):
          counts_per_row[:, item] = torch.sum((max_idx == item), dim=1)
      _, sorted_walks = torch.topk(counts_per_row, dim=1, sorted=True, largest=True, k=walk_hidden.shape[0])
      return prog_repr, sorted_walks

    return prog_repr


class WalkSetEmbed(nn.Module):
  def __init__(self, prog_dict):
    super(WalkSetEmbed, self).__init__()
    self.tok_encoding = TokenEncoder(prog_dict, cmd_args.embed_dim, cmd_args.dropout)
    self.walk_encoding = WalkEncoder(cmd_args.embed_dim, cmd_args.nhead, cmd_args.transformer_layers,  cmd_args.dim_feedforward, cmd_args.dropout)
    self.prob_encoding = ProgramDeepset(cmd_args.embed_dim, cmd_args.dropout)

  def forward(self, node_idx, edge_idx, node_val_mat):
    seq_tok_embed = self.tok_encoding(node_idx, edge_idx, node_val_mat)
    walk_repr, attn_weights = self.walk_encoding(seq_tok_embed)
    if cmd_args.desc_gen:
      prog_repr, sorted_walks = self.prob_encoding(walk_repr)
    else:
      prog_repr, sorted_walks = self.prob_encoding(walk_repr), None

    selected_walks = None
    if sorted_walks is not None:
      nodes_in_walk = (seq_tok_embed.shape[0]+1)//3
      selected_walks = []
      attn_of_interest = attn_weights[0].reshape(seq_tok_embed.shape[1], seq_tok_embed.shape[2],
                                                 attn_weights[0].shape[-1], -1)
      agg_attn_weights = torch.sum(attn_of_interest, dim=-2)[:,:, :nodes_in_walk]
      for sample_idx in range(agg_attn_weights.shape[1]):
        selected_walks.append((sorted_walks[sample_idx],
                               torch.topk(agg_attn_weights[sorted_walks[sample_idx]][:,sample_idx,:],
                                          dim=-1, k=nodes_in_walk, sorted=True, largest=True)))
    return prog_repr, selected_walks
