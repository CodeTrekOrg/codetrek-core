import os
import re
import pickle

from tqdm import tqdm

from ..constants import UNK, TOK_PAD

def get_or_add(type_dict, key):
    if key in type_dict:
        return type_dict[key]
    val = len(type_dict)
    type_dict[key] = val
    return val

class ProgDict:
  def __init__(self, data_dir, dict_name):
    with open(os.path.join(data_dir, dict_name), 'rb') as f:
      d = pickle.load(f)
    self.node_types = d['node_types']
    self.edge_types = d['edge_types']
    self.token_vocab = d['token_vocab']
    
  def node_idx(self, node_name):
    if node_name in self.node_types:
      return self.node_types[node_name]
    return self.node_types[UNK]

  def edge_idx(self, edge_name):
    if edge_name in self.edge_types:
      return self.edge_types[edge_name]
    return self.edge_types[UNK]

  @staticmethod
  def tokenize(s):
    return re.split(r'\W+|_', s)

  @staticmethod
  def build_dict(data_dir, dict_name, phases=['train', 'dev', 'test']):
    print("building the dictionary...")
    node_types = {}
    edge_types = {}
    token_vocab = {}
    
    for key in [UNK]:
      get_or_add(node_types, key)
      get_or_add(edge_types, key)
    
    for key in [TOK_PAD, UNK]:
      get_or_add(token_vocab, key)

    for phase in phases:
      files = os.listdir(os.path.join(data_dir, phase))
      for fname in tqdm(files):
        if not fname.startswith("graph_"): continue
        with open(os.path.join(data_dir, phase, fname), 'rb') as f:
          g = pickle.load(f)

        for node in g.nodes:
          if 'label' in g.nodes[node]:
            get_or_add(node_types, g.nodes[node]['label'])
          if 'values' in g.nodes[node]:
            [get_or_add(token_vocab, key) for key in ProgDict.tokenize(g.nodes[node]['values'])]

        for edge in g.edges:
          get_or_add(edge_types, g.edges[edge]['label'])
          
    with open(os.path.join(data_dir, dict_name), 'wb') as f:
      d = {
        'node_types': node_types,
        'edge_types': edge_types,
        'token_vocab': token_vocab
      }
      pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
      
      print("dictionary saved in", os.path.join(data_dir, dict_name), '\nsummary:')
      print('  #node types', len(node_types))
      print('  #edge types', len(edge_types))
      print('  #tokens', len(token_vocab))
