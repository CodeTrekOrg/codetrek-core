import os
import torch
import json
import random, time
import numpy as np
from tqdm import tqdm

from collections import namedtuple
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader

from ..constants import UNK
from .walker import Walker
from ..preprocessing.graph import GraphBuilder
from .dict import ProgDict

RawData = namedtuple('RawData', ['node_idx', 'edge_idx', 'node_val_idx', 'label'])

def collate_raw_data(list_samples):
  label = []
  max_node_len = 0
  max_edge_len = 0
  min_walks = list_samples[0].node_idx.shape[1]
  max_walks = 0
  for s in list_samples:
    label.append(s.label)
    max_node_len = max(s.node_idx.shape[0], max_node_len)
    max_edge_len = max(s.edge_idx.shape[0], max_edge_len)            
    min_walks = min(s.node_idx.shape[1], min_walks)
    max_walks = max(s.node_idx.shape[1], max_walks)

  full_node_idx = np.zeros((max_node_len, min_walks, len(list_samples)), dtype=np.int16)
  full_edge_idx = np.zeros((max_edge_len, min_walks, len(list_samples)), dtype=np.int16)

  for i, s in enumerate(list_samples):
    node_mat, edge_mat = s.node_idx, s.edge_idx
    full_node_idx[:node_mat.shape[0], :, i] = node_mat
    full_edge_idx[:edge_mat.shape[0], :, i] = edge_mat
    
  full_node_idx = torch.LongTensor(full_node_idx)
  full_edge_idx = torch.LongTensor(full_edge_idx)
  label = torch.LongTensor(label)
  
  _, word_dim = list_samples[0].node_val_idx
  sp_shape = (max_node_len, min_walks, len(list_samples), word_dim)
  list_coos = []
  word_dim = 0
  for i, s in enumerate(list_samples):
    coo, word_dim = s.node_val_idx
    if coo.shape[0]:
      row_ids = (coo[:, 0] * sp_shape[1] + coo[:, 1]) * sp_shape[2] + i
      list_coos.append(np.stack((row_ids, coo[:, 2])))
  if len(list_coos):
    list_coos = torch.LongTensor(np.concatenate(list_coos, axis=1))
    vals = torch.ones((list_coos.shape[1],))
  else:
    list_coos = torch.LongTensor(size=[2, 0])
    vals = torch.ones((0,))
  node_val_mat = (list_coos, vals, (sp_shape[0] * sp_shape[1] * sp_shape[2], sp_shape[3]))
  return full_node_idx, full_edge_idx, node_val_mat, label
    

def make_mat(list_walks, max_n_nodes, max_n_edges):
  mat_node = np.zeros((max_n_nodes, len(list_walks)), dtype=np.int16)
  mat_edge = np.zeros((max_n_edges, len(list_walks)), dtype=np.int16)

  for i, (seq_nodes, seq_edges) in enumerate(list_walks):
        mat_node[:len(seq_nodes), i] = seq_nodes
        mat_edge[:len(seq_edges), i] = seq_edges
  return mat_node, mat_edge

def get_or_unk(type_dict, key):
    if key in type_dict:
        return type_dict[key]
    return type_dict[UNK]

class AnchorNotFoundError(Exception):
    def __init__(self, message="Anchor not found."):
        self.message = message
        super().__init__(self.message)

def make_mat_from_raw(walks, node_types, edge_types):
  list_walks = []
  max_n_nodes = 0
  max_n_edges = 0
  for walk_num, walk in enumerate(walks):
    seq_nodes = []
    seq_edges = []
    for idx, step in enumerate(walk):
      if idx % 2 == 0:
        seq_nodes.append(get_or_unk(node_types, step['label']))
      else:
        seq_edges.append(get_or_unk(edge_types, step['label']))
    max_n_nodes = max(max_n_nodes, len(seq_nodes))
    max_n_edges = max(max_n_edges, len(seq_edges))
    list_walks.append((seq_nodes, seq_edges))
  node_mat, edge_mat = make_mat(list_walks, max_n_nodes, max_n_edges)
  return node_mat, edge_mat

class AbstractWalkDataset(Dataset):
  def __init__(self, args, prog_dict, data_dir, biases, phase):
    super(AbstractWalkDataset, self).__init__()
    self.args = args
    self.prog_dict = prog_dict
    self.biases = biases
    self.phase = phase

  def get_train_loader(self, args):
    return DataLoader(self,
                      batch_size=args.batch_size,
                      shuffle=True,
                      drop_last=True,
                      collate_fn=collate_raw_data)

  def get_test_loader(self, args):
    return DataLoader(self,
                      batch_size=args.batch_size,
                      shuffle=False,
                      drop_last=False,
                      collate_fn=collate_raw_data)

  def get_item_from_rawfile(self, walker, label):
    walks = walker.generate_walks(num_walks=self.args.num_walks, num_steps=self.args.num_steps)
    node_mat, edge_mat = make_mat_from_raw(walks, self.prog_dict.node_types, self.prog_dict.edge_types)

    node_val_coo = []
    for walk_idx, walk in enumerate(walks):
      for step_idx, step in enumerate(walk):
        if step_idx % 2 == 1: continue
        for tok in ProgDict.tokenize(step['values']):
          node_val_coo.append((step_idx//2, walk_idx, get_or_unk(self.prog_dict.token_vocab , tok)))
    node_val_coo = (np.array(node_val_coo, dtype=np.int32), len(self.prog_dict.token_vocab))
    return RawData(node_mat, edge_mat, node_val_coo, 0 if label == 'NEGATIVE' else 1)

class OnlineWalkDataset(AbstractWalkDataset):
  def __init__(self, args, prog_dict, data_dir, biases, phase):
    super(OnlineWalkDataset, self).__init__(args, prog_dict, data_dir, biases, phase)
    self.samples = []
    for item in os.listdir(os.path.join(data_dir, phase)):
      if item.startswith('sample_') and item.endswith('.json'):
        self.samples.append(item)
    if phase == 'eval':
      with open('eval_samples.txt', 'w') as f:
        for sample in self.samples:
          f.write(sample + '\n')

  def __getitem__(self, idx):
    with open(os.path.join(self.args.data_dir, self.phase, self.samples[idx]), 'r') as f:
      sample_info = json.load(f)
      graph_path = os.path.join(self.args.data_dir, self.phase, self.samples[idx]).replace('/sample_', '/graph_')
      graph_path = graph_path[:graph_path.rfind('.sol_')] + '.sol.pkl'
      sample_graph = GraphBuilder.load(graph_path)
    anchor = None
    if sample_info['anchor'] not in sample_graph.nodes:
      for node in sample_graph.nodes(data=True):
        if node[1]['label'] == 'schema_functions':
          if node[1]['raw_values']['name'] == sample_info['function']:
            anchor = node[0]
      if anchor is None:
        with open('silent_skips.txt', 'a') as f:
          f.write(self.samples[idx] + '\n')
        raise AnchorNotFoundError
    else:
      anchor = sample_info['anchor']

    walker = Walker(sample_graph, [anchor], self.biases)
    label = sample_info['label']
    return self.get_item_from_rawfile(walker, label)
  
  def __len__(self):
    return len(self.samples)

def eval_path_based_nn_args(nn_args, device):
  node_idx, edge_idx, node_val_mat, label = nn_args
  if device.startswith('mps'):
    node_val_mat = torch.sparse_coo_tensor(*node_val_mat).to_dense().to(device)
  else:
    node_val_mat = torch.sparse_coo_tensor(*node_val_mat).to(device)
  edge_idx = edge_idx.to(device)
  nn_args = {'node_idx': node_idx.to(device),
               'edge_idx': edge_idx,
               'node_val_mat': node_val_mat}
  return nn_args, label

def path_based_arg_constructor(nn_args, device):
  node_idx, edge_idx, node_val_mat, label = nn_args
  if node_val_mat is not None:
    if device.startswith('mps'):
      node_val_mat = torch.sparse_coo_tensor(*node_val_mat).to_dense().to(device)
    else:
      node_val_mat = torch.sparse_coo_tensor(*node_val_mat).to(device)
  if edge_idx is not None:
    edge_idx = edge_idx.to(device)
    
  nn_args = {'node_idx': node_idx.to(device),
              'edge_idx': edge_idx,
              'node_val_mat': node_val_mat,
              'label': label.to(device)}
  return nn_args

def binary_eval_dataset(model, phase, eval_loader, device, fn_parse_eval_nn_args=eval_path_based_nn_args):
  model.eval()
  pred_probs = []
  true_labels = []
  eval_iter = iter(eval_loader)
  silent_skips = 0
  print("Running eval on", phase, "set with", len(eval_loader), "batches.")
  pbar = tqdm(total=len(eval_loader))
  random.seed(int(time.time()))
  run_id = random.randint(1000,9999)
  print("Run ID:", run_id)
  with open(f'eval_report_{phase}_{run_id}.txt', 'a') as f:
    f.write('TRUE_LABEL,PRED_LABEL,PRED_PROB,SAMPLE'+'\n')
  while True:
    try:
      nn_args = next(eval_iter)
    except StopIteration:
        break
    except AnchorNotFoundError:
        silent_skips += 1
        continue
    with torch.no_grad():
      nn_args, label = fn_parse_eval_nn_args(nn_args, device)
      pred = model(**nn_args).data.cpu().numpy()
      pred_probs += pred.flatten().tolist()
      true_labels += label.data.numpy().flatten().tolist()

    pbar.update(1)
  pbar.close()

  if silent_skips > 0:
    print('Had to silently skip', silent_skips, 'samples.')

  if phase in ['dev', 'train', 'test']:
    roc_auc = roc_auc_score(true_labels, pred_probs)
    pred_label = np.where(np.array(pred_probs) > 0.5, 1, 0)
    acc = np.mean(pred_label == np.array(true_labels, dtype=pred_label.dtype))
    print("ROC=", roc_auc, ", ACC=", acc)
    fn = 0
    fp = 0
    if phase == "test":
      with open(f'eval_report_{phase}_{run_id}.txt', 'a') as f:
        for idx in range(len(pred_label)):
          f.write(f'{true_labels[idx]},{pred_label[idx]},{pred_probs[idx]},?\n')
          if true_labels[idx] == 0 and pred_label[idx] == 1:
            fp += 1
          if true_labels[idx] == 1 and pred_label[idx] == 0:
            fn += 1
        f.write('===========================\n')
        f.write(f'ROC AUC: {roc_auc}\n')
        f.write(f'    ACC: {acc}\n')
        f.write(f'     FP: {fp/len(pred_label)} ({fp} out of {len(pred_label)})\n')
        f.write(f'     FN: {fn/len(pred_label)} ({fn} out of {len(pred_label)})\n')

      print("saved the evaluation result in:", f'eval_report_{phase}_{run_id}.txt')
    return roc_auc
  else:
    pred_label = np.where(np.array(pred_probs) > 0.5, 1, 0)
    print("PRED_LABEL:",pred_label,"\nTRUE_LABEL:",true_labels)
    return None
