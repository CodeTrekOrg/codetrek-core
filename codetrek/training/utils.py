import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

class InitUtils:
  @staticmethod
  def _uniform(t):
        if len(t.size()) == 2:
            fan_in, fan_out = t.size()
        elif len(t.size()) == 3:
            fan_in = t.size()[1] * t.size()[2]
            fan_out = t.size()[0] * t.size()[2]
        else:
            fan_in = np.prod(t.size())
            fan_out = np.prod(t.size())
        
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        t.uniform_(-limit, limit)

  @staticmethod
  def _param_init(m):
        if isinstance(m, nn.Parameter):
            InitUtils._uniform(m.data)
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
            InitUtils._uniform(m.weight.data)
        elif isinstance(m, nn.Embedding):
            InitUtils._uniform(m.weight.data)

class SparseMatrixMult(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sp_mat, dense_mat):
        ctx.save_for_backward(sp_mat, dense_mat)
        return torch.mm(sp_mat, dense_mat)

    @staticmethod
    def backward(ctx, grad_output):        
        sp_mat, dense_mat = ctx.saved_variables
        grad_matrix1 = grad_matrix2 = None
        assert not ctx.needs_input_grad[0]
        if ctx.needs_input_grad[1]:
            grad_matrix2 = Variable(torch.mm(sp_mat.data.t(), grad_output.data))
        return grad_matrix1, grad_matrix2

def gnn_spmm(sp_mat, dense_mat):
  return SparseMatrixMult.apply(sp_mat, dense_mat)
