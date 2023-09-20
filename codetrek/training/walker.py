import random
from ..configs import cmd_args

class Walker:
  def __init__(self, graph, anchors, biases):
    random.seed(cmd_args.seed)
    self.graph = graph
    self.biases = biases
    self.anchors = anchors

  def pretty_print(self, walks):
    for walk in walks:
      for idx, step in enumerate(walk):
        if isinstance(step, dict):
          if 'values' in step:
            print('['+step['values']+']', end="")
          else:
            print('['+step['label']+']', end="")
        else:
          print('['+step+']', end="")
        if idx != len(walk) - 1: print(" --> ", end="")
        else: print()
    print("==============")

  def generate_walks(self, num_walks, num_steps):
    walks = []
    for anchor in self.anchors:
      walks += self._generate_walks(anchor, num_walks//len(self.anchors), num_steps)
    if len(walks) > num_walks:
      walks = walks[:num_walks]
    else:
      last_walk = walks[-1]
      for _ in range(num_walks - len(walks)):
        walks.append(last_walk)
    return walks

  def _sample_node(self, nodes):
    weights = []
    for node in nodes:
      weights.append(self.biases[self.graph.nodes[node]['label']] if self.graph.nodes[node]['label'] in self.biases else 1)
    return random.choices(nodes, weights, k=1)[0]

  def _postprocess_walks(self, walks, num_walks):
    # add edge information to the walks
    for walk_idx in range(len(walks)):
      with_edges = []
      for step in range(len(walks[walk_idx]) - 1):
        node1 = self.graph.nodes[walks[walk_idx][step]]
        node2 = self.graph.nodes[walks[walk_idx][step+1]]
        with_edges.append(node1)
        with_edges.append(self.graph.get_edge_data(walks[walk_idx][step],walks[walk_idx][step+1]))
        if step == len(walks[walk_idx]) - 2:
          with_edges.append(node2)
      walks[walk_idx] = with_edges
    return walks

  def _generate_walks(self, anchor, num_walks, num_steps):
    iter = 0
    walks = []
    while iter < num_walks * 3:
      curr_node = anchor
      curr_walk = [curr_node]

      for _ in range(num_steps):
        neighbors = list(self.graph.neighbors(curr_node))
        prev_node = curr_node
        curr_node = self._sample_node(neighbors)
        if curr_node not in curr_walk:
          curr_walk.append(curr_node)
        else:
          curr_node = prev_node

      if curr_walk not in walks:
        walks.append(curr_walk)        
      if len(walks) >= num_walks:
        break
      iter += 1

    return self._postprocess_walks(walks, num_walks)
