import random

class Walker:
  def __init__(self, graph, anchors, biases, seed):
    random.seed(seed)
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
    walks = [self._generate_walks(anchor, num_walks // len(self.anchors), num_steps) for anchor in self.anchors]
    walks = [walk for sublist in walks for walk in sublist]
    if len(walks) > num_walks:
      walks = walks[:num_walks]
    else:
      last_walk = walks[-1]
      walks.extend([last_walk] * (num_walks - len(walks)))
    return walks

  def _sample_node(self, nodes):
    return random.choices(nodes, [self.biases.get(self.graph.nodes[node]['label'], 1) for node in nodes], k=1)[0]

  def _postprocess_walks(self, walks):
    # add edge information to the walks
    for walk_idx in range(len(walks)):
      this_walk = walks[walk_idx]
      with_edges = []
      for step in range(len(this_walk) - 1):
        curr_step = this_walk[step]
        next_step = this_walk[step+1]
        with_edges.append(self.graph.nodes[curr_step])
        with_edges.append(self.graph.get_edge_data(curr_step, next_step))
        if step == len(this_walk) - 2:
          with_edges.append(self.graph.nodes[next_step])
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

    return self._postprocess_walks(walks)
