import pickle
import networkx as nx

from ..constants import UNK, EMPTY, NONE

class GraphBuilder:
  def __init__(self, tables_path):
    self.tables_path = tables_path
    with open(tables_path, 'rb') as f:
      self.tables = pickle.load(f)
    self._build()

  def _combine_values(self, row):
    vals = []
    for v in row:
      if not self._is_id(v, row[v]):
        if row[v] is None:
          vals.append(NONE)
        else:
          vals.append(row[v])
    return " ".join(vals)

  def _build(self):
    self.graph = nx.Graph()

    # add the main nodes
    for table, rows in self.tables.items():
      for row in rows:
        if "id" in row:
          self.graph.add_node(row["id"], label=table, values=self._combine_values(row), raw_values=row)

    # add the edge-only rows as edges
    for table, rows in self.tables.items():
      for row in rows:
        if "id" in row: continue
        a, b = row.values()
        self.graph.add_edge(a, b, label="_".join([table, *row.keys()]))

    # add foreign key edges
    list_nodes = list(self.graph.nodes)
    for node1_idx in range(len(list_nodes)):
      node1 = list_nodes[node1_idx]
      graph_node1 = self.graph.nodes[node1]
      for node2_idx in range(node1_idx + 1, len(list_nodes)):
        node2 = list_nodes[node2_idx]
        graph_node2 = self.graph.nodes[node2]
        if 'label' not in graph_node1 or 'label' not in graph_node2:
          continue

        row1 = graph_node1['raw_values']
        row2 = graph_node2['raw_values']

        for attr1 in row1:
          for attr2 in row2:
            if row1[attr1] == row2[attr2] and self._is_id(attr1, row1[attr1]) and self._is_id(attr1, row2[attr2]):
              if (attr1 == 'id' and attr2 != 'id') or (attr1 != 'id' and attr2 == 'id'):
                self.graph.add_edge(node1, node2, label="_".join([graph_node1['label'], attr1, graph_node2['label'], attr2]))

    # postprocessing nodes
    for node in self.graph.nodes:
      if 'label' not in self.graph.nodes[node]:
        self.graph.nodes[node]['label'] = UNK
        self.graph.nodes[node]['values'] = EMPTY
        self.graph.nodes[node]['raw_values'] = {}

  def _is_id(self, v, i):
    if i is None or v == 'source_mapping' or len(i) < 7:
        return False
    return i.startswith('#') and i.endswith('#')

  def save(self, filename):
    assert self.graph is not None
    pickle.dump(self.graph, open(filename, 'wb'))

  @staticmethod
  def load(filename):
    with open(filename, 'rb') as f:
      return pickle.load(f)
