# Copyright evgps

# Licensed under the MIT license:

#     http://www.opensource.org/licenses/mit-license.php

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Code created based on https://github.com/google-research/nasbench
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import json
import sys

from absl import app
from absl import flags
from absl import logging

from .graph_util import gen_is_edge_fn, is_full_dag, hash_module, is_isomorphic, num_edges
import numpy as np

def generate_graphs(output_file="generated_graphs.json", max_vertices=7, num_ops=3, max_edges=9, verify_isomorphism=True):
  total_graphs = 0    # Total number of graphs (including isomorphisms)
  # hash --> (matrix, label) for the canonical graph associated with each hash
  buckets = {}

  logging.info('Using %d vertices, %d op labels, max %d edges',
               max_vertices, num_ops, max_edges)
  for vertices in range(2, max_vertices+1):
    for bits in range(2 ** (vertices * (vertices-1) // 2)):
      # Construct adj matrix from bit string
      matrix = np.fromfunction(gen_is_edge_fn(bits),
                               (vertices, vertices),
                               dtype=np.int8)

      # Discard any graphs which can be pruned or exceed constraints
      if (not is_full_dag(matrix) or
          num_edges(matrix) > max_edges):
        continue

      # Iterate through all possible labelings
      for labeling in itertools.product(*[range(num_ops)
                                          for _ in range(vertices-2)]):
        total_graphs += 1
        labeling = [-1] + list(labeling) + [-2]
        fingerprint = hash_module(matrix, labeling)

        if fingerprint not in buckets:
          buckets[fingerprint] = (matrix.tolist(), labeling)

        # This catches the "false positive" case of two models which are not
        # isomorphic hashing to the same bucket.
        elif verify_isomorphism:
          canonical_graph = buckets[fingerprint]
          if not is_isomorphic(
              (matrix.tolist(), labeling), canonical_graph):
            logging.fatal('Matrix:\n%s\nLabel: %s\nis not isomorphic to'
                          ' canonical matrix:\n%s\nLabel: %s',
                          str(matrix), str(labeling),
                          str(canonical_graph[0]),
                          str(canonical_graph[1]))
            sys.exit()

    logging.info('Up to %d vertices: %d graphs (%d without hashing)',
                 vertices, len(buckets), total_graphs)

  with open(output_file, 'w') as f:
    json.dump(buckets, f, sort_keys=True)

