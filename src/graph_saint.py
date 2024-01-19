# This file is an extended copy of 'torch_geometric.data.graph_saint.py',
# since torch_geometric 1.4.0 does not incorporate this file,
# so we rewrite a permutation-based random-walk sampler.

import copy
import os.path as osp

import torch
from tqdm import tqdm
from torch_sparse import SparseTensor, rw, saint


class MySAINTSampler(object):
    r"""A new random-walk sampler for GraphSAINT that samples initial nodes
    by iterating over node permutations. The benefit is that we can leverage
    this sampler for subgraph-based inference.

    Args:
        data (torch_geometric.data.Data): The graph data object.
        batch_size (int): The number of walks to sample per batch.
        walk_length (int): The length of each random walk.
        sample_coverage (int): How many samples per node should be used to
            compute normalization statistics. (default: :obj:`50`)
        save_dir (string, optional): If set, will save normalization
            statistics to the :obj:`save_dir` directory for faster re-use.
            (default: :obj:`None`)
        log (bool, optional): If set to :obj:`False`, will not log any
            progress. (default: :obj:`True`)
    """

    def __init__(self, data, batch_size, sample_type='random_walk', walk_length=2, sample_coverage=50,
                 save_dir=None, log=True):
        assert data.edge_index is not None
        assert 'node_norm' not in data
        assert 'edge_norm' not in data
        assert sample_type in ('node', 'random_walk')

        self.N = N = data.num_nodes
        self.E = data.num_edges

        self.adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                                value=data.edge_attr, sparse_sizes=(N, N))

        self.data = copy.copy(data)
        self.data.edge_index = None
        self.data.edge_attr = None

        self.sample_type = sample_type
        self.batch_size = batch_size
        # self.num_steps = num_steps
        self.walk_length = walk_length
        self.sample_coverage = sample_coverage
        self.log = log

        path = osp.join(save_dir or '', self.__filename__)
        if save_dir is not None and osp.exists(path):  # pragma: no cover
            self.node_norm, self.edge_norm = torch.load(path)
        else:
            self.node_norm, self.edge_norm = self.__compute_norm__()
            if save_dir is not None:  # pragma: no cover
                torch.save((self.node_norm, self.edge_norm), path)

    @property
    def __filename__(self):
        return f'{self.__class__.__name__.lower()}_{self.sample_coverage}.pt'

    def __sample_nodes__(self):
        """Sampling initial nodes by iterating over the random permutation of nodes"""
        tmp_map = torch.arange(self.N, dtype=torch.long)
        all_n_id = torch.randperm(self.N, dtype=torch.long)
        node_samples = []
        for s_id in range(0, self.N, self.batch_size):
            init_n_id = all_n_id[s_id:s_id+self.batch_size]  # [batch_size]

            if self.sample_type == 'random_walk':
                n_id = self.adj.random_walk(init_n_id, self.walk_length)  # [batch_size, walk_length+1]
                n_id = n_id.flatten().unique()  # [num_nodes_in_subgraph]
                tmp_map[n_id] = torch.arange(n_id.size(0), dtype=torch.long)
                res_n_id = tmp_map[init_n_id]
            elif self.sample_type == 'node':
                n_id = init_n_id
                res_n_id = torch.arange(n_id.size(0), dtype=torch.long)
            else:
                raise ValueError('Unsupported value type {}'.format(self.sample_type))

            node_samples.append((n_id, res_n_id))

        return node_samples

    def __sample__(self, num_epoches):
        samples = []
        for _ in range(num_epoches):
            node_samples = self.__sample_nodes__()
            for n_id, res_n_id in node_samples:
                adj, e_id = self.adj.saint_subgraph(n_id)
                samples.append((n_id, e_id, adj, res_n_id))

        return samples

    def __compute_norm__(self):
        node_count = torch.zeros(self.N, dtype=torch.float)
        edge_count = torch.zeros(self.E, dtype=torch.float)

        if self.log:
            pbar = tqdm(total=self.sample_coverage)
            pbar.set_description('GraphSAINT Normalization')

        num_samples = len(self) * self.sample_coverage
        for _ in range(self.sample_coverage):
            samples = self.__sample__(1)

            for n_id, e_id, _, _ in samples:
                node_count[n_id] += 1
                edge_count[e_id] += 1

            if self.log:
                pbar.update(1)

        row, col, _ = self.adj.coo()

        edge_norm = (node_count[col] / edge_count).clamp_(0, 1e4)
        edge_norm[torch.isnan(edge_norm)] = 0.1

        node_count[node_count == 0] = 0.1
        node_norm = num_samples / (node_count * self.N)

        return node_norm, edge_norm

    def __get_data_from_sample__(self, sample):
        n_id, e_id, adj, res_n_id = sample
        data = self.data.__class__()
        data.num_nodes = n_id.size(0)
        row, col, value = adj.coo()
        data.edge_index = torch.stack([row, col], dim=0)
        data.edge_attr = value

        for key, item in self.data:
            if item.size(0) == self.N:
                data[key] = item[n_id]
            elif item.size(0) == self.E:
                data[key] = item[e_id]
            else:
                data[key] = item

        data.node_norm = self.node_norm[n_id]
        data.edge_norm = self.edge_norm[e_id]

        data.n_id = n_id
        data.res_n_id = res_n_id
        data.e_id = e_id

        return data

    def __len__(self):
        return (self.N + self.batch_size-1) // self.batch_size

    def __iter__(self):
        for sample in self.__sample__(1):
            data = self.__get_data_from_sample__(sample)
            yield data