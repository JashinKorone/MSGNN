import torch
import math
import numpy as np
import torch.distributed as dist

from torch_geometric.data import Data, ClusterData, ClusterLoader, NeighborSampler
from torch.utils.data import DataLoader, TensorDataset, IterableDataset, Dataset
from graph_saint import MySAINTSampler


class SAINTDataset(IterableDataset):
    def __init__(self, inputs, edge_index, edge_attr, num_nodes, batch_size,
                 shuffle=False, use_dist_sampler=False, rep_eval=None, shuffle_order='node_first',
                 saint_sample_type='random_walk', saint_batch_size=100,
                 saint_walk_length=2, saint_sample_coverage=50, use_saint_norm=False):
        self.inputs = inputs

        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.shuffle = shuffle
        # whether to use distributed sampler when available
        self.use_dist_sampler = use_dist_sampler
        # number of repeats to run evaluation, set to None for training
        self.rep_eval = rep_eval
        # choices: 'date_first' or 'node_first'
        self.shuffle_order = shuffle_order
        # the type of the sampling method used in GraphSAINT
        self.saint_sample_type = saint_sample_type
        # the number of initial nodes for random walk at each batch
        self.saint_batch_size = saint_batch_size
        # the length of each random walk
        self.saint_walk_length = saint_walk_length
        # the number of samples per node used to compute normalization statistics
        self.saint_sample_coverage = saint_sample_coverage
        # whether to use SAINT-based node/edge normalization tricks
        self.use_saint_norm = use_saint_norm

        # use 'epoch' as the random seed to shuffle data for distributed training
        self._epoch = None

        self._graph_sampler = self._make_graph_sampler()
        self._length = self.get_length()

    def _make_graph_sampler(self):
        graph = Data(edge_index=self.edge_index,
                     edge_attr=self.edge_attr,
                     num_nodes=self.num_nodes)

        saint_sampler = MySAINTSampler(
            graph, self.saint_batch_size,
            sample_type=self.saint_sample_type, walk_length=self.saint_walk_length,
            sample_coverage=self.saint_sample_coverage, save_dir=None, log=False)

        return saint_sampler

    def get_subgraph(self, saint_data):
        graph = {
            'type': 'subgraph',
            'edge_index': saint_data.edge_index,
            'edge_attr': saint_data.edge_attr,
            'cent_n_id': saint_data.n_id,
            'res_n_id': saint_data.res_n_id,
            'e_id': saint_data.e_id,
        }

        if self.use_saint_norm:
            graph['node_norm'] = saint_data.node_norm
            graph['edge_norm'] = saint_data.edge_norm

        return graph

    def __iter__(self):
        repeats = 1 if self.rep_eval is None else self.rep_eval

        for rep in range(repeats):
            if self.use_dist_sampler and dist.is_initialized():
                # ensure that all processes share the same graph dataflow
                # set seed as epoch for training, and rep for evaluation
                torch.manual_seed(self._epoch)

            if self.rep_eval is not None:
                # fix random seeds for repetitive evaluation
                # this attribute should not be set during training
                torch.manual_seed(rep)

            if self.shuffle_order == 'node_first':
                for saint_data in self._graph_sampler:
                    g = self.get_subgraph(saint_data)
                    inputs = list(
                        map(lambda x: x[:, g['cent_n_id']], self.inputs)
                    )
                    dataset_len = inputs[0].size(0)
                    indices = list(range(dataset_len))

                    if self.use_dist_sampler and dist.is_initialized():
                        # distributed sampler reference: torch.utils.data.distributed.DistributedSampler
                        if self.shuffle:
                            # ensure that all processes share the same permutated indices
                            tg = torch.Generator()
                            tg.manual_seed(self._epoch)
                            indices = torch.randperm(dataset_len, generator=tg).tolist()

                        world_size = dist.get_world_size()
                        node_rank = dist.get_rank()
                        num_samples_per_node = int(math.ceil(dataset_len * 1.0 / world_size))
                        total_size = world_size * num_samples_per_node

                        # add extra samples to make it evenly divisible
                        indices += indices[:(total_size - dataset_len)]
                        assert len(indices) == total_size

                        # get sub-batch for each process
                        # Node (rank=x) get [x, x+world_size, x+2*world_size, ...]
                        indices = indices[node_rank:total_size:world_size]
                        assert len(indices) == num_samples_per_node
                    elif self.shuffle:
                        np.random.shuffle(indices)

                    num_batches = (len(indices) + self.batch_size - 1) // self.batch_size
                    for batch_id in range(num_batches):
                        start = batch_id * self.batch_size
                        end = (batch_id + 1) * self.batch_size
                        yield list(map(lambda x: x[indices[start: end]], inputs)), g, torch.LongTensor(indices[start: end])
            elif self.shuffle_order == 'date_first':
                dataset_len = self.inputs[0].size(0)
                indices = list(range(dataset_len))
                if self.use_dist_sampler and dist.is_initialized():
                    # distributed sampler reference: torch.utils.data.distributed.DistributedSampler
                    if self.shuffle:
                        # ensure that all processes share the same permutated indices
                        tg = torch.Generator()
                        tg.manual_seed(self._epoch)
                        indices = torch.randperm(dataset_len, generator=tg).tolist()

                    world_size = dist.get_world_size()
                    node_rank = dist.get_rank()
                    num_samples_per_node = int(math.ceil(dataset_len * 1.0 / world_size))
                    total_size = world_size * num_samples_per_node

                    # add extra samples to make it evenly divisible
                    indices += indices[:(total_size - dataset_len)]
                    assert len(indices) == total_size

                    # get sub-batch for each process
                    # Node (rank=x) get [x, x+world_size, x+2*world_size, ...]
                    indices = indices[node_rank:total_size:world_size]
                    assert len(indices) == num_samples_per_node
                elif self.shuffle:
                    np.random.shuffle(indices)

                num_batches = (len(indices) + self.batch_size - 1) // self.batch_size
                for batch_id in range(num_batches):
                    start = batch_id * self.batch_size
                    end = (batch_id + 1) * self.batch_size
                    batch_inputs = list(
                        map(lambda x: x[indices[start: end]], self.inputs)
                    )
                    batch_indices = torch.LongTensor(indices[start: end])

                    for saint_data in self._graph_sampler:
                        g = self.get_subgraph(saint_data)
                        graph_batch_inputs = list(
                            map(lambda x: x[:, g['cent_n_id']], batch_inputs)
                        )
                        yield graph_batch_inputs, g, batch_indices
            else:
                raise Exception(f'Unsupported shuffle_order {self.shuffle_order}')

    def get_length(self):
        if self.use_dist_sampler and dist.is_initialized():
            dataset_len = self.inputs[0].size(0)
            world_size = dist.get_world_size()
            num_samples_per_node = int(math.ceil(dataset_len * 1.0 / world_size))
        else:
            num_samples_per_node = self.inputs[0].size(0)
        length = (num_samples_per_node + self.batch_size - 1) // self.batch_size
        length *= len(self._graph_sampler)

        return length

    def __len__(self):
        return self._length

    def set_epoch(self, epoch):
        # self.set_epoch() will be called by BasePytorchTask on each epoch when using distributed training
        self._epoch = epoch


class SimpleDataset(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs
        sz = inputs[0].size()
        self.date_size = sz[0]
        self.num_nodes = sz[1]

    def __getitem__(self, idx):
        i = idx // self.num_nodes        # date index
        j = idx % self.num_nodes         # node index
        return list(map(lambda x: x[i,j].unsqueeze(0), self.inputs)), j, i

    def __len__(self):
        return self.date_size * self.num_nodes