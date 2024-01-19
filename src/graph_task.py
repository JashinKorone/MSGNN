import os
import time
import argparse
import json
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd
from torch_scatter import scatter
import logging as lg

from base_task import add_config_to_argparse, BaseConfig, BasePytorchTask, \
    LOSS_KEY, BAR_KEY, SCALAR_LOG_KEY, VAL_SCORE_KEY
from dataset import SAINTDataset, SimpleDataset
from data_utils import load_new_data
from nbeats import NBeatsModel
from msgnn import MainModel
from graph_optim import TruncateSGD, TruncateAdam


class Config(BaseConfig):
    def __init__(self):
        super().__init__()
        # Reset base variables
        self.max_epochs = 1000
        self.early_stop_epochs = 15
        self.infer = False
        self.enable_multi_scale = True

        # for data loading
        self.data_fp = '../data/daily_us_7.csv'
        self.start_date = '2020-03-01'
        self.min_peak_size = -1  # min peak confirmed cases selected by country level
        self.lookback_days = 14  # the number of days before the current day for daily series
        self.lookahead_days = 1
        self.forecast_date = '2020-06-29'
        self.horizon = 7
        self.val_days = 1  # the number of days used for validation
        self.label = 'confirmed_target'
        self.use_mobility = False

        self.model_type = 'msgnn'
        self.rnn_type = 'nbeats'
        self.date_emb_dim = 2

        self.use_gbm = False
        self.use_lr = True

        # for krnn
        self.cnn_dim = 32
        self.cnn_kernel_size = 3
        self.rnn_dim = 32
        self.rnn_dups = 10

        # for transformer
        self.tfm_layer_num = 8
        self.tfm_head_num = 8
        self.tfm_hid_dim = 32
        self.tfm_ff_dim = 32
        self.tfm_max_pos = 500
        self.tfm_node_dim = 5
        self.tfm_dropout = 0.1
        self.tfm_block_num = -1
        self.tfm_cnn_kernel_size = 1

        # for n_beats
        self.block_size = 3
        self.hidden_dim = 32
        self.id_emb_dim = 8

        # for gcn
        self.gcn_dim = 64
        self.gcn_type = 'gcn'
        self.gcn_aggr = 'max'
        self.gcn_norm = 'none'
        self.gcn_layer_num = 2
        self.gcn_node_dim = 4
        self.gcn_edge_dim = 4
        self.gcn_dropout = 0.1

        # for gov gate
        self.use_gov_gate = False
        self.gov_id_dim = 32
        self.gov_hid_dim = 32

        # per-gpu training batch size, real_batch_size = batch_size * num_gpus * grad_accum_steps
        # per-gpu training batch size, real_batch_size = batch_size * num_gpus * grad_accum_steps
        self.batch_size = 4
        self.lr = 1e-3  # the learning rate

        # batch sample type
        self.use_saintdataset = True
        self.saint_batch_size = 3000
        self.saint_sample_type = 'random_walk'
        self.saint_walk_length = 2
        self.saint_shuffle_order = 'node_first'

        # graph optimization (deprecated)
        self.optim_graph = False
        self.graph_fp = '../data/us_graph.cpt'
        self.graph_lr = 1e-4  # learning rate for graph adjacent matrix
        self.graph_opt_type = 'TruncateAdam'  # TruncateAdam, TruncateSGD, Adam
        self.graph_gravity = 0.1  # sparse regularization coefficients
        self.graph_eta = 0.01  # \eta * || A - A_{prior} ||_2^2

        # consistency loss
        # the usage of 'xxxx_loss_node_num'
        #   -1: use all nodes,
        #   0: not use this loss,
        #   >0: use a certain number of randomly selected nodes
        self.topo_loss_node_num = -1
        self.topo_loss_weight = 0.01
        self.topo_loss_epoch_start = 3
        self.pair_loss_node_num = -1
        self.pair_loss_weight = 0.0

        # temp options
        self.use_node_weight = True
        self.mape_eps = 10
        self.sparse_gate_weight = 0.0
        self.sparse_gate_epoch_start = 3

        self.prepro_type = 'none'
        self.use_popu_norm = True
        self.use_logy = False
        self.use_fea_zscore = False
        self.use_adapt_norm = False
        self.use_default_edge = False
        self.abla_type = 'none'
        self.fea_day_offset = 1

        self.data_aug_scales = '1'  # a list of scales applied for training data augmentation


class WrapperNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        if self.config.model_type == 'nbeats':
            self.net = NBeatsModel(config)
        elif self.config.model_type == 'msgnn':
            self.net = MainModel(config)
        else:
            raise Exception(
                'Unsupported model type {}'.format(config.model_type))

        if config.use_lr:
            self.weight_lr = nn.Parameter(torch.Tensor(self.config.lookback_days, self.config.lookahead_days))
            self.b_lr = nn.Parameter(torch.Tensor([0.0] * self.config.lookahead_days))

        if config.use_gov_gate:
            self.state_emb = nn.Embedding(self.config.num_nodes, self.config.gov_id_dim)
            self.gov_gru = nn.GRU(input_size=self.config.day_gov_fea_dim,
                                  hidden_size=self.config.gov_hid_dim,
                                  batch_first=True)

            self.state_weight = nn.Parameter(torch.Tensor(self.config.gov_hid_dim, self.config.lookahead_days))
            self.gov_weight = nn.Parameter(torch.Tensor(self.config.gov_id_dim, self.config.lookahead_days))

        self.reset_parameters()

    def gov_map(self, input_day_gov):
        sz = input_day_gov.size()
        x = input_day_gov.view(-1, sz[2], sz[3])
        _, h = self.gov_gru(x)

        h = h[0,:,:].view(sz[0],sz[1],-1)
        return h

    def state_map(self, input_day, g):
        sz = input_day.size()
        n_id = g['cent_n_id']
        id_emb = self.state_emb(n_id.reshape(1,sz[1]).expand(sz[0],sz[1]).long())

        return id_emb

    def lr(self, input_day):
        sz = input_day.size()
        label_idx = self.config.label_fea_idx
        ts = input_day[:, :, :, label_idx]
        if self.config.use_logy:
            ts = ts.expm1()
        pred = torch.matmul(ts, torch.softmax(self.weight_lr, dim=0)) + self.b_lr
        if self.config.use_logy:
            pred = torch.log1p(pred)
        pred = pred.view(sz[0], sz[1], self.config.lookahead_days)
        return pred

    def reset_parameters(self):
        if self.config.use_lr:
            nn.init.xavier_uniform_(self.weight_lr)
        if self.config.use_gov_gate:
            nn.init.xavier_uniform_(self.gov_weight)
            nn.init.xavier_uniform_(self.state_weight)

    def forward_ori(self, input_day, g):
        out = self.net(input_day, g)
        if self.config.use_lr:
            out = out + self.lr(input_day)
        return out

    def forward(self, input_day, input_day_gov, g):
        ori_out = self.forward_ori(input_day, g)
        if self.config.use_gov_gate:
            gov_hid = self.gov_map(input_day_gov)
            state_hid = self.state_map(input_day, g)
            state_gate = torch.sigmoid(torch.matmul(state_hid, self.state_weight))
            gov_gate = torch.tanh(torch.matmul(gov_hid, self.gov_weight))

            out = ori_out * (1 + state_gate * gov_gate)
        else:
            out, state_gate, gov_gate = ori_out, torch.ones_like(ori_out), torch.ones_like(ori_out)
        return out, state_gate, gov_gate


class GraphNet(nn.Module):
    def __init__(self, config, edge_weight):
        super().__init__()

        self.config = config
        self.net = WrapperNet(config)
        if config.optim_graph:
            self.edge_weight = nn.Parameter(edge_weight)
        else:
            self.edge_weight = None

    def forward(self, input_day, input_day_gov, g):
        return self.net(input_day, input_day_gov, g)

    def get_net_parameters(self):
        return self.net.parameters()

    def get_graph_parameters(self):
        yield self.edge_weight


def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)


class Task(BasePytorchTask):
    def __init__(self, config):
        super().__init__(config)
        self.main_feas = None
        self.county_node_index = None
        self.state_node_index = None
        self.country_node_index = None
        self.country_node = None
        self.edge_weight = None
        self.edge_index = None
        self.log('Initialize {}'.format(self.__class__))

        self.init_data()
        self.log('======Data initialized======')
        self.init_graph()
        self.log('=======Graph initialized======')
        self.adjust_for_ablation_study()
        self.loss_func = nn.MSELoss()
        # self.loss_func = nn.L1Loss()
        self.log('Config:\n{}'.format(
            json.dumps(self.config.to_dict(), ensure_ascii=False, indent=4)
        ))

    def adjust_for_ablation_study(self):
        if self.config.abla_type == 'gat':
            self.config.gcn_type = 'gat'
        elif self.config.abla_type == 'flat':
            edge_sid = (self.edge_type == 0).sum().item()
            self.edge_index = self.edge_index[:, edge_sid:]
            self.edge_weight = self.edge_weight[edge_sid:]
            self.edge_type = self.edge_type[edge_sid:]
            self.config.num_edges = self.edge_weight.shape[0]
        elif self.config.abla_type.startswith('sep'):
            cur_node_type = int(self.config.abla_type[3:])
            node_sid, node_eid = None, None
            for idx, x in enumerate(self.node_type_list):
                if node_sid is None:
                    if x == cur_node_type:
                        node_sid = idx
                        node_eid = idx
                elif x == cur_node_type:
                    node_eid = idx
            node_eid += 1

            edge_sid = (self.edge_type == 0).sum().item()
            self.edge_index = self.edge_index[:, edge_sid:]
            self.edge_weight = self.edge_weight[edge_sid:]
            self.edge_type = self.edge_type[edge_sid:]
            sel_edge_mask = (self.edge_index[0] >= node_sid) & (self.edge_index[0] < node_eid)
            self.edge_index = self.edge_index[:, sel_edge_mask] - node_sid
            self.edge_weight = self.edge_weight[sel_edge_mask]
            self.edge_type = self.edge_type[sel_edge_mask]
            self.config.num_edges = self.edge_weight.shape[0]

            self.node_type = torch.zeros_like(self.node_type[node_sid:node_eid])
            self.node_type_list = [0] * (node_eid-node_sid)
            self.config.num_node_types = 1
            self.node_name = self.node_name[node_sid:node_eid]
            self.node_weight = self.node_weight[cur_node_type:cur_node_type+1]
            self.use_node_weight = False
            if self.node_popu is not None:
                self.node_popu = self.node_popu[node_sid:node_eid]
            self.nodes = self.nodes[node_sid:node_eid]
            self.config.num_nodes = node_eid - node_sid

            self.train_day_inputs = self.train_day_inputs[:, node_sid:node_eid]
            self.train_day_gov_inputs = self.train_day_gov_inputs[:, node_sid:node_eid]
            self.train_gbm_outputs = self.train_gbm_outputs[:, node_sid:node_eid]
            self.train_outputs = self.train_outputs[:, node_sid:node_eid]

            self.val_day_inputs = self.val_day_inputs[:, node_sid:node_eid]
            self.val_day_gov_inputs = self.val_day_gov_inputs[:, node_sid:node_eid]
            self.val_gbm_outputs = self.val_gbm_outputs[:, node_sid:node_eid]
            self.val_outputs = self.val_outputs[:, node_sid:node_eid]

            self.test_day_inputs = self.test_day_inputs[:, node_sid:node_eid]
            self.test_day_gov_inputs = self.test_day_gov_inputs[:, node_sid:node_eid]
            self.test_gbm_outputs = self.test_gbm_outputs[:, node_sid:node_eid]
            self.test_outputs = self.test_outputs[:, node_sid:node_eid]
        else:
            pass

    def init_data(self, data_fp=None):
        if data_fp is None:
            data_fp = self.config.data_fp

        # load data
        self.county_node, self.state_node = [], []
        self.config.label_fea_name = f'{self.config.label[:-7]}.rolling({self.config.horizon}).sum()'
        day_inputs, day_gov_inputs, outputs, dates, nodes,\
        self.main_feas, self.gov_feas, self.node_popu, self.fea_scaler = \
        load_new_data(data_fp, self.config, logger=self.log)
        for location in nodes:
            if '~' in location:
                self.county_node.append(location)
            else:
                self.state_node.append(location)
        self.state_node.remove('US')
        self.country_node = ['US']
        self.country_node_index = [0]
        country_node_index = self.country_node_index
        used_state_node = self.state_node
        used_county_node = self.county_node
        self.state_node_index = list(range(country_node_index[0] + 1, len(used_state_node)+1))
        self.county_node_index = list(range(len(self.state_node_index)+1, len(used_county_node)+len(used_state_node)+1))
        state_node_dict = dict(zip(self.state_node, self.state_node_index))
        county_node_dict = dict(zip(self.county_node, self.county_node_index))
        mapper_dict = {}
        for pair in used_county_node:
            tmp_pair = pair.split(' ~ ')
            mapper_dict[county_node_dict[pair]] = state_node_dict[tmp_pair[0]]
        if not os.path.exists('../data/state_index.json'):
            with open('../data/state_index.json', 'w+') as f_state:
                json.dump(state_node_dict, f_state)
        if not os.path.exists('../data/county_index.json'):
            with open('../data/county_index.json', 'w+') as f_county:
                json.dump(county_node_dict, f_county)
        if not os.path.exists('../data/mapper.json'):
            with open('../data/mapper.json', 'w+') as f_mapper:
                json.dump(mapper_dict, f_mapper)

        self.config.adapt_norm_eps = 1
        self.config.label_fea_idx = dict(zip(self.main_feas, range(len(self.main_feas))))[self.config.label_fea_name]
        if self.node_popu is not None:
            self.node_popu = self.node_popu.to(self.device)

        gbm_outputs = outputs
        # numpy default dtype is float64, but torch default dtype is float32
        self.day_inputs = day_inputs
        self.day_gov_inputs = day_gov_inputs
        self.outputs = outputs
        self.gbm_outputs = gbm_outputs
        self.dates = dates  # share index with sample id
        self.nodes = nodes  # share index with node id

        # fulfill config
        self.config.num_nodes = self.day_inputs.shape[1]
        self.config.day_seq_len = self.day_inputs.shape[2]
        self.config.day_fea_dim = self.day_inputs.shape[3]
        self.config.day_gov_fea_dim = self.day_gov_inputs.shape[3]
        # self.config.edge_fea_dim = self.edge_attr.shape[1]

        # Filter by label dates
        use_dates = [
            pd.to_datetime(item) for item in dates
            if pd.to_datetime(item) <= pd.to_datetime(self.config.forecast_date)
        ]
        test_divi = len(use_dates) - 1
        val_divi = test_divi - self.config.horizon
        train_divi = val_divi - self.config.val_days
        if self.config.infer:
            # use all achieved train data
            train_divi = val_divi + 1

        print(dates[train_divi], dates[val_divi], dates[test_divi])

        self.train_day_inputs = self.day_inputs[:train_divi+1]
        self.train_day_gov_inputs = self.day_gov_inputs[:train_divi+1]
        self.train_gbm_outputs = self.gbm_outputs[:train_divi+1]
        self.train_outputs = self.outputs[:train_divi+1]
        self.train_dates = self.dates[:train_divi+1]

        if self.config.data_aug_scales != '1':
            data_aug_scales = [float(s) for s in self.config.data_aug_scales.split(',')]
            scale_fea_end = -1
            print(f'Data Augmentation Scaling {data_aug_scales} for {self.main_feas[:scale_fea_end]}')
            def aug_scale(day_input, is_label=False):
                if is_label:
                    aug_inputs = [day_input * s for s in data_aug_scales]
                else:
                    scale_part = day_input[:, :, :, :scale_fea_end]
                    invar_part = day_input[:, :, :, scale_fea_end:]
                    aug_inputs = []
                    for s in data_aug_scales:
                        aug_part = scale_part * s
                        aug_part = torch.cat([aug_part, invar_part], dim=-1)
                        aug_inputs.append(aug_part)
                aug_input = torch.cat(aug_inputs, dim=0)
                return aug_input
            self.train_day_inputs = aug_scale(self.train_day_inputs)
            self.train_day_gov_inputs = aug_scale(self.train_day_gov_inputs)
            self.train_gbm_outputs = aug_scale(self.train_gbm_outputs, is_label=True)
            self.train_outputs = aug_scale(self.train_outputs, is_label=True)
            self.train_dates = self.train_dates * len(data_aug_scales)

        if self.config.infer:
            self.val_day_inputs = self.day_inputs[:train_divi+1]
            self.val_day_gov_inputs = self.day_gov_inputs[:train_divi+1]
            self.val_gbm_outputs = self.gbm_outputs[:train_divi+1]
            self.val_outputs = self.outputs[:train_divi+1]
            self.val_dates = self.dates[:train_divi+1]
        else:
            self.val_day_inputs = self.day_inputs[val_divi:val_divi+1]
            self.val_day_gov_inputs = self.day_gov_inputs[val_divi:val_divi+1]
            self.val_gbm_outputs = self.gbm_outputs[val_divi:val_divi+1]
            self.val_outputs = self.outputs[val_divi:val_divi+1]
            self.val_dates = self.dates[val_divi:val_divi+1]

        self.test_day_inputs = self.day_inputs[test_divi:test_divi+1]
        self.test_day_gov_inputs = self.day_gov_inputs[test_divi:test_divi+1]
        self.test_gbm_outputs = self.gbm_outputs[test_divi:test_divi+1]
        self.test_outputs = self.outputs[test_divi:test_divi+1]
        self.test_dates = self.dates[test_divi:test_divi+1]

    def init_graph(self, graph_fp=None):
        if graph_fp is None:
            graph_fp = self.config.graph_fp
        graph_dict = torch.load(graph_fp)

        self.edge_index = graph_dict['edge_index']
        self.edge_weight = graph_dict['edge_weight']
        self.edge_type = graph_dict['edge_type'].to(self.device)
        self.node_type = graph_dict['node_type'].to(self.device)
        self.node_type_list = list(graph_dict['node_type'].numpy())

        self.node_name = graph_dict['node_name']
        if self.config.num_nodes != len(self.node_name):
            data_node_set = set(self.nodes)
            graph_node_set = set(self.node_name)
            print('New nodes in data', data_node_set - graph_node_set)
            print('Missing nodes in data', graph_node_set - data_node_set)
            raise Exception('Please regenerate GNN topo before running')
        self.config.num_edges = self.edge_weight.shape[0]
        self.config.num_node_types = int(graph_dict['node_type'].max()) + 1
        self.config.num_edge_types = int(graph_dict['edge_type'].max()) + 1

        base_ones = torch.ones_like(self.node_type, dtype=torch.float)
        node_type_count = scatter(base_ones, self.node_type, dim_size=self.config.num_node_types, reduce='sum')
        # the weight of the bottom nodes is equal to 1
        self.node_weight = 1.0 / node_type_count * node_type_count.max()

    def make_sample_dataloader(self, day_inputs, day_gov_inputs, gbm_outputs, outputs, shuffle=False):
        if self.config.use_saintdataset:
            lg.log(lg.INFO, 'Data Sampling...')
            dataset = SAINTDataset(
                [day_inputs, day_gov_inputs, gbm_outputs, outputs],
                self.edge_index, self.edge_weight, self.config.num_nodes,
                self.config.batch_size, shuffle=shuffle,
                shuffle_order=self.config.saint_shuffle_order,
                saint_sample_type=self.config.saint_sample_type,
                saint_batch_size=self.config.saint_batch_size,
                saint_walk_length=self.config.saint_walk_length,
            )

            return DataLoader(dataset, batch_size=None)
        else:
            dataset = SimpleDataset([day_inputs, day_gov_inputs, gbm_outputs, outputs])
            def collate_fn(samples):
                day_inputs = torch.cat([item[0][0] for item in samples]).unsqueeze(0)   # [1,bs,seq_length,feature_dim]
                day_gov_inputs = torch.cat([item[0][1] for item in samples]).unsqueeze(0)   # [1,bs,seq_length,feature_dim]
                gbm_outputs = torch.cat([item[0][-2] for item in samples]).unsqueeze(0)
                outputs = torch.cat([item[0][-1] for item in samples]).unsqueeze(0)
                node_ids = torch.LongTensor([item[1] for item in samples])   # [bs]
                date_ids = torch.LongTensor([item[2] for item in samples])   # [bs]
                return [[day_inputs, day_gov_inputs, gbm_outputs, outputs], {'cent_n_id':node_ids,'type':'random'}, date_ids]

            return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle, collate_fn=collate_fn)

    def build_train_dataloader(self):
        return self.make_sample_dataloader(
            self.train_day_inputs, self.train_day_gov_inputs, self.train_gbm_outputs, self.train_outputs, shuffle=True
        )

    def build_val_dataloader(self):
        return self.make_sample_dataloader(
            self.val_day_inputs, self.val_day_gov_inputs, self.val_gbm_outputs, self.val_outputs, shuffle=False
        )

    def build_test_dataloader(self):
        return self.make_sample_dataloader(
            self.test_day_inputs, self.test_day_gov_inputs, self.test_gbm_outputs, self.test_outputs, shuffle=False
        )

    def build_optimizer(self, model):
        model_opt = torch.optim.Adam(self.model.get_net_parameters(), lr=self.config.lr)
        if self.config.optim_graph:
            kwargs = {
                'lr': self.config.graph_lr,
            }

            if self.config.graph_opt_type == 'Adam':
                opt_class = torch.optim.Adam
            elif self.config.graph_opt_type == 'TruncateSGD':
                kwargs['gravity'] = self.config.graph_gravity
                opt_class = TruncateSGD
            elif self.config.graph_opt_type == 'TruncateAdam':
                kwargs['gravity'] = self.config.graph_gravity
                kwargs['lr_truncate'] = self.config.graph_lr
                opt_class = TruncateAdam
            else:
                raise Exception("Unsupported graph optimizer '{}'".format(self.config.graph_opt_type))

            graph_opt = opt_class(self.model.get_graph_parameters(), **kwargs)

            return model_opt, graph_opt
        else:
            return model_opt

    def train_step(self, batch, batch_idx):
        inputs, g, _ = batch
        # prepare inputs, outputs
        input_day, input_day_gov, y_gbm, y = inputs
        if self.config.use_gbm:  # deprecated
            y = y - y_gbm
        if self.config.use_adapt_norm:
            norm_eps = self.config.adapt_norm_eps
            input_norm = input_day.mean(dim=-2, keepdim=True) + norm_eps
            y_norm = input_norm[:, :, :, self.config.label_fea_idx] + norm_eps
            input_day = (input_day+norm_eps) / input_norm
            y = (y+norm_eps) / y_norm
        else:
            norm_eps = 0
            input_norm = 1
            y_norm = 1
        # prepare graph
        g['edge_type'] = self.edge_type[g['e_id']]
        g['node_type'] = self.node_type[g['cent_n_id']]
        if self.config.optim_graph:
            g['edge_attr_prior'] = g['edge_attr']
            g['edge_attr'] = self.model.edge_weight[g['e_id']]

        y_hat, _, _ = self.model(input_day, input_day_gov, g)
        assert(y.size() == y_hat.size())

        if self.config.use_node_weight:
            node_weight = self.node_weight[g['node_type']]\
                .reshape(1, y.shape[1], 1)
            loss = weighted_mse_loss(y_hat, y, node_weight)
        else:
            node_weight = None
            loss = self.loss_func(y_hat, y)
        y_loss_i = loss.item()

        if self.config.optim_graph:
            graph_loss = self.loss_func(g['edge_attr'], g['edge_attr_prior'])
            loss += self.config.graph_eta * graph_loss

        if self.config.topo_loss_weight > 0 and \
            self._current_epoch >= self.config.topo_loss_epoch_start:
            # get topo_edge_index
            edge_index = g['edge_index']
            node_type = g['node_type']
            i, j = 1, 0
            node_type_j = node_type[edge_index[j]]
            node_type_i = node_type[edge_index[i]]
            topo_edge_index = edge_index[:, node_type_i == node_type_j-1]
            # calculate aggregated y
            if self.config.use_adapt_norm:
                y = y * y_norm - norm_eps
                y_hat = y_hat * y_norm - norm_eps
            if self.config.use_logy:
                y = y.expm1()  # exp(y)-1, where y = log(1+label)
                y_hat = y_hat.expm1()  # exp(y_hat)-1, where y = log(1+label_hat)
            if self.config.use_popu_norm:
                popu = self.node_popu[g['cent_n_id']]\
                    .reshape(1, g['cent_n_id'].shape[0], 1)
                y = y * popu / 10**5
                y_hat = y_hat * popu / 10**5
            y_j = y[:, topo_edge_index[j], :]
            y_hat_j = y_hat[:, topo_edge_index[j], :]
            y_agg = scatter(y_j, topo_edge_index[i], dim=-2, dim_size=y.shape[-2], reduce='sum')
            y_hat_agg = scatter(y_hat_j, topo_edge_index[i], dim=-2, dim_size=y_hat.shape[-2], reduce='sum')
            # use agg mask to ignore bottom node
            bottom_node_type = node_type.max()
            agg_mask = node_type < bottom_node_type
            ym = y[:, agg_mask]
            ym_hat = y_hat[:, agg_mask]
            ym_agg = y_agg[:, agg_mask]
            ym_hat_agg = y_hat_agg[:, agg_mask]
            eps = self.config.mape_eps
            topo_loss = self.loss_func((ym_hat_agg+eps)/(ym_agg+eps), torch.ones_like(ym_agg)) + \
                self.loss_func((ym_hat_agg+eps)/(ym_agg+eps), (ym_hat+eps)/(ym+eps),)
            loss += self.config.topo_loss_weight * topo_loss
            topo_loss_i = topo_loss.item()
        else:
            topo_loss_i = 0

        # judge to avoid useless computation
        if self.config.pair_loss_node_num != 0 and self.config.pair_loss_weight > 0:
            pair_edge_index = g['edge_index']  # consider every pair in the graph
            if self.config.pair_loss_node_num > 0:
                num_edges = pair_edge_index.shape[1]
                rand_eids = torch.randperm(num_edges, device=loss.device)[:self.config.pair_loss_node_num]
                pair_edge_index = pair_edge_index[:, rand_eids]
            i, j = 1, 0
            logy_j = y[:, pair_edge_index[j], :]
            logy_i = y[:, pair_edge_index[i], :]
            logy_j_hat = y_hat[:, pair_edge_index[j], :]
            logy_i_hat = y_hat[:, pair_edge_index[i], :]
            pair_loss = weighted_mse_loss(
                (logy_j_hat - logy_j).exp(),  # (y_j_hat+1) / (y_j+1)
                (logy_i_hat - logy_i).exp(),  # (y_i_hat+1) / (y_i+1)
                0.5*(logy_j + logy_i),  # pay more attention to large nodes
            )
            loss += self.config.pair_loss_weight * pair_loss
            pair_loss_i = pair_loss.item()
        else:
            pair_loss_i = 0

        if self.config.sparse_gate_weight > 0:
            gate_loss = self.model.net.net.gcn_coef.mean()
            if self._current_epoch >= self.config.sparse_gate_epoch_start:
                loss += self.config.sparse_gate_weight * gate_loss
            gate_loss_i = gate_loss.item()
        else:
            gate_loss_i = 0

        loss_i = loss.item()  # scalar loss
        # log all kinds of losses for debug
        loss_info = {
            'loss': loss_i,
            'y_loss': y_loss_i,
            'topo_loss': topo_loss_i,
            'pair_loss': pair_loss_i,
            'gate_loss': gate_loss_i,
        }

        return {
            LOSS_KEY: loss,
            BAR_KEY: loss_info,
            SCALAR_LOG_KEY: loss_info,
        }

    def eval_step(self, batch, batch_idx, tag):
        inputs, g, rows = batch
        input_day, input_day_gov, y_gbm, y = inputs
        if self.config.use_adapt_norm:
            norm_eps = self.config.adapt_norm_eps
            input_norm = input_day.mean(dim=-2, keepdim=True) + norm_eps
            y_norm = input_norm[:, :, :, self.config.label_fea_idx] + norm_eps
            input_day = (input_day + norm_eps) / input_norm
        else:
            norm_eps = 0
            input_norm = 1
            y_norm = 1
        forecast_length = y.size()[-1]
        g['edge_type'] = self.edge_type[g['e_id']]
        g['node_type'] = self.node_type[g['cent_n_id']]
        if self.config.optim_graph:
            g['edge_attr_prior'] = g['edge_attr']
            g['edge_attr'] = self.model.edge_weight[g['e_id']]

        y_hat, state_gate, gov_gate = self.model(input_day, input_day_gov, g)
        if self.config.use_gbm:
            y_hat += y_gbm
        assert(y.size() == y_hat.size())

        if self.config.use_adapt_norm:
            y_hat = y_hat * y_norm - norm_eps
        if self.config.use_logy:
            y = y.expm1()  # exp(y)-1, where y = log(1+label)
            y_hat = y_hat.expm1()  # exp(y_hat)-1, where y = log(1+label_hat)
        if self.config.use_popu_norm:
            popu = self.node_popu[g['cent_n_id']]\
                .reshape(1, g['cent_n_id'].shape[0], 1)
            y = y * popu / 10**5
            y_hat = y_hat * popu / 10**5

        if g['type'] == 'subgraph' and 'res_n_id' in g:  # if using SAINT sampler
            cent_n_id = g['cent_n_id']
            res_n_id = g['res_n_id']
            # Note: we only evaluate predictions on those initial nodes (per random walk)
            # to avoid duplicated computations
            y = y[:, res_n_id]
            y_hat = y_hat[:, res_n_id]
            cent_n_id = cent_n_id[res_n_id]
        else:
            cent_n_id = g['cent_n_id']

        if self.config.use_saintdataset:
            index_ptr = torch.cartesian_prod(
                torch.arange(rows.size(0)),
                torch.arange(cent_n_id.size(0)),
                torch.arange(forecast_length)
            )

            label = pd.DataFrame({
                'row_idx': rows[index_ptr[:, 0]].data.cpu().numpy(),
                'node_idx': cent_n_id[index_ptr[:, 1]].data.cpu().numpy(),
                'forecast_idx': index_ptr[:,2].data.cpu().numpy(),
                'val': y.flatten().data.cpu().numpy()
            })

            pred = pd.DataFrame({
                'row_idx': rows[index_ptr[:, 0]].data.cpu().numpy(),
                'node_idx': cent_n_id[index_ptr[:, 1]].data.cpu().numpy(),
                'forecast_idx': index_ptr[:,2].data.cpu().numpy(),
                'val': y_hat.flatten().data.cpu().numpy()
            })
        else:
            index_ptr = torch.cartesian_prod(
                torch.arange(rows.size(0)),
                torch.arange(forecast_length)
            )

            label = pd.DataFrame({
                'row_idx': rows[index_ptr[:, 0]].data.cpu().numpy(),
                'node_idx': cent_n_id[index_ptr[:, 0]].data.cpu().numpy(),
                'forecast_idx': index_ptr[:,1].data.cpu().numpy(),
                'val': y.flatten().data.cpu().numpy()
            })

            pred = pd.DataFrame({
                'row_idx': rows[index_ptr[:, 0]].data.cpu().numpy(),
                'node_idx': cent_n_id[index_ptr[:, 0]].data.cpu().numpy(),
                'forecast_idx': index_ptr[:,1].data.cpu().numpy(),
                'val': y_hat.flatten().data.cpu().numpy()
            })

        pred = pred.groupby(['row_idx', 'node_idx', 'forecast_idx']).mean()
        label = label.groupby(['row_idx', 'node_idx', 'forecast_idx']).mean()

        return {
            'label': label,
            'pred': pred,
            'info': [state_gate, gov_gate]
            # 'atten': atten_context
        }

    def eval_epoch_end(self, outputs, tag, dates):
        pred = pd.concat([x['pred'] for x in outputs], axis=0)
        label = pd.concat([x['label'] for x in outputs], axis=0)
        pred = pred.groupby(['row_idx', 'node_idx','forecast_idx']).mean()
        label = label.groupby(['row_idx', 'node_idx', 'forecast_idx']).mean()
        info = [x['info'] for x in outputs]
        # atten_context = [x['atten'] for x in outputs]

        align_nodes = label.reset_index().node_idx.map(lambda x: self.nodes[x]).values
        align_dates = label.reset_index().row_idx.map(lambda x: dates[x]).values

        loss = np.mean(np.abs(pred['val'].values - label['val'].values))
        scores = self.produce_score(pred, label, dates)

        log_dict = {
            '{}_loss'.format(tag): loss,
            '{}_mae'.format(tag): scores['mean_mistakes'],
            '{}_mape'.format(tag): scores['mape'],
            # '{}_mean_mistakes'.format(tag): scores['mean_mistakes'],
            # '{}_mean_label'.format(tag): scores['mean_label'],
            # '{}_mean_predict'.format(tag): scores['mean_predict']
        }
        type_mae_sum = 0
        type_mape_sum = 0
        for type_id in range(self.config.num_node_types):
            cur_pred = pred[
                pred.index.get_level_values(1).map(lambda x: self.node_type_list[x]) == type_id
            ]
            cur_label = label[
                label.index.get_level_values(1).map(lambda x: self.node_type_list[x]) == type_id
            ]
            cur_scores = self.produce_score(cur_pred, cur_label, dates)
            log_dict[f'{tag}_type-{type_id}_mae'] = cur_scores['mean_mistakes']
            log_dict[f'{tag}_type-{type_id}_mape'] = cur_scores['mape']
            type_mae_sum += cur_scores['mean_mistakes']
            type_mape_sum += cur_scores['mape']
        log_dict[f'{tag}_type-mean_mae'] = type_mae_sum / self.config.num_node_types
        log_dict[f'{tag}_type-mean_mape'] = type_mape_sum / self.config.num_node_types

        out = {
            BAR_KEY: log_dict,
            SCALAR_LOG_KEY: log_dict,
            VAL_SCORE_KEY: - type_mape_sum,
            'pred': pred,
            'label': label,
            'scores': scores,
            'dates': align_dates,
            'nodes': align_nodes,
            'info': info,
            'y_scale': 'linear',
            'epoch': self._passed_epoch,
            # 'atten': atten_context
        }

        return out

    def produce_score(self, pred, label, dates=None):
        # y_hat = pred.apply(lambda x: np.expm1(x))
        # y = label.apply(lambda x: np.expm1(x))
        y_hat = pred
        y = label
        mape_eps = self.config.mape_eps
        mape_df = np.abs((y_hat+mape_eps)/(y+mape_eps)-1).reset_index(drop=False)

        mape_val = np.abs((y_hat.values+1)/(y.values+1)-1).mean()
        mean_mistakes = np.abs(y_hat.values - y.values).mean()
        mean_label = np.abs(y.values).mean()
        mean_predict = np.abs(y.values).mean()
        eval_df = pd.concat([y_hat.rename(columns={'val': 'pred'}),
                             y.rename(columns={'val': 'label'})],
                            axis=1).reset_index(drop=False)

        eval_df['mape'] = mape_df['val']
        if dates is not None:
            eval_df['date'] = eval_df.row_idx.map(lambda x: dates[x])
        eval_df['nodes'] = eval_df.node_idx.map(lambda x: self.nodes[x])

        def produce_percent_count(m_df):
            res = pd.Series()
            res['pred'] = m_df['pred'].mean()
            res['label'] = m_df['label'].mean()
            res['mistake'] = np.abs(m_df['pred'] - m_df['label']).mean()
            return res

        scores = {
            'mape': mape_val,
            'mean_mistakes': mean_mistakes,
            'mean_label': mean_label,
            'mean_predict': mean_predict
        }
        for name, metric in [
            ('mistakes', eval_df),
        ]:
            scores[name] = metric.groupby(
                'row_idx').apply(produce_percent_count)
            if dates is not None:
                scores[name]['date'] = dates

        return scores

    def val_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'val')

    def val_epoch_end(self, outputs):
        val_out = self.eval_epoch_end(outputs, 'val', self.val_dates)
        return val_out

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'test')

    def test_epoch_end(self, outputs):
        test_out = self.eval_epoch_end(outputs, 'test', self.test_dates)
        return test_out


if __name__ == '__main__':
    start_time = time.time()

    # build argument parser and config
    config = Config()
    parser = argparse.ArgumentParser(description='COVID-19 Forecasting Task')
    add_config_to_argparse(config, parser)

    # parse arguments to config
    args = parser.parse_args()
    config.update_by_dict(args.__dict__)

    # build task
    task = Task(config)
    # Set random seed before the initialization of network parameters
    # Necessary for distributed training
    task.set_random_seed()
    net = GraphNet(task.config, task.edge_weight)
    task.init_model_and_optimizer(net)
    task.log('Build Neural Nets')
    # select epoch with best validation accuracy
    best_epochs = 50
    if not task.config.skip_train:
        task.fit()
        best_epochs = task._best_val_epoch
        print('Best validation epochs: {}'.format(best_epochs))

    # Resume the best checkpoint for evaluation
    task.resume_best_checkpoint()
    val_eval_out = task.val_eval()
    test_eval_out = task.test_eval()
    # dump evaluation results of the best checkpoint to val out
    task.dump(val_out=val_eval_out,
              test_out=test_eval_out,
              epoch_idx=-1,
              is_best=True,
              dump_option=1)
    task.log('Best checkpoint (epoch={}, {}, {})'.format(
        task._passed_epoch, val_eval_out[BAR_KEY], test_eval_out[BAR_KEY]))

    if task.is_master_node:
        for tag, eval_out in [
            ('val', val_eval_out),
            ('test', test_eval_out),
        ]:
            print('-'*15, tag)
            scores = eval_out['scores']['mistakes']
            print('-'*5, 'mistakes')
            print('Average:')
            print(scores.mean().to_frame('mistakes'))
            print('Daily:')
            print(scores)

    task.log('Training time {}s'.format(time.time() - start_time))
