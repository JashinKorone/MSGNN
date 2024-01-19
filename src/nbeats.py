import numpy as np
import torch
import math
from torch import nn
from torch.nn import functional as F

class NBeatsNet(nn.Module):

    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    GENERIC_BLOCK = 'generic'

    def __init__(self,
                 input_channels=1,
                 stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
                 nb_blocks_per_stack=3,
                 forecast_length=14,
                 backcast_length=1,
                 thetas_dims=(4, 8),
                 share_weights_in_stack=False,
                 hidden_channels=64):
        super(NBeatsNet, self).__init__()
        self.input_channels = input_channels
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.hidden_channels = hidden_channels
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.stack_types = stack_types
        self.stacks = []
        self.thetas_dim = thetas_dims
        self.parameters = []
        self.conv1d_flatten = nn.Conv1d(input_channels, hidden_channels, kernel_size=3, stride=1, dilation=1, padding=1)
        self.parameters.extend(self.conv1d_flatten.parameters())
        # print(f'| N-Beats')
        # print(f'     | -- {self.conv1d_flatten}')
        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))
        self.parameters = nn.ParameterList(self.parameters)

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        # print(f'| --  Stack {stack_type.title()} (#{stack_id}) (share_weights_in_stack={self.share_weights_in_stack})')
        blocks = []
        for block_id in range(self.nb_blocks_per_stack):
            block_init = NBeatsNet.select_block(stack_type)
            if self.share_weights_in_stack and block_id != 0:
                block = blocks[-1]  # pick up the last one when we share weights.
            else:
                block = block_init(self.hidden_channels, self.thetas_dim[stack_id],
                                   self.backcast_length, self.forecast_length)
                self.parameters.extend(block.parameters())
            # print(f'     | -- {block}')
            blocks.append(block)
        return blocks

    @staticmethod
    def select_block(block_type):
        if block_type == NBeatsNet.SEASONALITY_BLOCK:
            return SeasonalityBlock
        elif block_type == NBeatsNet.TREND_BLOCK:
            return TrendBlock
        else:
            return GenericBlock

    def forward(self, backcast):
        forecast = torch.zeros(size=(backcast.size()[0], self.forecast_length)).to(backcast.get_device())  # maybe batch size here.
        backcast = self.conv1d_flatten(backcast)     # (batch_size, input_channels, backcast_length) --> (batch_size, hidden_channels, backcast_length)
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast)
                backcast = backcast - b
                forecast = forecast + f
        return backcast, forecast


def seasonality_model(thetas, t):
    p = thetas.size()[-1]
    assert p <= thetas.shape[1], 'thetas_dim is too big.'
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    s1 = torch.FloatTensor([np.cos(2 * np.pi * i * t) for i in range(p1)])  # H/2-1
    s2 = torch.FloatTensor([np.sin(2 * np.pi * i * t) for i in range(p2)])
    S = torch.cat([s1, s2]).to(thetas.get_device())
    return thetas.matmul(S)


def trend_model(thetas, t):
    p = thetas.size()[-1]
    assert p <= 4, 'thetas_dim is too big.'
    T = torch.FloatTensor([t ** i for i in range(p)]).to(thetas.get_device())
    return thetas.matmul(T)


def linspace(backcast_length, forecast_length):
    lin_space = np.linspace(-backcast_length, forecast_length, backcast_length + forecast_length, endpoint=False)
    b_ls = lin_space[:backcast_length] / (backcast_length + forecast_length)
    f_ls = lin_space[backcast_length:] / (backcast_length + forecast_length)
    return b_ls, f_ls


class Block(nn.Module):

    def __init__(self, hidden_channels, thetas_dim, backcast_length=10, forecast_length=5,share_thetas=False,
                 nb_harmonics=None):
        super(Block, self).__init__()
        self.hidden_channels = hidden_channels
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas

        # (batch_size, hidden_channels, backcast_length)
        self.conv1d_1 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=1, dilation=1, padding=1)   # zero_padding to align length
        self.conv1d_2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=1, dilation=1, padding=1)
        self.conv1d_3 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=1, dilation=1, padding=1)
        self.conv1d_4 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=1, dilation=1, padding=1)

        # (batch_size, hidden_channels, thetas_dim)
        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Linear(backcast_length, thetas_dim)
        else:
            self.theta_b_fc = nn.Linear(backcast_length, thetas_dim)
            self.theta_f_fc = nn.Linear(backcast_length, thetas_dim)


        self.backcast_linspace, self.forecast_linspace = linspace(backcast_length, forecast_length)

    def forward(self, x):
        x = F.relu(self.conv1d_1(x))
        x = F.relu(self.conv1d_2(x))
        x = F.relu(self.conv1d_3(x))
        x = F.relu(self.conv1d_4(x))
        return x


class SeasonalityBlock(Block):

    def __init__(self, hidden_channels, thetas_dim, backcast_length=10, forecast_length=5):
        super(SeasonalityBlock, self).__init__(hidden_channels, forecast_length, backcast_length,
                                                   forecast_length, share_thetas=True)

        self.forecast_pool = nn.AdaptiveMaxPool1d(1)
        self.forecast_map = nn.Linear(hidden_channels, forecast_length)

    def forward(self, x):
        x = super(SeasonalityBlock, self).forward(x)
        backcast = seasonality_model(self.theta_b_fc(x), self.backcast_linspace)   # (batch_size, hidden_channels, backcast_length)
        forecast = seasonality_model(self.theta_f_fc(x), self.forecast_linspace)   # (batch_size, hidden_channels, forecast_length)
        forecast = self.forecast_map(self.forecast_pool(forecast).squeeze())       # (batch_size, forecast_length)

        return backcast, forecast


class TrendBlock(Block):

    def __init__(self, hidden_channels, thetas_dim,   backcast_length=10, forecast_length=5):
        super(TrendBlock, self).__init__(hidden_channels, thetas_dim, backcast_length,
                                         forecast_length, share_thetas=True)
        self.forecast_pool = nn.AdaptiveMaxPool1d(1)
        self.forecast_map = nn.Linear(hidden_channels, forecast_length)

    def forward(self, x):
        x = super(TrendBlock, self).forward(x)
        backcast = trend_model(self.theta_b_fc(x), self.backcast_linspace)
        forecast = trend_model(self.theta_f_fc(x), self.forecast_linspace)
        forecast = self.forecast_map(self.forecast_pool(forecast).squeeze())       # (batch_size, forecast_length)

        return backcast, forecast


class GenericBlock(Block):

    def __init__(self, hidden_channels, thetas_dim,  backcast_length=10, forecast_length=5):
        super(GenericBlock, self).__init__(hidden_channels, thetas_dim, backcast_length, forecast_length)

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)
        self.forecast_pool = nn.AdaptiveMaxPool1d(1)
        self.forecast_map = nn.Linear(hidden_channels, forecast_length)

    def forward(self, x):
        x = super(GenericBlock, self).forward(x)

        theta_b = F.relu(self.theta_b_fc(x))
        theta_f = F.relu(self.theta_f_fc(x))

        backcast = self.backcast_fc(theta_b)
        forecast = self.forecast_map(self.forecast_pool(F.relu(self.forecast_fc(theta_f))).squeeze())

        return backcast, forecast


class NBeatsModel(nn.Module):
    def __init__(self, config):
        super(NBeatsModel, self).__init__()
        self.week_em = nn.Embedding(7, config.date_emb_dim)
        self.id_em = nn.Embedding(config.num_nodes, config.id_emb_dim)
        self.lookahead_days = config.lookahead_days
        self.lookback_days = config.lookback_days

        day_input_dim = config.day_fea_dim - 1 + self.week_em.embedding_dim + self.id_em.embedding_dim

        self.day_n_beats = NBeatsNet(
                                    input_channels=day_input_dim,
                                    stack_types=('generic', 'generic'),
                                    nb_blocks_per_stack=config.block_size,
                                    forecast_length=config.lookahead_days,
                                    backcast_length=config.lookback_days,
                                    thetas_dims=(4, 8),
                                    share_weights_in_stack=True,
                                    hidden_channels=config.hidden_dim,
                                    )


    def add_date_embed(self, input_day):
        # last 1 dims correspond to weekday
        x = input_day[:, :, :, :-1]
        weekday = self.week_em(input_day[:, :, :, -1].long())
        # return torch.cat([x, month, day, weekday], dim=-1)
        return torch.cat([x, weekday], dim=-1)

    def add_id_embed(self, input, g):
        # add cent id index
        sz = input.size()
        cent_id = self.id_em(g['cent_n_id'].reshape(1,sz[1],1).expand(sz[0],sz[1],sz[2]).long())
        return torch.cat([input, cent_id], dim=-1)

    def permute_feature(self, x):
        sz = x.size()
        x = x.view(-1, sz[-2], sz[-1])
        x = x.permute(0, 2, 1)
        return x

    def forward(self, input_day, g):
        sz = input_day.size()
        # forecast_lr = self.linear_reg(input_day)
        input_day = self.add_date_embed(input_day)
        input_day = self.add_id_embed(input_day, g)
        input_day = self.permute_feature(input_day)

        _, forecast_day = self.day_n_beats(input_day)

        pred = forecast_day.squeeze().view(sz[0],sz[1],self.lookahead_days)
        return pred


class NBeatsEncoder(nn.Module):
    def __init__(self, config, hidden_dim):
        super().__init__()
        self.week_em = nn.Embedding(7, config.date_emb_dim)
        self.id_em = nn.Embedding(config.num_nodes, config.id_emb_dim)
        self.lookahead_days = config.lookahead_days
        self.lookback_days = config.lookback_days
        self.hidden_dim = hidden_dim

        day_input_dim = config.day_fea_dim - 1 + self.week_em.embedding_dim + self.id_em.embedding_dim

        self.day_n_beats = NBeatsNet(
            input_channels=day_input_dim,
            stack_types=('generic', 'generic'),
            nb_blocks_per_stack=config.block_size,
            forecast_length=config.lookahead_days,
            backcast_length=config.lookback_days,
            thetas_dims=(4, 8),
            share_weights_in_stack=True,
            hidden_channels=hidden_dim)

    def add_date_embed(self, input_day):
        # last 1 dims correspond to weekday
        x = input_day[:, :, :, :-1]
        weekday = self.week_em(input_day[:, :, :, -1].long())
        # return torch.cat([x, month, day, weekday], dim=-1)
        return torch.cat([x, weekday], dim=-1)

    def add_id_embed(self, input, g):
        # add cent id index
        sz = input.size()
        cent_id = self.id_em(g['cent_n_id'].reshape(1,sz[1],1).expand(sz[0],sz[1],sz[2]).long())
        return torch.cat([input, cent_id], dim=-1)

    def permute_feature(self, x):
        sz = x.size()
        x = x.view(-1, sz[-2], sz[-1])
        x = x.permute(0, 2, 1)
        return x

    def forward(self, input_day, g):
        # [batch_size, node_num, seq_len, fea_dim]
        sz = input_day.size()
        # forecast_lr = self.linear_reg(input_day)
        input_day = self.add_date_embed(input_day)
        input_day = self.add_id_embed(input_day, g)
        input_day = self.permute_feature(input_day)

        backcast_day, forecast_day = self.day_n_beats(input_day)

        backcast_day = backcast_day.view(sz[0], sz[1], self.hidden_dim, self.lookback_days)
        forecast_day = forecast_day.squeeze().view(sz[0], sz[1], self.lookahead_days)

        return backcast_day, forecast_day