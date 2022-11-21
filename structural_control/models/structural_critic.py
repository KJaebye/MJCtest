# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class StructuralValue
#   @author: Modified from khrylib by Ye Yuan, modified by Kangyao Huang
#   @created date: 12.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #
import torch
from lib.models.mlp import MLP
from lib.utils.tools import init_fc_weights
from lib.core.running_norm import RunningNorm


class StructuralValue(torch.nn.Module):
    def __init__(self, cfg_spec, agent):
        super().__init__()
        self.cfg_spec = cfg_spec
        self.agent = agent
        self.observation_dim = agent.observation_dim
        cur_dim = self.observation_dim
        self.norm = RunningNorm(self.observation_dim)
        if 'pre_mlp' in cfg_spec:
            self.pre_mlp = MLP(cur_dim, cfg_spec['pre_mlp'], cfg_spec['htype'])
            cur_dim = self.pre_mlp.output_dim
        else:
            self.pre_mlp = None
        if 'gnn_spec' in cfg_spec:
            self.gnn = GNNSimple(cur_dim, cfg_spec['gnn_spec'])
            cur_dim = self.gnn.out_dim
        else:
            self.gnn = None
        if 'mlp' in cfg_spec:
            self.mlp = MLP(cur_dim, cfg_spec['mlp'], cfg_spec['htype'])
            cur_dim = self.mlp.output_dim
        else:
            self.mlp = None
        self.value_head = torch.nn.Linear(cur_dim, 1)
        init_fc_weights(self.value_head)

    def batch_data(self):
        pass

    def forward(self, x):
        x = self.norm(x)
        if self.pre_mlp is not None:
            x = self.pre_mlp(x)
        if self.gnn is not None:
            x = self.gnn(x)
        if self.mlp is not None:
            x = self.mlp(x)
        print(len(x))
        value = self.value_head(x)
        return value
