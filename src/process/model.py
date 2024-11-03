import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import GatedGraphConv
from torch_geometric.nn import global_mean_pool, global_sort_pool
from torch_geometric.nn.aggr import Aggregation

torch.manual_seed(2020)


def get_conv_mp_out_size(in_size, last_layer, mps):
    size = in_size

    for mp in mps:
        size = round((size - mp["kernel_size"]) / mp["stride"] + 1)

    size = size + 1 if size % 2 != 0 else size

    return int(size * last_layer["out_channels"])


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)


class Net(nn.Module):
    def __init__(self, gated_graph_conv_args, emb_size, device):
        super(Net, self).__init__()
        if gated_graph_conv_args["out_channels"] == -1:
            gated_graph_conv_args["out_channels"] = emb_size
        self.inp_mlp = nn.Linear(emb_size, gated_graph_conv_args["out_channels"]).to(
            device
        )
        self.ggc = GatedGraphConv(**gated_graph_conv_args).to(device)
        self.k = 5
        self.l_o1 = nn.Linear(
            gated_graph_conv_args["out_channels"] * self.k,
            gated_graph_conv_args["out_channels"],
        ).to(device)
        self.l_o2 = nn.Linear(
            gated_graph_conv_args["out_channels"],
            gated_graph_conv_args["out_channels"],
        ).to(device)
        self.l_o = nn.Linear(gated_graph_conv_args["out_channels"] * 2, 2).to(device)
        self.dropout = nn.Dropout(p=0.1)

        self.aggr = Aggregation().to(device)

        self.conv_l1 = torch.nn.Conv1d(
            gated_graph_conv_args["out_channels"],
            gated_graph_conv_args["out_channels"],
            3,
        ).to(device)
        self.maxpool1 = torch.nn.MaxPool1d(3, stride=2).to(device)
        self.conv_l2 = torch.nn.Conv1d(
            gated_graph_conv_args["out_channels"],
            gated_graph_conv_args["out_channels"],
            1,
        ).to(device)
        self.maxpool2 = torch.nn.MaxPool1d(2, stride=2).to(device)

        self.concat_dim = gated_graph_conv_args["out_channels"] * 2
        self.conv_l1_for_concat = torch.nn.Conv1d(
            self.concat_dim, self.concat_dim, 3
        ).to(device)
        self.maxpool1_for_concat = torch.nn.MaxPool1d(3, stride=2).to(device)
        self.conv_l2_for_concat = torch.nn.Conv1d(
            self.concat_dim, self.concat_dim, 1
        ).to(device)
        self.maxpool2_for_concat = torch.nn.MaxPool1d(2, stride=2).to(device)

        self.mlp_z = nn.Linear(in_features=self.concat_dim, out_features=2).to(device)
        self.mlp_y = nn.Linear(
            in_features=gated_graph_conv_args["out_channels"], out_features=2
        ).to(device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        data_x = self.inp_mlp(data.x)
        x, edge_index = data_x, data.edge_index
        batch_index = data.batch
        x = self.ggc(x, edge_index)

        # fill_value = x.detach().min() - 1
        x_i, _ = self.aggr.to_dense_batch(data_x, batch_index)
        x, _ = self.aggr.to_dense_batch(x, batch_index)
        c_i = torch.cat((x, x_i), dim=-1)
        Y_1 = self.maxpool1(F.relu(self.conv_l1(x.transpose(1, 2))))
        Y_2 = self.maxpool2(F.relu(self.conv_l2(Y_1))).transpose(1, 2)
        Z_1 = self.maxpool1_for_concat(
            F.relu(self.conv_l1_for_concat(c_i.transpose(1, 2)))
        )
        Z_2 = self.maxpool2_for_concat(F.relu(self.conv_l2_for_concat(Z_1))).transpose(
            1, 2
        )
        before_avg = torch.mul(
            self.mlp_y(self.dropout(Y_2)), self.mlp_z(self.dropout(Z_2))
        )
        avg = before_avg.mean(dim=1)
        # result = self.sigmoid(avg).squeeze(dim=-1)
        result = avg
        return result

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))