import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, scatter

class CMPNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CMPNNConv, self).__init__()
        
        self.lin = nn.Linear(in_channels, out_channels)
        self.theta1 = nn.Parameter(torch.randn(in_channels, out_channels))
        self.theta2 = nn.Parameter(torch.randn(in_channels, out_channels))
        self.theta_ee = nn.Parameter(torch.randn(out_channels))  # Parameter per edge

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.theta1)
        nn.init.xavier_uniform_(self.theta2)
        nn.init.constant_(self.theta_ee, 1)

    def forward(self, x, edge_index):
        # Add self-loops to the adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Propagate information
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i: Node features of source nodes
        # x_j: Node features of target nodes

        # Compute the message for each edge
        message = self.theta_ee * (
            torch.matmul(x_i, self.theta1) +
            torch.matmul(x_j, self.theta2) -
            torch.matmul(x_i, self.theta2)
        )

        return message

    def aggregate(self, inputs, index, dim_size):
        # Aggregate messages using max pooling
        return scatter(inputs, index, dim=0, reduce='max')

    def update(self, aggr_out):
        # Apply ReLU activation function
        return F.relu(aggr_out)
    
# Complete model incorporating the custom MPNN layers
class Net(nn.Module):
    def __init__(self, in_chnl, hid_chnl):
        super(Net, self).__init__()

        # Initial projection layers
        self.lin1_1 = nn.Linear(in_chnl, hid_chnl)
        self.bn1_1 = nn.BatchNorm1d(hid_chnl)
        self.lin1_2 = nn.Linear(hid_chnl, hid_chnl)

        # Custom MPNN convolutional layers
        self.conv1 = CMPNNConv(hid_chnl, hid_chnl)
        self.bn1 = nn.BatchNorm1d(hid_chnl)
        self.conv2 = CMPNNConv(hid_chnl, hid_chnl)
        self.bn2 = nn.BatchNorm1d(hid_chnl)
        self.conv3 = CMPNNConv(hid_chnl, hid_chnl)
        self.bn3 = nn.BatchNorm1d(hid_chnl)

        # Graph pooling layers
        self.linears_prediction = nn.ModuleList()
        for layer in range(1 + 3):
            self.linears_prediction.append(nn.Linear(hid_chnl, hid_chnl))

    def forward(self, x, edge_index,  batch):
        # Projection step
        h = self.lin1_2(F.relu(self.bn1_1(self.lin1_1(x))))
        hidden_rep = [h]

        # MPNN convolutional layers
        h = F.relu(self.bn1(self.conv1(h, edge_index)))
        node_pool_over_layer = h
        hidden_rep.append(h)
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        node_pool_over_layer = node_pool_over_layer + h
        hidden_rep.append(h)
        h = F.relu(self.bn3(self.conv3(h, edge_index)))
        node_pool_over_layer = node_pool_over_layer + h
        hidden_rep.append(h)

        # Graph pooling
        gPool_over_layer = 0
        for layer, layer_h in enumerate(hidden_rep):
            g_pool = global_mean_pool(layer_h, batch)
            gPool_over_layer += F.dropout(self.linears_prediction[layer](g_pool), 0.5, training=self.training)

        return node_pool_over_layer, gPool_over_layer
