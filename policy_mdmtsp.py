from cmpnn import Net
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.distributions import Categorical
from ortools_tsp import solve
import matplotlib.pyplot as plt
import random



class Agentembedding(nn.Module):
    def __init__(self, node_feature_size, key_size, value_size, n_agent):
        super(Agentembedding, self).__init__()
        self.key_size = key_size

        self.q_agent = nn.Linear((1+n_agent) * node_feature_size, key_size)
        self.k_agent = nn.Linear(node_feature_size, key_size)
        self.v_agent = nn.Linear(node_feature_size, value_size)

    def forward(self, f_c, f):
        q = self.q_agent(f_c)
        k = self.k_agent(f)
        v = self.v_agent(f)
        u = torch.matmul(k, q.transpose(-1, -2)) / math.sqrt(self.key_size)
        u_ = F.softmax(u, dim=-2).transpose(-1, -2)
        agent_embedding = torch.matmul(u_, v)

        return agent_embedding


class AgentAndNode_embedding(torch.nn.Module):
    def __init__(self, in_chnl, hid_chnl, n_agent, key_size, value_size, dev):
        super(AgentAndNode_embedding, self).__init__()

        self.n_agent = n_agent

        # mpnn
        self.gin = Net(in_chnl=in_chnl, hid_chnl=hid_chnl).to(dev)
        # agent attention embed
        self.agents = torch.nn.ModuleList()
        for i in range(n_agent):
            self.agents.append(Agentembedding(node_feature_size=hid_chnl, key_size=key_size, value_size=value_size,n_agent=n_agent).to(dev))

    def forward(self, batch_graphs, n_nodes, n_batch):

        # get node embedding using mpnn

        nodes_h, g_h = self.gin(x=batch_graphs.x, edge_index=batch_graphs.edge_index, batch=batch_graphs.batch)
        nodes_h = nodes_h.reshape(n_batch, n_nodes, -1)
        g_h = g_h.reshape(n_batch, 1, -1)

        # concatenate the embeddings of the first n_agent nodes (depots)
        depots_h = nodes_h[:, :self.n_agent, :]
        depots_h_flattened = depots_h.reshape(n_batch, -1)  # Flatten to [batch_size, n_agent*hid_chnl]
 

        # Flatten graph embedding as well
        g_h_flattened = g_h.reshape(n_batch, -1)

        # Concatenate the flattened tensors
        depot_cat_g = torch.cat((g_h_flattened, depots_h_flattened), dim=-1)
        depot_cat_g = depot_cat_g.unsqueeze(1)
        #print('Global Embedding shape = ', depot_cat_g.shape)

        # remaining nodes (excluding depots)
        nodes_h_no_depot = nodes_h[:, self.n_agent:, :]

        # get agent embedding
        agents_embedding = []
        for i in range(self.n_agent):
            agents_embedding.append(self.agents[i](depot_cat_g, nodes_h_no_depot))

        agent_embeddings = torch.cat(agents_embedding, dim=1)
        #print(agent_embeddings.shape)

        return agent_embeddings, nodes_h_no_depot


class Policy(nn.Module):
    def __init__(self, in_chnl, hid_chnl, n_agent, key_size_embd, key_size_policy, val_size, clipping, dev):
        super(Policy, self).__init__()
        self.c = clipping
        self.key_size_policy = key_size_policy
        self.key_policy = nn.Linear(hid_chnl, self.key_size_policy).to(dev)
        self.q_policy = nn.Linear(val_size, self.key_size_policy).to(dev)

        # embed network
        self.embed = AgentAndNode_embedding(in_chnl=in_chnl, hid_chnl=hid_chnl, n_agent=n_agent,
                                            key_size=key_size_embd, value_size=val_size, dev=dev)


    def forward(self, batch_graph, n_nodes, n_batch):

        agent_embeddings, nodes_h_no_depot = self.embed(batch_graph, n_nodes, n_batch)
        k_policy = self.key_policy(nodes_h_no_depot)
        q_policy = self.q_policy(agent_embeddings)
        u_policy = torch.matmul(q_policy, k_policy.transpose(-1, -2)) / math.sqrt(self.key_size_policy)
        imp = self.c * torch.tanh(u_policy)
        prob = F.softmax(imp, dim=-2)


        return prob


def action_sample(pi):
    #print("Pi of agent 0: ", pi[:,0,:])
    #print("Pi shape: ", pi.shape)
    #print("Pi transpose shape: ", pi.transpose(2, 1).shape)
    dist = Categorical(pi.transpose(2, 1))
    #print("dist shape: ", dist)

    action = dist.sample()
    #print("action ", action)
    log_prob = dist.log_prob(action)
    return action, log_prob

def get_reward(action, data, n_agent, plot):
    data = data*1000
    #action%=n_agent
    # Assuming the first 'n_agent' entries are unique depots for each agent
    depots = data[:, :n_agent, :].tolist()
    n_nodes = data.size(1)  # This includes depots + cities

    sub_tours = [[[] for _ in range(n_agent)] for _ in range(data.shape[0])]
    for batch_index in range(data.shape[0]):
        #print("Action matrix for batch {}: {}".format(batch_index, action[batch_index].tolist()))
        for node_index, agent_index in enumerate(action.tolist()[batch_index]):
            #print("Agent index = " ,agent_index)
            city_index = node_index + n_agent  # Adjust index to skip depots
            #print("agent index = ", agent_index)
            if city_index < n_nodes:  # Ensure index is within bounds
                city = data[batch_index, city_index, :].tolist()
                #print(city)
                #print(batch_index,agent_index)
                agent_index %= n_agent  # Ensure valid agent indices
                #print("agent index = ", agent_index)
                sub_tours[batch_index][agent_index].append(city)

    # Insert depot as the start and end point of each tour
    for batch_index in range(data.shape[0]):
        for agent_index in range(n_agent):
            sub_tours[batch_index][agent_index].insert(0, depots[batch_index][agent_index])
            sub_tours[batch_index][agent_index].append(depots[batch_index][agent_index])

    subtour_max_lengths = [0 for _ in range(data.shape[0])]
    for batch_index in range(data.shape[0]):
        for agent_index in range(n_agent):
            tour_instance = sub_tours[batch_index][agent_index]
            #tour_length = solve(tour_instance,agent_index) / 1000
            tour_length = solve(tour_instance, agent_index, plot) / 1000
            if tour_length > subtour_max_lengths[batch_index]:
                subtour_max_lengths[batch_index] = tour_length

            #print(f"Objective for agent {agent_index} in batch {batch_index}: {tour_length}")
            #print("Route for agent {}: {}".format(agent_index, " -> ".join(str(x) for x in tour_instance)))

    return subtour_max_lengths



if __name__ == '__main__':
    from torch_geometric.data import Data
    from torch_geometric.data import Batch
    dev = 'cpu'
    torch.manual_seed(1)

    n_agent = 5
    n_nodes = 20
    n_batch = 1
    # get batch graphs data list
    fea = torch.randint(low=0, high=100, size=[n_batch, n_nodes, 2]).to(torch.float)  # [batch, nodes, fea]
    
    adj = torch.ones([fea.shape[0], fea.shape[1], fea.shape[1]])
    
    data_list = [Data(x=fea[i], edge_index=torch.nonzero(adj[i]).t()) for i in range(fea.shape[0])]
    #print(data_list)
    # generate batch graph
    batch_graph = Batch.from_data_list(data_list=data_list).to(dev)
    #print(batch_graph)
    # Create a figure
    plt.figure(figsize=(10, 6))
    # test policy

    policy = Policy(in_chnl=fea.shape[-1], hid_chnl=64, n_agent=n_agent, key_size_embd=64,
                    key_size_policy=64, val_size=64, clipping=10, dev=dev)

    pi = policy(batch_graph, n_nodes, n_batch)

    grad = torch.autograd.grad(pi.sum(), [param for param in policy.parameters()], allow_unused=True)

    action, log_prob = action_sample(pi)
    #print(action)

    rewards = get_reward(action, fea, n_agent, plot=1)

    print(rewards)

    plt.show()
