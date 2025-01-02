from policy_mdmtsp import Policy, action_sample, get_reward
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
import numpy as np
import matplotlib.pyplot as plt


def test(model, dataset, device, plot, depot_indices=[0,1,2,3,4]):

    # batch graph
    adj = torch.ones([dataset.shape[0], dataset.shape[1], dataset.shape[1]])  # adjacent matrix fully connected
    data_list = [Data(x=dataset[i], edge_index=torch.nonzero(adj[i], as_tuple=False).t(), as_tuple=False) for i in range(dataset.shape[0])]
    batch_graph = Batch.from_data_list(data_list=data_list).to(device)

    # get pi
    pi = model(batch_graph, n_nodes=dataset.shape[1], n_batch=dataset.shape[0])
    # sample action and calculate log probabilities
    action, log_prob = action_sample(pi)
    print(f"log_prob: {log_prob}")
    print(f"action: {action}")

    # get reward for each batch
    reward = get_reward(action, dataset, 5, plot)  # reward: tensor [batch, 1]

    return np.array(reward).mean()


if __name__ == '__main__':

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_agent = 3
    # n_nodes = [400, 500, 600, 700, 800, 900, 1000]
    # batch_size = 512
    # depots = [0,1,2,3,4]
    n_nodes = [200]
    seeds = [1, 2, 3]
    batch_size = 1
    seeds = [1]
    depots = [8, 22, 56, 190]

    # load net
    policy = Policy(in_chnl=2, hid_chnl=64, n_agent=n_agent, key_size_embd=64,
                  key_size_policy=64, val_size=64, clipping=10, dev=dev)

    # Path to uniformly distributed data trained model
    path = './saved_model_MDMTSP/{}.pth'.format(str(50) + '_' + str(n_agent) + '_lr' + str(0.0001) + '_cmpnn')

    # Path to mixed distributed data trained model
    # path = './saved_model_MDMTSP/{}.pth'.format(str(50) + '_' + str(n_agent) + '_lr' + str(0.0001) + '_cmpnn_goodData')

    policy.load_state_dict(torch.load(path, map_location=torch.device(dev)))

    for size in n_nodes:
        testing_data = torch.load('./testing_data/testing_data_' + str(size) + '_' + str(batch_size))
        print(f"testing_data size: {testing_data.shape}")

        results_per_seed = []
        for seed in seeds:
            # print('Size:', size, 'Seed:', seed)
            torch.manual_seed(seed)
            objs = []
            for j in range(batch_size):
                data = testing_data[j].unsqueeze(0)
                print(f"data: {data}")

                # testing
                obj = test(policy, data, dev, plot=1, depot_indices=depots)
                objs.append(obj)

                print('Max sub-tour length for instance', j, 'is', obj, 'Mean obj so far:', format(np.array(objs).mean(), '.4f'))
            results_per_seed.append(format(np.array(objs).mean(), '.4f'))
        print('Size:', size, results_per_seed)

    plt.show()
