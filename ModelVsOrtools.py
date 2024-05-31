from policy_mdmtsp import Policy, action_sample, get_reward
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
import numpy as np
import matplotlib.pyplot as plt
from vrp_mdtsp import entrance
from test import test


if __name__ == '__main__':

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_agent = 5
    n_nodes = [400, 500, 600, 700, 800, 900, 1000]
    batch_size = 512
    seeds = [1, 2, 3]


    # load net
    policy = Policy(in_chnl=2, hid_chnl=64, n_agent=n_agent, key_size_embd=64,
                    key_size_policy=64, val_size=64, clipping=10, dev=dev)
    path = './saved_model_MDMTSP/{}.pth'.format(str(50) + '_' + str(n_agent) + '_lr' + str(0.0001) + '_cmpnn')
    policy.load_state_dict(torch.load(path, map_location=torch.device(dev)))

    for size in n_nodes:
        
        testing_data = torch.load('./testing_data/testing_data_' + str(size) + '_' + str(batch_size))
        
        results_per_seed_model = []
        results_per_seed_ortools = []
        for seed in seeds:
            print('Size:', size, 'Seed:', seed)
            torch.manual_seed(seed)
            objs_model = []
            objs_ortools = []
            for j in range(batch_size):
                # data = torch.rand(size=[1, size, 2])  # [batch, nodes, fea], fea is 2D location
                data = testing_data[j].unsqueeze(0)
                # testing
                obj_model = test(policy, data, dev, plot=0)
                objs_model.append(obj_model)
                print('MODEL: Max sub-tour length for instance', j, 'is', obj_model, 'Mean obj so far:', format(np.array(objs_model).mean(), '.4f'))

                obj_ortools = entrance(cnum=  size, anum=n_agent , batch_index=0,  batch_size=1, coords=data ,depot_indices = [0,1,2,3,4], timeLimitation=1800)
                objs_ortools.append(obj_ortools)
                print('ORtools: Max sub-tour length for instance', j, 'is', obj_ortools, 'Mean obj so far:', format(np.array(objs_ortools).mean(), '.4f'),'\n')
            results_per_seed_model.append(format(np.array(objs_model).mean(), '.4f'))
            results_per_seed_ortools.append(format(np.array(objs_ortools).mean(), '.4f'))
        print('(Model) Size:', size, results_per_seed_model)
        print('(Ortools) Size:', size, results_per_seed_ortools)


#(Model) Size: 1000 ['6.9089']
#(Ortools) Size: 1000 ['17.7156']