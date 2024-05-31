from policy_mdmtsp import Policy, action_sample, get_reward
from vrp_mdtsp import entrance
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
import numpy as np
import matplotlib.pyplot as plt
import GA_FuzzyMutation
from GA_FuzzyMutation import main as GA_main
import time
import sys
import os

def calculate_better_probability(objs_ortools, objs_model):
    better_count = sum(1 for ortools, model in zip(objs_ortools, objs_model) if model < ortools)
    total_count = len(objs_ortools)
    probability = better_count / total_count if total_count > 0 else 0
    return probability

if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_agent = 5
    n_nodes = [50]
    batch_size = 1
    seeds = [1]

    # Load net
    policy = Policy(in_chnl=2, hid_chnl=64, n_agent=n_agent, key_size_embd=64,
                    key_size_policy=64, val_size=64, clipping=10, dev=dev)
    
    path = './saved_model_MDMTSP/{}.pth'.format(str(50) + '_' + str(n_agent) + '_lr' + str(0.0001) + '_cmpnn')
    #path = './saved_model_MDMTSP/{}.pth'.format(str(50) + '_' + str(n_agent) + '_lr' + str(0.0001) + '_cmpnn_goodData')
    policy.load_state_dict(torch.load(path, map_location=torch.device(dev)))
    
    probabilities = []
    means_model = []
    means_ortools = []

    for size in n_nodes:
        testing_data = torch.load('./testing_data/testing_data_' + str(size) + '_' + str(batch_size))
        #testing_data = torch.load('./testing_data/testing_data_Neyman_' + str(size) + '_' + str(batch_size))
        
        for seed in seeds:
            print('Size:', size, 'Seed:', seed)
            torch.manual_seed(seed)
            objs_ortools = []
            objs_model = []
            
            for j in range(batch_size):
                data = testing_data[j].unsqueeze(0)
                # Testing
                start_time = time.time()

                # Suppress stdout
                original_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
   
                try:
                    obj1, solution_data, _ = GA_main(Population_Number=100, max_generations=10, n_agent=n_agent, n_nodes=size, data=data)
                finally:
                    # Restore stdout
                    sys.stdout.close()
                    sys.stdout = original_stdout
                end_time = time.time()
                elapsed_time = int(end_time - start_time)
                objs_model.append(obj1)

                # Suppress stdout
                sys.stdout = open(os.devnull, 'w')

                try:
                    obj2 = entrance(cnum=size, anum=n_agent, batch_index=0, batch_size=batch_size, coords=solution_data, timeLimitation=elapsed_time)
                finally:
                    # Restore stdout
                    sys.stdout.close()
                    sys.stdout = original_stdout
                objs_ortools.append(obj2)

                print("ORTOOLS OBJECTIVE = ", obj2)
                print("MODEL OBJECTIVE = ", obj1)

            probability = calculate_better_probability(objs_ortools, objs_model)
            probabilities.append(probability)

            mean_model = np.mean(objs_model)
            mean_ortools = np.mean(objs_ortools)
            means_model.append(mean_model)
            means_ortools.append(mean_ortools)

            print('For N=', size)
            print('Probability that model is better than OR-Tools:', probability)
            print('Mean objective of model:', mean_model)
            print('Mean objective of OR-Tools:', mean_ortools)

    print('Probabilities:', probabilities)
    print('Mean objectives of model:', means_model)
    print('Mean objectives of OR-Tools:', means_ortools)
