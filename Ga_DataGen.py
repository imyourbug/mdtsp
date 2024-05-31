import torch
from GA_FuzzyMutation import main as FGA_main 
import os
import sys


def Ga_data_gen(no_nodes, batch_size, flag):
    testing_data = torch.rand(size=[batch_size, no_nodes, 2])
    new_data = []
    sum=0
    for i in range(batch_size):
        data = testing_data[i].unsqueeze(0)
        # Suppress stdout
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
   
        try:
           solution_value, solution_data, depots = FGA_main(data = data, max_generations=10, Population_Number=100,n_nodes=no_nodes,n_agent=5)
        
        finally:
            # Restore stdout
            sys.stdout.close()
            sys.stdout = original_stdout
        new_data.append(solution_data.squeeze(0))
        sum+=solution_value
        print("For batch index  = ", i , " result = ",  solution_value)

    print(new_data)
   
    torch.save(new_data, './testing_data/testing_data_GA_'+str(no_nodes)+'_'+str(batch_size))
    print("Mean = ",  sum/batch_size)


if __name__ == '__main__':
    n_nodes = 50
    b_size = 256
    flag = 'testing'
    torch.manual_seed(3)

    Ga_data_gen(n_nodes, b_size, flag)