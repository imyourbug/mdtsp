import genetic_algorithm
import GA_FuzzyMutation
from genetic_algorithm import main as GA_main
from GA_FuzzyMutation import main as GA_Fuzzy_main
import matplotlib.pyplot as plt
import torch
import os
import sys
import numpy as np

def calculate_better_probability(objs_ortools, objs_model):
    better_count = sum(1 for ortools, model in zip(objs_ortools, objs_model) if model < ortools)
    total_count = len(objs_ortools)
    probability = better_count / total_count if total_count > 0 else 0
    return probability

if __name__ == '__main__':
    # POPULATION NUMBER
    Population_Number = 100

    # MAX GENERATIONS
    max_generations = 10

    n_agent = 5
    n_nodes = [50]
    batch_size = 1
    seeds = [1]

    probabilities = []
    means_GA = []
    means_FGA = []

    for size in n_nodes:
        #testing_data = torch.load('./testing_data/testing_data_' + str(size) + '_' + str(batch_size))
        testing_data = torch.load('./testing_data/testing_data_' + str(size) + '_' + str(batch_size))
        for seed in seeds:
            print('Size:', size, 'Seed:', seed)
            torch.manual_seed(seed)
            objs_GA = []
            objs_FGA = []
            
            for j in range(batch_size):
                data = testing_data[j].unsqueeze(0)

                # Suppress stdout
                original_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
   
                try:
                    solution1 = GA_main(data, Population_Number, max_generations, n_agent,size)
                finally:
                    # Restore stdout
                    sys.stdout.close()
                    sys.stdout = original_stdout

                objs_GA.append(solution1)

                # Suppress stdout
                sys.stdout = open(os.devnull, 'w')

                try:
                    solution2,_,_ = GA_Fuzzy_main(data, Population_Number, max_generations, n_agent, size)
                finally:
                    # Restore stdout
                    sys.stdout.close()
                    sys.stdout = original_stdout
                objs_FGA.append(solution2)

                print("GA OBJECTIVE = ", solution1)
                print("FGA OBJECTIVE = ", solution2)

            probability = calculate_better_probability(objs_GA, objs_FGA)
            probabilities.append(probability)

            mean_FGA = np.mean(objs_FGA)
            mean_GA = np.mean(objs_GA)
            means_GA.append(mean_GA)
            means_FGA.append(mean_FGA)

            print('For N=', size)
            print('Probability that the GA is better than FGA:', probability)
            print('Mean objective of GA:', mean_GA)
            print('Mean objective of FGA:', mean_FGA)

    print('Probabilities:', probabilities)
    print('Mean objectives of GA:', means_GA)
    print('Mean objectives of FGA:', means_FGA)

    plt.show()


    
