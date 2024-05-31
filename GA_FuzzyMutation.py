from policy_mdmtsp import Policy, action_sample, get_reward
from test import test
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
import numpy as np
import matplotlib.pyplot as plt
from vrp_mdtsp import entrance
import time
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import copy
import random

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_data(dataset):
    # Plotting

    #plt.figure(figsize=(10, 8))  # Set the size of the plot
    plt.scatter(dataset[:, 0], dataset[:, 1], label = 'Nodes')  # Scatter plot of nodes
    plt.title('Random 2D Coordinates for MDMTSP')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.grid(True)
    #plt.show()

def initialize_chromosomes(n_agents, n_nodes):
    # Each chromosome is an array of depot indices, one for each agent
    return np.random.randint(0, n_nodes, size=n_agents)

def calculate_objective(chromosome, data, policy, dev, n_agents):
    # Ensure we have the right format for data
    data_new = []
    coords = data.squeeze(0).clone()
    
    # Initialize new coordinates list
    coords_new = torch.zeros_like(coords)
    
    n_agents = len(chromosome)
    # Place depot nodes at the beginning
    for i in range(n_agents):
        coords_new[i] = coords[chromosome[i]]
    
    # Place the remaining nodes
    index = n_agents
    for k in range(len(coords)):
        if k not in chromosome:
            coords_new[index] = coords[k]
            index += 1
            if index >= len(coords):
                break
    
    data_new = coords_new.unsqueeze(0)
    
    set_seeds()
    # Test the model given the specific depots
    obj = test(policy, data_new, dev, plot=0)
    
    return obj, data_new


def calculate_fitness(chromosomes,data,policy,dev, n_agents):
    results=[]

    results = [calculate_objective(chromo, data.clone(), policy,dev, n_agents) for chromo in chromosomes]

    # Unpack the results into separate lists
    objectives = []
    data_list = []
    objectives, data_list = zip(*results)
    objectives = np.array(objectives)

    # FITNESS = 1 / model output
    return 1/objectives, objectives, data_list

def remove_duplicate_chromosomes(elites):
    unique_elites = []
    seen = set()
    for elite in elites:
        # Convert chromosome list to a tuple for hashability
        chromosome_tuple = tuple(elite)
        if chromosome_tuple not in seen:
            seen.add(chromosome_tuple)
            unique_elites.append(elite)
    return unique_elites

def selection(chromosomes, fitness_scores, Population_Number, elitism_rate=0.1):
    # Randomness
    np.random.seed(None)
    
    # Calculate the number of elites based on the elitism rate and population number
    num_elites = int(Population_Number * elitism_rate)
    
    # Identify the indices of the elite chromosomes based on fitness scores
    elite_indices = np.argsort(-fitness_scores)[:num_elites]
    elites = [chromosomes[i] for i in elite_indices]

    elites = remove_duplicate_chromosomes(elites)
    num_elites = len(elites)

    # Identify non-elite indices by subtracting elite indices from the total list of indices
    non_elite_indices = list(set(range(len(chromosomes))) - set(elite_indices))
    
    # Prepare to select non-elites based on their normalized fitness probabilities
    selected_chromosomes = []
    if non_elite_indices:  # Ensure there are non-elites to select from
        total_fitness = np.sum([fitness_scores[i] for i in non_elite_indices])
        if total_fitness > 0:
            survival_probabilities = [fitness_scores[i] / total_fitness for i in non_elite_indices]
            # Calculate remaining spots to fill in the population considering the number of elites already chosen
            remaining_spots = max(0, int(np.random.normal(loc=Population_Number // 3, scale=Population_Number//10)) - num_elites)
            
            # Select non-elites based on survival probabilities
            if remaining_spots > 0:
                selected_indices = np.random.choice(non_elite_indices, size=remaining_spots, p=survival_probabilities)
                selected_chromosomes = [chromosomes[i] for i in selected_indices]
        else:
            # If total fitness is zero, distribute remaining spots equally among non-elites
            remaining_spots = max(0, int(np.random.normal(loc=Population_Number // 3, scale=10)) - num_elites)
            selected_chromosomes = [chromosomes[i] for i in np.random.choice(non_elite_indices, size=min(remaining_spots, len(non_elite_indices)), replace=False)]

    # Return combined elites (unchanged) and selected non-elites
    return elites + selected_chromosomes





def crossover(selected_chromosomes, crossover_prob):
    offsprings = []
    cross_number=0
    for i in range(0, len(selected_chromosomes), 2):
        if np.random.random() < crossover_prob:
            cross_number+=1
            parent1 = selected_chromosomes[i]
            parent2 = selected_chromosomes[i+1] if i+1 < len(selected_chromosomes) else selected_chromosomes[0]
            
            # Random Crossover point
            point = np.random.randint(1, len(parent1))

            # Create the 2 offsprings
            offspring1 = np.concatenate([parent1[:point], parent2[point:]])
            offsprings.append(offspring1)
            offspring2 = np.concatenate([parent2[:point], parent1[point:]])
            offsprings.append(offspring2)
        
    return offsprings

def setup_fuzzy_system():
    # Adjusting the universe for diversity from 0 to 0.08
    diversity = ctrl.Antecedent(np.arange(0, 0.08, 0.001), 'diversity')
    mutation_rate = ctrl.Consequent(np.arange(0, 0.2, 0.01), 'mutation_rate')

    # Manually define more sensitive membership functions for diversity
    diversity['low'] = fuzz.trimf(diversity.universe, [0, 0, 0.02])
    diversity['medium'] = fuzz.trimf(diversity.universe, [0.01, 0.04, 0.05])
    diversity['high'] = fuzz.trimf(diversity.universe, [0.03, 0.09, 0.09])

    # Define more responsive membership functions for mutation rate
    mutation_rate['low'] = fuzz.trimf(mutation_rate.universe, [0, 0, 0.05])
    mutation_rate['medium'] = fuzz.trimf(mutation_rate.universe, [0.02, 0.05, 0.09])
    mutation_rate['high'] = fuzz.trimf(mutation_rate.universe, [0.06, 0.2, 0.2])

    # Fuzzy rules with more rapid response
    rule1 = ctrl.Rule(diversity['low'], mutation_rate['high'])
    rule2 = ctrl.Rule(diversity['medium'], mutation_rate['medium'])
    rule3 = ctrl.Rule(diversity['high'], mutation_rate['low'])

    # Control system
    mutation_control_system = ctrl.ControlSystem([rule1, rule2, rule3])
    mutation_control_simulation = ctrl.ControlSystemSimulation(mutation_control_system)
    return mutation_control_simulation

def calculate_mutation_rate(diversity_input):
    mutation_rate_simulation = setup_fuzzy_system()
    mutation_rate_simulation.input['diversity'] = diversity_input
    mutation_rate_simulation.compute()
    return mutation_rate_simulation.output['mutation_rate']

def calculate_diversity(fitness_scores):
    """Calculate standard deviation of fitness scores."""
    return np.std(fitness_scores)

def mutate(next_generation, fitness_scores, n_nodes):
    """Apply mutations based on fuzzy logic derived from population diversity."""
    diversity = calculate_diversity(fitness_scores)
    print("DIVERSITY = ", diversity)

    mutation_probability = calculate_mutation_rate(diversity)

    print("Mutation Prob = ", mutation_probability)
    
    offsprings = []
    chromosome = []
    for i in range(len(next_generation)):
        chromosome = next_generation[i]
        for j in range(len(chromosome)):
            if np.random.random() < mutation_probability:
                chromosome[j] = np.random.randint(0, n_nodes)  # assuming `n_nodes` is in scope
                offsprings.append(chromosome)
    print("MUTATION NUMBER  = ", len(offsprings))
    return offsprings



def Population_Number_Maintenance(chromosomes,crossover_probability,Population_Number, n_nodes,n_agent,data,policy,dev):
    
    additional_crossover = []
    additional_mutation = []
    random_clones = []

    ### ADDITIONAL CROSSOVERS 
    new_popNumber = len(chromosomes)

    if(new_popNumber<Population_Number):
        additional_crossover = crossover(chromosomes, crossover_probability)
        new_popNumber = len(chromosomes) + len(additional_crossover)
        if new_popNumber >= Population_Number :
            # If more than needed, trim the list
            additional_crossover = additional_crossover[:Population_Number - new_popNumber]

        print("NEW POPULATION NUMBER AFTER ADDITIONAL CROSSOVERS: ",len(chromosomes) + len(additional_crossover))
    
    if len(additional_crossover)>0:
        chromosomes = np.concatenate((chromosomes,additional_crossover))

    
    # ADDITIONAL MUTATIONS
    if(new_popNumber<Population_Number):
        fitness_scores,_,_=calculate_fitness(additional_crossover,data,policy,dev, n_agent)
        additional_mutation = mutate(additional_crossover, fitness_scores, n_nodes)
        new_popNumber = len(chromosomes) + len(additional_mutation)
        if new_popNumber >= Population_Number:
            # If more than needed, trim the list
            additional_mutation = additional_mutation[:Population_Number - new_popNumber]
        print("NEW POPULATION NUMBER AFTER ADDITIONAL MUTATIONS: ",len(chromosomes) + len(additional_mutation))
    
    if len(additional_mutation)>0:
        chromosomes = np.concatenate((chromosomes,additional_mutation))
    
    # RANDOM CLONES
    new_popNumber = len(chromosomes)
    if(new_popNumber<Population_Number):
        random_clones = [initialize_chromosomes(n_agent, n_nodes) for _ in range(Population_Number-new_popNumber)]
        print("NEW POPULATION NUMBER AFTER ADDITIONAL RANDOM CLONES: ",new_popNumber+len(random_clones))

    if len(random_clones)>0:
        chromosomes = np.concatenate((chromosomes,random_clones))
 
    return chromosomes


def main(data, Population_Number, max_generations, n_agent, n_nodes):
    start_time = time.time()
    seed = 1
    torch.manual_seed(seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("### FUZZY GENETIC ALGORITHM ###\n\n")

    ### LOAD POLICY MODEL
    path = './saved_model_MDMTSP/{}.pth'.format(str(50) + '_' + str(n_agent) + '_lr' + str(0.0001) + '_cmpnn')
    policy = Policy(in_chnl=2, hid_chnl=64, n_agent=n_agent, key_size_embd=64,
                        key_size_policy=64, val_size=64, clipping=10, dev=dev)
    
    policy.load_state_dict(torch.load(path, map_location=torch.device(dev)))
    
    # Extract coordinates from data
    coords = data.squeeze(0)
    #plot_data(coords)

    # Model output
    obj = test(policy, data, dev, plot=0)
    solution_value = obj
    #plt.show(block=True)
    print('Initial objective:', obj, '\n\n')

    
    # CHROMOSOME RANDOM INITIALIZATION
    chromosomes = [initialize_chromosomes(n_agent, n_nodes) for _ in range(Population_Number)]  # For example, 10 chromosomes

    # Settings before run
    solutions = []
    solution_index = []
    data_lists=[]
    new_chromosomes = chromosomes
    
    print("INITIAL POPULATION NUMBER: ", Population_Number)

    for gen in range(max_generations):

        print("\n\n### GENERATION ", gen, " ###\n")
        ### FITNESS CALCULATION

        fitness_scores,objs, data_list_main = calculate_fitness(copy.deepcopy(chromosomes),copy.deepcopy(data),policy=policy,dev=dev, n_agents=n_agent)
        
        best_individual_value  = np.min(objs)
        index = np.argmin(objs)
        print("MINIMUM OBJECTIVE: ",best_individual_value," in index " , index)
        print("BEST INDIVIDUAL = ", chromosomes[index])
        solutions.append(best_individual_value)
        solution_index.append(index)
        data_lists.append(data_list_main[index])

        ### SELECTION
        
        selected_chromosomes = selection(chromosomes, fitness_scores, Population_Number)     
        print("NUMBER OF SELECTIONS: ", len(selected_chromosomes))

        ### CROSSOVER
        crossover_probability = 0.6
        next_generation = crossover(selected_chromosomes,crossover_probability)
        print("CROSSOVER OFFSPRINGS NUMBER: ", len(next_generation))

        ### MUTATION

        # Call mutate
        fitness_scores,_,_ = calculate_fitness(next_generation,copy.deepcopy(data),policy,dev, n_agent)
        mutated_offspring = mutate(next_generation, fitness_scores, n_nodes)
    
        chromosomes = selected_chromosomes + next_generation + mutated_offspring
        
        new_popNumber = len(chromosomes)
        if new_popNumber>Population_Number:
            new_popNumber=Population_Number
        print("NEW POPULATION NUMBER: ",new_popNumber)

        ### POPULATION MAINTENANCE PHASE
        print("\n####  POPULATION NUMBER MAINTENANCE  ####")
        crossover_probability = 0.4
        chromosomes = Population_Number_Maintenance(copy.deepcopy(chromosomes),crossover_probability,Population_Number, n_nodes,n_agent, copy.deepcopy(data), policy,dev)
    
        
    # TAKE THE BEST VALUE
    solution_value = solutions[max_generations-1]
    solution_data = data_lists[max_generations-1]

    print("\n\nBEST INDIVIDUAL VALUE = ", solution_value )
    #plot_data(solution_data.squeeze(0))

    #print("\n\nBEST INDIVIDUAL VALUE = ", solution_value )
    #plt.figure(figsize=(6, 6))
    #test(policy,solution_data,dev, plot=1)

    # Test the solution data (plot=1 for plotting)
    test(policy,solution_data,dev, plot=0)

    return solution_value, solution_data, chromosomes[0]

if __name__ == '__main__':

    # POPULATION NUMBER
    Population_Number = 100

    # MAX GENERATIONS
    max_generations = 10

    n_agent = 5
    n_nodes = 50
    data = torch.load('./testing_data/testing_data_' + str(n_nodes) + '_' + str(1))

    ### UNCOMMENT FOR ORTOOLS CALCULATION
    #start_time = time.time()
    _,solution_data,_=main(data, Population_Number, max_generations, n_agent, n_nodes)
    #end_time = time.time()
    #elapsed_time = int(end_time-start_time)
    #print("ORTOOLS SOLUTION FOR THE CHOSEN DEPOTS (for the same running time of ", elapsed_time, 'seconds : ')

    #entrance(cnum=  n_nodes, anum=n_agent , batch_index=0,  batch_size=1, coords=solution_data , timeLimitation=elapsed_time)
    
    plt.show()
        
