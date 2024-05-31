import matplotlib.pyplot as plt
import math
import torch
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def create_data_model(instances):
    """Stores the data for the problem."""
    data = {'locations': instances, 'num_vehicles': 1, 'depot': 0}
    return data

def compute_euclidean_distance_matrix(locations):
    """Creates callback to return distance between points."""
    distances = {}
    for from_counter, from_node in enumerate(locations):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                distances[from_counter][to_counter] = (int(
                    math.hypot((from_node[0] - to_node[0]),
                               (from_node[1] - to_node[1]))))
    return distances

def get_route(manager, routing, solution):
    """Extracts the route coordinates for plotting."""
    index = routing.Start(0)
    route = []
    while not routing.IsEnd(index):
        node_index = manager.IndexToNode(index)
        route.append(node_index)
        index = solution.Value(routing.NextVar(index))
    route.append(manager.IndexToNode(index))  # Append the end point (depot)
    return route

def plot_route(route, locations,agent_index):


    """Plots the route using matplotlib."""
    # Extract x and y coordinates from the route
    x_coords = [locations[i][0] for i in route]
    y_coords = [locations[i][1] for i in route]
    
    plt.plot(x_coords, y_coords, 'o-', label=f'Vehicle {agent_index}')
    plt.plot(x_coords[0], y_coords[0], 'yo')  # Depot as a red dot
    plt.scatter(x_coords[0], y_coords[0], s=100)  # Start depot
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Traveling Salesman Route (MODEL)')
    plt.legend()
    plt.grid(True)
    #plt.show()

def solve(instance, agent_index, plot):
    """Solves the TSP and plots the route."""
    data = create_data_model(instance)
    manager = pywrapcp.RoutingIndexManager(len(data['locations']),
                                           data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)
    distance_matrix = compute_euclidean_distance_matrix(data['locations'])
    transit_callback_index = routing.RegisterTransitCallback(
        lambda from_index, to_index: distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]
    )
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    solution = routing.SolveWithParameters(search_parameters)
    if plot ==1:
        if solution:
            route = get_route(manager, routing, solution)
            plot_route(route, data['locations'], agent_index)
        
    return solution.ObjectiveValue()

if __name__ == '__main__':
    n_nodes = 10
    agent_index = 2
    instances = torch.rand(size=[n_nodes, 2])  # [batch, nodes, features]
    solve(instances.numpy() * 1000,agent_index, plot=1)  # Convert torch tensor to numpy array and scale for plotting
    plt.show()