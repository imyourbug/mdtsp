from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.data import Batch

C = 10000

def Euclidean_distance(coords):
    city_square = torch.sum(coords ** 2, dim=1, keepdim=True)
    city_square_tran = torch.transpose(city_square, 1, 0)
    cross = -2 * torch.matmul(coords, torch.transpose(coords, 1, 0))
    dist = city_square + city_square_tran + cross
    dist = torch.sqrt(dist)
    for m in range(dist.size(0)):
        dist[m, m] = 0.0
    dist = dist * C
    return dist.long().numpy()

def create_data_model(cnum, anum, depots,batch_size, batch_index, coords= None):
    """Stores the data for the problem."""


    data = {}
    data['coords'] = coords[batch_index]
    data['num_vehicles'] = anum
    data['depots'] = depots
    data['distance_matrix'] = []

    dist = Euclidean_distance(coords[batch_index])
    for c in range(cnum):
        data['distance_matrix'].append(list(dist[c]))
    return data, coords

def print_solution(data, manager, routing, solution):
    ###Prints solution on console and returns the routes for plotting.
    max_route_distance = 0
    routes = []
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = f'Route for vehicle {vehicle_id}:\n'
        route_distance = 0
        route = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(node_index)
            plan_output += f'{node_index} -> '
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)

        node_index = manager.IndexToNode(index)
        route.append(node_index)  # add end depot
        plan_output += f'{node_index}\n'
        plan_output += f'Distance of the route: {route_distance / C}m\n'
        #print(plan_output)
        routes.append(route)
        max_route_distance = max(route_distance, max_route_distance)
    print(f'Maximum of the route distances: {max_route_distance / C}m')
    return routes, max_route_distance / C

def plot_solution(data, routes):
    ###Plots the routes using matplotlib
    #plt.figure(figsize=(6, 6))
    coords = data['coords'].numpy()
    for vehicle_id, route in enumerate(routes):
        route_coords = coords[route]
        plt.plot(route_coords[:, 0], route_coords[:, 1], marker='o', label=f'Vehicle {vehicle_id}')
        plt.scatter(route_coords[0, 0], route_coords[0, 1], s=100)  # Start depot

    #plt.scatter(coords[:, 0], coords[:, 1], c='blue', label='Cities')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('MDTSP Solution Visualization (ORTOOLS)')
    plt.legend()
    plt.grid(True)
    #plt.show()

def entrance(cnum, anum,batch_size, batch_index,coords ,depot_indices=[0,1,2,3,4], timeLimitation=1800):
    """Solve the MDTSP problem."""
    #depots = [0, 1, 2, 3, 4]  # Example depots for 5 vehicles
    #depots = [0, 0, 0, 0, 0]  # Example depots for 5 vehicles --> MTSP
    depots= depot_indices
    data, coords = create_data_model(cnum, anum, depots,batch_size, batch_index, coords=coords)
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], depots,depots)

    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    dimension_name = 'Distance'
    routing.AddDimension(transit_callback_index, 0, 10000000, True, dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.time_limit.seconds = timeLimitation

    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        routes, max_route_distance = print_solution(data, manager, routing, solution)
        #plot_solution(data, routes)

    #return solution.ObjectiveValue()/(C*100)
    return max_route_distance
if __name__ == '__main__':
    n_nodes = 50
    n_agents = 5
    objs = 0
    batch_size = 1
    data = torch.load('./testing_data/testing_data_' + str(n_nodes) + '_' + str(1))
    #data = torch.load('./testing_data/testing_data_NonHom_' + str(n_nodes) + '_' + str(1))
    depot_indices = [0,1,2,3,4]
    #depot_indices = [0,1]


    coords = data.squeeze(0)
    print(entrance(cnum=  n_nodes, anum=n_agents , batch_index=0,  batch_size=1, coords=data ,depot_indices = depot_indices, timeLimitation=2))
    plt.show()



