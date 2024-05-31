from policy_mdmtsp import Policy, action_sample, get_reward
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
from validation import validate
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

def save_result_list(result_list, filename):
    with open(filename, 'wb') as f:
        pickle.dump(result_list, f)

def load_result_list(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        return []

def train(batch_size, no_nodes, policy_net, l_r, no_agent, iterations, device, save_interval=100, result_save_interval=20):
    # Prepare validation data
    validation_data = torch.load('./validation_data/validation_data_' + str(no_nodes) + '_' + str(batch_size))
    best_so_far = np.inf
    validation_results = []

    # Check for existing saved state
    #checkpoint_path = f"./training_state_MDMTSP/checkpoint_{no_nodes}_{no_agent}_lr{l_r}_mpnn.pth"
    checkpoint_path = f"./training_state_MDMTSP/checkpoint_{no_nodes}_{no_agent}_lr{l_r}_mpnn_goodData.pth"

    start_iteration = 0
    result_list_path = './result_list.pkl'
    result_list = load_result_list(result_list_path)
    
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        policy_net.load_state_dict(checkpoint["model_state"])
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=l_r)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_iteration = checkpoint["iteration"]
        best_so_far = checkpoint["best_so_far"]
        validation_results = checkpoint["validation_results"]
    else:
        # If no checkpoint, create a new optimizer
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=l_r)
        
    # Load data generated by the genetic algorithm
    good_data = torch.load('./testing_data/testing_data_GA_' + str(no_nodes) + '_' + str(256))
    good_data = torch.stack([d.clone().detach() for d in good_data])

    # Load data generated by the Neyman-Scott process
    Neyman_data = torch.load('./testing_data/testing_data_Neyman_' + str(no_nodes) + '_' + str(1000))
    Neyman_data = torch.stack([d.clone().detach() for d in Neyman_data])


    for itr in range(start_iteration, iterations):
        # Prepare training data
        # Select 50 good data samples
        good_data_samples = good_data[np.random.choice(len(good_data), 50, replace=False)]

        # Select 50 Neyman data samples
        Neyman_data_samples = Neyman_data[np.random.choice(len(Neyman_data), 50, replace=False)]
        
        # Generate 412 random data samples
        random_data_samples = torch.rand(size=[412, no_nodes, 2])


        # Combine good data and random data
        combined_data = torch.cat((good_data_samples, random_data_samples, Neyman_data_samples), dim=0)
        
        # Shuffle combined data
        combined_data = combined_data[torch.randperm(combined_data.size(0))]
        
        # Create the batch graph
        adj = torch.ones([combined_data.shape[0], combined_data.shape[1], combined_data.shape[1]])
        data_list = [Data(x=combined_data[i], edge_index=torch.nonzero(adj[i], as_tuple=False).t()) for i in range(combined_data.shape[0])]
        batch_graph = Batch.from_data_list(data_list=data_list).to(device)

        # Get pi
        pi = policy_net(batch_graph, n_nodes=combined_data.shape[1], n_batch=batch_size)

        # Sample action and calculate log probabilities
        action, log_prob = action_sample(pi)
        
        reward = get_reward(action, combined_data, no_agent, plot=0)

        # Compute loss
        loss = torch.mul(torch.tensor(reward, device=device)-2, log_prob.sum(dim=1)).sum()

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Collect result for plotting
        avg_reward = sum(reward) / batch_size
        result_list.append(avg_reward)
        if itr % 100 == 0:
            print('\nIteration:', itr)
        
        print(format(avg_reward, '.4f'))

        # Validate and save best nets
        if (itr + 1) % 100 == 0:
            validation_result = validate(validation_data, policy_net, no_agent, device, plot=0)
            print(validation_result)
            if validation_result < best_so_far:
                torch.save(policy_net.state_dict(), './saved_model_MDMTSP/{}_{}_lr{}_cmpnn.pth'.format(str(no_nodes), str(no_agent), str(l_r)))
                #torch.save(policy_net.state_dict(), './saved_model_MDMTSP/{}_{}_lr{}_cmpnn_goodData.pth'.format(str(no_nodes), str(no_agent), str(l_r)))
                print('Found better policy, and the validation result is:', format(validation_result, '.4f'))
                validation_results.append(validation_result)
                best_so_far = validation_result

        # Save checkpoint periodically
        if (itr + 1) % 20 == 0:
            checkpoint = {
                "model_state": policy_net.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "iteration": itr + 1,
                "best_so_far": best_so_far,
                "validation_results": validation_results
            }
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(checkpoint, checkpoint_path)

        # Save result list periodically
        if (itr + 1) % result_save_interval == 0:
            save_result_list(result_list, result_list_path)
        #print(result_list)

    return validation_results, result_list

if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(2)

    torch.autograd.set_detect_anomaly(True)
    n_agent = 5
    n_nodes = 50
    batch_size = 512
    lr = 1e-4
    iteration = 10000

    policy = Policy(in_chnl=2, hid_chnl=64, n_agent=n_agent, key_size_embd=64,
                  key_size_policy=64, val_size=64, clipping=10, dev=dev)

    best_results, result_list = train(batch_size, n_nodes, policy, lr, n_agent, iteration, dev)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(result_list, label='Average Reward per Batch')
    plt.title('Training Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig('./training_convergence_plot.png')
    plt.show()