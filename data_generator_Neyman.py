import torch
from GA_FuzzyMutation import plot_data
import matplotlib.pyplot as plt

def neyman_scott_process(no_nodes, batch_size, flag, parent_lambda=10, cluster_size_mean=10, cluster_std=0.05):
    """
    Generates data points based on the Neyman-Scott (Poisson Cluster) process.

    Parameters:
    - no_nodes: Number of nodes (points) to generate.
    - batch_size: Number of batches.
    - flag: Indicates whether data is for 'testing' or 'validation'.
    - parent_lambda: Intensity (rate) of the parent Poisson process.
    - cluster_size_mean: Mean number of offspring per parent.
    - cluster_std: Standard deviation of the clusters around each parent.
    """
    data = torch.zeros((batch_size, no_nodes, 2))
    cluster_size_mean = no_nodes/10

    for b in range(batch_size):
        points = []
        cluster_means = []
        while len(points) < no_nodes:
            # Step 1: Generate parent points according to a homogeneous Poisson process
            num_parents = int(torch.poisson(torch.tensor(parent_lambda, dtype=torch.float32)).item())
            parent_points = torch.rand((num_parents, 2))

            # Step 2: Generate offspring points around each parent point
            for parent in parent_points:
                num_offspring = int(torch.poisson(torch.tensor(cluster_size_mean, dtype=torch.float32)).item())
                offspring_points = torch.randn((num_offspring, 2)) * cluster_std + parent
                points.extend(offspring_points.tolist())
                cluster_means.append(parent.tolist())

                if len(points) >= no_nodes:
                    break

        # Ensure the first 5 nodes are close to the cluster means
        points = points[:no_nodes]
        for i in range(min(5, len(cluster_means))):
            points[i] = cluster_means[i]
        
        data[b] = torch.tensor(points)

    if flag == 'validation':
        torch.save(data, './validation_data/validation_data_Neyman_' + str(no_nodes) + '_' + str(batch_size))
    elif flag == 'testing':
        torch.save(data, './testing_data/testing_data_Neyman_' + str(no_nodes) + '_' + str(batch_size))
    else:
        print('flag should be "testing", or "validation"')
    return data

if __name__ == '__main__':
    n_nodes = 1000
    b_size = 100
    flag = 'testing'
    torch.manual_seed(3)

    data = neyman_scott_process(n_nodes, b_size, flag)
    plot_data(data[0].squeeze(0))
    plt.show()
