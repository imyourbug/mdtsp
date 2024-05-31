# DeepRL-MDTSP

A solution to the Multiple Depot Travelling Salesman (MDTSP) problem using Deep Reinforcement Learning, Attention Networks, Graph Neural Network (GNN), Genetic Algorithm (GA), and Fuzzy Logic.

## Table of Contents
- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [References](#references)

## Description
This project proposes a novel computational intelligence architecture that combines DeepRL, GNNs, and Attention Networks to effectively solve the MDTSP. It also uses a fuzzy genetic algorithm that performs optimal depot selection using the pre-trained model.

Idea from the paper "A reinforcement learning approach for optimizing multiple traveling salesman problems over graphs".

## Features
- **GNN and Attention Mechanisms**: Creation of node, graph, and agent embeddings leading to a stochastic agent-to-node assignment mechanism.
- **Deep Reinforcement Learning**: Utilizes these embeddings to learn a strong policy and find an optimal solution for the MDTSP.
- **Genetic Algorithm and Fuzzy Logic**: Determines the best depot for each agent using the RL model. The integration of fuzzy logic enhances the algorithm's performance.

## Installation
Ensure you have the following dependencies installed:

- [OR-Tools](https://developers.google.com/optimization/install/python)
- Pytorch
- CUDA
- [Pytorch Geometric](https://github.com/pyg-team/pytorch_geometric)
- [Scikit-fuzzy](https://github.com/scikit-fuzzy/scikit-fuzzy)

## Usage
- `train.py`: Contains the training code.
- `test.py` and `validation.py`: Perform testing and validation.
- `validation.py` : Validates the results during training.
- `testing_data.py` : Contains the data for testing. Create new data if needed.
- `data_generator.py` (Uniform Distribution) and `data_generator_Newman.py` (Newman-Scott process): Generate data.
- `GA_DataGen.py`: Generates data using the genetic algorithm.
- `ortools_tsp.py`: Solves the TSP problem for reward calculations.
- `vrp.mdtsp.py`: Solves the MDTSP for baseline comparison with the model.
- `cmpnn.py`: Contains the Message Passing Neural Network.
- `policy_mdtsp.py`: Contains code related to the policy.
- `genetic_algorithm.py`: Implements a simple genetic algorithm.
- `GA_FuzzyMutation.py`: Implements the fuzzy-mutation genetic algorithm.
- `saved_model_MDMTSP.py`: Contains the two trained models
- 
- Other files: Perform comparisons between OR-Tools, RL-model, and genetic algorithms used in the project report.


## Contact
- Name: Odysseas Karagiannidis
- Email: okaragia@ece.auth.gr

## References
1. Code structure inspired by [zcaicaros/DRL-MTSP](https://github.com/zcaicaros/DRL-MTSP).
2. Hu, Yujiao, Yuan Yao, and Wee Sun Lee. "A reinforcement learning approach for optimizing multiple traveling salesman problems over graphs." Knowledge-Based Systems 204 (2020): 106244. DOI: [10.1016/j.knosys.2020.106244](https://doi.org/10.1016/j.knosys.2020.106244).

```bibtex
@article{hu2020reinforcement,
  title={A reinforcement learning approach for optimizing multiple traveling salesman problems over graphs},
  author={Hu, Yujiao and Yao, Yuan and Lee, Wee Sun},
  journal={Knowledge-Based Systems},
  volume={204},
  pages={106244},
  year={2020},
  publisher={Elsevier}
}




