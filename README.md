# DeepRL-MDTSP
A solution to the Multiple Depot Travelling Salesman (MDTSP) problem using Deep Reinforcment Learning, Attention Networks, Graph Neural Network (GNN), Genetic Algorithm (GA) and Fuzzy Logic.

## Table of Contents
- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Description
- This homework proposes a novel computational intelligence architecture, which combines DeepRL, GNNs, and Attention Networks to effectively solve the MDTSP. It also uses a fuzzy genetic algorithm that does the optimal depot selection using the pre-trained model.
- Idea from paper "A reinforcement learning approach for optimizing multiple traveling salesman problems over graphs".
  Paper reference:
  ```bibtex
   @article{hu2020reinforcement,
  title={A reinforcement learning approach for optimizing multiple traveling salesman problems over graphs},
  author={Hu, Yujiao and Yao, Yuan and Lee, Wee Sun},
  journal={Knowledge-Based Systems},
  volume={204},
  pages={106244},
  year={2020},
  publisher={Elsevier}
- Code structure from https://github.com/zcaicaros/DRL-MTSP/blob/main/README.md

## Features
- GNN and Attention Mechanisms: Creation of node, graph and agent embeddings leading to a stochastic agent to node assignment mechanism.
- Deep Reinforcement Learning: Uses these embeddings to learn a strong policy and find a very good solution for the MDTSP.
- Genetic Algorithm and Fuzzy Logic: Finds the best depot for each agent using the RL model. The fuzzy logic is integrated into the algorithm and enhances it's performance.

## Installation
Step-by-step instructions on how to get the development environment running.

### Prerequisites
List any prerequisites, such as software or libraries that need to be installed beforehand.

### Contact
- Name: Odysseas Karagiannidis
- Email: okaragia@ece.auth.gr
- GitHub: IthakeCanWait

```sh
# Example: Install Python and pip
sudo apt-get install python3
sudo apt-get install python3-pip

### Contact
