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
Instructions on how to use the project will be added here.

## Contributing
We welcome contributions! Please follow these steps to contribute:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## License
Specify the license under which your project is distributed. For example:

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
- Name: Odysseas Karagiannidis
- Email: okaragia@ece.auth.gr

## References
1. Hu, Yujiao, Yuan Yao, and Wee Sun Lee. "A reinforcement learning approach for optimizing multiple traveling salesman problems over graphs." Knowledge-Based Systems 204 (2020): 106244. DOI: [10.1016/j.knosys.2020.106244](https://doi.org/10.1016/j.knosys.2020.106244).

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



