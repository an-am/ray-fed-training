# Federated Learning with RayFed in Python

RayFed is an open-source **party-based** distributed computing framework for Federated
Learning (FL) **built upon the Ray ecosystem**. It acts as a connector layer that brings
Federated Learning capabilities to Ray, allowing multiple parties, i.e. the federated clients,
to jointly train models without sharing raw data. By leveraging Ray Core, RayFed inherits
Ray’s scheduling, distributed object store and resource management features for scaling
machine learning tasks. More on RayFed [here](https://github.com/ray-project/rayfed).

In this implementation, a **fixed set** of federated clients train their local neural network on their own dataset, satisfying **data locality** principles.
A **server** aggregates new model weights, following the **Federated Averaging** algorithm, for a total of 10 rounds.
