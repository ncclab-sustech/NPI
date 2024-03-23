# Mapping Effective Connectivity by Virtually Perturbing a Surrogate Brain

### Introduction

<img src=".\img\NPI_framework.jpg" alt="NPI_framework" style="zoom:80%;" />

This repository provides demonstration scripts for generating synthetic neural data and for inferring EC (Effective Connectivity) using the NPI (Neural Perturbational Inference) framework. These demos serve as a foundational insight into the methodologies applied within our research.

### System Requirements

**Operating Systems**: Windows 10, Ubuntu 20.04

**Python Version**: 3.8+

**Dependencies**: See `requirements.txt` file for details.

**Non-standard Hardware Requirements**: No non-standard hardware is required to run this software.

**Note on PyTorch**: We recommend installing the GPU version of PyTorch, especially if you plan to simulate large datasets. Simulating data can be computationally intensive and may take an extended period when running on CPU.

### Installation Guide

1. Clone the repository: `git clone https://github.com/ncclab-sustech/NPI`.
2. Navigate to the project directory: `cd NPI`.
3. Install dependencies: `pip install -r requirements.txt`.

Typical install time is approximately 10 minutes on a "normal" desktop computer.

### Demo

1. To run the demo, launch Jupyter Notebook: `jupyter notebook`.
2. Open the `NPI_demo.ipynb` file within the Jupyter Notebook interface.
3. Follow the instructions within the notebook for running the demo.
4. Expected output: The notebook will display output cells after execution. (NPI-inferred FC closely resembles empirical-FC, while NPI-inferred EC shows a strong correspondence with both real-EC and SC.)
5. Expected run time for the demo is approximately 20 minutes on a "normal" desktop computer.

### Instructions for Use

To reproduce the quantitative results, please follow the steps outlined in `NPI_demo.ipynb`.

If you have any questions or encounter any issues, please do not hesitate to contact us.

### Whole-brain EC Resources

In the `Whole-brain_EC` folder, we offer EC data across various Atlases to aid your brain connectivity research. The entry at the i-th row and j-th column represents the EC from brain region i to brain region j.

### Simulation Data Resources

In the `MFM_simulation_data` and `RNN_simulation_data` folders, we provide code for generating simulation data along with the generated simulation datasets themselves. These datasets are intended for use in validating the effectiveness of the NPI framework.