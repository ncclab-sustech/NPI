# NPI

[Mapping effective connectivity by virtually perturbing a surrogate brain](https://arxiv.org/abs/2301.00148)

## **Abstract**

Effective connectivity (EC), indicative of the causal interactions between brain regions, is fundamental to understanding information processing in the brain. Traditional approaches, which infer EC from neural responses to stimulations, are not suited for mapping whole-brain EC in humans due to being invasive and having limited spatial coverage of stimulations. To address this gap, we present Neural Perturbational Inference (NPI), a data-driven framework designed to map EC across the entire brain. NPI employs an artificial neural network trained to learn large-scale neural dynamics as a computational surrogate of the brain. NPI maps EC by perturbing each region of the surrogate brain and observing the resulting responses in all other regions. NPI captures the directionality, strength, and excitatory/inhibitory properties of brain-wide EC. Our validation of NPI, using models having ground-truth EC, shows its superiority over Granger causality and dynamic causal modeling. Applying NPI to resting-state fMRI data from diverse datasets reveals consistent and structurally supported EC. Further validation using a cortico-cortical evoked potentials dataset reveals a significant correlation between NPI-inferred EC and real stimulation propagation pathways. By transitioning from correlational to causal understandings of brain functionality, NPI marks a stride in decoding the brain's functional architecture and facilitating both neuroscience research and clinical applications.

## **Introduction**

<img src=".\img\NPI_framework.jpg" alt="NPI_framework" style="zoom:80%;" />

This repository contains the code and documentation of the NPI framework, a tool designed for mapping whole-brain EC in the brain. NPI operates by first training an ANN to mimic the complex dynamics of the brain based on the observed neural data. Once trained, virtual perturbations are applied to specific regions of the ANN to simulate the effect of neural stimulation. By monitoring the responses in other brain regions, NPI constructs a comprehensive map of the causal relationships, detailing how each area influences others across the brain. This process allows for the assessment of not only the presence but also the direction and excitatory/inhibitory properties of neural connections, contributing to a deeper comprehension of the brain's functional architecture.

## **Requirements**

**Operating Systems**: Windows 10, Ubuntu 20.04

**Python Version**: 3.12+

**Dependencies**: See `requirements.txt` file for details.

**Non-standard Hardware Requirements**: No non-standard hardware is required to run this software.

**Note on PyTorch**: We recommend installing the GPU version of PyTorch, especially if you plan to simulate large datasets. Running NPI can be computationally intensive and may take an extended period when running on CPU.

## **Installation**

1. Clone the repository: `git clone https://github.com/ncclab-sustech/NPI`.
2. Navigate to the project directory: `cd NPI`.
3. Install dependencies: `pip install -r requirements.txt`.

Typical installation time is approximately 5 minutes on a "normal" desktop computer.

## **How to run**

The notebook `.ipynb` files contain step-by-step instructions, including how to load and prepare your data, train an ANN as a surrogate brain, simulate long time series under noise-driven conditions and calculate the model functional connectivity (FC), and perturb the surrogate brain to obtain EC.

## **Demo examples**

Demos `NPI_demo(RNN).ipynb` and `NPI_demo(WBM).ipynb` show the application of NPI to simulated data derived from recurrent neural network (RNN) and whole-brain model (WBM) under given structural connectivity (SC). These demos serve to illustrate the process of applying NPI to infer EC, and comparing the NPI-inferred EC with ground-truth EC and SC. Demo `NPI_demo(fMRI).ipynb` shows applying of NPI on real individual fMRI data from HCP dataset to infer individual EC.

**Demo 1: RNN synthetic data**

- **Overview**: Demo `NPI_demo(RNN).ipynb` applies NPI to ground-truth RNN simulated data. It encompasses the generation of synthetic neural signals, obtaining ground-truth EC by direct perturbation of the RNN, applying NPI to infer EC from the RNN simulated data, and a comparison of the NPI-inferred EC with the ground-truth EC and SC, assessing the accuracy and reliability of the NPI framework in capturing directed causal interactions.
- **Simulation data resources**: Located in the `RNN_simulation data` folder, we provide the code necessary for using RNN to generate synthetic neural data and ground-truth EC. The folder contains detailed descriptions of the model parameters and simulation process.

**Demo 2: WBM synthetic data**

- **Overview**: Demo `NPI_demo(WBM).ipynb` applies NPI to WBM simulated data. It encompasses the generation of synthetic neural signals, obtaining ground-truth EC by direct perturbation of the WBM, applying NPI to infer EC from the WBM simulated data, and a comparison of the NPI-inferred EC with the ground-truth EC and SC, assessing the accuracy and reliability of the NPI framework in capturing direct causal interactions.
- **Simulation data resources**: Located in the `WBM_simulation data` folder, we provide the code necessary for using RNN to generate synthetic neural data and ground-truth EC. The folder contains detailed descriptions of the model parameters and simulation process.

**Demo 3: fMRI data from HCP dataset**

- **Overview**: Demo `NPI_demo(fMRI).ipynb` applies NPI to real fMRI data from HCP dataset of an individual. It encompasses applying NPI to infer EC from BOLD signals of an individual and a comparison of model FC and empirical FC.

**How to run the demos**

1. **Launch Jupyter Notebook**: Start from the project directory by launching the Jupyter Notebook.
2. **Open the demo notebook**: Select the demo notebook corresponding to either RNN or WBM synthetic data.
3. **Follow the instructions**: Each notebook provides step-by-step instructions for loading data, training the model, applying NPI, and analyzing results.
4. **Expected output**: The notebook will display output cells after execution. (Model FC closely resembles empirical FC, while NPI-inferred EC shows a strong correspondence with both ground-truth EC and SC.)
5. **Expected run time**: Expected total run time for the three demos is approximately 30 minutes on a "normal" desktop computer.

## **Whole-brain EC resources**

Within the `Whole-brain_EC` directory, we provide EC data across various brain atlases, offering insights into brain connectivity research. The EC matrices represent the causal interactions between different brain regions, derived from resting-state fMRI data of subjects within the Human Connectome Project (HCP) dataset.

Specifically, we calculated the EC matrices based on the MMP (360 regions), AAL (116 regions), MSDL (39 regions) atlases across 800 subjects and the Schaefer2018 atlases (with 100, 200, ..., 1000 regions) across 100 subjects. For each atlas, we calculated the individual EC matrices for all subjects and then averaged these matrices at the group level to obtain a representative EC matrix for each atlas. All EC matrices are normalized so that the strongest connection in each matrix has a value of 1.

We provide a notebook file `Whole-brain_EC/Whole-brain_EC.ipynb` that allow for the visualization of these EC matrices. Offering these detailed EC resources, we aim to facilitate research into the structural and functional interplay of the brain and support the development of novel diagnostic and therapeutic strategies in neuroscience.

<img src=".\img\ebc.png" alt="NPI_framework" style="zoom:80%;" />

## **How to cite**

If you use NPI in your research, please cite our paper:

- Luo, Z., Peng, K., Liang, Z. et al. Mapping effective connectivity by virtually perturbing a surrogate brain. [ https://doi.org/10.48550/arXiv.2301.00148](https://doi.org/10.48550/arXiv.2301.00148)

## **Contact**

For any questions/comments, please contact [NCC Lab](https://www.sustech.edu.cn/en/faculties/liuquanying.html).

## **Copyright**

Copyright © 2024 NCC Lab, Southern University of Science and Technology, Shenzhen, China.