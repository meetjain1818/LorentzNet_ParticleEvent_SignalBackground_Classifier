# LorentzNet for Signal/Background Classification in Particle Physics

This repository contains a PyTorch Geometric implementation of a Graph Neural Network (GNN) inspired by the principles of LorentzNet, adapted for a signal versus background classification task in particle physics event data. The primary goal is to leverage physics-informed architectures to distinguish between rare signal processes and dominant background processes.

## Introduction

Particle physics experiments generate vast amounts of complex data. Machine learning, particularly Graph Neural Networks (GNNs), has shown great promise in analyzing these datasets, for example, in classifying collision events as either containing a rare signal of new physics or being a known Standard Model background process.

This project implements a GNN pipeline that takes inspiration from LorentzNet, which emphasizes Lorentz group equivariance/invariance. While this implementation might not be a strict reproduction of the original LorentzNet for jet tagging, it aims to incorporate key ideas such_ as:

*   Using 4-vector information of particles.
*   Calculating Lorentz-invariant quantities as edge or global features.
*   Building graph structures from particle interactions.

The goal is to improve classification performance and potentially gain insights into the underlying physics driving the model's decisions.

## Features

*   **Data Loading:** Scripts to load event data from common formats (e.g., CSV, TXT).
*   **Preprocessing:** Filtering events based on physics criteria (e.g., number of b-jets, photons).
*   **Feature Engineering:** Calculation of high-level physics variables (DeltaR, invariant masses).
*   **GNN Data Conversion:** Conversion of tabular event data into PyTorch Geometric `Data` objects suitable for GNNs. The GNN graph representation includes:
    *   **Nodes:** Particles (jets, photons) with features like `[Eta, Phi, pT, E, b_tag_label]`.
    *   **Edges:** Interactions between particles, with features like `DeltaR`.
    *   **Global Features:** Event-level features like `invariant_mass_2j1p`, `invariant_mass_2j`, `leading_isophoton_pT`.
*   **Training Pipeline:** Script for training the GNN model with options for optimizers, loss functions, and learning rate schedulers.
*   **Evaluation:** Calculation of standard metrics (Accuracy, AUC, F1-score).
*   **Interpretability:**
    *   Functions to compute average node and edge feature importance using gradient-based saliency.
    *   Analysis of feature importance for different prediction outcomes (True Positives, True Negatives, False Positives, False Negatives).
    *   Visualization of importance scores.
*   **Hyperparameter Tuning:** A utility function similar to `GridSearchCV` for K-fold cross-validation based hyperparameter optimization.
*   **Model Checkpointing:** Functions to save and load trained model states.


## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/meetjain1818/LorentzNet_ParticleEvent_SignalBackground_Classifier.git
    cd lorentznet-event-classification
    ```

2.  **Install dependencies:**
    Install PyTorch according to your CUDA version from [pytorch.org](https://pytorch.org/). Then, install PyTorch Geometric and other packages:
    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation

### Input Data Format

The initial data is expected to be in a tabular format (e.g., CSV or tab-separated TXT files), where each row represents an event, and columns represent various particle features and event-level information.
Example columns might include:
`eventno`, `jet1_Eta`, `jet1_Phi`, `jet1_pT`, `jet1_E`, `jet1_btag`, ..., `jet15_btag`, `isophoton1_Eta`, ..., `isophotoncount`.

### Conversion to GNN Format

The pipeline includes scripts to:
1.  Filter events based on criteria like the number of b-tagged jets and isolated photons (see `src/preprocess.py`).
2.  Select specific particles (e.g., leading photons, specific b-jets) and engineer features like DeltaR and invariant masses.
3.  Convert these processed events into a list of dictionaries, where each dictionary represents a graph structure suitable for GNNs. This is typically saved as an intermediate JSON file (`./LorentzNet_Preprocessing.ipynb`). The structure of these dictionaries is defined as:
    ```json
    [
      {
        "eventno": 123,
        "nodes": [[eta1, phi1, pt1, e1], [eta2, phi2, pt2, e2], ...], // Node features
        "edges": [deltaR1, deltaR2, ...],                             // Edge features (DeltaR)
        "edge_index": [[source_idx_array], [target_idx_array]],       // Connectivity
        "node_labels": [0, 1, 1, ...],                                // 0 for isolated photon, 1 for jet
        "jet_btag_label": [-1, 1.0, 0.0, ...],                      // B-tag for jets, -1 for isolated photons
        "inv_mass_2j1p": 150.5,
        "inv_mass_2j": 90.2,
        "isophoton_pT": 55.3,
        "event_label": 1,                                              // 0 for background, 1 for signal
        "x_coords": [[E1, Px1, Py1, Pz1], [E2, Px2, Py2, Pz2], ...], // 4-Vectors of jets and isolated photons
        "h_scalars":[[particle_type, b_tag_status, invariant_mass]]
      },
      ...
    ]
    ```
4.  Finally, these dictionaries are converted into a list of PyTorch Geometric `Data` objects (see `./convert_to_pygData.py` for the conversion function `convert_all_to_pyg_graphs`).

## LorentzNet Architecture

The GNN architecture is defined in `./LGEB_block.py` or `./LGEBwithEdgeAttr_block.py`. The specific implementation aims to incorporate principles of Lorentz-invariant features in the edge computations and uses 4-vector components (`E, px, py, pz`) and Lorentz invariant quantities (`particle_type, b_tag_status, invariant_mass`) as scalar features.


## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact

*   Meet Jain - meetjain1818@gmail.com

Project Link: [https://github.com/meetjain1818/LorentzNet_ParticleEvent_SignalBackground_Classifier.git](https://github.com/meetjain1818/LorentzNet_ParticleEvent_SignalBackground_Classifier.git)