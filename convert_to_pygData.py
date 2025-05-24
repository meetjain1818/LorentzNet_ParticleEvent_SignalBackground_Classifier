import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import json 
import os


def convert_to_pyg_graph(graph, normalize_features):
    """
    Convert the graph dictionary into a PyTorch Geometric Data object
    and add graph-level features like jet_multiplicity, inv_mass, and the event label.

    Parameters:
    -----------
    graph : dict
        Dictionary containing 'eventno', 'nodes', 'edges', and 'edge_index', 'jet_multiplicity', 'inv_mass', 'node_labels',
        'jet_btag_labels', 'event_label' for a single event.

    Returns:
    --------
    data : Data
        PyTorch Geometric Data object containing the graph and additional features.
    """
    # Extract node features, edge features, and edge index from the graph dictionary
    event_num = torch.tensor(graph['eventno'], dtype=torch.long)
    # edge_index = torch.tensor(graph['edge_index'], dtype=torch.long)  # Shape: (2, num_edges)
    edge_index_raw = graph['edge_index']
    if not edge_index_raw or len(edge_index_raw) == 0:  # Empty edge_index
        edge_index = torch.empty(2, 0, dtype=torch.long)  # Shape: (2, 0)
        edge_features = torch.empty(0, 1, dtype=torch.float)  # Shape: (0, 1)
    else:
        edge_index = torch.tensor(edge_index_raw, dtype=torch.long)
        # Ensure edge_index is 2D with shape [2, num_edges]
        if edge_index.dim() == 1:  # If 1D, assume it's a flat list like [0, 1]
            edge_index = edge_index.view(2, -1)  # Reshape to [2, num_edges]
        elif edge_index.dim() != 2 or edge_index.size(0) != 2:  # Invalid shape
            raise ValueError(f"Invalid edge_index shape: {edge_index.shape}. Expected [2, num_edges].")
    
    
    edge_features = torch.tensor(graph['edges'], dtype=torch.float).view(-1,1)  # Shape: (num_edges, 1)
    
    node_labels = torch.tensor(graph['node_labels'], dtype=torch.long).view(-1,1)

    jet_btag_raw = graph.get('jet_btag_label', [])

    # Ensure jet_btag_raw has the same length as the number of nodes
    if len(jet_btag_raw) != graph['num_nodes']:
        print(f"Warning: Event {graph.get('eventno', 'N/A')} - Mismatch between num_nodes ({num_nodes}) and length of jet_btag_label ({len(jet_btag_raw)}). Using NaNs for b-tags.")
        jet_btag_labels_tensor = torch.full((num_nodes, 1), float('nan'), dtype=torch.float)
    else:
        btag_labels_float = []
        for btag_val in jet_btag_raw:
            if btag_val is None or (isinstance(btag_val, str) and btag_val.lower() == 'nan'): # Handle 'nan' string if present
                btag_labels_float.append(float(-1))
            else:
                try:
                    btag_labels_float.append(float(btag_val))
                except (ValueError, TypeError):
                    print(f"Warning: Invalid b-tag value '{btag_val}' for event {graph.get('eventno', 'N/A')}. Using NaN.")
                    btag_labels_float.append(float(-1))

    if not btag_labels_float:
        jet_btag_labels_tensor = torch.empty((0, 1), dtype=torch.float) if num_nodes == 0 else torch.full((num_nodes, 1), float('nan'), dtype=torch.float)
    else:
        jet_btag_labels_tensor = torch.tensor(btag_labels_float, dtype=torch.float).view(-1, 1)

    node_features = torch.tensor(graph['nodes'], dtype=torch.float)
    h_scalars = torch.tensor(graph['h_scalars'], dtype = torch.float)
    x_coord = torch.tensor(graph['x_coords'], dtype = torch.float)

    if (normalize_features) & (node_features.size(0) != 0):
        node_features = (node_features - node_features.mean(dim=0)) / node_features.std(dim=0) #Normalised Features
    
    node_features_with_btaglabel = torch.cat((node_features[:, :2], jet_btag_labels_tensor), dim = 1)

    
            
    # Convert graph-level features (jet_multiplicity, inv_mass) and label to tensor
    graph_level_features = torch.tensor([graph['num_nodes'],
                                         graph['invMass_2leadingbj1p'],
                                         graph['invMass_2leadingbj'],
                                         graph['num_isophotons'],
                                         graph['num_btag_jets'],
                                         graph['leading_isophoton_pT']
                                        ], dtype=torch.float).view(1, -1)
    label_tensor = torch.tensor(graph['event_label'], dtype=torch.long)  # Event label (0 or 1)

    # Create the PyTorch Geometric Data object
    data = Data(
        x=node_features_with_btaglabel,               # Node features (3, num_features)
        edge_index=edge_index,         # Edge index (2, num_edges)
        edge_attr=edge_features,       # Edge features (num_edges, 1)
        y=label_tensor                 # Event label (0 or 1)
    )

    # Add custom graph-level features
    data.eventno = event_num
    data.number_of_nodes = graph_level_features[0, 0]
    data.inv_mass_2j1p = graph_level_features[0, 1]
    data.inv_mass_2j = graph_level_features[0, 2]
    data.num_isophotons = graph_level_features[0, 3]
    data.num_btag_jets = graph_level_features[0, 4]
    data.node_label = node_labels 
    data.jet_btag_label = jet_btag_labels_tensor
    data.isophoton_pT = graph_level_features[0, 5]
    data.h_scalars = h_scalars
    data.x_coords = x_coord

    return data

def convert_all_to_pyg_graphs(graphs,*, normalize_features = False):
    """
    Converts a list of event graphs to PyTorch Geometric-compatible Data objects.
    Also adds the graph-level features 'jet_multiplicity', 'inv_mass' and event labels.

    Parameters:
    -----------
    graphs : list of dicts
        List containing dictionaries with node, edge information for each event.
    dataframe : pd.DataFrame
        DataFrame containing the event-level features like 'jetmultiplicity', 'invmass_4j1p', and 'label'.

    Returns:
    --------
    pyg_graphs : list of Data
        List of PyTorch Geometric Data objects with added graph-level features and labels.
    """
    pyg_graphs = []
    print('Initializing the process...')
    with tqdm(total = len(graphs), desc = 'Progress', leave = True) as pbar:    
        for i, graph in enumerate(graphs):
            data = convert_to_pyg_graph(graph, normalize_features)
            pyg_graphs.append(data)
            pbar.update(1)
    print('Process completed successfully :)')
    return pyg_graphs