from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    auc, roc_curve, precision_recall_curve, confusion_matrix, 
    det_curve, DetCurveDisplay, class_likelihood_ratios
)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import random
import torch_geometric
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import DataLoader

def check_pyg_cuda_status():
    """
    Display the CUDA and PyTorch Geometric (PyG) configuration, including version and GPU status.
    Verifies the installation of key PyG dependencies (torch_scatter, torch_sparse, etc.) and runs a practical
    test with a GCNConv layer to ensure CUDA-based computations are functional.
    """
    print("System CUDA Status:")
    print(f"- PyTorch version: {torch.__version__}")
    print(f"- CUDA Version: {torch.version.cuda}")
    print(f"- CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"- CUDA version: {torch.version.cuda}")
        print(f"- CUDA device: {torch.cuda.get_device_name(0)}")
    
    print("\nPyTorch Geometric Status:")
    print(f"- PyG version: {torch_geometric.__version__}")
    
    try:
        import torch_scatter
        print("- torch_scatter: Installed")
    except ImportError:
        print("- torch_scatter: Not installed")
    
    try:
        import torch_sparse
        print("- torch_sparse: Installed")
    except ImportError:
        print("- torch_sparse: Not installed")
    
    try:
        import torch_cluster
        print("- torch_cluster: Installed")
    except ImportError:
        print("- torch_cluster: Not installed")
    
    try:
        import torch_spline_conv
        print("- torch_spline_conv: Installed")
    except ImportError:
        print("- torch_spline_conv: Not installed")

    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    x = torch.randn(2, 10)
    if torch.cuda.is_available():
        edge_index = edge_index.cuda()
        x = x.cuda()
        try:
            conv = torch_geometric.nn.GCNConv(10, 10).cuda()
            out = conv(x, edge_index)
            print("\nPractical Test: Successfully ran GCNConv on GPU")
        except Exception as e:
            print(f"\nPractical Test Failed: {str(e)}")

def set_seed(seed:int):
    """
    Set random seed for reproducibility across Python's random module, NumPy, and PyTorch.

    Parameters:
    - seed (int): The seed value to ensure deterministic behavior.
    """
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def init_setup(batch_size:int=70, use_cuda:bool=True, random_seed:int=1234) -> torch.device:
    """
    Initialize the computational environment by setting up CUDA, batch size, and seed for reproducibility.

    Parameters:
    - batch_size (int): The batch size for training.
    - use_cuda (bool): Flag to enable or disable CUDA usage.
    - random_seed (int): Seed for reproducibility.
    """
    BATCH_SIZE = batch_size
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name()}")
        print(f"GPU Memory Usage: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    check_pyg_cuda_status()
    set_seed(random_seed)
    return device


def train(model:torch.nn.Module, loader:torch_geometric.data.DataLoader, 
          device:torch.device, optimizer:torch.optim, criterion:torch.nn.modules.loss) -> float:
    """
    Trains a PyTorch Geometric model for one epoch using the provided data loader, optimizer, and loss function.

    Parameters:
    -----------
    model : torch.nn.Module
        The PyTorch Geometric model to be trained.
    loader : torch_geometric.data.DataLoader
        The DataLoader providing batches of graph data for training.
    device : torch.device
        The device (e.g., 'cuda' or 'cpu') to which the model and data should be moved for computation.
    optimizer : torch.optim.Optimizer
        The optimizer used to update the model parameters.
    criterion : torch.nn.modules.loss._Loss
        The loss function used to compute the training loss.

    Returns:
    --------
    float
        The average training loss across all batches in the loader.

    Notes:
    ------
    - The function sets the model to training mode using `model.train()`.
    - For each batch, it computes the loss, performs backpropagation, and updates the model parameters.
    - Loss values from all batches are accumulated and averaged to return the training loss for the epoch.

    Example:
    --------
    >>> train_loss = train(model, train_loader, device, optimizer, criterion)
    >>> print(f"Training Loss: {train_loss:.4f}")
    """
    model.train()
    total_loss = 0
    for data in loader:  
        data = data.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(data).squeeze()
        loss = criterion(out, data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)



def test(model:torch.nn.Module, loader:torch_geometric.data.DataLoader,
         device:torch.device, optimizer:torch.optim, criterion:torch.nn.modules.loss)  -> float:
    """
    Evaluates a PyTorch Geometric model on the test/validation dataset to compute accuracy.

    Parameters:
    -----------
    model : torch.nn.Module
        The PyTorch Geometric model to be evaluated.
    loader : torch_geometric.data.DataLoader
        The DataLoader providing batches of graph data for testing/validation.
    device : torch.device
        The device (e.g., 'cuda' or 'cpu') to which the model and data should be moved for computation.
    optimizer : torch.optim.Optimizer
        (Unused in this function but included for consistency with training).
    criterion : torch.nn.modules.loss._Loss
        (Unused in this function but included for consistency with training).

    Returns:
    --------
    float
        The accuracy of the model on the test/validation dataset, computed as the proportion of correctly 
        predicted labels.

    Notes:
    ------
    - The function sets the model to evaluation mode using `model.eval()` to disable dropout and other training-specific behaviors.
    - It iterates through each batch, computes predictions, and compares them to the ground truth labels.
    - Predicted labels are determined by applying a threshold of 0.5 to the model's output.
    - Accuracy is computed as the total number of correct predictions divided by the total number of samples in the dataset.

    Example:
    --------
    >>> test_accuracy = test(model, test_loader, device, optimizer, criterion)
    >>> print(f"Test Accuracy: {test_accuracy:.2%}")
    """
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data).squeeze()
        pred = (out > 0.5).float()
        correct += (pred == data.y).sum().item()
    return correct / len(loader.dataset)


def eval_model(y_true:list[float], y_pred_prob:list[float], y_pred_label:list[float],*, save_fig:bool = False, save_fig_path:str = None) -> None:
    """
    Evaluate the model's performance using various classification metrics and visualization.

    Parameters:
    - y_true (array-like): True class labels.
    - y_pred_prob (array-like): Predicted probabilities for the positive class.
    - y_pred_label (array-like): Predicted class labels.
    - save_fig (bool): If set to True then figure will be saved to specified filepath.
    - save_fig_path (str): Filepath where to save the figure.

    Displays metrics such as accuracy, precision, recall, F1-score, and likelihood ratios,
    and plots the ROC curve, Precision-Recall curve, confusion matrix heatmap, and prediction histogram.
    """
    accuracy = accuracy_score(y_true, y_pred_label)
    precision = precision_score(y_true, y_pred_label)
    recall = recall_score(y_true, y_pred_label)
    f1 = f1_score(y_true, y_pred_label)
    conf_matrix = confusion_matrix(y_true, y_pred_label)
    pos_LR, neg_LR = class_likelihood_ratios(y_true, y_pred_label)

    figure, ax = plt.subplots(2, 2, figsize=(8, 6), dpi=300)
    figure.suptitle("Model Evaluation Metrics", fontsize=16)

    # --- Metrics Table (0,0) ---
    ax[0,0].axis('off')
    metrics = [
        ['Positive LR', f"{pos_LR:.3f}"],
        ['Negative LR', f"{neg_LR:.3f}"],
        ['Accuracy', f"{accuracy:.3f}"],
        ['Precision', f"{precision:.3f}"],
        ['Recall', f"{recall:.3f}"],
        ['F1-Score', f"{f1:.3f}"]
    ]
    
    table = ax[0,0].table(
        cellText=metrics,
        colLabels=['Metric', 'Value'],
        cellLoc='center',
        loc='center',
        colColours=['#f0f0f0', '#f0f0f0']
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    # --- ROC Curve (0, 1) ---
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    ax[0, 1].plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {roc_auc:.3f}")
    ax[0, 1].plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    ax[0, 1].set_xlabel("False Positive Rate")
    ax[0, 1].set_ylabel("True Positive Rate")
    ax[0, 1].set_title('ROC Curve')
    ax[0, 1].legend(loc="lower right")
    ax[0, 1].grid(alpha=0.3)


    # --- Confusion Matrix (2,1) ---
    group_counts = ["{0:0.0f}".format(value) for value in conf_matrix.flatten()]
    group_percentage = np.array([np.round((row/total)*100, 2) for row, total in zip(conf_matrix, np.sum(conf_matrix, axis=1))]).flatten()
    labels = np.asarray([f"{counts}\n{percent}%" for counts, percent in zip(group_counts, group_percentage)]).reshape((2, 2))
    sns.heatmap(conf_matrix, annot=labels, fmt='', ax=ax[1, 0])
    ax[1, 0].set_xlabel("Predicted Labels")
    ax[1, 0].set_ylabel("True Labels")
    ax[1, 0].set_title("Confusion Matrix Heatmap")

    # --- Predicted Probabilities Histogram(2,2) ---
    ax[1, 1].hist(y_pred_prob, histtype='step', linewidth=2, label="Model's Prediction")
    ax[1, 1].hist(y_true, histtype='step', color='green', linewidth=2, label="Actual Labels", alpha=0.4)
    ax[1, 1].set_xlabel('Predicted Probabilities')
    ax[1, 1].set_ylabel('Frequency')
    ax[1, 1].set_title('Predicted Probabilities Histogram')
    ax[1, 1].legend()
    
    plt.tight_layout()
    if save_fig & (save_fig_path != None):
        plt.savefig(save_fig_path, bbox_inches = 'tight', pad_inches = 0.05)
        print(f"Figure saved to {save_fig_path} successfully :)")
    
    plt.show()
    
    return None


def Euclidean_distance(node1: list[float, float], node2: list[float, float]) -> float:
    """
    Compute the Euclidean distance between two nodes in a graph using (Eta, Phi) coordinates.

    Parameters:
    - node1 (list): A list containing Eta and Phi of the first node.
    - node2 (list): A list containing Eta and Phi of the second node.

    Returns:
    - float: The Euclidean distance considering periodic boundaries in Phi.
    """
    diff = node1 - node2
    if np.abs(diff[1]) > np.pi:
        diff[1] = 2 * np.pi - np.abs(diff[1])
    euclidean_distance = np.sqrt(np.sum(diff**2))
    return euclidean_distance

def invariant_mass(x: pd.Series) -> float:
    """
    Calculate the invariant mass of a system comprising four jets and one photon.

    Parameters:
    - x (pd.Series): A pandas Series containing energies and momenta (Px, Py, Pz) of
      jets and the isolated photon.

    Returns:
    - float: The invariant mass of the event.
    """
    total_energy = x.loc[['isophoton_E', 'jet1_E', 'jet2_E', 'jet3_E', 'jet4_E']].sum()
    total_px = x.loc[['isophoton_Px', 'jet1_Px', 'jet2_Px', 'jet3_Px', 'jet4_Px']].sum()
    total_py = x.loc[['isophoton_Py', 'jet1_Py', 'jet2_Py', 'jet3_Py', 'jet4_Py']].sum()
    total_pz = x.loc[['isophoton_Pz', 'jet1_Pz', 'jet2_Pz', 'jet3_Pz', 'jet4_Pz']].sum()

    inv_mass = np.sqrt(total_energy**2 - total_px**2 - total_py**2 - total_pz**2)
    return inv_mass


def visualize_graph(data, *, save_fig=False, save_fig_path=None) -> None:
    """
    Visualize a graph with node features, labels, and edge attributes.
    
    Parameters:
    -----------
    data : Data
        PyTorch Geometric Data object with:
        - x: Node features [η, φ, pT, E]
        - node_label: Node type labels
        - edge_attr: Edge distances (ΔR)
        - eventno: Event number
        - y: Signal (1) vs background (0)
    """
    G = to_networkx(data, to_undirected=True, edge_attrs=["edge_attr"])
    
    # Create labels and feature strings
    node_labels = {i: f'{label.item()}' for i, label in enumerate(data.node_label)}
    node_features = {
        i: (data.x[i][0].item(),   # η
            data.x[i][1].item(),   # φ
            data.x[i][2].item(),   # pT
            data.x[i][3].item(),   #E
            data.x[i][4].item())   # btag_label
        for i in range(data.num_nodes)
    }

    plt.figure(figsize=(5, 3))
    pos = nx.spring_layout(G, seed=42, k=0.15)

    # Draw base graph
    nx.draw(G, pos, with_labels=True, node_size=500,
            node_color='skyblue', font_size=10, 
            font_weight='bold', edge_color='gray', 
            labels=node_labels)

    # Add node feature annotations
    for node, (eta, phi, pT, E, btag) in node_features.items():
        x, y = pos[node]
        feature_str = (f"η: {eta:.2f}\n"
                       f"φ: {phi:.2f}\n"
                       f"pT: {pT:.1f}\n"
                       f"E: {E:.1f}\n"
                        f"btag: {btag:.1f}")
        plt.text(x+0.12, y-0.12, feature_str, 
                 ha='center', va='top', 
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'),
                 fontsize=8, fontfamily='monospace')

    # Add edge distances
    edge_labels = nx.get_edge_attributes(G, 'edge_attr')
    edge_labels = {key: round(edge_labels[key][0], 2) for key in edge_labels}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                font_size=8)

    # Custom title
    event_type = "Signal" if data.y.item() == 1 else "Background"
    plt.title(f"{event_type} Event No.{data.eventno.item()}", fontsize=12)

    if save_fig and save_fig_path:
        plt.savefig(save_fig_path, bbox_inches='tight', pad_inches=0.1, dpi = 300)
        print(f"Figure saved to {save_fig_path}")

    plt.show()
    return None


def get_graph_embeddings(model:torch.nn.Module, dataset:list[torch_geometric.data.Data], batch_size:int = 70) -> pd.DataFrame:
    """
    Extracts graph-level embeddings from a PyTorch Geometric model and combines them with true labels 
    and predicted probabilities in a DataFrame.

    Parameters:
    -----------
    model : torch.nn.Module
        The PyTorch Geometric model from which graph embeddings are extracted. The model should include 
        a `graph_embedding` attribute at the desired layer.
    dataset : list of torch_geometric.data.Data
        A dataset of graph data objects to be processed. Each graph data object must have a `y` attribute 
        representing the true label.
    batch_size : int, optional
        Batch size to use when processing the dataset (default is 70).

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the following columns:
        - Graph embeddings (one column per dimension of the embedding).
        - `true_label`: The true labels of the graphs.
        - `pred_prob`: The predicted probabilities for the graphs (as a float).

    Notes:
    ------
    - The function registers a forward hook to extract graph embeddings from a specific layer of the model.
    - The `graph_embedding` attribute is assumed to be accessible within the forward hook's output.
    - The function processes the dataset in batches, computes predictions, and detaches embeddings.
    - The returned DataFrame combines the graph embeddings, true labels, and predicted probabilities.

    Example:
    --------
    >>> embedding_df = get_graph_embeddings(model, dataset, batch_size=32)
    >>> print(embedding_df.head())
             0         1         2  ...  true_label  pred_prob
    0  -0.1286    0.4532   -0.0523  ...           1    0.8673
    1   0.2345   -0.1746    0.3891  ...           0    0.3541

    """
    
    graph_embeddings = []
    def hook_fn(module, input, output):
        graph_embeddings.append(module.graph_embedding.detach().cpu().tolist())
    
    handle = model.register_forward_hook(hook_fn)
    
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.eval()
    y_true_label = []
    y_pred_proba = []
    with torch.no_grad():
        for data in dataset_loader:
            y_true_label.extend(data.y.numpy())
            out = model(data)
            y_pred_proba.extend(out.numpy())
    
    handle.remove()
    graph_embedding_temp = []
    for embedding in graph_embeddings:
        graph_embedding_temp.extend(embedding)

    graph_embedding_temp = np.array(graph_embedding_temp)
    embedding_df = pd.DataFrame(graph_embedding_temp)
    embedding_df['true_label'] = y_true_label
    embedding_df['pred_prob'] = y_pred_proba
    embedding_df['pred_prob'] = embedding_df['pred_prob'].astype(float)

    return embedding_df


def get_labels_from_model(model:torch.nn.Module, dataset:list[torch_geometric.data.Data], thresh:float = 0.5) -> (list, list, list):
    """
    Extracts true labels, predicted probabilities, and predicted labels from a PyTorch model for a given dataset.

    Parameters:
    -----------
    model : torch.nn.Module
        The trained PyTorch model used to generate predictions for the dataset.
    dataset : list of torch.data.Data
        A dataset of graph data objects. Each graph data object must have a `y` attribute representing the true label.
    thresh : float, optional
        The threshold for converting predicted probabilities into binary labels (default is 0.5).

    Returns:
    --------
    tuple of (list, list, list)
        - y_true: List of true labels extracted from the dataset.
        - y_pred_prob: List of predicted probabilities for each graph.
        - y_pred_labels: List of binary predicted labels (0 or 1), derived by applying the threshold to `y_pred_prob`.

    Notes:
    ------
    - The function processes each graph in the dataset and extracts its true label (`y`) and predicted probability.
    - Predicted probabilities are obtained from the model's output, which is assumed to be a single scalar value per graph.
    - Binary labels are computed by comparing predicted probabilities to the specified threshold (`thresh`).

    Example:
    --------
    >>> y_true, y_pred_prob, y_pred_labels = get_labels_from_model(model, dataset, thresh=0.7)
    >>> print(y_true[:5])         # True labels
    [1, 0, 1, 1, 0]
    >>> print(y_pred_prob[:5])    # Predicted probabilities
    [0.85, 0.32, 0.91, 0.74, 0.28]
    >>> print(y_pred_labels[:5])  # Predicted binary labels
    [1, 0, 1, 1, 0]
    """
    y_true = []
    y_pred_prob = []
    for _, graph in enumerate(dataset):
        y_true.append(graph.y.float())
        out = model(graph).squeeze().detach().numpy()
        y_pred_prob.append(float(out))
    y_true, y_pred_prob = np.array(y_true), np.array(y_pred_prob)
    y_pred_labels = (y_pred_prob > thresh)
    return y_true, y_pred_prob, y_pred_labels