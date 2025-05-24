# Train-Validation Loop

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from tqdm import tqdm
import numpy as np
import os
import traceback
from typing import Tuple, Optional

def to_device(data, device):
    """Moves a PyG Data or Batch object to the specified device."""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    if hasattr(data, 'to'):
        return data.to(device)
    return data # If not a PyG object or tensor

# --- Training Function ---
def train_epoch(model: nn.Module,
                loader: DataLoader,
                optimizer: optim.Optimizer,
                criterion: nn.Module, # Loss function, e.g., nn.BCELoss() or nn.BCEWithLogitsLoss()
                device: torch.device,
                epoch_num: int,
                log_interval: int = 50):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model to train.
        loader: DataLoader for the training data.
        optimizer: The optimizer.
        criterion: The loss function.
        device: The device to train on ('cuda' or 'cpu').
        epoch_num (int): Current epoch number for logging.
        log_interval (int): How often to print batch loss.

    Returns:
        float: Average training loss for the epoch.
        float: Average training accuracy for the epoch.
    """
    model.train()  # Set the model to training mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    processed_batches = 0

    # Wrap loader with tqdm for a progress bar
    progress_bar = tqdm(loader, desc=f"Epoch {epoch_num+1} [Train]", leave=True)

    for batch_idx, batch_data in enumerate(progress_bar):
        try:
            batch_data = to_device(batch_data, device)
            optimizer.zero_grad()  # Clear previous gradients

            # Forward pass: model expects data.x and data.h if using MyLorentzNet example
            # Ensure your batch_data has these attributes correctly populated
            if not hasattr(batch_data, 'x_coords') or not hasattr(batch_data, 'h_scalars'):
                print(f"Warning: Batch {batch_idx} missing 'x' or 'h' attributes. Skipping.")
                continue

            outputs = model(batch_data)  # Get raw logits or sigmoid outputs

            # Ensure outputs and batch_data.y have compatible shapes and types
            if outputs is None:
                print(f"Warning: Model output is None for batch {batch_idx}. Skipping.")
                continue
            if batch_data.y is None:
                print(f"Warning: Batch {batch_idx} missing 'y' attribute. Skipping.")
                continue
                
            loss = criterion(outputs.squeeze(), batch_data.y.float().squeeze())

            loss.backward() 
            optimizer.step()

            total_loss += loss.item() * batch_data.num_graphs

            preds = (outputs.squeeze() >= 0.5).long()
            correct_predictions += (preds == batch_data.y.squeeze().long()).sum().item()
            total_samples += batch_data.num_graphs
            processed_batches += 1

            if (batch_idx + 1) % log_interval == 0:
                current_loss = total_loss / total_samples if total_samples > 0 else 0
                current_acc = correct_predictions / total_samples if total_samples > 0 else 0
                progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}")

        except Exception as e:
            print(f"\n--- ERROR during training batch {batch_idx} in epoch {epoch_num+1} ---")
            print(f"Exception Type: {type(e)}")
            print(f"Exception Value: {e}")
            print("Traceback:")
            traceback.print_exc()
            print("--- Skipping this batch ---")
            continue

    if processed_batches == 0:
        print(f"Warning: Epoch {epoch_num+1} [Train] - No batches processed. Check data or error messages.")
        return 0.0, 0.0

    avg_loss = total_loss / total_samples
    avg_accuracy = correct_predictions / total_samples
    return avg_loss, avg_accuracy

# --- Validation Function ---
def evaluate_model(model: nn.Module,
                   loader: DataLoader,
                   criterion: nn.Module,
                   device: torch.device,
                   epoch_num: int # For logging purposes
                   ) -> Tuple[float, float, float, Optional[dict]]:
    """
    Evaluates the model on the validation or test set.

    Args:
        model: The PyTorch model to evaluate.
        loader: DataLoader for the validation/test data.
        criterion: The loss function.
        device: The device to evaluate on.
        epoch_num (int): Current epoch number for logging context.

    Returns:
        Tuple containing:
            - float: Average validation/test loss.
            - float: Average validation/test accuracy.
            - float: ROC AUC score.
            - dict: Classification report dictionary (or None if error).
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_outputs_proba = []
    processed_batches = 0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch_num+1} [Eval]", leave=True)

    with torch.no_grad():  # Disable gradient calculations
        for batch_idx, batch_data in enumerate(progress_bar):
            try:
                batch_data = to_device(batch_data, device)

                if not hasattr(batch_data, 'x_coords') or not hasattr(batch_data, 'h_scalars'):
                    print(f"Warning: Eval Batch {batch_idx} missing 'x' or 'h' attributes. Skipping.")
                    continue

                outputs = model(batch_data)
                if outputs is None:
                    print(f"Warning: Model output is None for eval batch {batch_idx}. Skipping.")
                    continue
                if batch_data.y is None:
                    print(f"Warning: Eval Batch {batch_idx} missing 'y' attribute. Skipping.")
                    continue

                loss = criterion(outputs.squeeze(), batch_data.y.float().squeeze())
                total_loss += loss.item() * batch_data.num_graphs

                # Assuming outputs are probabilities (model ends with sigmoid)
                probabilities = outputs.squeeze().cpu().numpy()
                preds = (probabilities >= 0.5).astype(int)
                labels = batch_data.y.squeeze().cpu().numpy().astype(int)

                all_outputs_proba.extend(probabilities.tolist())
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())
                processed_batches +=1

            except Exception as e:
                print(f"\n--- ERROR during evaluation batch {batch_idx} in epoch {epoch_num+1} ---")
                print(f"Exception Type: {type(e)}")
                print(f"Exception Value: {e}")
                print("Traceback:")
                traceback.print_exc()
                print("--- Skipping this batch ---")
                continue

    if processed_batches == 0 or not all_labels:
        print(f"Warning: Epoch {epoch_num+1} [Eval] - No batches processed or no labels collected.")
        return 0.0, 0.0, 0.0, None

    avg_loss = total_loss / len(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    try:
        if len(np.unique(all_labels)) > 1:
            roc_auc = roc_auc_score(all_labels, all_outputs_proba)
        else:
            print("Warning: ROC AUC not computed because only one class present in y_true.")
            roc_auc = 0.0 # Or np.nan
    except ValueError as e:
        print(f"ValueError calculating ROC AUC: {e}. Setting AUC to 0.0.")
        roc_auc = 0.0

    report_dict = None
    try:
        report_dict = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    except Exception as e:
        print(f"Error generating classification report: {e}")


    return avg_loss, accuracy, roc_auc, report_dict

# --- Main Training Loop ---
def run_training_pipeline(
    model: nn.Module,
    train_dataset: list,
    val_dataset: list,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    device_str: str = 'cuda',
    model_save_path: str = './best_model.pt',
    log_interval_train: int = 50,
    verbose_frequency: int = 1
):
    """
    Main training and validation loop.
    """
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer and Loss
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.BCELoss()
    best_val_auc = 0.0

    train_loss_lst, train_acc_lst = [], []
    val_loss_lst, val_acc_lst = [], []

    print("\n--- Starting Training ---")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch, log_interval_train)
        val_loss, val_acc, val_auc, val_report = evaluate_model(model, val_loader, criterion, device, epoch)

        train_loss_lst.append(train_loss)
        train_acc_lst.append(train_acc)
        val_loss_lst.append(val_loss)
        val_acc_lst.append(val_acc)

        if ((epoch+1) % verbose_frequency) == 0:
            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}, Val   AUC: {val_auc:.4f}")
    
            if val_report:
                print(f"  Validation F1-Score (macro avg): {val_report['macro avg']['f1-score']:.4f}")
                print(f"  Validation F1-Score (weighted avg): {val_report['weighted avg']['f1-score']:.4f}")


        # Save the model if validation AUC improves
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            print(f"  New best validation AUC: {best_val_auc:.4f}. Saving model to {model_save_path}...")
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'auc': val_auc
                }, model_save_path)
            except Exception as e:
                print(f"Error saving model: {e}")
        print("-" * 50)

    print("\n--- Training Finished ---")
    print(f"Best Validation AUC achieved: {best_val_auc:.4f}")
    return train_loss_lst, train_acc_lst, val_loss_lst, val_acc_lst