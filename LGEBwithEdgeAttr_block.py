# Custom Lorentz Group Equivariant Block (LGEB)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, degree

def minkowski_dot_product(p1, p2):
    # p1, p2 are batches of 4-vectors: [..., 4]
    # Metric: (+, -, -, -)
    return p1[..., 0] * p2[..., 0] - torch.sum(p1[..., 1:] * p2[..., 1:], dim=-1)

def psi_normalization(z):
    # sgn(z) * log(|z| + 1)
    return torch.sign(z) * torch.log1p(torch.abs(z))


class LGEBConv(MessagePassing):
    def __init__(self,
                 scalar_in_channels: int,
                 scalar_out_channels: int,
                 edge_feature_dim: int, # Dimensionality of edge_attr\
                 coord_channels: int = 4, # Should always be 4 for 4-vectors
                 hidden_mlp_channels: int = 128,
                 edge_mlp_hidden_channels: int = 72,
                 coord_update_scale_c: float = 1e-5, # Hyperparameter c from paper
                 aggr: str = 'sum', # Aggregation for scalar messages
                 dropout_rate: float = 0.2):
        """
        Lorentz Group Equivariant Block (LGEB) Convolution Layer.

        Args:
            scalar_in_channels (int): Dimensionality of input scalar features (h_in).
            scalar_out_channels (int): Dimensionality of output scalar features (h_out).
            coord_channels (int): Dimensionality of coordinate features (x), typically 4.
            hidden_mlp_channels (int): Hidden dimension for MLPs updating h and x.
            edge_mlp_hidden_channels (int): Hidden dimension for MLPs computing edge messages.
            coord_update_scale_c (float): Scaling factor 'c' for coordinate updates.
            aggr (str): Aggregation method for scalar messages ('mean', 'sum', 'max').
            dropout_rate (float): Dropout rate for MLPs.
        """
        super(LGEBConv, self).__init__(aggr=aggr, flow="source_to_target") # Messages flow from j to i

        self.scalar_in_channels = scalar_in_channels
        self.scalar_out_channels = scalar_out_channels
        self.coord_channels = coord_channels
        self.coord_update_scale_c = coord_update_scale_c

        # MLP for edge message Φe: (2*h_dim + 3 scalars) -> hidden -> hidden
        # Inputs to Φe: hi, hj, ψ(||xi - xj||^2), ψ((xi, xj)), deltaR_ij
        # The Minkowski products are scalars.
        self.phi_e = nn.Sequential(
            nn.Linear(2 * scalar_in_channels + 2 + edge_feature_dim, edge_mlp_hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(edge_mlp_hidden_channels, edge_mlp_hidden_channels),
        )

        # MLP for edge weight Φm (for scalar update): edge_msg_dim -> 1 (then sigmoid)
        self.phi_m_weight = nn.Sequential(
            nn.Linear(edge_mlp_hidden_channels, 1),
            nn.Sigmoid()
        )

        # MLP for scalar feature update Φh: (h_in_dim + aggregated_msg_dim) -> hidden -> h_out_dim
        # Aggregated message dimension is edge_mlp_hidden_channels
        self.phi_h_update = nn.Sequential(
            nn.Linear(scalar_in_channels + edge_mlp_hidden_channels, hidden_mlp_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_mlp_channels, scalar_out_channels)
        )

        # MLP for coordinate attention Φx: edge_msg_dim -> 1 (scalar attention weight)
        self.phi_x_attention = nn.Sequential(
            nn.Linear(edge_mlp_hidden_channels, 1)
            # No sigmoid here, attention can be any scalar. Paper doesn't specify normalization.
        )

        # Linear layer for residual connection on scalar features (if channels change)
        if scalar_in_channels != scalar_out_channels:
            self.h_residual_transform = nn.Linear(scalar_in_channels, scalar_out_channels)
        else:
            self.h_residual_transform = nn.Identity()


    def forward(self, x_coords, h_scalars, edge_index, edge_attr):
        """
        Forward pass of the LGEB layer.

        Args:
            x_coords (Tensor): Node coordinate features (4-vectors), shape [N, coord_channels].
            h_scalars (Tensor): Node scalar features, shape [N, scalar_in_channels].
            edge_index (LongTensor): Graph connectivity, shape [2, E].
            edge_attr (Tensor): Edge features (e.g., DeltaR), shape [E, edge_feature_dim].

        Returns:
            Tuple[Tensor, Tensor]: Updated x_coords_out, h_scalars_out
        """
        if x_coords.shape[1] != self.coord_channels:
            raise ValueError(f"Input x_coords have {x_coords.shape[1]} channels, expected {self.coord_channels}")
        if h_scalars.shape[1] != self.scalar_in_channels:
             raise ValueError(f"Input h_scalars have {h_scalars.shape[1]} channels, expected {self.scalar_in_channels}")
        if edge_attr is not None and edge_attr.size(0) != edge_index.size(1):
            raise ValueError(f"Mismatch between number of edges in edge_attr ({edge_attr.size(0)}) and edge_index ({edge_index.size(1)})")
        # For scalar update:
        # propagate_type: (x_coord: Tensor, h_scalar: Tensor)
        aggregated_scalar_messages = self.propagate(
            edge_index,
            x_coord=x_coords,
            h_scalar=h_scalars,
            edge_features_prop=edge_attr,
            mode='scalar_update'
        )

        aggregated_coord_updates = self.propagate(
            edge_index,
            x_coord=x_coords,
            h_scalar=h_scalars,
            edge_features_prop=edge_attr,
            mode='coord_update'
        )
        # aggregated_coord_updates shape: [N, coord_channels]

        # --- 3. Update Scalar Features (h) --- (Equation 3.4)
        # h_i^{l+1} = h_i^l + Φh(h_i^l, aggregated_scalar_messages_i)
        h_scalars_combined = torch.cat([h_scalars, aggregated_scalar_messages], dim=1)
        h_scalars_out = self.h_residual_transform(h_scalars) + self.phi_h_update(h_scalars_combined)

        # --- 5. Update Coordinate Features (x) --- (Equation 3.3)
        # x_i^{l+1} = x_i^l + c * aggregated_coord_updates_i
        x_coords_out = x_coords + self.coord_update_scale_c * aggregated_coord_updates

        return x_coords_out, h_scalars_out

    def message(self, h_scalar_i, h_scalar_j, x_coord_i, x_coord_j, edge_features_prop, mode: str):
        """
        Computes messages from j to i.

        Args:
            h_scalar_i (Tensor): Scalar features of target nodes i, shape [E, scalar_in_channels].
            h_scalar_j (Tensor): Scalar features of source nodes j, shape [E, scalar_in_channels].
            x_coord_i (Tensor): Coordinate features of target nodes i, shape [E, coord_channels].
            x_coord_j (Tensor): Coordinate features of source nodes j, shape [E, coord_channels].
            mode (str): 'scalar_update' or 'coord_update' to determine what to return.

        Returns:
            Tensor: If mode is 'scalar_update', returns weighted edge messages mij_weighted.
                    If mode is 'coord_update', returns weighted coordinates xj_weighted_for_attention.
        """
        # --- 1. Calculate Lorentz Invariants for the edge (i, j) ---
        # Minkowski dot product (xi, xj)
        dot_product_xi_xj = minkowski_dot_product(x_coord_i, x_coord_j) # Shape [E]
        # Squared Minkowski norm ||xi - xj||^2
        diff_x = x_coord_i - x_coord_j # Shape [E, coord_channels]
        # For ||p||^2 = p_0^2 - p_vec^2. So ||xi-xj||^2 = (Ei-Ej)^2 - (px_i-px_j)^2 - ...
        # This is minkowski_dot_product(diff_x, diff_x)
        squared_norm_xi_minus_xj = minkowski_dot_product(diff_x, diff_x) # Shape [E]

        # Apply psi normalization ψ(z) = sgn(z) * log(|z| + 1)
        psi_dot_product = psi_normalization(dot_product_xi_xj)
        psi_squared_norm = psi_normalization(squared_norm_xi_minus_xj)

        if edge_features_prop is None:
            raise ValueError("edge_features_prop (DeltaR) is None in message function. This should be provided.")

        if edge_features_prop.dim() == 1:
            edge_features_prop_clean = torch.nan_to_num(edge_features_prop, nan=0.0).unsqueeze(-1)
        else:
            edge_features_prop_clean = torch.nan_to_num(edge_features_prop, nan=0.0)
        # Prepare input for Φe: concatenate [hi, hj, ψ_norm, ψ_dot_prod]
        # Reshape invariants to [E, 1] to concatenate with h_scalars [E, D_h]
        phi_e_input = torch.cat([
            h_scalar_i, h_scalar_j,
            psi_squared_norm.unsqueeze(-1),
            psi_dot_product.unsqueeze(-1),
            edge_features_prop_clean
        ], dim=-1) # Shape [E, 2*scalar_in_channels + 3]

        # --- Compute raw edge message mij --- (Equation 3.2, output of Φe)
        mij = self.phi_e(phi_e_input) # Shape [E, edge_mlp_hidden_channels]

        if mode == 'scalar_update':
            # --- 2. Compute Edge Weight wij (for scalar update) --- (Part of Equation 3.4)
            # wij = Φm(mij)
            wij = self.phi_m_weight(mij) # Shape [E, 1]
            # Return weighted message: wij * mij
            return wij * mij # Shape [E, edge_mlp_hidden_channels]

        elif mode == 'coord_update':
            # --- 4. Compute Coordinate Attention Weight αij --- (Part of Equation 3.3)
            # αij = Φx(mij)
            alpha_ij = self.phi_x_attention(mij) # Shape [E, 1]
            # Return xj weighted by attention: xj * αij
            return x_coord_j * alpha_ij # Shape [E, coord_channels]
        else:
            raise ValueError(f"Invalid mode in message function: {mode}")