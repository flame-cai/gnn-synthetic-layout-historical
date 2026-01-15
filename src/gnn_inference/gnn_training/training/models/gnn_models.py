import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing, SGConv, GATv2Conv, SplineConv

class MLP(nn.Module):
    """A simple Multi-Layer Perceptron for edge prediction."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
        self.layers.append(nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class EdgeClassifier(nn.Module):
    """
    A generic GNN-based edge classifier.
    It takes a GNN backbone and uses its output to predict edge labels.
    """
    def __init__(self, backbone, in_channels, edge_feat_dim, hidden_channels, dropout):
        super().__init__()
        self.backbone = backbone
        
        # The MLP takes the concatenated features of two nodes and optional edge features
        mlp_in_channels = 2 * hidden_channels + edge_feat_dim
        self.predictor = MLP(mlp_in_channels, hidden_channels, 2, num_layers=3, dropout=dropout)

    def forward(self, x, edge_index, edge_attr=None):
        # 1. Get node embeddings from the GNN backbone
        node_embeds = self.backbone(x, edge_index, edge_attr)

        # 2. For each edge, concatenate the embeddings of its source and target nodes
        source_embeds = node_embeds[edge_index[0]]
        target_embeds = node_embeds[edge_index[1]]
        
        edge_representation = torch.cat([source_embeds, target_embeds], dim=1)
        
        # 3. (Optional) Concatenate edge features
        if edge_attr is not None:
            edge_representation = torch.cat([edge_representation, edge_attr], dim=1)

        # 4. Predict edge logits
        edge_logits = self.predictor(edge_representation)
        return edge_logits

# --- GNN Backbone Definitions ---

class GCNBackbone(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None): # edge_attr is unused by GCN
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class GATBackbone(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, heads, dropout):
        super().__init__()
        self.convs = nn.ModuleList()
        # Edge attributes can be passed to GATv2Conv if needed, but it's not the standard implementation
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) -1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

# --- START OF SGC MODIFICATION ---

class SGCBackbone(nn.Module):
    """
    A Simple Graph Convolution backbone.
    It consists of a single SGConv layer that propagates features K times.
    """
    def __init__(self, in_channels, hidden_channels, num_layers):
        super().__init__()
        # The 'num_layers' from config corresponds to the 'K' parameter in SGConv
        self.conv = SGConv(in_channels, hidden_channels, K=num_layers, cached=False)

    def forward(self, x, edge_index, edge_attr=None):
        # SGConv does not use edge_attr or dropout between layers
        return self.conv(x, edge_index)

# --- END OF SGC MODIFICATION ---


# --- START OF MPNN MODIFICATION ---

class MPNNLayer(MessagePassing):
    """
    A single layer of the Message Passing Neural Network that uses edge attributes.
    """
    def __init__(self, in_channels, out_channels, edge_dim, dropout):
        super().__init__(aggr='add') # "Add" aggregation.
        # This network computes the message based on source, target, and edge features.
        # Message = MLP(source_node || target_node || edge_feature)
        self.message_net = nn.Sequential(
            nn.Linear(2 * in_channels + edge_dim, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # This network updates the node features after aggregation.
        self.update_net = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_attr has shape [E, edge_dim]

        # propagate() will call message() and update()
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Update node features: MLP(old_node_features || aggregated_messages)
        return self.update_net(torch.cat([x, out], dim=1))

    def message(self, x_i, x_j, edge_attr):
        # x_i: Source node features [E, in_channels]
        # x_j: Target node features [E, in_channels]
        # edge_attr: Edge features [E, edge_dim]
        
        # Concatenate features and pass through the message network
        tmp = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.message_net(tmp)


class MPNNBackbone(nn.Module):
    """
    A multi-layer MPNN backbone that uses edge attributes.
    """
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, edge_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        # First layer maps from input channels to hidden channels
        self.layers.append(MPNNLayer(in_channels, hidden_channels, edge_dim, dropout))
        # Subsequent layers map from hidden channels to hidden channels
        for _ in range(num_layers - 1):
            self.layers.append(MPNNLayer(hidden_channels, hidden_channels, edge_dim, dropout))

    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is None:
            # Create a dummy tensor if edge attributes are missing
            edge_attr = torch.zeros((edge_index.shape[1], 1), device=x.device)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x


class SplineConvBackbone(nn.Module):
    """
    A multi-layer SplineConv backbone that uses edge attributes.
    It includes a pre-processing step to reduce high-dimensional edge features.
    """
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, edge_dim, kernel_size, spline_edge_dim=3):
        super().__init__()
        
        self.edge_dim = edge_dim 
        self.spline_edge_dim = spline_edge_dim
        
        if self.edge_dim == 0:
            self.edge_preprocessor = None
        else:
            # self.edge_preprocessor = nn.Linear(self.edge_dim, self.spline_edge_dim)
            # --- MODIFICATION START: Using an MLP instead of a single Linear layer ---
            # Define an MLP with one hidden layer (e.g., of size 2 * spline_edge_dim)
            mlp_hidden_dim = 2 * spline_edge_dim 
            self.edge_preprocessor = nn.Sequential(
                nn.Linear(self.edge_dim, mlp_hidden_dim),
                nn.ReLU(),  # Add a non-linear activation function
                # nn.Dropout(p=0.2), # Optional: Add dropout for regularization
                nn.Linear(mlp_hidden_dim, self.spline_edge_dim)
            )
            # --- MODIFICATION END ---

        self.convs = nn.ModuleList()
        self.convs.append(SplineConv(in_channels, hidden_channels, dim=self.spline_edge_dim, kernel_size=kernel_size, aggr='sum'))
        for _ in range(num_layers - 1):
            self.convs.append(SplineConv(hidden_channels, hidden_channels, dim=self.spline_edge_dim, kernel_size=kernel_size, aggr='sum'))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is None:
            processed_edge_attr = torch.zeros((edge_index.shape[1], self.spline_edge_dim), device=x.device)
        else:
            processed_edge_attr = self.edge_preprocessor(edge_attr)
            
            processed_edge_attr = torch.sigmoid(processed_edge_attr)
  
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, processed_edge_attr)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# --- Model Factory ---
def get_gnn_model(model_name: str, config: dict, data_sample) -> nn.Module:
    """Factory function to create a GNN model for training."""
    model_cfg = config['model_configs'][model_name]
    
    in_channels = data_sample.num_node_features
    # This is the TRUE edge feature dimension from the data (can be 0)
    edge_feat_dim = data_sample.num_edge_features if hasattr(data_sample, 'edge_attr') and data_sample.edge_attr is not None else 0
    
    if in_channels == 0:
        raise ValueError("Input data has 0 node features.")
        
    if model_name == "GCN":
        backbone = GCNBackbone(in_channels, model_cfg['hidden_channels'], model_cfg['num_layers'], model_cfg['dropout'])
    elif model_name == "GAT":
        backbone = GATBackbone(in_channels, model_cfg['hidden_channels'], model_cfg['num_layers'], model_cfg['heads'], model_cfg['dropout'])
    elif model_name == "SGC":
        backbone = SGCBackbone(in_channels, model_cfg['hidden_channels'], model_cfg['num_layers'])
    
    # --- SIMPLIFICATION STARTS HERE ---
    elif model_name == "MPNN":
        # Let MPNN handle the case where edge_feat_dim might be 0
        backbone = MPNNBackbone(in_channels, model_cfg['hidden_channels'], model_cfg['num_layers'], model_cfg['dropout'], edge_dim=edge_feat_dim)
    elif model_name == "SplineCNN":
        # Pass the true edge_feat_dim. The backbone will handle the reduction.
        # Also pass the new spline_edge_dim from the config if it exists.
        spline_dim = model_cfg.get('spline_edge_dim', 3) # Default to 3 if not in config
        backbone = SplineConvBackbone(in_channels, model_cfg['hidden_channels'], model_cfg['num_layers'], model_cfg['dropout'], edge_dim=edge_feat_dim, kernel_size=model_cfg['kernel_size'], spline_edge_dim=spline_dim)
    # --- SIMPLIFICATION ENDS HERE ---
    
    else:
        raise ValueError(f"Unknown GNN model: {model_name}")

    # Determine the number of channels output by the backbone for the final predictor MLP
    if model_name == "GAT":
        final_hidden_channels = model_cfg['hidden_channels'] * model_cfg['heads']
    else:
        final_hidden_channels = model_cfg['hidden_channels']

    # For the predictor, use the original edge_feat_dim, as it concatenates the original features
    return EdgeClassifier(backbone, in_channels, edge_feat_dim, final_hidden_channels, model_cfg['dropout'])