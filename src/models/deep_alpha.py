import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for dynamic feature weighting.
    Adapted for 1D feature vectors.
    """
    def __init__(self, input_dim, reduction=4):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // reduction, input_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [batch_size, input_dim]
        weight = self.fc(x)
        return x * weight

class DeepAlphaModel(nn.Module):
    """
    Cutting-edge HFT Alpha Prediction Model.
    Features: SE-Block, Multi-layer MLP with Dropout and BatchNorm.
    """
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], num_classes=3, dropout=0.2):
        super(DeepAlphaModel, self).__init__()

        self.se_block = SEBlock(input_dim)

        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim

        self.mlp = nn.Sequential(*layers)
        self.classifier = nn.Linear(curr_dim, num_classes)

    def forward(self, x):
        x = self.se_block(x)
        x = self.mlp(x)
        logits = self.classifier(x)
        return logits

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Formula: loss = -at * (1 - pt)^gamma * log(pt)
    """
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
