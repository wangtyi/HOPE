import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
    
    
class LinearProbe(nn.Module):
    """
    A simple linear probing model.

    Args:
        input_dim: Size of the input feature vector.
        target_classes: Number of output classes for prediction.

    Returns:
        Logits produced by the linear layer.
    """
    def __init__(self, input_dim, target_classes):
        super(LinearProbe, self).__init__()
        
        self.layer = nn.Linear(input_dim, target_classes)
        nn.init.xavier_uniform_(self.layer.weight)
        nn.init.zeros_(self.layer.bias)

    def forward(self, x):
        output = self.layer(x)
        
        return output

    

class FusionModel(nn.Module):
    """
    A fusion model that predicts both auxiliary labels and target labels from the input features.

    Args:
        input_dim: Size of the input feature vector.
        aux_classes: Number of auxiliary classes for the intermediate output.
        target_classes: Number of classes for the final target output.
        use_bn: Whether to apply batch normalization before the final linear layer.
        temperature: Whether to apply temperature scaling on the target logits.

    Returns:
        A tuple containing auxiliary output logits and target output logits (temperature-scaled if enabled).
    """
    def __init__(self, input_dim, aux_classes, target_classes, use_bn=False, temperature=False):
        super(FusionModel, self).__init__()

        self.temperature = temperature
        
        self.aux_layer = nn.Linear(input_dim, aux_classes)
        self.target_layer = (
            nn.Sequential(
                nn.BatchNorm1d(aux_classes),
                nn.Linear(aux_classes, target_classes),
            ) if use_bn
            else nn.Linear(aux_classes, target_classes)
        )
        if self.temperature:
            self.T = nn.Parameter(torch.ones(1))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        
    def forward(self, x):
        aux_output = self.aux_layer(x)
        target_output = self.target_layer(aux_output)
        if self.temperature:    
            target_output = target_output / self.T
       
        return aux_output, target_output

