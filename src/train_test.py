import torch
import numpy as np
from src.model import LinearProbe, FusionModel


def train_lp(features, 
             targets,
             device, 
             epochs=1):
    """
    Train a linear probe that predicts class labels from input features.

    Args:
        features: Input features.
        targets: True labels.
        device: "cpu" or "cuda".
        epochs: Number of training epochs.

    Returns:
        Trained LinearProbe model.
    """
    data_type = torch.float32
    model = LinearProbe(features.shape[1], len(np.unique(targets))).to(device, dtype=data_type)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        max_iter=1000,
        line_search_fn="strong_wolfe"
    )
    criterion = torch.nn.CrossEntropyLoss()

    all_feats = torch.tensor(features).to(device, dtype=data_type)
    all_targets = torch.tensor(targets).to(device, dtype=torch.long)

    def compute_loss(target_pred, target_true):
        """
        Compute classification loss plus scaled L2-based regularization.
        """
        base_loss = criterion(target_pred, target_true)

        l2_reg = 0.0
        for layer in model.modules():
            if isinstance(layer, torch.nn.Linear):
                d1, d2 = layer.weight.shape
                c = (d1 * d2) / 100
                l2_reg += (0.5 / c) * layer.weight.norm(p=2)
        loss = base_loss.mean() + l2_reg

        return loss

    model.train()
    for epoch in range(1, epochs + 1):
        def closure():
            optimizer.zero_grad()
            target_output = model(all_feats)
            loss = compute_loss(target_output, all_targets)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            target_output = model(all_feats)
            loss = compute_loss(target_output, all_targets)

        print(f"Train Epoch: {epoch} \t Loss: {loss.item():.6f}")

    return model


def test_lp(features, model, device):
    """
    Evaluate the trained linear probe on the test set.
    This function predicts class labels and probabilities for all samples in the input features. 
    It is suitable for standard classification evaluation where each sample has its own label.

    Args:
        features: Input features.
        model: Trained LinearProbe model.
        device: "cpu" or "cuda".

    Returns:
        target_preds and target_probs for all samples.
    """
    data_type = next(model.parameters()).dtype
    model.eval()
    with torch.no_grad():
        all_feats = torch.tensor(features).to(device, dtype=data_type)
        target_logits = model(all_feats)
        target_probs = torch.softmax(target_logits, dim=1).detach().cpu().numpy()

    target_preds = np.argmax(target_probs, axis=1)
    if target_probs.shape[1] == 2:
        target_probs = target_probs[:, 1]

    return target_preds, target_probs


def test_lp_MIL(features, model, device):
    """
    Perform batch-level prediction using a linear probe (MIL-style evaluation).
    This function aggregates per-sample probabilities into a single decision for the batch.

    Args:
        features: Input features for a batch of samples.
        model: Trained LinearProbe model.
        device: "cpu" or "cuda".

    Returns:
        Aggregated batch-level prediction and probability.
    """
    data_type = next(model.parameters()).dtype
    model.eval()
    with torch.no_grad():
        all_feats = torch.tensor(features).to(device, dtype=data_type)
        target_logits = model(all_feats)
        target_probs = torch.softmax(target_logits, dim=1).detach().cpu().numpy()

    # Aggregate probabilities across the batch
    if target_probs.shape[1] == 2:
        # Use the median probability of patches to represent the whole image
        target_prob = np.nanmedian(target_probs[:, 1])
        target_pred = 0 if target_prob < 0.5 else 1
    else:
        # Multi-class
        target_prob_per_class = np.nanmedian(target_probs, axis=0)
        target_pred = int(np.argmax(target_prob_per_class))
        target_prob = float(target_prob_per_class[target_pred])

    return target_pred, target_prob


def train_fusion(features, 
                 aux_labels, 
                 targets,
                 device,  
                 adam_epochs=10,
                 adam_lr=0.01,
                 adam_wd=0.0,
                 lbfgs_epochs=1,
                 lbfgs_lr=0.1,
                 log_epoch=1,
                 use_bn=False,
                 temperature=False):
    """
    Train a fusion model that jointly predicts auxiliary labels and target labels from the same input features.

    Args:
        features: Input features.
        aux_labels: Auxiliary task labels.
        targets: Target labels.
        device: "cpu" or "cuda".
        adam_epochs: Number of Adam training epochs.
        adam_lr: Adam learning rate.
        adam_wd: Adam weight decay.
        lbfgs_epochs: Number of L-BFGS epochs.
        lbfgs_lr: Learning rate for L-BFGS.
        log_epoch: Logging frequency.
        use_bn: Whether to include batch normalization.
        temperature: Whether to use temperature scaling.

    Returns:
        Trained FusionModel.
    """
    data_type = torch.float32
    input_dim = features.shape[1]
    aux_classes = len(np.unique(aux_labels))
    target_classes = len(np.unique(targets))

    model = FusionModel(
        input_dim, aux_classes, target_classes,
        use_bn=use_bn, temperature=temperature
    ).to(device, dtype=data_type)

    aux_criterion = torch.nn.CrossEntropyLoss()
    target_criterion = torch.nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(model.parameters(), lr=adam_lr, weight_decay=adam_wd)
    # optimizer1 = torch.optim.AdamW(model.parameters(), lr=adam_lr, weight_decay=adam_wd)
    optimizer2 = torch.optim.LBFGS(
        model.parameters(),
        max_iter=1000,
        lr=lbfgs_lr,
        line_search_fn="strong_wolfe"
    )

    all_feats = torch.tensor(features).to(device, dtype=data_type)
    all_aux_labels = torch.tensor(aux_labels).to(device, dtype=torch.long)
    all_targets = torch.tensor(targets).to(device, dtype=torch.long)

    def compute_loss(aux_pred, aux_true, target_pred, target_true):
        """
        Compute joint loss consisting of auxiliary loss, target loss, 
        and scaled L2-based regularization.
        """
        aux_loss = aux_criterion(aux_pred, aux_true)
        target_loss = target_criterion(target_pred, target_true)
        base_loss = aux_loss + target_loss

        l2_reg = 0.0
        for layer in model.modules():
            if isinstance(layer, torch.nn.Linear):
                d1, d2 = layer.weight.shape
                c = (d1 * d2) / 100
                l2_reg += (0.5 / c) * layer.weight.norm(p=2)
        loss = base_loss.mean() + l2_reg

        return aux_loss, target_loss, loss

    adam_epochs = int(adam_epochs)
    lbfgs_epochs = int(lbfgs_epochs)
    log_epoch = int(log_epoch)

    # Adam phase
    model.train()
    for epoch in range(1, adam_epochs + 1):
        optimizer1.zero_grad()
        aux_output, target_output = model(all_feats)
        aux_loss, target_loss, loss = compute_loss(aux_output, all_aux_labels, target_output, all_targets)
        loss.backward()
        optimizer1.step()

        if epoch % log_epoch == 0:
            print(f"[Adam] Train Epoch: {epoch} \t Aux Loss: {aux_loss.item():.6f}, Target Loss: {target_loss.item():.6f}, Total Loss: {loss.item():.6f}")

    # L-BFGS phase
    model.train()
    for epoch in range(adam_epochs + 1, adam_epochs + lbfgs_epochs + 1):
        def closure():
            optimizer2.zero_grad()
            aux_output, target_output = model(all_feats)
            aux_loss, target_loss, loss = compute_loss(aux_output, all_aux_labels, target_output, all_targets)
            loss.backward()
            return loss

        optimizer2.step(closure)

        with torch.no_grad():
            aux_output, target_output = model(all_feats)
            aux_loss, target_loss, loss = compute_loss(aux_output, all_aux_labels, target_output, all_targets)

        print(f"[L-BFGS] Train Epoch: {epoch} \t Aux Loss: {aux_loss.item():.6f}, Target Loss: {target_loss.item():.6f}, Total Loss: {loss.item():.6f}")

    return model


def test_fusion(features, model, device):
    """
    Test a trained fusion model and return predictions and probabilities for both auxiliary and target tasks.
    It is suitable for standard classification evaluation where each sample has its own label.

    Args:
        features: Input features.
        model: Trained FusionModel.
        device: "cpu" or "cuda".

    Returns:
        target_preds, target_probs, aux_preds and aux_probs for all samples.
    """
    data_type = next(model.parameters()).dtype
    model.eval()
    with torch.no_grad():
        all_feats = torch.tensor(features).to(device, dtype=data_type)
        aux_output, target_output = model(all_feats)
        aux_probs = torch.softmax(aux_output, dim=1).detach().cpu().numpy()
        target_probs = torch.softmax(target_output, dim=1).detach().cpu().numpy()

    aux_preds = np.argmax(aux_probs, axis=1)
    target_preds = np.argmax(target_probs, axis=1)

    if target_probs.shape[1] == 2:
        target_probs = target_probs[:, 1]

    return target_preds, target_probs, aux_preds, aux_probs


def test_fusion_MIL(features, model, device):
    """
    Perform batch-level prediction using a fusion model (MIL-style evaluation).
    This function aggregates per-sample probabilities into a single decision for the batch for the target task, 
    while returning sample-level probabilities for the auxiliary task.

    Args:
        features: Input features for a batch of samples.
        model: Trained FusionModel.
        device: "cpu" or "cuda".

    Returns:
        Aggregated target prediction, target probability, sample-level target probabilities, sample-level auxiliary probabilities.
    """
    data_type = next(model.parameters()).dtype
    model.eval()
    with torch.no_grad():
        all_feats = torch.tensor(features).to(device, dtype=data_type)
        aux_output, target_output = model(all_feats)
        aux_probs = torch.softmax(aux_output, dim=1).detach().cpu().numpy()
        target_probs = torch.softmax(target_output, dim=1).detach().cpu().numpy()

    # Aggregate probabilities across the batch
    if target_probs.shape[1] == 2:
        # Use the median probability of patches to represent the whole image
        target_prob = np.nanmedian(target_probs[:, 1])
        target_pred = 0 if target_prob < 0.5 else 1
    else:
        # Multi-class
        target_prob_per_class = np.nanmedian(target_probs, axis=0)
        target_pred = int(np.argmax(target_prob_per_class))
        target_prob = float(target_prob_per_class[target_pred])

    return target_pred, target_prob, target_probs, aux_probs


