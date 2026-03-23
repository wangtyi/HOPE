import numpy as np
import torch
from tqdm import tqdm


@torch.no_grad()
def extract_features(model, model_name, dataloader, device, show_progress=True):
    """
    Extract feature embeddings and labels from batches of images using a pretrained model.

    Args:
        model: The pretrained model used to extract feature embeddings.
        model_name: Id string specifying the pretrained model.
        dataloader: Dataloader providing samples.
        device: "cpu" or "cuda".
        show_progress: Whether to display a progress bar during feature extraction.

    Returns:
        A dictionary containing extracted embeddings and available labels (and regions if provided).
    """
    all_embeddings = []
    all_labels = []
    all_aux_labels = []
    all_regions = []

    data_type = next(model.parameters()).dtype

    for batch in tqdm(dataloader, total=len(dataloader), disable=not show_progress, desc="extracting features"):
        img = batch["image"].to(device, dtype=data_type)

        # Feature extraction
        if model_name == "uni":
            emb = model(img).detach().cpu().numpy()
        elif model_name == "musk":
            emb = model(
                image=img,
                with_head=False,
                out_norm=True,
                ms_aug=True
            )[0].detach().cpu().numpy()
        elif "phikon" in model_name:
            emb = model(img).last_hidden_state[:, 0, :].detach().cpu().numpy()
        elif model_name == "plip":
            emb = model.get_image_features(img).detach().cpu().numpy()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        all_embeddings.append(emb)
        all_labels.append(batch["label"].numpy())

        if "aux_label" in batch:
            all_aux_labels.append(batch["aux_label"].numpy())

        if "region" in batch:
            r = batch["region"]
            if isinstance(r, list):  # multiple tensors
                r = torch.stack(r, dim=1)
            all_regions.append(r.numpy())

    # Output
    asset_dict = {"embeddings": np.vstack(all_embeddings).astype(np.float32),
                  "labels": np.concatenate(all_labels)}

    if len(all_aux_labels) != 0:
        asset_dict["aux_labels"] = np.concatenate(all_aux_labels)

    if len(all_regions) != 0:
        asset_dict["regions"] = np.concatenate(all_regions)

    return asset_dict

