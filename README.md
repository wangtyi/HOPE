# HOPE

This repository contains the implementation of HOPE (Histology image analysis with Omnidirectional Patch Embeddings).

## Step 1. Create environment and install dependencies

```bash
# Create and activate conda environment
conda create -n hope python=3.10 -y
conda activate hope

# Install PyTorch
pip install "torch==2.6.0" "torchvision==0.21.0" --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install -r requirements.txt
```

**Install space-gm** (for CODEX analysis):
```bash
# Requires PyTorch Geometric
pip install torch-geometric==2.6.1
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

git clone https://gitlab.com/enable-medicine-public/space-gm.git 
pip install -e ./space-gm
```

## Step 2. Download H&E foundation models

To use `utils/load_model.py`, you need to manually download the following pretrained histology foundation models and place them in `./assets/ckpts/`:

| Model | Model ID (`pretrained_model_name`) | Download Source | Cache Path |
|-------|-----------------------------------|-----------------|------------|
| UNI2-h | `uni` | https://huggingface.co/MahmoodLab/UNI2-h | `./assets/ckpts/uni2-h/pytorch_model.bin` |
| MUSK | `musk` | https://github.com/lilab-stanford/MUSK | `./assets/ckpts/musk/` (auto-downloaded on first use) |
| Phikon-v2 | `phikon-v2` | https://huggingface.co/owkin/phikon-v2 | `./assets/ckpts/phikon-v2/` (auto-downloaded on first use) |
| PLIP | `plip` | https://github.com/PathologyFoundation/plip | `./assets/ckpts/plip/` (auto-downloaded on first use) |

**Note:** Install MUSK (requires installation before import):
```bash
git clone https://github.com/lilab-stanford/MUSK.git 
pip install -e ./MUSK
```

Example usage:
```python
from utils.load_model import load_histo_model

pre_model, transform = load_histo_model(
    pretrained_model_name='uni',  # or 'musk', 'phikon-v2', 'plip'
    device='cuda',
    ckpts_dir='./assets/ckpts'
)
```

## Step 3. Prepare datasets

The following datasets are used in this project:

| Dataset | Download Link | Files Used |
|---------|---------------|------------|
| **UPMC-HNC MIF data** | https://zenodo.org/records/13179600 | `upmc_raw_data.zip`, `upmc_labels.csv` |
| **UPMC-HNC H&E images** | https://zenodo.org/records/19163305 | H&E images, `region_id_mapping.csv` |
| **HCC MIF & H&E images** | https://zenodo.org/records/15392699 | MIF data + H&E images |


## Step 4. Usage

Refer to the Jupyter notebooks for example usage:
- `1_extract_spatial_omics_signatures.ipynb`: Derive disease-related spatial omics signatures using SPACE-GM
- `2_example_HNC_baseline.ipynb`: UPMC-HNC H&E direct extrapolation baseline
- `3_example_HNC_fusionModel.ipynb`: UPMC-HNC H&E fusion with spatial omics signatures
- `4_example_HCC_fusionModel.ipynb`: HCC internal validation with fusion model


## Project Structure

```
HOPE/
├── src/                          
│   ├── dataset.py               # Dataset classes for image loading
│   ├── histology_feature_extractor.py  # H&E feature extraction
│   ├── model.py                 # LinearProbe, FusionModel
│   └── train_test.py            # Training and testing functions
├── utils/                       
│   ├── load_model.py            # H&E foundation model loading utilities
│   ├── metrics.py               # Evaluation metrics
│   └── set_seed.py              # Random seed setting
├── assets/                       
│   ├── ckpts/                   # Pretrained model weights
│   │   ├── uni2-h/              # UNI2-h model
│   │   │   └── pytorch_model.bin
│   │   ├── musk/                # MUSK model (auto-downloaded)
│   │   ├── phikon-v2/           # Phikon-v2 model (auto-downloaded)
│   │   └── plip/                # PLIP model (auto-downloaded)
│   └── spacegm_upmc/            # Trained SPACE-GM models
│       └── model/               # Trained GNN model for spatial omics signature extraction
├── data/                         
│   ├── microE_annotations/      # Spatial omics derived signatures
│   └── upmc_split_indices.json  # Cross-validation split indices for UPMC-HNC
└── *.ipynb                       # Jupyter notebooks for experiments
```


