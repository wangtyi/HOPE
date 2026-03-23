import torchvision.transforms as transforms
import torch
import timm
import os
import huggingface_hub
from transformers import AutoImageProcessor, AutoModel
from transformers import CLIPProcessor, CLIPModel
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD


def load_histo_model(model_name, device, ckpts_dir):
    """
    Load a pretrained histology foundation model and its corresponding image transform.

    Supported models include UNI2-h, MUSK, Phikon-v2, and PLIP. Models are loaded
    from local checkpoint directories or downloaded from HuggingFace Hub as needed.

    Args:
        model_name: Name of the pretrained model ('uni', 'musk', 'phikon-v2', or 'plip').
        device: Device to load the model on ('cpu' or 'cuda').
        ckpts_dir: Root directory containing model checkpoints.

    Returns:
        A tuple of (model, transform).

    Raises:
        ValueError: If the specified model_name is not supported.
    """
    if model_name == 'uni':
        uni_kwargs = {
            'model_name': 'vit_giant_patch14_224',
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
        ckpt_path = os.path.join(ckpts_dir, 'uni2-h/pytorch_model.bin')
        model = timm.create_model(**uni_kwargs)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    elif model_name == 'musk':
        from musk import utils, modeling

        model_config = "musk_large_patch16_384"
        model = timm.create_model(model_config).eval()
        utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model, 'model|module', '', 
                                            local_dir=f"{ckpts_dir}/musk")
        model.to(device)
        model.eval()

        image_size = 384
        transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=3, antialias=True),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD) # mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
        ])

    elif 'phikon' in model_name:        
        model = AutoModel.from_pretrained(f"owkin/{model_name}", 
                                cache_dir=f'{ckpts_dir}/{model_name}')
        model.to(device)
        model.eval()

        image_processor = AutoImageProcessor.from_pretrained(f"owkin/{model_name}", 
                                cache_dir=f'{ckpts_dir}/{model_name}')
        image_size = image_processor.crop_size if "v2" in model_name else image_processor.size
        image_size = image_size["height"]  # 224
        transform = transforms.Compose([
                transforms.Resize(image_size, interpolation=3, antialias=True),
                transforms.CenterCrop((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)  # [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ])

    elif model_name == 'plip':
        model = CLIPModel.from_pretrained("vinid/plip", cache_dir=f'{ckpts_dir}/plip')
        model.to(device)
        model.eval()

        processor = CLIPProcessor.from_pretrained("vinid/plip", cache_dir=f'{ckpts_dir}/plip')
        image_size = processor.image_processor.crop_size
        image_size = image_size["height"]  # 224
        transform = transforms.Compose([
                transforms.Resize(image_size, interpolation=3, antialias=True),
                transforms.CenterCrop((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=processor.image_processor.image_mean, std=processor.image_processor.image_std)  # mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        ])

    else:
        raise ValueError(f"Unsupported model.")
    
    return model, transform
