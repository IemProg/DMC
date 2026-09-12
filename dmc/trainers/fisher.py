"""
Fisher Information utilities for FisherMergeTune (D1 + D2).

Computes the diagonal empirical Fisher in TEXT FEATURE SPACE,
consistent with how the existing MERGETUNE LMC operates.

The Fisher is computed at the zero-shot CLIP text features (w1)
using the downstream dataset (D2). This is a one-time computation
performed before continued training begins.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional

from clip import clip


CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "a photo of a {}, a type of texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


def get_zero_shot_text_features(clip_model, classnames, dataset_name, device):
    """
    Compute zero-shot CLIP text features for the given classnames.

    Args:
        clip_model: CLIP model (already on device)
        classnames: list of class name strings
        dataset_name: dataset name string (for template selection)
        device: torch device

    Returns:
        text_features: Tensor of shape [C, feat_dim], L2-normalized
    """
    temp = CUSTOM_TEMPLATES[dataset_name]
    prompts = [temp.format(c.replace("_", " ")) for c in classnames]
    prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features  # [C, 512]


def compute_feature_fisher(
    clip_model,
    dataloader,
    classnames,
    dataset_name,
    device,
    n_samples=1024,
    eps=1e-8,
):
    """
    Compute diagonal empirical Fisher over text feature space at ŵ₁.

    The Fisher measures how sensitive the zero-shot model's predictions
    are to perturbations in each dimension of the text feature matrix.

    F̃₁[c, d] = E_{(x,y)~D₂} [ (∂ loss / ∂ T₁[c, d])² ]

    where T₁ ∈ R^{C x feat_dim} are the zero-shot text features.

    Args:
        clip_model: CLIP model (on device, in eval mode)
        dataloader: downstream training data D₂
        classnames: list of class name strings
        dataset_name: dataset name (for template selection)
        device: torch device
        n_samples: max samples for Fisher estimation
        eps: numerical stability constant

    Returns:
        fisher_diag: Tensor of shape [C, feat_dim]
    """
    clip_model.eval()

    # Get zero-shot text features as a detached tensor
    text_features_base = get_zero_shot_text_features(
        clip_model, classnames, dataset_name, device
    )  # [C, feat_dim]
    feat_dim = text_features_base.shape[1]
    n_cls = text_features_base.shape[0]

    # Accumulator for squared gradients
    fisher_diag = torch.zeros(n_cls, feat_dim, device=device)
    logit_scale = clip_model.logit_scale.exp().detach()

    n_seen = 0
    for batch in dataloader:
        if n_seen >= n_samples:
            break

        images = batch["img"].to(device)
        labels = batch["label"].to(device)
        batch_size = images.size(0)

        # Create a leaf tensor copy of text features that requires grad
        text_features = text_features_base.clone().detach().requires_grad_(True)

        # Forward: compute logits using the zero-shot text features
        with torch.no_grad():
            image_features = clip_model.encode_image(images.type(clip_model.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.float()

        # Compute logits with grad enabled for text_features
        logits = logit_scale.float() * image_features @ text_features.float().t()
        loss = F.cross_entropy(logits, labels)

        # Backward to get gradient w.r.t. text_features
        loss.backward()

        with torch.no_grad():
            fisher_diag += (text_features.grad.detach() ** 2) * batch_size

        n_seen += batch_size

    # Normalize by number of samples
    fisher_diag = fisher_diag / max(n_seen, 1)

    # Clamp for numerical stability
    fisher_diag = torch.clamp(fisher_diag, min=eps)

    return fisher_diag


def normalize_fisher(fisher, mode="max"):
    """
    Normalize Fisher diagonal for numerical stability.

    Args:
        fisher: Tensor of shape [C, feat_dim]
        mode:
            'max'       - divide by global max (default)
            'mean'      - divide by global mean
            'per_class' - normalize each class row by its own max

    Returns:
        Normalized Fisher tensor (same shape)
    """
    if mode == "max":
        global_max = fisher.max().item()
        return fisher / (global_max + 1e-8)
    elif mode == "mean":
        global_mean = fisher.mean().item()
        return fisher / (global_mean + 1e-8)
    elif mode == "per_class":
        row_max = fisher.max(dim=1, keepdim=True).values
        return fisher / (row_max + 1e-8)
    return fisher


def fisher_weighted_distance(feat_w, feat_w1, fisher):
    """
    Fisher-weighted L2 distance in feature space (D1 endpoint term).

    d_F(w, ŵ₁) = Σ_{c,d} F̃₁[c,d] * (feat_w[c,d] - feat_w1[c,d])²

    This bounds KL(p(·;ŵ₁) || p(·;w)) up to third-order terms.

    Args:
        feat_w: current model text features [C, feat_dim]
        feat_w1: zero-shot CLIP text features [C, feat_dim]
        fisher: diagonal Fisher [C, feat_dim]

    Returns:
        Scalar tensor (the Fisher-weighted distance)
    """
    diff = feat_w - feat_w1
    return (fisher * diff.pow(2)).sum()


def fisher_weighted_path_loss(feat_w, feat_w1, fisher, alpha):
    """
    Task-1 path surrogate at interpolation point alpha (D2 term).

    R₁(ŵ₁ + α(w - ŵ₁)) = α² * (w - ŵ₁)ᵀ F̃₁ (w - ŵ₁)

    Summing over alpha ~ U[0,1] gives the symmetric path constraint.

    Args:
        feat_w: current model text features [C, feat_dim]
        feat_w1: zero-shot CLIP text features [C, feat_dim]
        fisher: diagonal Fisher [C, feat_dim]
        alpha: interpolation coefficient in (0, 1)

    Returns:
        Scalar tensor
    """
    return (alpha ** 2) * fisher_weighted_distance(feat_w, feat_w1, fisher)


def compute_prompt_fisher(
    clip_model,
    text_encoder,
    prompt_learner,
    dataloader,
    device,
    n_samples=1024,
    eps=1e-8,
):
    """
    Compute diagonal Fisher over prompt parameters ctx (Proposal A).

    F_prompt[i, j] = E_{(x,y)~D} [ (d CE / d ctx[i, j])^2 ]

    Computed at the current prompt_learner.ctx (typically the CoOp
    checkpoint = w2). Measures which prompt dimensions the downstream
    task is sensitive to.

    Args:
        clip_model: CLIP model (on device, provides image encoder + logit_scale)
        text_encoder: TextEncoder module
        prompt_learner: PromptLearner with loaded ctx (buffer)
        dataloader: downstream training data
        device: torch device
        n_samples: max samples for estimation
        eps: numerical stability

    Returns:
        fisher_diag: Tensor [N_ctx, ctx_dim] (e.g. [4, 512])
    """
    ctx_base = prompt_learner.ctx.clone().detach()  # [N_ctx, dim] or [n_cls, N_ctx, dim]
    if ctx_base.dim() == 2:
        fisher_shape = ctx_base.shape  # [N_ctx, dim]
    else:
        fisher_shape = ctx_base.shape[1:]  # [N_ctx, dim]

    fisher_diag = torch.zeros(fisher_shape, device=device)
    logit_scale = clip_model.logit_scale.exp().detach()
    dtype = clip_model.dtype

    n_seen = 0
    for batch in dataloader:
        if n_seen >= n_samples:
            break

        images = batch["img"].to(device)
        labels = batch["label"].to(device)
        batch_size = images.size(0)

        # Leaf copy of ctx requiring grad
        ctx_leaf = ctx_base.clone().detach().requires_grad_(True)

        # Build prompts (replicate prompt_learner.forward with ctx_leaf)
        if ctx_leaf.dim() == 2:
            ctx_expanded = ctx_leaf.unsqueeze(0).expand(prompt_learner.n_cls, -1, -1)
        else:
            ctx_expanded = ctx_leaf

        prompts = torch.cat(
            [prompt_learner.token_prefix, ctx_expanded, prompt_learner.token_suffix],
            dim=1,
        )

        # Text features through encoder
        text_features = text_encoder(prompts.type(dtype), prompt_learner.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Image features (no grad)
        with torch.no_grad():
            image_features = clip_model.visual(images.type(dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.float()

        logits = logit_scale.float() * image_features @ text_features.float().t()
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        with torch.no_grad():
            grad = ctx_leaf.grad.detach()
            if grad.dim() == 3:  # [n_cls, N_ctx, dim] -> average over classes
                grad = grad.mean(dim=0)
            fisher_diag += (grad ** 2) * batch_size

        n_seen += batch_size

    fisher_diag = fisher_diag / max(n_seen, 1)
    fisher_diag = torch.clamp(fisher_diag, min=eps)
    return fisher_diag


def save_fisher(fisher, path):
    """Save Fisher tensor to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(fisher, path)
    print(f"Fisher saved to {path}")


def load_fisher(path, device="cuda"):
    """Load Fisher tensor from disk."""
    fisher = torch.load(path, map_location=device)
    print(f"Fisher loaded from {path} (shape: {fisher.shape})")
    return fisher


def get_fisher_cache_path(output_dir, dataset_name, seed, backbone_name):
    """Get the standard cache path for a Fisher tensor."""
    cache_dir = os.path.join(output_dir, "fisher_cache")
    filename = f"{dataset_name}_{backbone_name.replace('/', '_')}_seed{seed}.pt"
    return os.path.join(cache_dir, filename)


def get_prompt_fisher_cache_path(output_dir, dataset_name, seed, backbone_name):
    """Get the cache path for a prompt-space Fisher tensor."""
    cache_dir = os.path.join(output_dir, "fisher_cache")
    filename = f"{dataset_name}_{backbone_name.replace('/', '_')}_seed{seed}_prompt.pt"
    return os.path.join(cache_dir, filename)
