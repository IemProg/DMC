"""DMC and MergeTune on top of CoOp / KgCoOp prompt learning.

This module registers the ``KgCoOp_COOP_LMC`` trainer, which implements both
the single-prompt MergeTune baseline and the decoupled DMC method of the
paper. Which one runs is decided by ``TRAINER.COOP.DPP``.

Notation (paper -> code)
------------------------
    f_w1        zero-shot CLIP text features from hand-crafted templates
    w2_hat      the stage-1 checkpoint (CoOp or KgCoOp), loaded via --resume-coop
    c           the single context of MergeTune          -> prompt_mid_learner
    c_spec      DMC specialisation prompt                -> prompt_mid_learner
    c_gen       DMC generalisation prompt                -> prompt_gen_learner
    R           cosine regulariser, Eq. (2)              -> cosine score term
    L_VA        Visual Anchor, Eq. (7)                   -> VA_W / VA_TAU
    beta        LMC path weight                          -> TRAINER.COOP.W_LMC
    lambda      MergeTune cosine weight                  -> TRAINER.COOP.W
    lambda_gen  DMC cosine weight on c_gen               -> TRAINER.COOP.DPP_W_GEN
    alpha       interpolation coefficient                -> sampled per step

Modes
-----
``DPP = False`` (MergeTune, Eq. 3)
    A single context is trained with
    ``L_CE(c) + W * R(f_c, f_w1) + W_LMC * E_alpha[L_CE(interp(f_w2, f_c))]``.

``DPP = True`` (DMC, Eq. 8)
    Two contexts are trained jointly, both initialised from ``w2_hat``:

        L = L_CE(f_spec) + VA_W * L_VA          (c_spec only)
          + DPP_W_GEN * R(f_gen, f_w1)          (c_gen only)
          + W_LMC * E_alpha[ L_CE(f_tilde(a)) ] (couples both)

    where ``f_tilde(a) = l2(f_gen + a * (f_spec - f_gen))``. The LMC term is
    the only coupling between the two prompts. Interpolation happens in
    text-feature space, not in prompt-parameter space, because the text
    encoder is non-linear and classification scores are computed as v . f_c.

Inference
---------
At test time a single fixed ``alpha`` (0.20 in the paper) selects one point on
the corridor; ``f_tilde(alpha)`` is computed once per dataset, so per-image
cost matches a single-prompt method. ``--eval-only-dpp`` sweeps alpha over
[0, 1] and reports accuracy at each point.

Also implemented here, disabled by default, are the single-prompt loss
modifications reported in the appendix (ZSDD, KL path, prompt-space EWC,
W-scheduling, test-time adaptive interpolation).
"""

import os
import os.path as osp
import json
import time
import itertools
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from collections import OrderedDict
import datetime


from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import numpy as np
from torchvision import transforms as T

from .fisher import (
    compute_feature_fisher,
    compute_prompt_fisher,
    normalize_fisher,
    save_fisher,
    load_fisher,
    get_fisher_cache_path,
    get_prompt_fisher_cache_path,
)

# For Optuna (optional)
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

_tokenizer = _Tokenizer()


def slerp(w_start, w_end, t, eps=1e-4):
    """Spherical linear interpolation between two vectors.

    Args:
        w_start: Starting vector (any shape, will be flattened for angle computation).
        w_end: Ending vector (same shape as w_start).
        t: Interpolation parameter in [0, 1].
        eps: Fallback to linear interpolation when angle < eps.

    Returns:
        Interpolated vector of same shape as inputs.
    """
    flat_start = w_start.reshape(-1)
    flat_end = w_end.reshape(-1)
    cos_theta = F.cosine_similarity(flat_start.unsqueeze(0), flat_end.unsqueeze(0)).clamp(-1, 1)
    theta = torch.acos(cos_theta)

    if theta.item() < eps:
        # Degenerate case: vectors nearly identical, fall back to linear
        return w_start + t * (w_end - w_start)

    sin_theta = torch.sin(theta)
    coeff_start = torch.sin((1 - t) * theta) / sin_theta
    coeff_end = torch.sin(t * theta) / sin_theta
    return coeff_start * w_start + coeff_end * w_end


def slerp_per_token(w_start, w_end, t, eps=1e-4):
    """Apply Slerp independently to each token (row) of the prompt matrix.

    Args:
        w_start: [N, D] prompt matrix.
        w_end: [N, D] prompt matrix.
        t: Interpolation parameter in [0, 1].
        eps: Fallback threshold.

    Returns:
        [N, D] interpolated prompt matrix.
    """
    result = torch.zeros_like(w_start)
    for i in range(w_start.shape[0]):
        result[i] = slerp(w_start[i], w_end[i], t, eps)
    return result


def _build_prompts_from_ctx(ctx, prompt_mid_learner, class_token_position="end"):
    """Build full prompt embeddings from interpolated ctx vectors.

    Handles class_token_position == "end" only. Asserts on other positions
    to prevent silent incorrect behavior.
    """
    assert class_token_position == "end", (
        f"SMC-Tune/WSPE prompt construction only supports class_token_position='end', "
        f"got '{class_token_position}'. For 'middle'/'front', the ctx-class token "
        f"ordering differs and needs explicit handling."
    )
    if ctx.dim() == 2:
        ctx = ctx.unsqueeze(0).expand(prompt_mid_learner.n_cls, -1, -1)
    return torch.cat([
        prompt_mid_learner.token_prefix,
        ctx,
        prompt_mid_learner.token_suffix,
    ], dim=1)


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

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




class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            temp = 'a photo of a'
            ctx_init = temp.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)


        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        # fixed not optimized
        self.register_buffer("ctx", ctx_vectors)


        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        clip_model_ = load_clip_to_cpu(cfg)
        clip_model_.cuda()

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts_}")
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_])
        prompts_ = prompts_.cuda()

        with torch.no_grad():
            text_features = clip_model_.encode_text(prompts_)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features

        # Free temp CLIP model immediately
        del clip_model_
        torch.cuda.empty_cache()

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS


        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1) # torch.Size([100, 4, 512])
        
        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts


class PromptMidLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP_CLIP.N_CTX
        ctx_init = cfg.TRAINER.COOP_CLIP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            temp = 'a photo of a'
            ctx_init = temp.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)


        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        # Add bias_vectors parameter (missing from original implementation)
        bias_vectors = torch.empty(1, 512, dtype=dtype)
        nn.init.normal_(bias_vectors, std=0.02)
        self.bias_vectors = nn.Parameter(bias_vectors)


        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        #print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model_ = load_clip_to_cpu(cfg)
        clip_model_.cuda()

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts_}")
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_])
        prompts_ = prompts_.cuda()

        with torch.no_grad():
            text_features = clip_model_.encode_text(prompts_)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features

        # Free temp CLIP model immediately to avoid GPU OOM when multiple learners are created
        del clip_model_
        torch.cuda.empty_cache()

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1) # torch.Size([100, 4, 512])
        
        prefix = self.token_prefix
        suffix = self.token_suffix


        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            # ... same middle logic as PromptLearner ...
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        ctx_i_half1,
                        class_i,
                        ctx_i_half2,
                        suffix_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            # ... same front logic as PromptLearner ...
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        class_i,
                        ctx_i,
                        suffix_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        
        else:
            raise ValueError


        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, fisher_diag=None):
        super().__init__()
        self.cfg = cfg
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model) # fixed coop model
        self.prompt_mid_learner = PromptMidLearner(cfg, classnames, clip_model) # mid coop model, which is learned
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.ori_embedding = self.prompt_learner.text_features
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # DPP: second learnable prompt (ctx_gen) for generalization
        if cfg.TRAINER.COOP.DPP:
            self.prompt_gen_learner = PromptMidLearner(cfg, classnames, clip_model)
            torch.cuda.empty_cache()  # Free temp CLIP models from PromptMidLearner init

        # Fisher weights for fisher_cosine loss type
        if fisher_diag is not None:
            # Per-class importance: average Fisher across feature dims -> [C]
            fisher_weights = fisher_diag.mean(dim=1)
            fisher_weights = fisher_weights / (fisher_weights.sum() + 1e-8)
            self.register_buffer("fisher_weights", fisher_weights)
        else:
            self.fisher_weights = None

    def forward(self, image, sign=None, text_features_mid=None):
        if sign is None:
            prompts = self.prompt_mid_learner()
            image_features = self.image_encoder(image.type(self.dtype))

            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(prompts, tokenized_prompts)
            text_features_old = self.ori_embedding

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()

            logits = logit_scale * image_features @ text_features.t()
            text_features_old = text_features_old / text_features_old.norm(dim=-1, keepdim=True)

            loss_type = self.cfg.TRAINER.COOP.LOSS_TYPE

            if loss_type == "cosine":
                cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
                score = cos(text_features, text_features_old)
                score = 1.0 - torch.mean(score)

            elif loss_type == "fisher_cosine":
                # Per-class cosine dissimilarity weighted by Fisher importance
                cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
                cos_per_class = 1.0 - cos(text_features, text_features_old)  # [C]
                score = (self.fisher_weights * cos_per_class).sum()

            elif loss_type == "l2":
                l2_dist = torch.norm(text_features - text_features_old, p=2, dim=1)
                score = torch.mean(l2_dist)

            else:
                raise ValueError(f"Unknown LOSS_TYPE: {loss_type}")

            return logits, score

        else:
            image_features = self.image_encoder(image.type(self.dtype))
            text_features = text_features_mid

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()

            logits = logit_scale * image_features @ text_features.t()

            return logits
        

@TRAINER_REGISTRY.register()
class KgCoOp_COOP_LMC(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        loss_type = cfg.TRAINER.COOP.LOSS_TYPE

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        # Compute Fisher if loss_type requires it
        fisher_diag = None
        if loss_type == "fisher_cosine":
            fisher_diag = self._get_fisher(cfg, clip_model, classnames)

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, fisher_diag=fisher_diag)
        self.w = cfg.TRAINER.COOP.W


        if self.cfg.RESUME_COOP and self.cfg.RESUME_COOP != 'None':
            print(f"Loading pretrained CoOp prompt_learner from {self.cfg.RESUME_COOP}")
            checkpoint_path_1 = osp.join(self.cfg.RESUME_COOP, "prompt_learner/model.pth.tar-100")
            checkpoint_path_2 = osp.join(self.cfg.RESUME_COOP, "prompt_learner/model-100.pth.tar")
            
            if osp.exists(checkpoint_path_1):
                checkpoint_path = checkpoint_path_1
            elif osp.exists(checkpoint_path_2):
                checkpoint_path = checkpoint_path_2
            else:
                raise FileNotFoundError(f"Neither {checkpoint_path_1} nor {checkpoint_path_2} exists")
            
            checkpoint = load_checkpoint(checkpoint_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            
            # Remove token information like in CoOp
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]
            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            self.model.prompt_learner.load_state_dict(state_dict, strict=False)


            # CRITICAL FIX: Initialize prompt_mid_learner with the same weights as prompt_learner
            # This ensures that testing before training gives correct results
            print("Initializing prompt_mid_learner with loaded prompt_learner weights")
            self.model.prompt_mid_learner.ctx.data.copy_(self.model.prompt_learner.ctx.data)

            
        # --- DPP: initialize prompt_gen_learner from w2 ---
        self.dpp = cfg.TRAINER.COOP.DPP
        self.dpp_gen_mode = cfg.TRAINER.COOP.DPP_GEN_MODE if self.dpp else "learned"

        if self.dpp and self.dpp_gen_mode == "learned":
            if self.cfg.RESUME_COOP and self.cfg.RESUME_COOP != 'None':
                print("DPP: Initializing prompt_gen_learner with loaded prompt_learner weights")
                self.model.prompt_gen_learner.ctx.data.copy_(self.model.prompt_learner.ctx.data)

        if self.dpp and self.dpp_gen_mode == "fixed_w1":
            print("DPP (fixed_w1 ablation): no learnable c_gen, gen endpoint = w1")

        print("Turning off gradients in both the image and the text encoder, and the prompt_learner")
        learnable_names = {"prompt_mid_learner.ctx"}
        if self.dpp and self.dpp_gen_mode == "learned":
            learnable_names.add("prompt_gen_learner.ctx")
        for name, param in self.model.named_parameters():
            if name not in learnable_names:
                param.requires_grad_(False)
            else:
                print(name)

        self.model.to(self.device)

        if self.dpp and self.dpp_gen_mode == "learned":
            # DPP (learned): optimizer for both ctx_base and ctx_gen
            combined_params = [
                {"params": self.model.prompt_mid_learner.parameters()},
                {"params": self.model.prompt_gen_learner.parameters()},
            ]
            self.optim = torch.optim.SGD(
                combined_params, lr=cfg.OPTIM.LR,
                momentum=0.9, weight_decay=5e-4
            )
            self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
            self.register_model("prompt_mid_learner", self.model.prompt_mid_learner, self.optim, self.sched)
            self.register_model("prompt_gen_learner", self.model.prompt_gen_learner, self.optim, self.sched)
            self.dpp_w_gen = cfg.TRAINER.COOP.DPP_W_GEN
            print(f"DPP: ctx_base=prompt_mid_learner, ctx_gen=prompt_gen_learner, W_gen={self.dpp_w_gen}")
        elif self.dpp and self.dpp_gen_mode == "fixed_w1":
            # DPP (fixed_w1): optimizer for ctx_base only
            self.optim = torch.optim.SGD(
                self.model.prompt_mid_learner.parameters(), lr=cfg.OPTIM.LR,
                momentum=0.9, weight_decay=5e-4
            )
            self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
            self.register_model("prompt_mid_learner", self.model.prompt_mid_learner, self.optim, self.sched)
            self.dpp_w_gen = 0.0
            print(f"DPP (fixed_w1): ctx_base only, gen=w1, no R loss")
        else:
            self.optim = build_optimizer(self.model.prompt_mid_learner, cfg.OPTIM)
            self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
            self.register_model("prompt_mid_learner", self.model.prompt_mid_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # --- Proposal A: Prompt-space Fisher (EWC on ctx parameters) ---
        self.prompt_fisher_w = cfg.TRAINER.COOP.PROMPT_FISHER_W
        if self.prompt_fisher_w > 0:
            self.ctx_init = self.model.prompt_learner.ctx.clone().detach().to(self.device)
            self.fisher_prompt = self._get_prompt_fisher(cfg, clip_model, classnames)
            self.fisher_prompt = self.fisher_prompt.to(self.device)
            print(f"Prompt-space Fisher: shape={self.fisher_prompt.shape}, "
                  f"weight={self.prompt_fisher_w}")

        # --- Proposal B: Task-1 KL path (w1 -> w) ---
        self.kl_path_w = cfg.TRAINER.COOP.KL_PATH_W
        if self.kl_path_w > 0:
            feat_w1 = self.model.ori_embedding.detach()
            self.feat_w1 = (feat_w1 / feat_w1.norm(dim=-1, keepdim=True)).to(self.device)
            print(f"KL path: feat_w1 shape={self.feat_w1.shape}, weight={self.kl_path_w}")

        # --- Proposal F: Zero-Shot Distribution Distillation (ZSDD) ---
        self.zsdd_w = cfg.TRAINER.COOP.ZSDD_W
        self.zsdd_tau = cfg.TRAINER.COOP.ZSDD_TAU
        if self.zsdd_w > 0:
            feat_w1 = self.model.ori_embedding.detach()
            self.zsdd_feat_w1 = (feat_w1 / feat_w1.norm(dim=-1, keepdim=True)).to(self.device)
            print(f"ZSDD: tau={self.zsdd_tau}, weight={self.zsdd_w}, "
                  f"feat_w1 shape={self.zsdd_feat_w1.shape}")

        # --- Visual Anchor ---
        self.va_w = cfg.TRAINER.COOP.VA_W
        self.va_tau = cfg.TRAINER.COOP.VA_TAU
        if self.va_w > 0:
            feat_w1 = self.model.ori_embedding.detach()
            self.va_feat_w1 = (feat_w1 / feat_w1.norm(dim=-1, keepdim=True)).to(self.device)
            # Augmentation pipeline for generating two views
            # CLIP ViT-B/16 uses 224x224 input
            img_size = cfg.INPUT.SIZE[0]
            _va_preset = getattr(cfg.TRAINER.COOP, "VA_AUG_PRESET", "default")
            _va_scale_ov = getattr(cfg.TRAINER.COOP, "VA_CROP_SCALE_MIN", -1.0)
            _va_jit_mul = getattr(cfg.TRAINER.COOP, "VA_JITTER_STRENGTH", -1.0)
            if _va_preset == "none":
                self.va_aug = T.Compose([])
                print(f"[VA aug] preset=none (identity)")
            else:
                if _va_preset == "weak":
                    _s, _j, _gp = 0.9, (0.1, 0.1, 0.05, 0.02), 0.0
                elif _va_preset == "strong":
                    _s, _j, _gp = 0.2, (0.8, 0.8, 0.4, 0.2), 0.2
                else:
                    _s, _j, _gp = 0.7, (0.4, 0.4, 0.2, 0.1), 0.1
                if _va_scale_ov >= 0.0:
                    _s = float(_va_scale_ov)
                if _va_jit_mul >= 0.0:
                    _j = tuple(v * float(_va_jit_mul) for v in _j)
                _ops = [
                    T.RandomResizedCrop(img_size, scale=(_s, 1.0), ratio=(0.75, 1.33)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ColorJitter(brightness=_j[0], contrast=_j[1], saturation=_j[2], hue=_j[3]),
                ]
                if _gp > 0.0:
                    _ops.append(T.RandomGrayscale(p=_gp))
                self.va_aug = T.Compose(_ops)
                print(f"[VA aug] preset={_va_preset} scale_min={_s} jitter={_j} grayscale_p={_gp}")
            print(f"Visual Anchor: tau={self.va_tau}, weight={self.va_w}, "
                  f"feat_w1 shape={self.va_feat_w1.shape}")

        # --- Proposal G: Adaptive W Scheduling ---
        self.w_schedule = cfg.TRAINER.COOP.W_SCHEDULE
        self.w_max = cfg.TRAINER.COOP.W
        self.w_min = cfg.TRAINER.COOP.W_MIN
        if self.w_min <= 0:
            self.w_min = self.w_max / 3.0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        if self.w_schedule != "none":
            print(f"W schedule: {self.w_schedule}, W_min={self.w_min:.2f}, "
                  f"W_max={self.w_max:.2f}, max_epoch={self.max_epoch}")

    def _get_fisher(self, cfg, clip_model, classnames):
        """Compute or load cached Fisher at the zero-shot checkpoint."""
        cache_path = get_fisher_cache_path(
            cfg.OUTPUT_DIR, cfg.DATASET.NAME, cfg.SEED, cfg.MODEL.BACKBONE.NAME
        )

        if osp.exists(cache_path):
            fisher_diag = load_fisher(cache_path, device=self.device)
        else:
            print(f"Computing diagonal Fisher ({cfg.TRAINER.COOP.FISHER_SAMPLES} samples)...")
            clip_for_fisher = load_clip_to_cpu(cfg)
            if cfg.TRAINER.COOP.PREC in ("fp32", "amp"):
                clip_for_fisher.float()
            clip_for_fisher.to(self.device)

            fisher_diag = compute_feature_fisher(
                clip_model=clip_for_fisher,
                dataloader=self.train_loader_x,
                classnames=classnames,
                dataset_name=cfg.DATASET.NAME,
                device=self.device,
                n_samples=cfg.TRAINER.COOP.FISHER_SAMPLES,
            )
            save_fisher(fisher_diag, cache_path)

            del clip_for_fisher
            torch.cuda.empty_cache()

        fisher_diag = normalize_fisher(fisher_diag, mode=cfg.TRAINER.COOP.FISHER_NORM)
        print(f"Fisher shape: {fisher_diag.shape}, "
              f"min: {fisher_diag.min():.6f}, max: {fisher_diag.max():.6f}, "
              f"mean: {fisher_diag.mean():.6f}, norm_mode: {cfg.TRAINER.COOP.FISHER_NORM}")
        return fisher_diag

    def _get_prompt_fisher(self, cfg, clip_model, classnames):
        """Compute or load cached prompt-space Fisher (Proposal A)."""
        cache_path = get_prompt_fisher_cache_path(
            cfg.OUTPUT_DIR, cfg.DATASET.NAME, cfg.SEED, cfg.MODEL.BACKBONE.NAME
        )

        if osp.exists(cache_path):
            fisher_diag = load_fisher(cache_path, device=self.device)
        else:
            print(f"Computing prompt-space Fisher ({cfg.TRAINER.COOP.FISHER_SAMPLES} samples)...")
            fisher_diag = compute_prompt_fisher(
                clip_model=clip_model,
                text_encoder=self.model.text_encoder,
                prompt_learner=self.model.prompt_learner,
                dataloader=self.train_loader_x,
                device=self.device,
                n_samples=cfg.TRAINER.COOP.FISHER_SAMPLES,
            )
            save_fisher(fisher_diag, cache_path)

        fisher_diag = normalize_fisher(fisher_diag, mode=cfg.TRAINER.COOP.FISHER_NORM)
        print(f"Prompt Fisher shape: {fisher_diag.shape}, "
              f"min: {fisher_diag.min():.6f}, max: {fisher_diag.max():.6f}, "
              f"mean: {fisher_diag.mean():.6f}")
        return fisher_diag

    def calculate_line_loss(self, feature_mid, image):
        logits = self.model(image, sign='LMC', text_features_mid=feature_mid)
        return logits



    def _get_current_w(self):
        """Get the current W value, applying schedule if enabled (Proposal G)."""
        if self.w_schedule == "none":
            return self.w
        # self.epoch is 0-indexed in DASSL (0 to max_epoch-1)
        progress = min(self.epoch / max(self.max_epoch - 1, 1), 1.0)
        if self.w_schedule == "linear":
            return self.w_min + (self.w_max - self.w_min) * progress
        elif self.w_schedule == "cosine":
            return self.w_min + 0.5 * (self.w_max - self.w_min) * (1 - np.cos(np.pi * progress))
        return self.w

    def _compute_loss_dpp(self, image, label):
        """DPP: Disentangled Prompt Pair loss.

        L = L_CE(ctx_base) + W_gen * L_cos(ctx_gen, w1) + W_LMC * L_lmc(ctx_gen -> ctx_base)

        ctx_base (prompt_mid_learner): CE only, free to specialize
        ctx_gen (prompt_gen_learner): cosine score only, stays near w1
        LMC path: interpolate feat_gen -> feat_base, ensure low CE along path

        When DPP_GEN_MODE == "fixed_w1": gen endpoint is fixed to w1 (no
        learnable c_gen, no R loss). Only c_base is trained with CE + VA + LMC.
        """
        # --- ctx_base: CE loss ---
        image_features = self.model.image_encoder(image.type(self.model.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.model.logit_scale.exp()

        prompts_base = self.model.prompt_mid_learner()
        feat_base = self.model.text_encoder(prompts_base, self.model.tokenized_prompts)
        feat_base = feat_base / feat_base.norm(dim=-1, keepdim=True)
        logits_base = logit_scale * image_features @ feat_base.t()
        loss_ce = F.cross_entropy(logits_base, label)

        # --- Generalization endpoint ---
        gen_mode = self.cfg.TRAINER.COOP.DPP_GEN_MODE
        feat_w1 = self.model.ori_embedding
        feat_w1 = feat_w1 / feat_w1.norm(dim=-1, keepdim=True)

        if gen_mode == "fixed_w1":
            # Ablation: use w1 directly as gen endpoint, no R loss
            feat_gen = feat_w1.detach()
            score_gen = torch.tensor(0.0, device=feat_base.device)
            loss = loss_ce
        else:
            # Default: learned c_gen with cosine regularizer
            prompts_gen = self.model.prompt_gen_learner()
            feat_gen = self.model.text_encoder(prompts_gen, self.model.tokenized_prompts)
            feat_gen = feat_gen / feat_gen.norm(dim=-1, keepdim=True)

            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
            score_gen = 1.0 - torch.mean(cos(feat_gen, feat_w1))
            loss = loss_ce + self.dpp_w_gen * score_gen

        # --- LMC path: feat_gen -> feat_base ---
        loss_lmc_val = 0.0
        if self.cfg.TRAINER.COOP.COOP_LMC:
            line_samples = np.arange(
                0.1, 1.01, 1.0 / float(self.cfg.TRAINER.COOP.NUM_SAMPLES)
            )
            total_loss_LMC = 0.0
            for t in line_samples:
                feat_mid = feat_gen + (feat_base - feat_gen) * t
                feat_mid = feat_mid / feat_mid.norm(dim=-1, keepdim=True)
                logits_mid = logit_scale * image_features @ feat_mid.t()
                total_loss_LMC += F.cross_entropy(logits_mid, label) / len(line_samples)

            loss = loss + self.cfg.TRAINER.COOP.W_LMC * total_loss_LMC
            loss_lmc_val = total_loss_LMC.item()

        # --- Visual Anchor on ctx_base (if enabled) ---
        loss_va_val = 0.0
        if self.va_w > 0:
            tau = self.va_tau
            logit_scale_va = self.model.logit_scale.exp().float()
            with torch.no_grad():
                x_a = self.va_aug(image)
                v_a = self.model.image_encoder(x_a.type(self.model.dtype)).float()
                v_a = v_a / v_a.norm(dim=-1, keepdim=True)
                del x_a
                s_w1 = logit_scale_va * v_a @ self.va_feat_w1.t()
                p_w1_va = F.softmax(s_w1 / tau, dim=-1)
                del v_a, s_w1

                x_b = image if self.cfg.TRAINER.COOP.VA_SINGLE_VIEW else self.va_aug(image)
                v_b = self.model.image_encoder(x_b.type(self.model.dtype)).float()
                v_b = v_b / v_b.norm(dim=-1, keepdim=True)
                del x_b

            # Student: ctx_base text features against augmented view
            s_w = logit_scale_va * v_b @ feat_base.float().t()
            log_p_w_va = F.log_softmax(s_w / tau, dim=-1)

            loss_va = F.kl_div(log_p_w_va, p_w1_va, reduction='batchmean')
            loss = loss + self.va_w * loss_va
            loss_va_val = loss_va.item()

        loss_summary = {
            "loss": loss.item(),
            "loss_ce": loss_ce.item(),
            "loss_score_gen": score_gen.item(),
            "loss_lmc": loss_lmc_val,
            "loss_va": loss_va_val,
            "acc": compute_accuracy(logits_base, label)[0].item(),
        }
        return loss, loss_summary

    def _compute_loss(self, image, label):
        """Compute the full loss with all active components.

        Components (all toggled by config flags):
          - CE + W * score (always on; W may follow a schedule)
          - Task-2 LMC path: w2 -> w  (COOP_LMC=True, weight W_LMC)
          - Prompt-space Fisher EWC    (PROMPT_FISHER_W > 0)  [Proposal A]
          - Task-1 KL path: w1 -> w   (KL_PATH_W > 0)        [Proposal B]
          - ZSDD: KL distillation      (ZSDD_W > 0)           [Proposal F]
        """
        if self.dpp:
            return self._compute_loss_dpp(image, label)

        current_w = self._get_current_w()
        output, score = self.model(image)
        loss_ce = F.cross_entropy(output, label)
        loss = loss_ce + current_w * score

        loss_lmc_val = 0.0
        loss_pf_val = 0.0
        loss_kl_val = 0.0

        # --- Task-2 LMC path (w2 -> w): original MERGETUNE ---
        if self.cfg.TRAINER.COOP.COOP_LMC:
            line_samples = np.arange(
                0.1, 1.01, 1.0 / float(self.cfg.TRAINER.COOP.NUM_SAMPLES)
            )
            interp_type = self.cfg.TRAINER.COOP.INTERP_TYPE

            if interp_type in ("slerp", "linear_prompt"):
                # Prompt-space interpolation: interpolate ctx, then encode
                # "slerp" = spherical, "linear_prompt" = linear (baseline for ablation)
                ctx_w2 = self.model.prompt_learner.ctx  # [N, D] or [C, N, D]
                ctx_w = self.model.prompt_mid_learner.ctx  # same shape
                use_per_token = self.cfg.TRAINER.COOP.SLERP_PER_TOKEN
                ctp = self.cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

                total_loss_LMC = 0.0
                for t in line_samples:
                    if interp_type == "slerp":
                        if use_per_token and ctx_w2.dim() == 2:
                            ctx_interp = slerp_per_token(ctx_w2, ctx_w, t)
                        else:
                            ctx_interp = slerp(ctx_w2, ctx_w, t)
                    else:
                        # linear_prompt: linear interpolation in prompt space
                        ctx_interp = ctx_w2 + (ctx_w - ctx_w2) * t

                    prompts_interp = _build_prompts_from_ctx(
                        ctx_interp, self.model.prompt_mid_learner, ctp
                    )
                    feature_mid = self.model.text_encoder(
                        prompts_interp, self.model.tokenized_prompts
                    )
                    output_LMC = self.calculate_line_loss(feature_mid, image)
                    total_loss_LMC += F.cross_entropy(output_LMC, label) / len(line_samples)
            else:
                # "linear" (default): feature-space interpolation (original MERGETUNE)
                feature_start = self.model.text_encoder(
                    self.model.prompt_learner(), self.model.tokenized_prompts
                )
                feature_end = self.model.text_encoder(
                    self.model.prompt_mid_learner(), self.model.tokenized_prompts
                )

                total_loss_LMC = 0.0
                for t in line_samples:
                    feature_mid = feature_start + (feature_end - feature_start) * t
                    output_LMC = self.calculate_line_loss(feature_mid, image)
                    total_loss_LMC += F.cross_entropy(output_LMC, label) / len(line_samples)

            loss = loss + self.cfg.TRAINER.COOP.W_LMC * total_loss_LMC
            loss_lmc_val = total_loss_LMC.item() if isinstance(total_loss_LMC, torch.Tensor) else total_loss_LMC

        # --- Proposal A: Prompt-space Fisher (EWC on ctx) ---
        if self.prompt_fisher_w > 0:
            ctx_diff = self.model.prompt_mid_learner.ctx - self.ctx_init
            if ctx_diff.dim() == 2:
                loss_pf = (self.fisher_prompt * ctx_diff.pow(2)).sum()
            else:
                loss_pf = (self.fisher_prompt.unsqueeze(0) * ctx_diff.pow(2)).sum()
            loss = loss + self.prompt_fisher_w * loss_pf
            loss_pf_val = loss_pf.item()

        # --- Proposal B: Task-1 KL path (w1 -> w) ---
        if self.kl_path_w > 0:
            line_samples_kl = np.arange(
                0.1, 1.01, 1.0 / float(self.cfg.TRAINER.COOP.NUM_SAMPLES)
            )
            # Reference distribution at w1 (no grad)
            with torch.no_grad():
                logits_w1 = self.model(image, sign='LMC', text_features_mid=self.feat_w1)
                p_w1 = F.softmax(logits_w1.float(), dim=-1)

            # Current text features
            feat_w = self.model.text_encoder(
                self.model.prompt_mid_learner(), self.model.tokenized_prompts
            )
            feat_w = feat_w / feat_w.norm(dim=-1, keepdim=True)

            total_loss_kl = 0.0
            for alpha in line_samples_kl:
                feat_interp = self.feat_w1 + alpha * (feat_w - self.feat_w1)
                logits_interp = self.model(image, sign='LMC', text_features_mid=feat_interp)
                log_p_interp = F.log_softmax(logits_interp.float(), dim=-1)
                total_loss_kl += F.kl_div(
                    log_p_interp, p_w1, reduction='batchmean'
                ) / len(line_samples_kl)

            loss = loss + self.kl_path_w * total_loss_kl
            loss_kl_val = total_loss_kl.item() if isinstance(total_loss_kl, torch.Tensor) else total_loss_kl

        # --- Proposal F: Zero-Shot Distribution Distillation (ZSDD) ---
        loss_zsdd_val = 0.0
        if self.zsdd_w > 0:
            tau = self.zsdd_tau
            # Zero-shot reference logits (no grad)
            with torch.no_grad():
                logits_w1 = self.model(image, sign='LMC', text_features_mid=self.zsdd_feat_w1)
                p_w1_soft = F.softmax(logits_w1.float() / tau, dim=-1)

            # Current logits (already computed as 'output', but recompute for tau)
            log_p_w_soft = F.log_softmax(output.float() / tau, dim=-1)

            # KL divergence scaled by tau^2 (standard distillation scaling)
            loss_zsdd = F.kl_div(log_p_w_soft, p_w1_soft, reduction='batchmean') * (tau ** 2)

            loss = loss + self.zsdd_w * loss_zsdd
            loss_zsdd_val = loss_zsdd.item()

        # --- Visual Anchor: augmentation-invariance KL ---
        loss_va_val = 0.0
        if self.va_w > 0:
            tau = self.va_tau
            logit_scale_va = self.model.logit_scale.exp().float()
            with torch.no_grad():
                # Teacher view: augment, encode, compute distribution, then free
                x_a = self.va_aug(image)
                v_a = self.model.image_encoder(x_a.type(self.model.dtype)).float()
                v_a = v_a / v_a.norm(dim=-1, keepdim=True)
                del x_a
                s_w1 = logit_scale_va * v_a @ self.va_feat_w1.t()
                p_w1_va = F.softmax(s_w1 / tau, dim=-1)
                del v_a, s_w1

                # Student view: augment, encode, then free augmented images
                x_b = image if self.cfg.TRAINER.COOP.VA_SINGLE_VIEW else self.va_aug(image)
                v_b = self.model.image_encoder(x_b.type(self.model.dtype)).float()
                v_b = v_b / v_b.norm(dim=-1, keepdim=True)
                del x_b

            # Student distribution (grad flows through feat_w to ctx)
            feat_w = self.model.text_encoder(
                self.model.prompt_mid_learner(), self.model.tokenized_prompts
            ).float()
            feat_w = feat_w / feat_w.norm(dim=-1, keepdim=True)
            s_w = logit_scale_va * v_b @ feat_w.t()
            log_p_w_va = F.log_softmax(s_w / tau, dim=-1)

            loss_va = F.kl_div(log_p_w_va, p_w1_va, reduction='batchmean')

            loss = loss + self.va_w * loss_va
            loss_va_val = loss_va.item()

        loss_summary = {
            "loss": loss.item(),
            "loss_ce": loss_ce.item(),
            "loss_score": score.item(),
            "loss_lmc": loss_lmc_val,
            "loss_prompt_fisher": loss_pf_val,
            "loss_kl_path": loss_kl_val,
            "loss_zsdd": loss_zsdd_val,
            "loss_va": loss_va_val,
            "acc": compute_accuracy(output, label)[0].item(),
        }
        if self.w_schedule != "none":
            loss_summary["current_w"] = current_w

        # Prompt-space diagnostic: log midpoint prompt norm (every 50 batches to reduce sync overhead)
        if (self.cfg.TRAINER.COOP.COOP_LMC
                and self.cfg.TRAINER.COOP.INTERP_TYPE in ("slerp", "linear_prompt")
                and self.batch_idx % 50 == 0):
            with torch.no_grad():
                ctx_w2 = self.model.prompt_learner.ctx
                ctx_w = self.model.prompt_mid_learner.ctx
                mid_linear = ctx_w2 + 0.5 * (ctx_w - ctx_w2)
                if self.cfg.TRAINER.COOP.SLERP_PER_TOKEN and ctx_w2.dim() == 2:
                    mid_slerp = slerp_per_token(ctx_w2, ctx_w, 0.5)
                else:
                    mid_slerp = slerp(ctx_w2, ctx_w, 0.5)
                loss_summary["norm_w2"] = ctx_w2.norm().item()
                loss_summary["norm_w"] = ctx_w.norm().item()
                loss_summary["norm_mid_linear"] = mid_linear.norm().item()
                loss_summary["norm_mid_slerp"] = mid_slerp.norm().item()

        return loss, loss_summary

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.COOP.PREC

        if prec == "amp":
            with autocast():
                loss, loss_summary = self._compute_loss(image, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        elif self.dpp:
            # DPP: use self.optim directly to avoid double-step from
            # model_backward_and_update iterating over two registered names
            loss, loss_summary = self._compute_loss(image, label)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        else:
            loss, loss_summary = self._compute_loss(image, label)
            self.model_backward_and_update(loss)

        if (self.batch_idx + 1) == self.num_batches:
            #self.update_lr()
            self.sched.step()
            #self.sched_.step()
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label


    def after_epoch(self):
        """Skip best-model saving during warmup epochs.

        During warmup (epoch < WARMUP_EPOCH), the LR is tiny (1e-5) so
        the model is essentially the CoOp initialization.  Saving it as
        'best' creates a degenerate checkpoint on fine-grained datasets
        where training can never surpass the initialization accuracy.
        """
        # DPP diagnostic: log disentanglement cosines every epoch
        if self.dpp:
            self._log_dpp_diagnostics()

        warmup = self.cfg.OPTIM.WARMUP_EPOCH
        if warmup > 0 and (self.epoch + 1) <= warmup:
            return
        super().after_epoch()

    @torch.no_grad()
    def _log_dpp_diagnostics(self):
        """Log cosine similarities between gen/base/w1 during DPP training."""
        prompts_base = self.model.prompt_mid_learner()
        feat_base = self.model.text_encoder(prompts_base, self.model.tokenized_prompts).float()
        feat_base = feat_base / feat_base.norm(dim=-1, keepdim=True)

        prompts_gen = self.model.prompt_gen_learner()
        feat_gen = self.model.text_encoder(prompts_gen, self.model.tokenized_prompts).float()
        feat_gen = feat_gen / feat_gen.norm(dim=-1, keepdim=True)

        feat_w1 = self.model.ori_embedding.clone().float()
        feat_w1 = feat_w1 / feat_w1.norm(dim=-1, keepdim=True)

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
        cos_gen_w1 = cos(feat_gen, feat_w1).mean().item()
        cos_base_w1 = cos(feat_base, feat_w1).mean().item()
        cos_gen_base = cos(feat_gen, feat_base).mean().item()

        # Store in a list attached to self
        if not hasattr(self, '_dpp_dynamics'):
            self._dpp_dynamics = []
        self._dpp_dynamics.append({
            "epoch": self.epoch + 1,
            "cos_gen_w1": cos_gen_w1,
            "cos_base_w1": cos_base_w1,
            "cos_gen_base": cos_gen_base,
        })

        # Save to JSON after each epoch
        import json
        dynamics_path = osp.join(self.cfg.OUTPUT_DIR, "dpp_dynamics.json")
        os.makedirs(osp.dirname(dynamics_path), exist_ok=True)
        with open(dynamics_path, 'w') as f:
            json.dump(self._dpp_dynamics, f, indent=2)

        if (self.epoch + 1) % 10 == 0 or self.epoch == 0:
            print(f"  [DPP] epoch {self.epoch+1}: "
                  f"cos(gen,w1)={cos_gen_w1:.4f}, "
                  f"cos(base,w1)={cos_base_w1:.4f}, "
                  f"cos(gen,base)={cos_gen_base:.4f}")

    def after_train(self):
        print("Finished training in KgCoOp_COOPLMC")

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            self.test(split="val")

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
            print("Do evaluation on {} set".format(split))
        else:
            data_loader = self.test_loader
            print("Do evaluation on test set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    @torch.no_grad()
    def test_wspe(self, split=None):
        """WSPE: Weight-Space Prediction Ensemble.

        Averages logits from K models interpolated along the w2->w* line
        in prompt parameter space. No retraining needed.
        """
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT
        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            data_loader = self.test_loader

        K = self.cfg.TRAINER.COOP.WSPE_K
        spacing = self.cfg.TRAINER.COOP.WSPE_SPACING
        tau = self.cfg.TRAINER.COOP.WSPE_TAU

        # Generate alpha values
        if spacing == "uniform":
            alphas = [k / (K - 1) for k in range(K)] if K > 1 else [1.0]
        elif spacing == "end_weighted":
            # Denser near alpha=1 (w*): use quadratic spacing
            alphas = [(k / (K - 1)) ** 0.5 for k in range(K)] if K > 1 else [1.0]
        else:
            raise ValueError(f"Unknown WSPE spacing: {spacing}")

        print(f"WSPE ensemble: K={K}, spacing={spacing}, tau={tau}")
        print(f"  alpha values: {[f'{a:.3f}' for a in alphas]}")

        # Get w2 (fixed CoOp) and w* (trained) prompt contexts
        ctx_w2 = self.model.prompt_learner.ctx  # [N, D]
        ctx_w_star = self.model.prompt_mid_learner.ctx  # [N, D]
        ctp = self.cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

        # Precompute text features for each alpha
        text_features_list = []
        for alpha in alphas:
            ctx_interp = ctx_w2 + alpha * (ctx_w_star - ctx_w2)
            prompts_interp = _build_prompts_from_ctx(
                ctx_interp, self.model.prompt_mid_learner, ctp
            )
            feat = self.model.text_encoder(
                prompts_interp, self.model.tokenized_prompts
            )
            feat = feat / feat.norm(dim=-1, keepdim=True)
            text_features_list.append(feat)

        logit_scale = self.model.logit_scale.exp()

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            image_features = self.model.image_encoder(input.type(self.model.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Average logits across ensemble members
            ensemble_logits = torch.zeros(
                input.shape[0], text_features_list[0].shape[0],
                device=input.device, dtype=torch.float32
            )
            for feat in text_features_list:
                logits = logit_scale * image_features @ feat.t()
                if tau != 1.0:
                    logits = logits / tau
                ensemble_logits += logits.float()
            ensemble_logits /= K

            self.evaluator.process(ensemble_logits, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = "wspe_{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        print(f"\nWSPE Results (K={K}, spacing={spacing}, tau={tau}):")
        for k, v in results.items():
            print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

        return list(results.values())[0]

    @torch.no_grad()
    def test_ttai(self, split=None, ttai_mlp_path=None):
        """TTAI: Test-Time Adaptive Interpolation.

        Phase 1: Train a small MLP on base training data to predict per-image
                 alpha from frozen visual features. Alpha indicates how much
                 to weight w* vs w1 (zero-shot). The MLP is saved to disk
                 so it can be reused for new-class evaluation.
        Phase 2: At inference, predict alpha per test image and interpolate
                 text features accordingly.

        Args:
            split: evaluation split ("val" or "test")
            ttai_mlp_path: if provided, load pre-trained MLP instead of
                training a new one. Use this for new-class evaluation after
                training the MLP on base classes.

        Uses existing w* and w2 checkpoints — zero retraining of prompts.
        """
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT
        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            data_loader = self.test_loader

        # --- Get text features for w* and w1 (cast to float32 for consistent matmul) ---
        ctx_w_star = self.model.prompt_mid_learner()
        feat_w_star = self.model.text_encoder(ctx_w_star, self.model.tokenized_prompts).float()
        feat_w_star = feat_w_star / feat_w_star.norm(dim=-1, keepdim=True)

        feat_w1 = self.model.ori_embedding.clone().float()
        feat_w1 = feat_w1 / feat_w1.norm(dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp().float()
        feat_dim = feat_w_star.shape[1]  # 512

        # Build MLP architecture (same for train or load)
        mlp = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(self.device)

        if ttai_mlp_path and osp.exists(ttai_mlp_path):
            # --- Load pre-trained MLP ---
            print(f"TTAI: Loading pre-trained MLP from {ttai_mlp_path}")
            mlp.load_state_dict(torch.load(ttai_mlp_path, map_location=self.device))
        else:
            # --- Phase 1: Train MLP on base training data ---
            print("TTAI Phase 1: Training alpha predictor on base training data...")

            # Collect visual features and alpha targets from training data
            all_vis_features = []
            all_alpha_targets = []

            for batch_idx, batch in enumerate(tqdm(self.train_loader_x, desc="Collecting training data")):
                input, label = self.parse_batch_train(batch)
                image_features = self.model.image_encoder(input.type(self.model.dtype)).float()
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Logits under w* and w1
                logits_w_star = logit_scale * image_features @ feat_w_star.t()
                logits_w1 = logit_scale * image_features @ feat_w1.t()

                # Soft alpha target: confidence gap on correct class
                # Higher alpha -> prefer w*, lower alpha -> prefer w1
                probs_w_star = F.softmax(logits_w_star, dim=1)
                probs_w1 = F.softmax(logits_w1, dim=1)

                # Confidence on correct class for each model
                conf_w_star = probs_w_star.gather(1, label.unsqueeze(1)).squeeze(1)
                conf_w1 = probs_w1.gather(1, label.unsqueeze(1)).squeeze(1)

                # Alpha target: sigmoid of confidence gap (maps to [0,1])
                # Positive gap -> w* better -> alpha near 1
                # Negative gap -> w1 better -> alpha near 0
                alpha_target = torch.sigmoid(5.0 * (conf_w_star - conf_w1))

                all_vis_features.append(image_features.float())
                all_alpha_targets.append(alpha_target.float())

            all_vis_features = torch.cat(all_vis_features, dim=0)    # [N_train, 512]
            all_alpha_targets = torch.cat(all_alpha_targets, dim=0)  # [N_train]

            print(f"  Training samples: {all_vis_features.shape[0]}")
            print(f"  Alpha target stats: mean={all_alpha_targets.mean():.3f}, "
                  f"std={all_alpha_targets.std():.3f}, "
                  f"min={all_alpha_targets.min():.3f}, max={all_alpha_targets.max():.3f}")

            mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
            n_train = all_vis_features.shape[0]
            batch_size = min(256, n_train)
            n_epochs = 50

            mlp.train()
            for ep in range(n_epochs):
                perm = torch.randperm(n_train, device=self.device)
                epoch_loss = 0.0
                n_batches = 0
                for i in range(0, n_train, batch_size):
                    idx = perm[i:i + batch_size]
                    vis_batch = all_vis_features[idx]
                    target_batch = all_alpha_targets[idx]

                    mlp_optimizer.zero_grad()
                    # Enable grad for MLP training within no_grad context
                    with torch.enable_grad():
                        pred_alpha = mlp(vis_batch).squeeze(1)
                        loss = F.mse_loss(pred_alpha, target_batch)
                        loss.backward()
                    mlp_optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                if (ep + 1) % 10 == 0:
                    print(f"  Epoch {ep+1}/{n_epochs}, loss: {epoch_loss/n_batches:.4f}")

            # Diagnostic: check alpha distribution on training data
            mlp.eval()
            with torch.no_grad():
                train_alphas = mlp(all_vis_features).squeeze(1)
            print(f"\n  Predicted alpha on train: mean={train_alphas.mean():.3f}, "
                  f"std={train_alphas.std():.3f}, "
                  f"min={train_alphas.min():.3f}, max={train_alphas.max():.3f}")

            # Save MLP for reuse in new-class evaluation
            mlp_save_path = osp.join(self.cfg.OUTPUT_DIR, "ttai_mlp.pth")
            os.makedirs(osp.dirname(mlp_save_path), exist_ok=True)
            torch.save(mlp.state_dict(), mlp_save_path)
            print(f"  MLP saved to {mlp_save_path}")

            # Free training data
            del all_vis_features, all_alpha_targets

        # --- Phase 2: Adaptive inference ---
        print(f"\nTTAI Phase 2: Adaptive inference on {split} set...")
        mlp.eval()
        all_test_alphas = []

        for batch_idx, batch in enumerate(tqdm(data_loader, desc="TTAI inference")):
            input, label = self.parse_batch_test(batch)
            image_features = self.model.image_encoder(input.type(self.model.dtype)).float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Predict per-image alpha
            alpha = mlp(image_features.float()).squeeze(1)  # [B]
            all_test_alphas.append(alpha)

            # Interpolate text features per image: feat(α) = feat_w1 + α*(feat_w* - feat_w1)
            alpha_expanded = alpha.unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
            feat_w1_expanded = feat_w1.unsqueeze(0)           # [1, C, D]
            feat_ws_expanded = feat_w_star.unsqueeze(0)       # [1, C, D]

            # Per-image interpolated text features: [B, C, D]
            feat_interp = feat_w1_expanded + alpha_expanded * (feat_ws_expanded - feat_w1_expanded)
            feat_interp = feat_interp / feat_interp.norm(dim=-1, keepdim=True)

            # Compute per-image logits: [B, C]
            # image_features: [B, D], feat_interp: [B, C, D]
            logits = logit_scale * torch.bmm(
                feat_interp, image_features.unsqueeze(2)
            ).squeeze(2)  # [B, C]

            self.evaluator.process(logits, label)

        results = self.evaluator.evaluate()
        all_test_alphas = torch.cat(all_test_alphas, dim=0)

        print(f"\nTTAI Results:")
        for k, v in results.items():
            tag = "ttai_{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)
            print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

        print(f"\n  Test alpha stats: mean={all_test_alphas.mean():.3f}, "
              f"std={all_test_alphas.std():.3f}, "
              f"min={all_test_alphas.min():.3f}, max={all_test_alphas.max():.3f}")

        return list(results.values())[0]

    @torch.no_grad()
    def test_dpp(self, split=None):
        """DPP: Evaluate with alpha sweep between ctx_gen and ctx_base.

        Sweeps alpha from 0 (pure gen/w1) to 1 (pure base specialization)
        in feature space. Reports accuracy at each alpha and finds optimal.
        """
        self.set_model_mode("eval")

        if split is None:
            split = self.cfg.TEST.SPLIT
        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
            print(f"DPP alpha sweep on val set")
        else:
            data_loader = self.test_loader
            print(f"DPP alpha sweep on test set")

        # Precompute text features for both prompts
        prompts_base = self.model.prompt_mid_learner()
        feat_base = self.model.text_encoder(prompts_base, self.model.tokenized_prompts).float()
        feat_base = feat_base / feat_base.norm(dim=-1, keepdim=True)

        feat_w1 = self.model.ori_embedding.clone().float()
        feat_w1 = feat_w1 / feat_w1.norm(dim=-1, keepdim=True)

        gen_mode = self.cfg.TRAINER.COOP.DPP_GEN_MODE
        if gen_mode == "fixed_w1":
            feat_gen = feat_w1
            print(f"  GEN_MODE=fixed_w1: using w1 as gen endpoint")
        else:
            prompts_gen = self.model.prompt_gen_learner()
            feat_gen = self.model.text_encoder(prompts_gen, self.model.tokenized_prompts).float()
            feat_gen = feat_gen / feat_gen.norm(dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp().float()

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
        cos_gen_w1 = torch.mean(cos(feat_gen, feat_w1)).item()
        cos_base_w1 = torch.mean(cos(feat_base, feat_w1)).item()
        cos_gen_base = torch.mean(cos(feat_gen, feat_base)).item()
        print(f"  cos(gen, w1)={cos_gen_w1:.4f}, cos(base, w1)={cos_base_w1:.4f}, "
              f"cos(gen, base)={cos_gen_base:.4f}")

        # Precompute all image features
        all_image_features = []
        all_labels = []
        for batch in tqdm(data_loader, desc="Encoding images"):
            input, label = self.parse_batch_test(batch)
            image_features = self.model.image_encoder(input.type(self.model.dtype)).float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features)
            all_labels.append(label)
        all_image_features = torch.cat(all_image_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Alpha sweep
        alphas = np.arange(0.0, 1.01, 0.05)
        print(f"\nAlpha sweep ({len(alphas)} points):")
        print(f"{'alpha':>6} | {'accuracy':>8}")
        print("-" * 20)

        best_alpha = 0.0
        best_acc = 0.0
        results_per_alpha = []

        for alpha in alphas:
            feat_interp = feat_gen + alpha * (feat_base - feat_gen)
            feat_interp = feat_interp / feat_interp.norm(dim=-1, keepdim=True)

            logits = logit_scale * all_image_features @ feat_interp.t()
            preds = logits.argmax(dim=1)
            acc = (preds == all_labels).float().mean().item() * 100.0

            results_per_alpha.append((alpha, acc))
            print(f"{alpha:6.2f} | {acc:8.2f}")

            if acc > best_acc:
                best_acc = acc
                best_alpha = alpha

        print(f"\nBest: alpha={best_alpha:.2f}, accuracy={best_acc:.2f}%")
        print(f"Endpoints: alpha=0 (gen)={results_per_alpha[0][1]:.2f}%, "
              f"alpha=1 (base)={results_per_alpha[-1][1]:.2f}%")

        return best_acc

    @torch.no_grad()
    def test_dpp_ablation(self, split=None):
        """Ablation: compare 3 generalization endpoints using the same c_base.

        Evaluates alpha sweep with:
          1. f_gen (learned generalization prompt) — our DMC
          2. w1 (zero-shot CLIP features) — fixed, no learning
          3. w2 (CoOp checkpoint features) — fixed, no learning

        Uses the SAME trained c_base for all 3. No retraining needed.
        """
        self.set_model_mode("eval")

        if split is None:
            split = self.cfg.TEST.SPLIT
        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            data_loader = self.test_loader
        print(f"DPP ablation on {split} set")

        # Precompute c_base features (shared across all variants)
        prompts_base = self.model.prompt_mid_learner()
        feat_base = self.model.text_encoder(prompts_base, self.model.tokenized_prompts).float()
        feat_base = feat_base / feat_base.norm(dim=-1, keepdim=True)

        # Three generalization endpoints
        # 1. f_gen (learned)
        prompts_gen = self.model.prompt_gen_learner()
        feat_gen = self.model.text_encoder(prompts_gen, self.model.tokenized_prompts).float()
        feat_gen = feat_gen / feat_gen.norm(dim=-1, keepdim=True)

        # 2. w1 (zero-shot)
        feat_w1 = self.model.ori_embedding.clone().float()
        feat_w1 = feat_w1 / feat_w1.norm(dim=-1, keepdim=True)

        # 3. w2 (CoOp checkpoint = prompt_learner, frozen)
        prompts_w2 = self.model.prompt_learner()
        feat_w2 = self.model.text_encoder(prompts_w2, self.model.tokenized_prompts).float()
        feat_w2 = feat_w2 / feat_w2.norm(dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp().float()

        # Diagnostics
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
        print(f"  cos(f_gen, w1) = {cos(feat_gen, feat_w1).mean():.4f}")
        print(f"  cos(f_gen, w2) = {cos(feat_gen, feat_w2).mean():.4f}")
        print(f"  cos(w1, w2)    = {cos(feat_w1, feat_w2).mean():.4f}")
        print(f"  cos(f_base, w1)= {cos(feat_base, feat_w1).mean():.4f}")

        # Precompute image features
        all_image_features = []
        all_labels = []
        for batch in tqdm(data_loader, desc="Encoding images"):
            input, label = self.parse_batch_test(batch)
            image_features = self.model.image_encoder(input.type(self.model.dtype)).float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features)
            all_labels.append(label)
        all_image_features = torch.cat(all_image_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        alphas = np.arange(0.0, 1.01, 0.05)
        endpoints = {
            'f_gen': feat_gen,
            'w1': feat_w1,
            'w2': feat_w2,
        }

        print(f"\n{'alpha':>6}", end="")
        for name in endpoints:
            print(f" | {name:>8}", end="")
        print()
        print("-" * 40)

        results = {name: [] for name in endpoints}

        for alpha in alphas:
            print(f"{alpha:6.2f}", end="")
            for name, feat_endpoint in endpoints.items():
                feat_interp = feat_endpoint + alpha * (feat_base - feat_endpoint)
                feat_interp = feat_interp / feat_interp.norm(dim=-1, keepdim=True)
                logits = logit_scale * all_image_features @ feat_interp.t()
                preds = logits.argmax(dim=1)
                acc = (preds == all_labels).float().mean().item() * 100.0
                results[name].append((alpha, acc))
                print(f" | {acc:8.2f}", end="")
            print()

        # Summary
        print(f"\n=== Ablation Summary ===")
        for name in endpoints:
            accs = [a for _, a in results[name]]
            best_idx = np.argmax(accs)
            best_alpha = alphas[best_idx]
            best_acc = accs[best_idx]
            acc_at_020 = accs[4]  # alpha=0.20 is index 4
            print(f"  {name:>5}: best α={best_alpha:.2f} acc={best_acc:.2f}%, "
                  f"α=0.00 acc={accs[0]:.2f}%, α=0.20 acc={acc_at_020:.2f}%, "
                  f"α=1.00 acc={accs[-1]:.2f}%")

        return results

    @torch.no_grad()
    def test_dpp_select(self):
        """DPP α selection via multiple criteria (no new-class images used).

        Criteria tested:
          1. Val-Acc: argmax base-val accuracy
          2. Proxy-HM: split base classes into halves, treat one as "proxy-new"
          3. Drift-Penalty: val_acc(α) - λ·cos_drift(α), for multiple λ values
          4. Gram-Align: structural preservation of base text features vs w1
          5. Combined: val_acc(α) · gram_align(α), balances accuracy and structure

        Reports selected α from each criterion.
        """
        self.set_model_mode("eval")

        if self.val_loader is not None:
            data_loader = self.val_loader
            print("DPP α selection on val set")
        else:
            data_loader = self.test_loader
            print("DPP α selection on test set (no val_loader)")

        # Precompute text features for both endpoints
        prompts_base = self.model.prompt_mid_learner()
        feat_base = self.model.text_encoder(prompts_base, self.model.tokenized_prompts).float()
        feat_base = feat_base / feat_base.norm(dim=-1, keepdim=True)

        prompts_gen = self.model.prompt_gen_learner()
        feat_gen = self.model.text_encoder(prompts_gen, self.model.tokenized_prompts).float()
        feat_gen = feat_gen / feat_gen.norm(dim=-1, keepdim=True)

        # w1 (zero-shot) text features
        feat_w1 = self.model.ori_embedding.clone().float()
        feat_w1 = feat_w1 / feat_w1.norm(dim=-1, keepdim=True)

        # w1 Gram matrix (pairwise similarity structure)
        gram_w1 = feat_w1 @ feat_w1.t()  # [C, C]

        logit_scale = self.model.logit_scale.exp().float()
        n_cls = feat_base.shape[0]

        # Precompute all image features and labels
        all_image_features = []
        all_labels = []
        for batch in tqdm(data_loader, desc="Encoding val images"):
            input, label = self.parse_batch_test(batch)
            image_features = self.model.image_encoder(input.type(self.model.dtype)).float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features)
            all_labels.append(label)
        all_image_features = torch.cat(all_image_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        print(f"  Val images: {all_image_features.shape[0]}, Classes: {n_cls}")

        # Random class splits for proxy-HM
        n_splits = 10
        class_splits = []
        for _ in range(n_splits):
            perm = np.random.permutation(n_cls)
            half = n_cls // 2
            class_splits.append((set(perm[:half].tolist()), set(perm[half:].tolist())))

        # Precompute w1 neighborhood structure for Neighborhood Preservation
        K_neighbors = min(5, n_cls - 1)  # K=5 nearest neighbors
        w1_sim = feat_w1 @ feat_w1.t()  # [C, C]
        # Zero out diagonal so a class isn't its own neighbor
        w1_sim.fill_diagonal_(-1.0)
        _, w1_nn = w1_sim.topk(K_neighbors, dim=1)  # [C, K]
        print(f"  Neighborhood K={K_neighbors}")

        # Effective rank of w1 for reference
        _, s_w1, _ = torch.svd(feat_w1)
        p_w1 = s_w1 / s_w1.sum()
        erank_w1 = torch.exp(-(p_w1 * torch.log(p_w1 + 1e-10)).sum()).item()
        print(f"  Effective rank at w1: {erank_w1:.2f}")

        # Alpha sweep
        alphas = np.arange(0.0, 1.01, 0.05)
        val_accs = []
        proxy_hms = []
        cos_drifts = []      # mean cosine drift from w1
        gram_aligns = []     # Gram matrix correlation with w1
        eff_ranks = []        # effective rank of prototype matrix
        neigh_preserv = []    # neighborhood preservation score

        header = (f"{'α':>5} | {'ValAcc':>7} | {'ProxyHM':>8} | "
                  f"{'CosDrift':>8} | {'GramAl':>7} | {'EffRank':>7} | {'NP':>6}")
        print(f"\n{header}")
        print("-" * 65)

        for alpha in alphas:
            feat_interp = feat_gen + alpha * (feat_base - feat_gen)
            feat_interp = feat_interp / feat_interp.norm(dim=-1, keepdim=True)

            logits = logit_scale * all_image_features @ feat_interp.t()
            preds = logits.argmax(dim=1)

            # 1. Val accuracy
            val_acc = (preds == all_labels).float().mean().item() * 100.0
            val_accs.append(val_acc)

            # 2. Proxy HM
            split_hms = []
            for split_a, split_b in class_splits:
                mask_a = torch.tensor([l.item() in split_a for l in all_labels], device=all_labels.device)
                mask_b = torch.tensor([l.item() in split_b for l in all_labels], device=all_labels.device)
                if mask_a.sum() > 0 and mask_b.sum() > 0:
                    acc_a = (preds[mask_a] == all_labels[mask_a]).float().mean().item() * 100.0
                    acc_b = (preds[mask_b] == all_labels[mask_b]).float().mean().item() * 100.0
                    if acc_a + acc_b > 0:
                        split_hms.append(2 * acc_a * acc_b / (acc_a + acc_b))
            proxy_hm = np.mean(split_hms) if split_hms else 0.0
            proxy_hms.append(proxy_hm)

            # 3. Cosine drift from w1 (mean across classes)
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
            cos_drift = (1.0 - cos(feat_interp, feat_w1).mean()).item()
            cos_drifts.append(cos_drift)

            # 4. Gram matrix alignment: correlation of pairwise similarities with w1
            gram_interp = feat_interp @ feat_interp.t()  # [C, C]
            # Flatten upper triangle (exclude diagonal)
            mask_upper = torch.triu(torch.ones(n_cls, n_cls, device=gram_interp.device), diagonal=1).bool()
            g1 = gram_w1[mask_upper]
            g2 = gram_interp[mask_upper]
            # Pearson correlation
            g1_centered = g1 - g1.mean()
            g2_centered = g2 - g2.mean()
            gram_corr = (g1_centered * g2_centered).sum() / (g1_centered.norm() * g2_centered.norm() + 1e-8)
            gram_align = gram_corr.item()
            gram_aligns.append(gram_align)

            # 5. Effective rank of prototype matrix
            _, s_vals, _ = torch.svd(feat_interp)
            p = s_vals / s_vals.sum()
            erank = torch.exp(-(p * torch.log(p + 1e-10)).sum()).item()
            eff_ranks.append(erank)

            # 6. Neighborhood preservation
            interp_sim = feat_interp @ feat_interp.t()
            interp_sim.fill_diagonal_(-1.0)
            _, interp_nn = interp_sim.topk(K_neighbors, dim=1)  # [C, K]
            # For each class, count how many of its w1 neighbors are preserved
            preserved = 0
            for c in range(n_cls):
                w1_set = set(w1_nn[c].cpu().tolist())
                interp_set = set(interp_nn[c].cpu().tolist())
                preserved += len(w1_set & interp_set)
            np_score = preserved / (n_cls * K_neighbors)
            neigh_preserv.append(np_score)

            print(f"{alpha:5.2f} | {val_acc:7.2f} | {proxy_hm:8.2f} | "
                  f"{cos_drift:8.4f} | {gram_align:7.4f} | {erank:7.2f} | {np_score:6.3f}")

        # Convert to arrays for criterion computation
        val_accs = np.array(val_accs)
        cos_drifts = np.array(cos_drifts)
        gram_aligns = np.array(gram_aligns)

        print(f"\n=== α Selection Results ===")

        # Criterion 1: Val-Acc
        idx = np.argmax(val_accs)
        print(f"  Val-Acc:           α={alphas[idx]:.2f} (val_acc={val_accs[idx]:.2f})")

        # Criterion 2: Proxy-HM
        idx = np.argmax(proxy_hms)
        print(f"  Proxy-HM:          α={alphas[idx]:.2f} (proxy_hm={proxy_hms[idx]:.2f})")

        # Criterion 3: Drift-Penalty for multiple λ
        # Normalize val_acc to [0,1] and cos_drift to [0,1] for fair combination
        va_norm = (val_accs - val_accs.min()) / (val_accs.max() - val_accs.min() + 1e-8)
        cd_norm = (cos_drifts - cos_drifts.min()) / (cos_drifts.max() - cos_drifts.min() + 1e-8)
        for lam in [0.5, 1.0, 2.0, 4.0]:
            score = va_norm - lam * cd_norm
            idx = np.argmax(score)
            print(f"  Drift-Pen (λ={lam:.1f}): α={alphas[idx]:.2f} (score={score[idx]:.4f})")

        # Criterion 4: Gram alignment
        idx = np.argmax(gram_aligns)
        print(f"  Gram-Align:        α={alphas[idx]:.2f} (corr={gram_aligns[idx]:.4f})")

        # Criterion 5: Combined val_acc * gram_align
        combined = va_norm * gram_aligns
        idx = np.argmax(combined)
        print(f"  ValAcc×Gram:       α={alphas[idx]:.2f} (combined={combined[idx]:.4f})")

        # Criterion 6: Combined val_acc * (1 - cos_drift)
        preservation = 1.0 - cd_norm
        combined2 = va_norm * preservation
        idx = np.argmax(combined2)
        print(f"  ValAcc×Preserve:   α={alphas[idx]:.2f} (combined={combined2[idx]:.4f})")

        # Criterion 7: Effective Rank — select α that maximizes rank
        eff_ranks = np.array(eff_ranks)
        idx = np.argmax(eff_ranks)
        print(f"  EffRank-Max:       α={alphas[idx]:.2f} (rank={eff_ranks[idx]:.2f}, w1={erank_w1:.2f})")

        # Criterion 7b: Effective Rank — select α where rank drops below w1's rank * threshold
        for thresh in [0.99, 0.95]:
            target = erank_w1 * thresh
            selected = alphas[0]
            for i, a in enumerate(alphas):
                if eff_ranks[i] >= target:
                    selected = a
                else:
                    break
            sel_idx = np.argmin(np.abs(alphas - selected))
            print(f"  EffRank-{thresh:.2f}:     α={selected:.2f} (threshold={target:.2f})")

        # Criterion 8: Neighborhood Preservation — select highest α with NP ≥ threshold
        neigh_preserv = np.array(neigh_preserv)
        for thresh in [1.0, 0.95, 0.90]:
            selected = alphas[0]
            for i, a in enumerate(alphas):
                if neigh_preserv[i] >= thresh:
                    selected = a
            sel_idx = np.argmin(np.abs(alphas - selected))
            print(f"  NeighPres≥{thresh:.2f}:   α={selected:.2f} (NP={neigh_preserv[sel_idx]:.3f})")

        # Criterion 8b: Neighborhood Preservation — select α at steepest NP drop
        np_diffs = np.diff(neigh_preserv)
        if len(np_diffs) > 0:
            steepest = np.argmin(np_diffs)  # most negative change
            selected = alphas[steepest]  # α just before the drop
            print(f"  NeighPres-Elbow:   α={selected:.2f} (NP drop={np_diffs[steepest]:.4f})")

        # ================================================================
        # Criterion 9: vMF Log-Likelihood Ratio (Option F)
        #
        # score(α) = -CE_val(α) + κ · mean_c[cos(feat_c(α), feat_c(w1))]
        #
        # Both terms are in nats (log-probability units):
        #   - First term: log P(val_data | feat(α)) = -CE
        #   - Second term: log P(feat(α) | w1) under vMF ∝ κ·cos
        #
        # No min-max normalization. κ controls the balance.
        # ================================================================

        # Compute CE on val set at each alpha
        val_ces = []
        for i, alpha in enumerate(alphas):
            feat_interp = feat_gen + alpha * (feat_base - feat_gen)
            feat_interp = feat_interp / feat_interp.norm(dim=-1, keepdim=True)
            logits = logit_scale * all_image_features @ feat_interp.t()
            ce = torch.nn.functional.cross_entropy(logits, all_labels).item()
            val_ces.append(ce)
        val_ces = np.array(val_ces)

        # cos_sim(α) = mean_c[cos(feat_c(α), feat_c(w1))]
        # Already computed as: cos_sim = 1.0 - cos_drifts
        cos_sims = 1.0 - cos_drifts

        print(f"\n  --- vMF Log-Likelihood Ratio (Option F) ---")
        print(f"  CE range: [{val_ces.min():.4f}, {val_ces.max():.4f}]")
        print(f"  cos_sim range: [{cos_sims.min():.4f}, {cos_sims.max():.4f}]")

        for kappa in [1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]:
            score = -val_ces + kappa * cos_sims
            idx = np.argmax(score)
            print(f"  vMF (κ={kappa:>5.0f}):     α={alphas[idx]:.2f} (score={score[idx]:.4f}, "
                  f"-CE={-val_ces[idx]:.4f}, κ·cos={kappa*cos_sims[idx]:.4f})")

        # ================================================================
        # Criterion 10: KL-based α selection
        #
        # score(α) = -CE_base(α) - β · KL(q_α || q_w1)
        #
        # Both terms in nats:
        #   - CE_base(α): cross-entropy on val labels (base-class fit)
        #   - KL(q_α || q_w1): how much α's predictions diverge from
        #     zero-shot CLIP's predictions (preservation of generalization)
        #
        # τ = temperature for softmax (τ>1 softer distributions)
        # β = weight on preservation term
        # ================================================================

        # Precompute zero-shot logits (constant across α)
        logits_w1 = logit_scale * all_image_features @ feat_w1.t()  # [N, C]

        kl_divs = []
        for i, alpha in enumerate(alphas):
            feat_interp = feat_gen + alpha * (feat_base - feat_gen)
            feat_interp = feat_interp / feat_interp.norm(dim=-1, keepdim=True)
            logits_alpha = logit_scale * all_image_features @ feat_interp.t()  # [N, C]

            # Compute KL(q_α || q_w1) at each temperature
            # Using τ=1 (raw logits) and τ=2 (softer)
            kl_per_tau = {}
            for tau in [1.0, 2.0]:
                q_alpha = F.softmax(logits_alpha / tau, dim=-1)  # [N, C]
                q_w1_soft = F.softmax(logits_w1 / tau, dim=-1)   # [N, C]
                # KL(q_α || q_w1) = Σ_c q_α(c) * log(q_α(c) / q_w1(c))
                # Add eps to avoid log(0)
                kl = (q_alpha * (torch.log(q_alpha + 1e-10) - torch.log(q_w1_soft + 1e-10))).sum(dim=-1)
                kl_mean = kl.mean().item()  # average over images
                kl_per_tau[tau] = kl_mean
            kl_divs.append(kl_per_tau)

        print(f"\n  --- KL-based α selection ---")
        # Print KL values at each α
        print(f"  {'α':>5} | {'CE':>8} | {'KL(τ=1)':>8} | {'KL(τ=2)':>8}")
        print(f"  {'-'*40}")
        for i, alpha in enumerate(alphas):
            print(f"  {alpha:5.2f} | {val_ces[i]:8.4f} | {kl_divs[i][1.0]:8.4f} | {kl_divs[i][2.0]:8.4f}")

        # Compute score for different β and τ combinations
        print(f"\n  --- KL selection results ---")
        for tau in [1.0, 2.0]:
            kl_arr = np.array([kl_divs[i][tau] for i in range(len(alphas))])
            print(f"  τ={tau:.0f}: CE range=[{val_ces.min():.4f}, {val_ces.max():.4f}], "
                  f"KL range=[{kl_arr.min():.4f}, {kl_arr.max():.4f}]")
            for beta in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
                score = -val_ces - beta * kl_arr
                idx = np.argmax(score)
                print(f"  KL (τ={tau:.0f}, β={beta:>4.1f}):  α={alphas[idx]:.2f} "
                      f"(score={score[idx]:.4f}, -CE={-val_ces[idx]:.4f}, "
                      f"-β·KL={-beta*kl_arr[idx]:.4f})")

        return

    def compute_gradient_conflict(self):
        """Compute gradient conflict γ = cos(∇_c L_CE, ∇_c R) at convergence.

        Measures the cosine similarity between the CE gradient and the
        cosine-score gradient with respect to the learned context c*.
        γ ≈ -1: strong conflict (opposing gradients)
        γ ≈  0: no conflict (orthogonal gradients)

        Averages over the full training set for stable estimation.
        No model updates — purely diagnostic.
        """
        self.set_model_mode("eval")
        ctx = self.model.prompt_mid_learner.ctx  # [N_CTX, D]

        # Accumulate gradients over entire training set
        grad_ce_accum = torch.zeros_like(ctx)
        grad_cos_accum = torch.zeros_like(ctx)
        n_batches = 0

        for batch in tqdm(self.train_loader_x, desc="Computing gradient conflict"):
            image, label = self.parse_batch_train(batch)

            # Shared forward: compute text features from ctx (grad flows to ctx)
            ctx.requires_grad_(True)
            if ctx.grad is not None:
                ctx.grad.zero_()

            prompts = self.model.prompt_mid_learner()
            text_features = self.model.text_encoder(prompts, self.model.tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Image features: detach from graph (frozen encoder, no grad needed)
            with torch.no_grad():
                image_features = self.model.image_encoder(image.type(self.model.dtype))
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logit_scale = self.model.logit_scale.exp()

            # --- CE gradient ---
            logits = logit_scale * image_features @ text_features.t()
            loss_ce = F.cross_entropy(logits, label)

            loss_ce.backward(retain_graph=True)
            grad_ce = ctx.grad.clone().detach()
            ctx.grad.zero_()

            # --- Cosine score gradient (text-only, no images needed) ---
            text_features_old = self.model.ori_embedding.detach().float()
            text_features_old = text_features_old / text_features_old.norm(dim=-1, keepdim=True)

            cos_fn = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
            score = 1.0 - torch.mean(cos_fn(text_features, text_features_old))

            score.backward()
            grad_cos = ctx.grad.clone().detach()
            ctx.grad.zero_()
            ctx.requires_grad_(False)

            grad_ce_accum += grad_ce
            grad_cos_accum += grad_cos
            n_batches += 1

        # Average over batches
        grad_ce_avg = grad_ce_accum / n_batches
        grad_cos_avg = grad_cos_accum / n_batches

        # Flatten to vectors and compute cosine similarity
        g_ce = grad_ce_avg.flatten()
        g_cos = grad_cos_avg.flatten()

        gamma = F.cosine_similarity(g_ce.unsqueeze(0), g_cos.unsqueeze(0)).item()

        # Also compute norms for diagnostics
        norm_ce = g_ce.norm().item()
        norm_cos = g_cos.norm().item()

        print(f"\n=== Gradient Conflict Analysis ===")
        print(f"  γ = cos(∇L_CE, ∇R) = {gamma:.4f}")
        print(f"  ||∇L_CE|| = {norm_ce:.6f}")
        print(f"  ||∇R||    = {norm_cos:.6f}")
        print(f"  Interpretation: {'strong conflict' if gamma < -0.5 else 'moderate conflict' if gamma < -0.2 else 'weak conflict' if gamma < 0 else 'aligned'}")

        return gamma

    def model_inference(self, input):
        return self.model(input)[0]


    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        print(names)

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            # model_file = "model.pth.tar-" + str(epoch)
            # Try both naming patterns
            model_file = "model.pth.tar-" + str(epoch)  # model.pth.tar-100

        for name in names:
            # model_path = osp.join(directory, name, model_file)
            # Check which file exists
            model_path = osp.join(directory, name, model_file)

            if osp.exists(model_path):
                model_path = model_path
            else:
                raise FileNotFoundError(f'Model not found at "{model_path}"')


            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            if "token_midfix" in state_dict:
                del state_dict["token_midfix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)


    def load_model_loop(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = f"model.pth.tar-{epoch}" if epoch is not None else None

        for name in names:
            if model_file:
                model_path = osp.join(directory, name, model_file)
                if not osp.exists(model_path):
                    raise FileNotFoundError('Model not found at "{}"'.format(model_path))

                checkpoint = load_checkpoint(model_path)
                state_dict = checkpoint["state_dict"]
                epoch = checkpoint["epoch"]

                # Ignore fixed token vectors
                if "token_prefix" in state_dict:
                    del state_dict["token_prefix"]

                if "token_suffix" in state_dict:
                    del state_dict["token_suffix"]

                if "token_midfix" in state_dict:
                    del state_dict["token_midfix"]

                # Load model state dict
                print("Loading weights to {} from '{}' (epoch = {})".format(name, model_path, epoch))
                self._models[name].load_state_dict(state_dict, strict=False)




    def load_model_merge(self, directory, lambda_val=None, model_prefix=None):
        if not directory:
            print("Note that load_model_merge() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        
        for name in names:
            # Determine which naming pattern to use
            model_files = []
            detected_prefix = None
            
            if osp.exists(directory):
                import glob
                
                if model_prefix:
                    # Use explicitly specified prefix
                    pattern = osp.join(directory, f"{model_prefix}_*.pth")
                    model_files = glob.glob(pattern)
                    detected_prefix = model_prefix
                    if model_files:
                        print(f"Using specified prefix: Found {len(model_files)} model files with pattern {model_prefix}_*.pth")
                    else:
                        print(f"No model files found with specified prefix {model_prefix}_*.pth")
                else:
                    # Auto-detect available pattern
                    # Try clip_coop_ties_lambda pattern first
                    pattern1 = osp.join(directory, f"clip_coop_ties_lambda_*.pth")
                    files1 = glob.glob(pattern1)
                    
                    # Try clip_kgcoop_ties_lambda pattern
                    pattern2 = osp.join(directory, f"clip_kgcoop_ties_lambda_*.pth")
                    files2 = glob.glob(pattern2)
                    
                    if files1:
                        model_files = files1
                        detected_prefix = "clip_coop_ties_lambda"
                        print(f"Auto-detected: Found {len(files1)} model files with pattern clip_coop_ties_lambda_*.pth")
                    elif files2:
                        model_files = files2
                        detected_prefix = "clip_kgcoop_ties_lambda"
                        print(f"Auto-detected: Found {len(files2)} model files with pattern clip_kgcoop_ties_lambda_*.pth")
            
            if not model_files:
                if model_prefix:
                    print(f"No model files found in {directory} with pattern {model_prefix}_*.pth")
                else:
                    print(f"No model files found in {directory} with either pattern:")
                    print(f"  - clip_coop_ties_lambda_*.pth")
                    print(f"  - clip_kgcoop_ties_lambda_*.pth")
                continue
                
            # Select model based on lambda parameter
            if lambda_val is not None:
                # Look for specific lambda value with detected prefix
                target_file = osp.join(directory, f"{detected_prefix}_{lambda_val}.pth")
                if osp.exists(target_file):
                    model_path = target_file
                    print(f"Using specified lambda model: {osp.basename(model_path)}")
                else:
                    print(f"Specified lambda model {detected_prefix}_{lambda_val}.pth not found")
                    # Fall back to first available model
                    model_files.sort()
                    model_path = model_files[0]
                    print(f"Falling back to: {osp.basename(model_path)}")
            else:
                # Sort model files and use the first one (lowest lambda first)
                model_files.sort()
                model_path = model_files[0]
                print(f"Found {len(model_files)} model files, using: {osp.basename(model_path)}")
            
            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint.get("epoch", "unknown")

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            if "token_midfix" in state_dict:
                del state_dict["token_midfix"]

            # Load model state dict
            print("Loading weights to {} from '{}' (epoch = {})".format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)


    def load_model_merge_dare(self, directory, model_name=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        print(names)

        # By default, the best model is loaded

        for name in names:
            model_path = osp.join(directory, model_name)
            # Check which file exists

            if not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            # epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            if "token_midfix" in state_dict:
                del state_dict["token_midfix"]

            # CRITICAL FIX: Rename parameters to match PromptMidLearner naming
            # DARE models have 'prompts.ctx' but PromptMidLearner expects 'ctx'
            renamed_state_dict = {}
            for key, value in state_dict.items():
                if key == "prompts.ctx":
                    renamed_state_dict["ctx"] = value
                    print(f"  Renamed parameter: {key} -> ctx")
                elif key == "prompts.bias_vectors":
                    renamed_state_dict["bias_vectors"] = value
                    print(f"  Renamed parameter: {key} -> bias_vectors")
                else:
                    # Keep all other parameters with original names
                    renamed_state_dict[key] = value

            print("Loading weights to {} " 'from "{}"'.format(name, model_path))
            # set strict=False
            self._models[name].load_state_dict(renamed_state_dict, strict=False)
