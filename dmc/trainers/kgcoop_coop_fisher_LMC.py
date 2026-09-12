"""
FisherMergeTune trainer for CoOp base method (D1 + D2).

Extends KgCoOp_COOP_LMC with:
  D1: Fisher-weighted proximity to zero-shot CLIP (replaces isotropic KgCoOp score)
  D2: Symmetric Task-1 path constraint (Fisher-weighted path toward w1)

Architecture:
  PromptLearner (fixed buffer)  = loaded from Stage 1 CoOp checkpoint (w2)
  PromptMidLearner (learnable)  = continued model (w), initialized from w2

Loss:
  L = L_task + lambda * L_fisher + beta2 * L_lmc2 + beta1 * L_lmc1
  where:
    L_task   = CE(logits_w, labels)                          -- downstream task
    L_fisher = fisher_weighted_distance(feat_w, feat_clip)   -- D1 endpoint
    L_lmc2   = avg_t CE(model(x, interp(feat_w2, feat_w, t)), labels)  -- Task-2 path
    L_lmc1   = avg_alpha alpha^2 * fisher_dist(feat_w, feat_clip)      -- D2 Task-1 path
"""

import os.path as osp
import time
import datetime

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import numpy as np

from .fisher import (
    compute_feature_fisher,
    normalize_fisher,
    fisher_weighted_distance,
    fisher_weighted_path_loss,
    save_fisher,
    load_fisher,
    get_fisher_cache_path,
    get_zero_shot_text_features,
    CUSTOM_TEMPLATES,
)

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    return model


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
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    """Fixed prompt learner — loaded from Stage 1 CoOp, not optimized."""

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize

        if ctx_init:
            temp = "a photo of a"
            ctx_init = temp.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if cfg.TRAINER.COOP.CSC:
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        # Fixed — not optimized
        self.register_buffer("ctx", ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        # Compute zero-shot text features for reference
        clip_model_ = load_clip_to_cpu(cfg)
        clip_model_.cuda()
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_]).cuda()
        with torch.no_grad():
            text_features = clip_model_.encode_text(prompts_)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.text_features = text_features

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts


class PromptMidLearner(nn.Module):
    """Learnable prompt — the continued model w, optimized during training."""

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP_CLIP.N_CTX
        ctx_init = cfg.TRAINER.COOP_CLIP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize

        if ctx_init:
            temp = "a photo of a"
            ctx_init = temp.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if cfg.TRAINER.COOP.CSC:
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        bias_vectors = torch.empty(1, 512, dtype=dtype)
        nn.init.normal_(bias_vectors, std=0.02)
        self.bias_vectors = nn.Parameter(bias_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        # Zero-shot text features (same as PromptLearner)
        clip_model_ = load_clip_to_cpu(cfg)
        clip_model_.cuda()
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_]).cuda()
        with torch.no_grad():
            text_features = clip_model_.encode_text(prompts_)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.text_features = text_features

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        elif self.class_token_position == "middle":
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
                    [prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [prefix_i, class_i, ctx_i, suffix_i], dim=1
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.prompt_mid_learner = PromptMidLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.ori_embedding = self.prompt_learner.text_features
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, sign=None, text_features_mid=None):
        if sign is None:
            # Main forward: use PromptMidLearner (the continued model w)
            prompts = self.prompt_mid_learner()
            image_features = self.image_encoder(image.type(self.dtype))
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(prompts, tokenized_prompts)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

            return logits, text_features

        else:
            # LMC forward: use provided interpolated text features
            image_features = self.image_encoder(image.type(self.dtype))
            text_features = text_features_mid

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

            return logits


@TRAINER_REGISTRY.register()
class KgCoOp_COOP_Fisher_LMC(TrainerX):
    """
    FisherMergeTune for CoOp (D1 + D2).

    Compared to KgCoOp_COOP_LMC:
      - D1: Fisher-weighted distance replaces isotropic KgCoOp cosine score
      - D2: Adds symmetric Task-1 path (Fisher-weighted toward w1)
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP (FisherMergeTune)")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.lam = cfg.TRAINER.COOP.W  # lambda: Fisher proximity weight
        self.beta2 = cfg.TRAINER.COOP.W_LMC  # beta2: Task-2 LMC weight
        self.beta1 = cfg.TRAINER.COOP.BETA1  # beta1: Task-1 path weight (D2)

        # --- Load Stage 1 CoOp checkpoint (w2) ---
        if self.cfg.RESUME_COOP and self.cfg.RESUME_COOP != "None":
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

            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]
            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            self.model.prompt_learner.load_state_dict(state_dict, strict=False)

            # Initialize prompt_mid_learner with loaded weights (w initialized at w2)
            print("Initializing prompt_mid_learner with loaded prompt_learner weights")
            self.model.prompt_mid_learner.ctx.data.copy_(self.model.prompt_learner.ctx.data)

        # --- Freeze everything except prompt_mid_learner.ctx ---
        print("Freezing all parameters except prompt_mid_learner.ctx")
        for name, param in self.model.named_parameters():
            if "prompt_mid_learner.ctx" not in name:
                param.requires_grad_(False)
            else:
                print(f"  Trainable: {name}")

        self.model.to(self.device)

        # --- Compute or load Fisher (D1 core) ---
        self._setup_fisher(clip_model, classnames)

        # --- Store zero-shot text features for Fisher distance computation ---
        self.feat_w1 = self.model.ori_embedding.detach()  # [C, 512]

        # --- Store fine-tuned (w2) text features for Task-2 LMC ---
        with torch.no_grad():
            prompts_w2 = self.model.prompt_learner()
            self.feat_w2 = self.model.text_encoder(
                prompts_w2, self.model.tokenized_prompts
            ).detach()
            self.feat_w2 = self.feat_w2 / self.feat_w2.norm(dim=-1, keepdim=True)

        # --- Optimizer ---
        self.optim = build_optimizer(self.model.prompt_mid_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_mid_learner", self.model.prompt_mid_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # --- Alpha values for LMC path interpolation ---
        n_alpha = cfg.TRAINER.COOP.NUM_SAMPLES
        self.alphas = np.arange(0.1, 1.01, 1.0 / float(n_alpha))

    def _setup_fisher(self, clip_model, classnames):
        """Compute or load cached Fisher at zero-shot checkpoint."""
        cfg = self.cfg
        cache_path = get_fisher_cache_path(
            cfg.OUTPUT_DIR, cfg.DATASET.NAME, cfg.SEED, cfg.MODEL.BACKBONE.NAME
        )

        if osp.exists(cache_path):
            self.fisher_diag = load_fisher(cache_path, device=self.device)
        else:
            print(f"Computing diagonal Fisher ({cfg.TRAINER.COOP.FISHER_SAMPLES} samples)...")
            # Need a fresh CLIP model on device for Fisher computation
            clip_for_fisher = load_clip_to_cpu(cfg)
            if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
                clip_for_fisher.float()
            clip_for_fisher.to(self.device)

            self.fisher_diag = compute_feature_fisher(
                clip_model=clip_for_fisher,
                dataloader=self.train_loader_x,
                classnames=classnames,
                dataset_name=cfg.DATASET.NAME,
                device=self.device,
                n_samples=cfg.TRAINER.COOP.FISHER_SAMPLES,
            )
            self.fisher_diag = normalize_fisher(
                self.fisher_diag, mode=cfg.TRAINER.COOP.FISHER_NORM
            )
            save_fisher(self.fisher_diag, cache_path)

            # Cleanup
            del clip_for_fisher
            torch.cuda.empty_cache()

        print(f"Fisher shape: {self.fisher_diag.shape}, "
              f"min: {self.fisher_diag.min():.6f}, "
              f"max: {self.fisher_diag.max():.6f}, "
              f"mean: {self.fisher_diag.mean():.6f}")

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
        else:
            loss, loss_summary = self._compute_loss(image, label)
            self.model_backward_and_update(loss)

        if (self.batch_idx + 1) == self.num_batches:
            self.sched.step()

        return loss_summary

    def _compute_loss(self, image, label):
        """Compute the 4-term FisherMergeTune loss."""

        # --- Term 1: Downstream task loss at w ---
        logits_w, feat_w = self.model(image)
        loss_task = F.cross_entropy(logits_w, label)

        # --- Term 2 (D1): Fisher-weighted endpoint proximity ---
        loss_fisher = fisher_weighted_distance(feat_w, self.feat_w1, self.fisher_diag)

        # --- Term 3: Task-2 LMC path (w2 -> w), same as original ---
        total_loss_lmc2 = 0.0
        for t in self.alphas:
            feat_interp = self.feat_w2 + t * (feat_w - self.feat_w2)
            output_lmc = self.model(image, sign="LMC", text_features_mid=feat_interp)
            total_loss_lmc2 += F.cross_entropy(output_lmc, label) / len(self.alphas)

        # --- Term 4 (D2): Symmetric Task-1 path (Fisher-weighted toward w1) ---
        total_loss_lmc1 = 0.0
        for alpha in self.alphas:
            total_loss_lmc1 += fisher_weighted_path_loss(
                feat_w, self.feat_w1, self.fisher_diag, alpha
            ) / len(self.alphas)

        # --- Total loss ---
        loss = (
            loss_task
            + self.lam * loss_fisher       # D1: Fisher endpoint
            + self.beta2 * total_loss_lmc2  # Task-2 LMC path (original)
            + self.beta1 * total_loss_lmc1  # D2: symmetric Task-1 path
        )

        loss_summary = {
            "loss": loss.item(),
            "loss_task": loss_task.item(),
            "loss_fisher": loss_fisher.item(),
            "loss_lmc2": total_loss_lmc2.item() if isinstance(total_loss_lmc2, torch.Tensor) else total_loss_lmc2,
            "loss_lmc1": total_loss_lmc1.item() if isinstance(total_loss_lmc1, torch.Tensor) else total_loss_lmc1,
            "acc": compute_accuracy(logits_w, label)[0].item(),
        }

        return loss, loss_summary

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

    def model_inference(self, input):
        return self.model(input)[0]

    def after_epoch(self):
        """Skip best-model saving during warmup epochs.

        During warmup (epoch < WARMUP_EPOCH), the LR is tiny (1e-5) so
        the model is essentially the CoOp initialization.  Saving it as
        'best' creates a degenerate checkpoint on fine-grained datasets
        where training can never surpass the initialization accuracy.
        """
        warmup = self.cfg.OPTIM.WARMUP_EPOCH
        if warmup > 0 and (self.epoch + 1) <= warmup:
            # Still save periodic checkpoints, just don't track best_val
            return
        super().after_epoch()

    def after_train(self):
        print("Finished training in KgCoOp_COOP_Fisher_LMC")
        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            self.test(split="val")

        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")
        self.close_writer()

    @torch.no_grad()
    def test(self, split=None):
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
            print(f"Do evaluation on {split} set")
        else:
            data_loader = self.test_loader
            print("Do evaluation on test set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]
            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]
            if "token_midfix" in state_dict:
                del state_dict["token_midfix"]

            print(f'Loading weights to {name} from "{model_path}" (epoch = {epoch})')
            self._models[name].load_state_dict(state_dict, strict=False)
