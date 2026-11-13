"""DMC and MergeTune on top of the MMA multi-modal adapter baseline.

Registers the ``MMA_LMC`` trainer. This is the adapter-based counterpart of
``kgcoop_coop_LMC``: instead of interpolating between two text prompts, it
operates on the multi-modal adapter parameters of MMA, and the LMC corridor is
again enforced in text-feature space.

Set ``TRAINER.COOP.DPP = True`` for DMC and ``False`` for the single-prompt
MergeTune baseline. Weights follow the same names as the prompt trainers:
``DPP_W_GEN`` (lambda_gen), ``W_LMC`` (beta) and ``VA_W`` (lambda_VA).

Stage-1 MMA checkpoints are supplied with ``--resume-coop``.
"""

import os
import os.path as osp
import copy

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from collections import OrderedDict

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import numpy as np
from tqdm import tqdm
from .zsclip import ZeroshotCLIP
from torchvision import transforms as T

_tokenizer = _Tokenizer()


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
    #"EuroSAT": "a photo of a {}.",
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

    def forward(self, prompts, tokenized_prompts, retrun_adapater_func=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        if retrun_adapater_func == None:
            x = self.transformer(x)
        else:
            x = self.transformer([x, retrun_adapater_func])
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class AdapterLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        self.n_cls = len(classnames)
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self._build_text_embedding(cfg, classnames, clip_model)

        # build multi-modal adapter
        self.text_adapter_func = lambda x: self.return_text_adapter(index=x)
        self.text_adapter = self._build_adapter(
            clip_model.ln_final.weight.shape[0], 
            len(clip_model.transformer.resblocks), 
            cfg.TRAINER.MMADAPTER.ADAPTER_START,
            cfg.TRAINER.MMADAPTER.ADAPTER_END,
            cfg.TRAINER.MMADAPTER.ADAPTER_DIM,
            clip_model.dtype
        )
        
        self.visual_adapter_func = lambda x: self.return_visual_adapter(index=x)
        self.visual_adapter = self._build_adapter(
            clip_model.visual.ln_post.weight.shape[0],
            len(clip_model.visual.transformer.resblocks), 
            cfg.TRAINER.MMADAPTER.ADAPTER_START,
            cfg.TRAINER.MMADAPTER.ADAPTER_END,
            cfg.TRAINER.MMADAPTER.ADAPTER_DIM,
            clip_model.dtype
        )

        self.shared_adapter = self._build_adapter(
            cfg.TRAINER.MMADAPTER.ADAPTER_DIM,
            len(clip_model.visual.transformer.resblocks), 
            cfg.TRAINER.MMADAPTER.ADAPTER_START,
            cfg.TRAINER.MMADAPTER.ADAPTER_END,
            cfg.TRAINER.MMADAPTER.ADAPTER_DIM,
            clip_model.dtype
        )
        self.adapter_scale = float(cfg.TRAINER.MMADAPTER.ADAPTER_SCALE)

    def return_text_adapter(self, index):
        return self.text_adapter[index], self.shared_adapter[index], self.adapter_scale

    def return_visual_adapter(self, index):
        return self.visual_adapter[index], self.shared_adapter[index], self.adapter_scale


    def _build_text_embedding(self, cfg, classnames, clip_model):
        dtype = clip_model.dtype
        text_ctx_init = cfg.TRAINER.MMADAPTER.TEXT_CTX_INIT

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [text_ctx_init + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_embedding", embedding)
        self.register_buffer("tokenized_prompts", tokenized_prompts)


    def _build_adapter(self, d_model, n_layers, l_start, l_end, mid_dim, dtype):

        adapter = [None] * (n_layers + 1)
        for i in range(l_start, l_end+1):
            if mid_dim == d_model:
                adapter[i] = nn.Sequential(
                    nn.Linear(d_model, mid_dim),
                    nn.ReLU()
                )
            else:
                adapter[i] = nn.Sequential(OrderedDict([
                    ("down", nn.Sequential(nn.Linear(d_model, mid_dim), nn.ReLU())),
                    ("up", nn.Linear(mid_dim, d_model))
                ]))
        adapter = nn.ModuleList([a for a in adapter])
        for m in adapter.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

        if dtype == torch.float16:
            for m in adapter.modules():
                m.half()
    
        return adapter
    
    def forward(self):
        embedding = self.token_embedding
        if self.text_adapter[0] is not None:
            token_embedding = self.text_adapter[0].down(embedding)
            shared_adapter = self.shared_adapter[0]
            token_embedding = shared_adapter(token_embedding)
            token_embedding = self.text_adapter[0].up(token_embedding)
            embedding = embedding + self.adapter_scale * token_embedding
        return embedding, self.text_adapter_func, self.visual_adapter_func


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.adapter_learner = AdapterLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.adapter_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.text_features_for_inference = None

        # DPP: second adapter set for generalization
        if cfg.TRAINER.COOP.DPP:
            self.adapter_gen_learner = AdapterLearner(cfg, classnames, clip_model)

    def encode_text(self, prompts, tokenized_prompts, text_adapter_func=None):
        if text_adapter_func is not None:
            text_features = self.text_encoder(
                prompts, tokenized_prompts, text_adapter_func
            )
        else:
            text_features = self.text_encoder(
                prompts, tokenized_prompts
            )
        return text_features
    
    def encode_image(self, image, visual_adapter_func=None):
        if visual_adapter_func is not None:
            image_features = self.image_encoder(
                [image.type(self.dtype), visual_adapter_func]
            )
        else:
            image_features = self.image_encoder(
                image.type(self.dtype)
            )
        return image_features

    def forward(self, image, sign=None, text_features_mid=None):
        if sign == 'MMA':  # main loss is from combined prompt_mid_learner
            token_embedding, text_adapter_func, visual_adapter_func = self.adapter_learner()
            tokenized_prompts = self.tokenized_prompts

            if self.adapter_learner.training:
                text_features = self.encode_text(
                    token_embedding, tokenized_prompts, text_adapter_func
                )
            else:
                if self.text_features_for_inference is None:
                    self.text_features_for_inference = self.encode_text(
                        token_embedding, tokenized_prompts, text_adapter_func
                    )   
                text_features = self.text_features_for_inference

            image_features = self.encode_image(image, visual_adapter_func)

            text_features = F.normalize(text_features, dim=-1)
            image_features = F.normalize(image_features, dim=-1)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

            return logits

        elif sign == 'LMC_MMA':
            # give the text_features_mid to the text_encoder
            token_embedding, text_adapter_func, visual_adapter_func = self.adapter_learner()
            tokenized_prompts = self.tokenized_prompts

            if self.adapter_learner.training:
                text_features = text_features_mid
            else:
                if self.text_features_for_inference is None:
                    self.text_features_for_inference = self.encode_text(
                        token_embedding, tokenized_prompts, text_adapter_func
                    )   
                text_features = self.text_features_for_inference

            image_features = self.encode_image(image, visual_adapter_func)

            text_features = F.normalize(text_features, dim=-1)
            image_features = F.normalize(image_features, dim=-1)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

            return logits

            

@TRAINER_REGISTRY.register()
class MMA_LMC(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
        
        print("Building original CLIP for zero-shot text features")
        _zs = ZeroshotCLIP(cfg)
        # Extract text features and free the full CLIP model
        class _ZSFeatures:
            pass
        self.clip_orig_model = _ZSFeatures()
        self.clip_orig_model.text_features = _zs.text_features.detach().clone()
        del _zs
        torch.cuda.empty_cache()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        # self.w = cfg.TRAINER.COOP.W

        if self.cfg.RESUME_COOP and self.cfg.RESUME_COOP != 'None':
            print(f"Loading pretrained MMAdapter adapter_learner from {self.cfg.RESUME_COOP}")
            
            # Find the maximum epoch checkpoint
            adapter_dir = osp.join(self.cfg.RESUME_COOP, "adapter_learner")
            if osp.exists(adapter_dir):
                import glob
                checkpoint_files = glob.glob(osp.join(adapter_dir, "model.pth.tar-*"))
                if checkpoint_files:
                    # Extract epoch numbers and find maximum
                    epochs = [int(f.split("-")[-1]) for f in checkpoint_files]
                    max_epoch = max(epochs)
                    checkpoint_path = osp.join(adapter_dir, f"model.pth.tar-{max_epoch}")
                    print(f"Found checkpoint at epoch {max_epoch}: {checkpoint_path}")
                else:
                    # Try model-best.pth.tar as fallback
                    checkpoint_path = osp.join(adapter_dir, "model-best.pth.tar")
                    if not osp.exists(checkpoint_path):
                        raise FileNotFoundError(f"No checkpoint files found in {adapter_dir}")
            else:
                raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
            
            if not osp.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
            
            
            checkpoint = load_checkpoint(checkpoint_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            
            # Ignore fixed token vectors
            if "token_embedding" in state_dict:
                del state_dict["token_embedding"]
            if "tokenized_prompts" in state_dict:
                del state_dict["tokenized_prompts"]

            self.model.adapter_learner.load_state_dict(state_dict, strict=False)

        # --- DPP: initialize adapter_gen from same checkpoint ---
        self.dpp = cfg.TRAINER.COOP.DPP
        if self.dpp:
            # ---------- ADAPTER-INIT ABLATION (env-var override) ----------
            # DMC endpoint-collapse investigation: skip copying the trained
            # base adapter into adapter_gen so it stays at its fresh Kaiming
            # init. With adapter_scale small, the residual is ~0 and the text
            # branch is effectively zero-shot CLIP -- a "CLIP-only" init for
            # the generalization endpoint. Enable by exporting
            #   DMC_GEN_INIT_FRESH=1
            _fresh_init = os.environ.get("DMC_GEN_INIT_FRESH", "0") == "1"
            if self.cfg.RESUME_COOP and self.cfg.RESUME_COOP != 'None':
                if _fresh_init:
                    print("DPP [ablation-B]: adapter_gen_learner kept at "
                          "fresh Kaiming init (~CLIP-only text features); "
                          "skipping copy from adapter_learner "
                          "(env DMC_GEN_INIT_FRESH=1)")
                else:
                    print("DPP: Initializing adapter_gen_learner from same checkpoint")
                    self.model.adapter_gen_learner.load_state_dict(
                        self.model.adapter_learner.state_dict(), strict=False
                    )
            # -------------------------------------------------------------

            # ---------- ADAPTER-CAPACITY ABLATION (env-var override) ----------
            # DMC endpoint-collapse investigation: override the residual scale
            # of the generalization adapter without touching the base adapter.
            # Canonical adapter_scale is 1e-3 (see configs/trainers/MMA_LMC/
            # vit_b16_ep5.yaml). Set DMC_GEN_ADAPTER_SCALE=1e-2 or 1e-1 in the
            # launcher environment to enlarge only adapter_gen.
            _gen_scale = os.environ.get("DMC_GEN_ADAPTER_SCALE")
            if _gen_scale is not None:
                _gs = float(_gen_scale)
                _orig = self.model.adapter_gen_learner.adapter_scale
                self.model.adapter_gen_learner.adapter_scale = _gs
                print(f"DPP [ablation-A]: adapter_gen.adapter_scale = "
                      f"{_orig:g} -> {_gs:g} (env DMC_GEN_ADAPTER_SCALE)")
            # -------------------------------------------------------------------

        print("Turning off gradients in both the image and the text encoder")

        # Determine which parameters should be trainable
        adapter_keywords = {"text_adapter", "visual_adapter", "shared_adapter"}
        for name, param in self.model.named_parameters():
            is_adapter = any(kw in name for kw in adapter_keywords)
            if not is_adapter:
                param.requires_grad_(False)

        # Double check
        num_trainable_params = 0
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
                num_trainable_params += param.data.nelement()
        print(f"Parameters to be updated: {sorted(enabled)}")
        print(f"Number of trainable parameters: {num_trainable_params}")

        self.model.to(self.device)

        if self.dpp:
            # DPP: single optimizer for both adapter sets
            combined_params = [
                {"params": self.model.adapter_learner.parameters()},
                {"params": self.model.adapter_gen_learner.parameters()},
            ]
            self.optim = torch.optim.SGD(
                combined_params, lr=cfg.OPTIM.LR,
                momentum=0.9, weight_decay=5e-4
            )
            self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
            self.register_model("adapter_learner", self.model.adapter_learner, self.optim, self.sched)
            self.register_model("adapter_gen_learner", self.model.adapter_gen_learner, self.optim, self.sched)
            self.dpp_w_gen = cfg.TRAINER.COOP.DPP_W_GEN
            print(f"DPP: adapter_base=adapter_learner, adapter_gen=adapter_gen_learner, W_gen={self.dpp_w_gen}")
        else:
            self.optim = build_optimizer(self.model.adapter_learner, cfg.OPTIM)
            self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
            self.register_model("adapter_learner", self.model.adapter_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # --- Visual Anchor ---
        self.va_w = cfg.TRAINER.COOP.VA_W
        self.va_tau = cfg.TRAINER.COOP.VA_TAU
        if self.va_w > 0:
            feat_w1 = self.clip_orig_model.text_features.detach()
            self.va_feat_w1 = (feat_w1 / feat_w1.norm(dim=-1, keepdim=True)).to(self.device)
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
            print(f"Visual Anchor: tau={self.va_tau}, weight={self.va_w}")

        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)

    def calculate_line_loss(self, feature_mid, image):
        # text_features = self.model.text_encoder(feature_mid, self.model.tokenized_prompts)
        # self.model.prompt_mid_learner.ctx.data.copy_(feature_mid)
        logits = self.model(image, sign='LMC_MMA', text_features_mid=feature_mid)
        return logits



    def _compute_loss_dpp(self, image, label):
        """DPP loss for MMA: two adapter sets with decoupled objectives."""
        # --- adapter_base: CE loss ---
        token_emb_base, text_adapter_base, vis_adapter_base = self.model.adapter_learner()
        tokenized_prompts = self.model.tokenized_prompts

        feat_base = self.model.encode_text(token_emb_base, tokenized_prompts, text_adapter_base)
        feat_base = F.normalize(feat_base, dim=-1)

        image_features = self.model.encode_image(image, vis_adapter_base)
        image_features = F.normalize(image_features, dim=-1)

        logit_scale = self.model.logit_scale.exp()
        logits_base = logit_scale * image_features @ feat_base.t()
        loss_ce = F.cross_entropy(logits_base, label)

        # --- adapter_gen: cosine score to w1 ---
        token_emb_gen, text_adapter_gen, _ = self.model.adapter_gen_learner()
        feat_gen = self.model.encode_text(token_emb_gen, tokenized_prompts, text_adapter_gen)
        feat_gen = F.normalize(feat_gen, dim=-1)

        feat_w1 = self.clip_orig_model.text_features
        feat_w1 = F.normalize(feat_w1, dim=-1)

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
        score_gen = 1.0 - torch.mean(cos(feat_gen, feat_w1))

        loss = loss_ce + self.dpp_w_gen * score_gen

        # --- LMC path: feat_gen -> feat_base ---
        loss_lmc_val = 0.0
        if self.cfg.TRAINER.COOP.COOP_LMC:
            line_samples = np.arange(0.1, 1.01, 1.0 / float(self.cfg.TRAINER.COOP.NUM_SAMPLES))
            total_loss_LMC = 0.0
            for t in line_samples:
                feat_mid = feat_gen + (feat_base - feat_gen) * t
                feat_mid = F.normalize(feat_mid, dim=-1)
                logits_mid = logit_scale * image_features @ feat_mid.t()
                total_loss_LMC += F.cross_entropy(logits_mid, label) / len(line_samples)

            loss = loss + self.cfg.TRAINER.COOP.W_LMC * total_loss_LMC
            loss_lmc_val = total_loss_LMC.item()

        # --- Visual Anchor on adapter_base ---
        loss_va_val = 0.0
        if self.va_w > 0:
            tau = self.va_tau
            logit_scale_va = logit_scale.float()
            with torch.no_grad():
                # Teacher view: vanilla CLIP encoder (no adapters)
                x_a = self.va_aug(image)
                v_a = self.model.encode_image(x_a).float()
                v_a = F.normalize(v_a, dim=-1)
                del x_a
                s_w1 = logit_scale_va * v_a @ self.va_feat_w1.t()
                p_w1_va = F.softmax(s_w1 / tau, dim=-1)
                del v_a, s_w1

                # Student view: vanilla CLIP encoder (no adapters)
                x_b = image if self.cfg.TRAINER.COOP.VA_SINGLE_VIEW else self.va_aug(image)
                v_b = self.model.encode_image(x_b).float()
                v_b = F.normalize(v_b, dim=-1)
                del x_b

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

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.COOP.PREC

        # DPP mode: use decoupled loss
        if self.dpp:
            if prec == "amp":
                with autocast():
                    loss, loss_summary = self._compute_loss_dpp(image, label)
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                loss, loss_summary = self._compute_loss_dpp(image, label)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            if (self.batch_idx + 1) == self.num_batches:
                self.sched.step()
            return loss_summary

        # Original MMA_LMC forward_backward
        if prec == "amp":
            with autocast():
                output = self.model(image, sign='MMA')
                loss_main_clip = F.cross_entropy(output, label)

                if self.cfg.TRAINER.COOP.COOP_LMC:
                    line_samples = np.arange(0.1, 1.01, 1.0 / float(self.cfg.TRAINER.COOP.NUM_SAMPLES))

                    # feature_start is MMA adapter weight generated text_features, feature_end is clip generated text_features
                    token_embedding, text_adapter_func, visual_adapter_func = self.model.adapter_learner()
                    tokenized_prompts = self.model.tokenized_prompts

                    if self.model.adapter_learner.training:
                        text_features = self.model.encode_text(
                            token_embedding, tokenized_prompts, text_adapter_func
                        )
                    else:
                        if self.model.text_features_for_inference is None:
                            self.model.text_features_for_inference = self.model.encode_text(
                                token_embedding, tokenized_prompts, text_adapter_func
                            )
                        text_features = self.model.text_features_for_inference

                    feature_start = text_features
                    feature_end = self.clip_orig_model.text_features
                    
                    total_loss_LMC = 0
                    for t in line_samples:
                        feature_mid = (feature_start + (feature_end - feature_start) * t)
                        output_LMC = self.calculate_line_loss(feature_mid, image)
                        loss_LMC = nn.CrossEntropyLoss()(output_LMC, label)
                        total_loss_LMC += loss_LMC / len(line_samples)

                    loss = loss_main_clip + self.cfg.TRAINER.COOP.W_LMC * total_loss_LMC
                else:
                    print("COOP_LMC is False")
                    loss = loss_main_clip

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image, sign='MMA')
            loss_main_clip = F.cross_entropy(output, label)

            if self.cfg.TRAINER.COOP.COOP_LMC:
                line_samples = np.arange(0.1, 1.01, 1.0 / float(self.cfg.TRAINER.COOP.NUM_SAMPLES))

                # feature_start is MMA adapter weight generated text_features, feature_end is clip generated text_features
                token_embedding, text_adapter_func, visual_adapter_func = self.model.adapter_learner()
                tokenized_prompts = self.model.tokenized_prompts

                if self.model.adapter_learner.training:
                    text_features = self.model.encode_text(
                        token_embedding, tokenized_prompts, text_adapter_func
                    )
                else:
                    if self.model.text_features_for_inference is None:
                        self.model.text_features_for_inference = self.model.encode_text(
                            token_embedding, tokenized_prompts, text_adapter_func
                        )
                    text_features = self.model.text_features_for_inference

                feature_start = text_features
                feature_end = self.clip_orig_model.text_features
                
                total_loss_LMC = 0
                for t in line_samples:
                    feature_mid = (feature_start + (feature_end - feature_start) * t)
                    output_LMC = self.calculate_line_loss(feature_mid, image)
                    loss_LMC = nn.CrossEntropyLoss()(output_LMC, label)
                    total_loss_LMC += loss_LMC / len(line_samples)

                loss = loss_main_clip + self.cfg.TRAINER.COOP.W_LMC * total_loss_LMC
            else:
                print("COOP_LMC is False")
                loss = loss_main_clip
                total_loss_LMC = 0

            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "loss_main_clip": loss_main_clip.item(),
            "loss_LMC": total_loss_LMC.item()*self.cfg.TRAINER.COOP.W_LMC,
            "weight_LMC": self.cfg.TRAINER.COOP.W_LMC,
            "acc": compute_accuracy(output, label)[0].item(),
        }

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

    def model_inference(self, input, sign=None):
        return self.model(input, sign=sign)

    @torch.no_grad()
    def test_dpp(self, split=None):
        """DPP alpha sweep for MMA: interpolate text features from adapter_gen to adapter_base."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT
        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            data_loader = self.test_loader
        print(f"DPP alpha sweep on {split} set")

        # Precompute text features for both adapter sets
        token_emb_base, text_adapter_base, vis_adapter_base = self.model.adapter_learner()
        feat_base = self.model.encode_text(token_emb_base, self.model.tokenized_prompts, text_adapter_base).float()
        feat_base = F.normalize(feat_base, dim=-1)

        token_emb_gen, text_adapter_gen, _ = self.model.adapter_gen_learner()
        feat_gen = self.model.encode_text(token_emb_gen, self.model.tokenized_prompts, text_adapter_gen).float()
        feat_gen = F.normalize(feat_gen, dim=-1)

        logit_scale = self.model.logit_scale.exp().float()

        # Diagnostics
        feat_w1 = self.clip_orig_model.text_features.float()
        feat_w1 = F.normalize(feat_w1, dim=-1)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
        print(f"  cos(gen, w1)={cos(feat_gen, feat_w1).mean():.4f}, "
              f"cos(base, w1)={cos(feat_base, feat_w1).mean():.4f}, "
              f"cos(gen, base)={cos(feat_gen, feat_base).mean():.4f}")

        # Precompute all image features (using adapter_base's visual adapters)
        all_image_features = []
        all_labels = []
        for batch in tqdm(data_loader, desc="Encoding images"):
            input, label = self.parse_batch_test(batch)
            image_features = self.model.encode_image(input, vis_adapter_base).float()
            image_features = F.normalize(image_features, dim=-1)
            all_image_features.append(image_features)
            all_labels.append(label)
        all_image_features = torch.cat(all_image_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Alpha sweep
        alphas = np.arange(0.0, 1.01, 0.05)
        print(f"\n{'alpha':>6} | {'accuracy':>8}")
        print("-" * 20)

        best_alpha, best_acc = 0.0, 0.0
        results_per_alpha = []
        for alpha in alphas:
            feat_interp = feat_gen + alpha * (feat_base - feat_gen)
            feat_interp = F.normalize(feat_interp, dim=-1)
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
    def test_wiseft(self, split=None):
        """WiSE-FT style adapter weight interpolation.

        Scales adapter outputs by alpha: adapter(alpha) = alpha * adapter_trained.
        At alpha=0: zero-shot CLIP (no adapter contribution).
        At alpha=1: full MMA.
        Traces a Pareto curve in adapter weight space without retraining.
        """
        self.set_model_mode("eval")

        if split is None:
            split = self.cfg.TEST.SPLIT
        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            data_loader = self.test_loader
        print(f"WiSE-FT adapter sweep on {split} set")

        # Save original adapter_scale
        orig_scale = self.model.adapter_learner.adapter_scale
        print(f"  Original adapter_scale: {orig_scale}")

        # Precompute all image features at each alpha
        # We need raw images so we encode per-alpha
        all_images = []
        all_labels = []
        for batch in tqdm(data_loader, desc="Loading batches"):
            input, label = self.parse_batch_test(batch)
            all_images.append(input)
            all_labels.append(label)
        all_labels = torch.cat(all_labels, dim=0)

        alphas = np.arange(0.0, 1.01, 0.05)
        print(f"\n{'alpha':>6} | {'accuracy':>8}")
        print("-" * 20)

        best_alpha, best_acc = 0.0, 0.0
        results_per_alpha = []
        for alpha in alphas:
            # Set adapter_scale to alpha * original
            scaled = alpha * orig_scale
            self.model.adapter_learner.adapter_scale = scaled

            # Must clear cached text features since scale changed
            self.model.text_features_for_inference = None

            # Evaluate
            all_preds = []
            for images in all_images:
                output = self.model(images, sign='MMA')
                preds = output.argmax(dim=1)
                all_preds.append(preds)
            all_preds = torch.cat(all_preds, dim=0)
            acc = (all_preds == all_labels).float().mean().item() * 100.0
            results_per_alpha.append((alpha, acc))
            print(f"{alpha:6.2f} | {acc:8.2f}")
            if acc > best_acc:
                best_acc = acc
                best_alpha = alpha

        # Restore original scale
        self.model.adapter_learner.adapter_scale = orig_scale
        self.model.text_features_for_inference = None

        print(f"\nBest: alpha={best_alpha:.2f}, accuracy={best_acc:.2f}%")
        print(f"Endpoints: alpha=0 (zero-shot)={results_per_alpha[0][1]:.2f}%, "
              f"alpha=1 (full MMA)={results_per_alpha[-1][1]:.2f}%")
        return best_acc

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        # self.evaluator_MMA.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input, sign='MMA')
            self.evaluator.process(output, label)



        results = self.evaluator.evaluate()
        # results_MMA = self.evaluator_MMA.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        # for k, v in results_MMA.items():
        #     tag = f"{split}/{k}"
        #     self.write_scalar(tag, v, self.epoch)

        # return list(results.values())[0], list(results_MMA.values())[0]
        return list(results.values())[0]




    def load_model(self, directory, epoch=None):

        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_embedding" in state_dict:
                del state_dict["token_embedding"]
            if "tokenized_prompts" in state_dict:
                del state_dict["tokenized_prompts"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)




    def load_model_loop(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch_loaded = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_embedding" in state_dict:
                del state_dict["token_embedding"]
            if "tokenized_prompts" in state_dict:
                del state_dict["tokenized_prompts"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch_loaded))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
            
            # Set model to eval mode
            self._models[name].eval()
            
            # Verify model state
            print(f"Model {name} state after loading: {self._models[name].training}")
            print(f"Model {name} parameters: {sum(p.numel() for p in self._models[name].parameters())}")
