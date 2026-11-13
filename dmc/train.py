import argparse
import torch
import time
import os

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.coop
import trainers.zsclip
import trainers.kgcoop
import trainers.kgcoop_coop_LMC
import trainers.kgcoop_coop_fisher_LMC
import trainers.mma
import trainers.mma_LMC


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    cfg.RESUME_COOP = args.resume_coop


    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """Register the config options used by the trainers in this repository.

    Naming note: every prompt-based trainer here (CoOp, KgCoOp, MergeTune and
    DMC) reads its options from the shared ``cfg.TRAINER.COOP`` namespace,
    which is inherited from the original CoOp codebase. The comments below map
    each option onto the symbol used in the paper.
    """
    from yacs.config import CfgNode as CN

    # ------------------------------------------------------------------
    # Shared prompt-learning options (CoOp / KgCoOp / MergeTune / DMC)
    # ------------------------------------------------------------------
    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16           # number of learnable context tokens (N)
    cfg.TRAINER.COOP.CSC = False          # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = False     # initialise context from these words
    cfg.TRAINER.COOP.PREC = "amp"         # fp16 | fp32 | amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"   # end | middle | front
    cfg.TRAINER.COOP.ALPHA = 1.0          # unused by the shipped trainers; kept for config compatibility

    # ------------------------------------------------------------------
    # MergeTune: single-prompt continued fine-tuning (Eq. 3 in the paper)
    #   L = L_CE(c) + W * R(f_c, f_w1) + W_LMC * E_alpha[ L_CE(interp) ]
    # ------------------------------------------------------------------
    cfg.TRAINER.COOP.W = 8.0              # lambda: weight on the cosine regulariser R
    cfg.TRAINER.COOP.LOSS_TYPE = "cosine" # form of R: "cosine" or "l2"
    cfg.TRAINER.COOP.W_LMC = 1.0          # beta: weight on the LMC path term
    cfg.TRAINER.COOP.NUM_SAMPLES = 5      # alpha samples per step for the LMC expectation
    cfg.TRAINER.COOP.COOP_LMC = True      # anchor the LMC path at the stage-1 checkpoint w2

    # ------------------------------------------------------------------
    # DMC: Decoupled Mode Connectivity (Eq. 8 in the paper)
    #
    # Two contexts are trained jointly and joined by an LMC corridor:
    #   c_spec  - specialisation prompt, trained by L_CE (+ Visual Anchor)
    #   c_gen   - generalisation prompt, trained by the cosine regulariser R
    #
    # Enabled with DPP=True. "DPP" (decoupled prompt pair) is the internal
    # name for DMC and is what the checkpoints and log files use.
    # ------------------------------------------------------------------
    cfg.TRAINER.COOP.DPP = False          # enable the decoupled two-prompt architecture
    cfg.TRAINER.COOP.DPP_W_GEN = 8.0      # lambda_gen: weight on R applied to c_gen
    cfg.TRAINER.COOP.DPP_GEN_MODE = "learned"  # "learned" (default) | "fixed_w1" (ablation)

    # Visual Anchor (Eq. 7): CPI-satisfying KL regulariser applied to c_spec.
    cfg.TRAINER.COOP.VA_W = 0.0           # lambda_VA: weight (0 disables VA)
    cfg.TRAINER.COOP.VA_TAU = 2.0         # tau: softmax temperature in the KL term
    cfg.TRAINER.COOP.VA_AUG_PRESET = "default"   # none | weak | default | strong
    cfg.TRAINER.COOP.VA_CROP_SCALE_MIN = -1.0    # override RandomResizedCrop lower bound (-1 = preset)
    cfg.TRAINER.COOP.VA_JITTER_STRENGTH = -1.0   # ColorJitter magnitude multiplier (-1 = preset)
    cfg.TRAINER.COOP.VA_SINGLE_VIEW = False      # student sees the un-augmented image (single-view objective)

    # ------------------------------------------------------------------
    # Single-prompt loss modifications (Appendix: "Single-Prompt Loss
    # Modifications"). Each is disabled by default; they exist so the
    # appendix table showing that none of them beats the single-prompt
    # ceiling can be reproduced.
    # ------------------------------------------------------------------
    cfg.TRAINER.COOP.PROMPT_FISHER_W = 0.0   # prompt-space EWC penalty weight
    cfg.TRAINER.COOP.KL_PATH_W = 0.0         # KL-divergence path constraint weight
    cfg.TRAINER.COOP.ZSDD_W = 0.0            # zero-shot distribution distillation weight
    cfg.TRAINER.COOP.ZSDD_TAU = 2.0          # ZSDD temperature
    cfg.TRAINER.COOP.W_SCHEDULE = "none"     # schedule on W: none | linear | cosine
    cfg.TRAINER.COOP.W_MIN = 0.0             # schedule floor (0 = auto, W / 3)

    # Fisher-weighted cosine variant (trainer: KgCoOp_COOP_Fisher_LMC).
    cfg.TRAINER.COOP.BETA1 = 0.3             # symmetric path weight
    cfg.TRAINER.COOP.FISHER_SAMPLES = 1024   # samples used to estimate the Fisher diagonal
    cfg.TRAINER.COOP.FISHER_NORM = "max"     # Fisher normalisation: max | per-class

    # Exploratory interpolation/ensembling variants. Not used by any shipped
    # script; retained because the evaluation code paths reference them.
    cfg.TRAINER.COOP.INTERP_TYPE = "linear"  # "linear" or "slerp"
    cfg.TRAINER.COOP.SLERP_PER_TOKEN = False
    cfg.TRAINER.COOP.WSPE_K = 5
    cfg.TRAINER.COOP.WSPE_SPACING = "uniform"
    cfg.TRAINER.COOP.WSPE_TAU = 1.0

    # ------------------------------------------------------------------
    # MMA (multi-modal adapter) baseline and MMA+DMC
    # ------------------------------------------------------------------
    cfg.TRAINER.MMADAPTER = CN()
    cfg.TRAINER.MMADAPTER.TEXT_CTX_INIT = ""
    cfg.TRAINER.MMADAPTER.PREC = "amp"       # fp16 | fp32 | amp
    cfg.TRAINER.MMADAPTER.ADAPTER_START = 4  # first transformer block to adapt
    cfg.TRAINER.MMADAPTER.ADAPTER_END = 12   # last transformer block to adapt
    cfg.TRAINER.MMADAPTER.ADAPTER_DIM = 32   # adapter bottleneck width
    cfg.TRAINER.MMADAPTER.ADAPTER_SCALE = 0.1

    # ------------------------------------------------------------------
    # Options inherited from the upstream CoOp/KgCoOp codebase.
    # ------------------------------------------------------------------
    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16
    cfg.TRAINER.COCOOP.CTX_INIT = False
    cfg.TRAINER.COCOOP.PREC = "amp"

    cfg.TRAINER.COOP_CLIP = CN()
    cfg.TRAINER.COOP_CLIP.N_CTX = 4
    cfg.TRAINER.COOP_CLIP.CTX_INIT = False

    cfg.TRAINER.SAM = CN()
    cfg.TRAINER.SAM.RHO = 0.05
    cfg.TRAINER.SAM.ADAPTIVE = False

    cfg.TRAINER.MODAL = "base2novel"

    cfg.LOSS = CN()
    cfg.LOSS.GM = False
    cfg.LOSS.NAME = ""
    cfg.LOSS.ALPHA = 0.
    cfg.LOSS.T = 1.
    cfg.LOSS.LAMBDA = 1.

    # ------------------------------------------------------------------
    # Base-to-novel protocol
    # ------------------------------------------------------------------
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all | base | new
    cfg.DATASET.NUM_SHOTS = 16             # shots per base class

    cfg.TEST.FINAL_MODEL = "best_val"


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_gradient_conflict:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.compute_gradient_conflict()
        return

    if args.eval_only_dpp_select:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test_dpp_select()
        return

    if args.eval_only_dpp:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test_dpp()
        return

    if args.eval_only_dpp_ablation:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test_dpp_ablation()
        return

    if args.eval_only_wiseft:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test_wiseft()
        return

    if args.eval_only_ttai:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test_ttai(ttai_mlp_path=args.ttai_mlp_path)
        return

    if args.eval_only_wspe:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test_wspe()
        return

    if args.eval_only_merge_dare:
        trainer.load_model_merge_dare(args.model_dir, model_name=args.model_name)
        trainer.test()
        return


    if args.eval_only_merge:
        trainer.load_model_merge(args.model_dir, lambda_val=args.lambda_val, model_prefix=args.model_prefix)
        trainer.test()
        return


    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return


    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir",
                        type=str,
                        default="",
                        help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--resume-coop",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=-1,
                        help="only positive value enables a fixed seed")
    parser.add_argument("--source-domains",
                        type=str,
                        nargs="+",
                        help="source domains for DA/DG")
    parser.add_argument("--target-domains",
                        type=str,
                        nargs="+",
                        help="target domains for DA/DG")
    parser.add_argument("--transforms",
                        type=str,
                        nargs="+",
                        help="data augmentation methods")
    parser.add_argument("--config-file",
                        type=str,
                        default="",
                        help="path to config file")
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer",
                        type=str,
                        default="",
                        help="name of trainer")
    parser.add_argument("--backbone",
                        type=str,
                        default="",
                        help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only",
                        action="store_true",
                        help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument("--load-epoch",
                        type=int,
                        help="load model weights at this epoch for evaluation")
    parser.add_argument("--no-train",
                        action="store_true",
                        help="do not call trainer.train()")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )

    # Eval only merge arguments
    parser.add_argument("--eval-only-merge",
                        action="store_true",
                        help="evaluation only")
    parser.add_argument("--eval-only-merge-dare",
                        action="store_true",
                        help="evaluation only")
    parser.add_argument("--lambda-val",
                        type=float,
                        default=None,
                        help="lambda value for merged model (e.g., 1.6). If not specified, uses the first available model.")
    parser.add_argument("--model-prefix",
                        type=str,
                        default=None,
                        help="Model file prefix. If not specified, auto-detects available pattern.")
    parser.add_argument("--model-name",
                        type=str,
                        default=None,
                        help="Model file name. If not specified, auto-detects available pattern.")
    parser.add_argument("--eval-only-dpp",
                        action="store_true",
                        help="DPP alpha sweep evaluation")
    parser.add_argument("--eval-only-dpp-select",
                        action="store_true",
                        help="DPP alpha selection via base-val proxy")
    parser.add_argument("--eval-only-dpp-ablation",
                        action="store_true",
                        help="DPP ablation: compare f_gen vs w1 vs w2 endpoints")
    parser.add_argument("--eval-only-wiseft",
                        action="store_true",
                        help="WiSE-FT adapter weight interpolation sweep")
    parser.add_argument("--eval-gradient-conflict",
                        action="store_true",
                        help="Compute gradient conflict gamma at MERGETUNE convergence")
    parser.add_argument("--eval-only-ttai",
                        action="store_true",
                        help="TTAI test-time adaptive interpolation evaluation")
    parser.add_argument("--ttai-mlp-path",
                        type=str,
                        default=None,
                        help="Path to pre-trained TTAI MLP. If not given, trains MLP on current train data.")
    parser.add_argument("--eval-only-wspe",
                        action="store_true",
                        help="WSPE ensemble evaluation only")
    args = parser.parse_args()
    main(args)
