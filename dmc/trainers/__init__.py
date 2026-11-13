"""Trainers for DMC and its baselines.

Stage 1 (fine-tune a PEFT baseline on base classes):
    coop        CoOp prompt learning
    kgcoop      KgCoOp prompt learning
    mma         MMA multi-modal adapters
    zsclip      zero-shot CLIP (no training)

Stage 2 (continued fine-tuning on top of a stage-1 checkpoint):
    kgcoop_coop_LMC         MergeTune and DMC for CoOp / KgCoOp
    mma_LMC                 MergeTune and DMC for MMA
    kgcoop_coop_fisher_LMC  Fisher-weighted cosine ablation (appendix)

Each module registers its trainer with Dassl's TRAINER_REGISTRY; selecting one
is done with the ``--trainer`` flag of ``train.py``.
"""
