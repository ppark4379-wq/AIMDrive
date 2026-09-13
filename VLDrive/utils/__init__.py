"""Training utilities shared by train.py, evaluate.py and tools/.

runtime                  offline caches, sha256 provenance, numeric adapters,
                         path resolving
vision_encoder_training  the stage-1 encoder training loop
trainer                  the stage-2 VLDrive training / preflight / evaluation loop
training                 pieces both loops share: batching, autocast, seeding,
                         gradient reporting, checkpoints
metrics                  trajectory metrics (ADE, FDE, per-horizon error) and
                         CCDP token-retention checks
"""
