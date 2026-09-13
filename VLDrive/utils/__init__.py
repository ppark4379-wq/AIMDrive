"""Training utilities shared by train.py, evaluate.py and tools/.

runtime      offline caches, sha256 provenance, numeric adapters, path resolving
training     batching, autocast, seeding and gradient reporting
trainer      the stage-2 VLDrive training / preflight / evaluation loop
metrics      trajectory metrics (ADE, FDE, per-horizon error)
"""
