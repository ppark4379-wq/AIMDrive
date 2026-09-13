"""Training utilities shared by train.py and evaluate.py.

config        YAML loading, overrides and path resolution
distributed   torchrun setup, rank helpers and samplers
logger        logging, iteration meters and TensorBoard writers
metrics       AverageMeter and accuracy
losses        vision encoder pre-training loss
checkpoint    saving, loading and resuming both stages
schedulers    learning-rate schedules of both stages
visualization TensorBoard images of camera views, detections and waypoints
"""
