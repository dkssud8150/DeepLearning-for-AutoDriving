cfg = {
    "seed" : 42,
    "img_size" : (1242, 375),
    "train_bs" : 16,
    "valid_bs" : 16,
    "num_classes" : 25,
    "n_fold" : 5,
    "model_name" : "regnet",
    "optimizer" : "AdamW", # adam, SGD, adamw
    "lr" : 1e-4,
    "scheduler" : "CosineAnnealingLR",
    "warmup_epochs" : 0,
    "weight_decay" : 1e-6,
    "sgd_momentum" : 0.999,
}