from types import SimpleNamespace

cfg = SimpleNamespace(**{})

# init
cfg.seed = 3
cfg.model = "models.nn_b1"
cfg.dataset = "data.ds_b1"

# stages
cfg.train = True
cfg.val = True
cfg.test = True
cfg.train_val = True
