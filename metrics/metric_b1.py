def calc_metric(cfg, pp_out, val_df, pre="val"):
    preds = pp_out["preds"].float()
    targets = pp_out["target"].float()
    if preds.shape != targets.shape or preds.numel() == 0:
        raise ValueError("RMSE requires nonempty predictions and targets with matching shapes")
    return {"rmse": (preds - targets).square().mean().sqrt().item()}
