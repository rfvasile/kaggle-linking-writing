def calc_metric(cfg, pp_out, val_df, pre="val"):
    if "loss" in pp_out:
        return {"loss": float(pp_out["loss"].mean())}
    return {}
