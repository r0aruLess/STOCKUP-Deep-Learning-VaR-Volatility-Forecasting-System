"""Run the adversarial study.   python scripts/attack_study.py"""
import pandas as pd
import torch
from _common import parse_args

from deepvar.config import REPORT_DIR
from deepvar.pipeline import (Forecaster, SecureForecaster, artifact_path, load_data,
                              trusted_history)
from deepvar.security.attacks import pgd_underestimate
from deepvar.security.evaluate import most_volatile_start, poison_sweep, run_attack
from deepvar.data.features import FEATURE_COLS, build_features
from deepvar.data.windows import make_windows

cfg, _ = parse_args("Adversarial robustness study")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
df = load_data(cfg)
fc = Forecaster.load(artifact_path(cfg))
sec = SecureForecaster.build(fc, cfg, trusted_history(cfg, df))
conf, mu = cfg["risk"]["confidence"], cfg["risk"]["mu"]

start = most_volatile_start(df, cfg["data"]["val_end"], length=20)
print(f"Attack launched at the most turbulent stretch of the test period: {df.index[start].date()}\n")

# 1) three feed-level attacks
rows = []
for atk, n, kw in [("flash_crash", 1, {}), ("fat_finger", 1, {}), ("vol_suppression", 10, {"alpha": 0.2})]:
    r = run_attack(fc, sec, df, atk, start, n, conf=conf, mu=mu, **kw)
    r.pop("series")
    rows.append(r)
summary = pd.DataFrame(rows).round(4)
print("== Attack summary (ratio = average sigma vs. the clean-data forecast; <1 means risk UNDER-estimated) ==")
print(summary[["attack", "n_days", "sigma_ratio_vanilla", "sigma_ratio_secure",
               "breaches_clean", "breaches_vanilla", "breaches_secure", "gate_detected"]].to_string(index=False))
summary.to_csv(REPORT_DIR / "attack_summary.csv", index=False)

# 2) how many poisoned days does it take to blind the model?
sweeps = []
for alpha in (0.2, 0.5):
    sw = poison_sweep(fc, sec, df, start, alpha=alpha, max_days=30, step=5, conf=conf, mu=mu)
    sweeps.append(sw)
sweep = pd.concat(sweeps)
print("\n== Poisoning sweep: consecutive poisoned days vs. forecast damage ==")
print(sweep.round(3).to_string(index=False))
sweep.to_csv(REPORT_DIR / "poison_sweep.csv", index=False)
for a, sw in sweep.groupby("alpha"):
    hit = sw[sw["vanilla_blinded"]]
    print(f"alpha={a}: vanilla model blinded after {int(hit['poisoned_days'].iloc[0]) if len(hit) else '>30'} poisoned days")

# 3) gradient (PGD) attack on the model input - white-box worst case
feats = build_features(df, cfg["data"]["vol_window"])
X = make_windows(fc.feat_scaler.transform(feats[FEATURE_COLS].values), fc.window)
X = X[-250:]
base = fc.target_scaler.inverse(fc.model(torch.from_numpy(X)).detach().numpy().reshape(-1, 1)).ravel()
print("\n== White-box PGD attack on the last 250 input windows ==")
for eps in (0.1, 0.25, 0.5):
    adv = pgd_underestimate(fc.model, X, eps=eps)
    out = fc.target_scaler.inverse(fc.model(torch.from_numpy(adv)).detach().numpy().reshape(-1, 1)).ravel()
    print(f"eps={eps:<5} avg predicted sigma falls {100 * (1 - out.mean() / base.mean()):5.1f}%")
print("\nreports written to reports/*.csv")
