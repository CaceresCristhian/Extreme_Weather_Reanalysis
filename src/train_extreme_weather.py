"""
Deep Temporal Sequence Modeling for Multi-Hazard Extreme Weather Detection.
Evaluated on NOAA GHCN-Daily physical station observations across contrasting climate biomes.
Implements chronological train/validation/test partitions, sequence-to-event formulation,
and comprehensive baseline comparisons (Persistence, Climatology, Logistic Regression, CNN, HistGBM).
"""

from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "01_raw" / "noaa_ghcn_daily_stations_2020_2024.parquet"
PROC_DIR = BASE_DIR / "data" / "02_processed"
FIG_DIR = BASE_DIR / "paper" / "figures"
TAB_DIR = BASE_DIR / "paper" / "tables"
EXP_DIR = BASE_DIR / "experiments"

PROC_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)
EXP_DIR.mkdir(parents=True, exist_ok=True)

# Device setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_properties(0).name
else:
    device = torch.device("cpu")
    gpu_name = "CPU"


class ClimateSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ExtremeWeatherTransformer(nn.Module):
    def __init__(self, in_dim=6, seq_len=7, d_model=64, n_heads=4):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=128, dropout=0.15, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        h = self.proj(x) + self.pos_emb
        trans = self.transformer(h)
        last_h = trans[:, -1, :]
        return self.classifier(last_h)


class ExtremeWeatherCNN1D(nn.Module):
    """Attention-free sequence baseline for RQ1 ablation: 1D temporal convolution with residual pooling."""
    def __init__(self, in_dim=6, seq_len=7, hidden_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        feat = self.conv(x.transpose(1, 2)).squeeze(-1)
        return self.classifier(feat)


def recalibrate_probabilities(p_val, y_val, p_test):
    """Post-hoc probability recalibration via Platt scaling on validation holdout."""
    eps = 1e-5
    p_val_c = np.clip(p_val, eps, 1.0 - eps)
    p_test_c = np.clip(p_test, eps, 1.0 - eps)
    logit_val = np.log(p_val_c / (1.0 - p_val_c)).reshape(-1, 1)
    logit_test = np.log(p_test_c / (1.0 - p_test_c)).reshape(-1, 1)
    calibrator = LogisticRegression()
    calibrator.fit(logit_val, y_val)
    return calibrator.predict_proba(logit_test)[:, 1]


def paired_bootstrap_onset(y_true, pers_yesterday, probs_a, probs_b, B=1000, seed=42, block_size=7):
    """Compute paired moving-block bootstrap 95% CIs on onset metric differences to account for serial correlation."""
    rng = np.random.default_rng(seed)
    onset_mask = (pers_yesterday == 0)
    y_sub = y_true[onset_mask]
    pa_sub = probs_a[onset_mask]
    pb_sub = probs_b[onset_mask]
    n = len(y_sub)

    n_blocks = int(np.ceil(n / block_size))
    diff_rec = []
    diff_prauc = []
    for _ in range(B):
        start_indices = rng.choice(max(1, n - block_size + 1), size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(s, min(s + block_size, n)) for s in start_indices])[:n]
        y_b = y_sub[idx]
        if len(np.unique(y_b)) < 2:
            continue
        rec_a = recall_score(y_b, (pa_sub[idx] >= 0.5).astype(int), zero_division=0) * 100.0
        rec_b = recall_score(y_b, (pb_sub[idx] >= 0.5).astype(int), zero_division=0) * 100.0
        prauc_a = average_precision_score(y_b, pa_sub[idx])
        prauc_b = average_precision_score(y_b, pb_sub[idx])
        diff_rec.append(rec_a - rec_b)
        diff_prauc.append(prauc_a - prauc_b)

    ci_rec = (float(np.percentile(diff_rec, 2.5)), float(np.percentile(diff_rec, 97.5)))
    ci_prauc = (float(np.percentile(diff_prauc, 2.5)), float(np.percentile(diff_prauc, 97.5)))
    return ci_rec, ci_prauc


def compute_multi_hazard_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Compute binary multi-hazard indicator flags according to physical thresholds."""
    df = df.copy()
    df["heat_extreme"] = (df["tmax_c"] >= 38.0).astype(int)
    df["freeze_extreme"] = (df["tmin_c"] <= -15.0).astype(int)
    df["precip_extreme"] = (df["prcp_mm"] >= 25.0).astype(int)
    df["wind_extreme"] = (df["wind_gust_ms"] >= 20.0).astype(int)
    df["is_extreme"] = (
        (df["heat_extreme"] == 1)
        | (df["freeze_extreme"] == 1)
        | (df["precip_extreme"] == 1)
        | (df["wind_extreme"] == 1)
    ).astype(int)
    return df


def load_and_preprocess_data():
    df = pd.read_parquet(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["station", "date"]).reset_index(drop=True)

    df = compute_multi_hazard_labels(df)

    # Cyclical day-of-year features
    doy = df["date"].dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    df["log_prcp"] = np.log1p(df["prcp_mm"])

    return df


def build_chronological_sequences(df, seq_len=7):
    feature_cols = ["tmax_c", "tmin_c", "log_prcp", "wind_gust_ms", "sin_doy", "cos_doy"]
    
    train_records = []
    val_records = []
    test_records = []

    for stn in df["station"].unique():
        sdf = df[df["station"] == stn].copy().reset_index(drop=True)
        feat_vals = sdf[feature_cols].values
        labels = sdf["is_extreme"].values
        dates = sdf["date"].values

        for i in range(len(sdf) - seq_len):
            x_win = feat_vals[i : i + seq_len]
            y_next = labels[i + seq_len]
            target_date = pd.to_datetime(dates[i + seq_len])
            last_step_extreme = labels[i + seq_len - 1]
            month = target_date.month

            rec = {
                "station": stn,
                "target_date": target_date,
                "X": x_win,
                "y": y_next,
                "last_step_extreme": last_step_extreme,
                "month": month,
            }

            if target_date <= pd.to_datetime("2023-12-31"):
                train_records.append(rec)
            elif target_date <= pd.to_datetime("2024-12-31"):
                val_records.append(rec)
            else:
                test_records.append(rec)

    return train_records, val_records, test_records, feature_cols


def extract_arrays(records):
    X = np.stack([r["X"] for r in records], axis=0).astype(np.float32)
    y = np.array([r["y"] for r in records], dtype=np.int64)
    pers = np.array([r["last_step_extreme"] for r in records], dtype=np.int64)
    stns = np.array([r["station"] for r in records])
    months = np.array([r["month"] for r in records])
    return X, y, pers, stns, months


def evaluate_onset_metrics(y_true, pers_yesterday, probs, threshold=0.5, target_flagged_pct=7.5):
    """
    Evaluates detection performance on hazard onsets:
    Subset: Days where yesterday was non-extreme (pers_yesterday == 0).
    Also evaluates precision and recall at matched operating point (top target_flagged_pct% flagged).
    """
    onset_mask = (pers_yesterday == 0)
    y_sub = y_true[onset_mask]
    p_sub = probs[onset_mask]
    pred_sub = (p_sub >= threshold).astype(int)

    n_sub = int(onset_mask.sum())
    n_pos = int(y_sub.sum())

    rec = float(recall_score(y_sub, pred_sub, zero_division=0) * 100.0)
    prec = float(precision_score(y_sub, pred_sub, zero_division=0) * 100.0)
    flagged = float(pred_sub.mean() * 100.0)
    auc = float(roc_auc_score(y_sub, p_sub)) if len(np.unique(y_sub)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_sub, p_sub)) if len(np.unique(y_sub)) > 1 else 0.0

    # Matched operating point: threshold set to flag exactly target_flagged_pct% of days
    k_flagged = max(1, int(np.round(n_sub * (target_flagged_pct / 100.0))))
    sorted_probs = np.sort(p_sub)[::-1]
    th_matched = float(sorted_probs[k_flagged - 1])
    pred_matched = (p_sub >= th_matched).astype(int)
    rec_matched = float(recall_score(y_sub, pred_matched, zero_division=0) * 100.0)
    prec_matched = float(precision_score(y_sub, pred_matched, zero_division=0) * 100.0)

    return {
        "n_sub": n_sub,
        "n_pos": n_pos,
        "Recall": rec,
        "Precision": prec,
        "Flagged_Pct": flagged,
        "AUC": auc,
        "PR_AUC": pr_auc,
        "th_matched": th_matched,
        "Recall_Matched": rec_matched,
        "Precision_Matched": prec_matched,
    }


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_prob)
    return {
        "AUC": auc,
        "PR_AUC": pr_auc,
        "F1": f1 * 100.0,
        "Recall": rec * 100.0,
        "Precision": prec * 100.0,
        "Brier": brier,
    }


def run_pipeline():
    print(f"Executing Deep Climate Pipeline on device: {gpu_name}")
    df = load_and_preprocess_data()
    total_days = len(df)
    total_extremes = df["is_extreme"].sum()
    print(f"Loaded {total_days:,} physical daily records. Total extremes: {total_extremes:,} ({total_extremes/total_days*100:.2f}%)")

    # Per hazard breakdown
    print(f"  Heat extremes (>=38C): {df['heat_extreme'].sum():,}")
    print(f"  Freeze extremes (<=-15C): {df['freeze_extreme'].sum():,}")
    print(f"  Precip extremes (>=25mm): {df['precip_extreme'].sum():,}")
    print(f"  Wind extremes (>=20m/s): {df['wind_extreme'].sum():,}")

    train_recs, val_recs, test_recs, feature_cols = build_chronological_sequences(df, seq_len=7)
    print(f"Sequences split chronologically: Train={len(train_recs):,}, Val={len(val_recs):,}, Test={len(test_recs):,}")

    X_train, y_train, pers_train, stn_train, month_train = extract_arrays(train_recs)
    X_val, y_val, pers_val, stn_val, month_val = extract_arrays(val_recs)
    X_test, y_test, pers_test, stn_test, month_test = extract_arrays(test_recs)

    # Normalization strictly using training statistics
    mean_f = X_train.mean(axis=(0, 1), keepdims=True)
    std_f = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
    X_train_norm = (X_train - mean_f) / std_f
    X_val_norm = (X_val - mean_f) / std_f
    X_test_norm = (X_test - mean_f) / std_f

    # 1. Baseline 1: Naive Persistence
    preds_pers_prob = pers_test.astype(float)
    metrics_pers = compute_metrics(y_test, preds_pers_prob)

    # 2. Baseline 2: Monthly Climatology by Station
    clim_lookup = {}
    train_df_stats = pd.DataFrame({"station": stn_train, "month": month_train, "y": y_train})
    for (stn, m), grp in train_df_stats.groupby(["station", "month"]):
        clim_lookup[(stn, m)] = grp["y"].mean()

    default_rate = y_train.mean()
    preds_clim_prob = np.array([clim_lookup.get((stn, m), default_rate) for stn, m in zip(stn_test, month_test)])
    metrics_clim = compute_metrics(y_test, preds_clim_prob)

    # 3. Baseline 3: Logistic Regression on last time-step
    lr = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
    lr.fit(X_train_norm[:, -1, :], y_train)
    preds_lr_prob = lr.predict_proba(X_test_norm[:, -1, :])[:, 1]
    metrics_lr = compute_metrics(y_test, preds_lr_prob)

    # 4. Baseline 4: HistGradientBoosting on flattened 7-day lookback
    hgb = HistGradientBoostingClassifier(
        max_iter=150, max_depth=5, learning_rate=0.05, class_weight="balanced", random_state=42
    )
    hgb.fit(X_train_norm.reshape(len(X_train), -1), y_train)
    preds_hgb_prob = hgb.predict_proba(X_test_norm.reshape(len(X_test), -1))[:, 1]
    metrics_hgb = compute_metrics(y_test, preds_hgb_prob)

    # 5. Attention-Free Sequence Baseline: 1D-CNN (PyTorch CUDA)
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    train_ds = ClimateSequenceDataset(X_train_norm, y_train)
    val_ds = ClimateSequenceDataset(X_val_norm, y_val)
    test_ds = ClimateSequenceDataset(X_test_norm, y_test)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / (y_train.sum() + 1e-6)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print("Training Attention-Free 1D-CNN Sequence Baseline...")
    cnn_model = ExtremeWeatherCNN1D(in_dim=len(feature_cols), seq_len=7, hidden_dim=64).to(device)
    cnn_optimizer = torch.optim.AdamW(cnn_model.parameters(), lr=0.001, weight_decay=1e-3)
    cnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(cnn_optimizer, T_max=20)
    cnn_best_loss = float("inf")
    cnn_best_weights = None

    for ep in range(20):
        cnn_model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            cnn_optimizer.zero_grad()
            out = cnn_model(bx)
            loss = criterion(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cnn_model.parameters(), max_norm=1.0)
            cnn_optimizer.step()
        cnn_scheduler.step()

        cnn_model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = cnn_model(bx)
                v_loss += criterion(out, by).item() * len(by)
        val_loss = v_loss / len(val_ds)
        if val_loss < cnn_best_loss:
            cnn_best_loss = val_loss
            cnn_best_weights = {k: v.cpu().clone() for k, v in cnn_model.state_dict().items()}

    cnn_model.load_state_dict({k: v.to(device) for k, v in cnn_best_weights.items()})

    # CNN test predictions
    cnn_model.eval()
    test_cnn_probs = []
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device)
            probs = torch.sigmoid(cnn_model(bx)).squeeze(-1).cpu().numpy()
            test_cnn_probs.extend(probs)
    preds_cnn_prob = np.array(test_cnn_probs)
    metrics_cnn = compute_metrics(y_test, preds_cnn_prob)

    # 6. Deep Temporal Transformer (PyTorch CUDA)
    print("Training Deep Temporal Transformer...")
    model = ExtremeWeatherTransformer(in_dim=len(feature_cols), seq_len=7, d_model=64, n_heads=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    best_val_loss = float("inf")
    best_weights = None
    epochs = 20
    train_losses = []
    val_losses = []

    for ep in range(epochs):
        model.train()
        tot_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tot_loss += loss.item() * len(by)
        scheduler.step()
        train_loss = tot_loss / len(train_ds)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                loss = criterion(out, by)
                v_loss += loss.item() * len(by)
        val_loss = v_loss / len(val_ds)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (ep + 1) % 5 == 0:
            print(f"Epoch {ep+1:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Load best checkpoint
    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    torch.save(best_weights, EXP_DIR / "best_extreme_weather_transformer.pt")

    # Evaluation on Test Set
    model.eval()
    all_probs = []
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device)
            logits = model(bx)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            all_probs.extend(probs)
    preds_deep_prob = np.array(all_probs)
    metrics_deep = compute_metrics(y_test, preds_deep_prob)

    # 7. Post-Hoc Probability Recalibration (Platt Scaling on 2024 Validation Set)
    print("Executing post-hoc probability recalibration via Platt scaling on validation holdout...")
    preds_lr_val = lr.predict_proba(X_val_norm[:, -1, :])[:, 1]
    preds_hgb_val = hgb.predict_proba(X_val_norm.reshape(len(X_val), -1))[:, 1]

    val_cnn_probs = []
    with torch.no_grad():
        for bx, _ in val_loader:
            bx = bx.to(device)
            val_cnn_probs.extend(torch.sigmoid(cnn_model(bx)).squeeze(-1).cpu().numpy())
    preds_cnn_val = np.array(val_cnn_probs)

    val_deep_probs = []
    with torch.no_grad():
        for bx, _ in val_loader:
            bx = bx.to(device)
            val_deep_probs.extend(torch.sigmoid(model(bx)).squeeze(-1).cpu().numpy())
    preds_deep_val = np.array(val_deep_probs)

    preds_lr_recal = recalibrate_probabilities(preds_lr_val, y_val, preds_lr_prob)
    preds_hgb_recal = recalibrate_probabilities(preds_hgb_val, y_val, preds_hgb_prob)
    preds_cnn_recal = recalibrate_probabilities(preds_cnn_val, y_val, preds_cnn_prob)
    preds_deep_recal = recalibrate_probabilities(preds_deep_val, y_val, preds_deep_prob)

    # Compute raw BSS and recalibrated BSS relative to Station-Month Climatology
    bs_clim = metrics_clim["Brier"]
    for m in [metrics_pers, metrics_clim, metrics_lr, metrics_cnn, metrics_hgb, metrics_deep]:
        m["BSS"] = 1.0 - (m["Brier"] / bs_clim)

    metrics_pers["BSS_Recal"] = metrics_pers["BSS"]
    metrics_clim["BSS_Recal"] = 0.0
    metrics_lr["Brier_Recal"] = brier_score_loss(y_test, preds_lr_recal)
    metrics_lr["BSS_Recal"] = 1.0 - (metrics_lr["Brier_Recal"] / bs_clim)
    metrics_cnn["Brier_Recal"] = brier_score_loss(y_test, preds_cnn_recal)
    metrics_cnn["BSS_Recal"] = 1.0 - (metrics_cnn["Brier_Recal"] / bs_clim)
    metrics_hgb["Brier_Recal"] = brier_score_loss(y_test, preds_hgb_recal)
    metrics_hgb["BSS_Recal"] = 1.0 - (metrics_hgb["Brier_Recal"] / bs_clim)
    metrics_deep["Brier_Recal"] = brier_score_loss(y_test, preds_deep_recal)
    metrics_deep["BSS_Recal"] = 1.0 - (metrics_deep["Brier_Recal"] / bs_clim)

    # Genuine Hazard Onset Evaluation (where yesterday was non-extreme: pers_test == 0)
    onset_pers = evaluate_onset_metrics(y_test, pers_test, preds_pers_prob)
    onset_clim = evaluate_onset_metrics(y_test, pers_test, preds_clim_prob)
    onset_lr = evaluate_onset_metrics(y_test, pers_test, preds_lr_prob)
    onset_cnn = evaluate_onset_metrics(y_test, pers_test, preds_cnn_prob)
    onset_hgb = evaluate_onset_metrics(y_test, pers_test, preds_hgb_prob)
    onset_deep = evaluate_onset_metrics(y_test, pers_test, preds_deep_prob)

    for m, o in [
        (metrics_pers, onset_pers),
        (metrics_clim, onset_clim),
        (metrics_lr, onset_lr),
        (metrics_cnn, onset_cnn),
        (metrics_hgb, onset_hgb),
        (metrics_deep, onset_deep),
    ]:
        m["Onset_Recall"] = o["Recall"]
        m["Onset_Precision"] = o["Precision"]
        m["Onset_AUC"] = o["AUC"]
        m["Onset_PR_AUC"] = o["PR_AUC"]

    ci_rec_deep_hgb, ci_prauc_deep_hgb = paired_bootstrap_onset(y_test, pers_test, preds_deep_prob, preds_hgb_prob)
    ci_rec_deep_cnn, ci_prauc_deep_cnn = paired_bootstrap_onset(y_test, pers_test, preds_deep_prob, preds_cnn_prob)
    print(f"\nPaired Bootstrap CIs (Deep Transformer vs HistGBM across N={onset_deep['n_pos']} onsets):")
    print(f"  Onset Recall Difference (95% CI): [{ci_rec_deep_hgb[0]:.1f}%, {ci_rec_deep_hgb[1]:.1f}%]")
    print(f"  Onset PR-AUC Difference (95% CI): [{ci_prauc_deep_hgb[0]:.3f}, {ci_prauc_deep_hgb[1]:.3f}]")
    print(f"Paired Bootstrap CIs (Deep Transformer vs 1D-CNN across N={onset_deep['n_pos']} onsets):")
    print(f"  Onset Recall Difference (95% CI): [{ci_rec_deep_cnn[0]:.1f}%, {ci_rec_deep_cnn[1]:.1f}%]")
    print(f"  Onset PR-AUC Difference (95% CI): [{ci_prauc_deep_cnn[0]:.3f}, {ci_prauc_deep_cnn[1]:.3f}]")

    print(f"\n--- HAZARD ONSET BENCHMARK (N = {onset_deep['n_pos']} onsets across {onset_deep['n_sub']} days) ---")
    print(f"Naive Persistence     : Onset Rec = {onset_pers['Recall']:.1f}%, Prec = {onset_pers['Precision']:.1f}%, Flagged = {onset_pers['Flagged_Pct']:.1f}%")
    print(f"Monthly Climatology   : Onset Rec = {onset_clim['Recall']:.1f}%, Prec = {onset_clim['Precision']:.1f}%, Flagged = {onset_clim['Flagged_Pct']:.1f}%")
    print(f"Logistic Regression   : Onset Rec = {onset_lr['Recall']:.1f}%, Prec = {onset_lr['Precision']:.1f}%, Flagged = {onset_lr['Flagged_Pct']:.1f}%")
    print(f"1D Temporal ConvNet   : Onset Rec = {onset_cnn['Recall']:.1f}%, Prec = {onset_cnn['Precision']:.1f}%, Flagged = {onset_cnn['Flagged_Pct']:.1f}%")
    print(f"HistGradientBoosting  : Onset Rec = {onset_hgb['Recall']:.1f}%, Prec = {onset_hgb['Precision']:.1f}%, Flagged = {onset_hgb['Flagged_Pct']:.1f}%")
    print(f"Temporal Transformer  : Onset Rec = {onset_deep['Recall']:.1f}%, Prec = {onset_deep['Precision']:.1f}%, Flagged = {onset_deep['Flagged_Pct']:.1f}%")

    # Print Full Comparative Table
    print("\n================== NOAA EXTREME WEATHER BENCHMARK ==================")
    header = f"{'Model':<28} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'F1 (%)':<8} | {'Recall':<8} | {'Prec':<8} | {'Brier':<8} | {'BSS_Raw':<8} | {'BSS_Recal':<10}"
    print(header)
    print("-" * len(header))
    for name, m in [
        ("Naive Persistence", metrics_pers),
        ("Monthly Climatology", metrics_clim),
        ("Logistic Regression (t)", metrics_lr),
        ("1D Temporal ConvNet (CNN)", metrics_cnn),
        ("HistGradientBoosting (7d)", metrics_hgb),
        ("Temporal Transformer", metrics_deep),
    ]:
        print(f"{name:<28} | {m['AUC']:<8.3f} | {m['PR_AUC']:<8.3f} | {m['F1']:<8.1f} | {m['Recall']:<8.1f} | {m['Precision']:<8.1f} | {m['Brier']:<8.4f} | {m['BSS']:<8.2f} | {m['BSS_Recal']:<10.2f}")

    # Generate Publication Figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 11))
    # Loss curves
    ax1.plot(range(1, epochs + 1), train_losses, "b-o", lw=1.8, markersize=3, label="Transformer Train Loss")
    ax1.plot(range(1, epochs + 1), val_losses, "r--s", lw=1.8, markersize=3, label="Transformer Val Loss")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.set_title("Training & Validation Convergence (NOAA Stations)", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # ROC Curves
    fpr_pers, tpr_pers, _ = roc_curve(y_test, preds_pers_prob)
    fpr_clim, tpr_clim, _ = roc_curve(y_test, preds_clim_prob)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, preds_lr_prob)
    fpr_cnn, tpr_cnn, _ = roc_curve(y_test, preds_cnn_prob)
    fpr_hgb, tpr_hgb, _ = roc_curve(y_test, preds_hgb_prob)
    fpr_deep, tpr_deep, _ = roc_curve(y_test, preds_deep_prob)
    ax2.plot(fpr_pers, tpr_pers, "k:", label=f"Persistence (AUC={metrics_pers['AUC']:.3f})")
    ax2.plot(fpr_clim, tpr_clim, "m-.", label=f"Climatology (AUC={metrics_clim['AUC']:.3f})")
    ax2.plot(fpr_lr, tpr_lr, "c:", label=f"Logistic Reg (AUC={metrics_lr['AUC']:.3f})")
    ax2.plot(fpr_cnn, tpr_cnn, "y-.", label=f"1D-CNN (AUC={metrics_cnn['AUC']:.3f})")
    ax2.plot(fpr_hgb, tpr_hgb, "g--", label=f"HistGBM (AUC={metrics_hgb['AUC']:.3f})")
    ax2.plot(fpr_deep, tpr_deep, "b-", lw=2, label=f"Temporal Transformer (AUC={metrics_deep['AUC']:.3f})")
    ax2.plot([0, 1], [0, 1], "gray", linestyle="--", alpha=0.4)
    ax2.set_xlabel("False Positive Rate", fontsize=10)
    ax2.set_ylabel("True Positive Rate", fontsize=10)
    ax2.set_title("ROC Curves (Chronological Holdout 2025-2026)", fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right", fontsize=8)

    # Precision-Recall Curves
    pr_prec_hgb, pr_rec_hgb, _ = precision_recall_curve(y_test, preds_hgb_prob)
    pr_prec_cnn, pr_rec_cnn, _ = precision_recall_curve(y_test, preds_cnn_prob)
    pr_prec_deep, pr_rec_deep, _ = precision_recall_curve(y_test, preds_deep_prob)
    ax3.plot(pr_rec_hgb, pr_prec_hgb, "g--", label=f"HistGBM (PR-AUC={metrics_hgb['PR_AUC']:.3f})")
    ax3.plot(pr_rec_cnn, pr_prec_cnn, "y-.", label=f"1D-CNN (PR-AUC={metrics_cnn['PR_AUC']:.3f})")
    ax3.plot(pr_rec_deep, pr_prec_deep, "b-", lw=2, label=f"Temporal Transformer (PR-AUC={metrics_deep['PR_AUC']:.3f})")
    ax3.set_xlabel("Recall", fontsize=10)
    ax3.set_ylabel("Precision", fontsize=10)
    ax3.set_title("Precision-Recall Curves (Imbalanced Extreme Hazard)", fontsize=11, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right", fontsize=8)

    # Hazard Breakdown per Station
    stn_hazard_counts = df.groupby("station")[["heat_extreme", "freeze_extreme", "precip_extreme", "wind_extreme"]].sum()
    stn_hazard_counts.plot(kind="bar", stacked=True, ax=ax4, colormap="viridis")
    ax4.set_title("Multi-Hazard Distribution Across 8 NOAA Stations", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Station", fontsize=10)
    ax4.set_ylabel("Observed Extreme Days (2020-2026)", fontsize=10)
    ax4.tick_params(axis="x", rotation=35)
    ax4.grid(True, alpha=0.3, axis="y")
    ax4.legend(["Heat (>=38C)", "Freeze (<=-15C)", "Precip (>=25mm)", "Wind (>=20m/s)"], fontsize=8)

    plt.tight_layout()
    fig_path = FIG_DIR / "fig1_extreme_weather_benchmark.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved publication figure to {fig_path}")

    num_heat_days = int(df["heat_extreme"].sum())
    num_freeze_days = int(df["freeze_extreme"].sum())
    num_precip_days = int(df["precip_extreme"].sum())
    num_wind_days = int(df["wind_extreme"].sum())
    w_pos_val = float(pos_weight.item())

    # LaTeX Macros
    with open(TAB_DIR / "macros.tex", "w") as f:
        f.write("% Auto-generated macros from NOAA GHCN-Daily Pipeline\n")
        f.write(f"\\newcommand{{\\numStations}}{{{len(df['station'].unique())}}}\n")
        f.write(f"\\newcommand{{\\numClimateRecords}}{{{total_days:,}}}\n")
        f.write(f"\\newcommand{{\\numTrainSeq}}{{{len(train_recs):,}}}\n")
        f.write(f"\\newcommand{{\\numValSeq}}{{{len(val_recs):,}}}\n")
        f.write(f"\\newcommand{{\\numTestSeq}}{{{len(test_recs):,}}}\n")
        f.write(f"\\newcommand{{\\numExtremeEvents}}{{{total_extremes:,}}}\n")
        f.write(f"\\newcommand{{\\prevalenceRate}}{{{total_extremes/total_days*100:.1f}}}\n")
        f.write(f"\\newcommand{{\\numHeatDays}}{{{num_heat_days:,}}}\n")
        f.write(f"\\newcommand{{\\numFreezeDays}}{{{num_freeze_days:,}}}\n")
        f.write(f"\\newcommand{{\\numPrecipDays}}{{{num_precip_days:,}}}\n")
        f.write(f"\\newcommand{{\\numWindDays}}{{{num_wind_days}}}\n")
        f.write(f"\\newcommand{{\\wPos}}{{{w_pos_val:.1f}}}\n")
        f.write(f"\\newcommand{{\\nOnsetEvents}}{{{onset_deep['n_pos']}}}\n")
        f.write(f"\\newcommand{{\\nOnsetDays}}{{{onset_deep['n_sub']}}}\n")
        f.write(f"\\newcommand{{\\onsetAucHgb}}{{{onset_hgb['AUC']:.3f}}}\n")
        f.write(f"\\newcommand{{\\onsetAucCnn}}{{{onset_cnn['AUC']:.3f}}}\n")
        f.write(f"\\newcommand{{\\onsetAucDeep}}{{{onset_deep['AUC']:.3f}}}\n")
        f.write(f"\\newcommand{{\\onsetPrAucHgb}}{{{onset_hgb['PR_AUC']:.3f}}}\n")
        f.write(f"\\newcommand{{\\onsetPrAucCnn}}{{{onset_cnn['PR_AUC']:.3f}}}\n")
        f.write(f"\\newcommand{{\\onsetPrAucDeep}}{{{onset_deep['PR_AUC']:.3f}}}\n")
        f.write(f"\\newcommand{{\\ciOnsetRecHgbLow}}{{{ci_rec_deep_hgb[0]:.1f}}}\n")
        f.write(f"\\newcommand{{\\ciOnsetRecHgbHigh}}{{{ci_rec_deep_hgb[1]:.1f}}}\n")
        f.write(f"\\newcommand{{\\ciOnsetPrAucHgbLow}}{{{ci_prauc_deep_hgb[0]:.3f}}}\n")
        f.write(f"\\newcommand{{\\ciOnsetPrAucHgbHigh}}{{{ci_prauc_deep_hgb[1]:.3f}}}\n")
        f.write(f"\\newcommand{{\\ciOnsetRecCnnLow}}{{{ci_rec_deep_cnn[0]:.1f}}}\n")
        f.write(f"\\newcommand{{\\ciOnsetRecCnnHigh}}{{{ci_rec_deep_cnn[1]:.1f}}}\n")
        f.write(f"\\newcommand{{\\ciOnsetPrAucCnnLow}}{{{ci_prauc_deep_cnn[0]:.3f}}}\n")
        f.write(f"\\newcommand{{\\ciOnsetPrAucCnnHigh}}{{{ci_prauc_deep_cnn[1]:.3f}}}\n")
        f.write(f"\\newcommand{{\\aucPers}}{{{metrics_pers['AUC']:.3f}}}\n")
        f.write(f"\\newcommand{{\\aucClim}}{{{metrics_clim['AUC']:.3f}}}\n")
        f.write(f"\\newcommand{{\\aucLr}}{{{metrics_lr['AUC']:.3f}}}\n")
        f.write(f"\\newcommand{{\\aucCnn}}{{{metrics_cnn['AUC']:.3f}}}\n")
        f.write(f"\\newcommand{{\\aucHgb}}{{{metrics_hgb['AUC']:.3f}}}\n")
        f.write(f"\\newcommand{{\\aucDeep}}{{{metrics_deep['AUC']:.3f}}}\n")
        f.write(f"\\newcommand{{\\prAucHgb}}{{{metrics_hgb['PR_AUC']:.3f}}}\n")
        f.write(f"\\newcommand{{\\prAucCnn}}{{{metrics_cnn['PR_AUC']:.3f}}}\n")
        f.write(f"\\newcommand{{\\prAucDeep}}{{{metrics_deep['PR_AUC']:.3f}}}\n")
        f.write(f"\\newcommand{{\\fOnePers}}{{{metrics_pers['F1']:.1f}}}\n")
        f.write(f"\\newcommand{{\\fOneClim}}{{{metrics_clim['F1']:.1f}}}\n")
        f.write(f"\\newcommand{{\\fOneLr}}{{{metrics_lr['F1']:.1f}}}\n")
        f.write(f"\\newcommand{{\\fOneCnn}}{{{metrics_cnn['F1']:.1f}}}\n")
        f.write(f"\\newcommand{{\\fOneHgb}}{{{metrics_hgb['F1']:.1f}}}\n")
        f.write(f"\\newcommand{{\\fOneDeep}}{{{metrics_deep['F1']:.1f}}}\n")
        f.write(f"\\newcommand{{\\recPers}}{{{metrics_pers['Recall']:.1f}}}\n")
        f.write(f"\\newcommand{{\\recClim}}{{{metrics_clim['Recall']:.1f}}}\n")
        f.write(f"\\newcommand{{\\recLr}}{{{metrics_lr['Recall']:.1f}}}\n")
        f.write(f"\\newcommand{{\\recCnn}}{{{metrics_cnn['Recall']:.1f}}}\n")
        f.write(f"\\newcommand{{\\recHgb}}{{{metrics_hgb['Recall']:.1f}}}\n")
        f.write(f"\\newcommand{{\\recDeep}}{{{metrics_deep['Recall']:.1f}}}\n")
        f.write(f"\\newcommand{{\\precPers}}{{{metrics_pers['Precision']:.1f}}}\n")
        f.write(f"\\newcommand{{\\precClim}}{{{metrics_clim['Precision']:.1f}}}\n")
        f.write(f"\\newcommand{{\\precLr}}{{{metrics_lr['Precision']:.1f}}}\n")
        f.write(f"\\newcommand{{\\precCnn}}{{{metrics_cnn['Precision']:.1f}}}\n")
        f.write(f"\\newcommand{{\\precHgb}}{{{metrics_hgb['Precision']:.1f}}}\n")
        f.write(f"\\newcommand{{\\precDeep}}{{{metrics_deep['Precision']:.1f}}}\n")
        f.write(f"\\newcommand{{\\onsetRecPers}}{{{onset_pers['Recall']:.1f}}}\n")
        f.write(f"\\newcommand{{\\onsetRecClim}}{{{onset_clim['Recall']:.1f}}}\n")
        f.write(f"\\newcommand{{\\onsetRecLr}}{{{onset_lr['Recall']:.1f}}}\n")
        f.write(f"\\newcommand{{\\onsetRecCnn}}{{{onset_cnn['Recall']:.1f}}}\n")
        f.write(f"\\newcommand{{\\onsetRecHgb}}{{{onset_hgb['Recall']:.1f}}}\n")
        f.write(f"\\newcommand{{\\onsetRecDeep}}{{{onset_deep['Recall']:.1f}}}\n")
        f.write(f"\\newcommand{{\\onsetPrecPers}}{{{onset_pers['Precision']:.1f}}}\n")
        f.write(f"\\newcommand{{\\onsetPrecClim}}{{{onset_clim['Precision']:.1f}}}\n")
        f.write(f"\\newcommand{{\\onsetPrecLr}}{{{onset_lr['Precision']:.1f}}}\n")
        f.write(f"\\newcommand{{\\onsetPrecCnn}}{{{onset_cnn['Precision']:.1f}}}\n")
        f.write(f"\\newcommand{{\\onsetPrecHgb}}{{{onset_hgb['Precision']:.1f}}}\n")
        f.write(f"\\newcommand{{\\onsetPrecDeep}}{{{onset_deep['Precision']:.1f}}}\n")
        f.write(f"\\newcommand{{\\onsetFlaggedLr}}{{{onset_lr['Flagged_Pct']:.1f}}}\n")
        f.write(f"\\newcommand{{\\onsetFlaggedCnn}}{{{onset_cnn['Flagged_Pct']:.1f}}}\n")
        f.write(f"\\newcommand{{\\onsetFlaggedHgb}}{{{onset_hgb['Flagged_Pct']:.1f}}}\n")
        f.write(f"\\newcommand{{\\onsetFlaggedDeep}}{{{onset_deep['Flagged_Pct']:.1f}}}\n")
        f.write(f"\\newcommand{{\\onsetRecMatchedDeep}}{{{onset_deep['Recall_Matched']:.1f}}}\n")
        f.write(f"\\newcommand{{\\onsetPrecMatchedDeep}}{{{onset_deep['Precision_Matched']:.1f}}}\n")
        f.write(f"\\newcommand{{\\onsetRecMatchedHgb}}{{{onset_hgb['Recall_Matched']:.1f}}}\n")
        f.write(f"\\newcommand{{\\onsetPrecMatchedHgb}}{{{onset_hgb['Precision_Matched']:.1f}}}\n")
        f.write(f"\\newcommand{{\\onsetRecMatchedCnn}}{{{onset_cnn['Recall_Matched']:.1f}}}\n")
        f.write(f"\\newcommand{{\\onsetPrecMatchedCnn}}{{{onset_cnn['Precision_Matched']:.1f}}}\n")
        f.write(f"\\newcommand{{\\prcpZeroPct}}{{{(df['prcp_mm'] == 0.0).mean() * 100:.1f}}}\n")
        f.write(f"\\newcommand{{\\brierClim}}{{{metrics_clim['Brier']:.4f}}}\n")
        f.write(f"\\newcommand{{\\brierPers}}{{{metrics_pers['Brier']:.4f}}}\n")
        f.write(f"\\newcommand{{\\brierLr}}{{{metrics_lr['Brier']:.4f}}}\n")
        f.write(f"\\newcommand{{\\brierCnn}}{{{metrics_cnn['Brier']:.4f}}}\n")
        f.write(f"\\newcommand{{\\brierHgb}}{{{metrics_hgb['Brier']:.4f}}}\n")
        f.write(f"\\newcommand{{\\brierDeep}}{{{metrics_deep['Brier']:.4f}}}\n")
        f.write(f"\\newcommand{{\\bssPers}}{{{metrics_pers['BSS']:.2f}}}\n")
        f.write(f"\\newcommand{{\\bssLr}}{{{metrics_lr['BSS']:.2f}}}\n")
        f.write(f"\\newcommand{{\\bssCnn}}{{{metrics_cnn['BSS']:.2f}}}\n")
        f.write(f"\\newcommand{{\\bssHgb}}{{{metrics_hgb['BSS']:.2f}}}\n")
        f.write(f"\\newcommand{{\\bssDeep}}{{{metrics_deep['BSS']:.2f}}}\n")
        f.write(f"\\newcommand{{\\bssRecalPers}}{{{metrics_pers['BSS_Recal']:.2f}}}\n")
        f.write(f"\\newcommand{{\\bssRecalClim}}{{{metrics_clim['BSS_Recal']:.2f}}}\n")
        f.write(f"\\newcommand{{\\bssRecalLr}}{{{metrics_lr['BSS_Recal']:.2f}}}\n")
        f.write(f"\\newcommand{{\\bssRecalCnn}}{{{metrics_cnn['BSS_Recal']:.2f}}}\n")
        f.write(f"\\newcommand{{\\bssRecalHgb}}{{{metrics_hgb['BSS_Recal']:.2f}}}\n")
        f.write(f"\\newcommand{{\\bssRecalDeep}}{{{metrics_deep['BSS_Recal']:.2f}}}\n")

    # LaTeX Table
    all_models = [
        ("Naive Persistence", metrics_pers),
        ("Monthly Climatology", metrics_clim),
        ("Logistic Regression ($t$)", metrics_lr),
        ("1D Temporal ConvNet (CNN)", metrics_cnn),
        ("HistGradientBoosting (7d)", metrics_hgb),
        ("Temporal Transformer (This Study)", metrics_deep),
    ]

    best_auc = max(m["AUC"] for _, m in all_models)
    best_pr_auc = max(m["PR_AUC"] for _, m in all_models)
    best_f1 = max(m["F1"] for _, m in all_models)
    best_rec = max(m["Recall"] for _, m in all_models)
    best_prec = max(m["Precision"] for _, m in all_models)
    best_onset_rec = max(m["Onset_Recall"] for _, m in all_models)
    best_brier = min(m["Brier"] for _, m in all_models)
    best_bss = max(m["BSS"] for _, m in all_models)
    best_bss_recal = max(m["BSS_Recal"] for _, m in all_models)

    def format_cell(val, best_val, fmt="{:.3f}", is_min=False):
        is_best = (val <= best_val + 1e-5) if is_min else (val >= best_val - 1e-5)
        formatted = fmt.format(val)
        return f"\\textbf{{{formatted}}}" if is_best else formatted

    with open(TAB_DIR / "tab_benchmark.tex", "w") as f:
        f.write(r"\begin{table*}[t]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Chronological Out-of-Sample Evaluation on NOAA GHCN-Daily Stations (Test Period: 2025--2026).}" + "\n")
        f.write(r"\label{tab:extreme_results}" + "\n")
        f.write(r"\resizebox{\textwidth}{!}{%" + "\n")
        f.write(r"\begin{tabular}{lccccccccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"\textbf{Architecture} & \textbf{ROC-AUC} & \textbf{PR-AUC} & \textbf{F1 (\%)} & \textbf{Precision (\%)} & \textbf{Recall (\%)} & \textbf{Onset Recall (\%)} & \textbf{Brier Score} & \textbf{BSS (Raw)} & \textbf{BSS (Recal.)} \\" + "\n")
        f.write(r"\midrule" + "\n")
        for name, m in all_models:
            c_auc = format_cell(m["AUC"], best_auc, "{:.3f}")
            c_pr = format_cell(m["PR_AUC"], best_pr_auc, "{:.3f}")
            c_f1 = format_cell(m["F1"], best_f1, "{:.1f}")
            c_prec = format_cell(m["Precision"], best_prec, "{:.1f}")
            c_rec = format_cell(m["Recall"], best_rec, "{:.1f}")
            c_onset = format_cell(m["Onset_Recall"], best_onset_rec, "{:.1f}")
            c_br = format_cell(m["Brier"], best_brier, "{:.4f}", is_min=True)
            c_bss = format_cell(m["BSS"], best_bss, "{:.2f}")
            c_bss_recal = format_cell(m["BSS_Recal"], best_bss_recal, "{:.2f}")
            f.write(f"{name} & {c_auc} & {c_pr} & {c_f1} & {c_prec} & {c_rec} & {c_onset} & {c_br} & {c_bss} & {c_bss_recal} " + r"\\" + "\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}}" + "\n")
        f.write(r"\end{table*}" + "\n")

    # Save results summary JSON
    results_summary = {
        "n_stations": int(len(df["station"].unique())),
        "total_days": int(total_days),
        "total_extremes": int(total_extremes),
        "prevalence_rate_pct": float(total_extremes / total_days * 100.0),
        "onset_events": int(onset_deep["n_pos"]),
        "onset_eval_days": int(onset_deep["n_sub"]),
        "models": {
            name: {k: float(v) for k, v in m.items()}
            for name, m in all_models
        },
        "bootstrap_diffs": {
            "deep_minus_hgb": {
                "onset_recall_ci_95": list(ci_rec_deep_hgb),
                "onset_prauc_ci_95": list(ci_prauc_deep_hgb),
            },
            "deep_minus_cnn": {
                "onset_recall_ci_95": list(ci_rec_deep_cnn),
                "onset_prauc_ci_95": list(ci_prauc_deep_cnn),
            }
        }
    }
    with open(EXP_DIR / "results_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print("Saved publication artifacts and results_summary.json successfully.")


if __name__ == "__main__":
    run_pipeline()
