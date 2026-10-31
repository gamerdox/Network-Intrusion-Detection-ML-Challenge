#!/usr/bin/env python3
"""Network Intrusion Detection ML Challenge (GDG on Campus) - final pipeline.

Usage:
    python train_and_predict.py --data-dir data --out-dir submissions_out

Writes two prediction files (the two final Kaggle submissions):
    A_hgb_hybrid.csv  - stage 2 = regularized HistGradientBoosting on all features
    B_rf_stage2.csv   - stage 2 = RandomForest trained only on Benign/Reconnaissance/Spoofing

Both share stage 1 (Availability detector built from sender-side features only).
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

from features import CLASSES, build_features, load_data, sender_features

HGB_PARAMS = dict(max_iter=150, learning_rate=0.05, max_leaf_nodes=8,
                  l2_regularization=5, min_samples_leaf=30, random_state=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--out-dir', default='submissions_out')
    ap.add_argument('--only', choices=['A', 'B', 'both'], default='both', help='which final model to run')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    train, test, y = load_data(args.data_dir)
    w_all = compute_sample_weight('balanced', y)          # rare classes count as much as common ones

    # ---- Stage 1: Availability vs everything else, from sender-side features only
    stage1 = HistGradientBoostingClassifier(**HGB_PARAMS).fit(sender_features(train).values, y, sample_weight=w_all)
    is_availability = stage1.predict_proba(sender_features(test).values)[:, 0] >= 0.5

    def finish(p3, name):
        """p3 = probabilities for [Benign, Reconnaissance, Spoofing] -> final labels."""
        idx = np.where(is_availability, 0, 1 + p3.argmax(1))
        pred = np.array(CLASSES)[idx]
        pd.DataFrame({'id': test['id'], 'Target': pred}).to_csv(os.path.join(args.out_dir, name), index=False)
        print(name, pd.Series(pred).value_counts().to_dict())

    # ---- Stage 2, model A: HistGradientBoosting on all rows, all features (no ports)
    if args.only in ('A', 'both'):
        Xa_tr, Xa_te = build_features(train), build_features(test)
        model_a = HistGradientBoostingClassifier(**HGB_PARAMS).fit(Xa_tr.values, y, sample_weight=w_all)
        finish(model_a.predict_proba(Xa_te.values)[:, 1:], 'A_hgb_hybrid.csv')

    # ---- Stage 2, model B: RandomForest on the three non-Availability classes only
    if args.only in ('B', 'both'):
        Xb_tr, Xb_te = build_features(train, port_extras=True), build_features(test, port_extras=True)
        keep = y != 0
        y3 = y[keep]
        model_b = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, max_features=0.4,
                                         n_jobs=-1, random_state=0)
        model_b.fit(Xb_tr.values[keep], y3, sample_weight=compute_sample_weight('balanced', y3))
        finish(model_b.predict_proba(Xb_te.values), 'B_rf_stage2.csv')

if __name__ == '__main__':
    main()
