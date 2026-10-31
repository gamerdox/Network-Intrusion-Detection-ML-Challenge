#!/usr/bin/env python3
"""Validation that mimics the competition's "unseen traffic behaviour" twist.

A random train/validation split is misleading here: near-identical flows land on both sides and scores look
~0.99. Instead we:
  1. cluster the flows INSIDE each class (KMeans) -> each cluster is a pseudo "behaviour profile"
  2. hold out whole clusters (StratifiedGroupKFold) so the model is scored on profiles it never saw
  3. report macro-F1 (the competition metric) for several design choices

Usage:  python validate_grouped_cv.py --data-dir data [--groupings 1]
(Roughly 6-8 minutes per grouping on a laptop.)
"""
import argparse
import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from features import CLASSES, build_features, load_data, sender_features
from train_and_predict import HGB_PARAMS

warnings.filterwarnings('ignore')


def make_groups(train, y, clusters_per_class, seed):
    X = build_features(train, keep_ports=True).values.astype(float)
    Z = StandardScaler().fit_transform(np.log1p(np.abs(X)))
    groups = np.zeros(len(y), int)
    for c in range(len(CLASSES)):
        m = y == c
        groups[m] = c * 100 + KMeans(clusters_per_class, n_init=5, random_state=seed).fit_predict(Z[m])
    return groups


def macro_f1(y, pred):
    return f1_score(y, pred, average='macro')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--groupings', type=int, default=1, help='how many different cluster layouts to average (1-3)')
    args = ap.parse_args()

    train, _, y = load_data(args.data_dir)
    X_noports = build_features(train).values
    X_ports = build_features(train, keep_ports=True).values
    X_b = build_features(train, port_extras=True).values
    X_sender = sender_features(train).values
    scores = {k: [] for k in ['HGB with raw port numbers (1 stage)', 'HGB without ports (1 stage)',
                              'Model A (2 stages, HGB)', 'Model B (2 stages, RandomForest)']}

    for g, (k, seed) in enumerate([(5, 0), (4, 1), (6, 2)][:args.groupings]):
        groups = make_groups(train, y, k, seed)
        oof = {name: np.zeros((len(y), 4)) for name in scores}
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X_noports, y, groups):
            w = compute_sample_weight('balanced', y[a])
            fit = lambda X: HistGradientBoostingClassifier(**HGB_PARAMS).fit(X[a], y[a], sample_weight=w)
            p_ports = fit(X_ports).predict_proba(X_ports[b])
            p_noports = fit(X_noports).predict_proba(X_noports[b])
            p_av = fit(X_sender).predict_proba(X_sender[b])[:, 0]
            a3 = a[y[a] != 0]
            rf = RandomForestClassifier(500, min_samples_leaf=2, max_features=0.4, n_jobs=-1, random_state=0)
            rf.fit(X_b[a3], y[a3], sample_weight=compute_sample_weight('balanced', y[a3]))
            p_rf = rf.predict_proba(X_b[b])

            def two_stage(p3):                      # same combination rule as train_and_predict.py
                out = np.zeros((len(b), 4)); out[:, 1:] = p3
                out[p_av >= 0.5] = [1, 0, 0, 0]
                return out
            oof['HGB with raw port numbers (1 stage)'][b] = p_ports
            oof['HGB without ports (1 stage)'][b] = p_noports
            oof['Model A (2 stages, HGB)'][b] = two_stage(p_noports[:, 1:])
            oof['Model B (2 stages, RandomForest)'][b] = two_stage(p_rf)
        for name in scores:
            scores[name].append(macro_f1(y, oof[name].argmax(1)))
        print(f'grouping {g + 1} done', flush=True)

    print('\nMacro-F1 on held-out behaviour profiles (higher is better):')
    for name, v in scores.items():
        print(f'  {name:40s} {np.mean(v):.4f}')
    print('\nNote: this proxy is deliberately harsh; the real public leaderboard score is higher.')


if __name__ == '__main__':
    main()
