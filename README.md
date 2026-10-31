# Network Intrusion Detection — GDG on Campus AI/ML Challenge

Multiclass classification of network-traffic flows into **Benign, Availability, Reconnaissance, Spoofing**
(metric: **macro-F1**). Kaggle competition: *network-intrusion-detection2-ml-challenge*.

## Result

| Final submission | Public leaderboard (macro-F1) |
|---|---|
| `A_hgb_hybrid.csv` | **0.96776** |
| `B_rf_stage2.csv`  | 0.96590 |

The public leaderboard covers only part of the hidden test set; the private part contains different traffic
profiles, so the final ranking may differ.

## The challenge: the test traffic is different from the training traffic

The organizers deliberately hold out traffic *behaviour profiles*, so a model that looks perfect on a random
validation split can fail on the test set. Two concrete findings drove the design:

1. **Train and test are almost perfectly separable** (a classifier can tell them apart with AUC ≈ 0.9995).
   The cause is timing/rate columns (e.g. `payload_bytes_per_second`, `*_pkts_per_sec`): in train they are
   enormous for the dominant *Availability* class (a timestamp artefact of extremely short flows) and are
   completely different in test.
2. **About 31 % of the test rows carry the same SYN-flood signature as train *Availability*** (one SYN packet,
   20-byte header, 120-byte payload) **but mostly with no reply from the target.** Models that rely on reply
   features (backward packets, ACK/RST counts, rates) label those rows Benign/Reconnaissance.

## Approach (two stages)

```
                     +--> sender-side features only --> Stage 1: Availability? (HistGradientBoosting)
flow record ---------+                                           | yes -> Availability
                     |                                           | no
                     +--> all features -----------------------> Stage 2: Benign / Reconnaissance / Spoofing
                                                                  A: HistGradientBoosting   B: RandomForest
```

* **Stage 1 – Availability detector.** Uses only forward-direction (`fwd_*`) features, the SYN count, protocol
  and service, so the decision does not depend on whether the target replied or on timing artefacts.
* **Stage 2 – the other three classes.**
  * **Model A:** regularized HistGradientBoosting (8 leaves, L2 = 5, learning rate 0.05) on all features.
  * **Model B:** RandomForest (500 trees) trained *only* on Benign/Reconnaissance/Spoofing rows, with extra
    features: destination-port range flags and a within-file port-rarity score (scanners touch many rare ports).
* **Common choices:** class-balanced sample weights (macro-F1 treats every class equally); raw port numbers
  removed as a robustness choice (see note in the table below); engineered ratios (forward/backward packets and payload,
  header/payload per packet, SYN/RST/ACK/FIN per packet, "no reply" flag, ephemeral-source-port flag).

## Validation (why a random split is not used)

`validate_grouped_cv.py` clusters the flows *inside each class* with KMeans — every cluster acts as a
pseudo "behaviour profile" — and then holds out **whole clusters** (StratifiedGroupKFold), so each model is scored
on profiles it has never seen.

Result of `python validate_grouped_cv.py` (one cluster layout, 5 folds, macro-F1). The cluster layout depends on the
library versions, so the numbers change between environments; two real runs are shown:



Differences between variants of up to ~0.1 change sign from one run to the other, so this validation **cannot reliably
rank the variants**. What it does show is that every variant lands around 0.64-0.74 on unseen behaviour profiles, far
below the ~0.99 that a random split would suggest. The design choices therefore rest mainly on the train-versus-test
analysis above and on the public leaderboard (A 0.96776, B 0.96590), not on small validation differences.

This proxy is intentionally harsh and cannot see every kind of shift (for example it did not reveal the
"no-reply" problem above — that was found by comparing train and test feature distributions), so it is used only as a rough
sanity check, not as a ranking of the variants or a prediction of the leaderboard score.

### What helped / what did not

| Idea | Effect (grouped validation) |
|---|---|
| Drop raw port numbers | helped in early runs with default model settings (0.668 → 0.713); with the final settings the sign flips between runs (table above), so this is a robustness choice (ports can memorize training profiles), not a proven gain |
| Strong regularization (few leaves, L2) | helped |
| Class-balanced weights | needed for macro-F1 |
| Sender-side Availability detector | fixes ~10.4k test rows; major leaderboard gain (not measurable in validation) |
| Train stage 2 on the 3 non-Availability classes only | RandomForest looked best in a 3-layout sweep of this stage, but not in every run (table above) |
| Undersampling the Availability class | hurt |
| LightGBM, ExtraTrees, logistic regression, simple blends | no gain / worse |
| Self-training on pseudo-labelled test rows | no gain |
| Dropping *all* timing features | much worse |
| Averaging predictions inside clusters of similar test flows | no clear gain |

## Reproduce

```bash
pip install -r requirements.txt
# download train.csv and test.csv from the Kaggle competition into ./data
python train_and_predict.py --data-dir data --out-dir submissions_out   # writes A_ and B_ prediction files
python validate_grouped_cv.py --data-dir data                          # optional: the grouped validation
```

`--only A` or `--only B` runs a single model. The notebooks in `colab/` (`Colab_Model_A.ipynb`, `Colab_Model_B.ipynb`)
reproduce each submission in Google Colab: they install the exact library versions from `requirements-exact.txt` and
then check that the regenerated predictions are identical to the submitted file.

`submissions/` contains the two prediction files that were actually submitted. With the exact versions in
`requirements-exact.txt` (Python 3.13, scikit-learn 1.9.1, pandas 3.0.5, numpy 2.5.3), `train_and_predict.py` regenerates them
byte-for-byte (fixed random seeds). With other library versions (for example Google Colab) about 0.3 % of rows
(roughly 100 of 33,161) come out differently; the files in `submissions/` are the official ones.

## Repository layout

```
features.py              feature engineering (full and sender-side)
train_and_predict.py     final pipeline -> the two submissions
validate_grouped_cv.py   grouped (held-out-profile) validation
colab/                   one Colab notebook per final model (reproduces the submitted predictions)
requirements-exact.txt   exact library versions used for the submitted files
submissions/             the two submitted prediction files
data/                    place the Kaggle CSVs here (not committed)
```

## Rules compliance and notes

* Only the provided competition data was used. No external labelled data, no manual labelling of test rows,
  no attempt to obtain hidden labels.
* Unlabeled test **features** were used in two places: to compare train/test feature distributions while
  diagnosing the shift, and for the within-file port-rarity feature (model B). No test labels are involved anywhere.
* This solution was developed with the help of an AI assistant (Claude, Anthropic), which the competition rules permit.
* Limitations: results on the private leaderboard may differ because it holds different traffic profiles; the
  validation proxy is approximate; the stage-1 threshold (0.5) and model settings were chosen on the training data only.
