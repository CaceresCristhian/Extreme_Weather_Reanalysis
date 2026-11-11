# Deep Temporal Sequence Modeling for Multi-Hazard Extreme Weather Detection

## Overview
This repository contains the implementation, empirical pipeline, and benchmark artifacts for multi-hazard extreme weather detection across diverse US climate zones using ground-sensor records from NOAA GHCN-Daily.

**Authors**: Cristhian David Cáceres Mateus, Talha Ali Khan (corresponding author: talhaali.khan@ue-germany.de)  
**Affiliation**: Department of Business, University of Europe for Applied Sciences, 14469 Potsdam, Germany  

## Data Provenance
- **Source**: NOAA National Centers for Environmental Information (NCEI) Global Historical Climatology Network Daily (GHCN-Daily).
- **Network**: 8 airport observation stations (2020-01-01 through 2026-10-17):
  - Phoenix Sky Harbor, AZ (`USW00023183`): Desert extreme heat
  - Las Vegas McCarran, NV (`USW00023169`): Arid high-heat benchmark
  - Houston Intercontinental, TX (`USW00012960`): Subtropical flood / heavy precipitation
  - Miami International, FL (`USW00012839`): Tropical storm / maritime humidity
  - Denver International, CO (`USW00003017`): High altitude cold front transitions
  - Minneapolis-St Paul, MN (`USW00014922`): Polar vortex / sub-zero freeze surges
  - Seattle-Tacoma, WA (`USW00024233`): Atmospheric river persistent rain
  - Chicago O'Hare, IL (`USW00094846`): Great Lakes freeze / wind extremes
- **Sample Count**: 19,856 daily station observations.
- **Extreme Event Frequency**: 2,163 hazard days (10.89% prevalence):
  - Heatwave ($T_{\max} \ge 38^\circ\text{C}$): 1,489 days
  - Freeze ($T_{\min} \le -15^\circ\text{C}$): 299 days
  - Precipitation ($P \ge 25\text{ mm}$): 375 days
  - Severe Wind ($v_{\text{gust}} \ge 20\text{ m/s}$): 1 day

## Chronological Experimental Protocol
- **Lookback Window**: $L = 7$ days to predict hazard occurrence on day $t+1$.
- **Training Set**: 2020-01-01 to 2023-12-31 (11,632 sequences).
- **Validation Set**: 2024-01-01 to 2024-12-31 (2,928 sequences).
- **Chronological Test Holdout**: 2025-01-01 to 2026-10-17 (5,240 sequences).

## Empirical Benchmark Results (Chronological Test Holdout: 2025–2026)
| Architecture | ROC-AUC | PR-AUC | F1 (%) | Precision (%) | Recall (%) | Onset Recall (%) | Brier | BSS (Raw) | BSS (Recal.) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Naive Persistence | 0.840 | 0.546 | **71.6** | 71.6 | 71.6 | 0.0 | 0.0656 | -0.22 | -0.22 |
| Monthly Climatology | 0.897 | 0.698 | 64.1 | **84.8** | 51.5 | 12.2 | **0.0539** | **0.00** | 0.00 |
| Logistic Regression ($t$) | 0.739 | 0.385 | 31.2 | 19.6 | 77.6 | **58.7** | 0.2213 | -3.10 | -0.64 |
| 1D Temporal ConvNet (CNN) | 0.929 | 0.770 | 63.7 | 52.1 | 82.0 | 41.9 | 0.0835 | -0.55 | 0.19 |
| HistGradientBoosting (7d) | **0.936** | 0.821 | 68.7 | 60.9 | 78.7 | 29.7 | 0.0681 | -0.26 | 0.25 |
| **Temporal Transformer** | 0.935 | **0.828** | 66.7 | 55.7 | **83.0** | 43.0 | 0.0689 | -0.28 | **0.27** |

*Key Findings*:
1. **Hazard Onset Lift**: On transitions from non-extreme days ($N = 172$ onsets across 4,634 test days), the Temporal Transformer achieves 43.0% onset recall vs. 29.7% for HistGBM (+13.3 pp unconstrained, 7-day moving-block bootstrap 95% CI: [7.3%, 19.2%]). When evaluated at a matched 7.5% alert budget, the Transformer achieves 43.0% onset recall (21.3% precision) vs. 34.3% (17.0% precision) for HistGBM, maintaining an 8.7 percentage point lead under identical alert constraints.
2. **Attention vs Convolutions**: The 1D-CNN sequence baseline achieves 41.9% onset recall (40.1% matched, difference vs. Transformer 95% CI: [-4.3%, 6.4%]), indicating that explicit multi-step sequence encodings provide competitive onset sensitivity relative to flattened tree feature tables.
3. **Platt Recalibration**: Post-hoc probability recalibration on 2024 validation logits eliminates class-weighting distortions, restoring positive Brier Skill Scores across models (Transformer: 0.27, HistGBM: 0.25, 1D-CNN: 0.19 vs Climatology 0.00).

## Reproduction Commands
```bash
pip install -r requirements.txt
python src/train_extreme_weather.py
pytest tests/test_pipeline.py
tools/tectonic.exe paper/main.tex --outdir paper/
tools/vale.exe paper/main.tex
```
