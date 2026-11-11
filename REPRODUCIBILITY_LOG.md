# Reproducibility Log: Multi-Hazard Extreme Weather Detection

**Manuscript Title**: Deep Temporal Sequence Modeling for Multi-Hazard Extreme Weather Detection across Ground Observation Networks  
**Authors**: Cristhian David Cáceres Mateus, Talha Ali Khan (corresponding author: talhaali.khan@ue-germany.de)  
**Affiliation**: Department of Business, University of Europe for Applied Sciences, 14469 Potsdam, Germany  
**Holdout Horizon**: 2025--2026 (Training: 2020--2023, Validation: 2024)  
**Log Updated**: October 2026  

---

## 1. Reproduction Steps

To execute the data preparation, model training, evaluation, table/macro generation, and compile the manuscript:

```bash
# 1. Environment requirements
pip install -r Extreme_Weather_Reanalysis_Fusion/requirements.txt

# 2. Run the test suite
pytest Extreme_Weather_Reanalysis_Fusion/tests

# 3. Execute the full training and evaluation pipeline
python Extreme_Weather_Reanalysis_Fusion/src/train_extreme_weather.py

# 4. Compile the manuscript PDF
tools/tectonic.exe Extreme_Weather_Reanalysis_Fusion/paper/main.tex --outdir Extreme_Weather_Reanalysis_Fusion/paper/

# 5. Lint manuscript prose
tools/vale.exe Extreme_Weather_Reanalysis_Fusion/paper/main.tex
```

---

## 2. Data Sources and Provenance

- **Data Source**: NOAA National Centers for Environmental Information (NCEI) Global Historical Climatology Network-Daily (GHCN-Daily).
- **Stations Evaluated**: 8 diverse airport climate stations representing contrasting US climate biomes:
  - Phoenix Sky Harbor, AZ (USW00023183): Sonoran desert extreme heat
  - Las Vegas McCarran, NV (USW00023169): Arid high-heat
  - Houston Intercontinental, TX (USW00012960): Subtropical precipitation deluge
  - Miami International, FL (USW00012839): Tropical convective storms
  - Denver International, CO (USW00003017): High-altitude freeze transitions
  - Minneapolis-St Paul, MN (USW00014922): Sub-zero polar freeze surges
  - Seattle-Tacoma, WA (USW00024233): Atmospheric river rainfall
  - Chicago O'Hare, IL (USW00094846): Great Lakes freeze and gale gusts
- **Observation Count**: 19,856 total station-days (January 1, 2020 through October 17, 2026).
- **Imputation Rules**: Missing precipitation zero-filled (`fillna(0.0)`), temperature forward-filled, peak wind gust speed WSFG used when coverage exceeds 30%, falling back to average daily wind speed AWND.
- **Multi-Hazard Definition**:
  $$y_t = \mathbb{I}((T_{\max, t} \ge 38^\circ\text{C}) \lor (T_{\min, t} \le -15^\circ\text{C}) \lor (P_t \ge 25\text{ mm}) \lor (v_{\text{gust}, t} \ge 20\text{ m/s}))$$
- **Prevalence**: 2,163 hazard days (10.9% overall prevalence; 1,489 heat, 299 freeze, 375 deluge, 1 wind extreme).

---

## 3. Evaluation Protocol and Baseline Suite

1. **Chronological Splitting**:
   - Training: 2020--2023 (11,632 sequences)
   - Validation: 2024 (2,928 sequences)
   - Chronological Test Holdout: 2025--2026 (5,240 sequences)
2. **Baselines**:
   - Naive Persistence: $y_{t+1} = y_t$
   - Monthly Climatology: Historical station-month empirical base rates
   - Logistic Regression: Single-day terminal features ($t$)
   - 1D Temporal ConvNet (CNN): Attention-free 2-layer temporal convolutional network (kernel 3, 64 filters)
   - HistGradientBoosting (HistGBM): Tabular tree ensemble over flattened 7-day lookback window
   - Temporal Transformer: 2-layer multi-head self-attention encoder over 7-day sequences
3. **Onset Transition Metric**:
   - Evaluated specifically on transitions where day $t$ is non-extreme ($y_t = 0$) and day $t+1$ is extreme ($y_{t+1} = 1$), isolating onset detection from autocorrelation ($N = 172$ test onsets across 4,634 non-extreme days).
   - Paired bootstrap test (1,000 replicates) reports onset recall gain under matched alert budgets over HistGBM: +13.3 percentage points (95% CI: [7.3%, 19.2%]), while showing comparable skill to 1D-CNN (95% CI: [-4.3%, 6.4%]).
4. **Brier Skill Score and Platt Recalibration**:
   - Raw scores yield negative BSS due to class-weighting shift ($w_{\text{pos}} = 8.5$).
   - Post-hoc Platt scaling fitted on 2024 validation logits restores positive skill on 2025--2026 test set:
     - Temporal Transformer: $\text{BSS}_{\text{recal}} = 0.27$
     - HistGBM: $\text{BSS}_{\text{recal}} = 0.25$
     - 1D-CNN: $\text{BSS}_{\text{recal}} = 0.19$
     - Climatology: $\text{BSS} = 0.00$

---

## 4. Verification and Artifacts

- **Pipeline Execution**: `src/train_extreme_weather.py` writes macros to `paper/tables/macros.tex`, table `paper/tables/tab_benchmark.tex`, figure `paper/figures/fig1_extreme_weather_benchmark.png`, and summary metrics to `experiments/results_summary.json`.
- **Test Suite**: 8 passed unit and integration tests (`pytest Extreme_Weather_Reanalysis_Fusion/tests`).
- **LaTeX Compilation**: Compiles via Tectonic with 0 fatal errors (791.6 KiB output).
- **Prose Linter**: Vale reports 0 errors, 0 warnings.
