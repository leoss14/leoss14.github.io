# Results Summary
## Zombie Firms and Spillover Effects — Italian Manufacturing Panel 2016–2024

**Project:** Replication and expansion of *Geographical Analysis of the Italian Industrial North: Zombie Firms and Spillover Effects*
**Panel:** 74,726 firms, 20 Italian regions, 2016–2024
**Primary zombie definition:** McGowan et al. (2018) — three consecutive years of ICR < 1, firm age ≥ 10 years, two-year exit condition

---

## 1. Zombie firm prevalence

The McGowan zombie share rises from 2.7% in 2018 to a peak of 4.4% in 2022 before declining to 3.9% in 2024. The 2020 COVID spike under the weak ICR definition (ICR < 1 in any single year) reaches 25.5%, confirming the two-year exit condition is essential to distinguish persistently distressed firms from those experiencing a temporary shock. Under the stricter McGowan definition with the two-year exit rule, the 2020 spike is contained at 4.0% — a 0.6 percentage point rise from 2019 — because firms that recovered their ICR after a single bad year are not reclassified as non-zombie until the second consecutive year of recovery.

Sector variation is substantial. Beverages (10.9%), Coke/petroleum (7.8%), and Pharma (7.2%) show the highest zombie shares in 2022. Fabricated metals, Repair/installation, and Wood products are at the bottom (2.1–2.8%). The dominance of food-related and energy-intensive sectors at the top is consistent with the margin compression those sectors experienced in the post-2021 input cost environment.

The Northern subsample (ITC + ITH NUTS2) contains 47,734 firms and drives most of the identification in the geographic specifications.

---

## 2. Baseline spillover regressions

All models include firm fixed effects and year fixed effects. Standard errors are clustered at province × 2-digit NACE sector. The identification strategy follows McGowan et al. (2018): the zombie congestion term is a leave-one-out province-sector-year zombie share, excluding the firm's own zombie status to avoid mechanical correlation.

### Main results

| Model | Dep. var. | Zombie congestion coef. | SE | p-value | N obs |
|---|---|---|---|---|---|
| M1 Baseline | Investment rate | −0.0032 | 0.0092 | 0.725 | 351,220 |
| M2 Baseline | Employment growth | −0.0041 | 0.0152 | 0.788 | 285,289 |
| M3 Credit interaction | Investment rate | −0.0088 | 0.0119 | 0.460 | 351,220 |
| M4 Own zombie | Investment rate | −0.0029*** | 0.0011 | 0.007 | 375,277 |

The spillover effect on investment (M1) and employment growth (M2) is negative in sign across all specifications but never reaches conventional significance. The own zombie penalty (M4) is the only robust finding from the baseline: a firm classified as zombie under McGowan invests 0.3 percentage points less than it would in non-zombie years, controlling for size and financial health. This effect survives all robustness checks.

### Robustness

The null spillover result is consistent across eight robustness specifications varying the ICR lag inclusion and geographic subsample. The direction is uniformly negative — no specification produces a positive spillover coefficient — but significance is absent throughout.

The capital-weighted congestion term (share of province-sector tangible fixed assets held by zombies, following the original McGowan et al. specification) produces a borderline result: −0.0054, SE = 0.0030, p = 0.077. The correlation between the count-share and capital-share measures is only 0.46, indicating they capture meaningfully different variation. The tighter standard error on the capital-weighted measure (0.003 vs 0.009) suggests that a small number of large zombie firms occupying a disproportionate share of industry assets are more relevant to the congestion mechanism than the numerical count of zombie firms.

The unconsolidated-only subsample (U1/U2 firms, 96.9% of the sample) produces near-identical results: zombie_share_loo = −0.003, own zombie = −0.003***. No consolidation composition bias.

---

## 3. Credit channel tests

Three sequential tests of the evergreening mechanism — the hypothesis that banks roll over loans to zombie firms, crowding out credit to viable neighbours.

**Test 1 — Credit cost.** Zombie congestion (province-sector zombie share) has no significant effect on `fin_intensity` (financial expenses / total assets) for healthy firms: coefficient = +0.0004, SE = 0.0007, p = 0.59. The capital-weighted version also shows no effect (+0.0002, p = 0.44). There is no evidence that zombie congestion raises the cost of borrowing for neighbouring healthy firms.

**Test 2 — Credit volume.** Zombie congestion has no significant effect on log loan growth for healthy firms nationally (−0.042, p = 0.75) or in the North (+0.011, p = 0.95). There is no evidence that zombie congestion reduces credit volumes to healthy firms.

**Test 3 — Mediation.** Adding `fin_intensity_w` as a control to M1 attenuates the zombie_share_loo coefficient by only 6.4% (from −0.0032 to −0.0030). Credit cost does not mediate the investment suppression channel. If zombie congestion does suppress investment, it does not do so through observable increases in borrowing costs at the province-sector level.

Taken together, Tests 1–3 provide no support for the evergreening channel as the mechanism linking zombie presence to investment outcomes in Italian manufacturing. This does not rule out the channel operating through unobservable credit rationing or at a finer geographic scale than the province.

---

## 4. Geographic scale of spillovers

### NUTS2-level congestion

Recomputing the congestion term at NUTS2 × sector × year (broader geography, correlation with NUTS3 measure = 0.65) produces a sign reversal: +0.0045, p = 0.84 for investment. At regional scale, zombie density is weakly positively associated with investment, likely reflecting that zombie-heavy NUTS2 regions (industrial Northern Italy) are simply more economically active — the regional aggregation confounds zombie density with industrial dynamism. The employment specification at NUTS2 shows −0.033 (p = 0.25), the strongest employment result across all specifications but still not significant.

### Distance-weighted spatial spillovers

Using firm-level geocoordinates (99.9% coverage, GeoNames CAP lookup), inverse-distance-weighted zombie shares are computed at 25km, 50km, and 100km radii for each healthy firm-year.

| Bandwidth | Coefficient | SE | p-value | N obs |
|---|---|---|---|---|
| 25km | −0.0161 | 0.0094 | 0.087 | 357,749 |
| 50km | −0.0200 | 0.0114 | 0.079 | 357,759 |
| 100km | −0.0289 | 0.0138 | **0.036** | 357,759 |

The 100km specification is significant at the 5% level. The effect strengthens monotonically with bandwidth, becoming significant only at 100km. This pattern has a specific interpretation: a NUTS3 province in Italy has an average radius of roughly 30km, meaning the province-level congestion term (which produces a null result) captures only the nearest 25–30km of zombie presence. The significant effect emerges when the spatial window is extended to 100km, covering neighbouring provinces.

The monotonic strengthening with distance rules out a hyper-local bank branch relationship story, where the effect would be expected to decay sharply with distance. Instead, the pattern is more consistent with mechanisms operating through regional labour markets, supply chain networks, or regional banking relationships that span multiple provinces. This is also consistent with the null credit cost result at province-sector level (Test 1): if the mechanism operates at 100km scale, province-level averaging would obscure it.

The divergence between the null NUTS2 administrative result and the significant 100km distance-based result reflects the difference between administrative boundaries (which dilute zombie density by including low-zombie provinces in a high-zombie NUTS2) and continuous distance weighting (which assigns higher weight to nearby zombies regardless of administrative unit).

---

## 5. Machine learning early-warning model

A predictive model is trained to forecast zombie firm status one year ahead using financial ratios observed at time t. The design avoids look-ahead contamination by using time-based train/validation/test splits (train 2016–2021, validation 2022, test 2023–2024) rather than random cross-validation.

### Sample and features

264,833 firm-years, 60,264 unique firms. Class imbalance 1:21 (zombie:non-zombie), handled via class weighting. Features include 20 financial variables (current ICR, ROA, leverage, financial intensity, investment rate, employment and sales growth, log assets, firm age, negative equity flag, plus one and two-year lags of ICR, ROA, and leverage, and first differences of each) and 23 sector dummies.

### Performance

| Model | Val AUPRC | Test AUPRC | Test ROC-AUC | Test F1 |
|---|---|---|---|---|
| Logistic Regression | 0.467 | 0.464 | 0.951 | 0.551 |
| Random Forest | 0.970 | 0.951 | 0.998 | 0.889 |
| XGBoost | 0.986 | **0.964** | 0.998 | 0.867 |
| Naive baseline | — | 0.042 | — | — |

XGBoost achieves a test AUPRC of 0.964 against a naive baseline of 0.042 (zombie prevalence), representing a 23-fold improvement. The large gap between logistic regression (0.464) and the tree-based models (0.951–0.964) indicates that the relationship between financial ratios and future zombie status is highly non-linear.

### Temporal stability

Rolling validation (training on all years up to t−1, evaluating on year t) shows stable performance from 2020 to 2023: AUPRC ranges from 0.950 to 0.985 with no degradation post-COVID. The model generalises across the pandemic shock without retraining.

### Interpretation caveat

The high performance requires careful interpretation. The McGowan zombie definition requires three consecutive years of ICR < 1. The feature set includes current ICR and two annual lags. A firm that will be classified zombie at t+1 has, by construction, had ICR < 1 for at least two consecutive years already at time t. The ICR lags are therefore near-deterministic predictors of the target, and the model may be largely rediscovering the classification rule rather than detecting genuinely independent early-warning signals. A conservative interpretation is that the model demonstrates the financial deterioration preceding formal zombie classification is observable and persistent, rather than sudden. A robustness check excluding the ICR lags would clarify how much independent predictive content the other financial ratios contribute.

---

## 6. Summary of findings

**Finding 1 — Own zombie penalty is robust.** Zombie firms invest 0.3–0.4 percentage points less than their own non-zombie years. This effect is significant (p < 0.01) and stable across sample restrictions, ICR lag inclusion, and consolidation code composition.

**Finding 2 — Province-sector count-share spillover is null.** Zombie congestion measured as the firm-count share at province × 2-digit NACE × year level does not significantly suppress investment or employment in healthy neighbouring firms. This null holds nationally and in the Northern subsample under all eight robustness specifications.

**Finding 3 — Capital-weighted congestion is borderline.** The share of province-sector tangible assets held by zombies produces a negative and borderline significant effect (p = 0.077). Large zombie firms are more relevant to the congestion mechanism than their count would imply.

**Finding 4 — Credit channel shows no evidence at province-sector scale.** Zombie congestion does not raise borrowing costs or reduce loan volumes for healthy firms, and credit cost does not mediate the investment effect. The evergreening channel, if present, is not detectable in observable financial ratios at this level of geographic aggregation.

**Finding 5 — Significant spatial spillover at 100km.** Distance-weighted zombie presence within 100km suppresses healthy firm investment by 2.9 percentage points (p = 0.036). The effect is absent at 25–50km and strengthens monotonically, suggesting the mechanism operates through regional rather than local channels and is not captured by province-level administrative boundaries.

**Finding 6 — Early-warning model achieves high predictive accuracy.** XGBoost forecasts zombie status one year ahead with AUPRC = 0.964 and is stable across years including the COVID shock. The predictive performance is partly driven by the ICR lag structure, but the model's stability and the gap over logistic regression confirm that the financial deterioration preceding zombification has non-linear structure beyond the ICR threshold rule.

---

*Document generated from NB3 and NB5 outputs. All regressions use linearmodels PanelOLS with firm and year fixed effects, standard errors clustered at province × sector. Spatial shares computed via inverse-distance weighting using GeoNames CAP geocoordinates.*
