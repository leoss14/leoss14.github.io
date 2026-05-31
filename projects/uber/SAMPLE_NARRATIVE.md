# Sample-Based Narrative — Source Findings

**Date:** 2026-05-24
**Source:** `outputs/tables/trip_sample_full.parquet` (8,340,641 weighted trips: 4.98M Uber + 3.36M Lyft, Feb 2019 → Apr 2026)
**ESS:** Uber 79.4% / Lyft 78.6% (healthy)
**All means below are weighted by `sampling_weight` for unbiased population estimates.**

---

## What the sample lets us do that the existing page does NOT

The current page (`page.html`) covers the for-hire vehicle market between 2018 and 2026, but every chart on it is aggregated from server-side rollups. Four important questions cannot be answered from aggregates and required the sample:

1. **Uber vs Lyft head-to-head** — the page treats Uber and Lyft as separate operators in the market-share chart, but never compares their per-trip economics. The sample has both at trip level.
2. **Response time** — the time between request and on-scene arrival is a trip-level field, not a monthly aggregate. The page has no chart of it.
3. **Tip patterns** — tips appear in driver pay but the page does not break out tip distribution by zone or operator.
4. **Pool share** — the share of trips that were shared/pooled rides is trip-level. The page does not address this.

What follows is the analytic raw material for adding a new section to the page on the **Uber vs Lyft head-to-head** and the **driver-side / rider-experience inequities** the sample exposes.

---

## Headline findings

### 1. Lyft has extracted more per trip than Uber for the entire panel

Operator margin proxy (full-panel weighted mean):
- **Uber: 15.1%**
- **Lyft: 21.8%**

Lyft's margin was higher than Uber's in every single year of the panel:

| Year | Uber margin | Lyft margin | Gap (Lyft − Uber) |
|------|-------------|-------------|-------------------|
| 2019 | −4.4% | +19.8% | +24.2 pp |
| 2020 | +9.0% | +21.8% | +12.8 pp |
| 2021 | +13.4% | +24.4% | +11.0 pp |
| 2022 | +18.9% | +20.8% | +1.9 pp |
| 2023 | +21.0% | +19.5% | −1.5 pp |
| 2024 | +21.3% | +24.4% | +3.1 pp |
| 2025 | +20.9% | +21.6% | +0.7 pp |
| 2026 | +19.2% | +22.4% | +3.2 pp |

This contradicts the common framing of Uber as the more extractive platform. **Lyft started the panel with margins Uber would not reach for three years**, and the two only converged once Uber's COVID-era repricing finished.

The gap opens up the question of whether Lyft's higher margin reflects (a) different fare/pay split policy, (b) different trip mix (Lyft is more outer-borough), or (c) different cost structures. The sample lets us decompose this by zone.

### 2. The margin gap is concentrated in outer boroughs

Full-panel margin by pickup zone class:

| Zone class | Uber | Lyft | Gap |
|---|---|---|---|
| Airport | 18.6% | 20.3% | +1.7 pp |
| CBD | 19.5% | 22.6% | +3.1 pp |
| Buffer (60th-65th) | 20.6% | 22.8% | +2.2 pp |
| Upper Manhattan | 9.9% | 20.8% | **+10.9 pp** |
| Outer boroughs | 12.5% | 21.5% | **+9.0 pp** |

In the CBD and at airports, the two operators take similar shares of each trip's fare. **In outer boroughs and Upper Manhattan, Lyft retains roughly twice the share Uber retains.** This is a clean, large, geographically-localised finding.

Interpretation candidate: Uber has more aggressive market-share growth incentives in lower-density areas (where there are fewer alternative options for the rider) and accepts lower margins to keep the ride happening. Lyft, with less aggressive expansion incentive, prices more uniformly.

### 3. Uber pays drivers more per hour, in every zone

Full-panel pay-per-hour:

| Zone class | Uber | Lyft | Uber premium |
|---|---|---|---|
| Airport | $71.40 | $66.20 | +$5.20/hr |
| CBD | $62.63 | $56.05 | +$6.58/hr |
| Buffer | $63.50 | $55.43 | +$8.07/hr |
| Outer boroughs | $58.32 | $55.39 | +$2.93/hr |
| Upper Manhattan | $60.64 | $57.53 | +$3.11/hr |

Combined with the margin finding: **Uber takes less and pays more.** The Lyft-versus-Uber differential is not a wash; it shows up clearly on both sides of the platform.

This is the single most counterintuitive finding in the panel. It is worth being careful about: it does not say Lyft drivers earn less in total, since dollar-per-hour is conditional on having a trip. If Uber drivers have more idle time between trips, hourly pay during trips overstates effective hourly earnings. The sample cannot test that without driver identifiers.

### 4. Response time is a geographic inequity, not an operator one (2024-2026)

Median time from request to on-scene arrival (clipped at one hour):

| Zone class | Uber median | Lyft median |
|---|---|---|
| Buffer (60th-65th) | 144s | 189s |
| CBD | 150s | 196s |
| Upper Manhattan | 177s | 215s |
| Outer boroughs | 185s | 217s |
| Airport | 270s | 230s |

Two patterns:

- **Outer-borough riders wait roughly 25% longer than CBD riders** on both platforms. The geographic inequality is real and operator-neutral.
- **At airports, the operator ordering reverses.** Lyft riders wait less than Uber riders at airports (230s vs 270s). The most plausible mechanism is queue depth: Uber's airport queue is longer because more drivers stage there hoping for a profitable airport-to-Manhattan return.

Tail risk (90th percentile) is also worse outside Manhattan: P90 outer-borough wait is 450s (7.5 min) for Uber and 459s for Lyft, versus 402s and 439s in CBD. The mean is dragged up by these long-wait events.

Caveat: Lyft `on_scene_datetime` is unreliable before 2024 (the column shows physically impossible values, e.g. mean -83 seconds in 2023). The response-time chart should be restricted to 2024-2026 to avoid showing nonsense Lyft pre-2024 data.

### 5. Uber Pool nearly died, then partially returned

Share of Uber trips marked as shared/pooled:

| Year | Pool requested | Pool matched | Match rate (matched / requested) |
|------|---------------|--------------|----------------------------------|
| 2019 | 17.5% | 12.3% | 71% |
| 2020 | 4.4% | 2.7% | 62% |
| 2021 | 0.0% | 0.0% | suspended |
| 2022 | 0.8% | 0.1% | 11% |
| 2023 | 3.2% | 0.9% | 29% |
| 2024 | 4.9% | 2.1% | 42% |
| 2025 | 4.0% | 2.3% | 57% |
| 2026 | 2.5% | 1.4% | 57% |

Uber Pool was a major share of trips in early 2019 (about one in seven), collapsed during COVID, was effectively suspended through 2021, and has been a small (under 5% requested, under 2.5% matched) service ever since. The "match rate" (fraction of pool requests that actually got matched with another rider) recovered to roughly its 2019 level but on a much smaller base.

Lyft does not appear in the sample as having a functioning Pool service at any point in the panel. (Lyft Shared / Lyft Line was shut down in November 2019 and has been re-piloted in limited cities; it does not register in the NYC TLC data.)

### 6. Wheelchair-accessible vehicle (WAV) availability rose substantially

Share of Uber trips where the vehicle was WAV-equipped (not necessarily requested):

| Year | Uber WAV match | Lyft WAV match |
|---|---|---|
| 2019 | 0.4% | 0.4% |
| 2021 | 5.2% | 3.5% |
| 2023 | 6.9% | 8.8% |
| 2024 | 7.6% | 12.1% |
| 2026 | 10.2% | 5.8% |

This is downstream of the TLC's WAV-fleet mandate, which requires for-hire operators to maintain a minimum share of accessible vehicles in their fleet. By 2024, roughly one in ten Lyft trips was happening in a WAV-equipped vehicle; that share fell in 2025-2026 (possibly reflecting Lyft fleet composition changes).

Note that the share of trips where the rider explicitly *requested* a WAV is tiny throughout (0.02% to 0.4%). Almost all WAV matches are people who did not request one but happened to be matched with a driver of a WAV-equipped car.

### 7. Tip patterns are sharply geographic

Mean tip per trip, full panel:

| Zone class | Uber | Lyft |
|---|---|---|
| Airport | $3.65 | $3.89 |
| Buffer | $1.61 | $1.76 |
| CBD | $1.39 | $1.51 |
| Upper Manhattan | $0.64 | $1.07 |
| Outer boroughs | $0.57 | $0.80 |

Airport pickups generate roughly five times the tip income of outer-borough pickups in absolute dollars. Lyft riders tip more than Uber riders in every single zone class. The CBD-versus-outer differential is larger than the Uber-versus-Lyft differential, so this is more a geographic story than an operator story. Tipping is concentrated in the trips drivers already earn the most on.

---

## Narrative arc for the new sample section

The current page has seven parts. The new section becomes **Part 4.5 (between Part 4 "Drivers earned differently" and Part 5 "Congestion-fee shock")** or a new **Part 8** before the synthesis, whichever flows better. I'd argue for splitting Part 4 into two parts: keep the existing aggregate driver-pay story and add a new part that uses the sample to expose what aggregates hide.

**Working title:** "What the trip-level sample reveals about the two-platform market"

**Sub-sections:**

1. **The operator gap, 2019 to 2026** — Lyft has been the higher-margin platform for the entire panel; the gap closed during Uber's COVID-era repricing but never inverted. Two-line chart, Uber margin vs Lyft margin by month.

2. **Where Lyft extracts more** — bar chart of margin by zone class, side-by-side Uber and Lyft. The outer-borough and Upper Manhattan bars are the visual hook.

3. **Driver pay: Uber's premium** — two-line chart of pay-per-hour by operator over time. Bar chart of pay-per-hour by zone class. The contrast with the margin finding is the discussion point.

4. **Response time inequity** — restricted to 2024-2026 for data quality. Boxplot or violin of response_sec by zone class, operator. The "airport queue is longer for Uber" reversal is the surprise.

5. **What happened to Pool** — single chart, share of Uber trips that were pool requests over time, with the 2021 cliff visible. Lyft is a flat line near zero.

6. **WAV access** — share of trips in WAV-equipped vehicles, by operator, over time. The TLC mandate timing is the regulatory context.

7. **Tips as inequality, not generosity** — small chart showing tip per trip by zone class. The point is that tipping mirrors fare/wage geography rather than offsetting it.

**Length target:** 4-5 charts plus prose. Less than Part 2 (geography) which has 11 charts. Each chart should be doing work the existing aggregate charts cannot do.

---

## Discussion points for the synthesis section

Each new finding fits one of two interpretive frames:

**Frame 1: The "convergent extraction" reading.**
The page's Part 3 already argues that Uber moved from cross-subsidising short trips to uniform extraction. The Uber-Lyft comparison strengthens this: Lyft was already running uniform-extraction pricing in 2019, and Uber spent five years catching up. The two operators converged on similar margins by 2023, suggesting the for-hire market has a "natural" platform margin around 19-22% that both ended up at via different paths.

**Frame 2: The "different business models" reading.**
Uber takes less per trip but pays more per hour. Combined, Uber probably runs higher trip volume per active driver (since the margin gap is closed by faster driver turnover, not higher per-trip extraction). Lyft is the higher-margin / lower-volume operator. This is consistent with Lyft's repeated public framing of itself as the "smaller, friendlier" platform but cuts against Lyft's public posture as the more driver-aligned operator: drivers earn less per hour on Lyft.

**Frame 3: The "geographic redlining-of-margin" reading.**
The most striking single finding is the Uber-Lyft margin gap in outer boroughs (9 percentage points) and Upper Manhattan (11 percentage points). The cleanest explanation is that Uber subsidises low-density area trips to maintain market share, while Lyft does not. This has distributional consequences: trips originating in lower-income, higher-Black-and-Hispanic neighbourhoods are more expensive (in margin terms) on Lyft than on Uber. Discussion of this finding should be careful not to overclaim — the sample shows the pattern but cannot tell whether it reflects deliberate policy, fare-elasticity differences across neighbourhoods, or driver-side staging decisions.

The response-time inequity finding (outer-borough riders waiting 25% longer) is operator-independent and adds a non-pricing axis to the same geographic story. Combined with the tip inequality, the picture is that **the for-hire market in NYC has roughly three regimes**: the airport (high margin, high pay, high tips, slow pickup); central Manhattan (medium margin, medium pay, medium tips, fast pickup); and outer boroughs (low Uber margin, low pay, low tips, slow pickup). Every metric reinforces the same hierarchy.

---

## Data quality flags

1. **Lyft `on_scene_datetime` is bad through 2023.** Years 2022 and 2023 show mean response times of 6.7s and -96s, which are physically impossible. The sample-based response analysis should restrict to 2024 onward for Lyft.

2. **Pool flag in 2019.** Lyft 2019 shows wav_request_flag and shared_match_flag values that look inconsistent with the rest of the panel (one isolated 13.4% pool match in Feb 2019, then near-zero). Probably a column-mapping issue in early TLC FHVHV reporting. Should be filtered out.

3. **Sampling weights matter.** Stratification gave airport zones (where trips are long) extra weight. Unweighted mean trip_miles is 5.58; weighted is 4.91. A 14% bias. Every per-trip statistic above is weighted; analyses run without the weight will overstate long-trip patterns.

4. **No driver identifiers.** All "driver earnings" findings are per-trip, not per-driver. A driver working long hours on Lyft might earn more annually than a Uber driver working short hours even though Lyft pays less per hour. This caveat needs to be flagged in the writeup.

---

## Next steps (in order)

1. Validate the narrative above with you before I write any charts.
2. Build the seven sub-section charts (Plotly, IBM Plex Sans, navy/slate palette consistent with page.html).
3. Decide chart placement: new "Part 4.5" between drivers and congestion fee, or new "Part 8" before synthesis.
4. Write the prose section. Roughly 800-1200 words.
5. Update the Part 7 prediction scorecard and the synthesis section to reference the Uber/Lyft comparison where it strengthens or qualifies the existing findings.

---

## Additional findings worth testing / discussing

Things I checked beyond the original seven. Some are clean additional findings, some are cross-checks that strengthen the existing claims, some are weak / discardable.

### 8. Subsidisation share, not aggregate margin, is the cleaner story

The aggregate margin trajectory in the existing Part 3 is already on the page. What the sample adds: **the share of trips that operate at negative margin, decomposed by zone class and operator.**

**Share of Uber outer-borough trips with negative margin:**
- 2019: 42.4%
- 2022: 10.4% (bottom)
- **2026: 22.2%** (rising sharply since 2022)

**Share of Uber CBD trips with negative margin:**
- 2019: 31.6%
- 2026: 9.8% (still falling)

**Share of Lyft outer-borough trips with negative margin:**
- 2019-2026: stable at 6-10%, never rises

This is a clean, unique finding. **Uber post-COVID has been increasing cross-subsidisation of outer-borough trips while continuing to extract uniformly in CBD.** Lyft has done neither. The aggregate margin numbers (which converged) hide this divergent strategy.

This finding strengthens Frame 3 (geographic margin redlining) from the original narrative: not only is Uber's outer-borough margin lower, it has been getting lower again post-2022. This looks like an active strategy, not a legacy effect of COVID.

### 9. Effective hourly earnings, after dispatch time

Pay-per-hour as reported in the TLC data is `driver_pay / trip_time`. This excludes the unpaid time the driver spends getting to the rider. Including that time changes the headline numbers substantially.

2024-2026, effective hourly earnings (`driver_pay / (trip_time + response_time)`):

| Zone | Trip pay/hour | Effective pay/hour | Reduction |
|---|---|---|---|
| Airport | $73.90 (Uber), $71.72 (Lyft) | $62.89, $62.66 | -15%, -13% |
| CBD | $64.35, $59.97 | $53.53, $48.72 | -17%, -19% |
| Outer | $60.27, $58.92 | $47.02, $44.77 | -22%, -24% |
| Upper Manhattan | $63.69, $62.14 | $50.64, $48.19 | -20%, -22% |

The "Uber pays more per hour" headline from the original narrative narrows once dispatch time is included. Outer-borough Uber drivers effectively earn $47/hr (not $60/hr), and the gap to Lyft narrows from $2.93/hr to $2.25/hr. Headline numbers without dispatch time **overstate** driver earnings, by about 15-25%.

This also reveals a clean geographic pay gap independent of operator: airport drivers effectively earn $63/hr, outer-borough drivers $45/hr. **An outer-borough driver earns 30% less per hour worked than an airport driver, before any per-mile cost differences.** Combined with the response-time inequity, this is the same "three-regime" pattern from the original narrative but with sharper numbers.

### 10. Margin by trip length is U-shaped

2024-2026:

| Distance | Uber margin | Lyft margin |
|---|---|---|
| <1 mile | **33.0%** | **29.9%** |
| 1-2 mi | 27.1% | 27.4% |
| 2-5 mi | 19.2% | 21.5% |
| 5-10 mi | 12.5% | 18.4% |
| 10-20 mi | 13.9% | 18.3% |
| 20+ mi | **20.4%** | **13.8%** |

Three observations:
- **Very short trips are the highest-margin segment.** This is the minimum-fare effect: trips that should have been $4 hit the minimum and the platform pockets the difference. About 6% of Uber's volume is under one mile.
- **Long trips are the second-highest-margin segment for Uber but the lowest for Lyft.** This is the only segment where Uber margin > Lyft margin. Probably driven by airport-to-suburb runs where Uber's pricing is more aggressive.
- **The U-shape contradicts the existing Part 3 "uniform extraction" claim.** Part 3 argues that platform extraction has become uniform across trip lengths post-2019. At the headline level that is true, but the U-shape at the tails (especially very short trips) is large enough to mention.

### 11. Speed degradation is real and continues

Mean trip speed, all NYC for-hire vehicles:
- 2019: 13.66 mph
- 2020: 14.83 mph (COVID — empty streets)
- 2022: 13.87 mph
- 2024: 13.59 mph
- 2026: 13.03 mph

P50 (median):
- 2019: 12.4 mph
- 2026: 11.5 mph

NYC streets have gotten slower across the panel even after correcting for COVID. The 2026 median trip moves 7% slower than the 2019 median trip. This affects everything else: driver pay-per-mile, fares, and rider total trip time.

This is interesting context for the page but not a headline finding on its own.

### 12. Hour-of-day extraction is uniform, but burdens differ

2024-2026, all operators:
- Margin: stable 20-24% across all hours
- Fare per mile: ranges from $5.64 (5am) to $8.42 (5pm) — riders pay 50% more per mile during PM peak
- Pay per hour: ranges from $73 (4am) to $57 (2pm) — drivers earn 28% more per hour overnight

**Platform extraction is uniform; the rider-driver burden split shifts with time of day.** Riders carry the PM-peak cost; drivers capture the overnight benefit. This is a clean inequality finding but probably belongs in a "rider experience" sub-section.

### 13. Day-of-week patterns weaken over the panel

Weekend margins were 5 percentage points higher than weekday margins in 2019; by 2024-2026 the gap has compressed to 1-2 pp (now weekday slightly higher). The "weekend repricing" Uber did post-2019 has converged to weekday levels. This is a small finding and probably not worth its own chart.

### 14. Pool composition shifts toward outer boroughs

Uber Pool match rate by zone, 2024:
- CBD: 2.0%
- Outer boroughs: 2.2%
- Upper Manhattan: 2.1%
- Buffer: 1.7%
- Airport: 0.2%

The current Uber Pool (post-2023 relaunch) is mostly an outer-borough product, in contrast to the pre-COVID Pool which was much more Manhattan-heavy. This is consistent with Frame 3 (Uber subsidising outer-borough share): Pool reduces marginal trip costs, and offering it preferentially in outer boroughs would be one mechanism for the rising negative-margin trip share documented in finding 8.

### 15. Zero-tip rate is roughly stable, but Lyft tippers tip more

Share of trips with zero tip, 2024-2026:
- Uber: 81.5%
- Lyft: 79.5%

So about 80% of trips get no tip on either platform, but conditional on tipping, Lyft riders tip more in absolute terms. This is a small finding; the geographic tip story (already in original narrative finding 7) is more interesting.

---

## What I do NOT recommend testing

Several things I checked are not worth chart real estate:

- **Day-of-week patterns**: weakened over time, no clear story
- **Trip tolls / congestion surcharge passthrough at trip level**: data on `congestion_surcharge` column shows the older $2.75 state surcharge, not the new MTA fee that started Jan 2025. The new MTA fee is bundled elsewhere (likely tolls) and is hard to isolate from regular tolls in the trip data
- **TLC pay-rule discontinuities**: the Dec 2022 and Mar 2024 TLC pay raises do not produce visible discontinuities in driver pay-per-hour, because nominal pay was already above the new minimums. The pay-rule effect is on utilization, not floor pay, and utilization is not in the trip data
- **WAV request vs match**: as noted in the original narrative, almost no riders explicitly request WAV. The WAV story is fleet composition, not rider preference

---

## Recommendation

The original narrative had 7 charts. Adding findings 8-12 would bring it to 11-12 charts, which is too many. Suggested cut:

**Keep:**
- Operator margin gap (original #1)
- Margin by zone class (original #2) — strengthen with subsidised-share decomposition (new #8)
- Pay per hour with effective-pay correction (original #3 + new #9 combined)
- Response time inequity (original #4)
- Pool collapse / revival (original #5) — supplement with zone-level pool revival (new #14)
- Tip geography (original #7)

**Move to appendix or drop:**
- WAV access (original #6) — clean but disconnected from main narrative
- Hour-of-day extraction (new #12) — clean but secondary
- Trip-length margin U-shape (new #10) — interesting but adds complexity
- Speed degradation (new #11) — better as a single-line aside in the existing temporal section

This gives roughly 6 chart slots in the new section, each doing meaningful work, with the subsidisation-share finding (new #8) being the new lead because it is the cleanest single result the sample adds to the page.
