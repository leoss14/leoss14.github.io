"""
Build threshold_sweep/comparison.html from the four artifacts_t{0,1,2,3}/ folders.

Reads:   artifacts_t{t}/{sample,reg_coef_main,ml_coefficients,clusters_1995,ml_forecast}.csv
Writes:  threshold_sweep/comparison.html

Run after sweep.py has finished:
    python threshold_sweep/build_html.py
"""
import os
import pandas as pd

HERE       = os.path.dirname(os.path.abspath(__file__))
THRESHOLDS = [0, 1, 2, 3]
OUT_PATH   = os.path.join(HERE, 'comparison.html')

def art_dir(t):
    return os.path.join(HERE, f'artifacts_t{t}')

def try_read(path):
    return pd.read_csv(path) if os.path.exists(path) else None

def fmt_coef(coef, lo, hi):
    sig = (lo > 0) or (hi < 0)
    cls = ('pos' if coef > 0 else 'neg') if sig else 'ns'
    return f'<td class="{cls}"><span class="c">{coef:+.3f}</span>'\
           f'<span class="ci">[{lo:+.3f}, {hi:+.3f}]</span></td>'

def load_all():
    """Return dict keyed by threshold, each value a dict of dataframes."""
    out = {}
    for t in THRESHOLDS:
        d = art_dir(t)
        if not os.path.isdir(d):
            print(f'  [skip] {d} not found')
            continue
        out[t] = {
            'sample':   try_read(os.path.join(d, 'sample.csv')),
            'reg':      try_read(os.path.join(d, 'reg_coef_main.csv')),
            'ml':       try_read(os.path.join(d, 'ml_coefficients.csv')),
            'clusters': try_read(os.path.join(d, 'clusters_1995.csv')),
            'forecast': try_read(os.path.join(d, 'ml_forecast.csv')),
        }
    return out

def section_sample(data):
    rows = []
    for t, d in data.items():
        n = len(d['sample']) if d['sample'] is not None else 0
        rows.append(f'<tr><td>{t}%</td><td class="num">{n}</td></tr>')
    body = '\n'.join(rows)
    return f'''<section><h2>1. Sample size</h2>
<table class="small"><thead><tr><th>Threshold</th><th>Countries</th></tr></thead>
<tbody>{body}</tbody></table></section>'''

def section_diff(data, baseline=1):
    """Members added/dropped relative to baseline threshold."""
    if baseline not in data or data[baseline]['sample'] is None:
        return ''
    base = set(data[baseline]['sample']['Country Code'])
    blocks = []
    for t, d in data.items():
        if t == baseline or d['sample'] is None:
            continue
        cur   = set(d['sample']['Country Code'])
        added = sorted(cur - base)
        drop  = sorted(base - cur)
        blocks.append(f'''<div class="diff-block">
<h4>{t}% vs {baseline}% baseline</h4>
<p><strong>Added ({len(added)}):</strong> <span class="mono">{', '.join(added) or '—'}</span></p>
<p><strong>Dropped ({len(drop)}):</strong> <span class="mono">{', '.join(drop) or '—'}</span></p>
</div>''')
    return f'''<section><h2>2. Sample composition vs {baseline}% baseline</h2>
{''.join(blocks)}</section>'''

def section_regression(data):
    """Matrix: rows = regressors, cols = thresholds, cells = coef + CI."""
    by_thr = {t: d['reg'].set_index('feature') for t, d in data.items() if d.get('reg') is not None}
    if not by_thr:
        return ''
    # ordered union of features, preserving the 1% baseline order where possible
    order = []
    if 1 in by_thr:
        order = list(by_thr[1].index)
    for t, df in by_thr.items():
        for f in df.index:
            if f not in order:
                order.append(f)
    head = '<tr><th>Variable</th>' + ''.join(f'<th>{t}%</th>' for t in by_thr) + '</tr>'
    rows = []
    for f in order:
        short = next((by_thr[t].loc[f, 'short'] for t in by_thr if f in by_thr[t].index), f)
        cells = []
        for t in by_thr:
            df = by_thr[t]
            if f in df.index:
                r = df.loc[f]
                cells.append(fmt_coef(float(r['coef']), float(r['ci_lo']), float(r['ci_hi'])))
            else:
                cells.append('<td class="absent">—</td>')
        rows.append(f'<tr><td class="label">{short}</td>{"".join(cells)}</tr>')
    return f'''<section><h2>3. Regression coefficients (panel fixed-effects, ECI as outcome)</h2>
<p class="note">Each cell shows the point estimate with the 95% confidence interval below. Green = positive and significant, red = negative and significant, grey = not significant.</p>
<div class="scroll"><table class="grid"><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table></div></section>'''

def section_ml(data):
    """ML predictor matrix. Rank features by abs_avg within each threshold,
    then show the union of top-12 features across thresholds."""
    by_thr = {}
    for t, d in data.items():
        if d.get('ml') is None:
            continue
        df = d['ml'].copy()
        df = df.sort_values('abs_avg', ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        by_thr[t] = df.set_index('feature')
    if not by_thr:
        return ''
    top_feats = []
    for t, df in by_thr.items():
        for f in df.index[:12]:
            if f not in top_feats:
                top_feats.append(f)
    # sort union by mean rank across thresholds (lower is better)
    def mean_rank(f):
        ranks = [by_thr[t].loc[f, 'rank'] for t in by_thr if f in by_thr[t].index]
        return sum(ranks) / len(ranks) if ranks else 999
    top_feats.sort(key=mean_rank)
    head = '<tr><th>Feature</th>' + ''.join(f'<th>{t}%</th>' for t in by_thr) + '</tr>'
    rows = []
    for f in top_feats:
        short = next((by_thr[t].loc[f, 'short'] for t in by_thr if f in by_thr[t].index), f)
        cells = []
        for t in by_thr:
            df = by_thr[t]
            if f in df.index:
                r = df.loc[f]
                cls = 'top' if r['rank'] <= 5 else ('mid' if r['rank'] <= 10 else 'low')
                cells.append(f'<td class="ml {cls}"><span class="rk">#{int(r["rank"])}</span>'
                             f'<span class="mag">{float(r["abs_avg"]):.4f}</span></td>')
            else:
                cells.append('<td class="absent">—</td>')
        rows.append(f'<tr><td class="label">{short}</td>{"".join(cells)}</tr>')
    return f'''<section><h2>4. Machine-learning predictor ranks</h2>
<p class="note">Rank by mean absolute coefficient (LASSO / Ridge / Elastic Net / Random Forest, all standardised). Top 5 highlighted, ranks 6-10 lighter, the rest unshaded.</p>
<div class="scroll"><table class="grid"><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table></div></section>'''

def section_clusters(data):
    """Per-threshold cluster blocks."""
    blocks = []
    for t, d in data.items():
        if d.get('clusters') is None:
            continue
        df = d['clusters']
        grouped = df.groupby('ClusterLabels')['Country'].apply(list).to_dict()
        # order clusters by size desc
        order = sorted(grouped, key=lambda k: -len(grouped[k]))
        cards = []
        for cl in order:
            members = sorted(grouped[cl])
            cards.append(f'''<div class="cl-card">
<h4>{cl} <span class="cnt">({len(members)})</span></h4>
<p class="mono small">{', '.join(members)}</p>
</div>''')
        blocks.append(f'''<div class="thr-block">
<h3>Threshold {t}%</h3>
<div class="cl-grid">{''.join(cards)}</div>
</div>''')
    return f'''<section><h2>5. Cluster membership (k-means on panel means, k=5)</h2>
{''.join(blocks)}</section>'''

def section_forecast(data, n=5):
    """Top n risers and decliners per threshold (mean Ensemble - Last_Known across horizon)."""
    blocks = []
    for t, d in data.items():
        if d.get('forecast') is None:
            continue
        df = d['forecast'].copy()
        if 'Ensemble' not in df.columns or 'Last_Known_ECI' not in df.columns:
            continue
        df['delta'] = df['Ensemble'] - df['Last_Known_ECI']
        agg = df.groupby(['Country Code', 'Country Name'])['delta'].mean().reset_index()
        agg = agg.sort_values('delta', ascending=False)
        risers   = agg.head(n)
        decliner = agg.tail(n).iloc[::-1]
        def mk(rows, cls):
            return ''.join(
                f'<tr class="{cls}"><td>{r["Country Name"]}</td>'
                f'<td class="num">{r["delta"]:+.3f}</td></tr>'
                for _, r in rows.iterrows()
            )
        blocks.append(f'''<div class="thr-block">
<h3>Threshold {t}%</h3>
<div class="fc-pair">
<table class="small"><thead><tr><th>Top {n} risers</th><th>Δ ECI</th></tr></thead>
<tbody>{mk(risers, 'pos')}</tbody></table>
<table class="small"><thead><tr><th>Top {n} decliners</th><th>Δ ECI</th></tr></thead>
<tbody>{mk(decliner, 'neg')}</tbody></table>
</div></div>''')
    return f'''<section><h2>6. Forecast top movers (mean projected change in ECI, horizon-averaged)</h2>
{''.join(blocks)}</section>'''

CSS = """
* { box-sizing: border-box; }
body { font-family: 'Public Sans', system-ui, sans-serif; font-size: 17px; line-height: 1.55;
       color: #222; background: #fafafa; margin: 0; padding: 40px 56px; max-width: 1400px; }
h1 { font-size: 30px; margin: 0 0 4px; }
h2 { font-size: 22px; margin: 36px 0 12px; border-bottom: 1px solid #d4d4d4; padding-bottom: 6px; }
h3 { font-size: 18px; margin: 18px 0 8px; color: #333; }
h4 { font-size: 15px; margin: 10px 0 4px; }
.subtitle { color: #666; margin: 0 0 8px; }
.note { color: #555; font-size: 14px; margin: 4px 0 12px; }
.mono { font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: 13.5px; }
.small { font-size: 14px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; margin: 8px 0; }
table.grid { width: 100%; }
table.grid th, table.grid td { border: 1px solid #d8d8d8; padding: 7px 10px; text-align: center; font-size: 14px; }
table.grid thead th { position: sticky; top: 0; background: #f0f0f0; z-index: 2; font-weight: 600; }
table.grid td.label { text-align: left; font-weight: 500; background: #f7f7f7; }
table.small th, table.small td { border-bottom: 1px solid #e2e2e2; padding: 5px 12px; text-align: left; font-size: 14px; }
table.small thead th { background: #f0f0f0; }
td.num { font-family: 'IBM Plex Mono', monospace; text-align: right; }
td.pos { background: #e7f3eb; }  td.pos .c { color: #1f6b3a; font-weight: 600; }
td.neg { background: #fbe9e9; }  td.neg .c { color: #a32424; font-weight: 600; }
td.ns  { background: #f4f4f4; }  td.ns  .c { color: #666; }
td .c, td .ci { display: block; font-family: 'IBM Plex Mono', monospace; }
td .c  { font-size: 13.5px; }
td .ci { font-size: 11.5px; color: #777; margin-top: 2px; }
td.absent { background: #fcfcfc; color: #bbb; }
td.ml .rk  { display: block; font-weight: 600; font-size: 13px; }
td.ml .mag { display: block; font-family: 'IBM Plex Mono', monospace; font-size: 11.5px; color: #555; }
td.ml.top { background: #e3edf7; }
td.ml.mid { background: #f1f5fa; }
td.ml.low { background: #fcfcfc; color: #777; }
.diff-block { margin: 10px 0; padding: 10px 14px; background: white; border-left: 3px solid #4a73a8; }
.cl-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 10px; }
.cl-card { background: white; border: 1px solid #e2e2e2; padding: 10px 14px; border-radius: 3px; }
.cl-card .cnt { color: #888; font-weight: 400; font-size: 13px; }
.thr-block { margin: 14px 0 22px; }
.fc-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
tr.pos td.num { color: #1f6b3a; } tr.neg td.num { color: #a32424; }
"""

def build_html(data):
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<title>Threshold sweep comparison</title>',
        '<link href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600&family=IBM+Plex+Mono&display=swap" rel="stylesheet">',
        f'<style>{CSS}</style></head><body>',
        '<h1>Threshold sweep: NR_THRESHOLD ∈ {0, 1, 2, 3}</h1>',
        '<p class="subtitle">Re-runs of the full pipeline with the natural-resource share cut-off varied. Headline pipeline uses 1%.</p>',
        section_sample(data),
        section_diff(data, baseline=1),
        section_regression(data),
        section_ml(data),
        section_clusters(data),
        section_forecast(data),
        '</body></html>',
    ]
    return '\n'.join(parts)

def main():
    data = load_all()
    if not data:
        print('No artifact folders found. Run sweep.py first.')
        return
    html = build_html(data)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Wrote {OUT_PATH}')
    print(f'Thresholds present: {sorted(data.keys())}')

if __name__ == '__main__':
    main()
