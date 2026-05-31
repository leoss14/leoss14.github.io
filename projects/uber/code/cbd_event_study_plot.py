"""
cbd_event_study_plot.py

Renders the event-time coefficient plot from cbd_event_study_coefs.csv.
Two panels (Uber, Lyft). Each panel shows event-time coefficients β_τ
with 95% confidence intervals, the reference point at τ=-1, and shading
to demarcate pre-period from post-period.

This is the canonical event-study visual: pre-period coefficients should
hover around zero if parallel trends holds. Visible drift in the pre-period
diagnoses parallel-trends violation.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
from _panel_loader import save_chart, base_layout, PALETTE, FONT

ROOT = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/uber')
TABLES = ROOT / 'outputs' / 'tables'

coefs = pd.read_csv(TABLES / 'cbd_event_study_coefs.csv')
tests = pd.read_csv(TABLES / 'cbd_event_study_tests.csv')

print('Coefficients loaded:')
print(coefs.groupby('operator').agg(n=('event_time', 'count'),
                                    min_t=('event_time', 'min'),
                                    max_t=('event_time', 'max')))
print()
print('Tests:')
print(tests.to_string(index=False))

OP_COLOR = {'Uber': PALETTE['navy'], 'Lyft': PALETTE['rose']}

fig = make_subplots(rows=1, cols=2, subplot_titles=('Uber', 'Lyft'),
                    horizontal_spacing=0.10, shared_yaxes=True)

for i, op in enumerate(['Uber', 'Lyft']):
    sub = coefs[coefs['operator'] == op].sort_values('event_time').copy()
    col = i + 1
    color = OP_COLOR[op]

    # 95% CI as error bars
    fig.add_trace(go.Scatter(
        x=sub['event_time'],
        y=sub['coef'] * 100,
        error_y=dict(
            type='data',
            array=(sub['ci_high'] - sub['coef']) * 100,
            arrayminus=(sub['coef'] - sub['ci_low']) * 100,
            thickness=1.2, width=4, color=color,
        ),
        mode='markers',
        marker=dict(color=color, size=7,
                    line=dict(color='white', width=1)),
        name=op,
        showlegend=False,
        hovertemplate=(f'{op}<br>Event time: %{{x}} months<br>'
                       'Coefficient: %{y:.1f}%<extra></extra>'),
    ), row=1, col=col)

    # Reference point at τ=-1 (already in data as 0,0)
    # Zero line
    fig.add_hline(y=0, line=dict(color=PALETTE['slate'], width=1,
                                  dash='solid'), opacity=0.4,
                  row=1, col=col)

    # Event line at τ=0
    fig.add_vline(x=0, line=dict(color=PALETTE['text'], width=1.5,
                                  dash='dash'), row=1, col=col)

    # Pre-trends test annotation in each panel
    op_test = tests[tests['operator'] == op].iloc[0]
    f_val = op_test['pre_trends_F']
    p_val = op_test['pre_trends_p']
    p_str = '<0.001' if p_val < 0.001 else f'{p_val:.3f}'
    fig.add_annotation(
        x=-11, y=0.97, xref=f'x{col if col > 1 else ""}', yref='paper',
        text=f'<b>Pre-trends F</b> = {f_val:.1f} (p {p_str})',
        showarrow=False, font=dict(size=11, color=PALETTE['slate']),
        align='left', xanchor='left',
    )

# Layout
fig.update_layout(**base_layout(height=480, width=980))
fig.update_layout(
    showlegend=False,
    margin=dict(l=70, r=30, t=60, b=70),
)

for col in [1, 2]:
    fig.update_xaxes(
        title='Months relative to CBD fee (Jan 2025 = 0, excluded)',
        tickvals=list(range(-12, 13, 3)),
        gridcolor=PALETTE['grid'], zeroline=False,
        row=1, col=col,
    )
fig.update_yaxes(
    title='Coefficient on log(base fare), % (95% CI)',
    gridcolor=PALETTE['grid'], zeroline=False,
    row=1, col=1,
)

# Improve subplot title styling
for ann in fig.layout.annotations[:2]:
    ann.font = dict(size=14, color=PALETTE['text'])

save_chart(fig, 'cbd/cbd_event_study_proper')
print('\nWrote outputs/cbd/cbd_event_study_proper.html')
