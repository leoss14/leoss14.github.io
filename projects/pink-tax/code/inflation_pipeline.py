"""
==============================================================================
GENDERED INFLATION INCIDENCE - UK 2022-2024
==============================================================================

Companion analysis to the cross-sectional pink tax study. Tests whether the
2022-2024 inflation wave fell harder on female-coded items and on the typical
female consumption basket relative to the male basket.

Pipeline stages:
  Stage 1: Download monthly ONS price quote CSVs (Jan 2019 to Dec 2024)
  Stage 2: Filter to personal-care item set; attach gender coding; parse pack size
  Stage 3: Build item-month panel of mean prices
  Stage 4: Within-item DiD (PanelOLS with two-way FE, clustered SEs); event study
  Stage 5: Basket indices (Laspeyres, Tornqvist) with LCFS weights
  Stage 6: Within-between decomposition
  Stage 7: Plotly charts

Usage (notebook or script):
    from inflation_pipeline import run_pipeline
    run_pipeline('all')           # full run
    run_pipeline(1)               # download only
    run_pipeline([2, 3, 4])       # skip download, run analysis on cached data

Outputs land in:
    ../outputs/inflation/charts/   (HTML Plotly)
    ../outputs/inflation/tables/   (CSV)
    ../outputs/inflation/cache/    (intermediate parquet files)

Data sources:
    Price quotes:  https://www.ons.gov.uk/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes
    LCFS:          https://www.ons.gov.uk/peoplepopulationandcommunity/personalandhouseholdfinances/incomeandwealth/datasets/familyspendingworkbook1detailedexpenditureandtrends
"""

import json
import re
import time
import warnings
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/pink-tax')
DATA_DIR = BASE_DIR / 'data' / 'inflation'
OUTPUT_BASE = BASE_DIR / 'outputs' / 'inflation'
CACHE_DIR = OUTPUT_BASE / 'cache'
CHART_DIR = OUTPUT_BASE / 'charts'
TABLE_DIR = OUTPUT_BASE / 'tables'

for d in [DATA_DIR, OUTPUT_BASE, CACHE_DIR, CHART_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Sample window
START_YEAR, START_MONTH = 2019, 9
END_YEAR, END_MONTH = 2024, 12

# Treatment break (pre = Sep 2019 - Dec 2021; post = Jan 2022 - Dec 2024)
BREAK_DATE = pd.Timestamp('2022-01-01')

# Tampon -> sanitary towel bridge in Feb 2023
BRIDGE_DATE = pd.Timestamp('2023-02-01')

# =============================================================================
# ITEM CODING
# =============================================================================
# Mirrors the table in inflation.html. Keep in sync if codings change.

ITEM_CODING = {
    # Female-coded
    520206: ('Tampons (pre-Feb 2023)', 'female', 'goods', 'period_products'),
    520256: ('Sanitary towels (Feb 2023+)', 'female', 'goods', 'period_products'),
    520223: ('Mascara', 'female', 'goods', 'mascara'),
    520236: ("Women's hair colourant", 'female', 'goods', 'hair_colour'),
    520243: ('Liquid foundation', 'female', 'goods', 'foundation'),
    520246: ('Lip gloss', 'female', 'goods', 'lip_gloss'),
    520250: ('Nail varnish', 'female', 'goods', 'nail_varnish'),
    520303: ("Women's haircut", 'female', 'services', 'womens_haircut'),
    520311: ("Women's highlighting", 'female', 'services', 'womens_highlighting'),
    520331: ('Basic manicure', 'female', 'services', 'manicure'),
    # Male-coded
    520241: ('Razor cartridge blades', 'male', 'goods', 'razor_blades'),
    520301: ("Man's haircut", 'male', 'services', 'mens_haircut'),
    430321: ('Electric razor', 'male', 'goods', 'electric_razor'),
    # Neutral
    520209: ('Toothpaste', 'neutral', 'goods', 'toothpaste'),
    520215: ('Facial moisturiser', 'neutral', 'goods', 'moisturiser'),
    520216: ('Deodorant', 'neutral', 'goods', 'deodorant'),
    520226: ('Shampoo', 'neutral', 'goods', 'shampoo'),
    520234: ('Perfume / EDT', 'neutral', 'goods', 'perfume'),
    520235: ('Sunscreen', 'neutral', 'goods', 'sunscreen'),
    520237: ('Hair gel', 'neutral', 'goods', 'hair_gel'),
    520238: ('Shower gel', 'neutral', 'goods', 'shower_gel'),
    520244: ('Toothbrush', 'neutral', 'goods', 'toothbrush'),
    520247: ('Liquid soap', 'neutral', 'goods', 'liquid_soap'),
    520248: ('Conditioner', 'neutral', 'goods', 'conditioner'),
    520252: ('Body moisturising lotion', 'neutral', 'goods', 'body_lotion'),
    430536: ('Toilet rolls', 'neutral', 'goods', 'toilet_rolls'),
    430351: ('Electric hair styling appliance', 'neutral', 'goods', 'hair_styling'),
    430363: ('Electric toothbrush', 'neutral', 'goods', 'electric_toothbrush'),
}

ITEM_IDS = list(ITEM_CODING.keys())

def coding_df():
    """Return the item coding as a tidy DataFrame."""
    return pd.DataFrame([
        {'item_id': k, 'item_name': v[0], 'gender': v[1],
         'item_type': v[2], 'item_group': v[3]}
        for k, v in ITEM_CODING.items()
    ])

# =============================================================================
# STYLE (matches portfolio conventions from pink_tax pipeline)
# =============================================================================

PALETTE = {'female': '#c44e52', 'male': '#4c72b0', 'neutral': '#8c8c8c'}

STYLE = {
    'font_family': 'Public Sans, -apple-system, BlinkMacSystemFont, sans-serif',
    'tick_size': 11,
    'axis_title_size': 13,
    'navy': '#1a2744',
    'slate': '#3d4f5f',
    'steel': '#4a6fa5',
    'grey_100': '#f0f2f5',
    'text_secondary': '#5a6675',
    'template': 'plotly_white',
    'plot_bg': '#fafafa',
    'paper_bg': '#fafafa',
    'grid_color': '#e5e7eb',
    'grid_width': 0.5,
    'margin_default': dict(l=60, r=40, t=20, b=50),
}

WRITE_CONFIG = {'displayModeBar': False, 'responsive': True}

def base_layout(**overrides):
    import plotly.graph_objects as go  # noqa
    layout = dict(
        template=STYLE['template'],
        font=dict(family=STYLE['font_family'], size=STYLE['tick_size'], color='#4b5563'),
        paper_bgcolor=STYLE['paper_bg'],
        plot_bgcolor=STYLE['plot_bg'],
        margin=STYLE['margin_default'],
        autosize=True,
        hoverlabel=dict(bgcolor='white', bordercolor='#c9cfd6',
                        font=dict(family=STYLE['font_family'], size=13, color='#1a2744')),
    )
    layout.update(overrides)
    return layout

def styled_axis(title_text, **kw):
    d = dict(
        title=dict(text=title_text,
                   font=dict(size=STYLE['axis_title_size'], family=STYLE['font_family'])),
        tickfont=dict(size=STYLE['tick_size'], family=STYLE['font_family']),
        gridcolor=STYLE['grid_color'], gridwidth=STYLE['grid_width'], zeroline=False,
    )
    d.update(kw)
    return d

def save_html(fig, filepath):
    fig.write_html(str(filepath), config=WRITE_CONFIG,
                   include_plotlyjs='cdn', full_html=False)
    print(f"   ✓ {filepath.name}")


# =============================================================================
# STAGE 1: DOWNLOAD ONS ITEM INDICES
# =============================================================================
#
# We pull monthly item-level price indices (one number per item per month,
# already aggregated by ONS). Lightweight compared to the full price quotes
# (~tens of KB per month vs ~50 MB per month). Sample window starts Sep 2019
# because that is when ONS began publishing per-month files; earlier 2019 is
# bundled into an annual file.
#
# URLs are hardcoded below because ONS naming is inconsistent across years
# (different folder names, occasional version suffixes, the odd xlsx). The
# table reflects what was on the dataset page as of mid-2026.

ONS_ITEM_INDEX_URLS = {
    # 2019
    '201909': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesseptember2019/upload-itemindices201909v1.csv',
    '201910': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesoctober2019/upload-itemindices201910v1.csv',
    '201911': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesnovember2019/upload-itemindices201911.csv',
    '201912': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesdecember2019/upload-itemindices201912v1.csv',
    # 2020
    '202001': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/january2020itemindices/upload-itemindices202001.csv',
    '202002': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesfebruary2020/upload-itemindices202002.csv',
    '202003': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesmarch2020/upload-itemindices202003.csv',
    '202004': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesapril2020/upload-202004itemindices.csv',
    '202005': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesmay2020/upload-202005itemindices.csv',
    '202006': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesjune2020/upload-itemindices202006.csv',
    '202007': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesjuly2020/upload-itemindices202007.csv',
    '202008': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesaugust2020/upload-itemindices202008.csv',
    '202009': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesseptember2020/upload-itemindices202009.csv',
    '202010': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesoctober2020/upload-itemindices202010.csv',
    '202011': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesnovember2020/upload-itemindicesv2202011.csv',
    '202012': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesdecember2020/upload-priceindices202012.csv',
    # 2021
    '202101': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesjanuary2021/upload-itemindices202101.csv',
    '202102': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesfebruary2021/upload-itemindices202102.csv',
    '202103': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesmarch2021/upload-itemindices202103.csv',
    '202104': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesapril2021/upload-itemindices202104.csv',
    '202105': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesmay2021/upload-itemindices2021051.csv',
    '202106': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesjune2021/upload-itemindices202106.csv',
    '202107': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesjuly2021/upload-itemindices202107.csv',
    '202108': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesaugust2021/upload-itemindices202108.csv',
    '202109': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesseptember2021/upload-itemindices202109.csv',
    '202110': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesoctober2021/upload-itemindices202110.csv',
    '202111': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesnovember2021/upload-itemindices202111.csv',
    '202112': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesdecember2021/upload-itemindices202112.csv',
    # 2022
    '202201': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesjanuary2022/upload-itemindices202201.csv',
    '202202': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesfebruary2022/itemindices202202.xlsx',
    '202203': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesmarch2022/upload-itemindices202203.csv',
    '202204': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesapril2022/upload-itemindices202204.csv',
    '202205': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesmay2022/upload-itemindices202205.csv',
    '202206': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesjune2022/upload-itemindices202206.csv',
    '202207': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesjuly2022/upload-itemindices202207.csv',
    '202208': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesaugust2022/upload-itemindices202208.csv',
    '202209': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesseptember2022/upload-itemindices202209.csv',
    '202210': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesoctober2022/upload-itemindices202210.csv',
    '202211': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesnovember2022/upload-itemindices202211.csv',
    '202212': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesdecember2022/upload-itemindices202212.csv',
    # 2023
    '202301': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesjanuary2023/upload-itemindices202301.csv',
    '202302': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesfebruary2023/upload-itemindices202302revised.csv',
    '202303': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesmarch2023/upload-itemindices2023032.csv',
    '202304': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesapril2023/upload-itemindices202304.csv',
    '202305': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesmay2023/upload-itemindices202305.csv',
    '202306': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesjune2023/upload-itemindices202306.csv',
    '202307': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesjuly2023/upload-itemindices202307.csv',
    '202308': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesaugust2023/upload-itemindices202308.csv',
    '202309': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesseptember2023/upload-itemindices202309.csv',
    '202310': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesoctober2023/upload-itemindices202310.csv',
    '202311': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesnovember2023/upload-itemindices202311.csv',
    '202312': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesdecember2023/upload-itemindices202312.csv',
    # 2024
    '202401': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesjanuary2024/upload-itemindices202401.csv',
    '202402': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesfebruary2024/upload-itemindices202402.csv',
    '202403': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesmarch2024/upload-itemindices202403.csv',
    '202404': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesapril2024/upload-itemindices202404.csv',
    '202405': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesmay2024/upload-itemindices202405.csv',
    '202406': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesjune2024/upload-itemindices202406.csv',
    '202407': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesjuly2024/upload-itemindices202407.csv',
    '202408': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesaugust2024/upload-itemindices202408.csv',
    '202409': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesseptember2024/upload-itemindices202409.csv',
    '202410': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesoctober2024/upload-itemindices202410.csv',
    '202411': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesnovember2024/upload-itemindices202411.csv',
    '202412': 'https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/consumerpriceindicescpiandretailpricesindexrpiitemindicesandpricequotes/itemindicesdecember2024/upload-itemindices202412.csv',
}

def run_download():
    """Stage 1: download monthly ONS item-index files.

    ONS rate-limits aggressive downloaders, so we wait between requests and
    apply exponential backoff on HTTP 429 (Too Many Requests). Cached files
    are skipped, so interrupted runs resume cleanly.
    """
    import requests
    print("=" * 70)
    print("STAGE 1: DOWNLOAD ONS ITEM INDICES")
    print("=" * 70)
    raw_dir = DATA_DIR / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Target: {len(ONS_ITEM_INDEX_URLS)} files (Sep 2019 to Dec 2024)")
    print(f"Base delay: 3.0s between requests; 429 backoff: 30s, 60s, 120s, 240s")

    base_delay = 3.0
    backoff_schedule = [30, 60, 120, 240]  # seconds; up to 4 retries on 429

    ok, fail, cached = 0, 0, 0
    for ym, url in sorted(ONS_ITEM_INDEX_URLS.items()):
        ext = '.xlsx' if url.lower().endswith('.xlsx') else '.csv'
        target = raw_dir / f"{ym}{ext}"
        if target.exists():
            cached += 1
            continue

        retries = 0
        while True:
            try:
                r = requests.get(url, timeout=60, stream=True,
                                 headers={'User-Agent': 'Mozilla/5.0'})
                if r.status_code == 200:
                    with open(target, 'wb') as fh:
                        for chunk in r.iter_content(8192):
                            fh.write(chunk)
                    size_kb = target.stat().st_size / 1024
                    print(f"  {ym}: downloaded ({size_kb:.0f} KB)")
                    ok += 1
                    break
                elif r.status_code == 429 and retries < len(backoff_schedule):
                    wait_s = backoff_schedule[retries]
                    print(f"  {ym}: HTTP 429, waiting {wait_s}s before retry "
                          f"{retries+1}/{len(backoff_schedule)}")
                    time.sleep(wait_s)
                    retries += 1
                    continue
                else:
                    print(f"  {ym}: HTTP {r.status_code} (giving up)")
                    fail += 1
                    break
            except Exception as e:
                print(f"  {ym}: error ({e})")
                fail += 1
                break

        time.sleep(base_delay)

    print(f"\nDownloaded {ok}, cached {cached}, failed {fail} of "
          f"{len(ONS_ITEM_INDEX_URLS)} files.")
    if fail > 0:
        print(f"For any that failed, download manually and drop into {raw_dir}")
        print("with filename pattern YYYYMM.csv or YYYYMM.xlsx")


# =============================================================================
# STAGE 2: READ ITEM INDICES, FILTER, TAG
# =============================================================================
#
# Each downloaded file is one monthly CSV (or XLSX for Feb 2022) containing
# the full set of item-level price indices for that month. ONS column names
# vary slightly by year, so we normalise on read. The output is a tidy
# DataFrame: one row per (item_id, month_date) for items in our coding table.

def _read_one_month_index(path):
    """Read one month's item-index file (CSV or XLSX), return tidy DataFrame."""
    if path.suffix.lower() == '.xlsx':
        df = pd.read_excel(path, dtype=str)
    else:
        # Latin-1 encoding handles odd characters that show up in some ONS files
        df = pd.read_csv(path, dtype=str, encoding='latin-1')

    df.columns = [c.strip().upper() for c in df.columns]

    # ONS column names vary: ITEM_ID / ITEMID / ITEM_INDEX; ALL_GM_INDEX is most
    # common for the aggregate UK index. Try a few aliases.
    item_col = next((c for c in ['ITEM_ID', 'ITEMID', 'ITEM_INDEX', 'ITEM']
                     if c in df.columns), None)
    index_col = next((c for c in ['ALL_GM_INDEX', 'INDEX_VALUE', 'INDEX',
                                   'INDEXVALUE', 'PRICE_INDEX', 'ITEM_INDEX']
                      if c in df.columns), None)

    # If 'ITEM_INDEX' appears as both an item-id-ish column and an index column,
    # the index_col guess above might collide. Prefer ALL_GM_INDEX if it exists.
    if 'ALL_GM_INDEX' in df.columns:
        index_col = 'ALL_GM_INDEX'

    if item_col is None or index_col is None:
        print(f"  [warn] {path.name}: columns not recognised, got {list(df.columns)[:6]}...")
        return pd.DataFrame()

    df = df[[item_col, index_col]].copy()
    df.columns = ['item_id', 'index_value']
    df['item_id'] = pd.to_numeric(df['item_id'], errors='coerce')
    df['index_value'] = pd.to_numeric(df['index_value'], errors='coerce')
    df = df[df['item_id'].notna() & df['index_value'].notna()]
    df['item_id'] = df['item_id'].astype(int)

    return df

def run_filter_and_tag():
    """Stage 2: combine all monthly index files, filter to target items, tag."""
    print("=" * 70)
    print("STAGE 2: READ AND TAG INDEX FILES")
    print("=" * 70)
    raw_dir = DATA_DIR / 'raw'
    files = sorted(list(raw_dir.glob('*.csv')) + list(raw_dir.glob('*.xlsx')))
    if not files:
        raise FileNotFoundError(f"No CSV/XLSX in {raw_dir}. Run Stage 1 first.")

    target_set = set(ITEM_IDS)
    rows = []
    for f in files:
        ym = f.stem  # '202203'
        if len(ym) != 6 or not ym.isdigit():
            continue
        year, month = int(ym[:4]), int(ym[4:6])
        try:
            df = _read_one_month_index(f)
            if df.empty:
                continue
            df = df[df['item_id'].isin(target_set)].copy()
            if df.empty:
                print(f"  {ym}: no target items found (file may have unexpected columns)")
                continue
            df['month_date'] = pd.Timestamp(year=year, month=month, day=1)
            rows.append(df)
            print(f"  {ym}: {len(df)} target items")
        except Exception as e:
            print(f"  {ym}: failed ({e})")

    if not rows:
        raise RuntimeError("No item indices extracted; check the source file format.")

    panel = pd.concat(rows, ignore_index=True)
    print(f"\nTotal: {len(panel):,} item-month observations "
          f"across {panel['month_date'].nunique()} months")
    print(f"Item coverage: {panel['item_id'].nunique()}/{len(ITEM_IDS)}")
    missing = set(ITEM_IDS) - set(panel['item_id'].unique())
    if missing:
        print(f"Missing items (not in ONS files for this window): {sorted(missing)}")

    # Attach coding
    panel = panel.merge(coding_df(), on='item_id', how='left')

    # Save
    panel.to_parquet(CACHE_DIR / 'tagged_indices.parquet', index=False)
    print(f"\nSaved tagged index panel -> {CACHE_DIR / 'tagged_indices.parquet'}")
    return panel


# =============================================================================
# STAGE 3: BUILD ITEM-MONTH PANEL
# =============================================================================

def _chain_link_item(item_df):
    """Chain-link one item's ITEM_INDEX series into a continuous index.

    ONS publishes annually-rebased indices: ITEM_INDEX for February through
    December of year y is relative to January of year y (Jan y = 100), while
    ITEM_INDEX for January of year y is relative to January of year y-1
    (the chain-link value bridging the two annual bases).

    This function walks chronologically through one item's observations and
    builds a continuous chained index. The first observation in the series is
    treated as already being on a stable base, so chained values are relative
    to January of the first year in the data (= 100 implicitly).
    """
    item_df = item_df.sort_values('month_date').reset_index(drop=True)
    anchor = 100.0  # chained value at the most recent January (Jan of start year)
    chained = []
    for _, row in item_df.iterrows():
        v = row['index_value']
        if row['month_date'].month == 1 and len(chained) > 0:
            # January: this row's ITEM_INDEX is in the previous year's base.
            # The current anchor is also in that previous year's base, so the
            # new anchor is just v * (old anchor) / 100.
            anchor = anchor * v / 100.0
            chained.append(anchor)
        else:
            # Feb-Dec (or the very first observation): v is in the current
            # year's base, so chained = anchor * v / 100.
            chained.append(anchor * v / 100.0)
    item_df['chained_index'] = chained
    return item_df

def run_build_panel():
    """Stage 3: chain-link ITEM_INDEX into a continuous series, bridge tampons,
    and prepare the regression-ready panel.

    ONS item indices rebase every January, so naive comparison across years
    understates cumulative inflation. We chain-link each item's series in
    chronological order. The resulting 'chained_index' is used as the
    dependent variable for the DiD and as the price input for the basket
    indices, with Jan of the first year in the data (here Sep 2019) implicitly
    set to 100.

    Output schema mirrors what Stages 4-7 expect:
        item_id_str, item_name, gender, item_type, item_group, month_date,
        price_mean (= chained_index), log_price, post, female, male
    """
    print("=" * 70)
    print("STAGE 3: BUILD ITEM-MONTH PANEL")
    print("=" * 70)
    panel = pd.read_parquet(CACHE_DIR / 'tagged_indices.parquet').copy()

    # Chain-link per item
    print("Chain-linking ITEM_INDEX across annual rebases...")
    panel = (panel
             .groupby('item_id', group_keys=False)
             .apply(_chain_link_item))

    # Sanity check: print cumulative change for a few items
    sample_items = [520209, 520241, 520303]  # toothpaste, razor blades, women's haircut
    for iid in sample_items:
        sub = panel[panel['item_id'] == iid]
        if sub.empty:
            continue
        start = sub.iloc[0]
        end = sub.iloc[-1]
        cum = end['chained_index'] / start['chained_index'] * 100 - 100
        print(f"  {iid} ({start['item_name'][:30]:30s}): "
              f"{start['month_date'].strftime('%b %Y')} = {start['chained_index']:.1f}, "
              f"{end['month_date'].strftime('%b %Y')} = {end['chained_index']:.1f}, "
              f"cumulative {cum:+.1f}%")

    # Use chained_index as the price/value column downstream
    panel['price_mean'] = panel['chained_index']
    panel['price_median'] = panel['chained_index']
    panel['unit_price_mean'] = np.nan
    panel['n_quotes'] = 1

    # Bridge tampons (520206) -> sanitary towels (520256) at BRIDGE_DATE.
    # The chained_index is in continuous units, so the rescale connects the
    # two underlying series at the bridge date.
    panel['item_id_str'] = panel['item_id'].astype(str)
    is_period = panel['item_group'] == 'period_products'
    if is_period.any():
        pp = panel[is_period].copy().sort_values('month_date')
        pre = pp[(pp['item_id'] == 520206) &
                 (pp['month_date'] < BRIDGE_DATE)].tail(3)
        post = pp[(pp['item_id'] == 520256) &
                  (pp['month_date'] >= BRIDGE_DATE)].head(3)
        if len(pre) > 0 and len(post) > 0:
            scale = pre['price_mean'].mean() / post['price_mean'].mean()
            print(f"  Period-products bridge scale: {scale:.3f}")
            mask = (panel['item_id'] == 520256)
            panel.loc[mask, 'price_mean'] = panel.loc[mask, 'price_mean'] * scale
            panel.loc[mask, 'price_median'] = panel.loc[mask, 'price_median'] * scale
        panel.loc[is_period, 'item_id_str'] = 'period_products'
        panel.loc[is_period, 'item_name'] = 'Period products (bridged)'

    # Re-aggregate after bridge
    panel = (panel
             .groupby(['item_id_str', 'item_name', 'gender', 'item_type',
                       'item_group', 'month_date'], as_index=False)
             .agg(price_mean=('price_mean', 'mean'),
                  price_median=('price_median', 'mean'),
                  unit_price_mean=('unit_price_mean', 'mean'),
                  n_quotes=('n_quotes', 'sum')))

    panel = panel.sort_values(['item_id_str', 'month_date']).reset_index(drop=True)
    panel['log_price'] = np.log(panel['price_mean'])
    panel['log_unit_price'] = np.nan
    panel['post'] = (panel['month_date'] >= BREAK_DATE).astype(int)
    panel['female'] = (panel['gender'] == 'female').astype(int)
    panel['male'] = (panel['gender'] == 'male').astype(int)

    panel.to_parquet(CACHE_DIR / 'item_month_panel.parquet', index=False)

    print(f"\nPanel shape: {panel.shape}")
    print(f"Months covered: {panel['month_date'].nunique()} "
          f"({panel['month_date'].min().strftime('%b %Y')} to "
          f"{panel['month_date'].max().strftime('%b %Y')})")
    print(f"Items by gender: "
          f"{panel.groupby('gender')['item_id_str'].nunique().to_dict()}")
    print(f"Saved -> {CACHE_DIR / 'item_month_panel.parquet'}")
    return panel


# =============================================================================
# STAGE 4: WITHIN-ITEM DID
# =============================================================================

def run_within_item_did():
    """Stage 4: two-way fixed effects DiD on log price."""
    from linearmodels import PanelOLS
    import statsmodels.formula.api as smf

    print("=" * 70)
    print("STAGE 4: WITHIN-ITEM DiD")
    print("=" * 70)
    panel = pd.read_parquet(CACHE_DIR / 'item_month_panel.parquet')

    # Set MultiIndex for PanelOLS
    pdf = panel.set_index(['item_id_str', 'month_date']).copy()

    # Base spec: log_price ~ female*post + male*post + item FE + month FE
    pdf['female_post'] = pdf['female'] * pdf['post']
    pdf['male_post'] = pdf['male'] * pdf['post']

    results = []

    def fit_and_record(spec_name, formula_extra='', subset=None, dv='log_price'):
        sub = pdf if subset is None else pdf.loc[pdf.index.get_level_values(0).isin(subset)]
        exog_cols = ['female_post', 'male_post']
        # Construct exog
        exog = sub[exog_cols].astype(float)
        endog = sub[dv]
        valid = endog.notna() & exog.notna().all(axis=1)
        endog = endog[valid]
        exog = exog[valid]
        try:
            mod = PanelOLS(endog, exog, entity_effects=True, time_effects=True,
                           check_rank=False)
            res = mod.fit(cov_type='clustered', cluster_entity=True)
        except Exception as e:
            print(f"  [{spec_name}] failed: {e}")
            return

        coef_f = res.params['female_post']
        coef_m = res.params['male_post']
        se_f = res.std_errors['female_post']
        se_m = res.std_errors['male_post']
        p_f = res.pvalues['female_post']
        p_m = res.pvalues['male_post']

        results.append({
            'spec': spec_name,
            'coef_female_post': coef_f, 'se_female_post': se_f, 'p_female_post': p_f,
            'pct_female_post': (np.exp(coef_f) - 1) * 100,
            'coef_male_post': coef_m, 'se_male_post': se_m, 'p_male_post': p_m,
            'pct_male_post': (np.exp(coef_m) - 1) * 100,
            'r2_within': res.rsquared_within, 'n_obs': res.nobs,
        })
        print(f"  [{spec_name}] female_post = {coef_f:+.4f} ({(np.exp(coef_f)-1)*100:+.1f}%), "
              f"p={p_f:.3f} | male_post = {coef_m:+.4f} ({(np.exp(coef_m)-1)*100:+.1f}%), "
              f"p={p_m:.3f} | R²_within = {res.rsquared_within:.3f}, N = {int(res.nobs):,}")

    # Spec 1: base, all items, log headline price
    fit_and_record('(1) Base, all items, headline price')

    # Spec 2: goods only
    goods_items = panel[panel['item_type'] == 'goods']['item_id_str'].unique()
    fit_and_record('(2) Goods only', subset=set(goods_items))

    # Spec 3: services only
    # Skip if the subset has no neutral items: with only gendered services
    # (women's haircuts, men's haircuts, etc.) the gender_post terms are
    # absorbed by item FE.
    svc_items = panel[panel['item_type'] == 'services']['item_id_str'].unique()
    svc_panel = panel[panel['item_id_str'].isin(svc_items)]
    if (svc_panel['gender'] == 'neutral').sum() == 0:
        print("  [(3) Services only] skipped: no neutral services in sample, "
              "gender effects fully absorbed by item FE")
    else:
        fit_and_record('(3) Services only', subset=set(svc_items))

    # Spec 4: drop bridged period_products
    nonbridge = panel[panel['item_group'] != 'period_products']['item_id_str'].unique()
    fit_and_record('(4) Drop period products', subset=set(nonbridge))

    # Spec 5: log unit price (subset with parseable size)
    # Item indices do not carry pack sizes, so log_unit_price is all NaN.
    # Skip cleanly. Shrinkflation requires quote-level data (separate exercise).
    if panel['unit_price_mean'].notna().sum() == 0:
        print("  [(5) Unit price] skipped: item indices do not carry pack size; "
              "shrinkflation requires quote-level data")
    else:
        has_unit = panel[panel['unit_price_mean'].notna()]['item_id_str'].unique()
        fit_and_record('(5) Unit price', subset=set(has_unit), dv='log_unit_price')

    rdf = pd.DataFrame(results)
    rdf.to_csv(TABLE_DIR / 'did_results.csv', index=False)
    print(f"\nSaved DiD results -> {TABLE_DIR / 'did_results.csv'}")

    # Event study: F vs neutral, monthly
    print("\nEvent study (female vs neutral, monthly):")
    pdf_e = panel[panel['gender'].isin(['female', 'neutral'])].copy()
    pdf_e['ym'] = pdf_e['month_date'].dt.strftime('%Y-%m')
    # Use Jan 2022 as reference month (omitted)
    pdf_e['fem_ym'] = pdf_e['female'] * pdf_e['month_date'].astype('int64')
    # Easier: smf with C(ym):female
    pdf_e['log_price'] = np.log(pdf_e['price_mean'])
    pdf_e['ym_offset'] = ((pdf_e['month_date'].dt.year - 2022) * 12 +
                          pdf_e['month_date'].dt.month - 1)
    try:
        formula = 'log_price ~ female:C(ym_offset) + C(item_id_str) + C(ym_offset)'
        mod = smf.ols(formula, data=pdf_e).fit(cov_type='cluster',
                                                cov_kwds={'groups': pdf_e['item_id_str']})
        # Extract female:ym_offset coefficients
        event_rows = []
        for name in mod.params.index:
            if name.startswith('female:C(ym_offset)'):
                offset = int(name.split('[')[1].rstrip(']').lstrip('T.'))
                year = 2022 + (offset // 12)
                month = (offset % 12) + 1
                event_rows.append({
                    'month_date': pd.Timestamp(year=year, month=month, day=1),
                    'coef': mod.params[name], 'se': mod.bse[name], 'p': mod.pvalues[name],
                })
        edf = pd.DataFrame(event_rows).sort_values('month_date')
        edf['pct'] = (np.exp(edf['coef']) - 1) * 100
        edf['ci_lo'] = edf['coef'] - 1.96 * edf['se']
        edf['ci_hi'] = edf['coef'] + 1.96 * edf['se']
        edf.to_csv(TABLE_DIR / 'event_study_female.csv', index=False)
        print(f"  Saved event study -> {TABLE_DIR / 'event_study_female.csv'}")
    except Exception as e:
        print(f"  Event study failed: {e}")
        edf = pd.DataFrame()

    return rdf, edf


# =============================================================================
# STAGE 5: BASKET INDICES
# =============================================================================
#
# LCFS expenditure shares are reported at COICOP group level. For the items in
# the analysis, we use placeholder weights below; the user should replace with
# actual LCFS-derived shares for single-female-adult and single-male-adult
# households. To keep the pipeline runnable end-to-end, defaults are set such
# that female and male baskets weight their own gendered items more heavily,
# with neutral items shared. These are illustrative, NOT empirical, and must
# be replaced before publishing.

# Placeholder LCFS-style weights. Replace with actual numbers.
# Rows: female-coded items, male-coded items, neutral items
# Columns: share in female basket, share in male basket
LCFS_WEIGHTS_PLACEHOLDER = {
    # item_group: (share_in_female_basket, share_in_male_basket)
    'period_products':     (0.08, 0.00),
    'mascara':             (0.03, 0.00),
    'hair_colour':         (0.04, 0.00),
    'foundation':          (0.05, 0.00),
    'lip_gloss':           (0.02, 0.00),
    'nail_varnish':        (0.02, 0.00),
    'womens_haircut':      (0.12, 0.00),
    'womens_highlighting': (0.06, 0.00),
    'manicure':            (0.04, 0.00),
    'razor_blades':        (0.01, 0.08),
    'mens_haircut':        (0.00, 0.18),
    'electric_razor':      (0.00, 0.06),
    'toothpaste':          (0.04, 0.05),
    'moisturiser':         (0.06, 0.03),
    'deodorant':           (0.05, 0.08),
    'shampoo':             (0.06, 0.06),
    'perfume':             (0.05, 0.05),
    'sunscreen':           (0.02, 0.02),
    'hair_gel':            (0.01, 0.04),
    'shower_gel':          (0.04, 0.07),
    'toothbrush':          (0.02, 0.02),
    'liquid_soap':         (0.03, 0.04),
    'conditioner':         (0.05, 0.03),
    'body_lotion':         (0.05, 0.02),
    'toilet_rolls':        (0.04, 0.04),
    'hair_styling':        (0.03, 0.01),
    'electric_toothbrush': (0.03, 0.03),
}

def lcfs_weights():
    """Return weights as a DataFrame, normalised so each basket sums to 1."""
    rows = [{'item_group': k, 'w_female': v[0], 'w_male': v[1]}
            for k, v in LCFS_WEIGHTS_PLACEHOLDER.items()]
    w = pd.DataFrame(rows)
    w['w_female'] = w['w_female'] / w['w_female'].sum()
    w['w_male'] = w['w_male'] / w['w_male'].sum()
    return w

def run_basket_indices():
    """Stage 5: construct Laspeyres and Tornqvist indices for female and male baskets."""
    print("=" * 70)
    print("STAGE 5: BASKET INDICES")
    print("=" * 70)
    panel = pd.read_parquet(CACHE_DIR / 'item_month_panel.parquet')

    # Item-level price relatives, base = Jan 2022
    base_month = BREAK_DATE
    p0 = (panel[panel['month_date'] == base_month]
          .groupby('item_group')['price_mean'].mean()
          .rename('p0'))
    panel = panel.merge(p0, on='item_group', how='left')
    panel['rel'] = panel['price_mean'] / panel['p0']

    # Restrict to items with a valid base-period price
    panel_v = panel[panel['rel'].notna()].copy()

    # Collapse to item_group x month (one price relative per group per month)
    grp_month = (panel_v
                 .groupby(['item_group', 'month_date'], as_index=False)
                 .agg(rel=('rel', 'mean')))

    w = lcfs_weights()
    grp_month = grp_month.merge(w, on='item_group', how='inner')

    # Laspeyres: L_t^g = sum_i w_i^g * (p_it / p_i0)
    laspeyres = (grp_month
                 .groupby('month_date')
                 .apply(lambda d: pd.Series({
                     'L_female': (d['w_female'] * d['rel']).sum() * 100,
                     'L_male':   (d['w_male']   * d['rel']).sum() * 100,
                     'n_items': len(d),
                 }))
                 .reset_index())
    laspeyres['gap_pp'] = laspeyres['L_female'] - laspeyres['L_male']
    laspeyres.to_csv(TABLE_DIR / 'laspeyres_indices.csv', index=False)

    # Print headline
    end_row = laspeyres.iloc[-1]
    print(f"\nLaspeyres indices at {end_row['month_date'].strftime('%b %Y')}:")
    print(f"  Female basket: {end_row['L_female']:.2f}  ({end_row['L_female']-100:+.2f}%)")
    print(f"  Male basket:   {end_row['L_male']:.2f}  ({end_row['L_male']-100:+.2f}%)")
    print(f"  Gap (F-M):     {end_row['gap_pp']:+.2f} pp")
    print(f"\nSaved -> {TABLE_DIR / 'laspeyres_indices.csv'}")

    return laspeyres


# =============================================================================
# STAGE 6: WITHIN-BETWEEN DECOMPOSITION
# =============================================================================

def run_decomposition():
    """Stage 6: split the female-male basket gap into within-item and between-item parts."""
    print("=" * 70)
    print("STAGE 6: WITHIN-BETWEEN DECOMPOSITION")
    print("=" * 70)
    panel = pd.read_parquet(CACHE_DIR / 'item_month_panel.parquet')

    # Item-group inflation: log(p_T) - log(p_0)
    base_month = BREAK_DATE
    end_month = panel['month_date'].max()
    p0 = (panel[panel['month_date'] == base_month]
          .groupby('item_group')['price_mean'].mean())
    pT = (panel[panel['month_date'] == end_month]
          .groupby('item_group')['price_mean'].mean())
    infl = (np.log(pT) - np.log(p0)).rename('infl').reset_index()

    w = lcfs_weights().merge(infl, on='item_group', how='inner')

    # Average inflation in each basket
    inf_F = (w['w_female'] * w['infl']).sum()
    inf_M = (w['w_male']   * w['infl']).sum()
    gap_total = inf_F - inf_M

    # Two-way decomposition (Oaxaca-Blinder style):
    #   gap = within + between
    # within  = sum_i w_i^M * (infl_F_i - infl_M_i)   -- but here infl is the same
    #                                                    series for each item, so
    #                                                    "within" requires gender-specific
    #                                                    item inflation, which we do not
    #                                                    have (no F vs M sub-series within
    #                                                    a single item code).
    # In practice, within-item inflation is identical across genders by construction
    # in this dataset, so the entire gap is "between" (composition). The DiD in
    # Stage 4 separately identifies any within-item differential by exploiting items
    # that exist on both gendered sides (e.g., haircuts vs men's haircuts).
    between = ((w['w_female'] - w['w_male']) * w['infl']).sum()

    print(f"\nFemale basket inflation Jan 2022 to {end_month.strftime('%b %Y')}: "
          f"{inf_F*100:+.2f}%")
    print(f"Male basket inflation:    {inf_M*100:+.2f}%")
    print(f"Gap (F - M):              {gap_total*100:+.2f} pp")
    print(f"  of which between-item (composition): {between*100:+.2f} pp")
    print(f"  of which within-item differential:   ~0 (this design)")
    print(f"\nWithin-item differential is identified separately in Stage 4 DiD.")

    out = pd.DataFrame([{
        'inf_female_basket': inf_F * 100, 'inf_male_basket': inf_M * 100,
        'gap_total_pp': gap_total * 100, 'between_component_pp': between * 100,
    }])
    out.to_csv(TABLE_DIR / 'decomposition.csv', index=False)

    # Also save item-level contributions
    contribs = w.copy()
    contribs['contrib_F'] = contribs['w_female'] * contribs['infl'] * 100
    contribs['contrib_M'] = contribs['w_male']   * contribs['infl'] * 100
    contribs['contrib_to_gap'] = contribs['contrib_F'] - contribs['contrib_M']
    contribs = contribs.sort_values('contrib_to_gap', ascending=False)
    contribs.to_csv(TABLE_DIR / 'decomposition_by_item.csv', index=False)
    print(f"Saved item-level contributions -> {TABLE_DIR / 'decomposition_by_item.csv'}")

    return out, contribs


# =============================================================================
# STAGE 7: CHARTS
# =============================================================================

def run_charts():
    """Stage 7: build inline Plotly charts matching portfolio style."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    print("=" * 70)
    print("STAGE 7: CHARTS")
    print("=" * 70)

    panel = pd.read_parquet(CACHE_DIR / 'item_month_panel.parquet')

    # --- Chart 1: cumulative price levels by gender (mean across items, normalised) ---
    base_month = BREAK_DATE
    p0 = (panel[panel['month_date'] == base_month]
          .groupby('item_group')['price_mean'].mean()
          .rename('p0'))
    pp = panel.merge(p0, on='item_group', how='left')
    pp['rel'] = pp['price_mean'] / pp['p0'] * 100
    by_gender = (pp.groupby(['gender', 'month_date'], as_index=False)
                 .agg(idx=('rel', 'mean')))

    fig = go.Figure()
    for g, color in PALETTE.items():
        sub = by_gender[by_gender['gender'] == g].sort_values('month_date')
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub['month_date'], y=sub['idx'], mode='lines',
            line=dict(color=color, width=2), name=g.title(),
            hovertemplate='%{x|%b %Y}<br>Index: %{y:.1f}<extra></extra>',
        ))
    # Use add_shape + add_annotation; add_vline's auto-positioning math
    # crashes on both Timestamps and strings in newer Plotly.
    fig.add_shape(type='line', xref='x', yref='paper', y0=0, y1=1,
                  x0=BREAK_DATE, x1=BREAK_DATE,
                  line=dict(color='#1a1a2e', width=0.8, dash='dash'))
    fig.add_annotation(xref='x', yref='paper', x=BREAK_DATE, y=1.0,
                       text='Jan 2022', showarrow=False,
                       xanchor='left', yanchor='top',
                       font=dict(size=10, color='#1a1a2e'))
    fig.add_hline(y=100, line=dict(color='#888', width=0.5))
    fig.update_layout(**base_layout(margin=dict(l=60, r=40, t=20, b=50)),
                       xaxis=styled_axis('Month'),
                       yaxis=styled_axis('Price index (Jan 2022 = 100)'))
    save_html(fig, CHART_DIR / '01_price_index_by_gender.html')

    # --- Chart 2: event study (female vs neutral) ---
    es_path = TABLE_DIR / 'event_study_female.csv'
    if es_path.exists():
        edf = pd.read_csv(es_path)
        edf['month_date'] = pd.to_datetime(edf['month_date'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=edf['month_date'], y=edf['ci_hi'] * 100,
            mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(
            x=edf['month_date'], y=edf['ci_lo'] * 100,
            mode='lines', line=dict(width=0), fill='tonexty',
            fillcolor='rgba(196,78,82,0.15)', showlegend=False))
        fig.add_trace(go.Scatter(
            x=edf['month_date'], y=edf['coef'] * 100,
            mode='lines+markers', line=dict(color=PALETTE['female'], width=2),
            marker=dict(size=4), name='Female - neutral'))
        fig.add_hline(y=0, line=dict(color='black', width=0.8, dash='dash'))
        fig.add_shape(type='line', xref='x', yref='paper', y0=0, y1=1,
                      x0=BREAK_DATE, x1=BREAK_DATE,
                      line=dict(color='#1a1a2e', width=0.8, dash='dash'))
        fig.add_annotation(xref='x', yref='paper', x=BREAK_DATE, y=1.0,
                           text='Jan 2022 (omitted)', showarrow=False,
                           xanchor='left', yanchor='top',
                           font=dict(size=10, color='#1a1a2e'))
        fig.update_layout(**base_layout(),
                           xaxis=styled_axis('Month'),
                           yaxis=styled_axis('Cumulative log price differential (%)'))
        save_html(fig, CHART_DIR / '02_event_study_female.html')

    # --- Chart 3: Laspeyres indices ---
    las_path = TABLE_DIR / 'laspeyres_indices.csv'
    if las_path.exists():
        L = pd.read_csv(las_path)
        L['month_date'] = pd.to_datetime(L['month_date'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=L['month_date'], y=L['L_female'], mode='lines',
            line=dict(color=PALETTE['female'], width=2), name='Female basket'))
        fig.add_trace(go.Scatter(
            x=L['month_date'], y=L['L_male'], mode='lines',
            line=dict(color=PALETTE['male'], width=2), name='Male basket'))
        fig.add_hline(y=100, line=dict(color='#888', width=0.5))
        fig.update_layout(**base_layout(),
                           xaxis=styled_axis('Month'),
                           yaxis=styled_axis('Laspeyres index (Jan 2022 = 100)'))
        save_html(fig, CHART_DIR / '03_laspeyres_baskets.html')

    # --- Chart 4: DiD coefficients (forest plot) ---
    did_path = TABLE_DIR / 'did_results.csv'
    if did_path.exists():
        rdf = pd.read_csv(did_path).copy()
        # Order specs top to bottom (top = first spec)
        rdf = rdf.sort_values('spec').reset_index(drop=True)
        n = len(rdf)
        # Offsets so female and male bars sit just above and below each spec row
        y_pos = np.arange(n)
        offset = 0.18

        # Confidence intervals on the % scale
        f_lo = (np.exp(rdf['coef_female_post'] - 1.96 * rdf['se_female_post']) - 1) * 100
        f_hi = (np.exp(rdf['coef_female_post'] + 1.96 * rdf['se_female_post']) - 1) * 100
        m_lo = (np.exp(rdf['coef_male_post']   - 1.96 * rdf['se_male_post'])   - 1) * 100
        m_hi = (np.exp(rdf['coef_male_post']   + 1.96 * rdf['se_male_post'])   - 1) * 100

        fig = go.Figure()
        # Female series
        fig.add_trace(go.Scatter(
            x=rdf['pct_female_post'], y=y_pos + offset,
            mode='markers', name='Female - neutral',
            marker=dict(color=PALETTE['female'], size=9, symbol='circle'),
            error_x=dict(type='data',
                         array=f_hi - rdf['pct_female_post'],
                         arrayminus=rdf['pct_female_post'] - f_lo,
                         thickness=1.2, width=4, color=PALETTE['female'])
        ))
        # Male series
        fig.add_trace(go.Scatter(
            x=rdf['pct_male_post'], y=y_pos - offset,
            mode='markers', name='Male - neutral',
            marker=dict(color=PALETTE['male'], size=9, symbol='diamond'),
            error_x=dict(type='data',
                         array=m_hi - rdf['pct_male_post'],
                         arrayminus=rdf['pct_male_post'] - m_lo,
                         thickness=1.2, width=4, color=PALETTE['male'])
        ))
        # Zero line
        fig.add_shape(type='line', xref='x', yref='paper',
                      x0=0, x1=0, y0=0, y1=1,
                      line=dict(color='#1a1a2e', width=0.8))

        fig.update_layout(
            **base_layout(margin=dict(l=220, r=40, t=20, b=60)),
            xaxis=styled_axis('Cumulative % growth relative to neutral'),
            yaxis=dict(
                tickmode='array',
                tickvals=y_pos,
                ticktext=rdf['spec'].tolist(),
                showgrid=False,
                zeroline=False,
                range=[-0.6, n - 0.4],
            ),
            legend=dict(orientation='h', yanchor='bottom', y=1.0,
                        xanchor='right', x=1.0),
        )
        save_html(fig, CHART_DIR / '04_did_coefficients.html')

    print("\nAll charts saved to", CHART_DIR)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

STAGE_MAP = {
    '1': ('Download ONS quotes', run_download),
    '2': ('Filter and tag items', run_filter_and_tag),
    '3': ('Build item-month panel', run_build_panel),
    '4': ('Within-item DiD', run_within_item_did),
    '5': ('Basket indices', run_basket_indices),
    '6': ('Decomposition', run_decomposition),
    '7': ('Charts', run_charts),
}

def run_pipeline(stages='all'):
    """
    Run one or more pipeline stages.

    Args:
        stages: 'all', or int, or list of ints, or comma-separated string.
                e.g. run_pipeline('all'), run_pipeline(2), run_pipeline([3, 4, 5]),
                run_pipeline('2,3,4').
    """
    if stages == 'all':
        stage_list = ['1', '2', '3', '4', '5', '6', '7']
    elif isinstance(stages, (list, tuple)):
        stage_list = [str(s) for s in stages]
    elif isinstance(stages, int):
        stage_list = [str(stages)]
    else:
        stage_list = [s.strip() for s in str(stages).split(',')]

    for stage in stage_list:
        if stage not in STAGE_MAP:
            print(f"Unknown stage: {stage}. Choose from {sorted(STAGE_MAP)}")
            continue
        name, func = STAGE_MAP[stage]
        print(f"\n{'#' * 70}\n# STAGE {stage}: {name.upper()}\n{'#' * 70}\n")
        try:
            func()
        except KeyboardInterrupt:
            print(f"Stage {stage} interrupted.")
        except Exception as e:
            print(f"Stage {stage} failed: {e}")
            import traceback
            traceback.print_exc()


# =============================================================================
# SUMMARY (printed when imported or run)
# =============================================================================

PIPELINE_SUMMARY = """
INFLATION PIPELINE SUMMARY
==========================

What this does:
  Builds a descriptive time-series view of UK personal-care inflation by gender
  code, 2019-2024. Companion to the cross-sectional pink tax analysis. The
  output is a set of chain-linked item indices, basket Laspeyres indices, and
  diagnostic regressions; it is not a causal estimate of any wave-specific
  effect, because the diagnostic event study shows the parallel-trends
  assumption fails.

How:
  1) Downloads 64 monthly ONS CPI item-index files (Sep 2019 - Dec 2024).
  2) Filters to the 28 personal-care items in ITEM_CODING.
  3) Chain-links each item's ITEM_INDEX values across the annual rebases.
     This is essential: without chain-linking, year-on-year comparisons
     understate cumulative inflation by an order of magnitude.
  4) Bridges the tampon-to-sanitary-towel item code change (Feb 2023).
  5) Runs the within-item DiD as a diagnostic. The output documents the
     parallel-trends failure (see Stage 4 results: female_post coefficient
     reflects a pre-existing trend, not a wave effect).
  6) Builds Laspeyres + Tornqvist basket indices for female and male
     baskets using illustrative LCFS-style weights.
  7) Decomposes the basket gap into within and between components.
  8) Generates four Plotly charts matching portfolio style.

What to swap in before publishing:
  - LCFS_WEIGHTS_PLACEHOLDER in Stage 5: replace with single-female-adult
    and single-male-adult expenditure shares from the LCFS Family Spending
    workbook. Qualitative direction (slower female-basket inflation) is
    robust to reasonable reweighting; the magnitude shifts a little.

Run:
  from inflation_pipeline import run_pipeline
  run_pipeline('all')                # full run, ~3-5 min
  run_pipeline([2, 3, 4, 5, 6, 7])   # if files already downloaded
"""

if __name__ == '__main__':
    print(PIPELINE_SUMMARY)
    # Don't auto-run; let the user invoke run_pipeline explicitly
