#!/usr/bin/env python3
"""
patch_generate_charts_5.py

Applies surgical edits to generate_charts_(5).py to improve chart quality:

  fig07 / fig08 — truncate to top 12 features by absolute magnitude
  fig09         — drop the noisy faint background lines (keep only highlighted countries)
  fig12         — drop Model 3a (no lag), keep only 3b headline; drop in-figure title
                  so the page can supply its own h4

Run once:  python3 patch_generate_charts_5.py
Then run V5 normally as before.

This script edits the file in place after taking a backup.
"""

import os, sys, shutil, re

HERE = os.path.dirname(os.path.abspath(__file__))

# Find generate_charts_(5).py — try common names
CANDIDATES = [
    "generate_charts_(5).py",
    "generate_charts(5).py",
    "generate_charts_5.py",
    "generate_charts_5_.py",
]
target = None
search_dirs = [HERE, os.path.join(HERE, "v2"), os.path.join(HERE, "..", "v2")]
for d in search_dirs:
    for n in CANDIDATES:
        p = os.path.join(d, n)
        if os.path.exists(p):
            target = os.path.abspath(p)
            break
    if target:
        break

if target is None:
    print("Could not find generate_charts_(5).py. Looked in:", search_dirs)
    sys.exit(1)

print(f"Patching: {target}")
backup = target + ".bak"
if not os.path.exists(backup):
    shutil.copy(target, backup)
    print(f"Backup written: {backup}")

src = open(target).read()
orig_len = len(src)


# ── EDIT 1: fig07 — truncate to top 12 by |abs_avg| ──────────────────────────
# After coef_df = coef_df.sort_values("abs_avg", ascending=True)..., add:
#     coef_df = coef_df.tail(12).reset_index(drop=True)
old_07 = 'coef_df = coef_df.sort_values("abs_avg", ascending=True).reset_index(drop=True)'
new_07 = (
    'coef_df = coef_df.sort_values("abs_avg", ascending=True).reset_index(drop=True)\n'
    '# Truncate to top 12 by absolute magnitude (page request: only show most important)\n'
    'coef_df = coef_df.tail(12).reset_index(drop=True)'
)
if old_07 in src:
    src = src.replace(old_07, new_07, 1)
    print("  ✓ fig07: truncated to top 12 features")
else:
    print("  ✗ fig07: pattern not found (already patched?)")


# ── EDIT 1b: fig07 — drop in-figure title ─────────────────────────────────────
# We change title() to None so the chart has no top text (page provides h4).
old_07t = (
    'title=title("LASSO, Ridge and Elastic Net Coefficients",\n'
    '                "Standardised inputs · Panel 1996–2014 training"),'
)
new_07t = "title=None,"
if old_07t in src:
    src = src.replace(old_07t, new_07t, 1)
    print("  ✓ fig07: in-figure title removed")
else:
    print("  ✗ fig07: title pattern not found")


# ── EDIT 2: fig08 — truncate to top 12 ───────────────────────────────────────
old_08 = 'rf_show = imp_show.sort_values("RF", ascending=True).copy()'
new_08 = (
    'rf_show = imp_show.sort_values("RF", ascending=True).tail(12).copy()  '
    '# top 12 only (page request)'
)
if old_08 in src:
    src = src.replace(old_08, new_08, 1)
    print("  ✓ fig08: truncated to top 12 features")
else:
    print("  ✗ fig08: pattern not found")


# ── EDIT 2b: fig08 — drop in-figure title ─────────────────────────────────────
old_08t = (
    'title=title("Random Forest Feature Importance",\n'
    '                "Mean Decrease in Impurity (MDI) · 200 trees · normalised"),'
)
new_08t = "title=None,"
if old_08t in src:
    src = src.replace(old_08t, new_08t, 1)
    print("  ✓ fig08: in-figure title removed")
else:
    print("  ✗ fig08: title pattern not found")


# ── EDIT 3: fig09 — drop the faint background lines ───────────────────────────
# The block adds grey lines for every non-highlighted country. Remove it by
# wrapping with `if False:`.
old_09 = (
    '    for cc in all_cc:\n'
    '        if cc in highlight: continue\n'
    '        h = hist[hist["Country Code"] == cc].sort_values("Year")\n'
    '        if len(h) == 0: continue\n'
    '        figZ.add_trace(go.Scatter(\n'
    '            x=h["Year"], y=h["Economic Complexity Index"], mode="lines",\n'
    '            line=dict(color=GREY_L, width=0.6), opacity=0.2,\n'
    '            showlegend=False, hoverinfo="skip",\n'
    '        ), row=1, col=panel_i)'
)
new_09 = (
    "    # Background lines for non-highlighted countries removed (page request)\n"
    "    # to reduce visual noise; only the 6 highlighted countries are drawn."
)
if old_09 in src:
    src = src.replace(old_09, new_09, 1)
    print("  ✓ fig09: background lines removed")
else:
    print("  ✗ fig09: pattern not found")


# ── EDIT 3b: fig09 — drop in-figure title ─────────────────────────────────────
old_09t = (
    'title=title("Projected ECI Trajectories, 2020–2030",\n'
    '                "Solid = historical · Dashed = ensemble forecast · Band = model disagreement"),'
)
new_09t = "title=None,"
if old_09t in src:
    src = src.replace(old_09t, new_09t, 1)
    print("  ✓ fig09: in-figure title removed")
else:
    print("  ✗ fig09: title pattern not found")


# ── EDIT 4: fig12 — drop Model 3a (no lag), keep only 3b ──────────────────────
# Change the loop to iterate over only [(m3b, "3b (with lag)", "#c23a3a")].
old_12 = (
    'for mi, (model, mname, col) in enumerate([(m3a, "3a (no lag)", "#4a6fa5"),\n'
    '                                            (m3b, "3b (with lag)", "#c23a3a")]):'
)
new_12 = (
    'for mi, (model, mname, col) in enumerate([(m3b, "Headline (with lagged ECI)", "#c23a3a")]):'
)
if old_12 in src:
    src = src.replace(old_12, new_12, 1)
    print("  ✓ fig12: Model 3a removed, only 3b shown")
else:
    print("  ✗ fig12: pattern not found")


# ── EDIT 4b: fig12 — drop in-figure title ─────────────────────────────────────
old_12t = (
    'title=title("Coefficient Estimates: Model 3a vs 3b",\n'
    '                "95% CI · Country-clustered SE"),'
)
new_12t = "title=None,"
if old_12t in src:
    src = src.replace(old_12t, new_12t, 1)
    print("  ✓ fig12: in-figure title removed")
else:
    print("  ✗ fig12: title pattern not found")


# ── EDIT 5: fig13 — drop in-figure title (handles multi-line) ─────────────────
old_13t = (
    'title=title("ECI vs Human Capital, by Production Value Quartile",\n'
    '                "Dotted lines = OLS trendline per quartile · forest-adjusted sample"),'
)
new_13t = "title=None,"
if old_13t in src:
    src = src.replace(old_13t, new_13t, 1)
    print("  ✓ fig13: in-figure title removed")
else:
    print("  · fig13: title pattern not found (may differ in your copy)")


# Generic title-stripper for the remaining charts the page references
# Uses balanced-paren matching to handle multi-line title=title(...) calls.
def strip_title(src_text, fig_name):
    """Find figN.update_layout( ... title=title(...), ... ) and replace title=title(...) with title=None."""
    start_marker = f"{fig_name}.update_layout("
    s = src_text.find(start_marker)
    if s < 0:
        return src_text, False
    # Search forward for 'title=title(' within this update_layout call
    title_idx = src_text.find("title=title(", s)
    if title_idx < 0:
        return src_text, False
    # Find matching close paren
    depth = 0
    i = title_idx + len("title=title")  # points at '('
    while i < len(src_text):
        ch = src_text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                # i points at the closing paren of title(...)
                # We expect a comma after it
                end = i + 1
                if end < len(src_text) and src_text[end] == ",":
                    end += 1
                return src_text[:title_idx] + "title=None," + src_text[end:], True
                break
        i += 1
    return src_text, False


for figname in ["fig04", "fig05", "fig06", "fig11", "fig17", "fig18", "fig19",
                "fig20", "fig21"]:
    src, changed = strip_title(src, figname)
    if changed:
        print(f"  ✓ {figname}: in-figure title removed")


# Write back
if len(src) != orig_len:
    with open(target, "w") as f:
        f.write(src)
    print(f"\nWrote {target} ({len(src):,} chars, was {orig_len:,})")
else:
    print("\nNo changes written.")
