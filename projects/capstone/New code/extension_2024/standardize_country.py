"""
standardize_country.py — local shim for extension_2024/

Replaces the v2 pipeline's `standardize_country` helper module, which was
authored by a teammate and never checked into this repo. Provides the same
public API expected by v2/1_cleaning_master_data_FINAL.ipynb:

    add_iso3(df, name_col, iso3_col) -> DataFrame
    ALIAS_TO_ISO3: dict[str, str]
    ISO3_TO_WB:    dict[str, str]

Implementation uses pycountry for fuzzy matching, with an alias table to
handle World Bank-style country names that pycountry does not match cleanly
(e.g. "Iran, Islamic Rep.").
"""

import pandas as pd
import pycountry


# Manual aliases for source-specific names that pycountry does not resolve
# cleanly. Extend this dict if you see unmatched names in the production
# warning ("dropped N rows with unmatched country names") at notebook run time.
ALIAS_TO_ISO3 = {
    # World Bank canonical forms
    "Bahamas, The":                       "BHS",
    "Congo, Dem. Rep.":                   "COD",
    "Congo, Rep.":                        "COG",
    "Czech Republic":                     "CZE",
    "Czechia":                            "CZE",
    "Egypt, Arab Rep.":                   "EGY",
    "Gambia, The":                        "GMB",
    "Hong Kong SAR, China":               "HKG",
    "Iran, Islamic Rep.":                 "IRN",
    "Korea, Dem. People's Rep.":          "PRK",
    "Korea, Rep.":                        "KOR",
    "Kyrgyz Republic":                    "KGZ",
    "Lao PDR":                            "LAO",
    "Macao SAR, China":                   "MAC",
    "Macedonia, FYR":                     "MKD",
    "North Macedonia":                    "MKD",
    "Micronesia, Fed. Sts.":              "FSM",
    "Russian Federation":                 "RUS",
    "Slovak Republic":                    "SVK",
    "Slovakia":                           "SVK",
    "St. Kitts and Nevis":                "KNA",
    "St. Lucia":                          "LCA",
    "St. Vincent and the Grenadines":     "VCT",
    "Syrian Arab Republic":               "SYR",
    "Turkiye":                            "TUR",
    "Turkey":                             "TUR",
    "Venezuela, RB":                      "VEN",
    "Vietnam":                            "VNM",
    "Viet Nam":                           "VNM",
    "West Bank and Gaza":                 "PSE",
    "Yemen, Rep.":                        "YEM",
    "Cape Verde":                         "CPV",
    "Cabo Verde":                         "CPV",
    "Cote d'Ivoire":                      "CIV",
    "Côte d'Ivoire":                      "CIV",
    "Eswatini":                           "SWZ",
    "Swaziland":                          "SWZ",
    "East Timor":                         "TLS",
    "Timor-Leste":                        "TLS",

    # EI / OWID / USGS short forms
    "USA":                                "USA",
    "United States":                      "USA",
    "United States of America":           "USA",
    "UK":                                 "GBR",
    "United Kingdom":                     "GBR",
    "Russia":                             "RUS",
    "Iran":                               "IRN",
    "South Korea":                        "KOR",
    "North Korea":                        "PRK",
    "Brunei":                             "BRN",
    "Brunei Darussalam":                  "BRN",
    "China, Hong Kong SAR":               "HKG",
    "Hong Kong":                          "HKG",
    "Macao":                              "MAC",
    "Macau":                              "MAC",
    "Taiwan":                             "TWN",
    "Taiwan, China":                      "TWN",
    "Bolivia":                            "BOL",
    "Tanzania":                           "TZA",
    "Moldova":                            "MDA",
    "Laos":                               "LAO",
    "Burma":                              "MMR",
    "Myanmar":                            "MMR",
    "Palestine":                          "PSE",
    "DR Congo":                           "COD",
    "Democratic Republic of Congo":       "COD",
    "Dem. Rep. Congo":                    "COD",
    "China Hong Kong SAR":                "HKG",
    "Republic of Korea":                  "KOR",
    "Korea, Republic of":                 "KOR",

    # Aggregates and non-countries that should be excluded (return None)
    "World":                              None,
    "OECD":                               None,
    "European Union":                     None,
    "Other Africa":                       None,
    "Other Asia Pacific":                 None,
    "Other CIS":                          None,
    "Other Europe":                       None,
    "Other Middle East":                  None,
    "Other Caribbean":                    None,
    "Other Eastern Africa":                None,
    "Other Middle Africa":                 None,
    "Other Southern Africa":               None,
    "Other Western Africa":                None,
    "Other South America":                 None,
    "Other Central America":               None,
    "Czechoslovakia":                     None,
    "East Germany":                       None,
    "Serbia and Montenegro":              None,
    "Netherlands Antilles":               None,
    "Other Northern Africa":              None,
    "Other S. & Cent. America":           None,
    "USSR":                               None,
    "Yugoslavia":                         None,
}


def _lookup_iso3(name):
    """Resolve a country name string to an ISO3 alpha code. None on failure."""
    if name is None or pd.isna(name):
        return None
    s = str(name).strip()
    if not s:
        return None
    if s in ALIAS_TO_ISO3:
        return ALIAS_TO_ISO3[s]
    # Exact pycountry lookup (handles ISO names, alpha2, alpha3, numeric)
    try:
        c = pycountry.countries.lookup(s)
        return c.alpha_3
    except LookupError:
        pass
    # Fuzzy fallback
    try:
        matches = pycountry.countries.search_fuzzy(s)
        if matches:
            return matches[0].alpha_3
    except (LookupError, Exception):
        pass

    # Footnote-digit fallback (handles "Brazil1", "India2" from EI Excel where
    # superscript footnote markers got concatenated onto the country name).
    # Strip trailing digits and retry the alias / pycountry chain ONCE.
    import re as _re
    stripped = _re.sub(r'\d+$', '', s).strip()
    if stripped and stripped != s:
        if stripped in ALIAS_TO_ISO3:
            return ALIAS_TO_ISO3[stripped]
        try:
            return pycountry.countries.lookup(stripped).alpha_3
        except LookupError:
            pass
        try:
            matches = pycountry.countries.search_fuzzy(stripped)
            if matches:
                return matches[0].alpha_3
        except (LookupError, Exception):
            pass

    return None


def add_iso3(df, name_col="Country Name", iso3_col="Country Code"):
    """Add an ISO3 column based on country names in name_col.

    Rows where the name cannot be resolved get NaN in iso3_col and should be
    filtered out by the caller.
    """
    df = df.copy()
    df[iso3_col] = df[name_col].apply(_lookup_iso3)
    return df


# Reverse map: ISO3 -> World Bank-style canonical name. The pycountry default
# names use the full ISO standard form; WB uses abbreviated forms for some
# countries. Overrides below match the WB conventions used in the existing
# Master.csv panel for backward compatibility.
ISO3_TO_WB = {c.alpha_3: c.name for c in pycountry.countries}
ISO3_TO_WB.update({
    "USA": "United States",
    "GBR": "United Kingdom",
    "RUS": "Russian Federation",
    "IRN": "Iran, Islamic Rep.",
    "KOR": "Korea, Rep.",
    "PRK": "Korea, Dem. People's Rep.",
    "EGY": "Egypt, Arab Rep.",
    "VEN": "Venezuela, RB",
    "YEM": "Yemen, Rep.",
    "BHS": "Bahamas, The",
    "GMB": "Gambia, The",
    "CZE": "Czech Republic",
    "SVK": "Slovak Republic",
    "KGZ": "Kyrgyz Republic",
    "LAO": "Lao PDR",
    "MKD": "Macedonia, FYR",
    "COD": "Congo, Dem. Rep.",
    "COG": "Congo, Rep.",
    "FSM": "Micronesia, Fed. Sts.",
    "SYR": "Syrian Arab Republic",
    "PSE": "West Bank and Gaza",
    "MAC": "Macao SAR, China",
    "HKG": "Hong Kong SAR, China",
    "VNM": "Vietnam",
    "CIV": "Cote d'Ivoire",
    "CPV": "Cabo Verde",
    "TUR": "Turkiye",
})
