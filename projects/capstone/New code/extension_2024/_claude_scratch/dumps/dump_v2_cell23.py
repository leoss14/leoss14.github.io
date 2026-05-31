"""Dump cell 23 of 3_Imputing_FINAL.ipynb to extract the HIC exclusion implementation."""
import json
NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/code/v2/3_Imputing_FINAL.ipynb'
with open(NB) as f:
    nb = json.load(f)
src = ''.join(nb['cells'][23].get('source', []))
print(src)
