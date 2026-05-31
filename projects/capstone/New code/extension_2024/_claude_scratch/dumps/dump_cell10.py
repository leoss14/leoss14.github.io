import json
NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e1_data_pull.ipynb'
with open(NB) as f:
    nb = json.load(f)
print(''.join(nb['cells'][10]['source']))
