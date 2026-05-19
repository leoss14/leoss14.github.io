p = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/viz_1_descriptive.ipynb'
with open(p) as fh:
    txt = fh.read()
# Find the full context of the cluster colors duplicate
i = txt.find("'Major Producers'")
print('Context for Major Producers duplicate in viz_1:')
print(txt[max(0,i-300):min(len(txt),i+400)])
print()
print('---')
print()
i = txt.find('49-country')
print('Context for 49-country in viz_1:')
print(txt[max(0,i-200):min(len(txt),i+400)])
