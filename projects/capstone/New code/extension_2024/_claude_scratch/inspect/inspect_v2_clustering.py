"""Inspect v2 clustering notebook structure."""
import json
NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/code/v2/4_Clustering_FINAL.ipynb'
with open(NB) as f:
    nb = json.load(f)
print(f'cells: {len(nb["cells"])}')
print()
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    if c['cell_type'] == 'markdown':
        head = src.split('\n')[0][:90]
        print(f'[{i:2d}] MD {head}')
    else:
        lines = src.split('\n')
        first = next((ln for ln in lines if ln.strip() and not ln.strip().startswith('#')), '')[:80]
        tags = []
        for kw, lbl in [('KMeans', 'KM'), ('AgglomerativeClustering', 'AGG'),
                        ('DBSCAN', 'DBS'), ('GaussianMixture', 'GMM'),
                        ('silhouette', 'SIL'), ('elbow', 'ELB'),
                        ('PCA', 'PCA'), ('TSNE', 'TSNE'), ('UMAP', 'UMAP'),
                        ('StandardScaler', 'SCL'),
                        ('to_csv', 'OUT')]:
            if kw in src:
                tags.append(lbl)
        print(f'[{i:2d}] PY {tags} {first}')
