"""Inspect v2 ML notebook structure."""
import json
NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/code/v2/5_ML_FINAL.ipynb'
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
        for kw, lbl in [('RandomForestRegressor', 'RFR'), ('RandomForestClassifier', 'RFC'),
                        ('XGB', 'XGB'), ('LGBM', 'LGB'), ('GradientBoosting', 'GBR'),
                        ('LinearRegression', 'LIN'), ('Ridge', 'RDG'), ('Lasso', 'LAS'),
                        ('SHAP', 'SHAP'), ('shap', 'SHAP'),
                        ('train_test_split', 'TTS'), ('KFold', 'KF'), ('cross_val', 'CV'),
                        ('GridSearchCV', 'GS'), ('TimeSeriesSplit', 'TSS'),
                        ('feature_importance', 'FI'), ('plot_importance', 'FI'),
                        ('StandardScaler', 'SCL'), ('PCA', 'PCA'),
                        ('to_csv', 'OUT'), ('to_pickle', 'PKL')]:
            if kw in src:
                tags.append(lbl)
        print(f'[{i:2d}] PY {tags} {first}')
