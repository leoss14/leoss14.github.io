/* =============================================
   SKILLS DATABASE
   Single source of truth for all skill tags
   across the portfolio. Edit here, updates everywhere.

   HOW TO USE:
   1. Add <script src="js/skills.js"></script> to any page
   2. Call renderSkills('container-id') or renderSkills('container-id', ['methods_ml', 'python'])
   3. Or access SKILLS directly for custom rendering

   CATEGORIES:
   - languages        : Programming languages and statistical environments
   - python           : Python ecosystem libraries
   - methods_ml       : Machine learning and classification techniques
   - methods_econ     : Econometric and causal inference methods
   - methods_spatial  : Spatial, geographic, and concentration analysis
   - methods_network  : Network and graph analysis
   - data_tools       : APIs, scraping, data access
   - software         : Desktop software and platforms

   SOURCED FROM:
   - Capstone (Moody's): PCA, K-Means, PanelOLS, Ridge/Lasso/ElasticNet,
     RF, XGBoost, LightGBM, SHAP, 5-fold CV
   - Central bank: Markov-switching, NLS, block-bootstrap HAC,
     sup-Wald/Chow, expanding-window CV, logistic regression
   - Uber NYC: K-Means, Gini/Lorenz, bootstrap CI, Moran's I,
     LISA, Jensen-Shannon divergence, OD flows, Haversine,
     KS test, PySAL/GeoPandas
   - Pink tax: TF-IDF, gradient boosting, HSV color extraction,
     web scraping, BeautifulSoup
   - Italian exports: product space, RCA, igraph, centrality
   - CV: DiD, IV, RDD, fixed effects, PyTorch
   ============================================= */

var SKILLS = {

    languages: {
        label: "Languages",
        items: [
            "Python",
            "R",
            "Stata",
            "SQL"
        ]
    },

    python: {
        label: "Python Ecosystem",
        items: [
            "pandas",
            "numpy",
            "scikit-learn",
            "statsmodels",
            "PyTorch",
            "GeoPandas",
            "PySAL",
            "Plotly",
            "matplotlib",
            "seaborn",
            "BeautifulSoup"
        ]
    },

    methods_ml: {
        label: "Machine Learning",
        items: [
            "K-Means Clustering",
            "PCA",
            "LASSO / Ridge / ElasticNet",
            "Gradient Boosting / XGBoost",
            "Random Forest",
            "TF-IDF",
            "SHAP Interpretability",
            "Cross-Validation"
        ]
    },

    methods_econ: {
        label: "Econometrics",
        items: [
            "Panel Fixed Effects",
            "Markov-Switching Models",
            "Nonlinear Least Squares",
            "Structural Break Tests",
            "Bootstrap Inference",
            "Causal Inference (DiD, IV, RDD)"
        ]
    },

    methods_spatial: {
        label: "Spatial & Geographic",
        items: [
            "Moran's I / LISA",
            "Gini / Lorenz Curves",
            "Choropleth Mapping",
            "OD Flow Analysis",
            "Jensen-Shannon Divergence"
        ]
    },

    methods_network: {
        label: "Network Analysis",
        items: [
            "Product Space",
            "Revealed Comparative Advantage",
            "igraph",
            "Centrality Measures"
        ]
    },

    data_tools: {
        label: "Data & APIs",
        items: [
            "FRED API",
            "Web Scraping",
            "USGS / World Bank / Comtrade",
            "Git"
        ]
    },

    software: {
        label: "Software",
        items: [
            "Excel"
        ]
    }
};


/* ----- Rendering helpers ----- */

/**
 * Render skill tags into a container element.
 *
 * @param {string} containerId  - ID of the target DOM element
 * @param {string[]} [categories] - Which categories to show (default: all)
 * @param {object} [options]
 * @param {boolean} [options.showLabels=true] - Show category headings
 * @param {string}  [options.tagClass='skill-tag'] - CSS class for each tag
 * @param {string}  [options.style='mono'] - 'mono' for code-style, 'pill' for rounded pills
 */
function renderSkills(containerId, categories, options) {
    var el = document.getElementById(containerId);
    if (!el) return;

    var opts = options || {};
    var showLabels = opts.showLabels !== false;
    var tagClass = opts.tagClass || 'skill-tag';
    var style = opts.style || 'mono';

    var cats = categories || Object.keys(SKILLS);
    var html = '';

    cats.forEach(function(key) {
        var cat = SKILLS[key];
        if (!cat) return;

        if (showLabels && cats.length > 1) {
            html += '<div class="skills-category-label">' + cat.label + '</div>';
        }
        html += '<div class="skills-row">';
        cat.items.forEach(function(item) {
            var cls = style === 'pill' ? 'interest-tag' : tagClass;
            html += '<span class="' + cls + '">' + item + '</span>';
        });
        html += '</div>';
    });

    el.innerHTML = html;
}
