"""Rubin's-rules pooling for Multiple Imputation by Chained Equations.

Used by downstream notebooks (e3, e4, e5, e6) that need to run analyses across
the M imputed datasets in Master_v2_imputations.parquet and combine results.

Reference: Rubin (1987) "Multiple Imputation for Nonresponse in Surveys".

Usage:
    from _mice_pool import pool_scalars, load_imputations, iter_imputations

    # Load all M imputed panels
    panels = load_imputations()   # list of M DataFrames

    # Or iterate one at a time (memory-friendly)
    betas, ses = [], []
    for panel in iter_imputations():
        res = run_regression(panel)
        betas.append(res.beta)
        ses.append(res.se)

    pooled = pool_scalars(betas, ses)
    print(f'beta = {pooled.point:.4f} (SE = {pooled.se:.4f})')
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import numpy as np
import pandas as pd

DEFAULT_PARQUET = Path(__file__).parent / 'intermediary' / 'Master_v2_imputations.parquet'


@dataclass
class PooledScalar:
    """Rubin's-rules pooled scalar estimate (e.g., a regression coefficient)."""
    point: float          # Combined point estimate (mean of M estimates)
    se: float             # Combined standard error
    df: float             # Degrees of freedom for t-test
    within_var: float     # Within-imputation variance (mean of variances)
    between_var: float    # Between-imputation variance (variance of means)
    total_var: float      # Total variance: within + (1 + 1/M) * between
    fmi: float            # Fraction of missing information
    m: int                # Number of imputations

    def ci(self, alpha: float = 0.05) -> tuple[float, float]:
        """Two-sided confidence interval using t-distribution with df degrees of freedom."""
        from scipy.stats import t
        z = t.ppf(1 - alpha / 2, df=self.df)
        return (self.point - z * self.se, self.point + z * self.se)

    def t_stat(self) -> float:
        """t-statistic for H0: point = 0."""
        return self.point / self.se if self.se > 0 else np.nan

    def p_value(self) -> float:
        """Two-sided p-value for H0: point = 0."""
        from scipy.stats import t
        return 2 * (1 - t.cdf(abs(self.t_stat()), df=self.df))

    def __repr__(self) -> str:
        return (f'PooledScalar(point={self.point:.4f}, se={self.se:.4f}, '
                f'df={self.df:.1f}, t={self.t_stat():.2f}, p={self.p_value():.4g})')


def pool_scalars(estimates: list[float], variances: list[float]) -> PooledScalar:
    """Combine M point estimates and their variances via Rubin's rules.

    Args:
        estimates: list of M scalar estimates (one per imputed dataset)
        variances: list of M variances (squared standard errors)

    Returns:
        PooledScalar with combined point, SE, and degrees of freedom.
    """
    estimates = np.asarray(estimates, dtype=float)
    variances = np.asarray(variances, dtype=float)
    m = len(estimates)
    if m != len(variances):
        raise ValueError(f'estimates ({m}) and variances ({len(variances)}) length mismatch')
    if m < 2:
        raise ValueError(f'need at least 2 imputations to compute between-variance, got {m}')

    point = estimates.mean()
    within_var = variances.mean()
    between_var = estimates.var(ddof=1)  # variance of point estimates across M

    # Rubin's total variance: U_bar + (1 + 1/M) * B
    total_var = within_var + (1 + 1 / m) * between_var
    se = np.sqrt(total_var)

    # Degrees of freedom (Rubin 1987)
    if between_var > 0:
        r = (1 + 1 / m) * between_var / within_var
        df = (m - 1) * (1 + 1 / r) ** 2
    else:
        df = np.inf

    # Fraction of missing information
    fmi = ((1 + 1 / m) * between_var) / total_var if total_var > 0 else 0.0

    return PooledScalar(
        point=float(point),
        se=float(se),
        df=float(df),
        within_var=float(within_var),
        between_var=float(between_var),
        total_var=float(total_var),
        fmi=float(fmi),
        m=m,
    )


def pool_vectors(estimates_list: list[np.ndarray], cov_list: list[np.ndarray]) -> dict:
    """Combine M vector estimates and their covariance matrices via Rubin's rules.

    Args:
        estimates_list: list of M arrays, each shape (k,)
        cov_list: list of M covariance matrices, each shape (k, k)

    Returns:
        Dict with 'point' (k,), 'cov' (k, k), 'se' (k,), 'df' per-parameter (k,)
    """
    k = len(estimates_list[0])
    estimates = np.array(estimates_list)  # (M, k)
    m = len(estimates)

    point = estimates.mean(axis=0)                                # (k,)
    within_cov = np.mean(cov_list, axis=0)                        # (k, k)
    between_cov = np.cov(estimates.T, ddof=1) if m > 1 else np.zeros((k, k))  # (k, k)
    total_cov = within_cov + (1 + 1 / m) * between_cov
    se = np.sqrt(np.diag(total_cov))

    # Per-parameter df
    diag_w = np.diag(within_cov)
    diag_b = np.diag(between_cov)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = (1 + 1 / m) * diag_b / diag_w
        df = (m - 1) * (1 + 1 / r) ** 2
    df = np.where(np.isfinite(df), df, np.inf)

    return {
        'point': point,
        'cov': total_cov,
        'se': se,
        'df': df,
        'within_cov': within_cov,
        'between_cov': between_cov,
        'm': m,
    }


def pool_predictions(predictions: list[np.ndarray]) -> dict:
    """Pool prediction arrays (e.g. forecasts) across M imputations.

    For predictions, the combined point estimate is the mean across M and the
    'prediction uncertainty' is just the between-imputation variance (we don't
    typically have per-prediction within-variance from a regression model).

    Args:
        predictions: list of M arrays, each of shape (n,) or (n, k)

    Returns:
        Dict with 'point' (mean prediction), 'between_sd' (uncertainty due to imputation)
    """
    arr = np.stack(predictions, axis=0)  # (M, ...)
    return {
        'point': arr.mean(axis=0),
        'between_sd': arr.std(axis=0, ddof=1),
        'm': len(predictions),
    }


def load_imputations(path: Path | str = DEFAULT_PARQUET) -> list[pd.DataFrame]:
    """Load all M imputed panels from the stacked parquet as a list of DataFrames.

    Each DataFrame has the same shape as Master_v2 (one panel per imputation).
    Loads all M into memory; use iter_imputations() for memory-efficient streaming.
    """
    full = pd.read_parquet(path)
    if 'imputation_id' not in full.columns:
        raise ValueError(f'expected imputation_id column in {path}')
    panels = [g.drop(columns='imputation_id').reset_index(drop=True)
              for _, g in full.groupby('imputation_id')]
    return panels


def iter_imputations(path: Path | str = DEFAULT_PARQUET) -> Iterator[tuple[int, pd.DataFrame]]:
    """Yield (imputation_id, panel_df) one at a time. Memory-friendly for large M."""
    full = pd.read_parquet(path)
    if 'imputation_id' not in full.columns:
        raise ValueError(f'expected imputation_id column in {path}')
    for imp_id, g in full.groupby('imputation_id'):
        yield int(imp_id), g.drop(columns='imputation_id').reset_index(drop=True)


def n_imputations(path: Path | str = DEFAULT_PARQUET) -> int:
    """Quick check of M without loading the full file."""
    full = pd.read_parquet(path, columns=['imputation_id'])
    return int(full['imputation_id'].nunique())


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
if __name__ == '__main__':
    # Synthetic test: known truth, see if pooling recovers it
    np.random.seed(0)
    true_beta = 2.5
    M = 10
    # Simulate M regression results where each gets a noisy estimate of true_beta
    betas = [true_beta + np.random.normal(0, 0.1) for _ in range(M)]
    ses_2 = [0.05 ** 2 + np.random.uniform(0, 0.001) for _ in range(M)]

    pooled = pool_scalars(betas, ses_2)
    print('Self-test:')
    print(f'  True beta:     {true_beta}')
    print(f'  Pooled point:  {pooled.point:.4f}')
    print(f'  Pooled SE:     {pooled.se:.4f}')
    print(f'  df:            {pooled.df:.1f}')
    print(f'  CI (95%):      ({pooled.ci()[0]:.4f}, {pooled.ci()[1]:.4f})')
    print(f'  FMI:           {pooled.fmi:.3f}')
    print(f'  Repr:          {pooled!r}')
