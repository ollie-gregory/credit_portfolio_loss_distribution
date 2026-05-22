import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm

everyday_green = '#11B67A'
racing_green = '#024731'
heritage_green = '#006A4D'

class Simulator:

    def __init__(self):
        self.portfolio_df = None
        self.cov = None
        self.C = None
        self.losses = None

    def _make_pd(self, cov, eps=1e-8):
        cov = (cov + cov.T) / 2
        lam_min = np.linalg.eigvalsh(cov).min()
        delta = max(0.0, -lam_min + eps)
        return cov + delta * np.eye(cov.shape[0])
    
    def _chol_from_cov(self, cov, eps=1e-8):
        cov_pd = self._make_pd(cov, eps=eps)
        C = np.linalg.cholesky(cov_pd)
        return cov_pd, C
    
    def _prep_inputs(self, loan_book):
        EAD = loan_book['ead'].to_numpy()
        LGD = loan_book['lgd'].to_numpy()
        PD = loan_book['pd'].to_numpy()
        RSQ = loan_book['rsq'].to_numpy()
        return EAD, LGD, PD, RSQ
    
    def _compute_thresholds(self, PD, eps=1e-12):
        PD_clipped = np.clip(PD, eps, 1.0 - eps)
        return norm.ppf(PD_clipped)

    def _prep_sim_vectors(self, PD, RSQ, LGD, EAD):
        thr = self._compute_thresholds(PD)[:, None]
        sqrt_rsq = np.sqrt(np.clip(RSQ, 0.0, 1.0))[:, None]
        weight = (LGD * EAD)[:, None]
        return thr, sqrt_rsq, weight
    
    def _build_factor_loadings(self, factor_index, M, N, C):

        F = np.zeros((M, N), dtype=float)

        for i, (_, row) in enumerate(self.portfolio_df[['region', 'sector']].iterrows()):
            r, s = row['region'], row['sector']
            F[i, factor_index[r]] = 1.0
            F[i, factor_index[s]] = 1.0

        FC = F @ C
        row_norms = np.linalg.norm(FC, axis=1, keepdims=True)
        FC_hat = FC / row_norms
        return FC_hat
    
    def simulate_losses(
        self,
        portfolio_df,
        factor_index,
        cov,
        n_scenarios=100_000,
        chunk=1_000,
        seed=42,
        return_z=False,
        return_defaults=False,
        eps_pd=1e-8
    ):
        self.portfolio_df = portfolio_df.reset_index(drop=True)
        self.cov, self.C = self._chol_from_cov(np.asarray(cov, float), eps=eps_pd)
        C = self.C

        M = len(self.portfolio_df)
        N = len(factor_index)

        FC_hat = self._build_factor_loadings(factor_index, M, N, C)
        EAD, LGD, PD, RSQ = self._prep_inputs(self.portfolio_df)
        thr, sqrt_rsq, weight = self._prep_sim_vectors(PD, RSQ, LGD, EAD)

        rng = np.random.default_rng(seed)

        losses_out = np.empty(n_scenarios)

        filled = 0

        with tqdm(total=n_scenarios, desc='Running Simulations') as pbar:
            while filled < n_scenarios:
                s = min(chunk, n_scenarios - filled)
                sl = slice(filled, filled + s)

                Z = rng.standard_normal(size=(N, s))
                
                Y = FC_hat @ Z
                Xi = rng.standard_normal(size=(M, s))
                X = sqrt_rsq * Y + np.sqrt(1.0 - sqrt_rsq ** 2) * Xi

                defaults = X <= thr

                losses_out[sl] = (defaults * weight).sum(axis=0)

                filled += s
                pbar.update(s)
        
        self.losses = pd.Series(losses_out)

        return self.losses
    
    def analytic_EL(self):
        EAD, LGD, PD, _ = self._prep_inputs(self.portfolio_df)
        return float(np.sum(PD*LGD*EAD))
    
    def var_quantile(self, alpha=0.99):
        return float(np.quantile(self.losses.to_numpy(), alpha))
    
    def expected_shortfall(self, alpha=0.99):
        arr = self.losses.to_numpy()
        q = np.quantile(arr, alpha)
        tail = arr[arr >= q]
        return float(tail.mean()) if len(tail) > 0 else float(q)
    
    def plot_loss_distribution(self):

        arr = self.losses.to_numpy()
        denom = self.portfolio_df['ead'].sum()

        fig, ax = plt.subplots(figsize=(6.75, 3.75), dpi=300)

        ax.hist(arr / denom, bins=60, color=everyday_green, edgecolor=None)

        ax.axvline(self.analytic_EL() / denom, color=racing_green, lw=2, label='Analytic EL')
        ax.axvline(arr.mean() / denom, color=racing_green, lw=2, ls=':', label='Simulated EL')
        ax.axvline(self.var_quantile(0.95) / denom, color=racing_green, lw=2, ls='--', label='VaR 95%')
        ax.axvline(self.var_quantile(0.99) / denom, color=racing_green, lw=2, ls='-.', label='VaR 99%')

        ax.set_title('Portfolio Loss Distribution')
        ax.set_xlabel('Loss Rate')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid()
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1))
        fig.tight_layout()
        return fig
    


systematic_shocks = pd.read_csv('data/systematic_shocks.csv', index_col='Quarter')
cov = systematic_shocks.cov()

portfolio = pd.read_csv('data/loan_book.csv')
pd_mapping = pd.read_csv('data/pd_mapping.csv')

portfolio = pd.merge(portfolio, pd_mapping, left_on='credit_rating', right_on='Credit Rating')

factors = list(cov.columns)
factor_index = {name: j for j, name in enumerate(factors)}

sim = Simulator()
losses = sim.simulate_losses(
    portfolio,
    factor_index,
    cov,
    n_scenarios=10_000
)

fig = sim.plot_loss_distribution()

fig.savefig('loss_distribution.png', dpi=300)

print(f'Analytic EL: {sim.analytic_EL() / portfolio['ead'].sum()}')
print(f'Simulated EL: {sim.losses.mean() / portfolio['ead'].sum()}')
print(f'VaR (99%): {sim.var_quantile(0.99) / portfolio['ead'].sum()}')
print(f'VaR (95%): {sim.var_quantile(0.95) / portfolio['ead'].sum()}')
print(f'Expected Shortfall (99%): {sim.expected_shortfall(0.99) / portfolio['ead'].sum()}')