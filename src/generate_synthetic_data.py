import numpy as np
import pandas as pd

def generate_loan_book(n=50000, rsq=0.2, seed=42):

    np.random.seed(seed)

    # categories
    ratings = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
    sectors = ["CRE", "Corporate", "FI"]
    regions = ["US", "UK"]

    # rating distribution
    rating_probs = np.array([0.05, 0.10, 0.20, 0.30, 0.20, 0.10, 0.05])

    # sector distribution
    sector_probs = np.array([0.3, 0.5, 0.2])

    # region distribution
    region_probs = np.array([0.3, 0.7])

    # sample categorical fields
    credit_rating = np.random.choice(ratings, size=n, p=rating_probs)
    sector = np.random.choice(sectors, size=n, p=sector_probs)
    region = np.random.choice(regions, size=n, p=region_probs)

    # EAD
    ead = np.random.lognormal(mean=12, sigma=1.0, size=n)

    # LGD by sector
    lgd = np.zeros(n)
    lgd[sector == 'CRE'] = np.random.uniform(0.4, 0.6, size=(sector == 'CRE').sum())
    lgd[sector == 'Corporate'] = np.random.uniform(0.3, 0.5, size=(sector == 'Corporate').sum())
    lgd[sector == 'FI'] = np.random.uniform(0.2, 0.4, size=(sector == 'FI').sum())

    df = pd.DataFrame({
        "loan_id": np.arange(1, n+1),
        "ead": ead,
        "lgd": lgd,
        "credit_rating": credit_rating,
        "rsq": rsq,
        "sector": sector,
        "region": region
    })

    return df

def generate_factors(uk_gdp, us_gdp, seed=42):

    np.random.seed(seed)

    n = len(uk_gdp)

    # noise terms
    eps_cre = np.random.normal(size=n)
    eps_corp = np.random.normal(size=n)
    eps_fi = np.random.normal(size=n)

    # sector loadings
    a_cre, b_cre = 0.6, 0.6
    a_corp, b_corp = 0.5, 0.5
    a_fi, b_fi = 0.4, 0.4

    # ensure unit variance
    def build_factor(a, b, eps):
        return a*us_gdp + b*uk_gdp + np.sqrt(1 - a**2 - b**2) * eps
    
    df = pd.DataFrame({
        "US": us_gdp,
        "UK": uk_gdp,
        "CRE": build_factor(a_cre, b_cre, eps_cre),
        "Corporate": build_factor(a_corp, b_corp, eps_corp),
        "FI": build_factor(a_fi, b_fi, eps_fi),
    })

    return df

def generate_pd_vector():
    df = pd.DataFrame({
        "AAA": [0.0001],
        "AA": [0.0003],
        "A": [0.0008],
        "BBB": [0.002],
        "BB": [0.01],
        "B": [0.05],
        "CCC": [0.20],
    })

    df = df.T.rename(columns={0: "pd"})

    return df



macro_shocks = pd.read_csv('data/macro_shocks.csv')
us_gdp = macro_shocks['us_gdp'].to_numpy()
uk_gdp = macro_shocks['uk_gdp'].to_numpy()

loan_book = generate_loan_book()
systematic_shocks = generate_factors(uk_gdp, us_gdp)
pd_vector = generate_pd_vector()

factors = systematic_shocks.columns.to_list()
systematic_shocks['Quarter'] = macro_shocks['Quarter']
systematic_shocks = systematic_shocks[['Quarter'] + factors]

loan_book.to_csv('data/loan_book.csv', index=False)
systematic_shocks.to_csv('data/systematic_shocks.csv', index=False)
pd_vector.to_csv('data/pd_mapping.csv', index_label='Credit Rating')