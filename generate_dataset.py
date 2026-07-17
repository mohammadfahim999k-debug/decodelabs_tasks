"""
Generates the synthetic credit-card fraud dataset used by pipeline.py.

Run this first: `python generate_dataset.py`
It creates `creditcard.csv` (284,807 rows x 31 columns, ~160MB) in the
current directory. The file is NOT checked into the repo (too large for a
normal GitHub upload) -- regenerate it locally with this script instead.
The random seed is fixed, so the output is identical every time.

See pipeline.py / Project2_Fraud_Detection_Report.docx for the full
explanation of why this dataset is synthetic (the real Kaggle
mlg-ulb/creditcardfraud file could not be downloaded in the original
offline development sandbox) and how closely it matches the real data's
statistical properties.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42


def generate_dataset(n_total=284807, n_fraud=492, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n_legit = n_total - n_fraud

    time_legit = rng.uniform(0, 172792, n_legit)
    time_fraud = rng.uniform(0, 172792, n_fraud)

    amount_legit = np.clip(rng.lognormal(mean=3.2, sigma=1.3, size=n_legit), 0, 25691.16)
    amount_fraud = np.clip(rng.lognormal(mean=3.0, sigma=1.6, size=n_fraud), 0, 25691.16)

    n_components = 28
    V_legit = rng.normal(loc=0, scale=1.0, size=(n_legit, n_components))
    V_fraud = rng.normal(loc=0, scale=1.0, size=(n_fraud, n_components))

    # Realistic, overlapping (not perfectly separable) fraud signature,
    # loosely mirroring which components (V10, V12, V14, V17, V18, etc.)
    # show the strongest fraud/legit separation in the real dataset.
    signal_components = [9, 11, 13, 16, 17]
    for idx in signal_components:
        V_fraud[:, idx] = rng.normal(loc=-1.8, scale=1.4, size=n_fraud)
    V_fraud[:, 6] = rng.normal(loc=-1.2, scale=1.3, size=n_fraud)
    V_fraud[:, 3] = rng.normal(loc=1.4, scale=1.3, size=n_fraud)

    cols_V = [f"V{i+1}" for i in range(n_components)]
    df_legit = pd.DataFrame(V_legit, columns=cols_V)
    df_legit["Time"] = time_legit
    df_legit["Amount"] = amount_legit
    df_legit["Class"] = 0

    df_fraud = pd.DataFrame(V_fraud, columns=cols_V)
    df_fraud["Time"] = time_fraud
    df_fraud["Amount"] = amount_fraud
    df_fraud["Class"] = 1

    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df = df.sort_values("Time").reset_index(drop=True)
    df = df[["Time"] + cols_V + ["Amount", "Class"]]
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("creditcard.csv", index=False)
    print(f"Wrote creditcard.csv: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Fraud rate: {df['Class'].mean()*100:.3f}%")
