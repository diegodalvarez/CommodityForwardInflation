import requests
import pandas as pd


# -----------------------------
# Settings
# -----------------------------

API_KEY = "lse_live_e5ba8980b8a6b4dce6d469d4ffdbcfec"

BASE_URL = "https://api.londonstrategicedge.com/vault"

headers = {
    "x-api-key": API_KEY
}


# -----------------------------
# Download catalog
# -----------------------------

response = requests.get(
    f"{BASE_URL}/catalog",
    headers=headers
)

response.raise_for_status()

catalog = response.json()


# -----------------------------
# Convert to dataframe
# -----------------------------

df_catalog = pd.DataFrame(catalog)


# -----------------------------
# Inspect datasets
# -----------------------------

print("Available datasets:")
print(df_catalog["dataset"].value_counts())


# -----------------------------
# Search for ES / S&P / SP500
# -----------------------------

df_es = (
    df_catalog
    .loc[
        lambda x: x.symbol.astype(str).str.contains(
            "ES|S&P|SP|500",
            case=False,
            na=False
        )
    ]
    [
        [
            "dataset",
            "symbol",
            "name",
            "first_tick",
            "last_tick"
        ]
    ]
)


print("\nMatching symbols:")
print(df_es)


# -----------------------------
# If nothing found, inspect futures
# -----------------------------

df_futures = (
    df_catalog
    .loc[
        lambda x: x.dataset.astype(str).str.contains(
            "future",
            case=False,
            na=False
        )
    ]
    [
        [
            "dataset",
            "symbol",
            "name",
            "first_tick",
            "last_tick"
        ]
    ]
)


print("\nFuture datasets:")
print(df_futures.head(100))