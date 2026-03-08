from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import os

data = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))

texts = data.data

df = pd.DataFrame({
    "text": texts
})

os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/corpus.csv", index=False)