import pandas as pd

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)

df_abs = pd.read_parquet("data/abstract_pairs.parquet")

for idx, row in df_abs.iterrows():
    print("="*120)
    print(f"ROW {idx}")
    print("="*120)

    print("\nRAW ABSTRACT:")
    print(repr(row["input_text"]))  # or whatever the column is named

    print("\nTEACHER SUMMARY (TL;DR):")
    print(repr(row["summary"])) # replace with the real column name

    print("\n")

