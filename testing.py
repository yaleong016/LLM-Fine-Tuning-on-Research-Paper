import pandas as pd

pd.set_option("display.max_colwidth", None)   # Show full text in each cell
pd.set_option("display.max_columns", None)   # Show all columns
pd.set_option("display.width", 0)            # Don't wrap to next line

df_abs = pd.read_parquet("data/abstract_pairs.parquet")

print(df_abs.iloc[1])   # first row, full text