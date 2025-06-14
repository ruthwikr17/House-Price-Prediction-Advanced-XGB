import pandas as pd

df = pd.read_csv("../data/Indian_Real_Estate_Clean_Data.csv")

# 1. Mean price by property type
print("Average Price by Property Type:")
print(
    df.groupby("property_type")["Price"]
    .agg(["mean", "count"])
    .sort_values(by="mean", ascending=False)
)

# 2. Mean price by BHK
print("\nAverage Price by BHK:")
print(
    df.groupby("BHK")["Price"]
    .agg(["mean", "count"])
    .sort_values(by="mean", ascending=False)
)


print("\nAverage Price by Property Type and BHK:")
print(df.groupby(["property_type", "BHK"])["Price"].mean().unstack().fillna("-"))
