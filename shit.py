import pandas as pd 

df = pd.read_csv("/nethome/akolchina/Combi/data/NPAtlas_download_2024_09.tsv", sep="\t")
print(df["origin_type"].value_counts())