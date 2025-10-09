import pandas as pd
df = pd.concat([pd.read_csv('results/dct.csv'),
                pd.read_csv('results/ksvd.csv'),
                pd.read_csv('results/pca.csv'),
                pd.read_csv('results/generic_compress.csv')],
               ignore_index=True, sort=False)
df.to_csv('results/all_results.csv', index=False)
print(df)
