import pandas as pd
df = pd.DataFrame(columns=('col1', 'col2', 'col3'))
for i in range(5):
   df.loc[i] = ['<some value for first>', '<some value for second>', '<some value for third>']

print(df.head())