import pandas as pd
import numpy as np
np.random.seed(1)

df = pd.read_csv('Akhbarona.ma.csv')
df['category'] = '0'
df = df.rename(columns = {'Body' :'text'}, inplace = False)
print(df.head(5000).to_csv("human_genarated_dataset1.csv",index_label=False))

df1 = pd.read_csv('human_genarated_dataset1.csv', index_col=0)
df1 = df1[['text','category']]
# Removing unnamed columns using drop function
#df1 = df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

print(df1.head(5000).to_csv("human_genarated_dataset.csv",index=False))
