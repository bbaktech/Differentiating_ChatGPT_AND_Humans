import pandas as pd
import numpy as np
np.random.seed(1)

df = pd.read_parquet('train-00000-of-00001-b2881e1b9f14c3b1.parquet')
df = df[['output']]
df['category'] = '1'
df = df.rename(columns = {'output' :'text'}, inplace = False)
df.head(5000).to_csv('chatgpt_genarated_dataset.csv', index=False)
