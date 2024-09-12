import pandas as pd
df_csv_append = pd.DataFrame()
df1 = pd.read_csv("chatgpt_genarated_dataset.csv")
df2 = pd.read_csv("human_genarated_dataset.csv")
df_csv_append = pd.concat([pd.read_csv("chatgpt_genarated_dataset.csv"),pd.read_csv("human_genarated_dataset.csv")],ignore_index=True)
print(df_csv_append.to_csv("human_chatgpt_genarated_dataset.csv",index=False))
