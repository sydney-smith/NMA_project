import pandas as pd
import numpy as np

df = pd.read_csv('NMA_features.csv', index_col=None)

df.dropna()
features = ['exponent', 'offset',
       'theta_cf', 'theta_pow', 'theta_band', 'beta_cf', 'beta_pow',
       'beta_band', 'gamma_cf', 'gamma_pow', 'gamma_band']
# print(df.head())
df_filtered = df[(df['response']!=0) | # filter out non-response
                 (df['contrast_right'] !=0) | # filter out where contrast right is 0
                  (df['contrast_left']) !=0] # filter out where contrast left is 0
df_filtered = df_filtered[['brain_area', 'contrast_diff','exponent', 'offset',
       'theta_cf', 'theta_pow', 'theta_band', 'beta_cf', 'beta_pow',
       'beta_band', 'gamma_cf', 'gamma_pow', 'gamma_band']]

brain_regs = list(set(df["brain_area"].tolist()))

feature_columns = ['exponent', 'offset','theta_cf', 'theta_pow', 'theta_band', 'beta_cf', 'beta_pow','beta_band', 'gamma_cf', 'gamma_pow', 'gamma_band']

new_columns = []
for br in brain_regs:
    br_cols = [br + "_" + x for x in feature_columns]
    new_columns.extend(br_cols)

# print(new_columns)
df_new = pd.DataFrame(columns = new_columns, index=None)

levels = df_filtered.groupby(['brain_area', 'contrast_diff'])[features].mean()
print (levels.apply(list))
# for br in brain_regs:
#     data_br = df_filtered[df_filtered["brain_area"] == br]
#     data_
#     for fc in feature_columns:
#         mv = data_br[fc].mean()
#     print (data_br)
#     break

# print (df_new)
# print(df.head())
# levels = df_filtered.groupby(['brain_area', 'contrast_diff'])[features].mean()
# a = levels.aggregate(list).values
# print(a)
