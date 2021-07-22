import pandas as pd
import numpy as np
import math
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
contrast_diffs = list(set(df["contrast_diff"].tolist()))
feature_columns = ['exponent', 'offset','theta_cf', 'theta_pow', 'theta_band', 'beta_cf', 'beta_pow','beta_band', 'gamma_cf', 'gamma_pow', 'gamma_band']

new_columns = ['contrast_diff']
for br in brain_regs:
    br_cols = [br + "_" + x for x in feature_columns]
    new_columns.extend(br_cols)

# print(new_columns)

# levels = df_filtered.groupby(['brain_area', 'contrast_diff'])[features].mean()
# print (levels.apply(list))
data_row_list = []
for contrast_diff in contrast_diffs:
    print(contrast_diff)
    data_row = [contrast_diff]
    for br in brain_regs:
        # print(br)
        df_temp = df_filtered[(df["brain_area"] == br) & (df["contrast_diff"] == contrast_diff)]
        # print (df_temp)
        br_cols = [br + "_" + x for x in feature_columns]
        for br_index, br_col in enumerate(br_cols):
            mean_value = 0
            mean_temp = df_temp[features[br_index]].mean()
            if not math.isnan(mean_temp):
                mean_value = mean_temp
            # print ( " mean value " + str(mean_value))
            data_row.append(float(mean_value))
    # print (len(data_row))
    # print (len(df_new.columns))
    data_row_list.append(data_row)

print (np.array(data_row_list).shape)

df_new = pd.DataFrame(data = np.array(data_row_list), columns = new_columns, index=None)

df_new.to_csv('out.csv')
