import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
df = pd.read_csv('NMA_features.csv', index_col=None)
df = df.dropna()
pca = PCA(n_components=2)

# pca.fit(df.values)
# pca.fit_transform(df.values)
features = ['response', 'response_time', 'exponent', 'offset',
       'theta_cf', 'theta_pow', 'theta_band', 'beta_cf', 'beta_pow',
       'beta_band', 'gamma_cf', 'gamma_pow', 'gamma_band']
x = df.loc[:, features].values
y = df.loc[:,['contrast_diff']].values
x = StandardScaler().fit_transform(x)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['pca1', 'pca2'])

finalDf = pd.concat([principalDf, df[['response']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = ['gamma_pow', 'theta_pow', 'beta_pow']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['response'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'pca1'], finalDf.loc[indicesToKeep, 'pca2'], c = color, s = 50)
ax.legend(targets)
# ax.grid()                                     ^
plt.show()
