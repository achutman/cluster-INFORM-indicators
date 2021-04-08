# -*- coding: utf-8 -*-
"""
Created on Thu 08 April 2021 

@author: Achut Manandhar, Data Scientist, Migration and Displacement Initiative, Save the Children International
Contact: William Low, Project Lead, Migration and Displacement Initiative, Save the Children International

Cluster INFORM indices for a particular year 2020
using Dirichlet Process Gaussian Mixture Model (DPGMM)
https://scikit-learn.org/stable/modules/mixture.html
https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html#sklearn.mixture.BayesianGaussianMixture
INFORM risk index
https://drmkc.jrc.ec.europa.eu/inform-index
Better documented code @ https://github.com/achutman/predictive-displacement-cluster-INFORM-indicators

# Methodolodgy/Algorithm/Pseudocode
informIndicatorsClusterDpgmmScript.py
1. Libraries, parameters, settings
1.1 Import necessary libraries
1.2 Define all parameters and choose various settings
2. Load data, generate and process labels and features
2.1 Read INFORM risk index
2.2 Only using the 6 pillar indices for the latest year 2020
2.3 Standardize data using all available data from all countries
2.4 PCA to generate lower dimensional representation to aid visualization
3. Train/test
3.1 Train model ***Key model training step***
3.2 Predict labels for test data ***Key model testing step***
4. Save/plot outputs
4.1 Save data and forecasts      
4.2 Plot outputs

"""
# 1. Libraries, parameters, settings
# 1.1 Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import KMeans

# 1.2 Define all parameters and choose various settings
INFORMyear = 2020
PCA_ncomp = 3
cluster_NCOMP = 20
# cluster_NINIT = 1000
cluster_NINIT = 100

# 2. Load data, generate and process labels and features
# 2.1 Read INFORM risk index
# Cluster INFORM indices for a particular year 2020
# 2.2 Only using the 6 pillar indices for the latest year 2020
indicators = ['Natural Hazard','Human Hazard','Socio-Economic Vulnerability','Vulnerable Groups','Institutional','Infrastructure']
pathData = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\INFORM'
df = pd.read_excel(os.path.join(pathData,'raw','INFORM2020_TREND_2010_2019_v040_ALL_2.xlsx'))
df = df.drop(['IndicatorId', 'SurveyYear', 'IndicatorType', 'INFORMYear'],axis=1) 
df = df.loc[df['INFORMYear']==INFORMyear,:]   
idxKeep = [np.any(indName in indicators) for indName in df['IndicatorName']]
df = df.loc[idxKeep,:]
df = df.set_index(['Iso3','IndicatorName'])
df = df.unstack()
# Save processed INFORM 6 key indicators
#df.to_csv(r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\INFORM\processed\INFORM2020_2010_2019_v040_2_ISO3_6pillars.csv')
# Read processed INFORM 6 key indicators
df = pd.read_csv(r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\INFORM\processed\INFORM2020_2010_2019_v040_2_ISO3_6pillars.csv',index_col='Iso3')

# 2.3 Standardize data using all available data from all countries
# Note - there is no need to separate training/testing because we are only interested in learning clusters based on all available data from all countries
zmuv = StandardScaler()
zmuv = zmuv.fit(df.values)
X = zmuv.fit_transform(df.values)

# 2.4 PCA to generate lower dimensional representation to aid visualization
# Could also cluster these PCA components instead of raw values
pca = PCA(n_components=PCA_ncomp)
# pca = PCA(n_components=2)
pca = pca.fit(X)
Xpc = pca.fit_transform(X)

# Plot explained variance
fig,ax = plt.subplots(figsize=(6,5))
ax.stem(pca.explained_variance_)
ax.set_title('PCA explained variance',fontsize='x-large')
ax.set_xlabel('Principal Components',fontsize='x-large')
ax.set_ylabel('Explained Variance (%)',fontsize='x-large')
plt.show()

# 2D PCA plot
fig,ax = plt.subplots(figsize=(6,5))
ax.plot(Xpc[:,0],Xpc[:,1],'.')
ax.set_title('PCA representation',fontsize='x-large')
ax.set_xlabel('1st Principal Component',fontsize='x-large')
ax.set_ylabel('2nd Principal Component',fontsize='x-large')
plt.show()

# 3D PCA plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Xpc[:,0],Xpc[:,1],Xpc[:,2], '.')
plt.show()

# 3. Train/test
# Note - both training and testing dataset include all available data
# 3.1 Train model ***Key model training step***
# Cluster using Bayesian Mixture Model
bgm = BayesianGaussianMixture(n_components=cluster_NCOMP,n_init=cluster_NINIT).fit(X)
print(bgm.weights_)
print(bgm.means_)
# 3.2 Predict labels for test data ***Key model testing step***
cluster_labels = bgm.predict(X)
df['Cluster'] = bgm.predict(X)

# # ***Reference*** - compare with Kmeans clustering (if you feel like it)
# # Cluster using Kmeans
# 3.1 Train model
# kmeans = KMeans(n_clusters=cluster_NCOMP,n_init=cluster_NINIT).fit(X)
# print(kmeans.labels_)
# 3.2 Predict labels for test data
# cluster_labels = kmeans.labels_
# df['Cluster'] = kmeans.labels_
# # Save cluster labels
# # df.to_csv(r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\INFORM\processed\INFORM2020_2010_2019_v040_2_ISO3_6pillars_KmeansK20_I100.csv')


# 4. Save/plot outputs
# 4.1 Save learned model and cluster labels
# Save cluster labels
# df.to_csv(r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\INFORM\processed\INFORM2020_2010_2019_v040_2_ISO3_6pillars_DPGMMK10.csv')
# Save learned DPGMM model
# fileSave = os.path.join(pathSave,'INFORM2020_2010_2019_v040_2_ISO3_6pillars_Pc3DpgmmK20I1000.sav')
# pickle.dump(bgm, open(fileSave, 'wb'))

# 4.2 Plot outputs
# Plot and save cluster probabilities
fig,ax=plt.subplots()
ax.stem(np.arange(cluster_NCOMP),bgm.weights_)
ax.set_xticks(np.arange(cluster_NCOMP))
ax.set_xlabel('Clusters',fontsize='x-large')
ax.set_ylabel('Cluster Probabilities',fontsize='x-large')
fig.tight_layout()
# fig.savefig(os.path.join(pathSave,'clusterPis.png'),dpi=300)
# fig.savefig(os.path.join(pathSave,'clusterPis_lowres.png'),dpi=100)

print(len(np.unique(bgm.predict(X))))
clusterProb = bgm.predict_proba(X)

# # Random plots to understand/debug
# plt.stem(clusterProb[:,0]),plt.show()
# plt.stem(clusterProb[:,2]),plt.show()
# plt.stem(clusterProb[:,4]),plt.show()

# 3d plot of learned cluster labels
markers = ['o','v','^','1','2','P','*','+','x','|','_']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for nClus in np.arange(bgm.n_components):
    ax.scatter(Xpc[cluster_labels==nClus,0],Xpc[cluster_labels==nClus,1],Xpc[cluster_labels==nClus,2], '.')    
    print('Cluster %d\n\t'%nClus)
    print(df.loc[cluster_labels==nClus,'Iso3'].values)
ax.set_title('DPGMM Clusters (N clusters = %d)'%cluster_NCOMP,fontsize='x-large')
ax.set_xlabel('1st PC',fontsize='x-large')
ax.set_ylabel('2nd PC',fontsize='x-large')
ax.set_zlabel('3rd PC',fontsize='x-large')
ax.legend(['C0','C1','C2','...'])
fig.tight_layout()
