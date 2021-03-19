# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 17:28:54 2021

@author: A.Manandhar

Cluster INFORM indices for a particular year 2020
Kmeans

"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cluster INFORM indices for a particular year 2020

# Only using the 6 pillar indices for the latest year 2020
indicators = ['Natural Hazard','Human Hazard','Socio-Economic Vulnerability','Vulnerable Groups','Institutional','Infrastructure']
pathData = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\INFORM'
df = pd.read_excel(os.path.join(pathData,'raw','INFORM2020_TREND_2010_2019_v040_ALL_2.xlsx'))
df = df.drop(['IndicatorId', 'SurveyYear', 'IndicatorType', 'INFORMYear'],axis=1) 
df = df.loc[df['INFORMYear']==2020,:]   
idxKeep = [np.any(indName in indicators) for indName in df['IndicatorName']]
df = df.loc[idxKeep,:]
df = df.set_index(['Iso3','IndicatorName'])
df = df.unstack()
# Save processed INFORM 6 key indicators
#df.to_csv(r'...\INFORM2020_2010_2019_v040_2_ISO3_6pillars.csv')
# Read processed INFORM 6 key indicators
df = pd.read_csv(r'...\INFORM2020_2010_2019_v040_2_ISO3_6pillars.csv',index_col='Iso3')

# Standardize data
from sklearn.preprocessing import StandardScaler
zmuv = StandardScaler()
zmuv = zmuv.fit(df.values)
X = zmuv.fit_transform(df.values)

# PCA to generate lower dimensional representation to aid visualization
# Could also cluster these PCA components instead of raw values
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
# pca = PCA(n_components=2)
pca = pca.fit(X)
Xpc = pca.fit_transform(X)

fig,ax = plt.subplots(figsize=(6,5))
ax.stem(pca.explained_variance_)
ax.set_title('PCA explained variance',fontsize='x-large')
ax.set_xlabel('Principal Components',fontsize='x-large')
ax.set_ylabel('Explained Variance (%)',fontsize='x-large')
plt.show()

# 2D plot
fig,ax = plt.subplots(figsize=(6,5))
ax.plot(Xpc[:,0],Xpc[:,1],'.')
ax.set_title('PCA representation',fontsize='x-large')
ax.set_xlabel('1st Principal Component',fontsize='x-large')
ax.set_ylabel('2nd Principal Component',fontsize='x-large')
plt.show()

# 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Xpc[:,0],Xpc[:,1],Xpc[:,2], '.')
plt.show()

# Cluster using KMeans
# Could use other techniques. Please see https://scikit-learn.org/stable/modules/clustering.html
from sklearn.cluster import KMeans
Nclusters = 5
kmeans = KMeans(n_clusters=Nclusters)
kmeans = kmeans.fit(X)
print(kmeans.labels_)
df['Cluster'] = kmeans.labels_
# Save cluster labels
# df.to_csv(r'...\INFORM2020_2010_2019_v040_2_ISO3_6pillars_KmeansK5.csv')

# 2d plot
fig,ax = plt.subplots(figsize=(6,5))
for nClus in np.arange(Nclusters):
    ax.plot(Xpc[kmeans.labels_==nClus,0],Xpc[kmeans.labels_==nClus,1],'.',markersize=10)
    print('Cluster %d\n\t'%nClus)
    print(df.index.values[kmeans.labels_==nClus])
ax.set_title('Kmeans Clusters (N clusters = 5)',fontsize='x-large')
ax.set_xlabel('1st Principal Component',fontsize='x-large')
ax.set_ylabel('2nd Principal Component',fontsize='x-large')
ax.legend(['Cluster 0','Cluster 1','Cluster 2','Cluster 3','Cluster 4'])

# 3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for nClus in np.arange(Nclusters):
    ax.scatter(Xpc[kmeans.labels_==nClus,0],Xpc[kmeans.labels_==nClus,1],Xpc[kmeans.labels_==nClus,2], '.')    
    print('Cluster %d\n\t'%nClus)
    print(df.index.values[kmeans.labels_==nClus])
ax.set_title('Kmeans Clusters (N clusters = 5)',fontsize='x-large')
ax.set_xlabel('1st PC',fontsize='x-large')
ax.set_ylabel('2nd PC',fontsize='x-large')
ax.set_zlabel('3rd PC',fontsize='x-large')
ax.legend(['C0','C1','C2','C3','C4'])
fig.tight_layout()


