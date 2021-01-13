# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 15:38:07 2021

@author: A.Manandhar
"""
import os
import pandas as pd
import numpy as np

# Plot countries and corresp. cluster indices in GIS

# Load ISO2 to ISO3 matching spreadsheet
dfIso23 = pd.read_csv(r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\UN DESA\Population\processed\IBAN_Countries_Iso2_Iso3.csv',engine='python')

# Load saved cluster labels
# Kmeans
# dfCluster = pd.read_csv(r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\INFORM\processed\INFORM2020_2010_2019_v040_2_ISO3_6pillars_KmeansK5.csv')
# DPGMM
dfCluster = pd.read_csv(r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\INFORM\processed\INFORM2020_2010_2019_v040_2_ISO3_6pillars_DPGMMK10.csv')
# ISO2 to ISO3 matching
dfCluster = dfCluster.merge(dfIso23,left_on='Iso3',right_on='ISO3',how='left')
dfCluster = dfCluster.drop(['Iso3', 'Human Hazard', 'Infrastructure', 'Institutional','Natural Hazard', 'Socio-Economic Vulnerability', 'Vulnerable Groups','Country','Numeric'],axis=1)

# Load shapefile for World COuntry boundaries
import geopandas as gpd
pathName = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\QGIS'
dfWorldGeo = gpd.read_file(os.path.join(pathName,'UIA_World_Countries_Boundaries-shp','World_Countries__Generalized_.shp'))
print(dfWorldGeo.keys())
dfWorldGeo = dfWorldGeo.merge(dfCluster,left_on='ISO',right_on='ISO2',how='left').drop(['ISO2'],axis=1)
print(dfWorldGeo.keys())
print(dfWorldGeo.shape)
dfWorldGeo = dfWorldGeo.dropna(axis=0)
print(dfWorldGeo.shape)
# dfWorldGeo.to_file(os.path.join(pathName,'UIA_World_Countries_Boundaries_Inform_Cluster','UIA_World_Boundaries_Kmeans.shp'))
# dfWorldGeo.to_file(os.path.join(pathName,'UIA_World_Countries_Boundaries_Inform_Cluster','UIA_World_Boundaries_DPGMM.shp'))