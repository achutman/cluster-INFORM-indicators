# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 15:38:07 2021

@author: Achut Manandhar, Data Scientist, Migration and Displacement Initiative, Save the Children International
Contact: William Low, Project Lead, Migration and Displacement Initiative, Save the Children International

Plot DPGMM clusters as shape files
Better documented code @ https://github.com/achutman/predictive-displacement-cluster-INFORM-indicators

Plot countries and corresping cluster labels in GIS

# Methodolodgy/Algorithm/Pseudocode
mergeInformClusterCntyLabelsCntyShapefileScript.py
1. Libraries, parameters, settings
1.1 Import necessary libraries
2. Load data, generate and process data
2.1 Load shapefile for World Cuuntry boundaries
2.2 Load ISO2 to ISO3 matching spreadsheet
2.3 Load saved cluster labels
2.4 INFORM data processing - Merge ISO2 with ISO3 per country
2.5 Merge cluster label with shape file info per country
3. Save/plot outputs
3.1 Save updated shape file (to be plotted in QGIS or ArcMap)

"""
# 1. Libraries, parameters, settings
# 1.1 Import necessary libraries
import os
import pandas as pd
import numpy as np
import geopandas as gpd

# 2. Load data, generate and process data
# 2.1 Load shapefile for World Cuuntry boundaries
pathName = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\QGIS'
dfWorldGeo = gpd.read_file(os.path.join(pathName,'UIA_World_Countries_Boundaries-shp','World_Countries__Generalized_.shp'))
print(dfWorldGeo.keys())

# 2.2 Load ISO2 to ISO3 matching spreadsheet
dfIso23 = pd.read_csv(r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\UN DESA\Population\processed\IBAN_Countries_Iso2_Iso3.csv',engine='python')

# 2.3 Load saved cluster labels
# Kmeans
# dfCluster = pd.read_csv(r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\INFORM\processed\INFORM2020_2010_2019_v040_2_ISO3_6pillars_KmeansK5.csv')
# DPGMM
dfCluster = pd.read_csv(r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\INFORM\processed\INFORM2020_2010_2019_v040_2_ISO3_6pillars_DPGMMK10.csv')

# 2.4 INFORM data processing - Merge ISO2 with ISO3 per country
dfCluster = dfCluster.merge(dfIso23,left_on='Iso3',right_on='ISO3',how='left')
dfCluster = dfCluster.drop(['Iso3', 'Human Hazard', 'Infrastructure', 'Institutional','Natural Hazard', 'Socio-Economic Vulnerability', 'Vulnerable Groups','Country','Numeric'],axis=1)

# 2.5 Merge cluster label with shape file info per country
dfWorldGeo = dfWorldGeo.merge(dfCluster,left_on='ISO',right_on='ISO2',how='left').drop(['ISO2'],axis=1)
print(dfWorldGeo.keys())
print(dfWorldGeo.shape)
dfWorldGeo = dfWorldGeo.dropna(axis=0)
print(dfWorldGeo.shape)

# 3. Save/plot outputs
# 3.1 Save updated shape file (to be plotted in QGIS or ArcMap)
# dfWorldGeo.to_file(os.path.join(pathName,'UIA_World_Countries_Boundaries_Inform_Cluster','UIA_World_Boundaries_Kmeans.shp'))
# dfWorldGeo.to_file(os.path.join(pathName,'UIA_World_Countries_Boundaries_Inform_Cluster','UIA_World_Boundaries_DPGMM.shp'))