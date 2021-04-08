# cluster-INFORM-indicators
Clusters countries based on six INFORM indicators related to hazard, vulnerability, and coping capacity.

Dirichlet Process Gaussian Mixture Model (DPGMM)
- https://scikit-learn.org/stable/modules/mixture.html
- https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html#sklearn.mixture.BayesianGaussianMixture

INFORM risk index
- https://drmkc.jrc.ec.europa.eu/inform-index

scripts
- informIndicatorsClusterKmeansScript.py shows an example of clustering using Kmeans
- informIndicatorsClusterDpgmmScript.py shows an example of clustering using Dirichlet Process Gaussian Mixture Model
- mergeInformClusterCntyLabelsCntyShapefileScript.py shows an example of merging learned cluster labels with country shapefile data

Outputs:

- Sample outputs for two clustering techniques
- Learned models, plots, and corresponding spreadsheets

Data:
- All data used in this work are publicly available from corresponding original sources
- Processed data will be made available via HDX https://data.humdata.org/, whose link will be updated here soon.

Please also refer to the methodoogies in the scripts.
