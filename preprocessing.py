import pandas as pd
import numpy as np

from sta211.visu import histo_count, hist_cross_categories, cluster_dendogram, correlation
from sta211.datasets import load_train_dataset, get_quantitatives
from sta211.preprocessing import get_pca_components, get_kmeans_clusters
from sklearn.cluster import FeatureAgglomeration, AgglomerativeClustering
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y, variables = load_train_dataset(variables_family=None)
df = load_train_dataset(return_dataframe=True)


# duplicated = set()
# for c, r, n in summary_linked_variables(df, min_ratio=0.95):
#     duplicated.add(r)
# print("{} duplicated variables: {}".format(len(duplicated), ", ".join(duplicated)))

# Nom de dimensions utiles pour ACP
nb_components, percent = get_pca_components(X, y)
print("Components for PCA on all variables: {} ({:.2f}%)".format(nb_components, 100.0 * percent))


# Kmeans sur toutes les variables
df = get_kmeans_clusters(X, y)
hist_cross_categories(df, "class", "cluster", output_directory="output", suffix="all")

# Kmeans par groupe de variables
for f in ["fac", "fou", "kar", "mor", "pix", "zer"]:
    X_sub, y_sub = load_train_dataset(variables_family=f)
    df = get_kmeans_clusters(X_sub, y_sub)
    hist_cross_categories(df, "class", "cluster", output_directory="output", suffix=f)


# Classification de variables (Clusters)
n_clusters = 6
pipe = Pipeline([
    ("sc", StandardScaler()),
    ("fa", FeatureAgglomeration(distance_threshold=0, n_clusters=None, linkage="ward")),
])

# Classification de variables (Dendogramme)
pipe.fit(X)
cluster_dendogram(pipe._final_estimator, output_directory="output", p=4, truncate_mode="level", labels=list(get_quantitatives()))


for n_clusters in range(3, 9, 1):
    pipe.set_params(fa__n_clusters=n_clusters, fa__distance_threshold=None)
    pipe.fit(X)
    clusters = pipe._final_estimator.labels_

    df2 = pd.DataFrame({"cluster": clusters, "family": list(map(lambda v: v[0:3], variables))}, dtype=pd.api.types.CategoricalDtype(ordered=False))
    df2["variable"] = map(lambda o: np.nan, variables)

    histo_count(df2, first_col="cluster", second_col="family", n_clusters=n_clusters, output_directory="output")
