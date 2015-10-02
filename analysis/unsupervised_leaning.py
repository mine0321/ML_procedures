# coding:utf-8


def k_means(samples, n_cluster, axis=None):
    from sklearn.cluster.dtw_k_means_ import KMeans

    if not axis == None:
        samples = map(lambda data: data[axis], samples)

    kmeans_model = KMeans(n_clusters=n_cluster, random_state=1).fit(samples)
    return kmeans_model

