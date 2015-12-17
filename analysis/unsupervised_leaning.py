# coding:utf-8


def dtw_k_means(samples, n_cluster, axis=None):
    """
    k_meansのクラスタ重心の距離計算にDTW距離を利用
    重心はユークリッド距離であるためデータ長は均一でなければならない

    samples : サンプルの挿入(if axis, pandas else numpy)
    n_cluster : クラスタ数(int)
    axis : sampleがpandas型の場合axisでculumns指定(string)
    """
    from sklearn.cluster.dtw_k_means_ import KMeans

    if axis is not None:
        samples = map(lambda data: data[axis], samples)

    kmeans_model = KMeans(n_clusters=n_cluster, random_state=1).fit(samples)
    return kmeans_model

def k_means(samples, n_cluster, axis=None):
    """
    sklearnのkmeans++
    距離はユークリッド限定

    samples : サンプルの挿入(if axis, pandas else numpy)
    n_cluster : クラスタ数(int)
    axis : sampleがpandas型の場合axisでculumns指定(string)
    """
    from sklearn.cluster import KMeans

    if axis is not None:
        samples = map(lambda data: data[axis], samples)

    kmeans_model = KMeans(n_clusters=n_cluster, random_state=1).fit(samples)
    return kmeans_model

def precomputed_AP(similarity,damping=0.5):

    """
    sklearnのAffinityPropagation
    類似度を事前計算時はこちらを使用
    類似度は値が大きいほど類似しているように設定

    similarity : 類似度の挿入．類似度が高いほど値を高くする(nunpy)
    damping : 減衰率(float)
    """
    from sklearn.cluster import AffinityPropagation
    AP_model = AffinityPropagation(affinity='precomputed',damping=damping).fit(similarity)
    return AP_model
