import fastcluster
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

def cluster(data, method = "ward"):
    """
    wrapper for fastcluster clustering
    Arg:
        data: (N*D) np.ndarray, the data
        method: the cluster method to be used
    return:
        np.ndarray in scipy linkage format
    """
    assert method in ["single", "complete", "average", "weighted", "ward", "centroid", "median"]
    return fastcluster.linkage(data, method=method, metric='euclidean', preserve_input=True)

def visualize_cluster(linkage):
    """
    wrapper for dendrogram visualization, this function is very slow
    args:
        linkage: (N*4) np.ndarray, the result of fastcluster.linkage or scipy hierarchy result
    """
    plt.figure()
    df = hierarchy.dendrogram(linkage)
    plt.show()
