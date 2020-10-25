from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import fastcluster
from scipy.cluster import hierarchy
import shutil
from typing import List, Dict, Tuple

def PCA_preview(data, top_feature:int = None):
    """
    preview the relation of explained_variance_ratio_ v.s. n_components for PCA on the data to be loaded

    args:
        data: np.ndarray of (N*d) shape
    """
    if top_feature != None:
        n_components = np.min([top_feature, data.shape[1]])
    else:
        n_components = data.shape[1]
    pca = PCA(n_components = n_components)
    extracted = pca.fit_transform(data)
    plt.plot(np.arange(1,n_components+1),pca.explained_variance_ratio_.cumsum()[:n_components]);
    plt.title(f"Explained information from top {n_components+1} features");
    plt.show()

def clustering(data, method = "ward"):
    """
    wrapper for fastcluster clustering
    Arg:
        data: (N*D) np.ndarray, the data
        method: the cluster method to be used
    """
    assert method in ["single", "complete", "average", "weighted", "ward", "centroid", "median"]
    return fastcluster.linkage(data, method=method, metric='euclidean', preserve_input=True)

def visualize_clustering(linkage):
    """
    wrapper for dendrogram visualization, this function is kinda slow
    args:
        linkage: (N*4) np.ndarray, the result of fastcluster.linkage or scipy hierarchy result
    """
    plt.figure()
    df = hierarchy.dendrogram(linkage)
    plt.show()

def traverse_save(tree, save_path, path_list, name = "root", smallest_size = 5, print_mode = True):
    """
    A traverse_saving saving clusters in different level of similarity
    This save will be in DFS searching of the hierarchy tree

    Tree: scipy.cluster.hierarchy.ClusterNode, the root node of the heriachy
          tree from scipy.cluster.hierarchy.to_tree function
    path_list: list of str, the path of images
    save_path: str, the path to be saved
    name: str, the name of root folder to be saved, will not affect the child folders
    smallest_size: int, the smallest cluster size that will be saved as an individual folder
    print: Bool, decide if creating new folders will be noticed
    """
    # create a new path
    _current_path = os.path.join(save_path, name)
    os.makedirs(_current_path, exist_ok=True)
    # set node as the current node
    _current_node = tree
    #copy all the images in this cluster
    for i in _current_node.pre_order():
        shutil.copyfile(src = path_list[i], dst = os.path.join(_current_path,f"{i}.png"))
    try:
        _left_try = _current_node.get_left()
        _right_try = _current_node.get_right()
        # if it is a leaf, skip
        if isinstance(_left_try, hierarchy.ClusterNode):
            # if the number of images under this cluster is smaller than 5, skip(too small)
            if len(_left_try.pre_order()) >=5 :
                if print == True:
                    print(f"making {os.path.join(_current_path, 'left')} for {len(_left_try.pre_order())} images")
                traverse_save(tree = _left_try,
                              save_path = _current_path,
                              path_list = path_list,
                              name = "left",
                              smallest_size = smallest_size,
                              print_mode = print_mode
                              )
        # if it is a leaf, skip
        if isinstance(_right_try, hierarchy.ClusterNode):
            # if the number of images under this cluster is smaller than 5, skip(too small)
            if len(_right_try.pre_order())>=5:
                if print == True:
                    print(f"making {os.path.join(_current_path, 'right')} for {len(_right_try.pre_order())} images")
                traverse_save(tree = _right_try,
                              save_path = _current_path,
                              path_list = path_list,
                              name = "right",
                              smallest_size = smallest_size,
                              print_mode = print_mode
                              )
    except AttributeError:
      pass

def cluster_process(data,
                    path_list: List = [],
                    PCA__n_component: int = 5,
                    hierarchy__method: str = "ward",
                    save_path: str = "",
                    root_name:str = "root",
                    smallest_size: int = 5,
                    print_mode:bool = True):
    """
    the wrapper of clusering procedure, cluster and save the images in a clustering manner
    """
    assert hierarchy__method in ["single", "complete", "average", "weighted", "ward", "centroid", "median"]
    _pca = PCA(n_components = PCA__n_component)
    _data = pca.fit_transform(data)
    _linkage = clustering(data = _data, method = hierarchy__method)
    _tree = hierarchy.to_tree(_linkage)
    traverse_save(tree = _tree,
                  save_path = save_path,
                  path_list = path_list,
                  name = root_name,
                  smallest_size = smallest_size,
                  print_mode = print_mode)
