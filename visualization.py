import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_clusters(tfidf_matrix, cluster_labels):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(tfidf_matrix.toarray())

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis', marker='o', edgecolor='k')
    plt.title('Resume Clusters (PCA)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(scatter, label='Cluster')
    plt.show()
