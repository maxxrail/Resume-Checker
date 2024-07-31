from sklearn.cluster import KMeans

def cluster_resumes(tfidf_matrix, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(tfidf_matrix)
    return kmeans.labels_, kmeans
