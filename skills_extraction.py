import numpy as np
import pandas as pd

def extract_top_skills(tfidf_matrix, resume_data, tfidf_vectorizer, kmeans, num_clusters):
    terms = tfidf_vectorizer.get_feature_names_out()
    top_terms_per_cluster = {}

    num_top_terms = 10
    for i in range(num_clusters):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        cluster_tfidf_sum = np.sum(tfidf_matrix[cluster_indices], axis=0)
        top_term_indices = np.argsort(cluster_tfidf_sum.A.flatten())[::-1][:num_top_terms]
        top_terms = [terms[index] for index in top_term_indices]
        top_terms_per_cluster[i] = top_terms

    top_skills_per_cluster = []
    for cluster, terms in top_terms_per_cluster.items():
        top_skills_per_cluster.append({"Cluster": cluster, "Top Skills": ", ".join(terms)})

    return pd.DataFrame(top_skills_per_cluster)
