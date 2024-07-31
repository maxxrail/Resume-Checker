from data_preprocessing import load_and_preprocess_data
from text_vectorization import vectorize_text
from clustering import cluster_resumes
from scoring import calculate_and_normalize_scores
from visualization import visualize_clusters
from skills_extraction import extract_top_skills

def main():
    # Step 1: Load and preprocess data
    resume_data = load_and_preprocess_data('/home/maxx/Documents/GitHub/Resume-Checker/UpdatedResumeDataSet.csv')
    
    # Step 2: Vectorize the preprocessed text data
    tfidf_matrix, tfidf_vectorizer = vectorize_text(resume_data['Processed_Resume'])
    
    # Step 3: Cluster the resumes
    num_clusters = 10
    resume_data['Cluster'], kmeans = cluster_resumes(tfidf_matrix, num_clusters)
    
    # Step 4: Calculate and normalize scores
    key_skills = ['python', 'machine learning', 'data analysis', 'deep learning', 'statistics', 'sql']
    resume_data = calculate_and_normalize_scores(resume_data, key_skills)
    
    # Step 5: Visualize the clustering results
    visualize_clusters(tfidf_matrix, resume_data['Cluster'])
    
    # Step 6: Extract and display top skills per cluster
    top_skills_per_cluster = extract_top_skills(tfidf_matrix, resume_data, tfidf_vectorizer, kmeans, num_clusters)
    print(top_skills_per_cluster)

if __name__ == "__main__":
    main()
