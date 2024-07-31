from data_preprocessing import load_and_preprocess_data
from text_vectorization import vectorize_text
from clustering import cluster_resumes
from scoring import calculate_and_normalize_scores
from visualization import visualize_clusters
from skills_extraction import extract_top_skills

def main():
    # Step 1: Load and preprocess data
    resume_data = load_and_preprocess_data('UpdatedResumeDataSet.csv')
    print("Data loaded and preprocessed")

    # Step 2: Vectorize the preprocessed text data
    tfidf_matrix, tfidf_vectorizer = vectorize_text(resume_data['Processed_Resume'])
    print("Text data vectorized")

    # Step 3: Cluster the resumes
    num_clusters = 10
    resume_data['Cluster'], kmeans = cluster_resumes(tfidf_matrix, num_clusters)
    print("Resumes clustered")

    # Step 4: Calculate and normalize scores
    key_skills = ['python', 'machine learning', 'data analysis', 'deep learning', 'statistics', 'sql']
    resume_data = calculate_and_normalize_scores(resume_data, key_skills)
    print("Scores calculated and normalized")

    # Step 5: Visualize the clustering results
    visualize_clusters(tfidf_matrix, resume_data['Cluster'])
    print("Clusters visualized")

    # Step 6: Extract and display top skills per cluster
    top_skills_per_cluster = extract_top_skills(tfidf_matrix, resume_data, tfidf_vectorizer, kmeans, num_clusters)
    print("Top skills per cluster:")
    print(top_skills_per_cluster)

    # Convert top skills per cluster to a string format and map to the resume_data DataFrame
    skills_series = top_skills_per_cluster['Top Skills'].apply(lambda x: ", ".join(x))
    resume_data['Top_Skills'] = resume_data['Cluster'].map(skills_series.to_dict())

    # Sort resumes by score in descending order and add ranking
    resume_data.sort_values(by='Normalized_Score', ascending=False, inplace=True)
    resume_data.reset_index(drop=True, inplace=True)
    resume_data['Ranking'] = resume_data.index + 1

    # Save results to a CSV file in the repository
    resume_data.to_csv('resume_analysis_results.csv', index=False)
    print("Results saved to resume_analysis_results.csv")

if __name__ == "__main__":
    main()
