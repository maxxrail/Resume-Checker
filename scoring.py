def calculate_score(processed_resume, key_skills):
    words = processed_resume.split()
    score = sum(1 for word in words if word in key_skills)
    return score

def calculate_and_normalize_scores(resume_data, key_skills):
    resume_data['Score'] = resume_data['Processed_Resume'].apply(lambda x: calculate_score(x, key_skills))
    max_score = resume_data['Score'].max()
    resume_data['Normalized_Score'] = resume_data['Score'] / max_score * 10
    return resume_data
