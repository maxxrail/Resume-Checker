# Resume Parser with AI and Machine Learning

This project aims to build a resume parser using AI and machine learning techniques to automate the process of skimming through resumes and identifying the best candidates for job positions. The parser processes resumes, identifies key skills, clusters similar resumes, and assigns a score to each resume based on the identified skills.

## Project Structure

The project is organized into several Python scripts, each responsible for a specific part of the process:

1. **`main.py`**: The main script to run the entire process.
2. **`data_preprocessing.py`**: Functions related to data loading and preprocessing.
3. **`text_vectorization.py`**: Functions for text vectorization using TF-IDF.
4. **`clustering.py`**: Functions for clustering the resumes.
5. **`scoring.py`**: Functions to calculate and normalize scores.
6. **`visualization.py`**: Functions to visualize the clustering results.
7. **`skills_extraction.py`**: Functions to extract and display top skills per cluster.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/resume-parser.git
   cd resume-parser
