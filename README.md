# Resume Parser with AI and Machine Learning

This project is a resume parser using AI and machine learning techniques to automate the process of skimming through resumes and identifying the best candidates for job positions. The parser processes resumes, identifies key skills, clusters similar resumes, and assigns a score to each resume based on the identified skills. The resumes are then ranked from best to worst based on the set keywords and saved into a csv.

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
   git clone https://github.com/maxxrail/resume-parser.git
   cd resume-parser

2. Install dependancies:
   ```sh
   pip install -r requirements.txt

## Running Program
1. Edit key skills in **`main.py`**

2. Use command to run program:
    ```sh
   python3 main.py
   
4.  Close PCA chart to see Top skills per cluster
