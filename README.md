Automated Resume Relevance Check System
Project Overview
This is a Streamlit web application designed to help recruiters, hiring managers, or job seekers evaluate the relevance of a resume to a specific job description. The system analyzes both a resume and a job description (JD) and provides a relevance score based on a combination of keyword matching and semantic similarity.

Features
File Upload: Supports uploading resumes and JDs in PDF, DOCX, or TXT formats.

Text Input: Option to paste JD text directly into the application.

Skill Extraction: Automatically extracts "must-have" and "good-to-have" skills from the JD. These lists are editable.

Relevance Scoring: Calculates a final score based on two components:

Hard Match Score: A weighted score based on the percentage of "must-have" and "good-to-have" skills found in the resume.

Soft Match Score: A contextual score based on the semantic similarity of the entire resume and JD texts, using either TF-IDF or Sentence-Transformers embeddings.

Detailed Results: Provides a breakdown of matched and missing skills, a final relevance verdict (High, Medium, Low), and suggestions for improvement.

Dashboard: Stores and displays a history of all evaluations in a local SQLite database, allowing for easy tracking and filtering of results.

Getting Started
Prerequisites
To run this application locally, you will need Python 3.7 or 
Install the required Python packages. It is recommended to use a virtual environment.

Bash

pip install -r requirements.txt
If sentence-transformers isn't available or you encounter issues, the app will automatically fall back to TF-IDF for soft matching.

Running the App
After installation, run the application from your terminal:

Bash

streamlit run app.py
The application will open in your default web browser.

Deployment
This application can be easily deployed to the Streamlit Community Cloud. For successful deployment, ensure your repository contains the following files:

app.py (your main application code)

requirements.txt (listing all dependencies)

Any sample files or data the app needs (.pdf, .docx, etc.).

How It Works
File Processing: The app uses pdfplumber and docx2txt to extract raw text from uploaded documents.

Skill Matching: The rapidfuzz library is used for fuzzy string matching to find skills, making the matching process robust to minor spelling differences.

Semantic Similarity:

TF-IDF: A classic machine learning technique that measures how important a word is to a document in a collection.

Sentence-Transformers: A more advanced deep learning model that converts sentences into numerical vectors to find semantic similarity. The app defaults to this method if the library is available, as it generally provides more accurate soft matching.

Database: sqlite3 is used to create and manage a simple local database to persist evaluation results.

Acknowledgements
Streamlit: For providing an incredible framework for building and sharing data apps.

Hugging Face: For the sentence-transformers models.

License
This project is licensed under the MIT License - see the LICENSE file for details.












T

