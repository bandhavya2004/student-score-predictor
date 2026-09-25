# Student Score Predictor

A Flask web app that predicts a student's final exam score based on study hours, attendance, and previous exam performance, using a linear regression model trained from scratch with NumPy.

## Features
- Predicts final exam score from 3 inputs: study hours, attendance %, previous score
- Linear regression model implemented from scratch using NumPy's Normal Equation (no scikit-learn dependency — built this way after hitting a Windows security restriction, which led to a deeper understanding of how the algorithm works under the hood)
- Model trained and evaluated with Mean Absolute Error and R² metrics
- Clean, responsive UI

## Tech Stack
- **Backend:** Python, Flask
- **Data/ML:** pandas, NumPy (custom Linear Regression implementation)
- **Frontend:** HTML, CSS, Jinja2 templates

## Running Locally

\\\ash
git clone https://github.com/bandhavya2004/student-score-predictor.git
cd student-score-predictor
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
python app.py
\\\

Open http://127.0.0.1:5000 in your browser.

## Live Demo
https://student-score-predictor-5x16.onrender.com
