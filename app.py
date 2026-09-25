from flask import Flask, render_template, request
import json

app = Flask(__name__)

# Load the trained model
with open('model.json') as f:
    model = json.load(f)

def predict(study_hours, attendance, previous_score):
    return (
        model['intercept']
        + model['study_hours_coef'] * study_hours
        + model['attendance_coef'] * attendance
        + model['previous_score_coef'] * previous_score
    )

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        study_hours = float(request.form.get('study_hours'))
        attendance = float(request.form.get('attendance'))
        previous_score = float(request.form.get('previous_score'))
        prediction = round(predict(study_hours, attendance, previous_score), 1)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
