from flask import Flask, request, jsonify, render_template
from wtforms import Form, SelectField, DecimalField, IntegerField, validators, BooleanField, RadioField
import pickle
import numpy as np

# Load the model
with open('data/pipe_xgb.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)


class InfoInput(Form):
    age = IntegerField('',
                      [validators.DataRequired(),
                      validators.NumberRange(min=0, max=1000)])
    hypertension = RadioField('', choices=[(0, 'Yes'),(1, 'No')])
    heart_disease = RadioField('', choices=[(0, 'Yes'),(1, 'No')])
    bmi = DecimalField('',
                            [validators.DataRequired(),
                            validators.NumberRange(min=0, max=50)],
                            places=2)
    hemoglobin = DecimalField('',
                            [validators.DataRequired(),
                            validators.NumberRange(min=0, max=50)],
                            places=2)
    blood_glucose = IntegerField('',
                                [validators.DataRequired(),
                                validators.NumberRange(min=0, max=1000)])
    gender = SelectField(u'--선택해주세요',
                        choices=[(1, '남자'), (0, '여자')])
    smoking = SelectField('', choices=[(0, 'Current'), (1, 'Ever'), (2, 'Former'), (3, 'Never'), (4, 'Not Current') ])


def map_smoking_values(num):
  list = [0, 0, 0, 0, 0]
  list[num] = 1
  return list

@app.route('/')
def home():
    form = InfoInput(request.form)
    return render_template('home.html', form=form)

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    bmi = float(request.form['bmi'])
    hemoglobin = float(request.form['hemoglobin'])
    blood_glucose = int(request.form['blood_glucose'])
    gender = int(request.form['gender'])
    smoking = map_smoking_values(int(request.form['smoking']))
    
    
    features = np.array([age, hypertension, heart_disease, bmi, hemoglobin, blood_glucose, gender, *smoking]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)
    
    # Return the result
    # return jsonify({'prediction': int(prediction[0])})
    # return render_template('results.html', results=int(prediction[0]))
    return render_template('results.html', results=int(prediction[0]))
if __name__ == '__main__':
    app.run(debug=True)
