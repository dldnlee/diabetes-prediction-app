from flask import Flask, request, jsonify, render_template
from wtforms import Form, SelectField, DecimalField, IntegerField, validators, BooleanField, RadioField
import pickle
import numpy as np

# Load the model
with open('data/lgbm_best.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)


class InfoInput(Form):
    age = IntegerField('',
                      [validators.DataRequired(),
                      validators.NumberRange(min=0, max=1000)])
    hypertension = RadioField('', [validators.DataRequired()], choices=[(0, 'Yes'),(1, 'No')])
    heart_disease = RadioField('', [validators.DataRequired()], choices=[(0, 'Yes'),(1, 'No')])
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

def result_text(classification, probability):

  if classification == 0 :
    proba = float(probability[0][0]) * 100
    return ('음성 확률: ' + str(round(proba, 2)) + '%')
  elif classification == 1:
    proba = float(probability[0][1]) * 100
    return ('양성 확률: ' + str(round(proba, 2)) + '%')
  else: return




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
    prediction_proba = model.predict_proba(features)

    text = result_text(int(prediction[0]), list(prediction_proba))

    return render_template('results.html', results=text)
if __name__ == '__main__':
    app.run(debug=True)
