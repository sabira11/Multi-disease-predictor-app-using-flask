from flask import Flask,redirect,url_for,render_template,request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
model=model=pickle.load(open('model.pkl','rb'))
app=Flask(__name__)

@app.route('/')
def welcome():
    return render_template('home.html')


@app.route('/heart_disease')
def heart_disease():
    return render_template('heart_disease.html')
@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    scaler = StandardScaler()
    final_features=scaler.fit_transform(final_features)

    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    
    res=""
    if output==1:
        res="Presence"
    else:
        res="Absence"

    return render_template('heart_disease.html', prediction_text='Heart disease {}'.format(res))
if __name__=='__main__':
    app.run(debug=True)



