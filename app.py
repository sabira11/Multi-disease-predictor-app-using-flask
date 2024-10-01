from flask import Flask,redirect,url_for,render_template,request,flash
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
hd_model=pickle.load(open('model.pkl','rb'))
dia_model=pickle.load(open('diabetics_model.pkl','rb'))
app=Flask(__name__)

@app.route('/')
def welcome():
    return render_template('home.html')


@app.route('/heart_disease')
def heart_disease():
    return render_template('heart_disease.html')
@app.route('/diabetics')
def diabetics():
    return render_template('diabetics_predict.html')
@app.route('/predict_heart_disease',methods=['POST'])
def predict_heart_disease():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    scaler = StandardScaler()
    final_features=scaler.fit_transform(final_features)

    prediction = hd_model.predict(final_features)

    output = round(prediction[0], 2)
    
    res=""
    if output==1:
        res="Presence"
    else:
        res="Absence"

    return render_template('heart_disease.html', prediction_text='Heart disease {}'.format(res))
@app.route('/predict_diabetics',methods=['POST'])
def predict_diabetics():
    try:
        # Initialize a list to hold the form values
        int_features = []
        for value in request.form.values():
            if value == '':  # If the field is empty, assign a default value (e.g., 0)
                int_features.append(0)  # Default value for untouched fields
            else:
                int_features.append(float(value))  # Convert to float if value is present
        
        # Convert list to numpy array and reshape for model input
        final_features = [np.array(int_features)]
        scaler = StandardScaler()
        final_features=scaler.fit_transform(final_features)
        prediction = dia_model.predict(final_features)
        
        output = round(prediction[0], 2)
        res = "might be Presence" if output == 1 else "might be Absence"
        
        return render_template('diabetics_predict.html', prediction_text=f'Diabetics Prediction test: {res}')
    
    except ValueError:
        flash("Invalid input. Please enter valid numeric values.")
        return render_template('diabetics_predict.html')
    
if __name__=='__main__':
    app.run(debug=True)



