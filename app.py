from flask import Flask,redirect,url_for,render_template,request,flash
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
#from keras.models import load_model 
#from keras.layers import Dense
#from tensorflow.python.keras.models import load_model

import h5py
#from keras.models import load_model
import cv2
model=tf.keras.models.load_model("skin_model.keras")

hd_model=pickle.load(open('model.pkl','rb'))
dia_model=pickle.load(open('diabetics_model.pkl','rb'))
classes=['Atopic Dermatitis','Basal Cell Carcinoma','Benign Keratosis-like Lesions','Eczema','Melanocytic Nevi','Melanoma']
img_size=(224,224)
def processImg(IMG_PATH):
    
    
    # Preprocess image
        img = cv2.imread(IMG_PATH)
        b, g, r = cv2.split(img)

    # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(5, 5))

    # Apply CLAHE to each channel
        r = clahe.apply(r.astype(np.uint8))
        g = clahe.apply(g.astype(np.uint8))
        b = clahe.apply(b.astype(np.uint8))

    # Merge the enhanced channels back
        enhanced_img = cv2.merge((b, g, r))
        
        # Resize the image to the desired size (e.g., 256x256)
        resized_image = cv2.resize(enhanced_img, img_size)
       
        #resized_image = cv2.resize(resized_image, img_size)
        resized_image = resized_image.astype('float32') / 255.0
        resized_image = np.expand_dims(resized_image, axis=0) 
        print(resized_image.shape)
        return resized_image
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
@app.route('/skin_disease')
def skin_disease():
    return render_template('skin_disease.html')
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

@app.route('/predict_skin_disease',methods=['POST'])
def predict_skin_disease():
    image_file = request.files["img"]
    image_path="./images/"+ image_file.filename
    image_file.save(image_path)
    #print(model.summary())
    Process_img = processImg(image_path)
    pred=model.predict(Process_img)
    predicted_class = np.argmax(pred)
    confidence_score = np.max(pred)
    confidence_score=round(confidence_score * 100, 2)
    predicted_label = classes[predicted_class]
    con_scores = [round(score * 100, 2) for score in pred[0]]
    prediction = list(zip(classes, con_scores))
   
    

    return render_template('skin_disease.html', prediction_t='Predicted disease : {}'.format(predicted_label),
                           confidence=confidence_score,prediction=prediction)
    
if __name__=='__main__':
    app.run(debug=True)



