#import necessary libraies
from flask import Flask, render_template, request

import numpy as np
import os

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

#load model
model =load_model("skin_disease_model5.h5")
print("@@ Model loaded")

def pred_skin_disease(skin_image):
    test_image = load_img(skin_image, target_size=(150,150))#load image
    print("@@ Got Image for prediction")

    test_image = img_to_array(test_image)/255   #convert image to np array
    test_image = np.expand_dims(test_image, axis = 0)#change 3d to 4d
    
    result = model.predict(test_image).round(3)  #predict disease or not
    print("@@ Row result = ", result)
    
    pred = np.argmax(result)  #get index in max value

    if pred == 0:
        return "Acne", 'Acne_Level_0.html'
    elif pred == 1:
        return "blister", 'blister.html'
    elif pred == 2:
        return "cold sore", 'cold sore.html'
    elif pred == 3:
        return "eczema", 'eczema.html'
    elif pred == 4:
        return "hives", 'hives.html'
    elif pred == 5:
        return "rosacea", 'rosacea.html'
    else:
        return "healthy plant", "healthy_body.html"
    
#------------>>>pred_disease<<---end

#create flask instance

app = Flask(__name__)

#render index file
@app.route("/", methods=['GET','POST'])
def home():
    return render_template('index.html')


@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files["image"]
        filename = file.filename
        print("@@ Input posted = ", filename)
        
        file_path = os.path.join("static/user uploaded", filename)
        file.save(file_path)
        
        print("@@ Predicting class.......")
        pred, output_page = pred_skin_disease(skin_image=file_path)
        
        return render_template(output_page, pred_output = pred, user_image = file_path)

if __name__ == "__main__":
    app.run()
    






















    