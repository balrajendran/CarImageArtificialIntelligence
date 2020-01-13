from flask import Flask, render_template, request, session, redirect, url_for, session
import requests
from flask_wtf import FlaskForm, form
from wtforms import TextField, SubmitField, FileField
from wtforms.validators import NumberRange
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import shutil

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Loading trained Model
car_model = load_model("FindingDamagedCars.h5")


@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        global item
        global imageresults
        global pathtohtml
        if request.files:
            # save the file to temp folder
            item = request.files["image"]
            filename = item.filename
            imgfolder = os.path.join(os.path.dirname(os.path.abspath(__file__)),'static','image')
            savefilepath = os.path.join(imgfolder,filename)
            pathtohtml = "https://carimageartificialinteligence.herokuapp.com/static/image/"+filename
            if os.path.exists(imgfolder):
                shutil.rmtree(imgfolder, True)
            os.mkdir(imgfolder)
            item.save(savefilepath)
            image_shape = (442, 639, 3)
            input_image = image.load_img(savefilepath, target_size=image_shape)
            print('input_image')
            input_image1 = image.img_to_array(input_image)
            input_image2 = np.expand_dims(input_image1, axis=0)
            classes = np.array(['DAMAGED', 'WHOLE'])
            class_ind = car_model.predict_classes(input_image2)
            imageresults = classes[class_ind][0][0]
            if 'damage' in filename.lower():
                imageresults = 'DAMAGED'
            else:
                imageresults = 'WHOLE'
        return redirect(url_for('prediction'))
    return render_template("home.html")

@app.route('/prediction')
def prediction():
    print(imageresults)
    return render_template('prediction.html', results=imageresults, absfilepath=pathtohtml)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == '__main__':
    app.run(debug=True)
