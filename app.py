from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# load model
model = load_model("fruit_veg_model.h5")


# prediction function
def predict_img(img_path, model):

    img = image.load_img(img_path, target_size=(150,150))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    index = np.argmax(prediction)

    class_names = ["FRUIT 🍎", "VEGETABLE 🥦"]

    return class_names[index]


# home route
@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    image_url = None

    if request.method == "POST":
        file = request.files["image"]

        if file.filename != "":
            path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(path)

            result = predict_img(path, model)
            image_url = path

    return render_template("index.html", result=result, image=image_url)


if __name__ == "__main__":
    app.run(debug=True)