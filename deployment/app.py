import numpy as np
from flask import Flask, request , jsonify , render_template
import pickle

# create flask app
app = Flask(__name__)

# loading the pickle model
model = pickle.load(open("model.pkl", "rb"))

# define homepage
@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    return render_template("index.html", prediction_text = "RESULT(if 1 then clicked else 0 then not) {}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)