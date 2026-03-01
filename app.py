from flask import Flask, render_template, request
from predict import predict_disease
import os

app = Flask(__name__)

# Upload folder setup
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]
    if file.filename == "":
        return "No selected file"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Safe prediction
    result = predict_disease(filepath)

    return render_template("index.html", prediction=result, image_path=file.filename)

if __name__ == "__main__":
    app.run(debug=True)