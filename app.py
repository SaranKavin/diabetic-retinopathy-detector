from flask import Flask, render_template, request
import os
import glob
import json
from inference import DRModel, is_fundus_image

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Initialize DR model
model = DRModel("models_weights/dr_model.pth")

# Load metrics if available
metrics_file = "models_weights/metrics.json"
metrics = None
if os.path.exists(metrics_file):
    with open(metrics_file, "r") as f:
        metrics = json.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        # Clear old uploaded images
        files = glob.glob(os.path.join(app.config["UPLOAD_FOLDER"], "*"))
        for f in files:
            os.remove(f)
        return render_template("index.html", metrics=metrics)

    if request.method == "POST":
        # Get patient details
        patient = {
            "patient_id": request.form.get("patient_id"),
            "name": request.form.get("patient_name"),
            "age": request.form.get("age"),
            "gender": request.form.get("gender"),
            "contact": request.form.get("contact"),
            "email": request.form.get("email"),
            "address": request.form.get("address"),
        }

        # Handle image upload
        if "file" not in request.files or request.files["file"].filename == "":
            return render_template("index.html", error="❌ No file uploaded", metrics=metrics)

        file = request.files["file"]
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        # ✅ Validate retina image before prediction
        if not is_fundus_image(filepath):
            return render_template(
                "index.html",
                patient=patient,
                error="❌ Invalid image. Please upload a retina fundus image.",
                metrics=metrics
            )

        # Prediction
        prediction, heatmap_path, confidence = model.predict(filepath)

        return render_template(
            "index.html",
            patient=patient,
            prediction=prediction,
            confidence=round(confidence, 2),
            img_path=filepath,
            heatmap_path=heatmap_path,
            metrics=metrics
        )

if __name__ == "__main__":
    app.run(debug=True)
