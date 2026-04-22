from ultralytics import YOLO
from flask import Flask, request, render_template
import os

# Load trained model
model = YOLO("best.pt")

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename != "":
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Run prediction
            results = model(filepath)

            for r in results:
                probs = r.probs
                prediction = model.names[probs.top1]
                confidence = float(probs.top1conf)

            return render_template(
                "index.html",
                prediction=prediction,
                confidence=round(confidence * 100, 2),
                image_path=filepath
            )

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
