from ultralytics import YOLO
from flask import Flask, request, render_template
from PIL import Image
import io
import base64

# Load trained model
model = YOLO("best.pt")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename != "":
            # Read file into memory
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            # Run prediction directly
            results = model(image)

            for r in results:
                probs = r.probs
                prediction = model.names[probs.top1]
                confidence = float(probs.top1conf)

            return render_template(
                "index.html",
                prediction=prediction,
                confidence=round(confidence * 100, 2),
                image_data=image_base64  # optional if you want to display
            )

    return render_template("index.html", prediction=None)

app.run(debug=True)