from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from PyPDF2 import PdfReader
from utils import clean_text

app = Flask(__name__)
CORS(app)

model = joblib.load("resume_classifier.pkl")

@app.route("/upload-resume", methods=["POST"])
def upload_resume():
    file = request.files["resume"]
    if file.filename.endswith(".pdf"):
        text = ''
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
        cleaned = clean_text(text)
        pred = model.predict([cleaned])[0]
        return jsonify({"role": pred})
    return jsonify({"error": "Invalid file format"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
