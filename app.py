from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from PyPDF2 import PdfReader
from utils import clean_text

app = Flask(__name__)
CORS(app)

# Optional: Set max upload size to 5MB
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB

# Load model
model = joblib.load("resume_classifier.pkl")

@app.route("/upload-resume", methods=["POST"])
def upload_resume():
    if "resume" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["resume"]
    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Invalid file format"}), 400

    try:
        text = ''
        reader = PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            text += page_text if page_text else ''
        
        cleaned = clean_text(text)
        pred = model.predict([cleaned])[0]
        return jsonify({"role": pred})
    
    except Exception as e:
        return jsonify({"error": f"Failed to parse PDF: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
