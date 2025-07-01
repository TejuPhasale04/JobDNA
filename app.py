from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from PyPDF2 import PdfReader
from utils import clean_text
import tempfile
import traceback

app = Flask(__name__)
CORS(app)

# Limit file upload size to 5MB
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

# Load the model once
model = joblib.load("resume_classifier.pkl")

@app.route("/upload-resume", methods=["POST"])
def upload_resume():
    try:
        # 1. Check file presence
        if "resume" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["resume"]

        # 2. Check file format
        if not file.filename.endswith(".pdf"):
            return jsonify({"error": "Only PDF files are allowed"}), 400

        # 3. Save file temporarily for reliable reading
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            file.save(temp.name)
            reader = PdfReader(temp.name)
            text = ''
            for page in reader.pages:
                page_text = page.extract_text()
                text += page_text if page_text else ''

        if not text.strip():
            return jsonify({"error": "Could not extract text from the PDF"}), 400

        # 4. Clean and predict
        cleaned = clean_text(text)
        prediction = model.predict([cleaned])[0]

        return jsonify({"role": prediction}), 200

    except Exception as e:
        # 5. Log any backend errors
        print("❌ Error during resume parsing:")
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
