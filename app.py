import os
import time
import torch
import json
import logging
import subprocess
import numpy as np
from flask import Flask, request, render_template
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from joblib import load

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # Limit upload to 2MB

logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = "microsoft/codebert-base"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

codebert_model_path = os.path.join(BASE_DIR, "codebert_finetuned")
if not os.path.exists(codebert_model_path):
    raise FileNotFoundError(f"Model directory not found: {codebert_model_path}")
codebert_model = RobertaForSequenceClassification.from_pretrained(codebert_model_path)

log_reg_path = os.path.join(BASE_DIR, "models", "trained_model.joblib")
if not os.path.exists(log_reg_path):
    raise FileNotFoundError(f"Logistic regression model not found: {log_reg_path}")
log_reg = load(log_reg_path)

def run_pylint(file_path):
    try:
        result = subprocess.run(
            ['pylint', file_path, '--output-format=json'],
            capture_output=True, text=True, timeout=15
        )
        output = result.stdout.strip() if result.stdout.strip() else result.stderr.strip()
        return output
    except subprocess.TimeoutExpired:
        logging.error("Pylint timed out")
        return ""
    except Exception as e:
        logging.error(f"Error running Pylint: {e}")
        return ""

def parse_pylint_output(output):
    errors = {'E': 0, 'W': 0, 'C': 0}
    error_details, warning_details, convention_details = [], [], []

    try:
        pylint_json = json.loads(output) if output else []
        for entry in pylint_json:
            msg_id = entry.get("message-id", "").upper()
            error_info = {
                "line": entry.get("line", 0),
                "column": entry.get("column", 0),
                "code": entry.get("symbol", ""),
                "message": entry.get("message", "")
            }

            if msg_id.startswith("E"):
                errors['E'] += 1
                error_details.append(error_info)
            elif msg_id.startswith("W"):
                errors['W'] += 1
                warning_details.append(error_info)
            elif msg_id.startswith('C'):
                errors['C'] += 1
                convention_details.append(error_info)
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing Pylint output: {e}")
        logging.error(f"Raw Pylint Output: {output}")

    return errors, error_details, warning_details, convention_details

def tokenize_code(code):
    return tokenizer(code, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return render_template("index.html", error="No file provided"), 400

    file = request.files['file']
    if file.filename == '':
        return render_template("index.html", error="No selected file"), 400

    upload_dir = os.path.join(BASE_DIR, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    file.save(file_path)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
    except Exception as e:
        return render_template("index.html", error=f"Failed to read uploaded file: {str(e)}"), 500

    try:
        start = time.time()
        pylint_output = run_pylint(file_path)
        logging.info(f"Pylint runtime: {time.time() - start:.2f}s")

        features, error_details, warning_details, convention_details = parse_pylint_output(pylint_output) if pylint_output else ({'E': 0, 'W': 0, 'C': 0}, [], [], [])
        logging.info(f"Extracted Features: {features}")
    except Exception as e:
        logging.error(f"Pylint error: {e}")
        features = {'E': 0, 'W': 0, 'C': 0}
        error_details, warning_details, convention_details = [], [], []

    try:
        start = time.time()
        clean_code = code.strip()
        tokenized = tokenize_code(clean_code)

        with torch.no_grad():
            outputs = codebert_model(**tokenized)
            logits = outputs.logits

            logging.info(f"Logits: {logits}")

            # Temperature scaling
            temperature = 1.5
            probabilities = torch.nn.functional.softmax(logits / temperature, dim=1)

            logging.info(f"Probabilities: {probabilities}")

            prediction = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][prediction].item()

            if confidence < 0.4:
                quality_codebert = "Uncertain"
            else:
                quality_codebert = "Good" if prediction == 0 else "Bad"

            logging.info(f"CodeBERT: {quality_codebert} ({confidence:.2f}), time: {time.time() - start:.2f}s")
    except Exception as e:
        logging.error(f"CodeBERT error: {e}")
        quality_codebert = "Error"
        confidence = 0.0

    try:
        feature_vector = np.array([features['E'], features['W'], features['C']]).reshape(1, -1)
        log_reg_pred = log_reg.predict(feature_vector)[0]
        quality_log_reg = "Good" if log_reg_pred == 0 else "Bad"
        logging.info(f"LogReg: {quality_log_reg}")
    except Exception as e:
        logging.error(f"LogReg error: {e}")
        quality_log_reg = "Error"

    os.remove(file_path)

    return render_template('results.html',
                           code=code,
                           quality_codebert=quality_codebert,
                           confidence=round(confidence * 100, 2),
                           quality_log_reg=quality_log_reg,
                           errors=features['E'],
                           warnings=features['W'],
                           conventions=features['C'],
                           error_details=error_details,
                           warning_details=warning_details,
                           convention_details=convention_details)

if __name__ == '__main__':
    app.run(debug=False, port=5000)
