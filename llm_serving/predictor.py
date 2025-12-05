# predictor.py (SageMaker-compliant script)
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, request, jsonify
import json
import logging

# Use port 8080 as required by SageMaker inference containers
HOST = '0.0.0.0'
PORT = 8080

app = Flask(__name__)
model = None
tokenizer = None
logging.basicConfig(level=logging.INFO)


# --- 1. MODEL LOADING ---
@app.before_first_request
def load_model():
    global model, tokenizer
    MODEL_ID = os.environ.get("MODEL_ID", "BioMistral/BioMistral-7B")
    logging.info(f"Loading unoptimized model: {MODEL_ID}...")

    try:
        # Load model in a high-precision dtype (e.g., FP16 or BF16) for baseline
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,  # Use FP16 for baseline VRAM/latency check
            device_map="auto",
            trust_remote_code=True
        ).eval()
        logging.info("Model loaded successfully!")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        # In a real environment, this error would stop the container
        model = None
        tokenizer = None


# --- 2. HEALTH CHECK ENDPOINT ---
# SageMaker pings this endpoint periodically to check container health.
@app.route('/ping', methods=['GET'])
def ping():
    if model is not None:
        return jsonify({'status': 'ok'}), 200
    return jsonify({'status': 'Model not loaded'}), 503


# --- 3. INFERENCE ENDPOINT ---
# This is where your AWS Lambda function will send the RAG prompt.
@app.route('/invocations', methods=['POST'])
def invocations():
    if model is None:
        return jsonify({'error': 'Model not ready'}), 503

    # SageMaker expects the data in the request body (often JSON or text)
    try:
        data = request.get_json(force=True)
        prompt = data.get("prompt")
        max_new_tokens = data.get("max_new_tokens", 256)
    except Exception as e:
        return jsonify({'error': f'Invalid request format: {e}'}), 400

    # Your core generation logic from the data scientist's notebook
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # TIME START (for your benchmarking)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    # TIME END

    response_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Return the response as JSON (SageMaker automatically handles content type)
    return jsonify({"answer": response_text})


# --- 4. START SERVER ---
if __name__ == '__main__':
    load_model()  # Load model directly when running locally for testing
    app.run(host=HOST, port=PORT)