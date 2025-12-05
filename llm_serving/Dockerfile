# Use a suitable base image with CUDA support
FROM nvcr.io/nvidia/pytorch:22.10-py3

# Set environment variables
ENV MODEL_ID="BioMistral/BioMistral-7B"
ENV PYTHONUNBUFFERED=TRUE
WORKDIR /opt/program

# Install dependencies (Flask, etc.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the inference script
COPY predictor.py /opt/program/

# Set the port and command
EXPOSE 8080

# Use a production WSGI server (Gunicorn) to run the Flask app
# For local testing, you can use python predictor.py, but Gunicorn is better
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "predictor:app"]