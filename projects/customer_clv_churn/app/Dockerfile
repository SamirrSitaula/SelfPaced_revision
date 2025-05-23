# Use official Python base image
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Copy your Flask app and models into the container
COPY . /app/
COPY data /app/data

COPY outputs /outputs

COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask will run on
EXPOSE 5000

# Start the Flask app
CMD ["python", "app.py"]

# Inside Dockerfile

# Copy the model into the correct path
COPY outputs/models/logistic_model.pkl /outputs/models/logistic_model.pkl
COPY outputs/models/scaler.pkl /outputs/models/scaler.pkl
