FROM python:3.8-buster

# Set working directory in the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY *.py ./
# COPY credentials.json ./    

# Create a directory to save the output (if it's not already created in the code)
RUN mkdir -p /app/output

# Set environment variable to prevent buffering of Python output
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV MODEL_BUCKET_NAME=utr-model-training-bucket
ENV UTR_BUCKET_NAME=utr_scraper_bucket
ENV MATCHES_BUCKET_NAME=matches-scraper-bucket

# Command to run your Python script (ensure your Python script writes to /app/output)
CMD ["python", "model.py"]
