# Use the official Python image with a more recent version
FROM python:3.11-slim

# Install Chrome and dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    xvfb \
    curl \
    && wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK to use gsutil for accessing GCS
RUN curl https://sdk.cloud.google.com | bash
ENV PATH $PATH:/root/google-cloud-sdk/bin

# Install ChromeDriver
RUN wget -q "https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/134.0.6998.165/linux64/chromedriver-linux64.zip" \
    && unzip chromedriver-linux64.zip \
    && mv chromedriver-linux64/chromedriver /usr/local/bin/ \
    && rm -rf chromedriver-linux64.zip chromedriver-linux64 \
    && chmod +x /usr/local/bin/chromedriver

# Create and set permissions for Chrome profile directory
RUN mkdir -p /tmp/chrome-profile \
    && chmod -R 777 /tmp

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set environment variables for Cloud Run
ENV DISPLAY=:99
ENV GCS_BUCKET_NAME=utr-scraper-bucket

# Expose the port that Cloud Run will use
EXPOSE 8080

# The CMD will download the files from GCS and run your script
CMD ["sh", "-c", "gsutil cp gs://$GCS_BUCKET_NAME/profile_id.csv /app/ && gsutil cp gs://$GCS_BUCKET_NAME/scraper.py /app/ && gsutil cp gs://$GCS_BUCKET_NAME/scrape_history_gcp.py /app/ && ls /app/ && rm -rf /tmp/chrome-profile/* && Xvfb :99 -screen 0 1280x1024x24 > /dev/null 2>&1 & python3 /app/scrape_history_gcp.py"]
