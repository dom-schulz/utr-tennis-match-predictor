# Use the official Python image with a more recent version
FROM python:3.11-slim

# Install Chrome and dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    xvfb \
    && wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Install ChromeDriver
RUN wget -q "https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/134.0.6998.165/linux64/chromedriver-linux64.zip" \
    && unzip chromedriver-linux64.zip \
    && mv chromedriver-linux64/chromedriver /usr/local/bin/ \
    && rm -rf chromedriver-linux64.zip chromedriver-linux64 \
    && chmod +x /usr/local/bin/chromedriver

# Create and set permissions for Chrome profile directory
RUN mkdir -p /tmp/chrome-profile \
    && chmod -R 777 /tmp

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set environment variables
ENV DISPLAY=:99
ENV GCS_BUCKET_NAME=utr_scraper_bucket
ENV PORT=8080

# Create a simple HTTP server script
RUN echo 'from http.server import HTTPServer, BaseHTTPRequestHandler\n\
import os\n\
import subprocess\n\
\n\
class Handler(BaseHTTPRequestHandler):\n\
    def do_GET(self):\n\
        self.send_response(200)\n\
        self.end_headers()\n\
        self.wfile.write(b"Starting UTR scraper...\\n")\n\
        # Run the scraper in the background\n\
        subprocess.Popen(["sh", "-c", "rm -rf /tmp/chrome-profile/* && Xvfb :99 -screen 0 1280x1024x24 > /dev/null 2>&1 & python3 scrape_history_gcp.py"])\n\
\n\
if __name__ == "__main__":\n\
    server = HTTPServer(("", int(os.environ.get("PORT", 8080))), Handler)\n\
    server.serve_forever()' > server.py

# Run the HTTP server
CMD ["python3", "server.py"]
