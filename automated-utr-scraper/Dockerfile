# Use the official Python image
FROM python:3.8-buster

# Install manually all the missing libraries
RUN apt-get update && apt-get install -y \
    gconf-service \
    libasound2 \
    libatk1.0-0 \
    libcairo2 \
    libcups2 \
    libfontconfig1 \
    libgdk-pixbuf2.0-0 \
    libgtk-3-0 \
    libnspr4 \
    libpango-1.0-0 \
    libxss1 \
    fonts-liberation \
    libappindicator1 \
    libnss3 \
    lsb-release \
    xdg-utils \
    wget

# Install Chrome
RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
    dpkg -i google-chrome-stable_current_amd64.deb || apt-get -fy install && \
    rm google-chrome-stable_current_amd64.deb

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set environment variables
ENV PORT=8080
ENV GCS_BUCKET_NAME=utr_scraper_bucket

# Create a simple HTTP server script
RUN echo 'from http.server import HTTPServer, BaseHTTPRequestHandler\n\
import os\n\
import subprocess\n\
import logging\n\
\n\
logging.basicConfig(level=logging.INFO)\n\
logger = logging.getLogger(__name__)\n\
\n\
class Handler(BaseHTTPRequestHandler):\n\
    def do_GET(self):\n\
        logger.info("Received request, starting scraper...")\n\
        self.send_response(200)\n\
        self.end_headers()\n\
        self.wfile.write(b"Starting UTR scraper...\\n")\n\
        # Run the scraper in the background\n\
        subprocess.Popen(["python3", "scrape_history_gcp.py"])\n\
\n\
if __name__ == "__main__":\n\
    logger.info("Starting HTTP server on port %s", os.environ.get("PORT", 8080))\n\
    server = HTTPServer(("", int(os.environ.get("PORT", 8080))), Handler)\n\
    server.serve_forever()' > server.py

# Run the HTTP server
CMD ["python3", "server.py"]