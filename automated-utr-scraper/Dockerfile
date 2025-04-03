# Use the official Python image with a more recent version
FROM python:3.11-slim

# Install Chrome and dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    xvfb \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    xdg-utils \
    && wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Install ChromeDriver (using webdriver-manager)
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

# Create necessary directories and set permissions
RUN mkdir -p /tmp/chrome-profile /tmp/.X11-unix \
    && chmod -R 777 /tmp \
    && chmod 1777 /tmp/.X11-unix

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
        # Run the scraper in the background with Xvfb\n\
        subprocess.Popen(["sh", "-c", "rm -rf /tmp/chrome-profile/* && Xvfb :99 -screen 0 1280x1024x24 > /dev/null 2>&1 & python3 scrape_history_gcp.py"])\n\
\n\
if __name__ == "__main__":\n\
    logger.info("Starting HTTP server on port %s", os.environ.get("PORT", 8080))\n\
    server = HTTPServer(("", int(os.environ.get("PORT", 8080))), Handler)\n\
    server.serve_forever()' > server.py

# Run the HTTP server
CMD ["python3", "server.py"]