# Use the official Selenium standalone Chrome image
FROM selenium/standalone-chrome

# Install required packages
USER root
RUN apt-get update && apt-get install -y \
    python3-venv \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Create a new directory with correct permissions
RUN mkdir -p /home/seluser/app && chown -R seluser:seluser /home/seluser/app
WORKDIR /home/seluser/app

# Switch to seluser (the user in selenium/standalone-chrome)
USER seluser

# Create and activate virtual environment
RUN python3 -m venv /home/seluser/app/venv
ENV PATH="/home/seluser/app/venv/bin:$PATH"

# Copy requirements first to leverage Docker cache
COPY --chown=seluser:seluser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY --chown=seluser:seluser . .

# Set environment variables
ENV DISPLAY=:99
ENV GCS_BUCKET_NAME=utr_scraper_bucket
ENV PORT=8080
ENV CHROME_BIN=/usr/bin/google-chrome
ENV CHROME_DRIVER=/usr/bin/chromedriver

# Create Chrome profile directory in user's home
RUN mkdir -p /home/seluser/chrome-profile

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
        subprocess.Popen(["sh", "-c", "rm -rf /home/seluser/chrome-profile/* && Xvfb :99 -screen 0 1280x1024x24 > /dev/null 2>&1 & /home/seluser/app/venv/bin/python3 scrape_history_gcp.py"])\n\
\n\
if __name__ == "__main__":\n\
    logger.info("Starting HTTP server on port %s", os.environ.get("PORT", 8080))\n\
    server = HTTPServer(("", int(os.environ.get("PORT", 8080))), Handler)\n\
    server.serve_forever()' > server.py

# Run the HTTP server
CMD ["/home/seluser/app/venv/bin/python3", "server.py"]
