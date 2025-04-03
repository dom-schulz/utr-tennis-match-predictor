# Use the official Selenium standalone Chrome image
FROM selenium/standalone-chrome

# Set the working directory
WORKDIR /app

# Create and activate virtual environment
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

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
        subprocess.Popen(["sh", "-c", "rm -rf /tmp/chrome-profile/* && Xvfb :99 -screen 0 1280x1024x24 > /dev/null 2>&1 & /app/venv/bin/python3 scrape_history_gcp.py"])\n\
\n\
if __name__ == "__main__":\n\
    logger.info("Starting HTTP server on port %s", os.environ.get("PORT", 8080))\n\
    server = HTTPServer(("", int(os.environ.get("PORT", 8080))), Handler)\n\
    server.serve_forever()' > server.py

# Run the HTTP server
CMD ["/app/venv/bin/python3", "server.py"]
