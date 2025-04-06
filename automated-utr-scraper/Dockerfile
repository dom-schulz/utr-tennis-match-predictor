# Use the official Python image as a base
FROM python:3.8-buster

# Install Chrome and ChromeDriver dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    curl \
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
    && rm -rf /var/lib/apt/lists/*

# Install Chrome
RUN apt-get update && apt-get install -y wget gnupg2 apt-transport-https ca-certificates
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
RUN echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list
RUN apt-get update && apt-get install -y google-chrome-stable

# Install specific compatible version of ChromeDriver
RUN CHROME_VERSION=$(google-chrome-stable --version | grep -oP '(?<=Google Chrome )[0-9]+') \
    && echo "Chrome version: $CHROME_VERSION" \
    && wget -q "https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/120.0.6099.109/linux64/chromedriver-linux64.zip" \
    && unzip chromedriver-linux64.zip \
    && mv chromedriver-linux64/chromedriver /usr/local/bin/ \
    && chmod +x /usr/local/bin/chromedriver \
    && rm -rf chromedriver-linux64.zip chromedriver-linux64

# Set up the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY *.py *.csv ./

# Verify Chrome installation
RUN which google-chrome-stable
RUN google-chrome-stable --version || true
RUN which chromedriver
RUN chromedriver --version || true

# Set environment variables
ENV DISPLAY=:99
ENV GCS_BUCKET_NAME=utr_scraper_bucket
ENV PORT=8080
ENV CHROME_BIN="/usr/bin/google-chrome-stable"
ENV CHROME_DRIVER="/usr/local/bin/chromedriver"
ENV PYTHONUNBUFFERED=1

# Create necessary directories and set permissions
RUN mkdir -p /tmp/chrome-profile /tmp/.X11-unix \
    && chmod -R 777 /tmp \
    && chmod 1777 /tmp/.X11-unix

# Add debug print for Chrome paths
RUN echo "Chrome binary: ${CHROME_BIN}" && \
    echo "ChromeDriver: ${CHROME_DRIVER}" && \
    echo "Chrome executable exists: $([ -f ${CHROME_BIN} ] && echo 'yes' || echo 'no')" && \
    echo "ChromeDriver exists: $([ -f ${CHROME_DRIVER} ] && echo 'yes' || echo 'no')"

# Run the scraper script
CMD ["python", "scrape_history_gcp.py"]