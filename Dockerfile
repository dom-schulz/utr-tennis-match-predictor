# Use the official Python image as a base
FROM python:3.8-buster

# Install Chrome and ChromeDriver
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

# Install ChromeDriver
RUN CHROMEDRIVER_VERSION=$(google-chrome --version | awk '{print $3}' | awk -F'.' '{print $1"."$2"."$3}') \
    && wget -q "https://chromedriver.storage.googleapis.com/$(wget -q -O - https://chromedriver.storage.googleapis.com/LATEST_RELEASE_$CHROMEDRIVER_VERSION)/chromedriver_linux64.zip" \
    && unzip chromedriver_linux64.zip \
    && mv chromedriver /usr/local/bin/ \
    && chmod +x /usr/local/bin/chromedriver \
    && rm chromedriver_linux64.zip

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
RUN google-chrome-stable --version
RUN which chromedriver
RUN chromedriver --version

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