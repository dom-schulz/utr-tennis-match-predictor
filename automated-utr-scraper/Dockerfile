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

# Install Chrome for Testing
RUN wget -q "https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/120.0.6099.109/linux64/chrome-linux64.zip" \
    && unzip chrome-linux64.zip \
    && mv chrome-linux64 /usr/local/bin/chrome \
    && rm chrome-linux64.zip \
    && chmod -R 755 /usr/local/bin/chrome

# Install ChromeDriver
RUN wget -q "https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/120.0.6099.109/linux64/chromedriver-linux64.zip" \
    && unzip chromedriver-linux64.zip \
    && mv chromedriver-linux64/chromedriver /usr/local/bin/ \
    && rm -rf chromedriver-linux64.zip chromedriver-linux64 \
    && chmod +x /usr/local/bin/chromedriver

# Set up the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY *.py *.csv ./

# Set environment variables
ENV DISPLAY=:99
ENV GCS_BUCKET_NAME=utr_scraper_bucket
ENV PORT=8080
ENV PATH="/usr/local/bin/chrome:${PATH}"
ENV CHROME_BIN="/usr/local/bin/chrome/chrome"
ENV CHROME_DRIVER="/usr/local/bin/chromedriver"

# Create necessary directories and set permissions
RUN mkdir -p /tmp/chrome-profile /tmp/.X11-unix \
    && chmod -R 777 /tmp \
    && chmod 1777 /tmp/.X11-unix \
    && chown -R root:root /usr/local/bin/chrome \
    && chown -R root:root /usr/local/bin/chromedriver

# Run the scraper script
CMD ["python", "scrape_history_gcp.py"]