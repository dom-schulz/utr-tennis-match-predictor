# Automated UTR Scraper

## Overview

This automated scraper collects historical match data from the Universal Tennis Rating (UTR) platform for professional tennis players. The primary goal is to gather comprehensive match history and performance data that will be used to train and optimize a tennis match prediction model.

## Deployment

This service is deployed automatically to Google Cloud Run using GitHub Actions. For detailed information on the deployment process, secret management, and local development setup, please refer to the [main README.md file](../../README.md#deployment).

## Core Files

[`scrape_history_gcp.py`](https://github.com/dom-schulz/utr-tennis-match-predictor/blob/main/automated-utr-scraper/scrape_history_gcp.py)

The "main" file of the scraper that orchestrates the entire process. It:
- Initializes logging and configuration
- Sets up connections to Google Cloud Storage
- Coordinates the scraping processes
- Includes extensive logging for debugging and future troubleshooting
- Manages error handling and retry logic

[`scraper.py`](https://github.com/dom-schulz/utr-tennis-match-predictor/blob/main/automated-utr-scraper/scraper.py)

Contains the core scraping functions and functionality:
- Originally received from a classmate (approximately 85% unchanged)
- Modified to run in a containerized environment on GCP
- Added `get_chrome_options()` to configure Chromedriver options for Docker
- Includes functions to navigate UTR website and extract match data
- Handles data extraction from the UTR profile pages

[`Dockerfile`](https://github.com/dom-schulz/utr-tennis-match-predictor/blob/main/automated-utr-scraper/Dockerfile)

Defines the container environment:
- Based on the Selenium standalone Chrome image (prebuilt docker image for these packages)
- Installs Python and required libraries
- Sets up the scraper code within the container
- Configures environment variables and entry points (i.e., `CMD ["python", "scrape_history_gcp.py"]`)
- Creates a reproducible environment for consistent execution during local and cloud testing

[`cloudbuild.yaml`](https://github.com/dom-schulz/utr-tennis-match-predictor/blob/main/automated-utr-scraper/cloudbuild.yaml)

Automates the Docker image build process:
- Connected to GitHub repository for continuous deployment
- Automatically builds a new Docker image whenever changes are committed
- Pushes the image to Google Artifact Registry
- Ensures the scraper always runs with the latest code

[`profile_id.csv`](https://github.com/dom-schulz/utr-tennis-match-predictor/blob/main/automated-utr-scraper/profile_id.csv)

Contains the list of UTR profile IDs to scrape:
- Each row represents a tennis player to be processed
- The scraper iterates through these profiles to collect match data

`credentials.json`(not in repository)
- Used for local testing with Google Cloud services
- Contains service account credentials for GCP authentication
- Excluded from git repository via `.gitignore` for security
- On the cloud, credentials are managed through GCP's built-in authentication

## Challenges and Solutions

This project presented several significant technical challenges that required extensive research and experimentation to overcome:

### Learning Google Cloud Platform
- Had to self-learn multiple Google Cloud services and their interactions without prior experience
- Worked extensively with documentation and leveraged AI tools to understand concepts
- Troubleshooting errors across connected services required developing a better understanding of GCP as a whole
- Navigated service permissions, networking, and resource management
- Developed a mental model of how different cloud services interact to create a cohesive system

### Docker Implementation
- This was my first experience working with Docker which required understanding the distinction between images and containers and how they interact
- Modifying `scraper.py` to run headless (without a display) within a container required significant time
- Testing locally before deployment helped identify environment-specific issues early
- Docker's consistency between local and cloud environments ultimately was very helpful (once understood)

### Google Cloud Logging and Debugging
- Tracking errors across five different GCP services required learning multiple logging interfaces (for debugging)
- Had to SSH into the Compute Engine VM frequently to read error logs and debug issues on the cloud
- Implemented structured logging in the application code to make logs more helpful

### Secret Management
- Implementing different credential handling strategies for local vs. cloud environments was time consuming
      - Local testing required mounting credentials to the Docker container, while cloud deployment used service accounts
- Ensuring sensitive information was never committed to Git required careful management of environment variables
- Learned to leverage GCP's built-in security services for managing sensitive credentials


## Future Scraper Enhancements

### Serverless Architecture
- Migrate from VM-based scraping to serverless Cloud Functions to process individual player data (ie. scrape for one player at a time via a Cloud Function for each)
- This would eliminate the need for Compute Engine and remove the runtime constraint 
- Would require redesigning the scraper to process single profiles efficiently

### Data Processing Pipeline
- Implement a more sophisticated data processing pipeline using Dataflow or Cloud Functions
- Integrate with BigQuery for more advanced query and analytics capabilities

### Cost Optimization
- Analyze and optimize cloud resource usage to reduce operational costs
- Implement more specific scheduling based on tournament calendars
- Consider regional pricing differences when deploying resources

### Continuous Integration Improvements
- Expand test coverage for the scraper components
- Implement automated end-to-end testing in CI/CD pipeline
- Add performance benchmarks to catch regressions
- Create development, staging, and production environments for safer deployments

**Note:** To recreate this scraper, you need a UTR account with a "Power Subscription" to access the detailed match history data. 