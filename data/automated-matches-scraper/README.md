## Deployment

This service is deployed automatically to Google Cloud Run using GitHub Actions. For detailed information on the deployment process, secret management, and local development setup, please refer to the [main README.md file](../../README.md#deployment).

## Core Files

[`matches.py`](https://github.com/dom-schulz/utr-tennis-match-predictor/blob/main/automated-matches-scraper/matches.py)

The "main" file of the scraper that orchestrates the entire process. It:
- Initializes logging and configuration
- Sets up connections to Google Cloud Storage
- Coordinates the scraping processes
- Includes extensive logging for debugging and future troubleshooting
- Manages error handling and retry logic

[`scraper.py`](https://github.com/dom-schulz/utr-tennis-match-predictor/blob/main/automated-matches-scraper/scraper.py)

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
- Configures environment variables and entry points (i.e., `CMD ["python", "matches.py"]`)
- Creates a reproducible environment for consistent execution during local and cloud testing

[`cloudbuild.yaml`](https://github.com/dom-schulz/utr-tennis-match-predictor/blob/main/automated-utr-scraper/cloudbuild.yaml)

Automates the Docker image build process:
- Connected to GitHub repository for continuous deployment
- Automatically builds a new Docker image whenever changes are committed
- Pushes the image to Google Artifact Registry
- Ensures the scraper always runs with the latest code
- Located in repo root directory

`credentials.json`(not in repository)
- Used for local testing with Google Cloud services
- Contains service account credentials for GCP authentication
- Excluded from git repository via `.gitignore` for security
- On the cloud, credentials are managed through GCP's built-in authentication

### Datasets

The repository contains sample datasets for reference. The production data is continuously updated and maintained in Google Cloud Storage (GCS) buckets.

[`profile_id.csv`](https://github.com/dom-schulz/utr-tennis-match-predictor/blob/main/automated-matches-scraper/profile_id.csv)

Contains the list of UTR profile IDs to scrape:
- Stored and written to in `utr_scraper_bucket` in Google Cloud Storage (GCS)
- Each row represents a tennis player to be processed
- The scraper iterates through these profiles to collect match data

[`utr_history.csv`](https://github.com/dom-schulz/utr-tennis-match-predictor/blob/main/automated-matches-scraper/utr_history.csv)

Contains historical UTR ratings for players:
- Stored and written to the `utr_scraper_bucket` in GCS
- Includes player names, dates, and corresponding UTR ratings
- Used to track player rating changes over time
- Processed into a dictionary format for efficient lookup during scraping

[`atp_utr_tennis_matches.csv`](https://github.com/dom-schulz/utr-tennis-match-predictor/blob/main/automated-matches-scraper/atp_utr_tennis_matches.csv)

Contains the collected tennis match data:
- Stored and written to in the `matches-scraper-bucket` in GCS
- Columns include date, player names, IDs, UTR ratings, tournament category, score, winner, etc.
- Updated with new matches after each scraping run
- Duplicates are automatically removed based on date and player combinations

**Note**: For additional details and architecture information, refer to the [UTR Scraper Documentation](https://github.com/dom-schulz/utr-tennis-match-predictor/blob/main/automated-utr-scraper/README.md).