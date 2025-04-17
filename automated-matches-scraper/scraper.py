from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import csv
from datetime import date
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import numpy as np
from datetime import datetime
import random
from dateutil.relativedelta import relativedelta
import pandas as pd
import os
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Function to get configured Chrome options for headless mode in Docker
def get_chrome_options():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    
    # Add user agent to avoid detection
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    # Disable images to save bandwidth and speed up scraping
    prefs = {"profile.managed_default_content_settings.images": 2}
    chrome_options.add_experimental_option("prefs", prefs)
    
    logger.info("Chrome options configured for headless mode")
    return chrome_options


### Sign In UTR ###
def sign_in(driver, log_in_url, email, password):
    logger.info(f"Attempting to sign in to UTR with email: {email[:3]}***")
    try:
        driver.get(log_in_url)
        time.sleep(2)  # Increased from 1 to ensure page loads

        # Verify page loaded correctly
        page_source = driver.page_source
        if "emailInput" not in page_source:
            logger.error("Login page elements not found. Page source: " + page_source[:200] + "...")
            raise Exception("Login page not loaded correctly")
        
        email_box = driver.find_element(By.ID, 'emailInput')
        password_box = driver.find_element(By.ID, 'passwordInput')
        login_button = driver.find_element(By.CSS_SELECTOR, 'button.btn.btn-primary.btn-xl.btn-block')

        email_box.clear()  # Clear in case there's text
        email_box.send_keys(email)
        logger.info("Email entered successfully")
        
        password_box.clear()  # Clear in case there's text
        password_box.send_keys(password)
        logger.info("Password entered successfully")
        
        time.sleep(0.5)
        login_button.click()
        logger.info("Login button clicked")
        
        # Wait longer for login to complete - especially important in headless mode
        time.sleep(4)  # Increased from 2.5

        # Verify login success by checking for elements that would only appear post-login
        if "Sign Out" in driver.page_source or "My Account" in driver.page_source:
            logger.info("Login successful")
        else:
            logger.warning("Login might have failed - typical post-login elements not found")
            logger.warning("Current URL after login attempt: " + driver.current_url)
    except Exception as e:
        logger.error(f"Error during sign in: {str(e)}")
        logger.error(traceback.format_exc())
        raise

### URL Modification ###
def edit_url(city, state, lat, long):
    d = str(date.today())
    d.replace('-', '/')

    url = f'https://app.utrsports.net/search?sportTypes=tennis,pickleball&startDate={d}&distance=10mi&utrMin=1&utrMax=16&utrType=verified&utrTeamType=singles&utrFitPosition=6&type=players&lat={lat}&lng={long}&locationInputValue={city},%20{state},%20USA&location={city},%20{state},%20USA' # initliaze url

    return url
###

### Formats Match Scores ###
def collect_scores(all_scores):
    score = ''
    p1_games = 0
    p2_games = 0
    for i in range(int(len(all_scores) / 2)):
        if len(all_scores[i].text) == 1:
            score = score + all_scores[i].text + '-' + all_scores[i+(int(len(all_scores) / 2))].text + ' '
            p1_games += int(all_scores[i].text)
            p2_games += int(all_scores[i+(int(len(all_scores) / 2))].text)
        else:
            score = score + all_scores[i].text[0] + '-' + all_scores[i+(int(len(all_scores) / 2))].text[0] + ' '
            p1_games += int(all_scores[i].text[0])
            p2_games += int(all_scores[i+int(len(all_scores) / 2)].text[0])
    score = score[:-1]
    return score, p1_games, p2_games
###

### Loads The Page ###
def load_page(driver, url):
    logger.info(f"Loading page: {url}")
    try:
        driver.get(url)
        time.sleep(2)  # Increased from 1 for more reliable loading
        logger.info(f"Page loaded successfully: {url[:60]}...")
        return True
    except Exception as e:
        logger.error(f"Error loading page {url}: {str(e)}")
        logger.error(traceback.format_exc())
        return False
###

### Scrolls The Page ###
def scroll_page(driver):
    logger.info("Starting page scroll")
    try:
        previous_height = driver.execute_script("return document.body.scrollHeight")
        scroll_count = 0
        max_scrolls = 10  # Limit scrolls to prevent infinite loops
        
        while scroll_count < max_scrolls:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(0.5)
            new_height = driver.execute_script("return document.body.scrollHeight")
            
            if new_height == previous_height:
                logger.info(f"Scrolling complete after {scroll_count+1} scrolls")
                break
                
            previous_height = new_height
            scroll_count += 1
            
        if scroll_count >= max_scrolls:
            logger.warning("Reached maximum scroll limit - page may not be fully loaded")
    except Exception as e:
        logger.error(f"Error during page scrolling: {str(e)}")
        logger.error(traceback.format_exc())
###

### Get UTR Rating ###
def scrape_player_matches(profile_ids, utr_history, matches, email, password, offset=0, stop=1, writer=None):
    
    # Initialize the Selenium WebDriver with headless options for Docker
    logger.info("Initializing Chrome driver for player matches scraping")
    chrome_options = get_chrome_options()
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        logger.info("Chrome driver initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Chrome driver: {str(e)}")
        logger.error(traceback.format_exc())
        return None

    url = 'https://app.utrsports.net/'
    today = date.today()

    sign_in(driver, url, email, password)

    y = 1
    for i in range(len(profile_ids)):
        
        # TESTING Purposes
        if i == 300:
            break
        
        if i == stop:
            break
        # if i % round(len(profile_ids)/100) == 0:
        #     print(f'Scraping..... {y}%')
        #     y += 1

        try:
            search_url = f"https://app.utrsports.net/profiles/{round(profile_ids['p_id'][i+offset])}"
        except:
            continue

        load_page(driver, search_url)
            
        scroll_page(driver)

        # Now that the page is rendered, parse the page with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        tournaments = soup.find_all("div", class_="eventItem__eventItem__2Xpsd")
       
        '''
        For each tournament, grab the data specified from each match in the tournament.
        Rework some data based on winners vs losers and errors that need exceptions.
        '''
        for tourney in tournaments:
            try:
                tourney_name = tourney.find("span", class_="").text
            except:
                continue
          
            if ',' in tourney_name:
                for j in range(len(tourney_name)):
                    if tourney_name[j] == ',':
                        tourney_name = tourney_name[:j]
                        break
            surface = "Hard"
            if tourney_name == "Wimbledon":
                slam = "Grand Slam"
                surface = "Grass"
            elif tourney_name == "French Open":
                slam = "Grand Slam"
                surface = "Clay"
            elif tourney_name == "US Open" or tourney_name == "Austrlian Open":
                slam = "Grand Slam"
                surface = "Hard"
            else:
                temp = ''
                _ = False
                slam = ''
                for ch in tourney_name:
                    if temp == 'ATP' and not _:
                        _ = True
                    elif ch == ' ' and _:
                        slam = temp
                        temp = ''
                        _ = False
                    temp = temp + ch
                if temp != '':
                    tourney_name = temp

                if tourney_name[0] == ' ':
                    tourney_name = tourney_name[1:]

            matches = tourney.find_all("div", class_="d-none d-md-block")

            for match in matches:
                tround = match.find("div", class_="scorecard__header__2iDdF").text
                r = ''
                for j in range(len(tround)):
                    if tround[-j-1] != '|':
                        r = tround[-1*j-1] + r
                    else:
                        r = r[1:]
                        break
                
                start = -1
                end = -1
                for j in range(len(tround)):
                    if tround[j] == '|' and start == -1:
                        start = j
                    elif tround[j] == '|' and start != -1:
                        end = j
                if end == -1:
                    match_date_str = tround[(start+2):j]
                else:
                    match_date_str = tround[(start+2):(end-1)]

                match_date = datetime.strptime(match_date_str, "%b %d").replace(year=datetime.now().year).date()
                if match_date > today:
                    match_date = match_date - relativedelta(year=datetime.now().year-1)

                data_row = [tourney_name, match_date, slam, 'Outdoor', surface, r]
                is_tie = False

                try:
                    winner_name = match.find("a", class_="flex-column player-name winner").text # throws error when TIE (COLLEGE MATCHES)
                    loser_name = match.find("a", class_="flex-column player-name").text
                except:
                    tie = match.find_all("a", class_="flex-column player-name")
                    winner_name, loser_name = tie[0].text, tie[1].text
                    is_tie = True

                try:
                    temp = False
                    for utrdata in utr_history[winner_name]:
                        if datetime.strptime(utrdata[1], '%Y-%m-%d').date() <= match_date:
                            w_utr = utrdata[0]
                            temp = True
                            break
                    if not temp:
                        w_utr = utr_history[winner_name][len(utr_history[winner_name])-1][0]
                    temp = False
                    for utrdata in utr_history[loser_name]:
                        if datetime.strptime(utrdata[1], '%Y-%m-%d').date() <= match_date:
                            l_utr = utrdata[0]
                            temp = True
                            break
                    if not temp:
                        l_utr = utr_history[loser_name][len(utr_history[loser_name])-1][0]
                except:
                    continue

                all_scores = match.find_all("div", "score-item")
                score, p1_games, p2_games = collect_scores(all_scores)
                score = score if score else 'W'
                if score == 'W':
                    continue

                sets = 0
                num_sets = 0
                for j in range(len(score)):
                    if j % 4 == 0:
                        num_sets += 1
                        try:
                            if int(score[j]) > int(score[j+2]):
                                sets += 1
                            else:
                                sets -= 1
                        except:
                            continue
                if num_sets < 3:
                    best_of = 3
                elif num_sets == 3 and abs(sets) == 1:
                    best_of = 3
                else:
                    best_of = 5

                data_row += [best_of]

                winner_name1 = ''
                a = False
                for ch in winner_name:
                    if ch == ' ':
                        a = True
                    elif a:
                        winner_name1 = winner_name1 + ch
                winner_name1 = winner_name1 + ' ' + winner_name[0] + '.'

                loser_name1 = ''
                a = False
                for ch in loser_name:
                    if ch == ' ':
                        a = True
                    elif a:
                        loser_name1 = loser_name1 + ch
                loser_name1 = loser_name1 + ' ' + loser_name[0] + '.'

                ri = random.randint(0,1)
                if ri == 0:
                    data_row += [winner_name1, w_utr, loser_name1, l_utr, winner_name1, p1_games, p2_games, score, 0]
                else:
                    data_row += [loser_name1, l_utr, winner_name1, w_utr, winner_name1, p1_games, p2_games, score, 1]

                if is_tie:
                    data_row[-1] = 0.5  # Mark ties properly

                # Log add data row
                logger.info(f'Adding data row: {data_row}')
                writer.writerow(data_row)

    # Close the driver
    driver.quit()
###




### Get UTR History ###
def scrape_utr_history(df, email, password, offset=0, stop=1, writer=None):
    # Initialize the Selenium WebDriver with headless options for Docker
    logger.info("Initializing Chrome driver for UTR history scraping")
    chrome_options = get_chrome_options()
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        logger.info("Chrome driver initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Chrome driver: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.DataFrame(columns=['first_name', 'last_name', 'date', 'utr'])
    
    # Create a list to store data rows
    data_rows = []
    url = 'https://app.utrsports.net/'
    
    try:
        sign_in(driver, url, email, password)
    except Exception as e:
        logger.error(f"Sign in failed, aborting scraping: {str(e)}")
        driver.quit()
        return pd.DataFrame(columns=['first_name', 'last_name', 'date', 'utr'])

    # Determine the actual number of profiles to process
    if stop == -1:
        end_idx = len(df)
    else:
        end_idx = min(stop, len(df))
    
    logger.info(f"Starting UTR history scrape for {end_idx - offset} profiles")
    
    # Track progress
    processed_count = 0
    success_count = 0

    for i in range(offset, end_idx):
        try:
            logger.info(f"Processing profile {i-offset+1}/{end_idx-offset}: {df['f_name'][i]} {df['l_name'][i]}")
            
            # Check if profile ID is valid
            if pd.isna(df['p_id'][i]) or df['p_id'][i] == '':
                logger.warning(f"Skipping profile with missing ID: {df['f_name'][i]} {df['l_name'][i]}")
                continue
                
            search_url = f"https://app.utrsports.net/profiles/{int(df['p_id'][i])}?t=6"
        except Exception as e:
            logger.error(f"Error preparing URL for profile at index {i}: {str(e)}")
            continue

        if not load_page(driver, search_url):
            logger.warning(f"Skipping profile {df['f_name'][i]} {df['l_name'][i]} due to page load failure")
            processed_count += 1
            continue

        time.sleep(0.5)  # Increased from 0.25
        scroll_page(driver)

        # Take a screenshot for debugging if needed
        try:
            # Only attempt screenshots in debug mode and if directory exists
            screenshot_dir = os.getenv("SCREENSHOT_DIR", None)
            if screenshot_dir:
                os.makedirs(screenshot_dir, exist_ok=True)
                screenshot_path = os.path.join(screenshot_dir, f"debug_screenshot_{df['f_name'][i]}_{df['l_name'][i]}.png")
                driver.save_screenshot(screenshot_path)
                logger.info(f"Saved screenshot to {screenshot_path}")
        except Exception as e:
            logger.warning(f"Could not save screenshot: {str(e)}")
            # Screenshot failure is non-critical, continue processing

        # Look for "Show all" button
        show_all_found = False
        try:
            logger.info("Looking for 'Show all' button")
            time.sleep(1)
            show_all = driver.find_element(By.LINK_TEXT, 'Show all')
            show_all.click()
            logger.info("Clicked 'Show all' button")
            show_all_found = True
        except Exception as e:
            logger.warning(f"First attempt to find 'Show all' button failed: {str(e)}")
            
            try:
                # Try again with a longer wait
                logger.info("Making second attempt to find 'Show all' button")
                time.sleep(3)
                show_all = driver.find_element(By.LINK_TEXT, 'Show all')
                show_all.click()
                logger.info("Clicked 'Show all' button on second attempt")
                show_all_found = True
            except Exception as e2:
                logger.error(f"Could not find 'Show all' button: {str(e2)}")
                logger.error(f"Debug info - Current URL: {driver.current_url}")
                logger.error(f"Page title: {driver.title}")
                
                # Check if we're still logged in
                if "Sign In" in driver.page_source or "Log In" in driver.page_source:
                    logger.error("Session appears to have expired, attempting to log in again")
                    try:
                        sign_in(driver, url, email, password)
                        # Try loading the profile again
                        load_page(driver, search_url)
                    except:
                        logger.error("Re-login attempt failed")
                
                processed_count += 1
                continue
        
        if show_all_found:
            time.sleep(1.5)  # Increased wait time after clicking "Show all"
            scroll_page(driver)

        # Now that the page is rendered, parse the page with BeautifulSoup
        logger.info("Parsing page content with BeautifulSoup")
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Debug page content
        if "UTR History" not in driver.page_source:
            logger.warning("UTR History section might be missing from the page")
        
        # Find the UTR history container
        container = soup.find("div", class_="newStatsTabContent__section__1TQzL p0 bg-transparent")
        
        if not container:
            logger.warning(f"UTR history container not found for {df['f_name'][i]} {df['l_name'][i]}")
            processed_count += 1
            continue
            
        utrs = container.find_all("div", class_="row")
        logger.info(f"Found {len(utrs)} UTR history entries")
        
        # Count how many records we're actually collecting
        record_count = 0
        
        for j in range(len(utrs)):
            if j == 0:
                continue
                
            try:
                utr = utrs[j].find("div", class_="newStatsTabContent__historyItemRating__GQUXw").text
                utr_date = utrs[j].find("div", class_="newStatsTabContent__historyItemDate__jFJyD").text
                
                logger.info(f"Found UTR: {utr} from date: {utr_date}")
                
                # Create data row with first_name, last_name (for compatibility with both column naming schemes)
                data_row = [df['f_name'][i], df['l_name'][i], utr_date, utr]
                
                # Add to data rows list
                data_rows.append(data_row)
                record_count += 1
                
                # If writer is provided, still write to CSV for backward compatibility
                if writer:
                    writer.writerow(data_row)
            except Exception as e:
                logger.error(f"Error extracting UTR data from row {j}: {str(e)}")
                continue
                
        logger.info(f"Extracted {record_count} UTR records for {df['f_name'][i]} {df['l_name'][i]}")
        processed_count += 1
        if record_count > 0:
            success_count += 1

    # Close the driver
    logger.info(f"Closing Chrome driver after scraping UTR history. Processed {processed_count} profiles with {success_count} successful extractions.")
    driver.quit()
    
    # Create DataFrame from collected data
    if not data_rows:
        logger.warning("No UTR history data was collected!")
        return pd.DataFrame(columns=['first_name', 'last_name', 'date', 'utr'])
        
    df_result = pd.DataFrame(data_rows, columns=['first_name', 'last_name', 'date', 'utr'])
    logger.info(f"Created DataFrame with {len(df_result)} total UTR records")
    return df_result
###