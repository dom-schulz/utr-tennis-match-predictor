from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from bs4 import BeautifulSoup
import time
import csv
from datetime import date
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import numpy as np
from datetime import datetime
import random
from dateutil.relativedelta import relativedelta
import os
import logging

# Set up logging to output to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

'''
NOTES:
    - Scraper can't go past Denver, CO (19th row in cities.csv). Gives error:
        * DevTools listening on ws://127.0.0.1:61306/devtools/browser/3da6171b-a2e7-40ec-88a5-55c69b3d4dfa
        * Created TensorFlow Lite XNNPACK delegate for CPU.
        * [24276:6160:0123/230738.058:ERROR:ssl_client_socket_impl.cc(878)] handshake failed; returned -1, SSL error code 1, net_error -101
        * [24276:6160:0123/230738.103:ERROR:ssl_client_socket_impl.cc(878)] handshake failed; returned -1, SSL error code 1, net_error -100
        * Attempting to use a delegate that only supports static-sized tensors with a graph that has dynamic-sized tensors (tensor#58 is a dynamic-sized tensor).
        * Traceback (most recent call last):
        * ...
'''

### Sign In UTR ###
def sign_in(driver, log_in_url, email, password):
    driver.get(log_in_url)
    time.sleep(2)  # Wait for page to load

    # Handle cookie consent banner if present
    try:
        cookie_button = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.ID, "onetrust-accept-btn-handler"))
        )
        cookie_button.click()
        time.sleep(1)
    except Exception:
        pass  # If no cookie banner, continue

    email_box = driver.find_element(By.ID, 'emailInput')
    password_box = driver.find_element(By.ID, 'passwordInput')
    login_button = driver.find_element(By.CSS_SELECTOR, 'button.btn.btn-primary.btn-xl.btn-block')

    email_box.send_keys(email)
    password_box.send_keys(password)
    time.sleep(0.5)
    login_button.click()

    time.sleep(2.5)

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
    driver.get(url)
    time.sleep(1)
###

### Scrolls The Page ###
def scroll_page(driver):
    previous_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(0.5)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == previous_height:
            break
        previous_height = new_height
###

### Get UTR Rating ###
def scrape_player_matches(profile_ids, utr_history, matches, email, password, offset=0, stop=1, writer=None):
    # Initialize the Selenium WebDriver (make sure you have the appropriate driver installed)
    driver = webdriver.Chrome()
    url = 'https://app.utrsports.net/'
    today = date.today()

    sign_in(driver, url, email, password)

    y = 1
    for i in range(len(profile_ids)):
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

                writer.writerow(data_row)

    # Close the driver
    driver.quit()
###

### Get UTR History ###
def scrape_utr_history(profile_ids, email, password, offset=0, stop=-1, writer=None):
    """Scrapes UTR history for a list of profile IDs."""
    driver = None
    try:
        driver = setup_driver()
        sign_in(driver, email, password)
        
        for index, row in profile_ids.iterrows():
            if stop != -1 and index >= stop:
                break
            if index < offset:
                continue
                
            profile_id = row['profile_id']
            first_name = row['first_name']
            last_name = row['last_name']
            
            logger.info(f"Scraping UTR history for {first_name} {last_name} (ID: {profile_id})")
            
            try:
                # Navigate to profile page
                driver.get(f"https://app.utrsports.net/Profile/Player/{profile_id}")
                time.sleep(2)  # Wait for page to load
                
                # Click on History tab
                history_tab = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'History')]"))
                )
                history_tab.click()
                time.sleep(2)  # Wait for history to load
                
                # Get all history rows
                history_rows = WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, "table.table tbody tr"))
                )
                
                # Process each history row
                for row in history_rows:
                    try:
                        cells = row.find_elements(By.TAG_NAME, "td")
                        if len(cells) >= 3:
                            date = cells[0].text.strip()
                            utr = cells[2].text.strip()
                            
                            if writer:
                                writer.writerow([first_name, last_name, date, utr])
                            else:
                                print(f"{first_name},{last_name},{date},{utr}")
                    except Exception as e:
                        logger.error(f"Error processing history row: {str(e)}")
                        continue
                
            except Exception as e:
                logger.error(f"Error scraping profile {profile_id}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Error in scrape_utr_history: {str(e)}")
        raise
    finally:
        if driver:
            driver.quit()