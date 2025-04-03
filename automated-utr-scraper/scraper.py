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
def scrape_utr_history(df, email, password, offset=0, stop=1, writer=None):
    start_time = time.time()
    timeout = 30  # 30 seconds timeout
    
    print(f"\nStarting scrape with offset={offset}, stop={stop}")
    print(f"Total profiles to process: {len(df.iloc[offset:stop])}")
    
    # Initialize Chrome options
    options = webdriver.ChromeOptions()
    options.binary_location = "/usr/local/bin/chrome/chrome"
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-software-rasterizer')
    options.add_argument('--remote-debugging-port=9222')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-dev-tools')
    options.add_argument('--no-default-browser-check')
    options.add_argument('--disable-infobars')
    options.add_argument('--disable-notifications')
    options.add_argument('--disable-popup-blocking')
    options.add_argument('--disable-web-security')
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--ignore-ssl-errors')
    options.add_argument('--ignore-certificate-errors-spki-list')
    options.add_argument('--user-data-dir=/tmp/chrome-profile')
    options.add_argument('--profile-directory=Default')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--disable-setuid-sandbox')
    options.add_argument('--disable-seccomp-filter-sandbox')
    options.add_argument('--disable-breakpad')
    options.add_argument('--disable-client-side-phishing-detection')
    options.add_argument('--disable-cast')
    options.add_argument('--disable-cast-streaming-hw-encoding')
    options.add_argument('--disable-cloud-import')
    options.add_argument('--disable-default-apps')
    options.add_argument('--disable-domain-reliability')
    options.add_argument('--disable-features=TranslateUI')
    options.add_argument('--disable-hang-monitor')
    options.add_argument('--disable-ipc-flooding-protection')
    options.add_argument('--disable-prompt-on-repost')
    options.add_argument('--disable-sync')
    options.add_argument('--disable-background-timer-throttling')
    options.add_argument('--disable-backgrounding-occluded-windows')
    options.add_argument('--disable-renderer-backgrounding')
    options.add_argument('--metrics-recording-only')
    options.add_argument('--mute-audio')
    options.add_argument('--no-first-run')
    options.add_argument('--password-store=basic')
    options.add_argument('--use-mock-keychain')
    options.add_argument('--single-process')
    options.add_argument('--disable-blink-features=AutomationControlled')
    
    try:
        # Use the installed ChromeDriver
        service = ChromeService('/usr/local/bin/chromedriver')
        driver = webdriver.Chrome(service=service, options=options)
        print("ChromeDriver initialized successfully")
    except Exception as e:
        print(f"Error initializing ChromeDriver: {str(e)}")
        raise
    
    log_in_url = "https://app.utrsports.net/login"
    
    try:
        sign_in(driver, log_in_url, email, password)
    except Exception as e:
        print(f"Error during sign in: {str(e)}")
        driver.quit()
        raise
    
    for i in range(len(df)):
        if i == stop:
                break
                
        try:
            search_url = f"https://app.utrsports.net/profiles/{round(df['p_id'][i+offset])}?t=6"
        except:
            continue

        load_page(driver, search_url)

        time.sleep(0.25)

        scroll_page(driver)

        try:
            time.sleep(1)
            show_all = driver.find_element(By.LINK_TEXT, 'Show all')
            show_all.click()
        except:
            try:
                time.sleep(2)
                show_all = driver.find_element(By.LINK_TEXT, 'Show all')
                show_all.click()
            except:
                print(f"{df['f_name'][i]} | {df['l_name'][i]} | {df['p_id'][i]}")
                continue
                
        time.sleep(1)

        scroll_page(driver)

        # Now that the page is rendered, parse the page with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        container = soup.find("div", class_="newStatsTabContent__section__1TQzL p0 bg-transparent")
        
        utrs = container.find_all("div", class_="row")
        
        for j in range(len(utrs)):
            if j == 0:
                continue
            utr = utrs[j].find("div", class_="newStatsTabContent__historyItemRating__GQUXw").text
            utr_date = utrs[j].find("div", class_="newStatsTabContent__historyItemDate__jFJyD").text

            data_row = [df['f_name'][i+offset], df['l_name'][i+offset], utr_date, utr]

            writer.writerow(data_row)

    # Close the driver
        driver.quit() 
###