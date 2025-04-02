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

def sign_in(driver, log_in_url, email, password):
    driver.get(log_in_url)

    time.sleep(1)

    email_box = driver.find_element(By.ID, 'emailInput')
    password_box = driver.find_element(By.ID, 'passwordInput')
    login_button = driver.find_element(By.CSS_SELECTOR, 'button.btn.btn-primary.btn-xl.btn-block')

    email_box.send_keys(email)
    password_box.send_keys(password)
    time.sleep(0.5)
    login_button.click()

    time.sleep(2.5)

def edit_url(city, state, lat, long):
    d = str(date.today())
    url = f"https://app.utrsports.net/events/search?city={city}&state={state}&lat={lat}&long={long}&date={d}"
    return url

def collect_scores(all_scores):
    scores = []
    for score in all_scores:
        scores.append(score.text)
    return scores

def load_page(driver, url):
    driver.get(url)
    time.sleep(2)

def scroll_page(driver):
    SCROLL_PAUSE_TIME = 2
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

def scrape_utr_history(df, email, password, offset=0, stop=1, writer=None):
    start_time = time.time()
    timeout = 30  # 30 seconds timeout
    
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
    log_in_url = "https://app.utrsports.net/login"
    
    try:
        sign_in(driver, log_in_url, email, password)
        
        for index, row in df.iloc[offset:stop].iterrows():
            if time.time() - start_time > timeout:
                print("Reached 30-second timeout limit")
                break
                
            p_id = row['p_id']
            f_name = row['f_name']
            l_name = row['l_name']
            
            url = f"https://app.utrsports.net/profile/{p_id}/history"
            driver.get(url)
            time.sleep(2)
            
            try:
                # Wait for the history table to load
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "table"))
                )
                
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                table = soup.find('table')
                
                if table:
                    rows = table.find_all('tr')[1:]  # Skip header row
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) >= 2:
                            date = cols[0].text.strip()
                            utr = cols[1].text.strip()
                            writer.writerow([f_name, l_name, date, utr])
                            
            except Exception as e:
                print(f"Error processing profile {p_id}: {str(e)}")
                continue
                
    finally:
        driver.quit()

def scrape_player_matches(profile_ids, utr_history, matches, email, password, offset=0, stop=1, writer=None):
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
    log_in_url = "https://app.utrsports.net/login"
    
    try:
        sign_in(driver, log_in_url, email, password)
        
        for index, row in profile_ids.iloc[offset:stop].iterrows():
            p_id = row['p_id']
            f_name = row['f_name']
            l_name = row['l_name']
            
            url = f"https://app.utrsports.net/profile/{p_id}/matches"
            driver.get(url)
            time.sleep(2)
            
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "table"))
                )
                
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                table = soup.find('table')
                
                if table:
                    rows = table.find_all('tr')[1:]
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) >= 4:
                            date = cols[0].text.strip()
                            opponent = cols[1].text.strip()
                            score = cols[2].text.strip()
                            utr = cols[3].text.strip()
                            writer.writerow([f_name, l_name, date, opponent, score, utr])
                            
            except Exception as e:
                print(f"Error processing profile {p_id}: {str(e)}")
                continue
                
    finally:
        driver.quit() 