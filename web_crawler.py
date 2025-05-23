import requests
from bs4 import BeautifulSoup
import time
import csv
import os
from datetime import datetime
from itertools import combinations, permutations

class GooglePlayCrawler:
    def __init__(self):
        # Base search terms
        base_terms = [
            "industrial", "control", "factory", "automation",
            "hmi", "interface", "iot", "industrial", "scada system",
            "automation", "modbus", "profinet", "ethernet/ip", "ethernet/tcp",
            "SCADA", "HMI", "PLC", "DCS", "Distributed control system",
            "Power", "Water", "Remote", "process", "monitor", "alert", "sematic",
            "Allen-Bradley", "Rockwell", "Cognex", "Omron", "Panasonic", "Siemens", "GE"    
        ]
        
        # Remove duplicates and normalize terms
        base_terms = list(set(term.lower() for term in base_terms))
        
        # Generate all possible combinations of 2 or more words
        self.search_terms = set()
        
        # Add all individual terms
        self.search_terms.update(base_terms)
        
        # Generate combinations of 2 words
        for term1, term2 in permutations(base_terms, 2):
            if term1 != term2:  # Avoid duplicates
                combined = f"{term1} {term2}"
                self.search_terms.add(combined)
        
        # Generate combinations of 3 words
        for term1, term2, term3 in permutations(base_terms, 3):
            if term1 != term2 and term2 != term3 and term1 != term3:  # Avoid duplicates
                combined = f"{term1} {term2} {term3}"
                self.search_terms.add(combined)
        
        # Generate combinations of 4 words
        for term1, term2, term3, term4 in permutations(base_terms, 4):
            if len(set([term1, term2, term3, term4])) == 4:  # Ensure all terms are unique
                combined = f"{term1} {term2} {term3} {term4}"
                self.search_terms.add(combined)
        
        # Add some specific meaningful combinations that might be missed by permutations
        specific_combinations = [
            "industrial control system",
            "factory automation system",
            "process control system",
            "distributed control system",
            "power monitoring system",
            "water treatment control",
            "remote monitoring system",
            "industrial automation control",
            "SCADA monitoring system",
            "HMI interface system",
            "PLC control system",
            "DCS control system",
            "modbus communication",
            "profinet network",
            "ethernet/ip protocol",
            "ethernet/tcp communication",
            "Allen-Bradley PLC",
            "Rockwell automation",
            "Siemens automation",
            "GE automation",
            "Omron PLC",
            "Panasonic automation",
            "Cognex vision system"
        ]
        
        self.search_terms.update(specific_combinations)
        
        # Convert set back to list and sort for consistent ordering
        self.search_terms = sorted(list(self.search_terms))
        
        # Create a timestamped filename for the CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f"industrial_control_apps_{timestamp}.csv"
        
        # Initialize unique apps tracking
        self.unique_apps = set()
        #self.max_unique_apps = 10000
        self.download_count = 0
        #self.max_downloads = 
    
    def crawl(self):
        with open(self.csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Search Term', 'App Title', 'Developer', 'Package Name', 'Play Store Link'])
            for term in self.search_terms:
                try:
                    max_retries = 3 
                    retry_delay = 60 


                    for retry in range(max_retries):
                        try:
                            search_url = "https://play.google.com/store/search?q=" + \
                                        term.replace(" ", "+") + "&c=apps"
                            
                            headers = {
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/108.0.0.0 Safari/537.36",
                                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                                "Accept-Language": "en-US,en;q=0.9",
                                "Accept-Encoding": "gzip, deflate, br",
                                "Connection": "keep-alive",
                            }
                            
                            response = requests.get(search_url, headers=headers, timeout=10)
                            response.raise_for_status()
                            
                            # Print raw HTML for debugging
                            print("\nRaw HTML snippet:")
                            print(response.text[:1000])
                            
                            soup = BeautifulSoup(response.text, 'html.parser')
                            
                            # Try multiple selectors for app cards
                            app_elements = soup.select("div.VfPpkd-aGsRMb") or \
                                        soup.select("div.card-content") or \
                                        soup.select("div.WHE7ib") or \
                                        soup.select("div[jscontroller][role='article']")
                            
                            print(f"\nSearching for term: {term}")
                            print(f"Found {len(app_elements)} potential app elements")
                            
                            if app_elements:
                                # Try different title selectors
                                title_selectors = [
                                    "h2.Epkrse", 
                                    "span.DdYX5",
                                    "div.b8cIId > div",
                                    "div.WsMG1c"
                                ]
                                
                                # Try different developer selectors
                                developer_selectors = [
                                    "div.KoLSrc",
                                    "span.wMUdtb",
                                    "div.KZnDLd"
                                ]
                                
                                for app in app_elements:
                                    # Try each title selector
                                    title_element = None
                                    for selector in title_selectors:
                                        title_element = app.select_one(selector)
                                        if title_element:
                                            break
                                            
                                    # Try each developer selector
                                    developer_element = None
                                    for selector in developer_selectors:
                                        developer_element = app.select_one(selector)
                                        if developer_element:
                                            break
                                    
                                    # Find link element
                                    link_element = app.find_parent('a') or app.select_one('a')
                                    
                                    if title_element and link_element:
                                        if 'href' in link_element.attrs:
                                            href = link_element['href']
                                            if 'id=' in href:
                                                package_name = href.split('id=')[-1].split('&')[0]
                                                play_store_link = f"https://play.google.com{href}"
                                                
                                                title = title_element.text.strip()
                                                developer = developer_element.text.strip() if developer_element else "Unknown"
                                                
                                                if package_name not in self.unique_apps:
                                                    self.unique_apps.add(package_name)
                                                    self.download_count += 1
                                                    
                                                    csv_writer.writerow([term, title, developer, package_name, play_store_link])
                                                    
                                                    print(f"\nFound app: {title}")
                                                    print(f"Developer: {developer}")
                                                    print(f"Package: {package_name}")
                                                    print(f"Link: {play_store_link}")
                            
                            print(f"Completed search for '{term}'. Found {self.download_count} apps so far.")
                            time.sleep(5)  # Sleep to avoid hitting the server too hard
                            break # Exit retry loop if successful

                        except requests.exceptions.HTTPError as e:
                            if e.response.status_code == 429:  # Too Many Requests
                                if retry < max_retries - 1:  # Don't sleep on last retry
                                    print(f"\nRate limited. Waiting {retry_delay} seconds before retry {retry + 1}/{max_retries}")
                                    time.sleep(retry_delay)
                                    retry_delay *= 2  # Double the delay for next retry
                                continue
                            raise  # Re-raise other HTTP errors

                except Exception as e:
                    print(f"Error searching for {term}: {str(e)}")
                    csv_writer.writerow([term, f"ERROR: {str(e)}", "", "", ""])
if __name__ == "__main__":
    crawler = GooglePlayCrawler()
    crawler.crawl()
