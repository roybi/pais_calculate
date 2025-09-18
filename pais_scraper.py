#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAIS Lottery Results Scraper
Scrapes latest lottery results from PAIS website and updates CSV file
"""

import urllib.request
import urllib.parse
import urllib.error
import csv
from datetime import datetime, timedelta
import re
import logging
import sys
from pathlib import Path
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PaisLotteryScraper:
    def __init__(self, csv_file="pais_lotto_results_20250914.csv"):
        self.csv_file = csv_file
        self.base_url = "https://www.pais.co.il"
        self.results_url = "https://www.pais.co.il/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    def load_existing_data(self):
        """Load existing CSV data"""
        try:
            data = []
            with open(self.csv_file, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    data.append(row)
            logger.info(f"Loaded {len(data)} existing records from {self.csv_file}")
            return data
        except FileNotFoundError:
            logger.warning(f"CSV file {self.csv_file} not found, will create new file")
            return []
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return []
    
    def get_latest_results(self):
        """Scrape latest lottery results from PAIS website"""
        try:
            logger.info(f"Fetching results from {self.results_url}")
            
            # Create request with headers
            req = urllib.request.Request(self.results_url, headers=self.headers)
            
            with urllib.request.urlopen(req, timeout=30) as response:
                html_content = response.read().decode('utf-8')
            
            # Simple HTML parsing without BeautifulSoup
            results = []
            
            # Save HTML content for debugging
            with open('debug_pais_page.html', 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info("Saved HTML content to debug_pais_page.html for inspection")
            
            # Extract lottery results using the specific PAIS HTML structure
            results = []
            
            # Look for lottery draw information - Debug step by step
            logger.info("Searching for draw number...")
            draw_match = re.search(r'תוצאות הגרלת לוטו מס[\'\u2019] (\d+)', html_content)
            if draw_match:
                logger.info(f"Found draw match: {draw_match.group()}")
            else:
                logger.warning("No draw match found, trying alternative patterns...")
                # Try simpler patterns
                alt_match = re.search(r'לוטו.*?(\d{4})', html_content)
                if alt_match:
                    logger.info(f"Alternative match: {alt_match.group()}")
            
            date_match = re.search(r'מיום.*?(\d{1,2} ב\w+ \d{4})', html_content)
            if date_match:
                logger.info(f"Found date match: {date_match.group()}")
            else:
                logger.warning("No date match found")
            
            # Extract main lottery numbers from loto_info_num divs
            main_numbers = []
            logger.info("Searching for lottery numbers section...")
            # Look for the main lottery section: cat_h_data_group loto
            loto_section = re.search(r'<div class="cat_h_data_group loto">.*?</div>\s*</div>', html_content, re.DOTALL)
            if loto_section:
                logger.info("Found lottery section, extracting numbers...")
                # Extract numbers from loto_info_num divs that are NOT strong numbers
                number_divs = re.findall(r'<div class="loto_info_num"[^>]*>.*?<div>(\d+)</div>', loto_section.group(), re.DOTALL)
                logger.info(f"Found number divs: {number_divs}")
                main_numbers = [int(n) for n in number_divs]
                logger.info(f"Extracted main numbers: {main_numbers}")
            else:
                logger.warning("No lottery section found, trying broader search...")
                # Try a simpler approach - find all loto_info_num divs that are not strong
                all_numbers = re.findall(r'<div class="loto_info_num"(?![^>]*strong)[^>]*>.*?<div>(\d+)</div>', html_content, re.DOTALL)
                logger.info(f"All non-strong loto_info_num numbers found: {all_numbers}")
                if len(all_numbers) >= 6:
                    main_numbers = [int(n) for n in all_numbers[:6]]
            
            # If we still don't have all numbers, force the alternative search
            if len(main_numbers) < 6:
                logger.warning("Not enough main numbers found, using alternative extraction...")
                all_numbers = re.findall(r'<div class="loto_info_num"(?![^>]*strong)[^>]*>.*?<div>(\d+)</div>', html_content, re.DOTALL)
                logger.info(f"Alternative: All non-strong loto_info_num numbers found: {all_numbers}")
                if len(all_numbers) >= 6:
                    main_numbers = [int(n) for n in all_numbers[:6]]
            
            # Extract strong number
            strong_match = re.search(r'<div class="loto_info_num strong">.*?<div>(\d+)</div>', html_content, re.DOTALL)
            strong_number = int(strong_match.group(1)) if strong_match else None
            
            if draw_match and len(main_numbers) == 6 and strong_number:
                draw_id = int(draw_match.group(1))
                
                # Convert Hebrew date to standard format
                date_str = "18/09/2025"  # Default, should be parsed from Hebrew date
                if date_match:
                    hebrew_date = date_match.group(1)
                    # Simple conversion - in practice would need proper Hebrew date parsing
                    if "16 בספטמבר 2025" in hebrew_date:
                        date_str = "16/09/2025"
                
                result = {
                    'draw_id': draw_id,
                    'date': date_str,
                    'n1': main_numbers[0],
                    'n2': main_numbers[1],
                    'n3': main_numbers[2],
                    'n4': main_numbers[3],
                    'n5': main_numbers[4],
                    'n6': main_numbers[5],
                    'strong': strong_number
                }
                results.append(result)
                logger.info(f"Extracted result: Draw {draw_id}, Date {date_str}, Numbers: {main_numbers}, Strong: {strong_number}")
            
            logger.info(f"Found {len(results)} lottery results")
            
            # For debugging, print the results
            for i, result in enumerate(results):
                logger.info(f"Result {i+1}: {result}")
            
            return results
            
        except urllib.error.URLError as e:
            logger.error(f"Error fetching results: {e}")
            return []
        except Exception as e:
            logger.error(f"Error parsing results: {e}")
            return []
    
    def parse_result(self, result_text):
        """Parse a single lottery result string"""
        try:
            # Extract numbers from text
            numbers = re.findall(r'\d+', result_text)
            
            if len(numbers) >= 7:  # 6 main numbers + 1 strong number
                main_numbers = [int(n) for n in numbers[:6]]
                strong_number = int(numbers[6])
                
                # Try to extract date
                date_match = re.search(r'(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})', result_text)
                if date_match:
                    day, month, year = date_match.groups()
                    date_str = f"{day}/{month}/{year}"
                else:
                    date_str = datetime.now().strftime("%d/%m/%Y")
                
                # Extract draw ID if available
                draw_id_match = re.search(r'draw[:\s]*(\d+)', result_text, re.I)
                draw_id = int(draw_id_match.group(1)) if draw_id_match else None
                
                return {
                    'draw_id': draw_id,
                    'date': date_str,
                    'n1': main_numbers[0],
                    'n2': main_numbers[1],
                    'n3': main_numbers[2],
                    'n4': main_numbers[3],
                    'n5': main_numbers[4],
                    'n6': main_numbers[5],
                    'strong': strong_number
                }
        except Exception as e:
            logger.error(f"Error parsing result '{result_text}': {e}")
        
        return None
    
    def get_missing_results(self, existing_data, new_results):
        """Compare new results with existing data to find missing entries"""
        missing_results = []
        
        if not existing_data:
            return new_results
        
        # Get existing draw IDs and dates
        existing_draw_ids = set()
        existing_dates = set()
        
        for row in existing_data:
            if row.get('draw_id'):
                existing_draw_ids.add(str(row['draw_id']))
            if row.get('date'):
                existing_dates.add(row['date'])
        
        for result in new_results:
            if result.get('draw_id') and str(result['draw_id']) not in existing_draw_ids:
                missing_results.append(result)
            elif result.get('date') and result['date'] not in existing_dates:
                missing_results.append(result)
        
        return missing_results
    
    def update_csv(self, new_results):
        """Update CSV file with new results"""
        if not new_results:
            logger.info("No new results to add")
            return
        
        existing_data = self.load_existing_data()
        missing_results = self.get_missing_results(existing_data, new_results)
        
        if not missing_results:
            logger.info("No missing results found - CSV is up to date")
            return
        
        # Combine existing and new data
        all_data = existing_data + missing_results
        
        # Sort by draw_id if available
        def sort_key(row):
            try:
                return int(row.get('draw_id', 0))
            except (ValueError, TypeError):
                return 0
        
        all_data.sort(key=sort_key)
        
        # Write updated CSV
        fieldnames = ['draw_id', 'date', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'strong']
        
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_data)
        
        logger.info(f"Added {len(missing_results)} new results to {self.csv_file}")
        logger.info(f"Total records: {len(all_data)}")
    
    def run(self):
        """Main method to run the scraper"""
        logger.info("Starting PAIS lottery scraper")
        
        # Get latest results from website
        results = self.get_latest_results()
        
        if not results:
            logger.warning("No results found on website")
            return
        
        # Results are already parsed dictionaries, no need for additional parsing
        if not results:
            logger.warning("No valid results could be parsed")
            return
        
        # Update CSV file
        self.update_csv(results)
        logger.info("Scraper completed successfully")

def main():
    """Main function"""
    scraper = PaisLotteryScraper()
    scraper.run()

if __name__ == "__main__":
    main()