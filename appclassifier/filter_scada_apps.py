import os
import json
import csv
import time
from typing import Set, List, Tuple

# Simple string-based patterns (no regex at all)
ICS_TERMS = {
    # Protocols
    "modbus": ["modbus", "modbus tcp", "modbus rtu", "modbus ascii"],
    "opcua": ["opc ua", "opc-ua", "opcua", "opc.tcp://"],
    "opc": ["opc da", "opc hda", "opc ae"],
    "ethernetip": ["ethernet ip", "ethernet/ip", "enip", "ethernet-ip"],
    "profinet": ["profinet"],
    "profibus": ["profibus"], 
    "ethercat": ["ethercat"],
    "dnp3": ["dnp3", "dnp 3"],
    "bacnet": ["bacnet", "bacnet ip", "bacnet/ip"],
    "bac0": ["bac0"],
    "iec": ["iec 60870", "iec 104", "iec 61850", "iec 61131"],
    "mms": ["mms protocol"],
    "canopen": ["canopen", "can open"],
    "j1939": ["j1939"],
    "hart": ["hart protocol", "hart communication"],
    "powerlink": ["powerlink"],
    "sercos": ["sercos"],
    "lonworks": ["lonworks", "lontalk"],
    "mbus": ["m-bus", "m bus", "mbus"],
    
    # Systems & Hardware
    "plc": ["plc", "plcs", "programmable logic controller"],
    "rtu": ["rtu", "rtus", "remote terminal unit"],
    "dcs": ["dcs", "distributed control system"],
    "hmi": ["hmi", "hmis", "human machine interface"],
    "scada": ["scada", "supervisory control"],
    "vfd": ["vfd", "variable frequency drive"],
    "inverter": ["inverter", "drive controller"],
    
    # Programming
    "ladder": ["ladder logic", "ladder diagram"],
    "structured_text": ["structured text", "st programming"],
    "function_block": ["function block", "function blocks"],
    "sfc": ["sequential function chart"],
    
    # Vendors - Siemens
    "siemens": ["siemens", "s7-1200", "s7-1500", "s7-300", "s7-400", "s7comm", "step 7", "step7", "wincc"],
    
    # Vendors - Allen Bradley / Rockwell
    "allen_bradley": ["allen bradley", "allen-bradley", "rockwell", "factorytalk", "logix", "rslogix"],
    
    # Vendors - Schneider
    "schneider": ["schneider", "modicon", "m340", "m580", "ecostruxure"],
    
    # Vendors - Omron
    "omron": ["omron", "cp1h", "cp1l", "cs1", "nx102", "nx701"],
    
    # Vendors - Mitsubishi
    "mitsubishi": ["mitsubishi", "fx3u", "fx5u", "q02u", "q03u", "gx works", "gx-works"],
    
    # Vendors - Others
    "beckhoff": ["beckhoff", "twincat", "ads route", "ams"],
    "wago": ["wago", "codesys", "3s-codesys"],
    "abb": ["abb", "800xa", "freelance"],
    "ge_fanuc": ["ge fanuc", "ge-fanuc"],
    "yokogawa": ["yokogawa", "centum"],
    "honeywell": ["honeywell", "experion"],
    "wonderware": ["wonderware", "intouch"],
    "citect": ["citect", "vijeo citect"],
    "ifix": ["ifix", "ge ifix"],
    "movicon": ["movicon"],
    "clearscada": ["clearscada"],
    "zenon": ["zenon"],
    "sparkplug": ["sparkplug b", "sparkplug-b"],
    
    # Ports
    "ports": ["port 502", ":502", "port 4840", ":4840", "port 44818", ":44818", 
              "port 2222", ":2222", "port 20000", ":20000", "port 2404", ":2404",
              "port 102", ":102", "port 47808", ":47808"],
    
    # Serial Communication
    "serial": ["rs232", "rs485", "rs-232", "rs-485", "serial port", "com port"],
    "bluetooth": ["bluetooth", "bluetooth le", "ble"],
}

# Negative terms that should exclude results
NEGATIVE_TERMS = [
    "factory reset", "plant care", "hydroponic", "smart home", "home automation",
    "arduino", "raspberry pi", "farming simulator", "garden", "3d printer",
    "minecraft", "game", "gaming", "entertainment", "music", "video player"
]

def normalize_text(text: str) -> str:
    """Simple text normalization without regex"""
    if not text:
        return ""
    
    # Simple HTML tag removal (basic approach)
    result = text
    while '<' in result and '>' in result:
        start = result.find('<')
        end = result.find('>', start)
        if start != -1 and end != -1:
            result = result[:start] + ' ' + result[end+1:]
        else:
            break
    
    # Basic entity decoding
    result = (result.replace('&amp;', '&')
                   .replace('&lt;', '<')
                   .replace('&gt;', '>')
                   .replace('&quot;', '"')
                   .replace('&#39;', "'")
                   .replace('&nbsp;', ' '))
    
    # Normalize whitespace
    result = ' '.join(result.split())
    
    return result.lower()

def find_ics_matches(text: str) -> Set[str]:
    """Find ICS terms using simple string matching"""
    if not text:
        return set()
    
    # Normalize text
    normalized_text = normalize_text(text)
    
    # Check for negative terms first
    for neg_term in NEGATIVE_TERMS:
        if neg_term in normalized_text:
            return set()  # Skip if negative term found
    
    matches = set()
    
    # Check each category of terms
    for category, terms in ICS_TERMS.items():
        for term in terms:
            if term.lower() in normalized_text:
                matches.add(f"{category}_{term.replace(' ', '_').replace('-', '_').replace('/', '_').replace(':', '_port')}")
                break  # Only count each category once
    
    return matches

def collect_text_fields(data: dict) -> str:
    """Collect all relevant text fields from app data"""
    fields = []
    
    # Basic fields
    fields.append(data.get('title', ''))
    fields.append(data.get('descriptionHtml', ''))
    fields.append(data.get('descriptionShort', ''))
    fields.append(data.get('creator', ''))
    
    # App details
    app_details = data.get('details', {}).get('appDetails', {})
    if app_details:
        fields.append(app_details.get('developerName', ''))
        fields.append(app_details.get('developerWebsite', ''))
    
    # Related links
    related_links = data.get('relatedLinks', {})
    if related_links:
        fields.append(related_links.get('privacyPolicyUrl', ''))
    
    return ' '.join(field for field in fields if field)

def process_jsonl_simple(jsonl_path: str, output_base: str, progress_every: int = 50000):
    """Process JSONL using simple string matching only"""
    print(f"Processing {jsonl_path} with simple string matching...")
    
    # Output files
    low_ics_path = output_base.replace('.csv', '_simple_low.csv')
    medium_ics_path = output_base.replace('.csv', '_simple_medium.csv') 
    high_ics_path = output_base.replace('.csv', '_simple_high.csv')
    
    print(f"Low ICS output -> {low_ics_path}")
    print(f"Medium ICS output -> {medium_ics_path}")
    print(f"High ICS output -> {high_ics_path}")
    
    stats = {
        'total_processed': 0,
        'total_matched': 0,
        'low_count': 0,
        'medium_count': 0,
        'high_count': 0
    }
    
    seen_packages = set()
    start_time = time.time()
    
    fieldnames = ['packageName', 'title', 'descriptionHtml', 'matched_keywords', 'keyword_count']
    
    try:
        with open(low_ics_path, 'w', newline='', encoding='utf-8') as low_file, \
             open(medium_ics_path, 'w', newline='', encoding='utf-8') as medium_file, \
             open(high_ics_path, 'w', newline='', encoding='utf-8') as high_file:
            
            low_writer = csv.DictWriter(low_file, fieldnames=fieldnames)
            medium_writer = csv.DictWriter(medium_file, fieldnames=fieldnames)
            high_writer = csv.DictWriter(high_file, fieldnames=fieldnames)
            
            # Write headers
            low_writer.writeheader()
            medium_writer.writeheader()
            high_writer.writeheader()
            
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line)
                        
                        # Extract package info
                        details = data.get('details', {}).get('appDetails', {})
                        package_name = details.get('packageName', '')
                        
                        if not package_name:
                            continue
                            
                        if package_name in seen_packages:
                            continue
                            
                        title = data.get('title', '')
                        description_html = data.get('descriptionHtml', '')
                        
                        if not description_html:
                            continue
                        
                        stats['total_processed'] += 1
                        
                        # Collect all text fields
                        all_text = collect_text_fields(data)
                        
                        # Find matches using simple string matching
                        matches = find_ics_matches(all_text)
                        
                        if matches:
                            seen_packages.add(package_name)
                            stats['total_matched'] += 1
                            keyword_count = len(matches)
                            
                            row_data = {
                                'packageName': package_name,
                                'title': title,
                                'descriptionHtml': description_html,
                                'matched_keywords': ','.join(sorted(matches)),
                                'keyword_count': keyword_count,
                            }
                            
                            # Categorize by score
                            if keyword_count < 3:
                                low_writer.writerow(row_data)
                                stats['low_count'] += 1
                            elif 4 <= keyword_count <= 6:
                                medium_writer.writerow(row_data)
                                stats['medium_count'] += 1
                            elif keyword_count > 7:
                                high_writer.writerow(row_data)
                                stats['high_count'] += 1
                        
                        # Progress reporting
                        if stats['total_processed'] % progress_every == 0:
                            elapsed = time.time() - start_time
                            rate = stats['total_processed'] / elapsed if elapsed > 0 else 0
                            
                            print(f"Processed {stats['total_processed']:,} | "
                                  f"matched {stats['total_matched']:,} | "
                                  f"low: {stats['low_count']} | "
                                  f"medium: {stats['medium_count']} | "
                                  f"high: {stats['high_count']} | "
                                  f"rate: {rate:.1f}/sec")
                            
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f"Error on line {line_num}: {e}")
                        continue
                        
    except Exception as e:
        print(f"Fatal error: {e}")
        raise
    
    # Final stats
    elapsed = time.time() - start_time
    print(f"\n=== FINAL STATS (Simple String Matching) ===")
    print(f"Total processed: {stats['total_processed']:,}")
    print(f"Total matched:   {stats['total_matched']:,}")
    print(f"Low ICS (< 3):   {stats['low_count']:,} -> {low_ics_path}")
    print(f"Medium ICS (4-6): {stats['medium_count']:,} -> {medium_ics_path}")
    print(f"High ICS (> 7):   {stats['high_count']:,} -> {high_ics_path}")
    print(f"Total time: {elapsed:.2f}s | Rate: {stats['total_processed']/elapsed:.1f} entries/sec")

def main():
    """Main function for simple string-based processing"""
    print("=== Simple String-Based SCADA/ICS Filter (No Regex) ===")
    
    input_jsonl = "gp-metadata-full.jsonl"
    output_base = "ics_matches_simple.csv"
    
    if not os.path.exists(input_jsonl):
        print(f"Error: Input file '{input_jsonl}' not found.")
        return
    
    # Test the matching function first
    print("\n=== Testing String Matching ===")
    test_cases = [
        "This app connects to Siemens S7-1200 PLC via Modbus TCP",
        "Control your factory with OPC UA and Profinet communication",
        "Smart home automation with Arduino",  # Should be excluded
        "Industrial HMI for Schneider Modicon M340"
    ]
    
    for i, test_text in enumerate(test_cases):
        matches = find_ics_matches(test_text)
        print(f"Test {i+1}: {len(matches)} matches - {sorted(matches) if matches else 'EXCLUDED/NO MATCH'}")
    
    try:
        process_jsonl_simple(input_jsonl, output_base)
        print(f"\n=== SUCCESS ===")
        print(f"Files created with '_simple' suffix")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()