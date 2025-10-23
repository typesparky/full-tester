import pandas as pd
import requests
import time
import re
import asyncio
import aiohttp
from pathlib import Path
import json

# --- Configuration ---
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"
OUTPUT_DIR = "polymarket_data"

# This is now for Option 1: Keyword Search
KEYWORD_SETS = {
    "1": {"description": "Fed & Interest Rates", "query": "fed rate interest rates"},
    "2": {"description": "US Inflation (CPI)", "query": "inflation cpi"},
    "3": {"description": "US GDP", "query": "gdp"},
    "4": {"description": "US Jobs & Unemployment", "query": "jobs unemployment nonfarm"},
    "5": {"description": "Treasury & Yields", "query": "treasury yield"},
    "6": {"description": "Politics (US Election)", "query": "election biden trump democrat republican politics"},
    "7": {"description": "Crypto (Bitcoin, Ethereum, ETF)", "query": "bitcoin ethereum etf"},
    "8": {"description": "Geopolitics & War", "query": "war conflict ceasefire russia ukraine china taiwan israel gaza"}
}


# --- Helper Functions ---

async def fetch_all_tags(session):
    """Fetches all available tags from the Polymarket API."""
    print("\nFetching all available market tags...")
    try:
        async with session.get(f"{GAMMA_API_BASE}/tags") as response:
            if response.status != 200:
                print(f"Error: Failed to fetch tags. API returned status {response.status}")
                return None
            tags_data = await response.json()
            
            tags_list = []
            if isinstance(tags_data, list): tags_list = tags_data
            elif isinstance(tags_data, dict):
                for key in ['data', 'tags', 'results']:
                    if key in tags_data and isinstance(tags_data[key], list):
                        tags_list = tags_data[key]; break
            
            if not tags_list:
                print("Warning: Could not find a list of tags in the API response.")
                return None

            tag_lookup = {
                tag['label'].lower(): tag['id'] 
                for tag in tags_list 
                if isinstance(tag, dict) and 'label' in tag and 'id' in tag
            }

            if not tag_lookup:
                 print("Warning: Processed 0 valid tags from the API response.")
            
            print(f"Successfully fetched and processed {len(tag_lookup)} valid tags.")
            return tag_lookup
    except Exception as e:
        print(f"Error fetching tags: {e}")
        return None

async def search_events_by_keyword(session, query_string, include_closed=False):
    """Searches for events containing a specific keyword query."""
    print(f"Searching for events with query: '{query_string}' (Include closed: {include_closed})...")
    params = {'q': query_string.strip()}
    if include_closed: params['keep_closed_markets'] = 1
        
    try:
        async with session.get(f"{GAMMA_API_BASE}/public-search", params=params) as response:
            if response.status != 200: return []
            search_data = await response.json()
            return search_data.get('events', [])
    except aiohttp.ClientError as e:
        print(f"  - Error searching for query '{query_string}': {e}")
        return []

async def fetch_events_by_tag_id(session, tag_id, include_closed=False, limit=100):
    """Fetches event summaries for a given tag_id, handling pagination."""
    print(f"Fetching events for tag_id: '{tag_id}' (Include closed: {include_closed})...")
    all_events = []
    offset = 0
    while True:
        params = {"tag_id": tag_id, "limit": limit, "offset": offset}
        if not include_closed:
             params["closed"] = "false" # Only add 'closed=false' if we are NOT including historical
             
        try:
            async with session.get(f"{GAMMA_API_BASE}/events", params=params) as response:
                if response.status != 200: break
                events_page = await response.json()
                if not events_page: break
                all_events.extend(events_page)
                if len(events_page) < limit: break
                offset += limit
                await asyncio.sleep(0.5)
        except aiohttp.ClientError as e:
            print(f"  - Error fetching events for tag_id {tag_id}: {e}")
            break
    return all_events

async def fetch_detailed_market_by_slug(session, slug: str):
    """Fetches the full, detailed market object using its slug."""
    if not slug: return None
    url = f"{GAMMA_API_BASE}/markets/slug/{slug}"
    try:
        async with session.get(url) as response:
            if response.status != 200: return None
            return await response.json()
    except aiohttp.ClientError:
        return None

def get_single_price_history(token_id):
    """Fetches and processes price history for a single token."""
    try:
        params = {"market": token_id, "interval": "max", "fidelity": 1440}
        response = requests.get(f"{CLOB_API_BASE}/prices-history", params=params)
        response.raise_for_status()
        data = response.json().get("history", [])
        if not data: return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['t'], unit='s')
        df['price'] = df['p']
        df = df.set_index('timestamp')
        return df[['price']]
    except requests.exceptions.RequestException:
        return pd.DataFrame()

async def process_and_save_markets(valid_detailed_markets, sector_name, sector_folder_name):
    """Helper function to process the final list of markets and save their data."""
    output_path = Path(OUTPUT_DIR)
    sector_output_path = output_path / sector_folder_name
    sector_output_path.mkdir(exist_ok=True)
    
    market_manifest_log = []
    saved_files = 0
    
    print(f"\nFetching historical odds for {len(valid_detailed_markets)} markets (sorted by volume)...")
    
    for i, market in enumerate(valid_detailed_markets):
        market_id = market.get('id')
        question = market.get('question', 'unknown_market')
        volume = float(market.get('volume', '0'))
        slug = market.get('slug', re.sub(r'[^a-zA-Z0-9_]+', '', question.replace(' ', '_')).lower()[:50])
        
        print(f"\n({i+1}/{len(valid_detailed_markets)}) Processing: {question} (Volume: ${volume:,.0f})")

        log_entry = {
            "rank": i + 1, "sector": sector_name, "question": question,
            "volume": volume, "slug": slug, "filename": "N/A", "status": "Failed"
        }

        reconstructed_tokens = []
        try:
            outcomes = json.loads(market.get('outcomes', '[]'))
            clob_token_ids = json.loads(market.get('clobTokenIds', '[]'))
            if len(outcomes) == len(clob_token_ids):
                for outcome_name, token_id in zip(outcomes, clob_token_ids):
                    reconstructed_tokens.append({'outcome': outcome_name, 'token_id': token_id})
            else: 
                print("  - Warning: Mismatch between outcomes and token IDs.")
                market_manifest_log.append(log_entry)
                continue
        except (json.JSONDecodeError, TypeError):
            print("  - Warning: Could not parse 'outcomes' or 'clobTokenIds'.")
            market_manifest_log.append(log_entry)
            continue

        if not market_id or not reconstructed_tokens:
            print("  - Market is missing ID or tokens. Skipping.")
            log_entry["status"] = "Failed: Missing ID or Tokens"
            market_manifest_log.append(log_entry)
            continue

        market_history_df = pd.DataFrame()
        
        for token in reconstructed_tokens:
            token_id = token.get('token_id'); outcome = token.get('outcome', 'unknown_outcome').lower()
            if not token_id: continue
            print(f"  - Fetching history for outcome: '{outcome}'...")
            history_df = get_single_price_history(token_id)
            if not history_df.empty: market_history_df[outcome] = history_df['price']
            time.sleep(0.3)

        if not market_history_df.empty:
            market_history_df = market_history_df.sort_index()
            filename = f"{slug}_{market_id[:6]}.csv"
            filepath = sector_output_path / filename
            market_history_df.to_csv(filepath)
            print(f"  ✅ Saved data to: {filepath}")
            saved_files += 1
            log_entry["filename"] = f"{sector_folder_name}/{filename}"
            log_entry["status"] = "Saved"
        else:
            print("  - No historical data found for any token in this market.")
            log_entry["status"] = "Failed: No History Found"
        market_manifest_log.append(log_entry)
            
    if market_manifest_log:
        manifest_df = pd.DataFrame(market_manifest_log)
        manifest_df = manifest_df[['rank', 'sector', 'question', 'volume', 'slug', 'filename', 'status']]
        manifest_filename = f"_manifest_{sector_folder_name}.csv"
        manifest_filepath = output_path / manifest_filename
        manifest_df.to_csv(manifest_filepath, index=False)
        print("\n" + "="*50); print(f"📊 Successfully created market manifest:\n   {manifest_filepath}")
    
    print("\n" + "="*50); print("🎉 Fetching Complete!")
    print(f"Successfully saved historical data for {saved_files} markets.")
    print(f"Data is located in the '{output_path}' directory."); print("="*50)

# --- Main Workflows ---

async def run_keyword_search_workflow(session):
    """Guides user through the pre-defined keyword search workflow."""
    print("\n--- 1: Fetch Markets by Keyword Search ---")
    for key, option in KEYWORD_SETS.items():
        print(f"  {key}: {option['description']}")
    print("\n  b: Back to main menu")
    print("="*60)
    
    choice = input("Enter your choice: ").strip()
    if choice.lower() == 'b': return
    if choice not in KEYWORD_SETS:
        print("\n❌ Invalid choice. Please try again.")
        return

    selected_option = KEYWORD_SETS[choice]
    keywords_to_search = selected_option['keywords'] # Get the list of keywords
    sector_name = selected_option['description']
    sector_folder_name = re.sub(r'[^a-zA-Z0-9_]+', '', sector_name.replace(' ', '_')).lower()
    
    include_closed_input = input("Include historical (closed) markets? (y/n) [default: n]: ").strip().lower()
    include_closed = True if include_closed_input == 'y' else False
    
    print(f"\n--- Fetching markets for: {selected_option['description']} ---")
    
    # Step 1: Search for events
    print("\nSearching for events matching your keywords...")
    tasks = [search_events_by_keyword(session, keyword, include_closed=include_closed) for keyword in keywords_to_search]
    event_results_by_keyword = await asyncio.gather(*tasks)
    
    all_events = [event for event_list in event_results_by_keyword for event in event_list]
    unique_events = {event['id']: event for event in all_events if event.get('id')}.values()
    print(f"\nFound {len(unique_events)} unique, relevant events.")

    # Step 2: Extract market SLUGS
    market_slugs = set()
    for event in unique_events:
        for market in event.get('markets', []):
            if market.get('slug'): market_slugs.add(market['slug'])
    print(f"Extracted {len(market_slugs)} unique market slugs from events.")

    # Step 3: Fetch full details
    print("\nFetching full details for each relevant market...")
    detail_tasks = [fetch_detailed_market_by_slug(session, slug) for slug in market_slugs]
    detailed_markets_results = await asyncio.gather(*detail_tasks)
    
    valid_detailed_markets = [market for market in detailed_markets_results if market]
    print(f"Successfully fetched details for {len(valid_detailed_markets)} markets.")

    if not valid_detailed_markets:
        print("No detailed market data could be fetched. Exiting.")
        return

    # Step 4: Sort by volume
    valid_detailed_markets.sort(key=lambda m: float(m.get('volume', '0')), reverse=True)
    
    await process_and_save_markets(valid_detailed_markets, sector_name, sector_folder_name)

async def run_tag_based_workflow(session):
    """Guides user through the interactive tag-filtering workflow."""
    print("\n--- 2: Fetch Markets by Tag ---")
    
    # Step 1: Get all tags and let user choose
    tag_lookup = await fetch_all_tags(session)
    if not tag_lookup:
        print("Aborting: Could not retrieve a valid tag list from Polymarket.")
        return
        
    print("\n--- Available Tags ---")
    sorted_tags = sorted(list(tag_lookup.keys()))
    col_width = max(len(tag) for tag in sorted_tags) + 4
    num_cols = 4
    for i in range(0, len(sorted_tags), num_cols):
        print("".join(tag.ljust(col_width) for tag in sorted_tags[i:i+num_cols]))
    print("="*60)
    
    tag_input = input("Enter the tags you want to search for, separated by commas (e.g., gdp, world affairs): ").strip().lower()
    if not tag_input:
        print("No tags entered. Aborting.")
        return
        
    chosen_tags = [t.strip() for t in tag_input.split(',')]
    target_tag_ids = []
    for tag in chosen_tags:
        if tag in tag_lookup:
            target_tag_ids.append(tag_lookup[tag])
        else:
            print(f"Warning: Tag '{tag}' not found in the list. Skipping it.")
            
    if not target_tag_ids:
        print("No valid tags were selected. Aborting.")
        return
        
    sector_name = f"Custom Tag Search ({tag_input})"
    sector_folder_name = f"custom_{re.sub(r'[^a-zA-Z0-9_]+', '', tag_input.replace(' ', '_')).lower()[:20]}"
    
    include_closed_input = input("Include historical (closed) markets? (y/n) [default: n]: ").strip().lower()
    include_closed = True if include_closed_input == 'y' else False
    
    # Step 2: Fetch events for the chosen tags
    print(f"\nFetching events for {len(target_tag_ids)} selected tag(s)...")
    tasks = [fetch_events_by_tag_id(session, tag_id, include_closed=include_closed) for tag_id in target_tag_ids]
    event_results_by_tag = await asyncio.gather(*tasks)
    
    all_events = [event for event_list in event_results_by_tag for event in event_list]
    unique_events = {event['id']: event for event in all_events if event.get('id')}.values()
    print(f"\nFound {len(unique_events)} unique, relevant events.")

    # Step 3: Extract market SLUGS
    market_slugs = set()
    for event in unique_events:
        for market in event.get('markets', []):
            if market.get('slug'): market_slugs.add(market['slug'])
    print(f"Extracted {len(market_slugs)} unique market slugs from events.")

    # Step 4: Fetch full details
    print("\nFetching full details for each relevant market...")
    detail_tasks = [fetch_detailed_market_by_slug(session, slug) for slug in market_slugs]
    detailed_markets_results = await asyncio.gather(*detail_tasks)
    
    valid_detailed_markets = [market for market in detailed_markets_results if market]
    print(f"Successfully fetched details for {len(valid_detailed_markets)} markets.")

    if not valid_detailed_markets:
        print("No detailed market data could be fetched. Exiting.")
        return

    # Step 5: Sort by volume
    valid_detailed_markets.sort(key=lambda m: float(m.get('volume', '0')), reverse=True)
    
    await process_and_save_markets(valid_detailed_markets, sector_name, sector_folder_name)

async def main():
    """Main menu loop."""
    async with aiohttp.ClientSession() as session:
        while True:
            print("\n" + "="*60)
            print("🚀 Polymarket Data Fetcher - Main Menu")
            print("="*60)
            print("  1: Fetch Markets by Keyword Search (Recommended)")
            print("  2: Fetch Markets by Tag (Advanced)")
            print("\n  q: Quit")
            print("="*60)
            
            choice = input("Enter your choice: ").strip()
            
            if choice == '1':
                await run_keyword_search_workflow(session)
            elif choice == '2':
                await run_tag_based_workflow(session)
            elif choice.lower() == 'q':
                print("Exiting.")
                break
            else:
                print("\n❌ Invalid choice. Please try again.")
            
            input("\nPress Enter to return to the main menu...")

if __name__ == "__main__":
    asyncio.run(main())

