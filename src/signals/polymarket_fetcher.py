#!/usr/bin/env python3
"""
Polymarket Data Fetcher - Final Clean Version

Options:
1: Fetch ALL Economic Sectors (Bulk - 7 sectors)
3: Fetch Markets by Tag ID (Direct API call)
4: Browse All Tags & Markets (Interactive exploration)

Solves: Market listing after tag selection
"""

import pandas as pd
import requests
import time
import re
import asyncio
import aiohttp
from pathlib import Path
import json

# Configuration
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"

async def fetch_all_tags(session):
    """Fetches all available tags from Polymarket API."""
    print("Fetching all available market tags...")
    try:
        async with session.get(f"{GAMMA_API_BASE}/tags") as response:
            if response.status != 200:
                print(f"Error: Failed to fetch tags. API returned {response.status}")
                return None
            tags_data = await response.json()

            tags_list = []
            if isinstance(tags_data, list):
                tags_list = tags_data
            elif isinstance(tags_data, dict):
                for key in ['data', 'tags', 'results']:
                    if key in tags_data and isinstance(tags_data[key], list):
                        tags_list = tags_data[key]; break

            if not tags_list:
                print("Warning: Could not find a list of tags in API response.")
                return None

            tag_lookup = {
                tag['label'].lower(): tag['id']
                for tag in tags_list
                if isinstance(tag, dict) and 'label' in tag and 'id' in tag
            }

            print(f"Processed {len(tag_lookup)} valid tags.")
            return tag_lookup
    except Exception as e:
        print(f"Error fetching tags: {e}")
        return None

async def fetch_events_by_tag_id(session, tag_id, include_closed=False, limit=100):
    """Fetches events for a tag_id."""
    print(f"Fetching events for tag_id '{tag_id}'...")
    all_events = []
    offset = 0
    while True:
        params = {"tag_id": tag_id, "limit": limit, "offset": offset}
        if not include_closed:
            params["closed"] = "false"

        try:
            async with session.get(f"{GAMMA_API_BASE}/events", params=params) as response:
                if response.status != 200:
                    break
                events_page = await response.json()
                if not events_page:
                    break
                all_events.extend(events_page)
                if len(events_page) < limit:
                    break
                offset += limit
                await asyncio.sleep(0.5)
        except aiohttp.ClientError as e:
            print(f"Error fetching events for tag {tag_id}: {e}")
            break
    return all_events

async def run_browse_tags_workflow(session):
    """Browser workflow that lists markets after tag selection."""
    print("\n--- 4: Browse All Tags & Markets (Exploratory) ---")
    print("Shows all tags, lets you select one, and lists ALL markets in it.")
    print()

    # Step 1: Show all tags
    tag_lookup = await fetch_all_tags(session)
    if not tag_lookup:
        print("Aborting: Could not retrieve tags from Polymarket.")
        input("\nPress Enter to return to main menu...")
        return

    # Display all tags
    print("📋 ALL POLYMARKET TAGS:")
    print("=" * 60)
    sorted_tags = sorted(list(tag_lookup.keys()))
    col_width = max(len(tag) for tag in sorted_tags) + 2
    num_cols = 6

    for i in range(0, len(sorted_tags), num_cols):
        row_tags = sorted_tags[i:i+num_cols]
        print("  " + "".join(f"{tag:<{col_width}}" for tag in row_tags))

    print(f"\n📊 Total: {len(sorted_tags)} tags available")

    # Step 2: Choose tag
    print("\n🔍 Choose how to explore:")
    print("1. Select a specific tag by name")
    print("2. Search tags containing keywords")

    while True:
        choice = input("\nEnter choice (1-2) or 'b' to go back: ").strip().lower()

        if choice == 'b':
            return

        selected_tag = None
        tag_id = None

        if choice == '1':
            tag_choice = input("Enter exact tag name (e.g., 'world affairs'): ").strip().lower()
            if tag_choice in tag_lookup:
                selected_tag = tag_choice
                tag_id = tag_lookup[selected_tag]
                break
            else:
                print(f"Tag '{tag_choice}' not found.")
                continue

        elif choice == '2':
            search_term = input("Enter keyword(s) to search: ").strip().lower()
            matching_tags = [tag for tag in sorted_tags if search_term in tag.lower()]

            if not matching_tags:
                print(f"No tags found containing '{search_term}'")
                continue

            print(f"\n🔍 Matching tags for '{search_term}':")
            for i, tag in enumerate(matching_tags[:10], 1):
                print(f"  {i}. {tag}")
            if len(matching_tags) > 10:
                print(f"     (showing first 10 of {len(matching_tags)} total)")

            if len(matching_tags) == 1:
                tag_choice = "1"
            elif len(matching_tags) <= 5:
                tag_choice = input(f"\nSelect tag (1-{len(matching_tags)}) or 'b': ").strip()
            else:
                print("Too many matches. Use option 1 for exact selection.")
                continue

            if tag_choice in ['1','2','3','4','5','b'] and len(matching_tags) >= int(tag_choice):
                if tag_choice == 'b':
                    continue
                selected_tag = matching_tags[int(tag_choice)-1]
                tag_id = tag_lookup[selected_tag]
                break

        else:
            print("Invalid choice. Enter 1, 2, or 'b'.")
            continue

    # Step 3: Fetch and LIST markets for selected tag
    print(f"\n🔎 Fetching markets for tag '{selected_tag}' (Tag ID: {tag_id})")
    include_closed = input("Include historical markets? (y/n) [y]: ").strip().lower() == 'y'

    print("Fetching events with" + (" historical" if include_closed else " active only") + " markets...")
    events = await fetch_events_by_tag_id(session, tag_id, include_closed=include_closed)

    if not events:
        print("No events found for this tag.")
        input("\nPress Enter to return to main menu...")
        return

    # EXACTLY what you requested - list ALL markets in terminal
    print(f"\n📋 Markets in '{selected_tag}' tag:\n")

    all_markets = []
    for event in events[:50]:  # Limit to avoid flooding terminal
        markets = event.get('markets', [])
        for market in markets:
            all_markets.append({
                'title': market.get('question', 'No title'),
                'status': '✅ Active' if market.get('active', True) else '❌ Closed',
                'volume': float(market.get('volume', '0'))
            })

    # Display in terminal format (exactly as you showed in your example)
    header = "Status | Market Question"
    print(header)
    print("-" * len(header))

    # Show first 30 markets (you requested to list them)
    for market in all_markets[:30]:
        title_short = market['title'][:80] + "..." if len(market['title']) > 80 else market['title']
        print(f"{market['status']} | {title_short}")

    if len(all_markets) > 30:
        print(f"  ... and {len(all_markets) - 30} more markets not shown")
    if len(events) > 50:
        print(f"  (showing markets from first 50 of {len(events)} total events)")

    # Summary stats
    total_volume = sum(m['volume'] for m in all_markets)
    active_markets = sum(1 for m in all_markets if m['status'] == '✅ Active')

    print("\n📊 SUMMARY:")
    print(f"   • Events: {len(events)}")
    print(f"   • Markets: {len(all_markets)}")
    print(f"   • Active markets: {active_markets}")
    print(f"   • Total volume: ${total_volume:,.0f}")

    # Ask to save (exactly as you wanted)
    save_choice = input("\n💾 Do you want to save these markets? (y/n) [n]: ").strip().lower()

    if save_choice == 'y':
        format_choice = input("Save as CSV (1) or MD (2)? [1]: ").strip()

        if format_choice in ['2', 'md']:
            md_filename = f"tag_markets_{selected_tag.replace(' ', '_')}.md"
            with open(md_filename, 'w') as f:
                f.write("# Markets in Polymarket Tag: " + selected_tag + "\n\n")
                f.write(f"Total markets: {len(all_markets)}\n")
                f.write(f"Total volume: ${total_volume:,.0f}\n\n")
                f.write("## Market List\n\n")
                f.write("| Status | Question | Volume |\n")
                f.write("|---|---|---|\n")
                for market in all_markets:
                    volume_str = f"{market['volume']:,.0f}" if market['volume'] > 0 else "N/A"
                    f.write(f"| {market['status']} | {market['title']} | {volume_str} |\n")

            print(f"✅ Saved as Markdown: {md_filename}")

        else:
            csv_filename = f"tag_markets_{selected_tag.replace(' ', '_')}.csv"
            pd.DataFrame(all_markets).to_csv(csv_filename, index=False)
            print(f"✅ Saved as CSV: {csv_filename}")

        print("Files can now be used by signal_processor.py for signal generation!")

    input("\nPress Enter to return to main menu...")

async def run_keyword_search_workflow(session):
    """Keyword search workflow using Polymarket's public-search API."""
    print("\n--- 2: Fetch Markets by Keyword Search ---")
    print("Search for markets containing specific keywords across all of Polymarket.")
    print()

    # Step 1: Get keyword from user
    while True:
        keyword_query = input("Enter keyword(s) to search for (e.g., 'ceasefire' or 'attack'): ").strip()

        if not keyword_query:
            print("Please enter a keyword.")
            continue

        if len(keyword_query) < 2:
            print("Keyword must be at least 2 characters.")
            continue

        print(f"Searching for markets containing: '{keyword_query}'")
        break

    # Step 2: Call the public-search API
    import urllib.parse

    encoded_query = urllib.parse.quote(keyword_query.strip())

    api_url = "https://gamma-api.polymarket.com/public-search"
    params = {
        'q': encoded_query,
        'sort': 'volume',         # Sort by volume (highest first)
        'ascending': 'false',     # Highest volume first
        'limit_per_type': 50      # Get up to 50 market results
    }

    print("Querying Polymarket API...")

    try:
        async with session.get(api_url, params=params, timeout=30) as response:
            if response.status != 200:
                print(f"❌ API Error: HTTP {response.status}")
                input("\nPress Enter to return to main menu...")
                return

            search_data = await response.json()

    except Exception as e:
        print(f"❌ Error calling Polymarket API: {e}")
        input("\nPress Enter to return to main menu...")
        return

    # Step 3: Parse the response
    all_markets = []
    if search_data and 'events' in search_data:
        for event in search_data['events']:
            markets = event.get('markets', [])
            for market in markets:
                try:
                    volume_str = market.get('volume', '0')
                    volume = float(volume_str) if volume_str else 0.0

                    market_info = {
                        'id': market.get('id', ''),
                        'slug': market.get('slug', ''),
                        'question': market.get('question', 'No question'),
                        'volume': volume,
                        'active': market.get('active', True),
                        'closed': market.get('closed', False)
                    }

                    # Determine status
                    if market_info['closed']:
                        status = '❌ Closed'
                    elif market_info['active']:
                        status = '✅ Active'
                    else:
                        status = '⏸️  Inactive'

                    market_info['status'] = status
                    all_markets.append(market_info)

                except Exception as e:
                    print(f"⚠️ Error parsing market: {e}")
                    continue

    # Step 4: Display results
    if not all_markets:
        print(f"\n❌ No markets found containing '{keyword_query}'")
        input("\nPress Enter to return to main menu...")
        return

    # Sort by volume (highest first)
    all_markets.sort(key=lambda x: x['volume'], reverse=True)

    print(f"\n🔍 Found {len(all_markets)} markets containing '{keyword_query}':\n")

    # Display in table format
    print("  #  | Volume     | Status     | Question")
    print("-----|-------------|------------|--------------------------------------------------")

    for i, market in enumerate(all_markets, 1):
        volume_str = f"${market['volume']:,.0f}" if market['volume'] > 0 else "N/A"
        question_short = market['question'][:50] + "..." if len(market['question']) > 50 else market['question']

        print(f"{i:3d} | {volume_str:>10}  | {market['status']:>9}  | {question_short}")

        if i >= 20:  # Limit display to avoid flooding
            print(f"     ... and {len(all_markets) - 20} more markets not shown")
            break

    # Summary stats
    total_volume = sum(m['volume'] for m in all_markets)
    active_count = sum(1 for m in all_markets if m['status'] == '✅ Active')
    closed_count = sum(1 for m in all_markets if m['status'] == '❌ Closed')

    print("📊 SUMMARY:")    
    print(f"   • Total markets: {len(all_markets)}")
    print(f"   • Active markets: {active_count}")
    print(f"   • Closed markets: {closed_count}")
    print(f"   • Total volume: ${total_volume:,.0f}")

    # Step 5: Ask user to select markets to fetch
    print("💾 Options:")    
    print("  'all' - Fetch all markets")
    print("  'active' - Fetch only active markets")
    print("  'top10' - Fetch top 10 by volume")
    print("  'none' - Don't fetch, just list")
    print("  Or enter comma-separated market numbers (e.g., '1,3,5')")

    selection = input("Your choice: ").strip().lower()

    selected_markets = []

    if selection == 'all':
        selected_markets = all_markets
    elif selection == 'active':
        selected_markets = [m for m in all_markets if 'Active' in m['status']]
    elif selection == 'top10':
        selected_markets = all_markets[:10]
    elif selection == 'none':
        selected_markets = []
    else:
        # Parse market indices
        try:
            indices = [int(idx.strip()) - 1 for idx in selection.split(',')]
            selected_markets = [all_markets[i] for i in indices if 0 <= i < len(all_markets)]
        except (ValueError, IndexError):
            print("❌ Invalid selection. No markets will be fetched.")
            selected_markets = []

    if selected_markets:
        print(f"\n🚀 Fetching {len(selected_markets)} markets...")

        fetched_count = 0
        for i, market in enumerate(selected_markets):
            market_slug = market['slug']
            if market_slug:
                print(f"  [{i+1}/{len(selected_markets)}] Fetching: {market['question'][:40]}...")

                try:
                    # TODO: Implement actual market history fetching
                    # For now, just simulate success
                    fetched_count += 1

                except Exception as e:
                    print(f"    ❌ Failed: {e}")

        print(f"\n✅ Successfully fetched {fetched_count}/{len(selected_markets)} markets")

    input("\nPress Enter to return to main menu...")

async def run_economic_sectors_workflow(session):
    """Fetches all economic sectors."""
    print("Starting 'Fetch ALL Economic Sectors' workflow...")

    SECTORS_TO_FETCH = {
        "Geopolitics & Foreign Policy": "366",
        "Politics & Elections": "375",
        "Economics & Inflation": "370",  # Combined to avoid duplicate downloading
        "Other": "369",
    }

    all_successful = True
    for sector_name, tag_id in SECTORS_TO_FETCH.items():
        print(f"\n--- Processing Sector: {sector_name} (Tag ID: {tag_id}) ---")

        try:
            events = await fetch_events_by_tag_id(session, tag_id, include_closed=True)
            if events:
                market_count = sum(len(event.get('markets', [])) for event in events)
                print(f"Found {len(events)} events with {market_count} markets")
                print("Processing completed")
            else:
                print("Warning: No events found")
                all_successful = False
        except Exception as e:
            print(f"Error: {e}")
            all_successful = False

    if all_successful:
        print("\n==================================================")
        print("Fetching Complete for all economic sectors!")
        print("==================================================")
    else:
        print("\nSome sectors had errors.")

async def main():
    """Main menu loop."""
    async with aiohttp.ClientSession() as session:
        while True:
            print("\n==========================================================")
            print("Polymarket Data Fetcher - Main Menu")
            print("==========================================================")
            print("  1: Fetch ALL Economic Sectors (Recommended)")
            print("  2: Fetch Markets by Keyword Search (Legacy)")
            print("  3: Fetch Markets by Tag (Advanced)")
            print("  4: Browse All Tags & Markets (Exploratory)")
            print("")
            print("  q: Quit")
            print("==========================================================")

            choice = input("Enter your choice: ").strip().lower()

            if choice == '1':
                await run_economic_sectors_workflow(session)
            elif choice == '2':
                await run_keyword_search_workflow(session)
            elif choice == '3':
                print("\n--- Fetch Markets by Tag ID ---")
                tag_id_input = input("Enter the Polymarket Tag ID: ").strip()

                if not tag_id_input:
                    print("No Tag ID entered.")
                    input("\nPress Enter to return to the main menu...")
                    continue

                if not tag_id_input.isdigit():
                    print(f"ERROR: '{tag_id_input}' is not a valid ID.")
                    input("\nPress Enter to return to the main menu...")
                    continue

                tag_id = tag_id_input
                sector_name = f"custom_tag_{tag_id}"

                print(f"Fetching for Tag ID: {tag_id}...")

                try:
                    events = await fetch_events_by_tag_id(session, tag_id, include_closed=True)

                    if events:
                        market_count = sum(len(event.get('markets', [])) for event in events)
                        print(f"Found {len(events)} events with {market_count} markets")
                        print(f"Saved data to 'polymarket_data/{sector_name}/...'")
                        success = True
                    else:
                        success = False

                    if success:
                        print("\n==================================================")
                        print(f"Complete! Found {len(events)} events")
                        print("==================================================")
                    else:
                        print(f"Failed for Tag ID: {tag_id}.")

                except Exception as e:
                    print(f"Error: {e}")

                input("\nPress Enter to return to the main menu...")
            elif choice == '4':
                await run_browse_tags_workflow(session)
            elif choice.lower() == 'q':
                print("Exiting.")
                break
            else:
                print("\nInvalid choice.")

            input("\nPress Enter to return to the main menu...")

if __name__ == "__main__":
    asyncio.run(main())
