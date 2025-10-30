#!/usr/bin/env python3
"""
SORT TAG MARKETS BY VOLUME

Extract all markets from tag_*.md files and sort them by volume (highest to lowest).
Shows which markets have the highest trading activity across all tag categories.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple

def extract_markets_from_md(filepath: Path) -> List[Tuple[str, str, str]]:
    """
    Extract markets from a markdown file with volume information.
    Returns list of (question, status, volume, filename) tuples.
    """
    markets = []

    if not filepath.exists():
        print(f"File not found: {filepath}")
        return markets

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by the market list table
    market_list_match = re.search(r'## Market List\s*\n\s*\|.*\|\s*\n\s*\|.*\|\s*\n\s*\|.*\|\s*\n\s*((?:\|.*\|\s*\n\s*)*)', content)

    if not market_list_match:
        print(f"No market list found in {filepath.name}")
        return markets

    market_table = market_list_match.group(1)

    # Parse each line of the table
    lines = market_table.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line or line.startswith('|---') or line.startswith('| Status'):
            continue

        # Parse markdown table row: | Status | Question | Volume |
        match = re.match(r'\| ([^|]+) \| ([^|]+) \| ([^|]+) \|', line)
        if match:
            status = match.group(1).strip()
            question = match.group(2).strip()
            volume_str = match.group(3).strip()

            # Parse volume - handle commas and "N/A"
            if volume_str == "N/A":
                volume = 0
            else:
                volume_str = volume_str.replace(',', '').replace('$', '')
                try:
                    volume = int(volume_str)
                except ValueError:
                    volume = 0

            markets.append((question, status, volume, filepath.name))

    return markets

def main():
    # Find all tag markdown files
    tag_files = list(Path('.').glob('tag_*.md'))

    if not tag_files:
        print("No tag_*.md files found in current directory")
        return

    print(f"Found {len(tag_files)} tag files:")
    for f in tag_files:
        print(f"  - {f.name}")

    # Extract all markets from all files
    all_markets = []

    for filepath in tag_files:
        markets = extract_markets_from_md(filepath)
        all_markets.extend(markets)
        print(f"  Extracted {len(markets)} markets from {filepath.name}")

    print(f"\nTotal markets extracted: {len(all_markets)}")

    # Sort by volume (highest to lowest), keeping only active markets
    active_markets = [(q, s, v, f) for (q, s, v, f) in all_markets if s == "✅ Active"]
    sorted_markets = sorted(active_markets, key=lambda x: x[2], reverse=True)

    print(f"Active markets after sorting: {len(sorted_markets)}")

    # Display top 20 highest volume markets
    print(f"\n{'='*100}")
    print("TOP 20 HIGHEST VOLUME MARKETS ACROSS ALL TAGS")
    print(f"{'='*100}")
    print("#".ljust(3) + "Question".ljust(60) + "Volume".rjust(15) + "Tag")
    print("-" * 100)

    for i, (question, status, volume, filename) in enumerate(sorted_markets[:20], 1):
        tag_name = filename.replace('tag_markets_', '').replace('.md', '')
        question_short = question[:55] + "..." if len(question) > 55 else question
        print("3")

    # Show summary by tag
    print(f"\n{'='*100}")
    print("MARKETS PER TAG CATEGORY")
    print(f"{'='*100}")

    tag_summary = {}
    for filepath in tag_files:
        tag_name = filepath.name.replace('tag_markets_', '').replace('.md', '')
        markets = extract_markets_from_md(filepath)
        active_count = len([m for m in markets if m[1] == "✅ Active"])
        inactive_count = len([m for m in markets if m[1] == "❌ Closed"])
        total_volume = sum([int(m[2]) for m in markets if isinstance(m[2], (int, float)) and m[2] > 0])

        tag_summary[tag_name] = {
            'total': len(markets),
            'active': active_count,
            'inactive': inactive_count,
            'total_volume': total_volume
        }

    # Sort tags by total volume
    sorted_tags = sorted(tag_summary.items(), key=lambda x: x[1]['total_volume'], reverse=True)

    print("Tag Category".ljust(15) + "Total".rjust(8) + "Active".rjust(8) + "Inactive".rjust(10) + "Total Volume".rjust(15))
    print("-" * 60)

    for tag_name, stats in sorted_tags:
        total_vol = f"${stats['total_volume']:,}"
        print("15")

    # Show active vs inactive breakdown
    total_active = sum([s['active'] for s in tag_summary.values()])
    total_inactive = sum([s['inactive'] for s in tag_summary.values()])
    grand_total_volume = sum([s['total_volume'] for s in tag_summary.values()])

    print("-" * 60)
    total_vol_formatted = f"${grand_total_volume:,}"
    print("15")

    # Create detailed export file
    output_file = Path("test_signals/all_tag_markets_sorted_by_volume.csv")
    print(f"\nSaving detailed results to: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("question,status,volume,tag_file\n")
        for question, status, volume, filename in sorted_markets:
            tag_name = filename.replace('tag_markets_', '').replace('.md', '')
            f.write(f'"{question}","{status}",{volume},"{tag_name}"\n')

    print("✅ Sorting complete!")

    # Show interesting insights
    print(f"\n{'='*60}")
    print("INTERESTING INSIGHTS")
    print(f"{'='*60}")

    # Highest volume individual market
    if sorted_markets:
        top_market = sorted_markets[0]
        print(f"🏆 Highest volume market: '{top_market[0][:60]}...' (${top_market[2]:,})")

    # Most active tag
    most_active_tag = max(tag_summary.items(), key=lambda x: x[1]['active'])
    print(f"📈 Most active tag: {most_active_tag[0]} ({most_active_tag[1]['active']} active markets)")

    # Average volumes
    avg_volume_all = grand_total_volume / total_active if total_active > 0 else 0
    print(",.0f")
    # Markets above average volume
    high_volume_count = len([m for m in sorted_markets if m[2] > avg_volume_all])
    print(",.1%")


if __name__ == "__main__":
    main()
