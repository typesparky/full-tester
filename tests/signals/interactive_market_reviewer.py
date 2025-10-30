#!/usr/bin/env python3
"""
INTERACTIVE MARKET REVIEW SYSTEM

Quality scoring and interactive review for Polymarket signals.
Filters out rhetorical/speech markets and ensures only substantive markets
are included in signals.

Usage:
python interactive_market_reviewer.py --scan
python interactive_market_reviewer.py --review <concept>
python interactive_market_reviewer.py --stats
"""

import sys
import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CORE.unified_signal_generator import UnifiedSignalGenerator
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MarketQualityScore:
    """Quality score breakdown for individual markets"""
    overall_score: int = 0
    substantive_score: int = 0  # 0-25: substantive vs rhetorical
    impact_score: int = 0      # 0-25: political impact potential
    volume_score: int = 0      # 0-25: volume validation
    relevance_score: int = 0   # 0-25: signal relevance
    flags: List[str] = field(default_factory=list)

    def get_classification(self) -> str:
        """Get recommended classification based on score"""
        if self.overall_score >= 80:
            return "✅ HIGH QUALITY"
        elif self.overall_score >= 60:
            return "⚠️ REQUIRES REVIEW"
        else:
            return "❌ LOW QUALITY"

class InteractiveMarketReviewer:
    """Interactive system for reviewing market quality"""

    def __init__(self):
        self.generator = UnifiedSignalGenerator(debug=False)
        self.reviewed_markets_file = Path("test_signals/reviewed_markets.json")
        self.quality_cache_file = Path("test_signals/market_quality_cache.json")

        # Load existing reviews
        self.reviewed_decisions = self._load_reviewed_decisions()
        self.quality_cache = self._load_quality_cache()

    def _load_reviewed_decisions(self) -> Dict:
        """Load previously reviewed market decisions"""
        if self.reviewed_markets_file.exists():
            try:
                with open(self.reviewed_markets_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load review decisions: {e}")
        return {}

    def _load_quality_cache(self) -> Dict:
        """Load cached quality scores"""
        if self.quality_cache_file.exists():
            try:
                with open(self.quality_cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load quality cache: {e}")
        return {}

    def save_decisions(self):
        """Save all review decisions and cache"""
        with open(self.reviewed_markets_file, 'w') as f:
            json.dump(self.reviewed_decisions, f, indent=2)

        with open(self.quality_cache_file, 'w') as f:
            json.dump(self.quality_cache, f, indent=2)

        logger.info(f"💾 Saved {len(self.reviewed_decisions)} reviewed decisions")

    def score_market_quality(self, question: str, volume: float, concept_name: str) -> MarketQualityScore:
        """Score market quality using multiple criteria"""
        score = MarketQualityScore()
        question_lower = question.lower()
        flags = []

        # === SUBSTANTIVE VS RHETORICAL (0-25) ===
        substantive_score = 25

        rhetorical_patterns = [
            (r"\bwill\s+\w+\s+say\s+", "speech pattern"),          # "will X say" (more flexible)
            (r"\bsay\s+['\"]?[^'\"]*(['\"])?\s+during\s+", "speech during event"), # "say X during Y"
            (r"\bsay\s+['\"]?[^'\"]*['\"]?\s+times?\b", "word count pattern"), # "say X times"
            (r"\bmention\b", "mention pattern"),                  # mention as standalone word
            (r"\bphrase\b", "phrase pattern"),                    # phrase as standalone word
        ]

        # Check for placeholder/template markets - zero score immediately
        placeholder_patterns = [
            (r'\b(person|party)\s+[a-z]\b', "person/party template"),  # Person A, Party B
            (r'\bcountry\s+([a-z]{1}|[a-z]{3})\b', "country template"), # Country X, Country ABC
            (r'\b(candidate|option)\s+[a-z0-9]+\b', "candidate/option template"), # Candidate 1, Option A
            (r'\b(person|party)\s+[a-z]\b.*\b(win|election)', "person/party template - election"), # Any person/party template in election context
        ]

        # Check for irrelevant/ironic content (like Nobel Peace Prize wars which are not real geopolitical signals)
        irrelevant_patterns = [
            (r'nobel\s+peace\s+prize', "nobel peace prize market"),  # Nobel Peace Prize markets are irrelevant for geopolitics
            (r'celebrity.*out', "celebrity leadership change"),     # Celebrity politics (not real)
            (r'rapper.*president', "entertainment politics"),       # Rapper politics
            (r'out\s+in\s+2025', "specific year leader change"),    # Generic "out in 2025" predictions
            (r'be\s+the\s+first\s+leader\s+out\s+in\s+\d{4}', "first leader out pattern"),  # "first leader out" pattern
        ]

        # Also check for specific template phrases we've seen
        template_phrases = [
            "will another person win",
            "will another party",
            "hold the second most seats",
            "win 2nd place",
            "win 2nd place",
            "person ",
            "party "
        ]

        # First check irrelevant content
        for pattern, flag_reason in irrelevant_patterns:
            if re.search(pattern, question_lower, re.IGNORECASE):
                flags.append(f"Irrelevant: {flag_reason}")
                substantive_score = 0  # Irrelevant markets get zero score
                break
        else:
            # Then check placeholder templates
            for pattern, flag_reason in placeholder_patterns:
                if re.search(pattern, question_lower, re.IGNORECASE):
                    flags.append(f"Template: {flag_reason}")
                    substantive_score = 0  # Template markets get zero score
                    break
            else:
                # Check for template phrases if regex didn't match
                for phrase in template_phrases:
                    if phrase in question_lower and len(question_lower.split()) < 15:  # Short template questions
                        flags.append("Template: placeholder phrase")
                        substantive_score = 0  # Template markets get zero score
                        break

        # Multi-pass pattern detection for speech markets - ZERO SCORE FOR SPEECH
        speech_detected = False
        for pattern, flag_reason in rhetorical_patterns:
            if re.search(pattern, question_lower, re.IGNORECASE):
                # Speech markets automatically get ZERO score - completely excluded
                flags.append(f"Rhetorical: {flag_reason}")
                speech_detected = True
                substantive_score = 0
                break  # Single speech pattern detection = auto-fail

        score.substantive_score = substantive_score

        # If speech detected, ZERO out the ENTIRE score - this market is excluded
        if speech_detected:
            score.overall_score = 0  # Force zero score - no review needed
            score.impact_score = 0
            score.volume_score = 0
            score.relevance_score = 0
        else:
            # === POLITICAL IMPACT (0-25) ===
            high_impact = ["election", "wins", "policy", "war", "sanctions", "government"]
            impact_score = min(25, sum(1 for word in high_impact if word in question_lower) * 5)
            score.impact_score = impact_score

            # === VOLUME VALIDATION (0-25) ===
            volume_score = min(25, volume / 1000)
            score.volume_score = volume_score

            # === SIGNAL RELEVANCE (0-25) ===
            relevance_score = 25
            score.relevance_score = relevance_score

        # Calculate overall (but don't override forced zero for speech markets)
        if not speech_detected:
            score.overall_score = sum([
                score.substantive_score,
                score.impact_score,
                score.volume_score,
                score.relevance_score
            ])

        score.flags = flags
        return score

    def scan_and_score_markets(self) -> Dict[str, List]:
        """Scan and score all markets by concept"""
        logger.info("🔍 Scanning and scoring all markets...")

        results = {}

        for concept_key, concept in self.generator.concepts.items():
            concept_results = []

            # Find markets matching concept keywords
            for manifest_file in Path("polymarket_data").glob("_manifest_*.csv"):
                try:
                    manifest_df = pd.read_csv(manifest_file)
                except Exception:
                    continue

                for _, market_row in manifest_df.iterrows():
                    question = market_row['question']

                    # Check keyword match
                    matches_keywords = any(
                        kw.lower() in question.lower()
                        for kw in concept.keywords
                    )

                    if not matches_keywords:
                        continue

                    # Create market info
                    market_info = {
                        'filename': market_row['filename'],
                        'question': question,
                        'volume': float(market_row.get('volume', 0)),
                        'active': market_row.get('active', True)
                    }

                    # Score market
                    score = self.score_market_quality(question, market_info['volume'], concept_key)

                    market_info.update({
                        'quality_score': score.overall_score,
                        'flags': score.flags,
                        'classification': score.get_classification()
                    })

                    concept_results.append(market_info)

            results[concept_key] = concept_results
            logger.info(f"    Found {len(concept_results)} markets for {concept.name}")

        return results

    def interactive_review(self, concept_key: str, reviews_per_session: int = 10):
        """Interactive review session"""
        if concept_key not in self.generator.concepts:
            logger.error(f"Unknown concept: {concept_key}")
            return

        logger.info(f"🎯 Reviewing markets for concept: {concept_key}")

        # Get markets needing review
        markets_to_review = []
        results = self.scan_and_score_markets()

        for market in results[concept_key]:
            market_key = market['filename']

            # Skip reviewed markets
            if market_key in self.reviewed_decisions:
                continue

            # Flag low quality markets (but exclude auto-filtered speech markets with score 0)
            if market['quality_score'] < 70 and market['quality_score'] > 0:
                market['_priority'] = 70 - market['quality_score']
                markets_to_review.append(market)

        # Sort by priority
        markets_to_review.sort(key=lambda x: x['_priority'], reverse=True)

        if not markets_to_review:
            logger.info(f"✅ No markets need review for {concept_key}")
            return

        logger.info(f"📋 {len(markets_to_review)} markets need review")

        reviewed_count = 0

        for i, market in enumerate(markets_to_review):
            if reviewed_count >= reviews_per_session:
                break

            print(f"\n{'='*80}")
            print(f"MARKET REVIEW #{i+1} (Priority: {market['_priority']})")
            print(f"{'='*80}")
            print(f"Question: {market['question']}")
            print(f"Volume: ${market['volume']:,.0f}")
            print(f"Score: {market['quality_score']}/100")
            print(f"Classification: {market['classification']}")

            if market['flags']:
                print(f"🚩 Flags: {', '.join(market['flags'])}")

            print("Decision: [I]nclude / [E]xclude / [S]kip / [Q]uit")
            choice = input("Choice: ").strip().upper()

            if choice == 'I':
                self.reviewed_decisions[market['filename']] = {
                    'decision': 'include',
                    'timestamp': pd.Timestamp.now().isoformat()
                }
                print("✅ INCLUDED")
                reviewed_count += 1

            elif choice == 'E':
                self.reviewed_decisions[market['filename']] = {
                    'decision': 'exclude',
                    'timestamp': pd.Timestamp.now().isoformat()
                }
                print("❌ EXCLUDED")
                reviewed_count += 1

            elif choice == 'Q':
                break

        self.save_decisions()

    def update_manual_flags(self):
        """Update manual flags from review decisions"""
        logger.info("📝 Updating manual flags...")

        # Group exclusions by concept
        concept_excludes = {}

        # Build reverse mapping
        filename_to_concept = {}
        for concept_key, markets in self.scan_and_score_markets().items():
            for market in markets:
                filename_to_concept[market['filename']] = concept_key

        # Collect exclusions
        for filename, decision in self.reviewed_decisions.items():
            if decision['decision'] == 'exclude':
                concept = filename_to_concept.get(filename)
                if concept:
                    if concept not in concept_excludes:
                        concept_excludes[concept] = []
                    concept_excludes[concept].append(filename)

        # Update flags
        updated_flags = self.generator.flagged_markets.copy() if self.generator.flagged_markets else {}

        for concept, files in concept_excludes.items():
            if concept not in updated_flags:
                updated_flags[concept] = {'excludes': []}
            if 'excludes' not in updated_flags[concept]:
                updated_flags[concept]['excludes'] = []

            existing = set(updated_flags[concept]['excludes'])
            new_files = [f for f in files if f not in existing]
            updated_flags[concept]['excludes'].extend(new_files)

            logger.info(f"  {concept}: Added {len(new_files)} excludes")

        with open(Path("CORE/signals_config.json"), 'w') as f:
            json.dump(updated_flags, f, indent=2)

        logger.info("✅ Manual flags updated")

    def show_stats(self):
        """Show review statistics"""
        logger.info("📊 REVIEW STATISTICS")
        logger.info("=" * 40)

        total_reviewed = len(self.reviewed_decisions)
        included = sum(1 for d in self.reviewed_decisions.values() if d['decision'] == 'include')
        excluded = sum(1 for d in self.reviewed_decisions.values() if d['decision'] == 'exclude')

        logger.info(f"Reviewed: {total_reviewed}")
        if total_reviewed > 0:
            logger.info(f"Included: {included} ({included/total_reviewed:.1f}%)")
            logger.info(f"Excluded: {excluded}")
        else:
            logger.info("No markets reviewed yet")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Interactive Market Review System")
    parser.add_argument('--scan', action='store_true', help='Scan and score all markets')
    parser.add_argument('--review', type=str, help='Interactive review for concept')
    parser.add_argument('--stats', action='store_true', help='Show review statistics')
    parser.add_argument('--update-flags', action='store_true', help='Update manual flags')

    args = parser.parse_args()

    reviewer = InteractiveMarketReviewer()

    if args.scan:
        results = reviewer.scan_and_score_markets()
        total_markets = sum(len(markets) for markets in results.values())
        logger.info(f"✅ Scanned {total_markets} markets")

    elif args.review:
        if args.review == 'all':
            concepts = list(reviewer.generator.concepts.keys())
            for concept in concepts:
                logger.info(f"\n🎯 Reviewing: {concept}")
                reviewer.interactive_review(concept, reviews_per_session=5)
        else:
            reviewer.interactive_review(args.review)

    elif args.stats:
        reviewer.show_stats()

    elif args.update_flags:
        reviewer.update_manual_flags()

    else:
        print("Interactive Market Reviewer")
        print("=" * 30)
        print("--scan              Scan and score markets")
        print("--review CONCEPT    Review specific concept")
        print("--review all        Review all concepts")
        print("--stats             Show statistics")
        print("--update-flags      Update manual flags")

if __name__ == "__main__":
    main()
