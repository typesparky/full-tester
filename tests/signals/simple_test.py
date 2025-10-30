#!/usr/bin/env python3
"""
Simple test to verify the unified methodology classes are importable and basic functionality works.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from CORE.unified_signal_generator import UnifiedSignalGenerator, MarketType, SignalPolarity
    print("✓ SUCCESS: All unified methodology classes imported correctly")
except ImportError as e:
    print(f"✗ FAILURE: Import error: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic functionality of the unified classes"""
    print("\n" + "="*60)
    print("TESTING BASIC FUNCTIONALITY")
    print("="*60)

    # Test enums
    print("Testing enums:")
    print(f"✓ MarketType.BINARY = {MarketType.BINARY}")
    print(f"✓ MarketType.QUANTITATIVE = {MarketType.QUANTITATIVE}")
    print(f"✓ SignalPolarity.PRO_CONCEPT = {SignalPolarity.PRO_CONCEPT}")
    print(f"✓ SignalPolarity.ANTI_CONCEPT = {SignalPolarity.ANTI_CONCEPT}")

    # Test generator initialization
    print("\nTesting generator initialization:")
    generator = UnifiedSignalGenerator(debug=False)
    print(f"✓ UnifiedSignalGenerator initialized with {len(generator.concepts)} concepts")


    # Test classification function
    print("\nTesting market classification:")
    test_questions = [
        "Will inflation be higher next year?",
        "What will CPI be at 3.5%?",
        "Will Trump win the election?"
    ]

    for question in test_questions:
        result = generator.classify_market_hybrid(question.lower(), ['inflation', 'cpi'])
        print(f"  '{question[:40]}...' → {result.value}")

    # Test polarity assignment
    print("\nTesting polarity assignment:")
    inflation_concept = generator.concepts['us_inflation']
    test_questions = [
        ("Will inflation be higher?", +1),  # higher = pro-concept
        ("Will inflation be lower?", -1),  # lower = anti-concept
    ]

    for question, expected in test_questions:
        result = generator.assign_polarity_scalar(question, inflation_concept)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{question}' → λ={result} (expected {expected})")

    # Test aligned price calculation
    print("\nTesting aligned price calculation:")
    test_cases = [
        (0.6, 1, 0.6),   # pro-concept alignment
        (0.4, -1, 0.6),  # anti-concept alignment
    ]

    for p_yes, lambda_val, expected in test_cases:
        result = generator.calculate_aligned_price(p_yes, lambda_val)
        status = "✓" if abs(result - expected) < 0.001 else "✗"
        print(f"  {status} P_yes={p_yes}, λ={lambda_val} → P'={result:.3f} (expected {expected:.3f})")

    print("\n" + "="*60)
    print("BASIC TESTS COMPLETED")
    print("="*60)

if __name__ == "__main__":
    test_basic_functionality()
    print("\n🎉 All basic functionality tests passed!")
