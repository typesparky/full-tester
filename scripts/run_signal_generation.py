#!/usr/bin/env python3
"""
RUN SIGNAL GENERATION

Generate market signals using modular configuration system.
Supports both unified methodology and simplified VWP generators.
"""

import sys
import os
import yaml
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modular signal generators
from src.signal_generators.vwp_signal import VWP_Signal  # Placeholder - will need to create
from src.signal_generators.vwev_rolled_signal import VWEV_Rolled_Signal
from src.filters import FilterManager  # Manifest-first filtering

def load_yaml_config(config_path: str) -> dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Generate market signals from configuration")
    parser.add_argument('--signal-config', required=True,
                       help='Path to signal configuration YAML file')
    parser.add_argument('--output-dir', default='data/signals',
                       help='Output directory for generated signals')

    args = parser.parse_args()

    # Load signal configuration
    signal_config = load_yaml_config(args.signal_config)

    # Generate signal based on generator type
    generator_type = signal_config.get('generator_type', 'vwp').lower()

    if generator_type == 'vwp':
        # Manifest-first filtering
        min_quality_score = signal_config.get('min_quality_score')
        input_manifest_files = signal_config.get('input_manifest_files', [])
        
        if input_manifest_files:
            # Initialize filter manager and apply manifest-first filtering
            filter_manager = FilterManager()
            filtered_file_paths = filter_manager.filter_manifests(input_manifest_files, min_quality_score)
            print(f"Filtered to {len(filtered_file_paths)} markets for signal generation")
        else:
            # Fallback to directory-based if no manifests (legacy)
            filtered_file_paths = None
            
        # Initialize VWP signal generator with polarity rules
        generator = VWP_Signal(
            filtered_file_paths=filtered_file_paths,
            input_dirs=signal_config.get('input_market_dirs', []),  # Keep for backward compatibility
            polarity_rules=signal_config.get('polarity_rules', {}),
            output_file=signal_config.get('output_file'),
            min_volume=1,
            min_quality_score=min_quality_score  # Pass through from config
        )

        # Generate signal
        signal_df = generator.generate()
        if signal_df is not None:
            generator.save_signal(signal_df)

            print(f"✅ Generated VWP signal: {signal_config['signal_name']}")
            print(f"📊 Records: {len(signal_df)}")
        else:
            print(f"❌ Failed to generate signal: {signal_config['signal_name']}")

    elif generator_type == 'unified':
        # Use unified methodology generator
        from src.signals.unified_signal_generator import UnifiedSignalGenerator
        generator = UnifiedSignalGenerator()
        # TODO: Implement unified generation from YAML config
        print("⚠️ Unified generator not yet implemented for YAML config")

    else:
        print(f"❌ Unknown generator type: {generator_type}")

if __name__ == "__main__":
    main()
