#!/usr/bin/env python3
"""
Data preparation script for Environmental LLM Fine-tuning.

Usage:
    python scripts/prepare_data.py \
        --input data/raw/climate_qa.json \
        --output data/processed/climate_qa_train.json \
        --format instruction \
        --template alpaca
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.data_processor import (
    prepare_training_data,
    load_data,
    save_data,
    save_jsonl,
    convert_qa_to_instruction,
    augment_dataset,
    split_dataset,
    generate_environmental_qa,
    clean_text,
    is_valid_example,
)
from src.dataset import print_dataset_stats


def setup_logging(verbose: bool = False):
    """Setup logging."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Prepare training data")
    
    parser.add_argument(
        "--input",
        type=str,
        help="Input data file (JSON/JSONL)",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path",
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["instruction", "qa", "chat"],
        default="instruction",
        help="Input data format",
    )
    
    parser.add_argument(
        "--template",
        type=str,
        choices=["alpaca", "chatml", "llama2", "phi"],
        default="alpaca",
        help="Instruction template",
    )
    
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Augment data with variations",
    )
    
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split into train/val/test",
    )
    
    parser.add_argument(
        "--generate_sample",
        action="store_true",
        help="Generate sample environmental data",
    )
    
    parser.add_argument(
        "--min_length",
        type=int,
        default=10,
        help="Minimum example length",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    setup_logging(args.verbose)
    
    print("\n" + "=" * 60)
    print("📊 Data Preparation for Environmental LLM")
    print("=" * 60 + "\n")
    
    if args.generate_sample:
        # Generate sample environmental data
        logger.info("Generating sample environmental Q&A data...")
        
        data = generate_environmental_qa()
        
        # Add more examples manually for better training
        additional_data = [
            {
                "instruction": "What is climate change?",
                "input": "",
                "output": "Climate change refers to long-term shifts in global temperatures and weather patterns. While climate change can occur naturally, human activities have been the main driver since the 1800s, primarily due to burning fossil fuels like coal, oil, and gas, which produces heat-trapping greenhouse gases."
            },
            {
                "instruction": "Explain the greenhouse effect.",
                "input": "",
                "output": "The greenhouse effect is a natural process where certain gases in Earth's atmosphere trap heat from the sun, keeping the planet warm enough to sustain life. These gases, including carbon dioxide, methane, and water vapor, act like a blanket around Earth. However, human activities have increased greenhouse gas concentrations, enhancing this effect and causing global warming."
            },
            {
                "instruction": "What are the main greenhouse gases?",
                "input": "",
                "output": "The main greenhouse gases are: 1) Carbon dioxide (CO2) - from burning fossil fuels and deforestation; 2) Methane (CH4) - from agriculture, landfills, and natural gas; 3) Nitrous oxide (N2O) - from fertilizers and industrial processes; 4) Fluorinated gases - from industrial applications; 5) Water vapor - the most abundant but varies naturally."
            },
            {
                "instruction": "How does renewable energy help combat climate change?",
                "input": "",
                "output": "Renewable energy helps combat climate change by replacing fossil fuel-based electricity generation with clean sources that produce little to no greenhouse gas emissions. Solar, wind, hydroelectric, and geothermal power can provide electricity without burning carbon-based fuels. Transitioning to renewables is essential for reducing global emissions and meeting climate targets."
            },
            {
                "instruction": "What is carbon neutrality?",
                "input": "",
                "output": "Carbon neutrality, also called net-zero carbon emissions, means achieving a balance between the carbon emissions produced and the carbon removed from the atmosphere. This can be achieved by reducing emissions as much as possible and then offsetting remaining emissions through carbon sequestration, reforestation, or carbon capture technologies."
            },
            {
                "instruction": "Explain ESG investing.",
                "input": "",
                "output": "ESG investing integrates Environmental, Social, and Governance factors into investment decisions. Environmental criteria examine how a company impacts nature, including carbon emissions and resource use. Social criteria look at relationships with employees, suppliers, and communities. Governance involves leadership, audits, and shareholder rights. ESG investing aims to generate long-term returns while promoting sustainable practices."
            },
            {
                "instruction": "What is the Paris Agreement?",
                "input": "",
                "output": "The Paris Agreement is a legally binding international treaty on climate change adopted in 2015 by 196 parties. Its goal is to limit global warming to well below 2°C, preferably 1.5°C, compared to pre-industrial levels. Countries submit Nationally Determined Contributions (NDCs) outlining their climate actions and must report regularly on emissions and implementation efforts."
            },
            {
                "instruction": "What are the impacts of sea level rise?",
                "input": "",
                "output": "Sea level rise caused by climate change has significant impacts: 1) Coastal flooding and erosion threatening communities; 2) Saltwater intrusion contaminating freshwater supplies; 3) Loss of coastal ecosystems like wetlands and mangroves; 4) Displacement of millions of people in low-lying areas; 5) Damage to coastal infrastructure and property; 6) Increased storm surge damage during hurricanes."
            },
            {
                "instruction": "Biến đổi khí hậu là gì?",
                "input": "",
                "output": "Biến đổi khí hậu là sự thay đổi lâu dài của nhiệt độ và các kiểu thời tiết toàn cầu. Mặc dù biến đổi khí hậu có thể xảy ra tự nhiên, nhưng các hoạt động của con người là nguyên nhân chính kể từ những năm 1800, chủ yếu do đốt nhiên liệu hóa thạch như than, dầu và khí đốt, tạo ra khí nhà kính giữ nhiệt."
            },
            {
                "instruction": "Năng lượng tái tạo là gì?",
                "input": "",
                "output": "Năng lượng tái tạo là năng lượng được tạo ra từ các nguồn tự nhiên có thể tái tạo, bao gồm năng lượng mặt trời, gió, thủy điện, địa nhiệt và sinh khối. Khác với nhiên liệu hóa thạch, các nguồn năng lượng này không cạn kiệt và tạo ra ít hoặc không có khí thải nhà kính, giúp giảm thiểu biến đổi khí hậu."
            },
        ]
        
        data.extend(additional_data)
        
        # Filter valid examples
        data = [d for d in data if is_valid_example(d, args.min_length)]
        
        logger.info(f"Generated {len(data)} examples")
        
    elif args.input:
        # Load and process input data
        logger.info(f"Loading data from: {args.input}")
        raw_data = load_data(args.input)
        logger.info(f"Loaded {len(raw_data)} raw examples")
        
        # Convert format if needed
        if args.format == "qa":
            data = convert_qa_to_instruction(raw_data)
        elif args.format == "chat":
            from src.data_processor import convert_chat_to_instruction
            data = convert_chat_to_instruction(raw_data)
        else:
            data = raw_data
        
        # Clean and filter
        data = [
            {
                "instruction": clean_text(d.get("instruction", "")),
                "input": clean_text(d.get("input", "")),
                "output": clean_text(d.get("output", "")),
            }
            for d in data
        ]
        data = [d for d in data if is_valid_example(d, args.min_length)]
        
        logger.info(f"After processing: {len(data)} examples")
        
    else:
        logger.error("Either --input or --generate_sample is required")
        sys.exit(1)
    
    # Augment if requested
    if args.augment:
        original_count = len(data)
        data = augment_dataset(data, augmentation_factor=2)
        logger.info(f"Augmented: {original_count} → {len(data)} examples")
    
    # Print stats
    print_dataset_stats(data)
    
    # Split and save
    if args.split:
        train, val, test = split_dataset(data)
        
        output_path = Path(args.output)
        save_data(train, str(output_path.parent / f"{output_path.stem}_train.json"))
        save_data(val, str(output_path.parent / f"{output_path.stem}_val.json"))
        save_data(test, str(output_path.parent / f"{output_path.stem}_test.json"))
        
        logger.info(f"Saved train/val/test splits to: {output_path.parent}")
    else:
        save_data(data, args.output)
        logger.info(f"Saved to: {args.output}")
    
    print("\n" + "=" * 60)
    print("✅ Data preparation complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
