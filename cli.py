#!/usr/bin/env python
import argparse
import json
import sys
import os

# Adjust path to import ocr_engine if run directly or as package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ocr_engine import TyreOCREngine


def main():
    parser = argparse.ArgumentParser(
        description="Tyre OCR Specification Extractor using Qwen-VL models."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the tyre image file (local path)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="HuggingFace model ID to use (default: Qwen/Qwen2-VL-2B-Instruct)."
    )
    parser.add_argument(
        "--mlx",
        action="store_true",
        help="Use Apple Silicon MLX framework optimization if available."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="json",
        choices=["json", "pretty"],
        help="Output format: 'json' (raw machine-readable) or 'pretty' (human-readable table)."
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save the JSON output (e.g. output.json)."
    )

    args = parser.parse_args()

    # Validate image path
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at '{args.image}'", file=sys.stderr)
        sys.exit(1)

    print(f"Starting Tyre OCR Extractor...", file=sys.stderr)
    print(f"Image: {args.image}", file=sys.stderr)
    print(f"Model: {args.model}", file=sys.stderr)
    print(f"Framework: {'MLX' if args.mlx else 'PyTorch/Transformers'}", file=sys.stderr)
    print("-" * 50, file=sys.stderr)

    try:
        # Load engine and run extraction
        engine = TyreOCREngine(model_id=args.model, use_mlx=args.mlx)
        result = engine.extract_tyre_info(args.image)
        
        # Save to file if specified
        if args.save:
            with open(args.save, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.save}", file=sys.stderr)

        # Output formatting
        if args.output == "json":
            print(json.dumps(result, indent=2))
        else:
            # Pretty printed output
            print("\n" + "=" * 50)
            print("         EXTRACTED TYRE SPECIFICATIONS")
            print("=" * 50)
            print(f" Brand:            {result.get('brand') or 'N/A'}")
            print(f" Model/Pattern:    {result.get('model') or 'N/A'}")
            print(f" Tyre Size:        {result.get('tyre_size') or 'N/A'}")
            print(f" Load Index:       {result.get('load_index') or 'N/A'}")
            print(f" Speed Rating:     {result.get('speed_rating') or 'N/A'}")
            print(f" DOT Code:         {result.get('dot_code') or 'N/A'}")
            print(f" Mfg Date:         {result.get('manufacturing_date') or 'N/A'}")
            print(f" Max Load/Press:   {result.get('max_load_pressure') or 'N/A'}")
            
            other_m = result.get('other_markings', [])
            print(f" Other Markings:   {', '.join(other_m) if other_m else 'None'}")
            
            print("-" * 50)
            print(" All Detected Sidewall Text:")
            for text in result.get('all_sidewall_text', []):
                print(f"  - {text}")
            print("=" * 50 + "\n")

    except Exception as e:
        print(f"Extraction failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
