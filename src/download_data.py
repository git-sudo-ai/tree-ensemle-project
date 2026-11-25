import argparse
from pathlib import Path
import urllib.request

VARIANT_URLS = {
    "V4": "https://archive.ics.uci.edu/static/public/222/data.csv",
}


def download_dataset(variant: str = "V4", out_dir: str = "data") -> Path:
    """Download the dataset for the specified lab variant into the local data folder."""
    key = variant.upper()
    if key not in VARIANT_URLS:
        raise ValueError(f"Unsupported variant: {variant}. Available: {list(VARIANT_URLS)}")
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / "bank_marketing.csv"
    urllib.request.urlretrieve(VARIANT_URLS[key], out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download UCI Bank Marketing dataset")
    parser.add_argument("--variant", default="V4", help="Lab variant code (default: V4)")
    parser.add_argument("--out-dir", default="data", help="Output directory (default: data)")
    args = parser.parse_args()
    saved = download_dataset(args.variant, args.out_dir)
    print(f"Dataset saved to {saved}")


if __name__ == "__main__":
    main()
