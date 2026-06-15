"""
download_data.py

Downloads the 3 benchmark datasets as ARFF files directly from OpenML.
Does NOT require the openml Python package — uses plain urllib.

Usage:
    python download_data.py

If downloads fail due to network restrictions, download manually:
    bank_marketing : https://www.openml.org/data/download/22044760
    california     : https://www.openml.org/data/download/22044717
    magic_telescope: https://www.openml.org/data/download/22044756

Place the files in the project root as:
    bank_marketing.arff
    california.arff
    magic_telescope.arff
"""

import os
import urllib.request

# Direct ARFF file download URLs from OpenML (no API key required)
DATASETS = {
    "bank_marketing.arff": "https://www.openml.org/data/download/22044760",
    "california.arff":     "https://www.openml.org/data/download/22044717",
    "magic_telescope.arff":"https://www.openml.org/data/download/22044756",
}


def download(filename: str, url: str, output_dir: str = ".") -> None:
    dest = os.path.join(output_dir, filename)
    if os.path.exists(dest):
        print(f"  [{filename}] already exists, skipping.")
        return
    print(f"  [{filename}] downloading...")
    try:
        urllib.request.urlretrieve(url, dest)
        size_kb = os.path.getsize(dest) // 1024
        print(f"  [{filename}] saved ({size_kb} KB)")
    except Exception as e:
        if os.path.exists(dest):
            os.remove(dest)
        raise RuntimeError(
            f"Download failed for {filename}.\n"
            f"  URL: {url}\n"
            f"  Error: {e}\n"
            f"  → Download manually and place in the project root."
        ) from e


if __name__ == "__main__":
    print("Downloading datasets from OpenML...\n")
    failed = []
    for filename, url in DATASETS.items():
        try:
            download(filename, url)
        except RuntimeError as e:
            print(f"  ERROR: {e}\n")
            failed.append(filename)

    if failed:
        print(f"\n{len(failed)} download(s) failed. See manual download URLs above.")
    else:
        print("\nAll datasets ready.")