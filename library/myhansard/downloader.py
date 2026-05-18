import time
from datetime import date, timedelta
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def build_paths(date: date) -> tuple[str, Path]:
    date_str = date.strftime("%d%m%Y")
    url = f"https://www.parlimen.gov.my/files/hindex/pdf/DR-{date_str}.pdf"
    filename = f"{date_str}.pdf"
    file_path = Path("data/raw") / filename
    return url, file_path


def build_dates(start_date: date, end_date: date) -> list[date]:
    dates = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:
            dates.append(current_date)
        current_date += timedelta(days=1)
    return dates


def download_one_pdf(date: date) -> str:
    url, file_path = build_paths(date)
    if file_path.exists():
        print(f"Skipped (exists): {file_path}")
        return "skipped"

    print(f"Downloading: {url}")

    for attempt in range(3):
        try:
            response = requests.get(url, verify=False, timeout=(5, 30))
            response.raise_for_status()
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if len(response.content) < 10_000:  # 10KB
                print(
                    f"Suspiciously small response ({len(response.content)} bytes) "
                    f"for {url}, skipping."
                )
                return "skipped"

            with open(file_path, "wb") as f:
                f.write(response.content)
            return "success"

        except requests.RequestException as e:
            if attempt < 2:
                print(f"Attempt {attempt + 1}/3 after {2**attempt}s: {e}")
                time.sleep(2**attempt)
                continue
            if isinstance(e, requests.HTTPError) and e.response is not None:
                if e.response.status_code == 404:
                    print(f"Weekend or no meeting: {url}")
                    return "skipped"
                else:
                    print(f"Failed to download {url}: {e}")
                    return "failed"

            elif isinstance(e, requests.Timeout):
                print(f"Timeout while downloading {url}: {e}")
                return "failed"

            else:
                print(f"Error downloading {url}: {e}")
                return "failed"


def download_by_date(start: str, end: str | None = None) -> dict:
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end) if end else start_date

    dates = build_dates(start_date, end_date)
    success_count = 0
    skipped_count = 0
    failed_count = 0

    failed_dates = []

    for d in dates:
        status = download_one_pdf(d)
        if status == "success":
            success_count += 1
        elif status == "skipped":
            skipped_count += 1
        else:
            failed_count += 1
            failed_dates.append(d.isoformat())

    if failed_dates:
        Path("data/failed_dates.txt").write_text("\n".join(failed_dates))
        print("Failed dates saved to data/failed_dates.txt")

    return {
        "success": success_count,
        "skipped": skipped_count,
        "failed": failed_count,
        "failed_dates": failed_dates if failed_dates else None,
    }
