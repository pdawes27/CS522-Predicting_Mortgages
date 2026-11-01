from playwright.sync_api import sync_playwright
from pathlib import Path
import re

output_dir = Path("loan_data")
output_dir.mkdir(exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch_persistent_context(
        user_data_dir="fannie_profile",
        headless=False,
        accept_downloads=True
    )

    page = browser.pages[0] if browser.pages else browser.new_page()

    page.goto(
        "https://datadynamics.fanniemae.com/data-dynamics/",
        wait_until="domcontentloaded"
    )

    input(
        "Log in and navigate to the quarterly Primary Dataset table. "
        "Once you see the table, press Enter here."
    )

    # Track downloads from all tabs, including newly opened tabs
    downloads = []

    def register_page(new_page):
        new_page.on("download", lambda download: downloads.append(download))

    browser.on("page", register_page)

    for open_page in browser.pages:
        register_page(open_page)

    # Find a specific year's row in any tab or iframe
    def find_year_row(year):
        year_pattern = re.compile(rf"^\s*{year}\s*$")

        for tab in browser.pages:
            for frame in tab.frames:
                try:
                    rows = frame.locator("tr").filter(
                        has=frame.locator("td, th").filter(
                            has_text=year_pattern
                        )
                    )

                    if rows.count() > 0:
                        return rows.first, tab

                except Exception:
                    continue

        return None, None

    # Download 2016-2025 and first two quarters of 2026
    for year in range(2016, 2027):

        quarter_count = 2 if year == 2026 else 4

        print(f"\nProcessing {year}...")

        row, tab = find_year_row(year)

        if row is None:
            print(f"ERROR: Could not find row for {year}")
            continue

        links = row.get_by_text(
            "Acquisition and Performance",
            exact=True
        )

        count = links.count()

        print(f"Found {count} download links for {year}")

        if count < quarter_count:
            print(f"ERROR: Missing quarterly links for {year}")
            continue

        for quarter in range(1, quarter_count + 1):

            # Skip already downloaded files
            existing = list(
                output_dir.glob(f"{year}Q{quarter}.*")
            )

            if existing:
                print(f"Skipping {year} Q{quarter} (already exists)")
                continue

            print(f"Downloading {year} Q{quarter}...")

            previous_count = len(downloads)

            try:
                links.nth(quarter - 1).click(timeout=30000)

                # Wait for the browser download event
                for _ in range(240):
                    if len(downloads) > previous_count:
                        break

                    tab.wait_for_timeout(250)

                if len(downloads) == previous_count:
                    print(
                        f"ERROR: No download started for "
                        f"{year} Q{quarter}"
                    )
                    continue

                download = downloads[-1]

                # Preserve actual file extension
                extension = "".join(
                    Path(download.suggested_filename).suffixes
                )

                if not extension:
                    extension = ".zip"

                filename = f"{year}Q{quarter}{extension}"
                file_path = output_dir / filename

                download.save_as(str(file_path))

                print(f"Saved: {file_path}")

            except Exception as e:
                print(
                    f"ERROR downloading {year} Q{quarter}: {e}"
                )

    print("\nFinished downloading available files.")

    browser.close()