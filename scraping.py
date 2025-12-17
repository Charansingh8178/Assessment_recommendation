import requests
import pandas as pd
import time
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/products/product-catalog/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behaviour",
    "S": "Simulations"
}

def extract_yes_no(span):
    if not span:
        return "Not Available"
    classes = span.get("class", [])
    if "-yes" in classes:
        return "Yes"
    if "-no" in classes:
        return "No"
    return "Not Available"

def scrape_catalog_listings():
    records = []
    start = 0
    step = 12

    while True:
        params = {"start": start, "type": 1}
        resp = requests.get(CATALOG_URL, headers=HEADERS, params=params)
        if resp.status_code != 200:
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        rows = soup.select("tr[data-entity-id]")

        if not rows:
            break

        print(f"[INFO] start={start} → {len(rows)} rows")

        for row in rows:
            name_cell = row.select_one("td.custom__table-heading__title a")
            if not name_cell:
                continue

            name = name_cell.get_text(strip=True)
            url = BASE_URL + name_cell["href"]
            remote_span = row.select_one("td:nth-of-type(2) span.catalogue__circle")
            remote_testing = extract_yes_no(remote_span)

            adaptive_span = row.select_one("td:nth-of-type(3) span.catalogue__circle")
            adaptive_support = extract_yes_no(adaptive_span)
            test_key = row.select_one("span.product-catalogue__key")
            if not test_key:
                continue  

            key = test_key.get_text(strip=True)
            test_type = TEST_TYPE_MAP.get(key)
            if not test_type:
                continue

            records.append({
                "assessment_name": name,
                "url": url,
                "test_type": test_type,
                "remote_testing": remote_testing,
                "adaptive_support": adaptive_support
            })

        start += step
        time.sleep(1)

    return pd.DataFrame(records)

def extract_section(soup, title):
    header = soup.find("h4", string=lambda x: x and title.lower() in x.lower())
    if not header:
        return "Not Available"

    content = header.find_next("p")
    if not content:
        return "Not Available"

    return content.get_text(" ", strip=True)
def enrich_with_detail_data(df):
    descriptions = []
    durations = []
    job_levels = []
    languages = []

    for url in tqdm(df["url"], desc="Opening assessment pages"):
        resp = requests.get(url, headers=HEADERS)
        if resp.status_code != 200:
            descriptions.append("Not Available")
            durations.append("Not Available")
            job_levels.append("Not Available")
            languages.append("Not Available")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")

        descriptions.append(extract_section(soup, "Description"))
        durations.append(extract_section(soup, "Assessment length"))
        job_levels.append(extract_section(soup, "Job levels"))
        languages.append(extract_section(soup, "Languages"))

        time.sleep(0.5)

    df["description"] = descriptions
    df["assessment_length"] = durations
    df["job_levels"] = job_levels
    df["languages"] = languages

    return df

def main():
    print("[STEP 1] Scraping catalog listings...")
    base_df = scrape_catalog_listings()

    print(f"[INFO] Found {len(base_df)} Individual Test Solutions")

    print("[STEP 2] Scraping detail pages...")
    final_df = enrich_with_detail_data(base_df)

    final_df.to_csv("shl_assessments.csv", index=False)
    print(f"[DONE] Saved {len(final_df)} rows to shl_assessments.csv")


if __name__ == "__main__":
    main()
