import streamlit as st
import pandas as pd
import asyncio
import aiohttp
import re
import time
import json
import os
from bs4 import BeautifulSoup
from io import BytesIO
from urllib.parse import urljoin, urlparse
from collections import deque

# Load CSS
def load_css():
    try:
        with open("style.css") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except:
        st.warning("No CSS loaded.")

load_css()

# --- Improved Email Regex ---
EMAIL_REGEX = re.compile(
    r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.(?:com|org|net|edu|gov|io|co|us|uk|ca|de|au|biz|info|ai|app|in|pk|nl|fr|it|es|se|ch|no|dk|pl|be|cz|at|ru|jp|cn|sg|hk|my|id|ph|za|br|mx|tr|ar|vn|gr|ro|pt|fi|ir|sa|nz)\b",
    re.IGNORECASE
)
FACEBOOK_REGEX = re.compile(r"https?://(www\.)?facebook\.com/[a-zA-Z0-9_\-./]+")
LINKEDIN_REGEX = re.compile(r"https?://(www\.)?linkedin\.com/[a-zA-Z0-9_\-./]+")
PRIVACY_EMAIL_REGEX = re.compile(r"(privacy|dpo|data\.protection|gdpr|compliance)@", re.IGNORECASE)

# Excluded sample emails
excluded_emails = {...}  # Keep your excluded list unchanged

# Save and Load local data
def save_to_local_storage(data):
    with open("local_storage.json", "w") as f:
        json.dump(data, f)

def load_from_local_storage():
    if os.path.exists("local_storage.json"):
        with open("local_storage.json", "r") as f:
            return json.load(f)
    return []

def download_partial_results(results, filename="partial_results.csv"):
    if results:
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        st.download_button("Download Partial Data", df.to_csv(index=False), filename, "text/csv")

# --- Main Crawler ---
async def crawl_website(url, session, semaphore, status, results, email_df_container, unique_emails, max_pages):
    collected_emails = set()
    facebook_url = ""
    linkedin_url = ""
    visited_urls = set()
    urls_to_visit = deque([(url, 0)])
    max_depth = 3
    base_domain = urlparse(url).netloc
    priority_paths = ["/contact", "/about", "/team", "/contact-us", "/get-in-touch", "/support"]

    try:
        async with semaphore:
            # Add priority paths
            for path in priority_paths:
                full_url = urljoin(url, path)
                if full_url not in visited_urls:
                    urls_to_visit.append((full_url, 0))

            while urls_to_visit and len(visited_urls) < max_pages:
                current_url, depth = urls_to_visit.popleft()
                if current_url in visited_urls or depth > max_depth:
                    continue

                visited_urls.add(current_url)
                status['current'] = current_url

                try:
                    async with session.get(current_url, timeout=10) as response:
                        if response.status != 200:
                            continue
                        html = await response.text()
                        soup = BeautifulSoup(html, "html.parser")
                        for tag in soup(["script", "style", "noscript"]):
                            tag.decompose()

                        # --- Smart text cleanup ---
                        text = soup.get_text(separator=" ")
                        # Fix merged endings like ".comThis" or ".ioContact"
                        text = re.sub(r"(\.[a-z]{2,})([A-Z])", r"\1 \2", text)
                        text = re.sub(r"(\.[a-z]{2,})([a-z])", r"\1 \2", text)

                        # --- Extract emails ---
                        found_emails = EMAIL_REGEX.findall(text)

                        # --- Post-cleanup ---
                        cleaned_emails = set()
                        for email in found_emails:
                            email = email.strip(".,;:()[]{}<>\"'! ")
                            email = re.sub(r"(\.[a-z]{2,})([A-Za-z]+)$", r"\1", email)
                            if re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", email):
                                cleaned_emails.add(email.lower())

                        filtered_emails = {
                            email for email in cleaned_emails
                            if email.lower() not in {e.lower() for e in excluded_emails}
                            and not PRIVACY_EMAIL_REGEX.search(email.lower())
                        }

                        collected_emails.update(filtered_emails)
                        unique_emails.update(filtered_emails)

                        # --- Social links ---
                        if not facebook_url:
                            match_fb = FACEBOOK_REGEX.search(html)
                            if match_fb:
                                facebook_url = match_fb.group()

                        if not linkedin_url:
                            match_ln = LINKEDIN_REGEX.search(html)
                            if match_ln:
                                linkedin_url = match_ln.group()

                        # --- Crawl new internal links ---
                        for a_tag in soup.find_all("a", href=True):
                            href = a_tag["href"]
                            full_url = urljoin(current_url, href)
                            parsed_url = urlparse(full_url)
                            if parsed_url.netloc == base_domain and full_url not in visited_urls:
                                urls_to_visit.append((full_url, depth + 1))

                except Exception:
                    continue

    except Exception:
        pass
    finally:
        email_str = "No Email Found" if not collected_emails else " * ".join(sorted(collected_emails))
        facebook_str = facebook_url if facebook_url else "No Facebook Found"
        linkedin_str = linkedin_url if linkedin_url else "No LinkedIn Found"

        result = {
            "Website": url,
            "Emails": email_str,
            "Facebook URL": facebook_str,
            "LinkedIn URL": linkedin_str,
            "Pages Scanned": len(visited_urls)
        }

        results.append(result)
        save_to_local_storage(results)
        email_df_container.dataframe(pd.DataFrame(results))
        status['scanned'] += 1

# --- Async Processor ---
async def process_all_urls(urls, status, results, email_df_container, unique_emails, max_pages):
    semaphore = asyncio.Semaphore(5)
    async with aiohttp.ClientSession() as session:
        tasks = [
            crawl_website(url, session, semaphore, status, results, email_df_container, unique_emails, max_pages)
            for url in urls
        ]
        await asyncio.gather(*tasks)

# --- Download Utility ---
def prepare_download_data(results):
    df = pd.DataFrame(results)
    output = BytesIO()
    df.to_csv(output, index=False)
    return output.getvalue(), "text/csv", "emails_social_links.csv"

# --- Streamlit UI ---
uploaded_file = st.file_uploader("Upload CSV or Excel File With URLs", type=["csv", "xlsx"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.success("File Loaded")
        st.write("**Preview:**", df.head())

        url_column = st.selectbox("Select URL Column", df.columns)
        max_pages = st.number_input("Maximum Pages to Scrape per Website", min_value=1, max_value=100, value=20, step=1)
        st.markdown("""
<style>* {margin: 0px; padding: 0px;}</style>
<p>The Number of maximum pages is directly proportional to the speed of Tool.</p>
<p>∴ Max Pages ∝ Tool Speed ∝ Emails Efficiency</p>
<p style='color:var(--indigo-color);'>- Developer</p>
""", unsafe_allow_html=True)

        if st.button("Start Extraction"):
            url_list = df[url_column].dropna().astype(str).tolist()
            total_urls = len(url_list)
            status = {"scanned": 0, "current": ""}
            unique_emails = set()
            results = []

            progress = st.progress(0)
            status_msg = st.empty()
            current_url_display = st.empty()
            estimate_time_display = st.empty()
            email_df_container = st.empty()
            valid_count_display = st.empty()

            start_time = time.time()

            async def update_ui():
                while status["scanned"] < total_urls:
                    elapsed = time.time() - start_time
                    percent = int((status["scanned"] / total_urls) * 100)
                    avg_time = elapsed / max(1, status["scanned"])
                    remaining = avg_time * (total_urls - status["scanned"])
                    mins, secs = divmod(int(remaining), 60)

                    progress.progress(min(percent, 100))
                    status_msg.markdown(f"Scanned Websites: **{status['scanned']} / {total_urls}**")
                    current_url_display.markdown(f"Currently Scanning: `{status['current']}`")
                    valid_count_display.markdown(f"Valid Emails Extracted: **{len(unique_emails)}**")
                    estimate_time_display.markdown(f"Estimated Time Remaining: **{mins} min {secs} sec**")
                    await asyncio.sleep(0.5)

            async def main_runner():
                await asyncio.gather(
                    process_all_urls(url_list, status, results, email_df_container, unique_emails, max_pages),
                    update_ui()
                )

            with st.spinner("Extracting..."):
                try:
                    asyncio.run(main_runner())
                except Exception as e:
                    st.error("Crash detected. Auto-saving current results.")
                    download_partial_results(results)
                    raise e

            st.success(f"Completed: {len(unique_emails)} unique emails found in {status['scanned']} websites scanned.")
            st.markdown("---")
            st.subheader("Download Full Results")
            file_data, mime_type, file_name = prepare_download_data(results)
            st.download_button("Download CSV", file_data, file_name, mime_type)

    except Exception as e:
        st.error(f"Error while processing file: {e}")
else:
    st.write("")
