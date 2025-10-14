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

# --- CSS Loader ---
def load_css():
    try:
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        pass

load_css()

# --- Regex Patterns ---
EMAIL_REGEX = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.(?:com|org|net|edu|gov|io|co|us|uk|ca|de|au|biz|info|ai|app|in|pk|nl|fr|it|es|se|ch|no|dk|pl|be|cz|at|ru|jp|cn|sg|hk|my|id|ph|za|br|mx|tr|ar|vn|gr|ro|pt|fi|ir|sa|nz)",
    re.IGNORECASE,
)
FACEBOOK_REGEX = re.compile(r"https?://(www\.)?facebook\.com/[a-zA-Z0-9_\-./]+")
LINKEDIN_REGEX = re.compile(r"https?://(www\.)?linkedin\.com/[a-zA-Z0-9_\-./]+")
PRIVACY_EMAIL_REGEX = re.compile(r"(privacy|dpo|data\.protection|gdpr|compliance)@", re.IGNORECASE)

# --- Helpers ---
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

# --- Extractor Core ---
async def crawl_website(url, session, semaphore, status, results, email_df_container, unique_emails, max_pages):
    collected_emails = set()
    facebook_url = ""
    linkedin_url = ""
    visited_urls = set()
    urls_to_visit = deque([(url, 0)])
    base_domain = urlparse(url).netloc
    priority_paths = ["/contact", "/about", "/team", "/support", "/get-in-touch", "/contact-us"]

    async with semaphore:
        try:
            for path in priority_paths:
                full_url = urljoin(url, path)
                if full_url not in visited_urls:
                    urls_to_visit.append((full_url, 0))

            while urls_to_visit and len(visited_urls) < max_pages:
                current_url, depth = urls_to_visit.popleft()
                if current_url in visited_urls or depth > 3:
                    continue

                visited_urls.add(current_url)
                status["current"] = current_url

                try:
                    async with session.get(current_url, timeout=10) as response:
                        if response.status != 200:
                            continue
                        html = await response.text(errors="ignore")
                        soup = BeautifulSoup(html, "html.parser")

                        # Remove script/style
                        for tag in soup(["script", "style", "noscript"]):
                            tag.decompose()

                        text = soup.get_text(separator=" ")

                        # --- Extract from text + mailto links ---
                        found_emails = set(EMAIL_REGEX.findall(text))
                        mailto_links = {
                            a["href"].replace("mailto:", "")
                            for a in soup.find_all("a", href=True)
                            if "mailto:" in a["href"]
                        }
                        found_emails.update(mailto_links)

                        # --- Clean emails ---
                        cleaned_emails = set()
                        for email in found_emails:
                            email = email.strip(".,;:()[]{}<>\"'! ")
                            if not PRIVACY_EMAIL_REGEX.search(email):
                                if re.fullmatch(EMAIL_REGEX, email):
                                    cleaned_emails.add(email.lower())

                        collected_emails.update(cleaned_emails)
                        unique_emails.update(cleaned_emails)

                        # --- Social Links ---
                        if not facebook_url:
                            fb_match = FACEBOOK_REGEX.search(html)
                            if fb_match:
                                facebook_url = fb_match.group()
                        if not linkedin_url:
                            ln_match = LINKEDIN_REGEX.search(html)
                            if ln_match:
                                linkedin_url = ln_match.group()

                        # --- Crawl internal links ---
                        for a_tag in soup.find_all("a", href=True):
                            href = a_tag["href"]
                            full_url = urljoin(current_url, href)
                            parsed = urlparse(full_url)
                            if parsed.netloc == base_domain and full_url not in visited_urls:
                                urls_to_visit.append((full_url, depth + 1))

                except Exception:
                    continue

        except Exception:
            pass
        finally:
            result = {
                "Website": url,
                "Emails": " * ".join(sorted(collected_emails)) if collected_emails else "No Email Found",
                "Facebook URL": facebook_url if facebook_url else "No Facebook Found",
                "LinkedIn URL": linkedin_url if linkedin_url else "No LinkedIn Found",
                "Pages Scanned": len(visited_urls),
            }
            results.append(result)
            save_to_local_storage(results)
            email_df_container.dataframe(pd.DataFrame(results))
            status["scanned"] += 1

# --- Async Runner ---
async def process_all_urls(urls, status, results, email_df_container, unique_emails, max_pages):
    semaphore = asyncio.Semaphore(5)
    async with aiohttp.ClientSession() as session:
        tasks = [
            crawl_website(url, session, semaphore, status, results, email_df_container, unique_emails, max_pages)
            for url in urls
        ]
        await asyncio.gather(*tasks)

# --- Download Data ---
def prepare_download_data(results):
    df = pd.DataFrame(results)
    output = BytesIO()
    df.to_csv(output, index=False)
    return output.getvalue(), "text/csv", "emails_social_links.csv"

# --- Streamlit UI ---
st.title("📧 Smart Email Extractor (Async + Streamlit)")
uploaded_file = st.file_uploader("Upload CSV or Excel file with URLs", type=["csv", "xlsx"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.success("✅ File Loaded Successfully")
        st.write("**Preview:**", df.head())

        url_column = st.selectbox("Select URL Column", df.columns)
        max_pages = st.number_input("Maximum Pages to Scan per Website", min_value=1, max_value=100, value=15)

        if st.button("🚀 Start Extraction"):
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
                    status_msg.markdown(f"**Scanned Websites:** {status['scanned']} / {total_urls}")
                    current_url_display.markdown(f"**Currently Scanning:** `{status['current']}`")
                    valid_count_display.markdown(f"**Emails Found So Far:** {len(unique_emails)}")
                    estimate_time_display.markdown(f"**Estimated Time Remaining:** {mins}m {secs}s")
                    await asyncio.sleep(0.5)

            async def main_runner():
                await asyncio.gather(
                    process_all_urls(url_list, status, results, email_df_container, unique_emails, max_pages),
                    update_ui()
                )

            with st.spinner("🔍 Extracting emails... please wait"):
                try:
                    asyncio.run(main_runner())
                except Exception as e:
                    st.error("⚠️ Crash detected — auto-saving current results.")
                    download_partial_results(results)
                    raise e

            st.success(f"✅ Completed: {len(unique_emails)} total emails found from {status['scanned']} websites.")
            st.markdown("---")
            file_data, mime_type, file_name = prepare_download_data(results)
            st.download_button("⬇️ Download Results", file_data, file_name, mime_type)

    except Exception as e:
        st.error(f"Error: {e}")
