import streamlit as st
import pandas as pd
import asyncio
import aiohttp
import re
import time
from bs4 import BeautifulSoup
from io import BytesIO
from urllib.parse import urljoin, urlparse
from collections import deque

# ────────────────────────────────────────────────
#  SESSION STATE — only temporary, cleared on rerun
# ────────────────────────────────────────────────
if 'scraping_results' not in st.session_state:
    st.session_state.scraping_results = []
if 'unique_emails' not in st.session_state:
    st.session_state.unique_emails = set()
if 'scraping_complete' not in st.session_state:
    st.session_state.scraping_complete = False

# ─── CSS (optional) ───
def load_css():
    try:
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        pass
load_css()

# ─── Regex ───
EMAIL_REGEX = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.(?:com|org|net|edu|gov|io|co|us|uk|ca|de|au|biz|info|ai|app|in|pk|nl|fr|it|es|se|ch|no|dk|pl|be|cz|at|ru|jp|cn|sg|hk|my|id|ph|za|br|mx|tr|ar|vn|gr|ro|pt|fi|ir|sa|nz)",
    re.IGNORECASE,
)
FACEBOOK_REGEX = re.compile(r"https?://(www\.)?facebook\.com/[a-zA-Z0-9_\-./]+")
LINKEDIN_REGEX  = re.compile(r"https?://(www\.)?linkedin\.com/[a-zA-Z0-9_\-./]+")
PRIVACY_EMAIL_REGEX = re.compile(r"(privacy|dpo|data\.protection|gdpr|compliance)@", re.IGNORECASE)

# ─── Reset function (used when starting new run) ───
def reset_scraping_state():
    st.session_state.scraping_results = []
    st.session_state.unique_emails = set()
    st.session_state.scraping_complete = False

# ─── Core crawler ───
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
            # Give priority to contact/about pages
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

                        for tag in soup(["script", "style", "noscript"]):
                            tag.decompose()

                        text = soup.get_text(separator=" ")

                        # Emails from text + mailto:
                        found_emails = set(EMAIL_REGEX.findall(text))
                        mailto_links = {
                            a["href"].replace("mailto:", "")
                            for a in soup.find_all("a", href=True)
                            if "mailto:" in a["href"]
                        }
                        found_emails.update(mailto_links)

                        # Clean
                        cleaned = set()
                        for email in found_emails:
                            email = email.strip(".,;:()[]{}<>\"'! ")
                            if not PRIVACY_EMAIL_REGEX.search(email):
                                if re.fullmatch(EMAIL_REGEX, email):
                                    cleaned.add(email.lower())

                        collected_emails.update(cleaned)
                        unique_emails.update(cleaned)

                        # Social
                        if not facebook_url:
                            fb_match = FACEBOOK_REGEX.search(html)
                            if fb_match:
                                facebook_url = fb_match.group()
                        if not linkedin_url:
                            ln_match = LINKEDIN_REGEX.search(html)
                            if ln_match:
                                linkedin_url = ln_match.group()

                        # Follow internal links
                        for a in soup.find_all("a", href=True):
                            full_url = urljoin(current_url, a["href"])
                            parsed = urlparse(full_url)
                            if parsed.netloc == base_domain and full_url not in visited_urls:
                                urls_to_visit.append((full_url, depth + 1))

                except Exception:
                    continue

        finally:
            result = {
                "Website": url,
                "Emails Found": " * ".join(sorted(collected_emails)) if collected_emails else "No Email Found",
                "Facebook Link": facebook_url or "No Facebook Found",
                "LinkedIn Link": linkedin_url or "No LinkedIn Found",
                "Pages Scanned": len(visited_urls),
            }
            results.append(result)

            if email_df_container:
                email_df_container.dataframe(pd.DataFrame(results))

            status["scanned"] += 1

# ─── Runner ───
async def process_all_urls(urls, status, results, email_df_container, unique_emails, max_pages):
    semaphore = asyncio.Semaphore(5)
    async with aiohttp.ClientSession() as session:
        tasks = [
            crawl_website(u, session, semaphore, status, results, email_df_container, unique_emails, max_pages)
            for u in urls
        ]
        await asyncio.gather(*tasks)

# ─── Download prep ───
def prepare_download_data():
    df = pd.DataFrame(st.session_state.scraping_results)
    output = BytesIO()
    df.to_csv(output, index=False)
    return output.getvalue(), "text/csv", "emails_social_links.csv"

# ────────────────────────────────────────────────
#                 MAIN UI
# ────────────────────────────────────────────────
st.markdown("<p class='h1'>Lead <span>Extractor</span></p>", unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file with URLs",
    type=["csv", "xlsx"],
    help="Upload to start scraping (previous data is cleared)"
)

if uploaded_file:
    reset_scraping_state()           # ← Always start fresh

    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.write("Preview:", df.head())

        url_column = st.selectbox("Select URL Column", df.columns)
        max_pages = st.number_input("Max Pages per Website", 1, 100, 15)

        if st.button("Start Extraction", type="primary"):
            url_list = df[url_column].dropna().astype(str).tolist()
            total_urls = len(url_list)

            if total_urls == 0:
                st.warning("No valid URLs found in selected column.")
                st.stop()

            status = {"scanned": 0, "current": ""}

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
                    status_msg.markdown(f"**Scanned:** {status['scanned']} / {total_urls}")
                    current_url_display.markdown(f"**Scanning:** `{status['current']}`")
                    valid_count_display.markdown(f"**Unique emails so far:** {len(st.session_state.unique_emails)}")
                    estimate_time_display.markdown(f"**ETA:** {mins}m {secs}s")

                    await asyncio.sleep(0.4)

            async def main_runner():
                await asyncio.gather(
                    process_all_urls(
                        url_list,
                        status,
                        st.session_state.scraping_results,
                        email_df_container,
                        st.session_state.unique_emails,
                        max_pages
                    ),
                    update_ui()
                )

            with st.spinner("Extracting emails & social links..."):
                try:
                    asyncio.run(main_runner())
                    st.session_state.scraping_complete = True
                except Exception as e:
                    st.error(f"Extraction failed: {e}")
                    raise

            st.success(f"Done. Found {len(st.session_state.unique_emails)} unique emails across {status['scanned']} sites.")

            if st.session_state.scraping_results:
                st.markdown("---")
                st.dataframe(pd.DataFrame(st.session_state.scraping_results))

                data, mime, fname = prepare_download_data()
                st.download_button(
                    "Download CSV",
                    data,
                    fname,
                    mime,
                    key="download_final"
                )

    except Exception as e:
        st.error(f"File reading error: {e}")

# ─── Sidebar ───
with st.sidebar:
    st.header("Controls")

    if st.session_state.scraping_complete:
        st.success("Results ready")
        st.write(f"Websites: {len(st.session_state.scraping_results)}")
        st.write(f"Unique emails: {len(st.session_state.unique_emails)}")

    if st.button("Clear Results", type="secondary"):
        reset_scraping_state()
        st.success("Results cleared")
        st.rerun()

st.markdown("---")
st.caption("No data is saved between sessions")
