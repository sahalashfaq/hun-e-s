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
import hashlib

# --- Initialize Session State ---
if 'scraping_results' not in st.session_state:
    st.session_state.scraping_results = []
if 'unique_emails' not in st.session_state:
    st.session_state.unique_emails = set()
if 'uploaded_file_hash' not in st.session_state:
    st.session_state.uploaded_file_hash = None
if 'current_file_name' not in st.session_state:
    st.session_state.current_file_name = None
if 'scraping_complete' not in st.session_state:
    st.session_state.scraping_complete = False

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

# --- Persistent Storage ---
def save_to_persistent_storage():
    """Save results to both session state and local storage"""
    # Convert set to list for JSON serialization
    data_to_save = {
        'results': st.session_state.scraping_results,
        'unique_emails': list(st.session_state.unique_emails),
        'file_name': st.session_state.current_file_name,
        'file_hash': st.session_state.uploaded_file_hash
    }
    
    # Save to file
    with open("persistent_storage.json", "w") as f:
        json.dump(data_to_save, f)
    
    # Also save to session state
    st.session_state.persisted_data = data_to_save

def load_from_persistent_storage():
    """Load results from persistent storage"""
    if os.path.exists("persistent_storage.json"):
        with open("persistent_storage.json", "r") as f:
            data = json.load(f)
            
            # Restore to session state
            st.session_state.scraping_results = data.get('results', [])
            st.session_state.unique_emails = set(data.get('unique_emails', []))
            st.session_state.current_file_name = data.get('file_name')
            st.session_state.uploaded_file_hash = data.get('file_hash')
            st.session_state.scraping_complete = len(st.session_state.scraping_results) > 0
            
            return True
    return False

def calculate_file_hash(file_content):
    """Calculate hash of uploaded file to detect changes"""
    return hashlib.md5(file_content).hexdigest()

def reset_scraping_state():
    """Reset scraping state for new file"""
    st.session_state.scraping_results = []
    st.session_state.unique_emails = set()
    st.session_state.scraping_complete = False

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
                "Emails Found": " * ".join(sorted(collected_emails)) if collected_emails else "No Email Found",
                "Facebook Link": facebook_url if facebook_url else "No Facebook Found",
                "LinkedIn Link": linkedin_url if linkedin_url else "No LinkedIn Found",
                "Pages Scanned": len(visited_urls),
            }
            results.append(result)
            # Save to persistent storage after each website
            save_to_persistent_storage()
            
            # Update dataframe display
            if email_df_container:
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
def prepare_download_data():
    """Prepare data for download from session state"""
    df = pd.DataFrame(st.session_state.scraping_results)
    output = BytesIO()
    df.to_csv(output, index=False)
    return output.getvalue(), "text/csv", "emails_social_links.csv"

# --- Load persistent data on startup ---
load_from_persistent_storage()

# --- Streamlit UI ---
st.title("Website Email & Social Link Extractor")

# Display existing results if available
if st.session_state.scraping_complete and st.session_state.scraping_results:
    st.info(f"📊 Found {len(st.session_state.scraping_results)} previously scraped results")
    
    # Show results summary
    df_previous = pd.DataFrame(st.session_state.scraping_results)
    st.dataframe(df_previous)
    
    # Download button for existing data
    file_data, mime_type, file_name = prepare_download_data()
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download Existing Results", 
            file_data, 
            file_name, 
            mime_type,
            key="download_existing"
        )
    with col2:
        if st.button("🗑️ Clear Existing Results"):
            reset_scraping_state()
            save_to_persistent_storage()
            st.rerun()

st.markdown("---")

# File uploader
uploaded_file = st.file_uploader(
    "Upload CSV or Excel file with URLs", 
    type=["csv", "xlsx"],
    help="Upload a new file to start fresh scraping"
)

if uploaded_file:
    # Calculate file hash to detect if it's a new file
    file_content = uploaded_file.getvalue()
    file_hash = calculate_file_hash(file_content)
    
    # Check if this is a new file
    is_new_file = (file_hash != st.session_state.uploaded_file_hash)
    
    if is_new_file:
        reset_scraping_state()
        st.session_state.uploaded_file_hash = file_hash
        st.session_state.current_file_name = uploaded_file.name
    
    try:
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Read file
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.success(f"✅ File Loaded: {uploaded_file.name}")
        st.write("Preview:", df.head())

        url_column = st.selectbox("Select URL Column", df.columns)
        max_pages = st.number_input("Maximum Pages to Scan per Website", min_value=1, max_value=100, value=15)
        
        # Show warning if resuming with existing data
        if st.session_state.scraping_results and not is_new_file:
            st.warning(f"⚠️ Found {len(st.session_state.scraping_results)} previously scraped results. New scraping will add to existing data.")

        if st.button("Start Extraction", type="primary"):
            url_list = df[url_column].dropna().astype(str).tolist()
            total_urls = len(url_list)
            status = {"scanned": 0, "current": ""}
            
            # Initialize containers
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
                    valid_count_display.markdown(f"**Emails Found So Far:** {len(st.session_state.unique_emails)}")
                    estimate_time_display.markdown(f"**Estimated Time Remaining:** {mins}m {secs}s")
                    await asyncio.sleep(0.5)

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

            with st.spinner("Extracting emails and social links... please wait"):
                try:
                    asyncio.run(main_runner())
                    st.session_state.scraping_complete = True
                    save_to_persistent_storage()
                    
                except Exception as e:
                    st.error("Error during extraction — saving current results.")
                    save_to_persistent_storage()
                    raise e

            # Final results
            st.success(f"✅ Completed: {len(st.session_state.unique_emails)} total emails found from {status['scanned']} websites.")
            st.markdown("---")
            
            # Display final results
            if st.session_state.scraping_results:
                final_df = pd.DataFrame(st.session_state.scraping_results)
                st.dataframe(final_df)
                
                # Download button
                file_data, mime_type, file_name = prepare_download_data()
                st.download_button(
                    "📥 Download Results", 
                    file_data, 
                    file_name, 
                    mime_type,
                    key="download_final"
                )

    except Exception as e:
        st.error(f"Error: {e}")

# Sidebar for additional options
with st.sidebar:
    st.header("Data Management")
    
    if st.session_state.scraping_complete:
        st.success("✅ Data available for download")
        
        # Show statistics
        st.subheader("Statistics")
        st.write(f"Total Websites: {len(st.session_state.scraping_results)}")
        st.write(f"Unique Emails: {len(st.session_state.unique_emails)}")
        
        # Export options
        st.subheader("📤 Export Options")
        
        if st.button("Export to JSON"):
            data_to_export = {
                'results': st.session_state.scraping_results,
                'unique_emails': list(st.session_state.unique_emails),
                'metadata': {
                    'export_date': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'total_websites': len(st.session_state.scraping_results),
                    'total_emails': len(st.session_state.unique_emails)
                }
            }
            
            json_str = json.dumps(data_to_export, indent=2)
            st.download_button(
                "Download JSON",
                json_str,
                "scraping_results.json",
                "application/json"
            )
    
    # Clear all data
    st.markdown("---")
    if st.button("Clear All Data", type="secondary"):
        reset_scraping_state()
        st.session_state.uploaded_file_hash = None
        st.session_state.current_file_name = None
        
        # Clear persistent storage file
        if os.path.exists("persistent_storage.json"):
            os.remove("persistent_storage.json")
        
        st.success("All data cleared!")
        st.rerun()

# Footer
st.markdown("---")
st.caption("🔒 Data is automatically saved and persists across sessions")
