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
# ── Session State ───────────────────────────────────────────────
defaults = {
    'scraping_results': [],
    'unique_emails': set(),
    'uploaded_file_hash': None,
    'current_file_name': None,
    'scraping_complete': False,
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── CSS (you can keep your style.css or use this minimal version) ──
st.markdown("""
    <style>
        .stProgress > div > div > div {
            background-color: #4CAF50;
        }
        .block-container {
            padding-top: 1.2rem !important;
            padding-bottom: 2rem !important;
        }
        h1, h2, h3 {
            margin-bottom: 0.6rem;
        }
        .stButton > button {
            margin-right: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# ── Regex Patterns ──────────────────────────────────────────────
EMAIL_REGEX = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.(?:com|org|net|edu|gov|io|co|us|uk|ca|de|au|biz|info|ai|app|in|pk|nl|fr|it|es|se|ch|no|dk|pl|be|cz|at|ru|jp|cn|sg|hk|my|id|ph|za|br|mx|tr|ar|vn|gr|ro|pt|fi|ir|sa|nz)",
    re.IGNORECASE
)
FACEBOOK_REGEX = re.compile(r"https?://(www\.)?facebook\.com/[a-zA-Z0-9_\-./]+")
LINKEDIN_REGEX  = re.compile(r"https?://(www\.)?linkedin\.com/[a-zA-Z0-9_\-./]+")
PRIVACY_EMAIL_REGEX = re.compile(r"(privacy|dpo|data\.protection|gdpr|compliance)@", re.IGNORECASE)

# ── Persistent Storage ──────────────────────────────────────────
def save_state():
    data = {
        'results': st.session_state.scraping_results,
        'unique_emails': list(st.session_state.unique_emails),
        'file_name': st.session_state.current_file_name,
        'file_hash': st.session_state.uploaded_file_hash
    }
    with open("persistent_storage.json", "w") as f:
        json.dump(data, f)

def load_state():
    if not os.path.exists("persistent_storage.json"):
        return False
    with open("persistent_storage.json", "r") as f:
        data = json.load(f)
    st.session_state.scraping_results   = data.get('results', [])
    st.session_state.unique_emails      = set(data.get('unique_emails', []))
    st.session_state.current_file_name  = data.get('file_name')
    st.session_state.uploaded_file_hash = data.get('file_hash')
    st.session_state.scraping_complete  = len(st.session_state.scraping_results) > 0
    return True

def reset_state():
    st.session_state.scraping_results   = []
    st.session_state.unique_emails      = set()
    st.session_state.scraping_complete  = False
    st.session_state.uploaded_file_hash = None
    st.session_state.current_file_name  = None

# ── Core crawler logic remains almost unchanged ─────────────────
# (omitted here for brevity — keep your crawl_website and process_all_urls functions)

# ── Helpers ─────────────────────────────────────────────────────
def prepare_csv_download():
    if not st.session_state.scraping_results:
        return None, None, None
    df = pd.DataFrame(st.session_state.scraping_results)
    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return output.getvalue(), "text/csv", "email_social_export.csv"

# ── Load previous data ──────────────────────────────────────────
load_state()

# ── Main UI ─────────────────────────────────────────────────────
st.title("Website Email & Social Link Extractor")

# ── Previous results ────────────────────────────────────────────
if st.session_state.scraping_complete and st.session_state.scraping_results:
    st.subheader("Previous Results")
    st.dataframe(pd.DataFrame(st.session_state.scraping_results), use_container_width=True)

    csv_data, mime, fname = prepare_csv_download()

    cols = st.columns([1,1,4])
    with cols[0]:
        st.download_button(
            "Download Results",
            csv_data,
            fname,
            mime,
            key="dl_prev"
        )
    with cols[1]:
        if st.button("Clear Results", type="secondary"):
            if os.path.exists("persistent_storage.json"):
                os.remove("persistent_storage.json")
            reset_state()
            save_state()
            st.rerun()

st.divider()

# ── New scraping section ────────────────────────────────────────
st.subheader("New Extraction")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file containing URLs",
    type=["csv", "xlsx"],
    help="One URL per row — select the correct column below"
)

if uploaded_file is not None:
    file_content = uploaded_file.getvalue()
    current_hash = hashlib.md5(file_content).hexdigest()

    if current_hash != st.session_state.uploaded_file_hash:
        reset_state()
        st.session_state.uploaded_file_hash = current_hash
        st.session_state.current_file_name  = uploaded_file.name

    uploaded_file.seek(0)

    try:
        df_input = pd.read_csv(uploaded_file) if uploaded_file.name.lower().endswith('.csv') else pd.read_excel(uploaded_file)
        st.success(f"File loaded: {uploaded_file.name}  ({len(df_input)} rows)")

        st.write("First few rows:")
        st.dataframe(df_input.head(5), use_container_width=True)

        url_column = st.selectbox("URL column", options=list(df_input.columns), index=0)
        max_pages  = st.number_input("Max pages per website", 1, 80, 12, step=1)

        st.divider()

        if st.button("Start Extraction", type="primary", use_container_width=True):
            urls = df_input[url_column].dropna().astype(str).tolist()
            if not urls:
                st.error("No valid URLs found in selected column.")
                st.stop()

            status = {"scanned": 0, "current": ""}

            progress       = st.progress(0)
            status_text    = st.empty()
            current_url_ui = st.empty()
            eta_ui         = st.empty()
            emails_so_far  = st.empty()
            results_table  = st.empty()

            start_time = time.time()

            async def ui_updater():
                while status["scanned"] < len(urls):
                    elapsed = time.time() - start_time
                    perc = status["scanned"] / len(urls)
                    avg = elapsed / max(1, status["scanned"])
                    eta_sec = avg * (len(urls) - status["scanned"])
                    m, s = divmod(int(eta_sec), 60)

                    progress.progress(perc)
                    status_text.markdown(f"**Processed:** {status['scanned']} / {len(urls)}")
                    current_url_ui.markdown(f"**Now scanning:**  {status['current'] or '—'}")
                    emails_so_far.markdown(f"**Unique emails found:** {len(st.session_state.unique_emails)}")
                    eta_ui.markdown(f"**ETA:** {m} min {s} sec")
                    await asyncio.sleep(0.4)

            async def runner():
                await asyncio.gather(
                    process_all_urls(
                        urls,
                        status,
                        st.session_state.scraping_results,
                        results_table,
                        st.session_state.unique_emails,
                        max_pages
                    ),
                    ui_updater()
                )

            with st.spinner("Extracting data..."):
                try:
                    asyncio.run(runner())
                    st.session_state.scraping_complete = True
                    save_state()
                except Exception as exc:
                    st.error("Extraction stopped with error — partial results saved.")
                    save_state()
                    raise

            st.success(f"Completed — {len(st.session_state.unique_emails)} unique emails found")

            if st.session_state.scraping_results:
                st.subheader("Extraction Results")
                st.dataframe(pd.DataFrame(st.session_state.scraping_results), use_container_width=True)

                csv_data, mime, fname = prepare_csv_download()
                if csv_data:
                    st.download_button(
                        "Download CSV",
                        csv_data,
                        fname,
                        mime,
                        key="dl_final",
                        use_container_width=True
                    )

    except Exception as e:
        st.error(f"File reading error: {e}")

st.divider()

st.caption("Results are automatically saved to persistent_storage.json")
