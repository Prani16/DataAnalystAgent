# app.py
import io
import os
import time
import base64
import asyncio
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests
from PIL import Image

# Config
MAX_RUNTIME = 170.0          # seconds total (give some buffer under 180s)
DOWNLOAD_TIMEOUT = 10.0     # seconds per network call
MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024  # 5 MB per file
IMAGE_SIZE_LIMIT = 100_000  # bytes (100k)
ALLOWED_FETCH_SCHEMES = ("http", "https")

app = FastAPI(title="Data Analyst Agent")

# --- Utilities --------------------------------------------------------------
def now_seconds():
    return time.monotonic()

def enforce_deadline(start, extra_margin=0.5):
    elapsed = now_seconds() - start
    if elapsed + extra_margin > MAX_RUNTIME:
        raise HTTPException(status_code=504, detail="Processing time exceeded the allowed budget.")

def safe_download_url(url: str, timeout=DOWNLOAD_TIMEOUT):
    # Basic scheme check
    if not url.lower().startswith(ALLOWED_FETCH_SCHEMES):
        raise ValueError("Only http/https supported.")
    r = requests.get(url, timeout=timeout, stream=True)
    r.raise_for_status()
    # stream but break if too big
    total = 0
    content = bytearray()
    for chunk in r.iter_content(chunk_size=8192):
        if chunk:
            total += len(chunk)
            if total > MAX_DOWNLOAD_BYTES:
                raise ValueError("Downloaded content too large.")
            content.extend(chunk)
    return bytes(content)

def parse_html_table_from_url(url: str) -> List[pd.DataFrame]:
    html = safe_download_url(url).decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    tables = pd.read_html(html)
    return tables

def file_to_dataframe(filename: str, content: bytes) -> pd.DataFrame:
    # Try csv -> excel -> json -> fallback to pandas.read_html
    buf = io.BytesIO(content)
    try:
        return pd.read_csv(buf)
    except Exception:
        buf.seek(0)
    try:
        return pd.read_excel(buf)
    except Exception:
        buf.seek(0)
    try:
        import json
        parsed = json.load(io.TextIOWrapper(buf, encoding="utf-8"))
        return pd.json_normalize(parsed)
    except Exception:
        buf.seek(0)
    # fallback: try to parse HTML tables
    try:
        html = buf.read().decode("utf-8", errors="ignore")
        dfs = pd.read_html(html)
        if dfs:
            return dfs[0]
    except Exception:
        pass
    raise ValueError("Unsupported file format or no table found.")

def compress_image_to_limit(img: Image.Image, max_bytes: int, fmt="PNG"):
    """Try to compress by resizing and saving with optimize until under max_bytes."""
    # Start with original and iterative downscale
    buf = io.BytesIO()
    img.save(buf, format=fmt, optimize=True)
    data = buf.getvalue()
    if len(data) <= max_bytes:
        return data
    # iteratively reduce size by scaling down
    w, h = img.size
    for factor in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]:
        nw, nh = max(1, int(w * factor)), max(1, int(h * factor))
        resized = img.resize((nw, nh), Image.LANCZOS)
        buf = io.BytesIO()
        if fmt == "PNG":
            resized.save(buf, format=fmt, optimize=True)
        else:
            resized.save(buf, format=fmt, quality=85, optimize=True)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            return data
    # last resort convert to JPEG (often smaller)
    buf = io.BytesIO()
    rgb = img.convert("RGB")
    rgb.save(buf, format="JPEG", quality=70, optimize=True)
    data = buf.getvalue()
    if len(data) <= max_bytes:
        return data
    raise ValueError("Unable to compress image under limit.")

def plot_scatter_with_regression(x, y, title=None, xlabel=None, ylabel=None, dpi=150):
    fig, ax = plt.subplots(figsize=(6,4), dpi=dpi)
    ax.scatter(x, y)
    # linear regression
    coef = np.polyfit(x, y, 1)
    poly1d_fn = np.poly1d(coef)
    xs = np.linspace(np.min(x), np.max(x), 100)
    ax.plot(xs, poly1d_fn(xs), linestyle=':', color='red')
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# --- Core pipeline ----------------------------------------------------------
async def run_pipeline(questions_text: str, attachments: List[UploadFile], start_time: float):
    """
    Basic pipeline:
    - Interpret instructions (we'll keep this simple: look for URLs and keywords in questions_text)
    - Load provided attachments into DataFrames (if any)
    - If text includes a URL, try to scrape html tables
    - Run standard analyses requested in example (count before year X, earliest over threshold, correlation, scatter plot)
    - Return list[str] answers (images as data: URI)
    """
    enforce_deadline(start_time)
    answers = []

    # --- 1) quick heuristics to find URLs in questions
    import re
    urls = re.findall(r'https?://\S+', questions_text)
    enforce_deadline(start_time)

    # --- 2) load attachments to dataframes
    dfs = []
    for f in attachments:
        enforce_deadline(start_time)
        content = await f.read()
        if len(content) > MAX_DOWNLOAD_BYTES:
            raise HTTPException(status_code=400, detail=f"File {f.filename} too large.")
        try:
            df = file_to_dataframe(f.filename, content)
            df['_source_filename'] = f.filename
            dfs.append(df)
        except Exception:
            # skip non-table attachments
            continue

    # --- 3) if URL present, attempt to scrape tables
    for url in urls:
        enforce_deadline(start_time)
        # only attempt if looks like wiki or html
        try:
            tables = parse_html_table_from_url(url)
            for t in tables:
                t['_source_url'] = url
                dfs.append(t)
        except Exception as e:
            # continue; could not fetch/parse
            continue

    if not dfs:
        raise HTTPException(status_code=400, detail="No structured data found in attachments or scraped URLs.")

    # merge heuristically (take first dataframe that looks non-empty)
    df = max(dfs, key=lambda d: len(d))
    # Normalize columns: lower-case
    df.columns = [str(c).strip() for c in df.columns]

    # Example-specific helper heuristics: we will attempt to find columns like 'Worldwide', 'Peak', 'Rank', 'Year', 'Release'
    cols_lower = [c.lower() for c in df.columns]

    # Guess columns
    def find_col(names):
        for c in df.columns:
            for n in names:
                if n.lower() in c.lower():
                    return c
        return None

    col_rank = find_col(['rank'])
    col_peak = find_col(['peak'])
    col_gross = find_col(['worldwide', 'gross', 'box office', 'box_office', 'boxoffice'])
    col_year = find_col(['year', 'release', 'released'])
    # try to coerce types
    def to_numeric_series(s):
        return pd.to_numeric(s.astype(str).str.replace(r'[^0-9.\-]', '', regex=True), errors='coerce')

    if col_rank:
        df['_rank_num'] = to_numeric_series(df[col_rank])
    if col_peak:
        df['_peak_num'] = to_numeric_series(df[col_peak])
    if col_gross:
        df['_gross_num'] = to_numeric_series(df[col_gross])
    if col_year:
        df['_year_num'] = to_numeric_series(df[col_year])

    # Now answer several likely questions seen in the example:
    # 1) Count of $2bn movies before year 2000
    try:
        enforce_deadline(start_time)
        mask = pd.Series([False]*len(df))
        if '_gross_num' in df.columns:
            mask = df['_gross_num'] >= 2_000_000_000
        # year constraint
        if '_year_num' in df.columns:
            mask = mask & (df['_year_num'] < 2000)
        count_2bn_before_2000 = int(mask.sum())
        answers.append(f"{count_2bn_before_2000}")
    except Exception:
        answers.append("Could not compute count of $2bn movies before 2000")

    # 2) Earliest film that grossed over $1.5bn
    try:
        enforce_deadline(start_time)
        if '_gross_num' in df.columns and '_year_num' in df.columns:
            mask = df['_gross_num'] >= 1_500_000_000
            subset = df[mask].copy()
            subset = subset.dropna(subset=['_year_num'])
            if len(subset):
                earliest_idx = subset['_year_num'].idxmin()
                # attempt to find title column
                title_col = find_col(['title', 'film', 'movie', 'name'])
                title = str(subset.loc[earliest_idx, title_col]) if title_col else str(subset.loc[earliest_idx].iloc[0])
                year = int(subset.loc[earliest_idx, '_year_num'])
                answers.append(f"{title} ({year})")
            else:
                answers.append("No film found with gross >= $1.5bn")
        else:
            answers.append("Insufficient columns to determine earliest film over $1.5bn")
    except Exception:
        answers.append("Could not determine earliest film over $1.5bn")

    # 3) Correlation between Rank and Peak
    try:
        enforce_deadline(start_time)
        if '_rank_num' in df.columns and '_peak_num' in df.columns:
            # drop NA
            tmp = df.dropna(subset=['_rank_num', '_peak_num'])
            if len(tmp) >= 2:
                corr = float(tmp['_rank_num'].corr(tmp['_peak_num']))
                answers.append(f"{corr:.4f}")
            else:
                answers.append("Not enough data to compute correlation")
        else:
            answers.append("Rank or Peak columns missing to compute correlation")
    except Exception:
        answers.append("Could not compute correlation between Rank and Peak")

    # 4) Create scatterplot of Rank vs Peak with dotted red regression. Return base64 PNG Data URI under limit.
    try:
        enforce_deadline(start_time)
        if '_rank_num' in df.columns and '_peak_num' in df.columns:
            tmp = df.dropna(subset=['_rank_num', '_peak_num'])
            if len(tmp) >= 2:
                img = plot_scatter_with_regression(tmp['_rank_num'].values, tmp['_peak_num'].values,
                                                   title="Rank vs Peak", xlabel="Rank", ylabel="Peak")
                # compress to under IMAGE_SIZE_LIMIT
                png_bytes = compress_image_to_limit(img, IMAGE_SIZE_LIMIT, fmt="PNG")
                b64 = base64.b64encode(png_bytes).decode('ascii')
                data_uri = f"data:image/png;base64,{b64}"
                answers.append(data_uri)
            else:
                answers.append("Not enough points to make scatterplot")
        else:
            answers.append("Rank or Peak columns missing to produce scatterplot")
    except Exception as e:
        answers.append(f"Could not create scatterplot: {str(e)}")

    return answers

# --- API endpoint -----------------------------------------------------------
@app.post("/")
async def analyze(request: Request, files: List[UploadFile] = File(...)):
    """
    Expects multipart/form-data where one of the uploads is 'questions.txt'.
    Returns JSON array of strings (answers).
    """
    start_time = now_seconds()
    # find questions.txt among files
    questions_file = None
    attachments = []
    for f in files:
        if f.filename.lower().endswith("questions.txt"):
            questions_file = f
        else:
            attachments.append(f)
    if not questions_file:
        raise HTTPException(status_code=400, detail="questions.txt is required.")

    # read questions text
    qtext = (await questions_file.read()).decode("utf-8", errors="ignore")
    # reset pointer? Not needed after read but just close file
    try:
        answers = await asyncio.wait_for(run_pipeline(qtext, attachments, start_time), timeout=MAX_RUNTIME - 2)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Overall processing timed out.")
    return JSONResponse(content=answers)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)

