
"""
collect_pairs.py
Builds two datasets:
  1. Abstract → 100–200 word summary
  2. Full-text → 200–300 word summary

Sources of summaries:
  - SciTLDR (curated, currently disabled)
  - Semantic Scholar TLDRs
  - GPT-4o-mini fallback (expand/write)

Dependencies:
  - aiohttp, asyncio
  - openai>=1.0.0
  - pyarrow
  - python-dotenv
  - arxiv
  - GROBID running on localhost:8070 (for full-text parsing)
"""

import os, asyncio, requests
from pathlib import Path
from dotenv import load_dotenv
import aiohttp
from openai import AsyncOpenAI
import pyarrow as pa
import pyarrow.parquet as pq

load_dotenv()

# ---------------- Config ----------------
S2_KEY = os.getenv("S2_API_KEY")
GROBID_URL = "http://127.0.0.1:8070/api"
PDF_DIR = Path("pdfs"); PDF_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
RES_DIR = Path("resources"); RES_DIR.mkdir(exist_ok=True)

# GPT-4o-mini costs (Sep 2025)
COST_INPUT_USD_PER_TOKEN  = 0.15 / 1e6
COST_OUTPUT_USD_PER_TOKEN = 0.60 / 1e6

# ---------------- OpenAI Async Client ----------------
_async_oai = None
def _async_client() -> AsyncOpenAI:
    global _async_oai
    if _async_oai is None:
        _async_oai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _async_oai

# ---------------- SciTLDR (disabled placeholder) ----------------
def load_scitldr() -> dict[str, str]:
    print("[INFO] SciTLDR temporarily disabled. Using S2 TLDR + GPT fallback.")
    return {}

# ---------------- Semantic Scholar (batch TLDRs) ----------------
async def s2_fetch_tldr_batch(arxiv_ids: list[str], session: aiohttp.ClientSession) -> dict[str, str]:
    if not S2_KEY:
        print("batch, no s2_key")
        return {}

    url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    params = {"fields": "tldr,title,externalIds"}
    headers = {"x-api-key": S2_KEY, "Content-Type": "application/json"}
    payload = {"ids": [f"arXiv:{aid}" for aid in arxiv_ids]}

    out: dict[str, str] = {}

    async with session.post(url, params=params, headers=headers, json=payload, timeout=30) as r:
        if r.status != 200:
            print(f"s2_batch r.status: {r.status}")
            return out

        items = await r.json()

        # items is a list, but may contain None
        for i, item in enumerate(items):
            if item is None:
                # No TLDR for this pid
                pid = arxiv_ids[i]
                out[pid] = ""   # mark empty
                continue

            ext = item.get("externalIds") or {}
            aid = ext.get("ArXiv") or ext.get("arXiv")
            tldr = (item.get("tldr") or {}).get("text")

            if aid:
                out[aid] = tldr or ""   # store empty string if no TLDR

    return out

async def s2_fetch_tldr_for_ids(arxiv_ids: list[str], batch_size: int = 100) -> dict[str, str]:
    if not S2_KEY:
        print("missing s2_key")
        return {}
    results = {}
    connector = aiohttp.TCPConnector(limit=8)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i in range(0, len(arxiv_ids), batch_size):
            chunk = arxiv_ids[i:i+batch_size]
            tasks.append(s2_fetch_tldr_batch(chunk, session))
        for coro in asyncio.as_completed(tasks):
            try:
                part = await coro
                results.update(part)
            except Exception as e:
                print(f"exception in s2: {e}")
                continue
    return results

# ---------------- GROBID (fulltext extraction) ----------------
def grobid_fulltext(pdf_path: Path) -> str:
    files = {"input": open(pdf_path, "rb")}
    try:
        r = requests.post(f"{GROBID_URL}/processFulltextDocument", files=files, timeout=60)
    finally:
        files["input"].close()
    if r.status_code != 200:
        print(f"[GROBID WARN] {pdf_path.name}: HTTP {r.status_code}")
        return ""
    return r.text

# ---------------- GPT fallback (ASYNC, with optional TLDR expansion) ----------------
async def gpt_expand_async(base_text: str, title: str, min_w: int, max_w: int, cost_tracker: dict, tldr: str = "") -> str:
    """
    Async GPT-4o-mini expansion/writing with token cost accounting.
    If `tldr` is provided but too short, the model is asked to expand/augment it.
    If `tldr` is empty, the model writes a summary from scratch.
    """

    if not os.getenv("OPENAI_API_KEY"):
        return base_text

    if tldr and tldr.strip():
        # Expand the TLDR into a longer, plain-language summary
        user_prompt = (
            f"The following is a short TLDR of a research paper. Expand it into a {min_w}-{max_w} word summary "
            f"for the general public. Make it accurate, clear, and avoid jargon. Incorporate context from the paper’s title "
            f"and abstract/full text if relevant.\n\n"
            f"Title: {title}\n\n"
            f"TLDR: {tldr}\n\n"
            f"Reference text:\n{base_text}"
        )
    else:
        # Write a new summary from scratch
        user_prompt = (
            f"Write a {min_w}-{max_w} word plain-language summary for the general public.\n"
            f"Be faithful to the source, avoid jargon, explain significance.\n\n"
            f"Title: {title}\n\nText:\n{base_text}"
        )

    resp = await _async_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a precise science communicator."},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=int(max_w * 2),
    )

    out = (resp.choices[0].message.content or "").strip()

    # cost tracking
    u = resp.usage
    if u:
        prompt_toks = u.prompt_tokens or 0
        comp_toks   = u.completion_tokens or 0
        cost = prompt_toks * COST_INPUT_USD_PER_TOKEN + comp_toks * COST_OUTPUT_USD_PER_TOKEN
        cost_tracker["tokens_in"]  += prompt_toks
        cost_tracker["tokens_out"] += comp_toks
        cost_tracker["usd"]        += cost

    return out
    
# ---------------- Main builder (async) ----------------
async def _build_dataset_async(arxiv_results, mode: str, len_range, used_ids: set[str]):
    scitldr = load_scitldr()
    rows: list[dict] = []
    stats = {"sci": 0, "s2": 0, "gpt": 0}
    cost_tracker = {"tokens_in": 0, "tokens_out": 0, "usd": 0.0}

    # 1) populates the selected articles with their pid, title, abstracts, and input_text (which is the abstract itself if mode==abs). ensures uniqueness
    selected = []
    for res in arxiv_results:
        pid = res.get_short_id()
        pid = pid.split("v")[0]   #remove the suffix
        if pid in used_ids:
            continue
        title = (res.title or "").strip()
        abstr = (res.summary or "").strip()
        
        #for abstracts, save the input_text to the model as the abstract
        if mode == "abstract":
            if not abstr:
                continue
            input_text = abstr
        #for full-papers, save the extracted(pdf) as thje input_text
        else:
            try:
                pdf_path = res.download_pdf(dirpath=str(PDF_DIR))
            except Exception as e:
                print(f"[PDF WARN] {pid}: {e}; skipping")
                continue
            input_text = grobid_fulltext(Path(pdf_path))
            if len(input_text.split()) < 800:
                print(f"[GROBID WARN] {pid}: body too short; skipping")
                continue
        selected.append((pid, title, abstr, input_text))
        used_ids.add(pid)

    #after the loop
    if not selected:
        print(f"[WARN] No candidates for {mode}.")
        return rows, stats, cost_tracker, used_ids

    # 2) Fetch S2 TLDRs in batch
    pids = [pid for pid, _, _, _ in selected]
    pids = [pid.split("v")[0] for pid in pids]
    # print(pids)
    s2_map = await s2_fetch_tldr_for_ids(pids)
    # print(s2_map)
    # print(len(s2_map))
    # for key in s2_map:
        # print(f"article: {key} tldr: {s2_map[key]}")
        # print(len(s2_map[key].split()))

    # 3) Assign sources or schedule GPT
    to_gpt, ready = [], []
    for pid, title, abstr, input_text in selected:
        print(f"curr pid: {pid}")
        summary, src = None, None
        if pid in scitldr:
            summary, src = scitldr[pid], "sci"; stats["sci"] += 1
        if not summary:
            tldr = s2_map.get(pid)
            if tldr:
                summary, src = tldr, "s2"; stats["s2"] += 1
                print(f"summary from s2: {summary}")
        # no valid summary from s2
        if not summary or summary == "": 
            to_gpt.append((pid, title, abstr, input_text))
        # summary exists but length is less than 100
        elif summary and len(summary.split()) < len_range[0]:
            to_gpt.append((pid, title, abstr, input_text))
        else:
            ready.append((pid, title, input_text, summary, src))

    # 4) Run GPT in parallel
    SEM = asyncio.Semaphore(16)
    async def one_gpt(pid, title, abstr, input_text):
        async with SEM:
            s = await gpt_expand_async(abstr or input_text, title, *len_range, cost_tracker)
            return pid, title, input_text, s, "gpt"

    gpt_tasks = [one_gpt(*args) for args in to_gpt]
    for fut in asyncio.as_completed(gpt_tasks):
        try:
            pid, title, input_text, s, src = await fut
            stats["gpt"] += 1
            ready.append((pid, title, input_text, s, src))
        except Exception as e:
            print(f"[GPT WARN] {e}")

    # 5) Assemble rows
    for pid, title, input_text, summary, src in ready:
        rows.append({
            "paper_id": pid,
            "title": title,
            "input_text": input_text,
            "summary": summary,
            "source": src,
        })

    return rows, stats, cost_tracker, used_ids

def build_dataset(arxiv_results, mode: str = "abstract", len_range=(100, 200), used_ids: set[str] | None = None):
    used_ids = used_ids if used_ids is not None else set()
    rows, stats, cost_tracker, used_ids = asyncio.run(_build_dataset_async(arxiv_results, mode, len_range, used_ids))
    total = len(rows)
    print(f"\n--- {mode.upper()} Coverage ---")
    if total == 0:
        print("[WARN] No examples collected.")
    else:
        for k, v in stats.items():
            pct = (v / total * 100.0) if total else 0.0
            print(f"{k}: {v} ({pct:.1f}%)")
    print(f"\n--- GPT cost ({mode}) ---")
    print(f"Input tokens: {cost_tracker['tokens_in']}")
    print(f"Output tokens: {cost_tracker['tokens_out']}")
    print(f"Estimated USD: ${cost_tracker['usd']:.2f}")
    return rows

def write_parquet(rows, path: Path):
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd")
    print(f"[✓] Saved {len(rows)} → {path}")

# expose constants
__all__ = ["build_dataset", "write_parquet", "DATA_DIR"]

