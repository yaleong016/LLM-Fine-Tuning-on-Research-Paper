
"""
test_collect.py

Fetch top-N arXiv papers on/before a cutoff date (newest -> older),
then call collect_pairs.build_dataset() to produce:
  - abstracts -> 100–200 word summaries
  - full-texts -> 200–300 word summaries

Writes:
  data/abstract_pairs.parquet
  data/fulltext_pairs.parquet
"""

from __future__ import annotations
import argparse
import datetime as dt
import time
from collections import Counter
import arxiv

from collect_pairs import build_dataset, write_parquet, s2_fetch_tldr_for_ids, DATA_DIR

# ----------------- arXiv date-window helpers -----------------
def _fmt_ymdhm(ts: dt.datetime) -> str:
    ts = ts.astimezone(dt.timezone.utc)
    return ts.strftime("%Y%m%d%H%M")

def _search_with_window(query_base: str, start_dt: dt.datetime, end_dt: dt.datetime) -> arxiv.Search:
    q = f"({query_base}) AND submittedDate:[{_fmt_ymdhm(start_dt)} TO {_fmt_ymdhm(end_dt)}]"
    return arxiv.Search(
        query=q,
        max_results=200,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

def fetch_before_date(query_base: str, cutoff_date_utc: dt.datetime, want: int,
                      lookback_days=365, window_days=7, page_size=100, pause_s=1.0) -> list[arxiv.Result]:
    client = arxiv.Client(num_retries=2, delay_seconds=2, page_size=page_size)
    results: list[arxiv.Result] = []
    end = cutoff_date_utc
    start_limit = cutoff_date_utc - dt.timedelta(days=lookback_days)
    while end > start_limit and len(results) < want:
        start = max(start_limit, end - dt.timedelta(days=window_days))
        search = _search_with_window(query_base, start, end)
        try:
            for r in client.results(search):
                if r.published and r.published <= cutoff_date_utc:
                    results.append(r)
                    if len(results) >= want:
                        break
        except arxiv.UnexpectedEmptyPageError:
            print(f"[arXiv WARN] Empty page for {start.date()}–{end.date()}; continuing…")
        except Exception as e:
            print(f"[arXiv WARN] {e}; continuing…")
        end = start
        time.sleep(pause_s)
    return results[:want]

# ----------------- CLI + main -----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=str, default="2025-08-09",
                    help="Cutoff date in YYYY-MM-DD. Default=today UTC.")
    ap.add_argument("--abs_n", type=int, default=10000, help="Number of abstracts to collect.")
    ap.add_argument("--ft_n", type=int, default=0, help="Number of full-texts to collect.")
    ap.add_argument("--lookback_days", type=int, default=1000)
    ap.add_argument("--window_days", type=int, default=30)
    ap.add_argument("--query", type=str, default="cat:cs.AI OR cat:cs.LG")
    return ap.parse_args()

def cutoff_to_eod_utc(cutoff_str: str | None) -> dt.datetime:
    if cutoff_str:
        y, m, d = map(int, cutoff_str.split("-"))
        return dt.datetime(y, m, d, 23, 59, tzinfo=dt.timezone.utc)
    now = dt.datetime.now(dt.timezone.utc)
    return dt.datetime(now.year, now.month, now.day, 23, 59, tzinfo=dt.timezone.utc)

def summarize_sources(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {"sci": 0.0, "s2": 0.0, "gpt": 0.0}
    c = Counter(r.get("source", "gpt") for r in rows)
    total = len(rows)
    return {k: (100.0 * c.get(k, 0) / total) for k in ("sci", "s2", "gpt")}


import os
import requests

S2_KEY = os.getenv("S2_API_KEY")

def s2_get_by_arxiv(arxiv_id: str) -> dict:
    """
    Fetch a paper from Semantic Scholar using an arXiv ID (PID).
    Returns the JSON response with metadata.
    """
    if not S2_KEY:
        raise RuntimeError("Missing S2_API_KEY in environment.")

    url = f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv_id}"
    params = {
        "fields": "title,year,authors,tldr,externalIds,url"
    }
    headers = {"x-api-key": S2_KEY}

    r = requests.get(url, headers=headers, params=params, timeout=30)
    if r.status_code != 200:
        return None
    return r.json()

def main():
    args = parse_args()
    cutoff = cutoff_to_eod_utc(args.cutoff)
    print(f"[INFO] Cutoff (UTC): {cutoff.isoformat()} | Query: {args.query}")

    # test = ['2507.10864', '2507.15867', '2507.10861', '2507.10854', '2507.10850', '2507.10846', '2507.10843', '2507.10835', '2507.10834', '2507.10831']
    # for id in test:
    #     article = s2_get_by_arxiv(id)   #we have to take the version away
    #     print(article)

    # Abstracts
    t0 = time.time()

    total_wanted = args.abs_n + args.ft_n
    results = fetch_before_date(
        args.query, cutoff, want=total_wanted,
        lookback_days=args.lookback_days, window_days=args.window_days
    )

    used_ids = set()
    abstracts = results[:args.abs_n]
    ft_candidates = results[args.abs_n:]
    fulltexts = [r for r in ft_candidates if r.get_short_id() not in used_ids][:args.ft_n]

    rows_abs = build_dataset(abstracts, mode="abstract", len_range=(100, 200), used_ids=used_ids)
    write_parquet(rows_abs, DATA_DIR / "abstract_pairs.parquet")

    # rows_ft = build_dataset(fulltexts, mode="fulltext", len_range=(200, 300), used_ids=used_ids)
    # write_parquet(rows_ft, DATA_DIR / "fulltext_pairs.parquet")

    t1 = time.time()
    print(f"[INFO] Retrieved {len(abstracts)} abstracts and {len(fulltexts)} fulltexts in {t1 - t0:.2f} seconds")

    # # Coverage summaries
    # abs_pct = summarize_sources(rows_abs)
    # ft_pct  = summarize_sources(rows_ft)
    # total_rows = rows_abs + rows_ft
    # overall_pct = summarize_sources(total_rows)

    # def fmt(p): return " | ".join(f"{k}: {p[k]:5.1f}%" for k in ("sci", "s2", "gpt"))
    # print("\n=== Source coverage (percent) ===")
    # print(f"Abstracts ({len(rows_abs)}): {fmt(abs_pct)}")
    # print(f"Full-texts ({len(rows_ft)}): {fmt(ft_pct)}")
    # print(f"Overall   ({len(total_rows)}): {fmt(overall_pct)}")


if __name__ == "__main__":
    main()