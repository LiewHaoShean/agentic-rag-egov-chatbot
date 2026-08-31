"""Offline RAG evaluation harness (Chapter 5 — System Evaluation Metrics).

Runs the fixed question set from scripts/eval_questions.py through the live
retrieval path (and optionally the full agent), records what was retrieved and
what was answered, and writes a CSV for manual scoring against the 2/1/0
relevance rubric.

Two metrics are produced automatically, with no human involvement and no extra
model calls:

  * Category routing accuracy — did a KWSP question retrieve KWSP chunks?
    Ground truth is `expected_category` in the question set.
  * Retrieval yield — how many chunks came back, and how concentrated they are
    in the expected agency (purity).

Two metrics remain manual, because they need a human reading the text. The CSV
ships with blank columns for them:

  * retrieval_score — 2 complete match, 1 partial match, 0 miss
  * grounded        — y/n, does the answer stay inside the retrieved context

Usage:
    python -m scripts.eval_rag                    # retrieval only (cheap)
    python -m scripts.eval_rag --with-answers     # + full agent, 4 LLM calls/q
    python -m scripts.eval_rag --limit 5          # smoke test
    python -m scripts.eval_rag --sleep 3          # pace against Vertex quota
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from datetime import datetime

from agent.tools import retrieve
from core.logging import get_logger
from scripts.eval_questions import QUESTIONS

log = get_logger(__name__)

# Transient transport failures are retried so they are not scored as misses.
RETRIEVAL_ATTEMPTS = 3
# Generation is far more failure-prone than retrieval (four provider calls per
# question, against a quota that rate-limits), so it gets more attempts and a
# longer backoff.
GENERATION_ATTEMPTS = 4
GENERATION_BACKOFF = 8  # seconds, multiplied by the attempt number

CSV_COLUMNS = [
    "id", "lang", "question", "expected_category",
    "top_category", "category_match", "purity",
    "num_chunks", "categories_returned", "latency_s",
    "answer", "answer_latency_s", "attempts", "chunk_ids",
    # --- filled in by hand afterwards ---
    # retrieval_score     2 complete match / 1 partial match / 0 miss
    # hallucination_score 2 severe / 1 minor / 0 zero hallucination (target)
    #   NOTE the hallucination scale runs the opposite way to the relevance
    #   scale. 0 is the desired outcome. This matches the rubric defined in
    #   the project's system evaluation metrics.
    "retrieval_score", "hallucination_score", "notes",
]


def evaluate_one(item: dict, with_answers: bool) -> dict:
    """Run one question through retrieval (and optionally the whole agent)."""
    row = {c: "" for c in CSV_COLUMNS}
    row.update(
        id=item["id"], lang=item["lang"], question=item["q"],
        expected_category=item["category"],
    )

    started = time.perf_counter()
    # The embedding endpoint intermittently drops HTTP/2 connections. That is a
    # transport blip, not a retrieval result, so retry rather than recording it
    # as a miss and skewing the accuracy figure.
    chunks = None
    for attempt in range(1, RETRIEVAL_ATTEMPTS + 1):
        try:
            chunks = retrieve(item["q"])
            break
        except Exception as exc:  # noqa: BLE001 — one failure must not end the run
            log.warning("%s retrieval attempt %d failed: %s", item["id"], attempt, exc)
            if attempt == RETRIEVAL_ATTEMPTS:
                row["notes"] = f"RETRIEVAL ERROR after {attempt} attempts: {exc}"
                return row
            time.sleep(2 * attempt)

    # Stamped HERE, not at the end of the function. Retrieval latency is a
    # property of embedding plus search; folding generation and its retry
    # backoff into the same figure would inflate it by an order of magnitude.
    row["latency_s"] = f"{time.perf_counter() - started:.2f}"

    cats = [c.get("category") or "?" for c in chunks]
    counts = Counter(cats)
    top_category = counts.most_common(1)[0][0] if counts else ""
    expected = item["category"]

    row.update(
        top_category=top_category,
        category_match="Y" if top_category == expected else "N",
        # Purity = share of retrieved chunks belonging to the expected agency.
        # A question can route correctly yet still pull in cross-agency noise.
        purity=f"{counts.get(expected, 0) / len(cats):.2f}" if cats else "0.00",
        num_chunks=len(chunks),
        categories_returned=" ".join(f"{k}={v}" for k, v in counts.most_common()),
        chunk_ids=" ".join(str(c.get("embedding_id", ""))[:8] for c in chunks),
    )
    # Kept out of the CSV (too long for a cell) and written to the companion
    # review file instead, since the 2/1/0 rubric cannot be scored without
    # reading the retrieved text.
    row["_chunks"] = chunks

    if with_answers:
        from agent.graph import run_agent
        from agent.nodes import BUSY_MESSAGE, BUSY_MESSAGES
        from services.gemini import get_llm

        # The graph catches provider errors itself and degrades to the busy
        # terminal, so a rate-limited question returns normally and never
        # raises. Matching the busy text is the only way to see the failure
        # from out here, and it has to cover every language the graph replies
        # in — an English-only check silently passed the Malay ones through.
        busy_texts = {BUSY_MESSAGE, *BUSY_MESSAGES.values()}

        gen_started = time.perf_counter()
        for attempt in range(1, GENERATION_ATTEMPTS + 1):
            try:
                result = run_agent(
                    user_text=item["q"], history=[],
                    user_id="eval", conversation_id="eval",
                )
                row["answer"] = (result.get("reply") or "").replace("\n", " ")
                if row["answer"].strip() in busy_texts:
                    log.warning("%s attempt %d returned the busy message",
                                item["id"], attempt)
                    get_llm.cache_clear()
                    if attempt == GENERATION_ATTEMPTS:
                        row["notes"] = (
                            f"BUSY MESSAGE after {attempt} attempts "
                            "(provider error, not an answer)"
                        )
                    else:
                        time.sleep(GENERATION_BACKOFF * attempt)
                        continue
                break
            except Exception as exc:  # noqa: BLE001
                log.warning("%s generation attempt %d failed: %s",
                            item["id"], attempt, exc)
                # get_llm is lru_cached, so a dropped HTTP/2 connection is
                # reused by every later call. Clearing the cache forces a new
                # client and a fresh connection on the next attempt.
                get_llm.cache_clear()
                if attempt == GENERATION_ATTEMPTS:
                    row["notes"] = f"GENERATION ERROR after {attempt} attempts: {exc}"
                else:
                    time.sleep(GENERATION_BACKOFF * attempt)
        row["answer_latency_s"] = f"{time.perf_counter() - gen_started:.2f}"
        row["attempts"] = str(attempt)

    return row


def write_review_file(rows: list[dict], path: str) -> None:
    """Human-readable dump of what each question retrieved.

    The 2/1/0 relevance rubric is scored against the retrieved TEXT, which is
    far too long for a CSV cell, so it is written here instead. Read this file
    alongside the CSV and record the score in the CSV.
    """
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Retrieval review sheet\n\n")
        fh.write("Two scores per question, recorded in the CSV.\n\n")
        fh.write("`retrieval_score` against the TOP chunk. "
                 "2 complete match, 1 partial match, 0 miss.\n\n")
        fh.write("`hallucination_score` against the answer. "
                 "2 severe (fabricated policy, invented links), "
                 "1 minor (true but absent from the chunks), "
                 "0 zero hallucination (fully supported). "
                 "NOTE this scale runs the opposite way. 0 is the target.\n\n")
        for r in rows:
            fh.write(f"\n---\n\n## {r['id']}  [{r['lang']}]  "
                     f"expected = {r['expected_category']}\n\n")
            fh.write(f"**{r['question']}**\n\n")
            fh.write(f"routing {r['category_match']} | "
                     f"purity {r['purity']} | {r['categories_returned']}\n\n")
            if r.get("answer"):
                fh.write(f"**Answer produced**\n\n> {r['answer']}\n\n")
            for i, c in enumerate(r.get("_chunks") or [], 1):
                text = (c.get("translate_text") or c.get("original_text") or "")
                label = "TOP CHUNK" if i == 1 else f"chunk {i}"
                fh.write(f"**{label}** [{c.get('category')}] "
                         f"{c.get('file_url', '')}\n\n")
                # NOT truncated. A 700-character cap here silently cut roughly
                # the last third off every chunk, and claims that sat past the
                # cut read as ungrounded during scoring. That produced five
                # false hallucination findings, including three PERKESO form
                # links that are genuine and present in the retrieved passages.
                # Chunks are bounded at 1000 characters by the splitter, so the
                # full body is short enough to print.
                fh.write(f"```\n{text.strip()}\n```\n\n")
            fh.write("Score ___   Notes ______________________________\n")


def summarise(rows: list[dict]) -> None:
    """Print the automated metrics that go straight into the report."""
    scored = [r for r in rows if r["category_match"] in ("Y", "N")]
    if not scored:
        print("\nNo successful retrievals — nothing to summarise.")
        return

    matched = sum(1 for r in scored if r["category_match"] == "Y")
    print("\n" + "=" * 62)
    print(f"{'CATEGORY ROUTING ACCURACY':<40}{matched}/{len(scored)}"
          f"  ({matched / len(scored):.1%})")
    purities = [float(r["purity"]) for r in scored if r["purity"]]
    print(f"{'MEAN RETRIEVAL PURITY':<40}{sum(purities) / len(purities):.2f}")
    lat = [float(r["latency_s"]) for r in scored if r["latency_s"]]
    print(f"{'MEAN LATENCY (seconds)':<40}{sum(lat) / len(lat):.2f}")

    for label, key in (("BY AGENCY", "expected_category"), ("BY LANGUAGE", "lang")):
        print("-" * 62)
        print(label)
        groups: dict[str, list[dict]] = {}
        for r in scored:
            groups.setdefault(r[key], []).append(r)
        for name, grp in sorted(groups.items()):
            hit = sum(1 for r in grp if r["category_match"] == "Y")
            print(f"  {name:<20}{hit}/{len(grp):<8}({hit / len(grp):.0%})")

    misses = [r for r in scored if r["category_match"] == "N"]
    if misses:
        print("-" * 62)
        print("MISROUTED QUESTIONS (inspect these for the report)")
        for r in misses:
            print(f"  {r['id']} [{r['expected_category']} -> {r['top_category']}] "
                  f"{r['question'][:52]}")
    print("=" * 62)


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline RAG evaluation harness.")
    parser.add_argument("--with-answers", action="store_true",
                        help="Also run the full agent (4 LLM calls per question).")
    parser.add_argument("--limit", type=int, help="Run only the first N questions.")
    parser.add_argument("--ids", help="Comma-separated question ids to re-run, "
                                      "e.g. Q12,Q13,Q16 (for provider failures).")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Seconds to wait between questions (quota pacing).")
    parser.add_argument("--out", help="CSV output path (default: timestamped).")
    args = parser.parse_args()

    if args.ids:
        wanted = {i.strip().upper() for i in args.ids.split(",")}
        items = [q for q in QUESTIONS if q["id"].upper() in wanted]
    else:
        items = QUESTIONS[:args.limit] if args.limit else QUESTIONS
    out_path = args.out or f"eval_results_{datetime.now():%Y%m%d_%H%M}.csv"

    mode = "retrieval + generation" if args.with_answers else "retrieval only"
    print(f"Evaluating {len(items)} questions ({mode})\n")

    rows: list[dict] = []
    for i, item in enumerate(items, 1):
        row = evaluate_one(item, args.with_answers)
        rows.append(row)
        flag = {"Y": "ok  ", "N": "MISS"}.get(row["category_match"], "ERR ")
        print(f"[{i:>2}/{len(items)}] {row['id']} {flag} "
              f"{row['expected_category']:<9} -> {row['top_category']:<9} "
              f"purity={row['purity']}  {row['question'][:44]}")
        if args.sleep and i < len(items):
            time.sleep(args.sleep)

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    review_path = out_path.rsplit(".", 1)[0] + "_review.md"
    write_review_file(rows, review_path)

    summarise(rows)
    print(f"\nCSV written to {out_path}")
    print("Score the retrieval_score (2/1/0) and grounded (y/n) columns by hand.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
