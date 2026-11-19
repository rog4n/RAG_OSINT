"""Extract article bodies and enrich them with spaCy; store results in SQLite.

Run from project root:
    python -m src.enrich_articles
    python -m src.enrich_articles --limit 10
    python -m src.enrich_articles --retry
"""
import argparse
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import certifi
import requests
import spacy
import trafilatura

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "new_articles.db"
USER_AGENT = "Mozilla/5.0 OSINT-Event-Fusion/1.0"
ENTITY_LABELS = {
    "PERSON", "ORG", "GPE", "LOC", "FAC", "EVENT",
    "NORP", "PRODUCT", "WORK_OF_ART",
}


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def open_db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("""
        CREATE TABLE IF NOT EXISTS article_nlp (
            article_id TEXT PRIMARY KEY,
            body_text TEXT,
            sentences_json TEXT NOT NULL DEFAULT '[]',
            entities_json TEXT NOT NULL DEFAULT '[]',
            keyphrases_json TEXT NOT NULL DEFAULT '[]',
            extraction_status TEXT NOT NULL,
            extraction_error TEXT,
            extracted_utc TEXT NOT NULL,
            FOREIGN KEY(article_id) REFERENCES articles(article_id)
        )
    """)
    con.commit()
    return con


def extract_body(session, url):
    try:
        response = session.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=(10, 25),
            verify=certifi.where(),
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if "html" not in content_type and "xml" not in content_type:
            return None, "unsupported_content_type", content_type[:200]

        kwargs = dict(
            url=response.url,
            include_comments=False,
            include_tables=False,
            include_links=False,
            output_format="txt",
        )
        text = trafilatura.extract(response.content, favor_precision=True, **kwargs)
        if not text or len(text.strip()) < 200:
            text = trafilatura.extract(response.content, favor_recall=True, **kwargs)
        if not text or len(text.strip()) < 200:
            return None, "no_readable_body", "Extractor found insufficient article text"
        return text.strip(), "extracted", None
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        return None, "http_error", f"HTTP {status}: {str(exc)[:200]}"
    except requests.RequestException as exc:
        return None, "request_error", f"{type(exc).__name__}: {str(exc)[:200]}"
    except Exception as exc:
        return None, "extraction_error", f"{type(exc).__name__}: {str(exc)[:200]}"


def _json_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except (ValueError, TypeError):
            return []
    return []


def preserve_previous_extraction_on_retry_failure(
    row, body, status, error, sentences, entities, phrases
):
    """Keep a prior successful body/NLP extraction if a retry fails."""
    previous_body = row["existing_body_text"] or ""
    previous_status = row["existing_extraction_status"] or ""
    if status == "extracted":
        return (
            body, sentences, entities, phrases, status, error,
            now_utc(), False,
        )
    if previous_status == "extracted" and len(previous_body.strip()) >= 200:
        previous_timestamp = row["existing_extracted_utc"] or now_utc()
        retry_error = (
            f"Retry failed; preserved previous successful extraction "
            f"from {previous_timestamp}: {error or status}"
        )
        return (
            previous_body,
            _json_list(row["existing_sentences_json"]),
            _json_list(row["existing_entities_json"]),
            _json_list(row["existing_keyphrases_json"]),
            "extracted",
            retry_error,
            previous_timestamp,
            True,
        )
    return body, sentences, entities, phrases, status, error, now_utc(), False


def enrich(nlp, title, body):
    """Extract body sentences separately from the headline.

    Headlines frequently omit terminal punctuation. Processing title + body as
    one spaCy document can merge the headline with the article's first sentence,
    creating a malformed claim candidate. Entities/keyphrases still use both.
    """
    title_doc = nlp((title or "")[:nlp.max_length]) if title else None
    body_doc = nlp((body or "")[:nlp.max_length])
    sentences = [s.text.strip() for s in body_doc.sents if s.text.strip()]

    entities, seen = [], set()
    for doc in (title_doc, body_doc):
        if doc is None:
            continue
        for ent in doc.ents:
            value = " ".join(ent.text.split())
            key = (value.casefold(), ent.label_)
            if ent.label_ in ENTITY_LABELS and value and key not in seen:
                seen.add(key)
                entities.append({"text": value, "label": ent.label_})

    # Noun chunks are candidate keyphrases, not verified facts.
    phrases, seen_phrases = [], set()
    for doc in (title_doc, body_doc):
        if doc is None:
            continue
        for chunk in doc.noun_chunks:
            phrase = " ".join(chunk.text.split()).strip(" \t\n.,:;")
            key = phrase.casefold()
            if 3 <= len(phrase) <= 100 and key not in seen_phrases:
                seen_phrases.add(key)
                phrases.append(phrase)
            if len(phrases) >= 40:
                break
        if len(phrases) >= 40:
            break
    return sentences, entities, phrases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retry", action="store_true",
                        help="Reprocess previously successful extractions too.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Maximum articles to process; 0 means all eligible.")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay between requests in seconds.")
    args = parser.parse_args()

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError as exc:
        raise SystemExit(
            "spaCy model missing. Run: python -m spacy download en_core_web_sm"
        ) from exc

    con = open_db()
    query = """
        SELECT a.article_id, a.title, a.url, a.source_name,
               n.body_text AS existing_body_text,
               n.sentences_json AS existing_sentences_json,
               n.entities_json AS existing_entities_json,
               n.keyphrases_json AS existing_keyphrases_json,
               n.extraction_status AS existing_extraction_status,
               n.extracted_utc AS existing_extracted_utc
        FROM articles a
        LEFT JOIN article_nlp n ON n.article_id = a.article_id
        WHERE a.url IS NOT NULL AND TRIM(a.url) != ''
    """
    if not args.retry:
        query += " AND (n.article_id IS NULL OR n.extraction_status != 'extracted')"
    query += " ORDER BY a.collected_utc, a.article_id"
    if args.limit > 0:
        rows = con.execute(query + " LIMIT ?", (args.limit,)).fetchall()
    else:
        rows = con.execute(query).fetchall()

    if not rows:
        print("No eligible articles. Use --retry to reprocess successful ones.")
        con.close()
        return

    session = requests.Session()
    counts = {}
    print(f"Processing {len(rows)} articles...")
    try:
        for i, row in enumerate(rows, 1):
            print(f"[{i}/{len(rows)}] {row['source_name']}: {(row['title'] or '')[:85]}")
            body, status, error = extract_body(session, row["url"])
            sentences, entities, phrases = [], [], []
            if status == "extracted":
                try:
                    sentences, entities, phrases = enrich(nlp, row["title"], body)
                except Exception as exc:
                    status = "nlp_error"
                    error = f"{type(exc).__name__}: {str(exc)[:200]}"

            (
                body, sentences, entities, phrases, status, error,
                extracted_utc, preserved_previous,
            ) = preserve_previous_extraction_on_retry_failure(
                row, body, status, error, sentences, entities, phrases
            )

            con.execute("""
                INSERT INTO article_nlp (
                    article_id, body_text, sentences_json, entities_json,
                    keyphrases_json, extraction_status, extraction_error, extracted_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(article_id) DO UPDATE SET
                    body_text=excluded.body_text,
                    sentences_json=excluded.sentences_json,
                    entities_json=excluded.entities_json,
                    keyphrases_json=excluded.keyphrases_json,
                    extraction_status=excluded.extraction_status,
                    extraction_error=excluded.extraction_error,
                    extracted_utc=excluded.extracted_utc
            """, (
                row["article_id"], body,
                json.dumps(sentences, ensure_ascii=False),
                json.dumps(entities, ensure_ascii=False),
                json.dumps(phrases, ensure_ascii=False),
                status, error, extracted_utc,
            ))
            con.commit()
            counts[status] = counts.get(status, 0) + 1
            if error:
                prefix = "    Preserved previous extraction; " if preserved_previous else "    "
                print(f"{prefix}{status}: {error}")
            else:
                print(f"    {len(body)} chars; {len(entities)} entities; "
                      f"{len(sentences)} sentences")
            if i < len(rows) and args.delay > 0:
                time.sleep(args.delay)
    finally:
        session.close()
        con.close()

    print("\n=== NLP ENRICHMENT SUMMARY ===")
    print(json.dumps(counts, indent=2))
    print(f"Database: {DB_PATH}")
    print("NER/keyphrases are model outputs, not verified facts.")


if __name__ == "__main__":
    main()
