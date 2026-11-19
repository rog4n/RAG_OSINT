
import argparse
import hashlib
import json
import math
import re
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import feedparser
import joblib
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import requests
import certifi


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "headline_svm.joblib"
DB_PATH = ROOT / "data" / "new_articles.db"
REPORT_DIR = ROOT / "reports"
PAIRWISE_AUDIT_PATH = REPORT_DIR / "cluster_pairwise_audit_NALP.csv"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

FEEDS = {
    "BBC World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "BBC UK": "https://feeds.bbci.co.uk/news/uk/rss.xml",
    "NPR World": "https://feeds.npr.org/1004/rss.xml",
    "NPR Politics": "https://feeds.npr.org/1014/rss.xml",
    "Guardian World": "https://www.theguardian.com/world/rss",
    "Guardian Environment": "https://www.theguardian.com/environment/rss",
    "UN News": "https://news.un.org/feed/subscribe/en/news/all/rss.xml",
}

# Conservative starting threshold. Evaluate and adjust using manually
# reviewed development examples, not the held-out MN-DS test set.
SIMILARITY_THRESHOLD = 0.205
ENTITY_SIMILARITY_THRESHOLD = 0.20
# Experimental safeguards; validate on the frozen, manually labeled pair set.
IDF_ENTITY_MIN_SCORE = 3.0
CLUSTER_MIN_PAIRWISE_SIMILARITY = 0.10
ENABLE_IDF_ENTITY_WEIGHTING = True
ENABLE_CLUSTER_COHERENCE = True

# Semantic matching is optional at runtime. Install sentence-transformers to
# enable embeddings; the script logs a warning and falls back to lexical rules
# if the package/model cannot be loaded.
ENABLE_SEMANTIC_MATCHING = True
SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_MERGE_THRESHOLD = 0.76
# Recall-recovery rule: lower semantic threshold only with distinctive,
# label-compatible named-entity overlap and no temporal/development conflict.
ENTITY_CONDITIONED_SEMANTIC_THRESHOLD = 0.70
ENTITY_CONDITIONED_SEMANTIC_UPPER = SEMANTIC_MERGE_THRESHOLD
SEMANTIC_STRONG_MATCH_THRESHOLD = 0.88
if not (
    0.0 <= ENTITY_CONDITIONED_SEMANTIC_THRESHOLD
    < ENTITY_CONDITIONED_SEMANTIC_UPPER
    <= SEMANTIC_STRONG_MATCH_THRESHOLD
    <= 1.0
):
    raise ValueError(
        "Invalid semantic thresholds: expected 0 <= entity-conditioned lower "
        "< semantic merge upper <= strong-match threshold <= 1."
    )

# Temporal guardrails: broad-topic/entity matches across distant dates should
# not merge unless the text match is exceptionally strong.
MAX_EVENT_DATE_GAP_DAYS = 14
TEMPORAL_OVERRIDE_COSINE = 0.50
MAX_PER_FEED = 100

# Event-development classification is deliberately conservative: uncertain
# articles remain "unknown" and are not blocked solely by this heuristic.
ENABLE_DEVELOPMENT_GATING = True
DEVELOPMENT_CONFLICT_POLICY = "separate_known_types"


def clean_text(value):
    value = str(value or "")
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_title(value):
    value = clean_text(value).lower()
    value = re.sub(r"[^\w\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def domain_of(url):
    try:
        domain = urlparse(url).netloc.lower()
        return domain.removeprefix("www.")
    except Exception:
        return ""


def entry_published(entry):
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        try:
            import calendar
            return datetime.fromtimestamp(
                calendar.timegm(parsed), tz=timezone.utc
            ).isoformat()
        except Exception:
            pass

    return entry.get("published", entry.get("updated", ""))


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                article_id TEXT PRIMARY KEY,
                source_name TEXT NOT NULL,
                domain TEXT NOT NULL,
                url TEXT NOT NULL,
                title TEXT NOT NULL,
                normalized_title TEXT NOT NULL,
                summary TEXT,
                published_utc TEXT,
                collected_utc TEXT NOT NULL,
                category TEXT,
                entities_json TEXT,
                cluster_id TEXT,
                category_model TEXT,
                category_decision_scores_json TEXT
            )
        """)
        # Safe, additive migration for existing databases.
        existing_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(articles)")
        }
        if "category_model" not in existing_columns:
            conn.execute("ALTER TABLE articles ADD COLUMN category_model TEXT")
        if "category_decision_scores_json" not in existing_columns:
            conn.execute(
                "ALTER TABLE articles ADD COLUMN category_decision_scores_json TEXT"
            )
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_articles_domain
            ON articles(domain)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_articles_cluster
            ON articles(cluster_id)
        """)

def parse_feed(feed_url):
    """Fetch and parse an RSS feed with certificate verification."""
    try:
        response = requests.get(
            feed_url,
            headers={"User-Agent": "Mozilla/5.0 OSINT-Event-Fusion/1.0"},
            timeout=20,
            verify=certifi.where(),
        )
        response.raise_for_status()

        feed = feedparser.parse(response.content)

        if getattr(feed, "bozo", False):
            print(f"  Feed parsing warning for {feed_url}: "
                  f"{feed.bozo_exception}")

        if not feed.entries:
            print(f"  No entries returned from {feed_url}")

        return feed

    except requests.RequestException as exc:
        print(f"  RSS request failed for {feed_url}: {exc}")
        return None

def fetch_articles():
    collected = []

    for source_name, feed_url in FEEDS.items():
        print(f"Fetching {source_name}...", flush=True)

        try:
            feed = parse_feed(feed_url)
            if feed is None:
                continue

            for entry in feed.entries[:MAX_PER_FEED]:
                title = clean_text(entry.get("title", ""))
                url = entry.get("link", "").strip()

                if not title or not url:
                    continue

                domain = domain_of(url) or domain_of(feed_url)
                if not domain:
                    continue

                # Stable identity for the same article URL.
                article_id = hashlib.sha256(
                    url.split("#")[0].encode("utf-8")
                ).hexdigest()[:24]

                collected.append({
                    "article_id": article_id,
                    "source_name": source_name,
                    "domain": domain,
                    "url": url,
                    "title": title,
                    "normalized_title": normalize_title(title),
                    "summary": clean_text(entry.get("summary", "")),
                    "published_utc": entry_published(entry),
                    "collected_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),
                })

            print(f"  Retrieved {len(feed.entries[:MAX_PER_FEED])} entries.")

        except Exception as exc:
            print(f"  Feed error: {exc}")

        time.sleep(0.2)

    return collected


def store_articles(articles):
    with sqlite3.connect(DB_PATH) as conn:
        before = conn.total_changes

        conn.executemany("""
            INSERT INTO articles (
                article_id, source_name, domain, url, title,
                normalized_title, summary, published_utc, collected_utc
            )
            VALUES (
                :article_id, :source_name, :domain, :url, :title,
                :normalized_title, :summary, :published_utc, :collected_utc
            )
            ON CONFLICT(article_id) DO UPDATE SET
                title = excluded.title,
                normalized_title = excluded.normalized_title,
                summary = excluded.summary,
                published_utc = excluded.published_utc
        """, articles)

        return len(articles)


def extract_entities(nlp, title, summary):
    doc = nlp(f"{title}. {summary}")
    entities = []

    for ent in doc.ents:
        if ent.label_ in {
            "PERSON", "ORG", "GPE", "LOC", "EVENT", "FAC"
        }:
            text = normalize_title(ent.text)

            # Skip very short entity strings to reduce accidental matches.
            if len(text) >= 3:
                entities.append({
                    "text": text,
                    "label": ent.label_,
                })

    # De-duplicate exact entity text within the article.
    seen = set()
    unique = []
    for entity in entities:
        key = (entity["text"], entity["label"])
        if key not in seen:
            unique.append(entity)
            seen.add(key)

    return unique


def load_articles():
    """Load article metadata, SVM labels, and optional full-text NLP fields."""
    with sqlite3.connect(DB_PATH) as conn:
        tables = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "article_nlp" in tables:
            query = """
                SELECT
                    a.*,
                    n.body_text AS body_text,
                    n.sentences_json AS sentences_json,
                    n.entities_json AS fulltext_entities_json,
                    n.keyphrases_json AS keyphrases_json,
                    n.extraction_status AS extraction_status
                FROM articles a
                LEFT JOIN article_nlp n ON n.article_id = a.article_id
                ORDER BY a.published_utc, a.article_id
            """
        else:
            query = """
                SELECT a.*, NULL AS body_text, NULL AS sentences_json,
                       NULL AS fulltext_entities_json, NULL AS keyphrases_json,
                       NULL AS extraction_status
                FROM articles a
                ORDER BY a.published_utc, a.article_id
            """
        return pd.read_sql_query(query, conn)


def save_enrichment(df, model, nlp):
    """Run the existing headline-only SVM and persist its predictions.

    Decision scores are raw LinearSVC margins, not calibrated probabilities.
    Full-text NER entities are merged when article_nlp enrichment exists.
    """
    titles = df["title"].fillna("").astype(str)
    predictions = model.predict(titles)

    decision_scores = None
    if hasattr(model, "decision_function"):
        try:
            decision_scores = model.decision_function(titles)
        except Exception as exc:
            print(f"Warning: could not calculate SVM decision scores: {exc}")

    merged_entity_json = []
    for _, row in df.iterrows():
        short_entities = extract_entities(
            nlp, row.get("title", ""), row.get("summary", "")
        )
        full_entities = []
        raw_full = row.get("fulltext_entities_json")
        if isinstance(raw_full, str) and raw_full.strip():
            try:
                parsed = json.loads(raw_full)
                if isinstance(parsed, list):
                    full_entities = [
                        item for item in parsed
                        if isinstance(item, dict)
                        and item.get("text")
                        and item.get("label")
                    ]
            except (ValueError, TypeError):
                pass

        unique = []
        seen = set()
        for entity in short_entities + full_entities:
            text_value = normalize_title(entity.get("text", ""))
            label = str(entity.get("label", ""))
            key = (text_value, label)
            if text_value and label and key not in seen:
                seen.add(key)
                unique.append({"text": text_value, "label": label})
        merged_entity_json.append(json.dumps(unique, ensure_ascii=False))

    df["category"] = predictions
    df["category_model"] = MODEL_PATH.name
    df["entities_json"] = merged_entity_json

    if decision_scores is not None:
        classes = list(model.classes_)
        df["category_decision_scores_json"] = [
            json.dumps(
                {str(label): float(score) for label, score in zip(classes, scores)},
                ensure_ascii=False,
            )
            for scores in decision_scores
        ]
    else:
        df["category_decision_scores_json"] = None

    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            """
            UPDATE articles
            SET category = ?,
                entities_json = ?,
                category_model = ?,
                category_decision_scores_json = ?
            WHERE article_id = ?
            """,
            [
                (
                    row["category"],
                    row["entities_json"],
                    row["category_model"],
                    row["category_decision_scores_json"],
                    row["article_id"],
                )
                for _, row in df.iterrows()
            ],
        )

    return df


def entity_tokens(entity_json):
    try:
        return {
            item["text"]
            for item in json.loads(entity_json or "[]")
        }
    except (ValueError, TypeError, KeyError):
        return set()



def entity_records(entity_json):
    """Return normalized named entities with labels; malformed data is ignored."""
    try:
        raw = json.loads(entity_json or "[]")
    except (ValueError, TypeError):
        return {}
    records = {}
    if not isinstance(raw, list):
        return records
    for item in raw:
        if not isinstance(item, dict):
            continue
        text_value = normalize_title(item.get("text", ""))
        label = str(item.get("label", "")).strip().upper()
        if text_value and label:
            records.setdefault(text_value, set()).add(label)
    return records



def classify_event_development(title, summary=""):
    """Rule-based development label; heuristic, not a verified event ontology."""
    text = normalize_title(f"{title} {summary}")

    patterns = [
        ("investigation", [
            r"\binvestigation\b", r"\binvestigators?\b", r"\binquiry\b",
            r"\bprobe\b", r"\bprosecutors?\b", r"\bcourt hears\b",
            r"\btrial\b", r"\bcharges?\b", r"\bpolice examine\b",
            r"\breleased audio\b", r"\baudio reveals\b",
            r"\bblack box\b", r"\baccident report\b",
        ]),
        ("response", [
            r"\bgovernment announces\b", r"\bgovernment vows\b",
            r"\bminister announces\b", r"\bnew measures\b",
            r"\bpolicy response\b", r"\bsafety rules\b",
            r"\bsanctions\b", r"\bpledges? aid\b", r"\blaunches? plan\b",
        ]),
        ("aftermath", [
            r"\bdeath toll\b", r"\bnumber of dead\b", r"\bcasualties\b",
            r"\bsurvivors?\b", r"\bmemorial\b", r"\bfuneral\b",
            r"\bcleanup\b", r"\brecovery\b", r"\bdisplaced\b",
            r"\bmissing\b", r"\bvictims?\b",
        ]),
        ("incident", [
            r"\bcrash\b", r"\battack\b", r"\bexplosion\b", r"\bshooting\b",
            r"\bfire\b", r"\bquake\b", r"\bearthquake\b", r"\bstorm\b",
            r"\bflood\b", r"\bstrikes?\b", r"\bblasts?\b", r"\bshoots?\b",
            r"\bkills?\b", r"\bkilled\b", r"\bdead\b", r"\bcollides?\b",
            r"\bderails?\b", r"\bhostage\b", r"\bseized\b",
        ]),
        ("follow_up", [
            r"\blatest\b", r"\bupdate\b", r"\bwhat we know\b",
            r"\bwhat happened next\b", r"\bnew details\b",
            r"\bdevelopments?\b", r"\bcontinues?\b",
        ]),
    ]

    matches = [label for label, expressions in patterns
               if any(re.search(expr, text) for expr in expressions)]
    # Ambiguous/mixed signals are left unclassified rather than guessed.
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Investigation and response markers are more specific than generic
        # incident words often present in retrospective headlines.
        for preferred in ("investigation", "response", "aftermath"):
            if preferred in matches:
                return preferred
    return "unknown"


def development_relationship(type_a, type_b):
    """Return a descriptive edge between different developments of a story."""
    if type_a == type_b or "unknown" in {type_a, type_b}:
        return ""
    relation_map = {
        frozenset(("incident", "investigation")): "investigates",
        frozenset(("incident", "aftermath")): "aftermath_of",
        frozenset(("incident", "response")): "responds_to",
        frozenset(("investigation", "aftermath")): "related_development",
        frozenset(("investigation", "response")): "related_development",
        frozenset(("aftermath", "response")): "related_development",
        frozenset(("incident", "follow_up")): "follows",
        frozenset(("investigation", "follow_up")): "updates",
        frozenset(("aftermath", "follow_up")): "updates",
        frozenset(("response", "follow_up")): "updates",
    }
    return relation_map.get(frozenset((type_a, type_b)), "related_development")


def passes_development_relationship_similarity(
    lexical_score,
    semantic_score,
    semantic_available,
    label_compatible_entity_count,
    label_compatible_idf_score,
    common_entity_count,
    entity_overlap,
    shared_entity_idf_score,
    temporal_conflict,
):
    """Whether a cross-phase pair has independent evidence of story similarity."""
    if temporal_conflict:
        return False
    return bool(
        lexical_score >= SIMILARITY_THRESHOLD
        or (
            semantic_available
            and semantic_score >= SEMANTIC_MERGE_THRESHOLD
        )
        or (
            semantic_available
            and ENTITY_CONDITIONED_SEMANTIC_THRESHOLD <= semantic_score
            < ENTITY_CONDITIONED_SEMANTIC_UPPER
            and label_compatible_entity_count >= 1
            and label_compatible_idf_score > IDF_ENTITY_MIN_SCORE
        )
        or (
            lexical_score >= ENTITY_SIMILARITY_THRESHOLD
            and common_entity_count >= 1
            and entity_overlap >= 0.20
            and (
                not ENABLE_IDF_ENTITY_WEIGHTING
                or shared_entity_idf_score >= IDF_ENTITY_MIN_SCORE
            )
        )
    )


def cluster_articles(df):
    if df.empty:
        return df

    df = df.reset_index(drop=True)

    # Keep event identity anchored in headline + RSS summary. Full article
    # text is a weak supplementary signal only: raw body text often contains
    # background context shared by otherwise distinct events.
    titles = df["title"].fillna("").astype(str).tolist()
    summaries = df["summary"].fillna("").astype(str)
    development_types = [
        classify_event_development(title, summary)
        for title, summary in zip(titles, summaries)
    ]
    df["development_type"] = development_types
    bodies = (
        df["body_text"].fillna("").astype(str)
        if "body_text" in df.columns
        else pd.Series([""] * len(df))
    )

    def useful_body_sentences(body, title, limit=5):
        """Select a few lead sentences with title-term relevance."""
        words = {
            w for w in re.findall(r"[a-z]{3,}", normalize_title(title).lower())
            if w not in {"says", "said", "after", "over", "amid", "what", "from"}
        }
        candidates = []
        for sentence in re.split(r"(?<=[.!?])\s+", str(body or "")):
            sentence = re.sub(r"\s+", " ", sentence).strip()
            if not 45 <= len(sentence) <= 350:
                continue
            low = sentence.lower()
            if any(x in low for x in (
                "subscribe", "newsletter", "cookie", "privacy policy",
                "all rights reserved", "advertisement", "read more"
            )):
                continue
            tokens = set(re.findall(r"[a-z]{3,}", low))
            overlap = len(words & tokens) / max(len(words), 1)
            # Prefer relevant early sentences; do not let later background
            # paragraphs dominate event identity.
            candidates.append((overlap, sentence))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return " ".join(sentence for _, sentence in candidates[:limit])

    headline_summary_texts = [
        f"{title} {title} {title} {summary[:500]}"
        for title, summary in zip(titles, summaries)
    ]
    body_signal_texts = [
        useful_body_sentences(body, title)
        for body, title in zip(bodies, titles)
    ]

    if len(titles) == 1:
        df["cluster_id"] = "event_0001"
        return df

    vectorizer_main = TfidfVectorizer(
        ngram_range=(1, 2), stop_words="english",
        sublinear_tf=True, min_df=1,
        max_df=0.90,
    )
    main_matrix = vectorizer_main.fit_transform(headline_summary_texts)
    main_similarity = cosine_similarity(main_matrix)

    # Body text contributes at most 15% of lexical similarity. This limits
    # generic background vocabulary from overriding the event-specific lead.
    body_similarity = None
    if any(text.strip() for text in body_signal_texts):
        vectorizer_body = TfidfVectorizer(
            ngram_range=(1, 2), stop_words="english",
            sublinear_tf=True, min_df=1, max_df=0.90,
        )
        try:
            body_matrix = vectorizer_body.fit_transform(body_signal_texts)
            body_similarity = cosine_similarity(body_matrix)
        except ValueError:
            body_similarity = None

    similarity_matrix = (
        0.85 * main_similarity + 0.15 * body_similarity
        if body_similarity is not None else main_similarity
    )

    # Semantic embeddings remain anchored to headline + RSS summary. Do not
    # embed the whole article: broad background text can inflate false matches.
    semantic_matrix = None
    semantic_enabled = False
    if ENABLE_SEMANTIC_MATCHING:
        try:
            from sentence_transformers import SentenceTransformer
            semantic_model = SentenceTransformer(SEMANTIC_MODEL_NAME)
            embeddings = semantic_model.encode(
                headline_summary_texts, normalize_embeddings=True,
                show_progress_bar=False
            )
            semantic_matrix = embeddings @ embeddings.T
            semantic_enabled = True
            print(f"Semantic matching enabled: {SEMANTIC_MODEL_NAME}")
        except Exception as exc:
            print(
                "WARNING: semantic matching unavailable; using lexical/entity "
                f"matching only. Reason: {exc}"
            )

    def parse_article_date(value):
        if not value or not str(value).strip():
            return None
        try:
            parsed = pd.to_datetime(value, utc=True, errors="coerce")
            return None if pd.isna(parsed) else parsed
        except Exception:
            return None

    article_dates = [
        parse_article_date(value)
        for value in df.get(
            "published_utc", pd.Series([""] * len(df))
        ).fillna("")
    ]

    def date_gap_days(i, j):
        if article_dates[i] is None or article_dates[j] is None:
            return None
        return abs((article_dates[i] - article_dates[j]).total_seconds()) / 86400.0

    entity_records_by_article = [
        entity_records(value)
        for value in df["entities_json"].fillna("[]")
    ]
    entity_sets = [set(records) for records in entity_records_by_article]

    # Corpus-level entity IDF: distinctive entities receive higher weights.
    entity_df = Counter(entity for entities in entity_sets for entity in entities)
    n_articles = max(len(entity_sets), 1)
    entity_idf = {
        entity: math.log((n_articles + 1) / (document_frequency + 1)) + 1.0
        for entity, document_frequency in entity_df.items()
    }

    parent = list(range(len(df)))
    pairwise_audit_rows = []

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def cluster_members(root):
        return [k for k in range(len(df)) if find(k) == root]

    def union_if_coherent(a, b):
        root_a, root_b = find(a), find(b)
        if root_a == root_b:
            return True
        members_a = cluster_members(root_a)
        members_b = cluster_members(root_b)
        # A cluster cannot contain two distinct known development phases.
        # Crucially, this is checked across ALL cross-cluster members, so an
        # unknown article cannot bridge incident -> investigation transitively.
        if ENABLE_DEVELOPMENT_GATING:
            known_a = {development_types[k] for k in members_a
                       if development_types[k] != "unknown"}
            known_b = {development_types[k] for k in members_b
                       if development_types[k] != "unknown"}
            if known_a and known_b and known_a.isdisjoint(known_b):
                return False

        # Complete-link coherence: every cross-cluster pair must have enough
        # lexical or semantic support. A single bridge cannot merge unrelated
        # events into a large connected component.
        for x in members_a:
            for y in members_b:
                if (
                    ENABLE_DEVELOPMENT_GATING
                    and development_types[x] != "unknown"
                    and development_types[y] != "unknown"
                    and development_types[x] != development_types[y]
                ):
                    return False
                lexical = float(similarity_matrix[x, y])
                semantic = (
                    float(semantic_matrix[x, y])
                    if semantic_matrix is not None else 0.0
                )
                if max(lexical, semantic) < CLUSTER_MIN_PAIRWISE_SIMILARITY:
                    return False
                gap = date_gap_days(x, y)
                if (
                    gap is not None
                    and gap > MAX_EVENT_DATE_GAP_DAYS
                    and lexical < TEMPORAL_OVERRIDE_COSINE
                    and semantic < SEMANTIC_STRONG_MATCH_THRESHOLD
                ):
                    return False
        parent[root_b] = root_a
        return True

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            score = float(similarity_matrix[i, j])
            semantic_score = (
                float(semantic_matrix[i, j])
                if semantic_matrix is not None else 0.0
            )
            gap_days = date_gap_days(i, j)

            common_entities = entity_sets[i] & entity_sets[j]
            all_entities = entity_sets[i] | entity_sets[j]
            entity_overlap = (
                len(common_entities) / len(all_entities)
                if all_entities else 0.0
            )
            # Only count an entity for the conditional semantic rule when its
            # NER label agrees in both articles (e.g. PERSON with PERSON).
            label_compatible_entities = {
                entity for entity in common_entities
                if entity_records_by_article[i][entity]
                & entity_records_by_article[j][entity]
            }
            shared_idf_score = sum(
                entity_idf.get(entity, 1.0) for entity in common_entities
            )
            label_compatible_idf_score = sum(
                entity_idf.get(entity, 1.0)
                for entity in label_compatible_entities
            )

            # Experiment A: entity-overlap merges now require a minimum
            # aggregate IDF score, rather than unweighted Jaccard alone.
            # Temporal conflict blocks weak lexical/entity matches across
            # distant publication dates. Strong direct text matches can pass.
            development_a = development_types[i]
            development_b = development_types[j]
            development_conflict = (
                ENABLE_DEVELOPMENT_GATING
                and development_a != "unknown"
                and development_b != "unknown"
                and development_a != development_b
            )

            temporal_conflict = (
                gap_days is not None
                and gap_days > MAX_EVENT_DATE_GAP_DAYS
                and score < TEMPORAL_OVERRIDE_COSINE
                and semantic_score < SEMANTIC_STRONG_MATCH_THRESHOLD
            )

            # A phase difference alone does NOT establish that two articles
            # describe the same broader event. Emit a relationship only when
            # the pair independently passes a story-similarity rule.
            has_story_similarity = passes_development_relationship_similarity(
                lexical_score=score,
                semantic_score=semantic_score,
                semantic_available=semantic_matrix is not None,
                label_compatible_entity_count=len(label_compatible_entities),
                label_compatible_idf_score=label_compatible_idf_score,
                common_entity_count=len(common_entities),
                entity_overlap=entity_overlap,
                shared_entity_idf_score=shared_idf_score,
                temporal_conflict=temporal_conflict,
            )
            development_relation = (
                development_relationship(development_a, development_b)
                if has_story_similarity else ""
            )

            if development_conflict:
                candidate_rule = "blocked_by_development_mismatch"
            elif temporal_conflict:
                candidate_rule = "blocked_by_temporal_gap"
            elif score >= SIMILARITY_THRESHOLD:
                candidate_rule = "similarity_threshold"
            elif semantic_matrix is not None and semantic_score >= SEMANTIC_MERGE_THRESHOLD:
                candidate_rule = "semantic_similarity"
            elif (
                semantic_matrix is not None
                and ENTITY_CONDITIONED_SEMANTIC_THRESHOLD <= semantic_score
                    < ENTITY_CONDITIONED_SEMANTIC_UPPER
                and not temporal_conflict
                and len(label_compatible_entities) >= 1
                and label_compatible_idf_score > IDF_ENTITY_MIN_SCORE
            ):
                candidate_rule = "entity_conditioned_semantic_similarity"
            elif (
                score >= ENTITY_SIMILARITY_THRESHOLD
                and len(common_entities) >= 1
                and entity_overlap >= 0.20
                and (
                    not ENABLE_IDF_ENTITY_WEIGHTING
                    or shared_idf_score >= IDF_ENTITY_MIN_SCORE
                )
            ):
                candidate_rule = (
                    "idf_entity_overlap"
                    if ENABLE_IDF_ENTITY_WEIGHTING
                    else "similarity_plus_entity_overlap"
                )
            else:
                candidate_rule = "no_direct_merge"

            # Experiment B: apply a cluster-level coherence check to every
            # candidate merge to reduce transitive semantic drift.
            if candidate_rule in {
                "similarity_threshold",
                "idf_entity_overlap",
                "similarity_plus_entity_overlap",
                "semantic_similarity",
                "entity_conditioned_semantic_similarity",
            }:
                if (
                    not ENABLE_CLUSTER_COHERENCE
                    or union_if_coherent(i, j)
                ):
                    merge_rule = candidate_rule
                else:
                    merge_rule = "blocked_by_cluster_coherence"
            else:
                merge_rule = candidate_rule

            pairwise_audit_rows.append({
                "article_id_a": df.iloc[i].get("article_id", ""),
                "article_id_b": df.iloc[j].get("article_id", ""),
                "title_a": titles[i],
                "title_b": titles[j],
                "domain_a": df.iloc[i].get("domain", ""),
                "domain_b": df.iloc[j].get("domain", ""),
                "cosine_similarity": score,
                "headline_summary_similarity": float(main_similarity[i, j]),
                "body_similarity": (
                    float(body_similarity[i, j])
                    if body_similarity is not None else None
                ),
                "semantic_similarity": semantic_score if semantic_enabled else None,
                "extraction_status_a": df.iloc[i].get("extraction_status", ""),
                "extraction_status_b": df.iloc[j].get("extraction_status", ""),
                "published_date_gap_days": gap_days,
                "temporal_conflict": temporal_conflict,
                "development_type_a": development_a,
                "development_type_b": development_b,
                "development_conflict": development_conflict,
                "development_relationship": development_relation,
                "development_link_requires_similarity": bool(development_relation),
                "shared_entity_count": len(common_entities),
                "shared_entities": " | ".join(sorted(common_entities)),
                "entity_jaccard_overlap": entity_overlap,
                "shared_entity_idf_score": shared_idf_score,
                "label_compatible_shared_entities": " | ".join(
                    sorted(label_compatible_entities)
                ),
                "label_compatible_shared_entity_idf_score": (
                    label_compatible_idf_score
                ),
                "merge_rule_for_pair": merge_rule,
                # Fill these labels manually after reviewing the pair.
                "manual_label": "",
                "manual_notes": "",
            })

    root_to_cluster = {}
    cluster_ids = []

    for i in range(len(df)):
        root = find(i)
        if root not in root_to_cluster:
            root_to_cluster[root] = f"event_{len(root_to_cluster) + 1:04d}"
        cluster_ids.append(root_to_cluster[root])

    df["cluster_id"] = cluster_ids

    # Save pair-level diagnostics separately from the cluster report.
    # "same_cluster_after_clustering" reflects transitive grouping too:
    # a pair can share a final cluster even if that pair did not directly
    # pass either merge rule.
    pairwise_audit = pd.DataFrame(pairwise_audit_rows)
    relationship_path = REPORT_DIR / "event_development_relationships.csv"
    if not pairwise_audit.empty:
        cluster_lookup = dict(zip(df["article_id"], df["cluster_id"]))
        pairwise_audit["cluster_id_a"] = pairwise_audit["article_id_a"].map(cluster_lookup)
        pairwise_audit["cluster_id_b"] = pairwise_audit["article_id_b"].map(cluster_lookup)
        pairwise_audit["same_cluster_after_clustering"] = (
            pairwise_audit["cluster_id_a"] == pairwise_audit["cluster_id_b"]
        )
        relationship_rows = pairwise_audit[
            pairwise_audit["development_relationship"].astype(str).ne("")
        ].copy()
    else:
        # Avoid leaving stale pairwise/relationship CSVs from a previous larger run.
        pairwise_audit = pd.DataFrame(columns=[
            "article_id_a", "article_id_b", "title_a", "title_b", "domain_a", "domain_b",
            "cosine_similarity", "headline_summary_similarity", "body_similarity",
            "semantic_similarity", "published_date_gap_days", "temporal_conflict",
            "development_type_a", "development_type_b", "development_conflict",
            "development_relationship", "development_link_requires_similarity",
            "shared_entity_count", "shared_entities", "entity_jaccard_overlap",
            "shared_entity_idf_score", "label_compatible_shared_entities",
            "label_compatible_shared_entity_idf_score", "merge_rule_for_pair",
            "manual_label", "manual_notes", "cluster_id_a", "cluster_id_b",
            "same_cluster_after_clustering",
        ])
        relationship_rows = pairwise_audit.copy()
    pairwise_audit.to_csv(PAIRWISE_AUDIT_PATH, index=False)
    relationship_rows.to_csv(relationship_path, index=False)
    print(f"Saved pairwise clustering audit: {PAIRWISE_AUDIT_PATH}")
    print(f"Saved development relationships: {relationship_path}")
    print(f"Article pairs logged: {len(pairwise_audit)}")

    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            "UPDATE articles SET cluster_id = ? WHERE article_id = ?",
            [
                (row["cluster_id"], row["article_id"])
                for _, row in df.iterrows()
            ],
        )

    return df


def build_cluster_report(df):
    rows = []

    for cluster_id, group in df.groupby("cluster_id", sort=False):
        domains = sorted(set(group["domain"].dropna()) - {""})
        normalized_titles = set(
            group["normalized_title"].dropna()
        )
        categories = Counter(group["category"].dropna())
        entities = Counter()

        for raw_entities in group["entities_json"].fillna("[]"):
            for entity in entity_tokens(raw_entities):
                entities[entity] += 1

        # Domain count is observable; publisher independence is not proven.
        if len(group) == 1:
            status = "single_article"
        elif len(domains) >= 2 and len(normalized_titles) >= 2:
            status = "multi_domain_candidate"
        else:
            status = "possible_duplicate_or_single_domain"

        rows.append({
            "cluster_id": cluster_id,
            "article_count": len(group),
            "distinct_domains": len(domains),
            "distinct_headlines": len(normalized_titles),
            "publisher_independence": "unverified",
            "corroboration_status": status,
            "predominant_category": (
                categories.most_common(1)[0][0] if categories else ""
            ),
            "categories": json.dumps(dict(categories)),
            "recurring_entities": " | ".join(
                entity for entity, count in entities.most_common(10)
                if count >= 2
            ),
            "domains": " | ".join(domains),
            "headlines": " || ".join(group["title"].astype(str)),
            "urls": " | ".join(group["url"].astype(str)),
        })

    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values(
            ["distinct_domains", "article_count"],
            ascending=False,
        )

    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-fetch", action="store_true",
        help="Recompute classification/clusters from the existing database without fetching RSS feeds.",
    )
    args = parser.parse_args(argv)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing trained model: {MODEL_PATH}"
        )

    init_db()

    model = joblib.load(MODEL_PATH)
    nlp = spacy.load("en_core_web_sm")

    fetched = [] if args.no_fetch else fetch_articles()
    if fetched:
        store_articles(fetched)
    elif args.no_fetch:
        print("RSS fetching skipped; using articles already in the database.")

    df = load_articles()
    if df.empty:
        raise RuntimeError("No articles available in the evidence store.")

    print(f"\nArticles stored across all runs: {len(df)}")

    df = save_enrichment(df, model, nlp)

    print("\n=== LINEAR SVM HEADLINE CLASSIFICATION ===")
    print(
        df[["title", "category", "category_model"]]
        .head(30)
        .to_string(index=False)
    )

    print("\n=== PREDICTED CATEGORY COUNTS ===")
    print(df["category"].value_counts().to_string())

    df = cluster_articles(df)
    clusters = build_cluster_report(df)

    article_report_path = REPORT_DIR / "collected_articles.csv"
    cluster_report_path = REPORT_DIR / "event_clusters.csv"
    summary_path = REPORT_DIR / "event_fusion_summary.json"

    df.to_csv(article_report_path, index=False)
    clusters.to_csv(cluster_report_path, index=False)

    multi_domain = (
        int((clusters["corroboration_status"] == "multi_domain_candidate").sum())
        if not clusters.empty else 0
    )

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "feeds_configured": len(FEEDS),
        "new_feed_entries_seen": len(fetched),
        "articles_in_database": len(df),
        "articles_with_extracted_body": (
            int(df["extraction_status"].eq("extracted").sum())
            if "extraction_status" in df.columns else 0
        ),
        "classifier_model": MODEL_PATH.name,
        "classifier_input": "headline only",
        "decision_scores_note": (
            "LinearSVC decision scores are raw margins, not probabilities."
        ),
        "clustering_text": (
            "Headline + RSS summary form the primary representation; selected "
            "title-relevant body sentences contribute 15% of lexical similarity. "
            "Semantic similarity uses headline + RSS summary only."
        ),
        "body_signal_weight": 0.15,
        "body_signal_selection": (
            "Up to five title-relevant sentences, each 45-350 characters; "
            "body text is supplementary rather than concatenated wholesale."
        ),
        "event_clusters": len(clusters),
        "multi_domain_candidates": multi_domain,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "idf_entity_min_score": IDF_ENTITY_MIN_SCORE,
        "cluster_min_pairwise_similarity": CLUSTER_MIN_PAIRWISE_SIMILARITY,
        "idf_entity_weighting_enabled": ENABLE_IDF_ENTITY_WEIGHTING,
        "cluster_coherence_enabled": ENABLE_CLUSTER_COHERENCE,
        "semantic_matching_enabled_requested": ENABLE_SEMANTIC_MATCHING,
        "semantic_model": SEMANTIC_MODEL_NAME,
        "semantic_merge_threshold": SEMANTIC_MERGE_THRESHOLD,
        "entity_conditioned_semantic_threshold": (
            ENTITY_CONDITIONED_SEMANTIC_THRESHOLD
        ),
        "entity_conditioned_semantic_upper_bound": (
            ENTITY_CONDITIONED_SEMANTIC_UPPER
        ),
        "entity_conditioned_semantic_rule": (
            "Requires semantic score in [0.70, 0.76), no temporal conflict, "
            "at least one shared named entity with matching NER label, and "
            "aggregate IDF score > 3.0."
        ),
        "transitive_development_guard": (
            "Every cross-cluster merge checks known development types across "
            "all members; unknown articles cannot bridge distinct known phases."
        ),
        "max_event_date_gap_days": MAX_EVENT_DATE_GAP_DAYS,
        "temporal_override_cosine": TEMPORAL_OVERRIDE_COSINE,
        "development_gating_enabled": ENABLE_DEVELOPMENT_GATING,
        "development_classifier": "conservative keyword rules; unknown when ambiguous",
        "entity_idf_method": "log((number_of_articles + 1)/(entity_document_frequency + 1)) + 1",
        "clustering_experiments": (
            "IDF entity-overlap gate and cluster coherence safeguard are enabled "
            "together in this implementation; compare against the frozen "
            "88-pair dataset before treating them as improvements."
        ),
        "independence_assessment": (
            "Unverified: distinct domains are counted, but ownership, "
            "syndication, and editorial independence are not established."
        ),
        "limitations": [
            "RSS feed coverage is incomplete.",
            "Semantic matching requires sentence-transformers and model weights; "
            "if unavailable, the script falls back to lexical/entity rules.",
            "Temporal safeguards use publication-date gap as a heuristic, not "
            "proof that two articles describe different events.",
            "Development labels use keyword heuristics and require manual validation; "
            "different development types are kept in separate clusters by default.",
            "Headline similarity can merge separate events or split one event.",
            "NER may miss or misidentify people, places, and organizations.",
            "Distinct domains do not prove independent reporting.",
            "Topic classification is not factual verification.",
        ],
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== EVENT FUSION SUMMARY ===")
    print(json.dumps(summary, indent=2))

    print("\nTop clusters:")
    if not clusters.empty:
        print(clusters[
            [
                "cluster_id",
                "article_count",
                "distinct_domains",
                "corroboration_status",
                "predominant_category",
            ]
        ].head(15).to_string(index=False))

    print("\nSaved reports:")
    print(article_report_path)
    print(cluster_report_path)
    print(summary_path)
    print(PAIRWISE_AUDIT_PATH)
    print(REPORT_DIR / "event_development_relationships.csv")
    print(f"Evidence database: {DB_PATH}")
    return 0


if __name__ == "__main__":
    main()
