# OSINT Intelligence Fusion & Evidence Analysis Platform

An experimental OSINT platform for collecting news, identifying events,
enriching articles, retrieving related evidence, and organizing
claim--evidence analysis into structured intelligence reports.

The system combines traditional machine-learning classification, NLP
enrichment, hybrid information retrieval, neural relevance and stance
models, and human review. It assists investigation and evidence
organization; it does not make definitive factual judgments
autonomously.

## Overview

The platform is organized around a local SQLite news database. It
ingests articles from configured RSS feeds, classifies headlines,
enriches collected articles with NLP features, and retrieves passages
that may be relevant to a claim.

For evidence analysis, it uses hybrid retrieval and reranking. Retrieved
claim--evidence pairs can then be passed to fine-tuned neural models to
estimate relevance and stance. Results are stored for review and can be
incorporated into intelligence reports.

The retrieval stage searches articles and passages collected in the
platform's own database. FEVER is used in model development and
fine-tuning; it is not the live evidence source.

## System workflow

``` text
RSS feeds
   |
   v
Article ingestion
   |
   v
Headline classification
   |
   v
SQLite article database
   |
   v
Article enrichment
   |-- sentence extraction
   |-- named-entity extraction
   |-- keyphrase / text features
   |
   v
Claim–evidence retrieval
   |-- TF-IDF retrieval
   |-- sentence-transformer retrieval
   |-- Reciprocal Rank Fusion (RRF)
   |-- cross-encoder reranking
   |
   v
NLI analysis of claim–evidence pairs
   |-- relevance score
   |-- stance signal
   |
   v
Evidence aggregation and human review
   |
   v
Intelligence report generation
```

## Key components

  --------------------------------------------------------------------------------------------------
  Component                                                      Responsibility
  -------------------------------------------------------------- -----------------------------------
  `event_fusion_development_aware_revised.py`                    Collects articles from configured
                                                                 RSS feeds, classifies headlines,
                                                                 and supports event
                                                                 grouping/reporting.

  `enrich_articles.py`                                           Enriches stored articles with
                                                                 sentence-level and NLP information.

  `evidence_verification_hybrid_event_constrained_reranked.py`   Retrieves candidate evidence using
                                                                 lexical and semantic methods,
                                                                 combines rankings with RRF, and
                                                                 reranks candidates with a
                                                                 cross-encoder.

  `nli_verification.py`                                          Runs the fine-tuned relevance and
                                                                 stance models on claim--evidence
                                                                 pairs.

  `claim_verification_aggregation.py`                            Organizes evidence signals and
                                                                 human assessments. Model
                                                                 predictions do not independently
                                                                 establish a factual verdict.

  `intelligence_report_generator.py`                             Produces structured intelligence
                                                                 reports, with optional LLM-assisted
                                                                 report writing if configured.

  `train_headline.py`                                            Trains the headline classifier and
                                                                 writes
                                                                 `models/headline_svm.joblib`.

  `predict.py`                                                   Runs standalone headline
                                                                 classification using the headline
                                                                 model.
  --------------------------------------------------------------------------------------------------

The exact command-line arguments and configuration options are defined
in each script. Check a script's `--help` output before running it.

## Machine-learning models

### Headline classification

The headline classifier uses TF-IDF features with a Linear Support
Vector Classifier (LinearSVC), trained using the MN-DS dataset.

The project has a recorded test result of 85.5% accuracy and 0.862 macro
F1 for one reported run. These figures describe headline classification
only; they are not evidence-verification metrics.

The trained model is expected at:

``` text
models/headline_svm.joblib
```

### Evidence retrieval and reranking

The evidence retrieval component combines lexical and semantic
retrieval. Sentence embeddings are generated with `all-MiniLM-L6-v2`;
candidate rankings are combined using Reciprocal Rank Fusion (RRF). A
cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reranks candidate
passages.

Retrieval and reranking scores are ranking signals. They are not
calibrated probabilities that a passage is true and do not independently
establish whether a claim is correct.

### Relevance and stance models

Two existing pretrained transformer models were fine-tuned using
FEVER-derived claim--evidence examples:

1.  **Relevance model:** estimates whether an evidence passage is
    relevant to a claim.
2.  **Stance model:** estimates whether evidence supports or refutes a
    claim.

The stance training described for this project used `SUPPORTS` and
`REFUTES` labels. Do not interpret it as a validated three-way
classifier for `SUPPORTS`, `REFUTES`, and `NOT ENOUGH INFO` unless
neutral-class training and evaluation are separately documented.

These models analyse claim--evidence pairs retrieved from the platform's
own news database. FEVER is a training resource, not the live evidence
corpus.

## FEVER-based fine-tuning

The `scripts/FEVER_preparation/` directory contains dataset preparation,
validation, training, evaluation, and test tooling used for
FEVER-related experiments.

The fine-tuning workflow adapts pretrained transformer models to
evidence relevance estimation and stance classification. Document the
exact base-model identifiers, dataset versions and splits, label
mapping, training parameters, and evaluation metrics in the
corresponding scripts and model metadata.

FEVER-derived training examples must not be confused with evidence
collected by the live OSINT ingestion pipeline. Check the dataset
licence and terms before redistributing any FEVER files. This repository
is not intended to redistribute the FEVER dataset or its associated
Wikipedia dump.

## Technology stack

-   Python and SQLite
-   scikit-learn, TF-IDF, and LinearSVC
-   PyTorch and Hugging Face Transformers
-   Sentence Transformers and cross-encoder reranking
-   spaCy
-   RSS parsing and feed ingestion
-   Optional Gemini integration for report generation

See `requirements.txt` and `requirements-nli.txt` for dependencies. Some
components may require additional language packages or pretrained model
downloads.

## Repository structure

A clean public repository can follow this layout:

``` text
.
├── README.md
├── README_NLI.md
├── requirements.txt
├── requirements-nli.txt
├── src/
│   ├── event_fusion_development_aware_revised.py
│   ├── enrich_articles.py
│   ├── evidence_verification_hybrid_event_constrained_reranked.py
│   ├── nli_verification.py
│   ├── claim_verification_aggregation.py
│   ├── intelligence_report_generator.py
│   ├── train_headline.py
│   └── predict.py
├── scripts/
│   └── FEVER_preparation/
├── models/
│   ├── headline_svm.joblib
│   └── README.md
├── test/
└── docs/
```

This is an intended public-facing layout, not a guarantee that every
file is present in every checkout. Preserve actual filenames and
document omitted or experimental scripts.

## Setup

### 1. Clone the repository

``` bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

### 2. Create and activate a virtual environment

macOS / Linux:

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

``` powershell
py -m venv .venv
.venv\\Scripts\\Activate.ps1
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For the NLI components:

``` bash
pip install -r requirements-nli.txt
```

## Configuration

Review the feed settings and configuration used by the ingestion and
report-generation scripts before running the system.

If optional external services are enabled, provide credentials through
environment variables or a local configuration file excluded from
version control. Never commit API keys, tokens, passwords, database
files, or private configuration.

A public `config.example.json` should contain placeholders only and
should match the configuration actually read by the application. Do not
assume a configuration file is required if the scripts do not load it.

## Running the pipeline

The scripts are independently executable, and their options can vary by
version. Inspect their help output first:

``` bash
python src/event_fusion_development_aware_revised.py --help
python src/enrich_articles.py --help
python src/evidence_verification_hybrid_event_constrained_reranked.py --help
python src/nli_verification.py --help
python src/intelligence_report_generator.py --help
```

A typical processing sequence is:

1.  Run article ingestion to collect RSS items into the configured
    SQLite database.
2.  Run article enrichment to extract sentence and entity information.
3.  Submit a claim to the evidence retrieval component.
4.  Run NLI analysis on retrieved claim--evidence pairs.
5.  Review evidence and human assessments.
6.  Generate an intelligence report.

Use the actual arguments and execution order supported by the current
scripts; this is not a guaranteed one-command deployment.

### Standalone headline prediction

After placing the trained model at `models/headline_svm.joblib`:

``` bash
python src/predict.py "Example news headline to classify"
```

The script can also accept a UTF-8 text file or an explicit model path.
Check:

``` bash
python src/predict.py --help
```

LinearSVC decision scores are raw margins, not probabilities.

## Model files

The headline model is expected at `models/headline_svm.joblib`.

The fine-tuned relevance and stance transformer checkpoints are
comparatively large. A practical arrangement is to host them in a model
repository (for example, Hugging Face Hub) and document their
identifiers and download instructions in `models/README.md`, rather than
committing large checkpoint files to Git.

Model documentation should specify:

-   Base model and checkpoint identifier
-   Training dataset and split
-   Label definitions and mapping
-   Fine-tuning procedure and parameters
-   Evaluation metrics and test-set details
-   Intended use and known limitations

Do not publish model weights or datasets unless their licences and
redistribution terms permit it.

## Limitations

-   **Not an automated fact-checker:** model outputs are assistance
    signals, not factual determinations.
-   **Evidence coverage:** results are limited by collected articles,
    publisher coverage, retrieval quality, and database freshness.
-   **Retrieval errors:** relevant passages may be missed, and
    irrelevant passages may be retrieved.
-   **Model scores:** relevance, stance, embedding similarity,
    cross-encoder scores, and LinearSVC margins are not automatically
    calibrated probabilities.
-   **Stance label coverage:** the described stance fine-tuning used
    `SUPPORTS` and `REFUTES`; neutral / `NOT ENOUGH INFO` handling must
    not be assumed validated.
-   **Publisher independence:** different domains do not necessarily
    represent independent confirmation; articles may repeat the same
    underlying source.
-   **LLM-generated text:** optional LLM-assisted report writing can
    introduce unsupported interpretations. Verify report statements
    against source articles and evidence.
-   **Dataset and domain shift:** benchmark-trained models may behave
    differently on current news, unfamiliar topics, languages, and
    publisher styles.

Human review and source inspection remain necessary before using reports
for consequential decisions.

## Responsible use and data

Use the platform for lawful research and analysis. Respect publisher
terms, copyright, privacy, and applicable data-protection rules when
collecting or storing articles. Keep personal data and credentials out
of public commits. Clearly distinguish sourced facts, model-generated
assessments, and analyst interpretations in reports.

## Project status

This is a research and development project. Model performance and
pipeline behaviour should be evaluated on representative, independently
reviewed data before operational or high-stakes use.
