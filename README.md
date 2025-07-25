# Thesis Hallucination Detection

This repository implements a pipeline to detect hallucinations in large language model (LLM) outputs based on similarity metrics, prompt features, and factuality signals.

## Repository Structure

```bash
├── data/
│   ├── raw/
│   │   └── prompts.csv           # 0. Raw prompts with categories, IDs, prompts, and reference texts
│   ├── intermediate/
│   │   ├── outputs.json          # Raw LLM responses (after fetch_responses.py)
│   │   ├── outputs_enriched.json # Added similarity & prompt features
│   │   ├── outputs_expanded.json # Added NLI contradiction scores
│   └── final/
│       └── feature_matrix.csv    # Merged feature matrix for modeling

├── notebooks/
│   └── EDA.ipynb                 # Interactive exploratory analysis

├── result/
│   ├── figures/                  # Plots and charts exported by eda.py
│   └── tables/                   # Summary tables exported by eda.py & modeling

├── src/
│   ├── fetch_responses.py        # 1. Queries LLMs and writes outputs.json
│   ├── similarity_test.py        # 2. Computes BERTScore, SBERT and prompt features → outputs_enriched.json
│   ├── eda.py                    # 3. Generates EDA tables/figures (uses outputs_enriched.json)
│   ├── compute_factuality_signals.py  # 4. Computes NLI contradiction → outputs_expanded.json
│   ├── assemble_feature_matrix.py     # 5. Merges enriched & expanded outputs into feature_matrix.csv
│   └── predictive_modeling.py    # 6. Trains & evaluates hallucination classifiers → result/tables

├── test.py                       # Unit tests for D-category prompts
├── test_D.py                     # Integration test for Category D pipeline

├── .env                          # Environment variables for API keys
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Pipeline Overview

1. **Raw Prompts** (`data/raw/prompts.csv`)
   Contains `prompt_id`, `category` (A–F), `prompt`, and `reference_text`.

2. **Fetch Responses** (`src/fetch_responses.py`)
   Reads `prompts.csv`, queries each LLM (OpenAI GPT-3.5, GPT-4o, Gemini, Claude, Cohere), and writes raw outputs to `data/intermediate/outputs.json`.

3. **Similarity & Prompt Features** (`src/similarity_test.py`)
   Computes:

   * **BERTScore F1** and **SBERT cosine similarity** between `response` and `reference_text`.
   * Prompt-level features: `token_len`, `char_len`, `entity_count`, `question_type`, `ambiguity_flag`, `pronoun_ratio`, `imperative_flag`, `readability_score`, `punct_density`.
     Saves enriched results to `data/intermediate/outputs_enriched.json`.

4. **Exploratory Data Analysis** (`src/eda.py` or `notebooks/EDA.ipynb`)
   Loads `outputs_enriched.json` to produce summary tables and plots, exporting to `result/figures/` and `result/tables/`.

5. **Factuality Signals** (`src/compute_factuality_signals.py`)
   Runs a RoBERTa-MNLI model on each response/reference pair to compute NLI contradiction probability, appending to `data/intermediate/outputs_expanded.json`.

6. **Feature Matrix Assembly** (`src/assemble_feature_matrix.py`)
   Merges `outputs_enriched.json` and `outputs_expanded.json`, selecting core features and labels into `data/final/feature_matrix.csv`.

7. **Predictive Modeling** (`src/predictive_modeling.py`)
   Loads `feature_matrix.csv`, trains logistic regression, decision tree, and random forest with 5-fold stratified CV, and saves performance results to `result/tables/`.

## Getting Started

1. **Clone the repo**

   ```bash
   git clone <repo_url>
   cd thesis-hallucination
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys** in `.env`:

   ```ini
   OPENAI_API_KEY=...
   GOOGLE_API_KEY=...
   ANTHROPIC_API_KEY=...
   COHERE_API_KEY=...
   ```

4. **Run the full pipeline**

   ```bash
   # Fetch responses
   python src/fetch_responses.py

   # Similarity & features
   python src/similarity_test.py

   # EDA
   python src/eda.py

   # Factuality signals
   python src/compute_factuality_signals.py

   # Assemble feature matrix
   python src/assemble_feature_matrix.py

   # Predictive modeling
   python src/predictive_modeling.py
   ```

5. **Inspect results**

   * **Figures**: `result/figures/`
   * **Tables**: `result/tables/`
   * **Feature matrix**: `data/final/feature_matrix.csv`

## Contributions

Contributions, issues, and feature requests are welcome. Feel free to open a pull request.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
