<h1 align="center">CHART: Configuration History for Adaptive Retrieval Tuning</h1>

<p align="center">
  <a href="https://github.com/chery-nextai">Chenchen Zhao</a><sup>*</sup>,
  <a href="https://github.com/chery-nextai">Yi Wei</a><sup>*</sup>,
  <a href="https://github.com/chery-nextai">Zhengyu Chen</a>,
  <a href="https://github.com/chery-nextai">Ma Xiao</a>,
  <a href="https://github.com/chery-nextai">Yijun Chen</a>,
  <a href="https://github.com/chery-nextai">Zhiquan Hu</a>,
  <a href="https://github.com/chery-nextai">Chao Han</a>,
  <a href="https://github.com/chery-nextai">Xiao Zhang</a><sup>†</sup>,
  <a href="https://github.com/chery-nextai">Chuang Dai</a>
</p>

<p align="center">NEXTAI Research Institute, Chery Group.</p>

<p align="center"><sub>* Equal contribution. † Corresponding author.</sub></p>

<p align="center">
  Budgeted configuration search for multimodal retrieval-augmented generation
</p>

<p align="center">
  <a href="#-overview">Overview</a> ·
  <a href="#-method">Method</a> ·
  <a href="#-demo">Demo</a> ·
  <a href="#-results">Results</a> ·
  <a href="#-getting-started">Getting started</a> ·
  <a href="#-reproducibility">Reproducibility</a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="Search space" src="https://img.shields.io/badge/Search%20space-288%20configurations-086B13">
  <img alt="Budget" src="https://img.shields.io/badge/Budget-30%20evaluations-101578">
  <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/License-MIT-760E80"></a>
</p>

## 💡 Overview

The best retrieval setup depends on the document collection. **CHART** searches text and image retrieval depth, chunk size, and embedding backend within a fixed budget, using observed rewards to choose each next configuration. It returns the highest-scoring configuration it evaluated.

**Redundancy-Discounted Support (RDS)** scores reference-answer support while discounting repeated facts. CHART selects the next configuration using trial memory and a distance-based exploration bonus; it uses no LLM for search decisions.

## 🧭 Method

<p align="center">
  <img src="assets/chart_framework.svg" alt="CHART framework: configuration search, episodic memory, geometry-aware selection, evidence retrieval, and reward feedback" width="100%">
</p>

Each configuration is a tuple of four choices:

| Component | Values |
| --- | --- |
| Text top-*k* | `5`, `10`, `20`, `40` |
| Image top-*k* | `0`, `1`, `3`, `5`, `10`, `20` |
| Chunk size | `64`, `128`, `256`, `512` |
| Embedding backend | `jina_nano`, `jina_small`, `qwen3_vl_2b` |

The search space has **288 configurations**. Each run starts with five shared evaluations, then makes 25 selections. CHART estimates each untried configuration's reward from its five nearest trials, adds a distance-based exploration bonus, and evaluates the highest-utility option.

For a question *q*, RDS combines the judge's support probability and expected semantic redundancy:

```text
reward(q) = P(support | q, evidence) × [1 − 0.5 × E(redundancy | q, evidence)]
configuration reward = mean reward(q) over the fixed question set
```

Images are transcribed into question-independent facts. RDS is an **evidence-level proxy**, not a recall metric or a direct measure of answer accuracy.

## 🎬 Demo

<p align="center">
  <img src="assets/chart_demo.gif" alt="Illustrative CHART vs Random 30-step reward trajectories for Pharmaceuticals, Finance EN, and TechSlides" width="100%">
</p>

<p align="center"><em>Illustrative trajectories selected post hoc; scores use historical JEV models. The complete 20-seed comparison below is the primary quantitative result.</em></p>

## 📈 Results

We compare CHART with Random, coordinate descent, TPE, and an LLM controller on three document collections. Each uses 50 fixed questions and 20 paired seeds; methods share five starting configurations and a 30-evaluation budget. **Reward@k** is the best RDS reward in the first *k* evaluations, averaged across seeds.

| Dataset | CHART @10 | Random @10 | CHART @30 | Random @30 | Coordinate descent @30 | TPE @30 | LLM @30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pharmaceuticals | **0.6881** | 0.6792 | **0.6969** | 0.6848 | 0.6950 | 0.6919 | 0.6886 |
| Finance EN | **0.7244** | 0.7074 | **0.7359** | 0.7203 | 0.7337 | 0.7326 | 0.7243 |
| Techslides | **0.7516** | 0.7473 | 0.7627 | 0.7532 | **0.7628** | 0.7601 | 0.7601 |

These are the complete **20-seed** results. CHART's mean Reward@30 is higher than Random's on all three datasets; coordinate descent is slightly higher on Techslides. These are point estimates. The [full report](docs/20261104_three_dataset_rep20_report.md) includes all five budgets and the rule-augmented ablation.

<p align="center">
  <img src="assets/chart_budget_curves.svg" alt="CHART reward curves over 20 paired seeds across Pharma, Finance, TechSlides, and their macro average" width="100%">
</p>

<p align="center"><em>Mean best-so-far RDS reward over 20 paired seeds; shading shows ±1 standard error. The macro panel averages the three datasets within each seed before computing uncertainty.</em></p>

## 🔧 Getting started

### 1. Install

```bash
git clone https://github.com/chery-nextai/CHART.git chart
cd chart
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `.env` with model endpoints, credentials, corpus paths, Elasticsearch, and local embedding paths. The example file lists the variables. Running a search also requires these services and prepared indexes.

### 2. Prepare data and indexes

The repository includes **50-question subsets**, but not source documents, images, model weights, or indexes. Obtain each corpus under its license and prepare 15 indexes per dataset: 12 text and 3 image indexes.

```bash
bash scripts/run.sh --validate-only
```

For index-building options, run `python scripts/build_ao_dataset_index.py --help`.

### 3. Run CHART

```bash
DATASET=techslides POLICIES=jit_no_rules SEEDS=20 STEPS=25 bash scripts/run.sh
```

`jit_no_rules` runs CHART; legacy `jit` adds rules and is an ablation. The old `mgrs` package name, `MGRS_*` variables, and `dqcr` scheme remain for compatibility with saved runs.

`DATASET` selects the collection; an empty `POLICIES` runs all six strategies. `SEEDS` sets repeats and `STEPS` sets decisions after initialization. Use `TEXT_TOPK`, `IMAGE_TOPK`, `CHUNKS`, and `BACKENDS` to narrow the search space; see [the frozen configuration](configs/definitive_runs.json).

## 📂 Data and indexes

| Dataset | Document pages | Included evaluation subset |
| --- | ---: | --- |
| `pharmaceuticals` | 2,313 | [`qa_screened50_from_200.json`](datasets/ao_ready_v1_subsets/pharmaceuticals/qa_screened50_from_200.json) |
| `finance_en` | 2,942 | [`qa_screened50_from_179.json`](datasets/ao_ready_v1_subsets/finance_en/qa_screened50_from_179.json) |
| `techslides` | 1,963 | [`qa_screened50_from_200.json`](datasets/ao_ready_v1_subsets/techslides/qa_screened50_from_200.json) |

Each subset has 50 questions with reference answers. Provide `corpus.json` and its image files, then set `MGRS_AO_CORPUS_ROOT` or a dataset-specific `MGRS_AO_CORPUS_DIR_*` in `.env`. Set `MGRS_INDEX_PREFIX_*` if index names differ. Follow the corpus publishers' licenses.

## 🔁 Reproducibility

The [frozen configuration](configs/definitive_runs.json) records the protocol and run IDs. The [result report](docs/20261104_three_dataset_rep20_report.md) contains the 20-seed results. Historical runs use `jit_no_rules` for CHART and `jit` for the rule-augmented ablation.

```bash
# After the three frozen run directories have been restored under runs/:
OUT=/tmp/chart_report.md bash scripts/reproduce_report.sh
```

`reproduce_report.sh` rebuilds the report offline from saved trajectories. **The required `runs/20261103_*` files are not included**, so a fresh clone needs those artifacts or a rerun. `scripts/run_search.sh` is the frozen rerun entry point; `rep6` in old run IDs is historical, and the final experiment has 20 repeats.

For comparable reruns, keep the judge and `MGRS_JUDGE_SCHEMA_TAG` aligned with the frozen protocol. Changing the judge or reward requires a new tag and run ID; do not reuse old response caches. Frozen runs used `Jev_AO`, `qwen3.7-flash` for image transcription, `qwen3.8-max` for the LLM baseline, and `γ = 0.5`. `dqcr` remains only as a legacy cache key for the same reward.

## 🗺️ Code map

```text
mgrs/
├── pipeline/             Search entry point, policies, run layout, and RDS
├── memory/               Configuration and reward memory
├── controllers/          LLM, coordinate descent, and other baselines
├── rag_pipeline/         Text and image retrieval, encoding, and model clients
├── analysis/             Frozen-run analysis and report generation
└── judge_client.py       Judge API adapters and response validation
scripts/
├── run.sh                General experiment entry point
├── run_search.sh         Frozen-protocol rerun entry point
├── build_ao_dataset_index.py
└── reproduce_report.sh Offline report generation from saved runs
configs/definitive_runs.json
datasets/ao_ready_v1_subsets/
docs/20261104_three_dataset_rep20_report.md
```

## 📜 License

Code is released under the [MIT License](LICENSE). Dataset and model terms are set by their publishers.
