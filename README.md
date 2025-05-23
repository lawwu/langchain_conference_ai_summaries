# LangChain Conference AI Summaries

Generating AI Summaries of transcripts from the LangChain Interrupt 2025 conference.

The transcripts are available here: https://lawwu.github.io/transcripts/index_langchain_interrupt_2025_official.html

## Setup

```bash
uv venv
uv sync
```

## Usage

```bash
python summarize_transcripts.py
```

The summaries will be saved to `summaries.json`.
A blog post will be generated in `langchain_interrupt_2025_summaries.md`.

## TODO

- Could generalize this to work on any conference