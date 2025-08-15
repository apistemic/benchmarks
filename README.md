# Apistemic Benchmarks

Since we do a lot of LLM-based company analysis at [apistemic](https://apistemic.com),
we decided to have one central place to keep track of all the benchmarks we do.
This repo thus covers many business/company-related LLM benchmarks.

## Ad-hoc Company Understanding
In this benchmark, we explore how well LLMs understand companies.
To do this, we prompt the name of the company to get an embedding.
These embeddings are then used in a complex regression task, i.e. scoring the competitiveness of two companies.
A task, that usually requires a complex understanding of companies, markets, business models, and more.

![benchmark of LLM embeddings](.data/plots/r2-scores-boxplot.png)
