"""Local news: collect candidate articles, fetch their text, label them.

``collect``  finds candidate URLs (GDELT DOC API, web search, hand-found)
``fetch``    downloads each one politely and extracts text + publish date

Article text lives only under ``raw/news/`` (gitignored, copyrighted). The
committed catalog ``labels/articles.csv`` holds metadata only.
"""
