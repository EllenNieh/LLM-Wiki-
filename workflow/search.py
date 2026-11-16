import sqlite3
import sys
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / 'index' / 'wiki.db'


def search(query, limit=5):
    with sqlite3.connect(DB) as conn:
        conn.row_factory = sqlite3.Row
        # FTS5 unicode61 primarily tokenizes words, so English technical terms work best.
        terms = [x for x in query.replace('？', ' ').replace('?', ' ').split() if x]
        if not terms:
            return []
        fts_query = ' OR '.join('"' + x.replace('"', '""') + '"' for x in terms)
        return [dict(r) for r in conn.execute('''
            SELECT c.id, d.title, d.url, snippet(chunk_search, 1, '[', ']', '…', 30) AS excerpt,
                   bm25(chunk_search) AS score
            FROM chunk_search
            JOIN chunks c ON c.id = chunk_search.id
            JOIN documents d ON d.source_id = c.source_id
            WHERE chunk_search MATCH ?
            ORDER BY score LIMIT ?''', (fts_query, limit))]


if __name__ == '__main__':
    if not DB.exists():
        raise SystemExit('Run python workflow/ingest.py first')
    query = ' '.join(sys.argv[1:]) or 'reentrancy security'
    for result in search(query):
        print(f"\n{result['id']} | {result['title']}\n{result['url']}\n{result['excerpt']}\n")
