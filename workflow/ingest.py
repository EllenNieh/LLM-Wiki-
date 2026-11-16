import hashlib
import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'sources' / 'manifest.json'
DB = ROOT / 'index' / 'wiki.db'
RAW = ROOT / 'sources' / 'raw'
LOG = ROOT / 'logs' / 'ingest_log.json'


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def normalize(s):
    return re.sub(r'\s+', ' ', s).strip()


def extract(html):
    soup = BeautifulSoup(html, 'lxml')
    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript', 'form']):
        tag.decompose()
    main = soup.find('main') or soup.find('article') or soup.body
    if main is None:
        return ''
    pieces = []
    for el in main.find_all(['h1', 'h2', 'h3', 'p', 'li', 'pre']):
        if el.find_parent(['p', 'li', 'pre']):
            continue
        text = normalize(el.get_text(' ', strip=True))
        if len(text) >= 25:
            pieces.append(text)
    return '\n\n'.join(pieces)


def chunks(text, max_chars=1400):
    parts = text.split('\n\n')
    output, current = [], ''
    for part in parts:
        if len(part) > max_chars:
            subparts = [part[i:i+max_chars] for i in range(0, len(part), max_chars)]
        else:
            subparts = [part]
        for sub in subparts:
            if current and len(current) + len(sub) + 2 > max_chars:
                output.append(current)
                current = ''
            current = (current + '\n\n' + sub).strip()
    if current:
        output.append(current)
    return output


def initialize(conn):
    conn.execute('CREATE TABLE IF NOT EXISTS documents (source_id TEXT PRIMARY KEY, title TEXT, url TEXT, topic TEXT, fetched_at TEXT, sha256 TEXT, char_count INTEGER)')
    conn.execute('CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, source_id TEXT, position INTEGER, content TEXT, FOREIGN KEY(source_id) REFERENCES documents(source_id))')
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunk_search USING fts5(id UNINDEXED, content, tokenize='unicode61')")
    conn.commit()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    DB.parent.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    sources = json.loads(MANIFEST.read_text(encoding='utf-8'))
    results = []
    with sqlite3.connect(DB) as conn:
        initialize(conn)
        for source in sources:
            entry = {'id': source['id'], 'url': source['url'], 'status': 'failed', 'attempts': 0}
            for attempt in range(1, 4):
                entry['attempts'] = attempt
                try:
                    response = requests.get(source['url'], headers={'User-Agent': 'SmartContractWikiCourseProject/1.0 (educational research)'}, timeout=25)
                    response.raise_for_status()
                    text = extract(response.text)
                    if len(text) < 500:
                        raise ValueError(f'Parsed article too short ({len(text)} chars)')
                    pieces = chunks(text)
                    if not pieces:
                        raise ValueError('No chunks generated')
                    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
                    with conn:
                        conn.execute('DELETE FROM chunk_search WHERE id IN (SELECT id FROM chunks WHERE source_id=?)', (source['id'],))
                        conn.execute('DELETE FROM chunks WHERE source_id=?', (source['id'],))
                        conn.execute('INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?)', (source['id'], source['title'], source['url'], source['topic'], utcnow(), digest, len(text)))
                        for i, content in enumerate(pieces, 1):
                            chunk_id = f"{source['id']}-{i:03d}"
                            conn.execute('INSERT INTO chunks VALUES (?,?,?,?)', (chunk_id, source['id'], i, content))
                            conn.execute('INSERT INTO chunk_search (id, content) VALUES (?,?)', (chunk_id, content))
                    (RAW / f"{source['id']}.txt").write_text(text, encoding='utf-8')
                    entry.update(status='success', char_count=len(text), chunk_count=len(pieces), sha256=digest)
                    break
                except (requests.RequestException, ValueError, sqlite3.Error) as exc:
                    entry['error'] = str(exc)
                    if attempt < 3:
                        time.sleep(attempt)
            results.append(entry)
            print(f"{source['id']}: {entry['status']} (attempts={entry['attempts']})")
    summary = {'run_at': utcnow(), 'successful_sources': sum(r['status']=='success' for r in results), 'total_sources': len(sources), 'results': results}
    LOG.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Successful sources: {summary['successful_sources']}/{len(sources)}")
    if summary['successful_sources'] < 5:
        raise SystemExit('Ingest acceptance failed: fewer than five sources succeeded. See logs/ingest_log.json')

if __name__ == '__main__':
    main()
