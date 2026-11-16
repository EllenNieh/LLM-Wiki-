"""Phase 2: source-grounded Gemini teaching-note graph. Run from project root."""
import argparse
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from search import DB, search  # noqa: E402

TOPICS = {
    'reentrancy': ('重入攻擊', ['reentrancy', 'checks effects interactions', 'external calls']),
    'smart-contract': ('智慧合約', ['smart contract', 'ethereum', 'EVM']),
    'gas': ('Gas 費用', ['gas', 'transaction fees', 'EVM']),
    'access-control': ('存取控制', ['access control', 'authorization', 'ownership']),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def retrieve(terms, limit=8):
    found = {}
    for term in terms:
        for hit in search(term, limit):
            found.setdefault(hit['id'], hit)
    if not found:
        return []
    with sqlite3.connect(DB) as conn:
        conn.row_factory = sqlite3.Row
        for item in found.values():
            row = conn.execute('SELECT content FROM chunks WHERE id=?', (item['id'],)).fetchone()
            item['content'] = row['content'] if row else ''
    return list(found.values())[:10]


def retrieval_check(hits, topic):
    if not hits:
        return False, 'no_hits'
    terms = {'reentrancy': 'reentran', 'smart-contract': 'smart contract',
             'gas': 'gas', 'access-control': 'access control'}
    needle = terms[topic]
    matches = sum(needle in h['content'].lower() for h in hits)
    return (matches >= 1, f'matching_chunks={matches}/{len(hits)}')


def ask_json(client, model, prompt):
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type='application/json',
        ),
    )
    if not response.text:
        raise ValueError('Gemini returned empty response')
    return json.loads(response.text)


def generate(client, model, title, hits, feedback=''):
    evidence = [{'id': h['id'], 'source': h['url'], 'text': h['content'][:1600]} for h in hits]
    prompt = f'''你是繁體中文區塊鏈初學者教材編輯。只使用提供的 Wiki 證據撰寫「{title}」。
不可添加 Wiki 沒有支持的具體事實。每個重點都要引用至少一個 chunk id，引用必須真實存在。
如果資料不足，請在 limitations 說明，不要猜測。
回傳 JSON 物件，欄位：title (字串), introduction (字串), points (陣列，每項包含 text 字串與 citations 字串陣列), limitations (字串)。
重點 3~5 項，內容簡明且易懂。前次檢查回饋：{feedback}
Wiki 證據：{json.dumps(evidence, ensure_ascii=False)}'''
    return ask_json(client, model, prompt)


def structural_check(draft, hits):
    if not isinstance(draft, dict) or not isinstance(draft.get('points'), list):
        return False, 'invalid_schema'
    allowed = {h['id'] for h in hits}
    if not 3 <= len(draft['points']) <= 5:
        return False, 'point_count_out_of_range'
    for point in draft['points']:
        if not isinstance(point, dict) or not isinstance(point.get('text'), str) or not point['text'].strip():
            return False, 'empty_point'
        refs = point.get('citations')
        if not isinstance(refs, list) or not refs or any(r not in allowed for r in refs):
            return False, 'missing_or_invalid_citation'
    return True, 'citations_and_schema_valid'


def fact_check(client, model, draft, hits):
    evidence = {h['id']: h['content'][:1600] for h in hits}
    prompt = f'''請以嚴格的證據審查員角色判斷每個教學重點是否受到引用的 Wiki 片段支持。
不要使用模型自身知識補足。回傳 JSON 物件：passed (布林), issues (字串陣列)。
若某項僅部分受到支持，passed 應為 false。
教材：{json.dumps(draft, ensure_ascii=False)}
證據：{json.dumps(evidence, ensure_ascii=False)}'''
    result = ask_json(client, model, prompt)
    if not isinstance(result.get('passed'), bool) or not isinstance(result.get('issues'), list):
        raise ValueError('Invalid fact check response')
    return result


def run(topic, model, max_retries=2):
    load_dotenv(ROOT / '.env')
    if not os.getenv('GEMINI_API_KEY'):
        raise SystemExit('Missing GEMINI_API_KEY in .env')
    if not DB.exists():
        raise SystemExit('Wiki database missing. Run python workflow/ingest.py first.')
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
    title, queries = TOPICS[topic]
    log = {'started_at': now(), 'topic': topic, 'model': model, 'events': [], 'status': 'needs_review'}
    feedback = ''
    draft = None
    for attempt in range(max_retries + 1):
        # First query is focused; subsequent attempts broaden the evidence search.
        active = queries[:min(len(queries), attempt + 1)]
        hits = retrieve(active)
        ok, reason = retrieval_check(hits, topic)
        log['events'].append({'step': 'retrieve_check', 'attempt': attempt + 1, 'queries': active, 'chunks': [h['id'] for h in hits], 'passed': ok, 'reason': reason})
        if not ok:
            feedback = reason
            continue
        try:
            draft = generate(client, model, title, hits, feedback)
            valid, reason = structural_check(draft, hits)
            log['events'].append({'step': 'structural_check', 'attempt': attempt + 1, 'passed': valid, 'reason': reason})
            if not valid:
                feedback = reason
                continue
            verdict = fact_check(client, model, draft, hits)
            log['events'].append({'step': 'fact_check', 'attempt': attempt + 1, **verdict})
            if verdict['passed']:
                log['status'] = 'approved'
                log['approved_chunk_ids'] = sorted({r for p in draft['points'] for r in p['citations']})
                break
            feedback = '; '.join(map(str, verdict['issues']))[:700]
        except Exception as exc:
            # Never log API key, prompt, or private environment data.
            log['events'].append({'step': 'api_or_parse_error', 'attempt': attempt + 1, 'error_type': type(exc).__name__})
            feedback = 'API or JSON error; retry'
    log['finished_at'] = now()
    (ROOT / 'logs').mkdir(exist_ok=True)
    (ROOT / 'outputs').mkdir(exist_ok=True)
    (ROOT / 'logs' / f'graph_{topic}.json').write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding='utf-8')
    if log['status'] == 'approved':
        (ROOT / 'outputs' / f'{topic}.json').write_text(json.dumps(draft, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'APPROVED: outputs/{topic}.json')
    else:
        print('NEEDS REVIEW: no approved output was saved')
    print(f'Log: logs/graph_{topic}.json')
    return log['status'] == 'approved'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', choices=TOPICS, default='reentrancy')
    parser.add_argument('--model', default=os.getenv('GEMINI_MODEL', 'gemini-3.5-flash-lite'))
    args = parser.parse_args()
    raise SystemExit(0 if run(args.topic, args.model) else 1)
