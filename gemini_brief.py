import chromadb
import os

# Loaded ONCE at startup
client = chromadb.PersistentClient(path='data/chromadb')
col = client.get_or_create_collection('viral_sequences')

GROQ_KEY = os.environ.get('GROQ_API_KEY', '')


def _get_context(sequence: str) -> str:
    try:
        results = col.query(query_texts=[sequence], n_results=5)
        return '\n'.join([
            f"{m['virus_name']} | Danger={m['label']} | {m.get('notes', 'N/A')}"
            for m in results['metadatas'][0]
        ])
    except Exception:
        return 'Database unavailable — using general knowledge'


def _build_prompt(sequence: str, scores: dict, context: str) -> str:
    threat = scores.get('combined', scores.get('threat_index', 50))
    return f"""You are a WHO pandemic biosurveillance AI. CRISIS SIGNAL DETECTED.

Threat Index: {threat}/100
K-mer Novelty: {scores.get('kmer', 50)}/100
ESM-2 Danger:  {scores.get('esm2', 50)}/100
Structural:    {scores.get('structural', 'N/A')}

5 Most Similar Known Viruses:
{context}

Write a WHO-style crisis brief with these sections:
1. SITUATION: What this sequence appears to be
2. THREAT LEVEL: Low/Medium/High/Critical with confidence %
3. WHY DANGEROUS: Structural and biological reasoning
4. SIMILAR OUTBREAKS: Historical comparison
5. RECOMMENDED ACTIONS: Specific immediate steps

Then provide a full Urdu translation."""


def _stream_groq(prompt: str):
    from groq import Groq
    client_groq = Groq(api_key=GROQ_KEY)
    stream = client_groq.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[{'role': 'user', 'content': prompt}],
        stream=True,
        max_tokens=2000
    )
    for chunk in stream:
        text = chunk.choices[0].delta.content
        if text:
            yield text


def _static_fallback(scores: dict):
    threat = scores.get('combined', scores.get('threat_index', 50))
    level = 'CRITICAL' if threat > 75 else 'HIGH' if threat > 50 else 'MEDIUM'
    yield f"""SITUATION REPORT — PROTEINWATCH BIOSURVEILLANCE

1. SITUATION
Novel viral sequence detected. Automated pipeline analysis complete.

2. THREAT LEVEL: {level} — Confidence {min(99, int(threat))}%
   Threat Index: {threat}/100

3. WHY DANGEROUS
   ESM-2 danger score: {scores.get('esm2', 50)}/100
   K-mer novelty: {scores.get('kmer', 50)}/100

4. SIMILAR OUTBREAKS
   Closest match identified in surveillance database.

5. RECOMMENDED ACTIONS
   • Notify WHO Global Outbreak Alert and Response Network
   • Deploy field investigation team
   • Issue travel health advisory if human-to-human transmission confirmed
   • Begin PCR diagnostic protocol development

اردو خلاصہ: نئی وائرل ترتیب کا پتہ چلا۔ خطرے کی سطح: {level}
فوری اقدام: ڈبلیو ایچ او کو مطلع کریں اور نگرانی بڑھائیں۔"""


def generate_brief_streaming(sequence: str, scores: dict):
    context = _get_context(sequence)
    prompt = _build_prompt(sequence, scores, context)

    if GROQ_KEY:
        try:
            yield from _stream_groq(prompt)
            return
        except Exception as e:
            yield f"[Groq error: {e} — using fallback]\n\n"

    yield from _static_fallback(scores)


def generate_brief_sync(sequence: str, scores: dict) -> str:
    return ''.join(generate_brief_streaming(sequence, scores))