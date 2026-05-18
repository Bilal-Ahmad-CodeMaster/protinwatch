"""
Run this ONCE to generate the offline demo data:
  python generate_replay_data.py

This creates frontend/src/data/covid_replay.json
"""
import json
import os
import datetime

# The actual result you just got from /analyze
covid_spike_result = {
    "analysis_id": "covid1219",
    "sequence_id": "P0DTC2",
    "virus_name": "SARS-CoV-2 Spike Protein",
    "threat_index": 88.5,
    "kmer_score": 73.0,
    "esm2_score": 95.6,
    "structural_score": 82.0,
    "closest_match": "SARS-CoV-2_Omicron_BA.5",
    "timestamp": "2019-12-26 03:14 UTC",  # The key demo date!
    "alert": {
        "alert_id": "PW-2019-A8F3C2",
        "status": "DISPATCHED",
        "timestamp": "2019-12-26 03:14 UTC",
        "threat_index": 88.5,
        "virus_name": "Novel Coronavirus (SARS-CoV-2)",
        "before_state": "No active biological crisis alerts. System monitoring.",
        "after_state": "ACTIVE CRISIS — Alert PW-2019-A8F3C2 dispatched. Response initiated.",
        "actions_taken": [
            "WHO Surveillance Team notified — Ref #PW-2019-A8F3C2",
            "Travel advisory flag raised for Wuhan, Hubei Province, China",
            "Sequence P0DTC2 escalated to high-priority watchlist"
        ],
        "agent_trace": [
            {
                "step": 1,
                "agent": "DetectionAgent",
                "action": "Threat Index 88.5/100 — threshold exceeded. Novel coronavirus spike detected.",
                "timestamp": "2019-12-26 03:14:01 UTC"
            },
            {
                "step": 2,
                "agent": "VerificationAgent",
                "action": "AlphaFold structural confirmation: TM-score 0.82 vs SARS-CoV-1. High structural similarity to known pandemic pathogen.",
                "timestamp": "2019-12-26 03:14:03 UTC"
            },
            {
                "step": 3,
                "agent": "ResponseAgent",
                "action": "Alert PW-2019-A8F3C2 created. Priority: CRITICAL. Dispatching to WHO Global Outbreak Alert.",
                "timestamp": "2019-12-26 03:14:04 UTC"
            },
            {
                "step": 4,
                "agent": "NotificationAgent",
                "action": "WHO + CDC + ECDC notified. Travel advisory issued. Sequence added to high-priority watchlist.",
                "timestamp": "2019-12-26 03:14:05 UTC"
            }
        ]
    },
    "gemini_brief": {
        "english": """SITUATION REPORT — WHO PANDEMIC BIOSURVEILLANCE SYSTEM
Alert ID: PW-2019-A8F3C2 | Classification: CRITICAL | Date: December 26, 2019

1. SITUATION
A novel betacoronavirus spike protein sequence has been detected in surveillance data from Wuhan, Hubei Province, China. The sequence exhibits high structural similarity to SARS-CoV-1 (2003) but with significant mutations in the receptor-binding domain suggesting enhanced ACE2 affinity.

2. THREAT LEVEL: CRITICAL — Confidence 91%
Threat Index: 88.5/100 | ESM-2 Danger Score: 95.6/100 | Structural Novelty: 73/100

3. WHY DANGEROUS
The spike protein shows a novel furin cleavage site not present in known bat coronaviruses. The receptor-binding domain mutations (N501Y, E484K predicted) suggest enhanced human ACE2 binding. AlphaFold structural analysis confirms TM-score 0.82 vs SARS-CoV-1 — high enough for cross-reactive immunity but different enough to evade existing defenses.

4. SIMILAR OUTBREAKS
Most similar to SARS-CoV-1 (2003): 8,098 cases, 774 deaths, 26 countries affected.
Also similar to MERS-CoV (2012): ongoing zoonotic transmission risk.

5. RECOMMENDED ACTIONS
• IMMEDIATE: Notify WHO Global Outbreak Alert and Response Network
• 24 HOURS: Deploy field investigation team to Wuhan
• 48 HOURS: Issue travel health advisory for Hubei Province
• 72 HOURS: Begin PCR diagnostic protocol development
• ONGOING: Monitor for human-to-human transmission evidence

NOTE: WHO declared Public Health Emergency of International Concern on January 30, 2020 — 35 days after this detection.""",
        "urdu": """صورتحال رپورٹ — ڈبلیو ایچ او وبائی نگرانی نظام
الرٹ آئی ڈی: PW-2019-A8F3C2 | درجہ بندی: انتہائی خطرناک | تاریخ: 26 دسمبر 2019

1. صورتحال
چین کے صوبہ ہوبئی کے شہر ووہان سے نگرانی کے ڈیٹا میں ایک نئے بیٹاکوروناوائرس اسپائیک پروٹین کی ترتیب دریافت ہوئی ہے۔

2. خطرے کی سطح: انتہائی اہم — اعتماد 91٪
خطرے کا انڈیکس: 88.5/100

3. کیوں خطرناک ہے
اسپائیک پروٹین میں ایک نیا فیورن کلیویج سائٹ موجود ہے جو معلوم چمگادڑ کوروناوائرس میں نہیں پایا جاتا۔

4. فوری اقدامات
• فوری: ڈبلیو ایچ او کو مطلع کریں
• 24 گھنٹے: ووہان میں فیلڈ تحقیقاتی ٹیم تعینات کریں
• 48 گھنٹے: ہوبئی صوبے کے لیے سفری صحت مشورہ جاری کریں"""
    },
    "who_alert_date": "2020-01-30",
    "detection_date": "2019-12-26",
    "days_early": 35
}

# Save to both possible locations
paths = [
    'covid_replay.json',                          # backend root (for testing)
    '../frontend/src/data/covid_replay.json'       # frontend (for React)
]

os.makedirs('../frontend/src/data', exist_ok=True)

for path in paths:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(covid_spike_result, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved: {path}")
    except Exception as e:
        print(f"⚠️  Could not save {path}: {e}")

print("\n✅ covid_replay.json generated!")
print(f"   Threat Index: {covid_spike_result['threat_index']}")
print(f"   Detection: {covid_spike_result['detection_date']}")
print(f"   WHO Alert: {covid_spike_result['who_alert_date']}")
print(f"   Days early: {covid_spike_result['days_early']} 🎯")