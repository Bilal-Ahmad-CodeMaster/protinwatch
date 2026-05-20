# scheduler.py
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

SCHEDULES = {
    '30min': '*/30 * * * *',
    '6h':    '0 */6 * * *',
    '12h':   '0 */12 * * *',
    '1day':  '0 8 * * *',
    '1week': '0 8 * * 1'
}

scheduler = AsyncIOScheduler()

async def run_full_pipeline():
    """Called on schedule — fetches new sequences and analyzes them"""
    print("⏰ Scheduled pipeline run started")
    try:
        from ncbi_fetcher import fetch_sequences
        from main import analyze  # <--- IMPORT MAIN FUNCTION
        
        # Since it runs every 30 mins, we only need to look 1 hour back
        sequences = fetch_sequences(hours_back=1)
        print(f"✅ Fetched {len(sequences)} sequences. Starting AI analysis...")

        for seq_data in sequences:
            # Run each fetched sequence through your AI pipeline
            await analyze({"sequence": seq_data['sequence']})

        print("✅ All new sequences analyzed and saved to history!")
    except Exception as e:
        print(f"❌ Scheduled run failed: {e}")

def update_schedule(label: str) -> dict:
    cron = SCHEDULES.get(label, SCHEDULES['6h'])
    scheduler.remove_all_jobs()
    scheduler.add_job(
        run_full_pipeline,
        CronTrigger.from_crontab(cron),
        id='main',
        misfire_grace_time=3600  # ⚠️ YEH LINE BOHOT ZAROORI HAI LIVE SERVER KE LIYE
    )
    return {'updated': True, 'schedule': label, 'cron': cron}