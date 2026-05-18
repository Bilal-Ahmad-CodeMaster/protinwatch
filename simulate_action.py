import uuid
import datetime


def simulate_alert_dispatch(sequence_id: str, threat_index: float, virus_name: str) -> dict:
    alert_id = f'PW-{datetime.datetime.now().year}-{str(uuid.uuid4())[:6].upper()}'
    ts = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')

    return {
        'alert_id': alert_id,
        'status': 'DISPATCHED',
        'timestamp': ts,
        'threat_index': threat_index,
        'virus_name': virus_name,
        'before_state': 'No active biological crisis alerts. System monitoring.',
        'after_state': f'ACTIVE CRISIS — Alert {alert_id} dispatched. Response initiated.',
        'actions_taken': [
            f'WHO Surveillance Team notified — Ref #{alert_id}',
            f'Travel advisory flag raised for origin region',
            f'Sequence {sequence_id} escalated to high-priority watchlist',
        ],
        'agent_trace': [
            {
                'step': 1,
                'agent': 'DetectionAgent',
                'action': f'Threat Index {threat_index}/100 — threshold exceeded',
                'timestamp': ts
            },
            {
                'step': 2,
                'agent': 'VerificationAgent',
                'action': 'AlphaFold structural confirmation checked',
                'timestamp': ts
            },
            {
                'step': 3,
                'agent': 'ResponseAgent',
                'action': f'Alert {alert_id} created and dispatched',
                'timestamp': ts
            },
            {
                'step': 4,
                'agent': 'NotificationAgent',
                'action': 'WHO + stakeholders notified',
                'timestamp': ts
            },
        ]
    }