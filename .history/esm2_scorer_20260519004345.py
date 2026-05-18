
from transformers import EsmTokenizer, EsmForSequenceClassification
import torch

# Loaded ONCE at server startup
tokenizer = EsmTokenizer.from_pretrained('./model')
model = EsmForSequenceClassification.from_pretrained('./model')
model.eval()

# torch.compile() only works on Linux/Mac — skip on Windows
import platform
if platform.system() != 'Windows':
    model = torch.compile(model)


def danger_score(sequence: str) -> dict:
    try:
        if len(sequence) < 50:
            return {'danger_score': 0.0, 'error': 'Sequence too short'}

        inputs = tokenizer(
            sequence,
            return_tensors='pt',
            truncation=True,
            max_length=512
        )
        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.softmax(logits, dim=1)[0]
        return {
            'danger_score': round(probs[1].item() * 100, 1),
            'safe_prob': round(probs[0].item() * 100, 1)
        }
    except Exception as e:
        return {'danger_score': 50.0, 'error': str(e)}