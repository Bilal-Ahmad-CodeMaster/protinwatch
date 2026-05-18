
# from transformers import EsmTokenizer, EsmForSequenceClassification
# import torch

# # Loaded ONCE at server startup
# tokenizer = EsmTokenizer.from_pretrained('./model')
# model = EsmForSequenceClassification.from_pretrained('./model')
# model.eval()

# # torch.compile() only works on Linux/Mac — skip on Windows
# import platform
# if platform.system() != 'Windows':
#     model = torch.compile(model)


# def danger_score(sequence: str) -> dict:
#     try:
#         if len(sequence) < 50:
#             return {'danger_score': 0.0, 'error': 'Sequence too short'}

#         inputs = tokenizer(
#             sequence,
#             return_tensors='pt',
#             truncation=True,
#             max_length=512
#         )
#         with torch.no_grad():
#             logits = model(**inputs).logits

#         probs = torch.softmax(logits, dim=1)[0]
#         return {
#             'danger_score': round(probs[1].item() * 100, 1),
#             'safe_prob': round(probs[0].item() * 100, 1)
#         }
#     except Exception as e:
#         return {'danger_score': 50.0, 'error': str(e)}

# esm2_scorer.py
import os
import torch
import platform
from transformers import EsmTokenizer, EsmForSequenceClassification

# Live cloud deployment aur local dono ke liye Hugging Face repository ID
MODEL_REPO = "arifhusnain/ProteinWatch"

print(f"🔄 Loading ESM-2 Model from Hugging Face: {MODEL_REPO}...")

try:
    # Yeh automatic Hugging Face se download karega aur cache mein save rakhega
    tokenizer = EsmTokenizer.from_pretrained(MODEL_REPO)
    model = EsmForSequenceClassification.from_pretrained(MODEL_REPO)
    model.eval()
    
    # Windows standard engine verification check (Linux/Render par pipeline ko fast karne ke liye)
    if platform.system() != 'Windows':
        model = torch.compile(model)
        
    print("✅ ESM-2 Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model from Hugging Face: {e}")
    raise e

def danger_score(sequence: str) -> dict:
    """
    Computes the danger score for a given viral sequence using the fine-tuned ESM-2 model.
    """
    try:
        inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            # Softmax apply karke danger probability nikalna
            probabilities = torch.softmax(logits, dim=1)
            # Assuming class 1 is the 'danger/threat' class
            score = float(probabilities[0][1].item()) * 100
            
        return {"danger_score": round(score, 1)}
    except Exception as e:
        print(f"Error during ESM-2 inference: {e}")
        return {"danger_score": 50.0, "error": str(e)}
