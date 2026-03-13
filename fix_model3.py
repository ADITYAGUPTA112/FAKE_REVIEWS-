import json

with open('c:/fake reviews/main.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('hybrid = HybridFakeReviewDetector(' in line for line in source) or any('TRAIN & EVALUATE HYBRID MODEL' in line for line in source):
            # Check if it's already initialized
            if not any('DistilBertForSequenceClassification' in line for line in source):
                for j, line in enumerate(source):
                    if 'device = torch.device(' in line or 'model.to(device)' in line or 'hybrid = HybridFakeReviewDetector(' in line:
                        source.insert(j, 'from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast\n')
                        source.insert(j+1, 'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n')
                        source.insert(j+2, 'tokenizer = DistilBertTokenizerFast.from_pretrained("./models/distilbert")\n')
                        source.insert(j+3, 'model = DistilBertForSequenceClassification.from_pretrained("./models/distilbert").to(device)\n')
                        source.insert(j+4, 'model.eval()\n')
                        break

with open('c:/fake reviews/main.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
