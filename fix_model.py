import json

with open('c:/fake reviews/main.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if '# Get BERT probabilities for test set' in line:
                source.insert(i, 'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n')
                source.insert(i+1, 'from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast\n')
                source.insert(i+2, 'tokenizer = DistilBertTokenizerFast.from_pretrained("./models/distilbert")\n')
                source.insert(i+3, 'model = DistilBertForSequenceClassification.from_pretrained("./models/distilbert").to(device)\n')
                found = True
                break
        if found:
            break

with open('c:/fake reviews/main.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
