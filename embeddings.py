from transformers import BertModel, BertTokenizer
import json
import torch


# Download pre-trained BERT model and tokenizer
model = BertModel.from_pretrained('bert-large-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

# Load the data from the JSON file
with open('roget_words.json', 'r') as f:
    data = json.load(f)

embeddings_dict = {}

for class_name, sections in data.items():
    class_embeddings = {}
    for section_name, word_lists in sections.items():
        section_embeddings = []
        for word_list in word_lists:
            preprocessed_word_list = [word.lower() for word in word_list]
            tokens = tokenizer(preprocessed_word_list, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**tokens)
                # Ensure the [CLS] token embedding is flattened and has consistent dimensions
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()  # Flatten the embedding
                section_embeddings.append(cls_embedding)
        class_embeddings[section_name] = section_embeddings
    embeddings_dict[class_name] = class_embeddings


print("Data processed and embeddings generated successfully!")

# Save embeddings to a JSON file
with open('embeddings_cls.json', 'w') as f:
    json.dump(embeddings_dict, f)

print("Embeddings saved successfully!")
