from transformers import MarianMTModel, MarianTokenizer
import sentencepiece  # Ensure sentencepiece is imported
from tqdm import tqdm

# Load the MarianMT model and tokenizer
model_name = "Helsinki-NLP/opus-mt-fr-en"  # French to English model
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Read the contents of the file
with open('./translation/RAG_data_fr.txt', 'r', encoding='utf-8') as file:
    src_text = file.readlines()

# Prepare the text for translation by adding the language code
src_text = [">>en<< " + line.strip() for line in src_text]

# Use tqdm to show progress
translated_text = []
for i in tqdm(range(0, len(src_text), 10), desc="Translating"):
    batch = src_text[i:i+10]
    tokens = tokenizer(batch, return_tensors="pt", padding=True)
    translated = model.generate(**tokens)
    translated_text.extend([tokenizer.decode(t, skip_special_tokens=True) for t in translated])

# Save the translated text to a new file
with open('./translation/RAG_data_en.txt', 'w', encoding='utf-8') as file:
    file.write("\n".join(translated_text))

print("Translation completed and saved to RAG_data_translated.txt")
