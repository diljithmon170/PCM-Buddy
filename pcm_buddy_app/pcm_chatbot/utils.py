import os
import torch
import numpy as np
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# -------------------------
# Device Setup
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32
print(f"🚀 Using device: {device}, dtype: {torch_dtype}")

# -------------------------
# Load Sentence Embedder
# -------------------------
print("🔁 Loading Sentence Transformer...")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# -------------------------
# BitsAndBytes 4-bit Config
# -------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# -------------------------
# Load Language Model (Gemma-2B)
# -------------------------
print("🔁 Loading Gemma-2B Model...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    quantization_config=bnb_config,
    device_map="auto"
)

print("✅ Model loaded successfully!")

# -------------------------
# PDF to Text Chunking
# -------------------------
def load_pdf_chunks(pdf_path, chunk_size=300):
    try:
        doc = fitz.open(pdf_path)
        full_text = " ".join(page.get_text() for page in doc)
        doc.close()
    except Exception as e:
        print(f"[ERROR] Failed to read {pdf_path}: {e}")
        return []

    return [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]

# -------------------------
# Build Embedding Index
# -------------------------
def build_index(chunks):
    embeddings = embedder.encode(chunks, convert_to_tensor=True, device=device)
    return chunks, embeddings.cpu().numpy()

# -------------------------
# Context Retrieval
# -------------------------
def retrieve_context(query, chunks, embeddings, top_k=3):
    if not embeddings.any():
        print("⚠️ No embeddings found. Skipping context retrieval.")
        return []

    query_vec = embedder.encode([query], convert_to_numpy=True, device=device)

    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)

    sims = cosine_similarity(query_vec, embeddings)[0]
    top_indices = np.argsort(sims)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# -------------------------
# Generate Answer
# -------------------------
def generate_answer(query, context, subject, marks):
    context_text = "\n".join(context)

    word_count_map = {
        1: 20, 2: 40, 3: 60, 4: 80, 5: 100,
    }
    token_limit_map = {
        1: 80, 2: 120, 3: 160, 4: 200, 5: 300,
    }

    desired_words = word_count_map.get(marks, 50)
    max_tokens = token_limit_map.get(marks, 150)

    prompt = f"""You are a helpful CBSE Class 10 {subject} tutor.
Use the NCERT context to answer the following question simply and clearly.
Keep the answer around {desired_words} words for a {marks}-mark question.

Context:
{context_text}

Question: {query}

Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.split("Answer:")[-1].strip()

# -------------------------
# Main Chat Loop
# -------------------------
print("📥 Loading and indexing NCERT PDFs...")

science_chunks = load_pdf_chunks(r"C:\Users\dilji\Documents\project\chatbot\pcm_buddy\pcm_buddy_app\pcm_chatbot\ncert_pdfs\science.pdf")
maths_chunks = load_pdf_chunks(r"C:\Users\dilji\Documents\project\chatbot\pcm_buddy\pcm_buddy_app\pcm_chatbot\ncert_pdfs\maths.pdf")

if not science_chunks or not maths_chunks:
    print("❌ Failed to load NCERT PDFs. Please check the file paths.")
    sci_texts, sci_embeds, math_texts, math_embeds = [], [], [], []
else:
    sci_texts, sci_embeds = build_index(science_chunks)
    math_texts, math_embeds = build_index(maths_chunks)
print("✅ Indexing complete. You can now ask questions!\n")

if __name__ == "__main__":
    while True:
        subject = input("Enter subject (science/maths or 'exit'): ").lower().strip()
        if subject == "exit":
            print("👋 Goodbye!")
            break
        if subject not in ["science", "maths"]:
            print("❌ Invalid subject. Please choose 'science' or 'maths'.")
            continue

        question = input("Enter your question: ").strip()
        if not question:
            print("❌ Question cannot be empty.")
            continue

        try:
            marks = int(input("How many marks should the answer be for? (1-5): ").strip())
            if marks < 1 or marks > 5:
                raise ValueError
        except ValueError:
            print("❌ Please enter a number between 1 and 5.")
            continue

        context = retrieve_context(
            question,
            sci_texts if subject == "science" else math_texts,
            sci_embeds if subject == "science" else math_embeds
        )
        answer = generate_answer(question, context, subject, marks)
        print("\n📘 Answer:\n", answer)

# -------------------------
# Get Answer Function
# -------------------------
def get_answer(query, subject, marks=2):
    # ...existing code to select context...
    if subject.lower() in ["physics", "chemistry", "science"]:
        context = retrieve_context(query, sci_texts, sci_embeds)
        subj = "Science"
    elif subject.lower() == "maths":
        context = retrieve_context(query, math_texts, math_embeds)
        subj = "Maths"
    else:
        return "Sorry, I can only answer Physics, Chemistry, or Maths questions."
    return generate_answer(query, context, subj, marks)
