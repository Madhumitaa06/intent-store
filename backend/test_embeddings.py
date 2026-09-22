from sentence_transformers import SentenceTransformer

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded successfully!")

text = "This is a test document about artificial intelligence."

embedding = model.encode(text)

print("Embedding generated successfully!")
print("Embedding dimensions:", len(embedding))