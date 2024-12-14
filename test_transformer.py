from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
embeddings = model.encode(["Test sentence", "Another test sentence"])
print(embeddings)

