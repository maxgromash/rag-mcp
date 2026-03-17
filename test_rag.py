from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
query = "What is the difference between stdio and HTTP transports?"

# Загружаем индексы
db_f = FAISS.load_local("faiss_fixed_index", embeddings, allow_dangerous_deserialization=True)
db_s = FAISS.load_local("faiss_structural_index", embeddings, allow_dangerous_deserialization=True)

print(f"\n🔍 Вопрос: {query}")

# Используем similarity_search_with_score
print("\n=== РЕЗУЛЬТАТ FIXED SIZE ===")
res_f, score_f = db_f.similarity_search_with_score(query, k=1)[0]
print(f"Score (L2 Distance): {score_f:.4f}") # Чем меньше, тем лучше
print(f"File: {res_f.metadata['source_file']}")
print(f"Content: {res_f.page_content}...")

print("\n=== РЕЗУЛЬТАТ STRUCTURAL ===")
res_s, score_s = db_s.similarity_search_with_score(query, k=1)[0]
print(f"Score (L2 Distance): {score_s:.4f}")
print(f"File: {res_s.metadata['source_file']}")
print(f"Content: {res_s.page_content}...")