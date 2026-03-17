import os
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIG ---»
DOCS_PATH = "./resources" # Путь к твоим .mdx файлам
MODEL_NAME = "nomic-embed-text"

# 1. ЗАГРУЗКА ДОКУМЕНТОВ
print("🚀 Загрузка документов...")
# Используем UnstructuredMarkdownLoader, так как mdx — это расширенный markdown
loader = DirectoryLoader(DOCS_PATH, glob="*.mdx", loader_cls=UnstructuredMarkdownLoader)
raw_documents = loader.load()

# 2. ИНИЦИАЛИЗАЦИЯ ЭМБЕДДИНГОВ (OLLAMA)
embeddings = OllamaEmbeddings(model=MODEL_NAME)

# --- СТРАТЕГИЯ 1: Recursive / Fixed Size ---
print("📦 Чанкинг: Стратегия 'Fixed Size'...")
fixed_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    add_start_index=True
)
docs_fixed = fixed_splitter.split_documents(raw_documents)

# Добавляем метаданные
for i, d in enumerate(docs_fixed):
    d.metadata["chunk_id"] = i
    d.metadata["strategy"] = "fixed"
    d.metadata["source_file"] = os.path.basename(d.metadata.get("source", "unknown"))

# --- СТРАТЕГИЯ 2: Structural (с защитой от переполнения) ---
print("🏗️ Чанкинг: Стратегия 'Structural'...")

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# Добавляем второй уровень нарезки для защиты от гигантских чанков
final_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

docs_structural = []

for full_doc in raw_documents:
    # 1. Сначала режем по заголовкам
    header_splits = header_splitter.split_text(full_doc.page_content)

    # 2. Каждый кусок от заголовка режем на части, если он больше 1000 символов
    # split_documents принимает список документов, так что прогоняем через него
    sub_splits = final_splitter.split_documents(header_splits)

    for i, seg in enumerate(sub_splits):
        seg.metadata["source_file"] = os.path.basename(full_doc.metadata.get("source", "unknown"))
        seg.metadata["chunk_id"] = f"struct_{i}"
        seg.metadata["strategy"] = "structural"
        docs_structural.append(seg)

print(f"Итог структурной нарезки: {len(docs_structural)} чанков")

# 3. СОЗДАНИЕ И СОХРАНЕНИЕ ИНДЕКСОВ
print("💾 Создание векторных баз FAISS...")
db_fixed = FAISS.from_documents(docs_fixed, embeddings)
db_fixed.save_local("faiss_fixed_index")

db_structural = FAISS.from_documents(docs_structural, embeddings)
db_structural.save_local("faiss_structural_index")

print("\n✅ Готово! Создано два индекса.")
print(f"Fixed: {len(docs_fixed)} чанков")
print(f"Structural: {len(docs_structural)} чанков")