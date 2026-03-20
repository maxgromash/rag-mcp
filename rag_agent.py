from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

class MCPAgent:
    def __init__(self, index_path="faiss_structural_index"):
        # 1. Загружаем то, что подготовили в прошлый раз
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vector_db = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})

        # 2. Инициализируем LLM
        self.llm = ChatOllama(model="llama3.2", temperature=0)  # Температура 0 для точности

    def ask(self, query: str, use_rag: bool = True):
        if not use_rag:
            # Режим БЕЗ RAG: просто отправляем вопрос модели
            return self.llm.invoke(query).content

        # Режим С RAG:
        # Создаем шаблон промпта
        template = """You are an expert on Model Context Protocol (MCP). 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know.
        
        Context: {context}
        
        Question: {question}
        Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        # Цепочка: поиск -> сборка контекста -> промпт -> модель -> чистый текст
        chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )

        return chain.invoke(query)

class AdvancedMCPAgent(MCPAgent):
    def __init__(self, index_path="faiss_structural_index"):
        super().__init__(index_path)
        print("🤖 Загрузка реранкера...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.threshold = 0.3 # Порог "уверенности"

    def rewrite_query(self, query: str):
        system_context = (
            "You are an expert in Model Context Protocol (MCP) by Anthropic. "
            "Expand the query into technical search terms. DO NOT assume MCP means Microsoft. "
            "Return ONLY the expanded query text."
        )
        return self.llm.invoke(f"{system_context}\n\nQuery: {query}").content

    def ask_advanced(self, query: str, use_rewrite=True, use_rerank=True):
        # 1. Query Rewrite
        search_query = self.rewrite_query(query) if use_rewrite else query

        # 2. Поиск (Retriever)
        initial_docs = self.vector_db.similarity_search(search_query, k=10)

        # 3. Реранкинг и Фильтрация
        if use_rerank:
            pairs = [[search_query, doc.page_content] for doc in initial_docs]
            scores = self.reranker.predict(pairs)
            scored_docs = sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)
            relevant_docs = [doc for score, doc in scored_docs if score > self.threshold]
        else:
            relevant_docs = initial_docs[:3]

        # Усиление: Режим "Не знаю"
        if not relevant_docs:
            return ("К сожалению, в моей базе знаний нет достаточно надежной информации по этому вопросу. "
                    "Пожалуйста, уточните запрос (релевантность ниже порога).")

        # 4. Генерация с источниками и цитатами
        return self.generate_grounded_answer(query, relevant_docs[:3])

    def generate_grounded_answer(self, question, docs):
        context_blocks = []
        for doc in docs:
            source = doc.metadata.get('source_file', 'unknown')
            cid = doc.metadata.get('chunk_id', 'unknown')
            context_blocks.append(f"--- SOURCE: {source} | ID: {cid} ---\n{doc.page_content}")

        context_text = "\n\n".join(context_blocks)

        system_prompt = (
            "You are a strict technical assistant. Use ONLY the provided context to answer. "
            "For every key claim, you MUST provide a direct quote from the context in double quotes. "
            "Example: 'The protocol uses JSON-RPC (Source: architecture.mdx)'. "
            "At the end of your answer, provide a 'SOURCES' section listing the files used."
        )

        full_prompt = f"{system_prompt}\n\nContext:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
        return self.llm.invoke(full_prompt).content

# Пример использования
if __name__ == "__main__":
    agent = MCPAgent()
    q = "How do I implement a stdio transport in MCP?"

    print("--- ОТВЕТ БЕЗ RAG ---")
    print(agent.ask(q, use_rag=False))

    print("\n--- ОТВЕТ С RAG ---")
    print(agent.ask(q, use_rag=True))
