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
            # Маленькая, но мощная модель для реранкинга (всего ~80МБ)
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            self.threshold = 0.3  # Порог отсечения: всё что ниже — "шум"

    def rewrite_query(self, query: str):
        # Добавляем жесткую установку контекста
        system_context = (
            "You are an expert in Model Context Protocol (MCP) by Anthropic. "
            "Your task is to expand the user's search query into a technical description. "
            "DO NOT assume MCP means Microsoft. It is Model Context Protocol. "
            "Return ONLY the expanded query, no conversational filler."
        )

        prompt = f"{system_context}\n\nQuery: {query}\n\nExpanded technical query:"
        return self.llm.invoke(prompt).content

    def ask_advanced(self, query: str, use_rewrite=True, use_rerank=True):
        # 1. Query Rewrite
        search_query = self.rewrite_query(query) if use_rewrite else query

        # 2. Retrieval: Берем Top-10 (с запасом для фильтрации)
        initial_docs = self.vector_db.similarity_search(search_query, k=10)

        if not use_rerank:
            # Если реранкинг выключен, просто берем первые 3
            context = "\n".join([d.page_content for d in initial_docs[:3]])
        else:
            # 3. Reranking: Скармливаем пары (запрос, чанк) в модель
            pairs = [[search_query, doc.page_content] for doc in initial_docs]
            scores = self.reranker.predict(pairs)

            # Сортируем по скору
            scored_docs = sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)

            # 4. Фильтрация по порогу
            filtered_docs = [doc for score, doc in scored_docs if score > self.threshold]

            # Берем финальный Top-3 после фильтрации
            final_docs = filtered_docs[:3]
            context = "\n".join([d.page_content for d in final_docs])

        # 5. Генерация ответа (стандартный промпт)
        return self.generate_answer(query, context)

    def generate_answer(self, question, context):
        if not context:
            return "Извините, в базе данных нет релевантной информации для ответа на этот вопрос."

        template = f"Use the context to answer. Context: {context}\nQuestion: {question}\nAnswer:"
        return self.llm.invoke(template).content


# Пример использования
if __name__ == "__main__":
    agent = MCPAgent()
    q = "How do I implement a stdio transport in MCP?"

    print("--- ОТВЕТ БЕЗ RAG ---")
    print(agent.ask(q, use_rag=False))

    print("\n--- ОТВЕТ С RAG ---")
    print(agent.ask(q, use_rag=True))
