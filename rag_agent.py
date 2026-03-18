from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class MCPAgent:
    def __init__(self, index_path="faiss_structural_index"):
        # 1. Загружаем то, что подготовили в прошлый раз
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vector_db = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})

        # 2. Инициализируем LLM
        self.llm = ChatOllama(model="llama3.2", temperature=0) # Температура 0 для точности

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

# Пример использования
if __name__ == "__main__":
    agent = MCPAgent()
    q = "How do I implement a stdio transport in MCP?"

    print("--- ОТВЕТ БЕЗ RAG ---")
    print(agent.ask(q, use_rag=False))

    print("\n--- ОТВЕТ С RAG ---")
    print(agent.ask(q, use_rag=True))