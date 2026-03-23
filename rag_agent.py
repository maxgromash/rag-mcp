import os
import json
import re
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

class MCPAgent:
    def __init__(self, index_path="faiss_structural_index"):
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Индекс не найден: {index_path}")
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.llm = ChatOllama(model="llama3.2", temperature=0)
        self.vector_db = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)

class AdvancedMCPAgent(MCPAgent):
    def __init__(self, index_path="faiss_structural_index"):
        super().__init__(index_path)
        print("🤖 Loading Reranker (Cross-Encoder)...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.threshold = 0.2

    def rewrite_query(self, query: str, state: str = ""):
        system = (
            "You are an MCP Technical Architect. Rewrite the user's request into a search query "
            "for the Model Context Protocol documentation. Use terms like: 'server implementation', "
            "'SDK handlers', 'transport configuration'. Context: {state}. Return ONLY the query string."
        )
        return self.llm.invoke(f"{system}\n\nUser: {query}").content

class ConversationalMCPAgent(AdvancedMCPAgent):
    def __init__(self, index_path="faiss_structural_index"):
        super().__init__(index_path)
        self.chat_history = []
        self.task_state = {
            "goal": "Not started",
            "constraints": [],
            "fixed_terms": {}
        }

    def update_task_state(self, user_input):
        """Интеллектуальное обновление состояния задачи"""
        state_prompt = f"""
        Current Task State: {json.dumps(self.task_state)}
        New Message: {user_input}
        
        TASK: Update the JSON state. 
        Rules:
        1. If user changes a decision (e.g. choice of transport), OVERWRITE the old value in fixed_terms.
        2. If user sets a security limit (e.g. read-only), add it to 'constraints' and REMOVE any conflicting permissions.
        3. Keep 'goal' concise.
        
        Return ONLY valid JSON.
        """
        try:
            raw_res = self.llm.invoke(state_prompt).content
            match = re.search(r'\{.*\}', raw_res, re.DOTALL)
            if match:
                self.task_state = json.loads(match.group())
        except Exception as e:
            print(f"⚠️ State Update Error: {e}")

    def ask_chat(self, query: str):
        self.update_task_state(query)

        # Переписываем запрос с учетом текущего State
        search_query = self.rewrite_query(query, state=str(self.task_state))

        # RAG Search
        docs = self.vector_db.similarity_search(search_query, k=8)
        pairs = [[search_query, d.page_content] for d in docs]
        scores = self.reranker.predict(pairs)
        scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        relevant_docs = [d for s, d in scored_docs if s > self.threshold]

        # Генерация ответа
        answer = self.generate_final_answer(query, relevant_docs[:3])
        self.chat_history.append((query, answer))
        return answer

    def generate_final_answer(self, question, docs):
        # Подготовка контекста и истории
        context = "\n\n".join([f"[Source: {d.metadata.get('source_file')}] {d.page_content}" for d in docs]) if docs else "No docs."
        history = "\n".join([f"U: {u}\nA: {a[:100]}..." for u, a in self.chat_history[-3:]])

        system_prompt = f"""
        YOU ARE A SENIOR MCP DEVELOPER. 
        Identity: You use @modelcontextprotocol/sdk, NOT Express/REST.
        Language: Russian.
        
        STRICT RULES:
        1. CURRENT PROJECT STATE: {json.dumps(self.task_state)}
        2. If constraints say 'Read-only', NEVER provide code for DELETE, POST, or UPDATE.
        3. If fixed_terms specify 'stdio', use 'StdioServerTransport'. Never mention TCP/HTTP.
        4. ALWAYS use direct "Quotes" from Context and cite (Source: file.mdx).
        5. If Context is missing, admit it but answer using Task State rules.
        6. DO NOT repeat your memory block in the answer. Just answer the user.
        """

        full_input = f"{system_prompt}\n\nHistory:\n{history}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        return self.llm.invoke(full_input).content