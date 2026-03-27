import os
import json
import re
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

class ConversationalMCPAgent:
    def __init__(self, index_path="faiss_structural_index"):
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found: {index_path}")

        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.llm = ChatOllama(
            model="llama3.2",
            temperature=0,
            num_ctx=4096,        # Ограничиваем окно контекста, чтобы не перегружать RAM
            repeat_penalty=1.2,  # Оптимизация против зацикливания (которое мы видели в логах)
            num_predict=512      # Ограничиваем длину ответа для скорости
        )
        self.vector_db = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        self.chat_history = []
        self.task_state = {"goal": "Not defined", "constraints": [], "fixed_terms": {}}

    def update_task_state(self, user_input):
        """Обновление памяти задачи с защитой от ошибок формата"""
        prompt = f"""
        Update the MCP Project State JSON based on the user message.
        Current State: {json.dumps(self.task_state)}
        User Message: {user_input}
        
        Rules:
        - Return ONLY a valid JSON block.
        - If 'stdio' is chosen, put it in fixed_terms.transport.
        - If 'read-only' is set, add to constraints.
        """
        try:
            res = self.llm.invoke(prompt).content
            # Ищем JSON в блоке кода или просто в тексте
            json_match = re.search(r'(\{.*\})', res, re.DOTALL)
            if json_match:
                new_state = json.loads(json_match.group(1))
                self.task_state.update(new_state)
        except:
            pass

    def ask_chat(self, query: str):
        self.update_task_state(query)

        # Поиск (RAG)
        docs = self.vector_db.similarity_search(query, k=8)
        scores = self.reranker.predict([[query, d.page_content] for d in docs])
        relevant_docs = [d for s, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True) if s > 0.1]

        # Промпт для генерации
        context = "\n\n".join([f"[Source: {d.metadata.get('source_file')}] {d.page_content}" for d in relevant_docs[:3]])
        history = "\n".join([f"U: {u}\nA: {a[:100]}..." for u, a in self.chat_history[-2:]])

        system_prompt = f"""
        You are a Senior MCP Developer. Use @modelcontextprotocol/sdk. 
        Project State: {json.dumps(self.task_state)}
        History: {history}
        Context: {context}
        
        STRICT RULES:
        1. Answer in Russian.
        2. If 'read-only' constraint exists, NEVER show code with DELETE/POST/UPDATE.
        3. Use direct quotes from context if available.
        4. Do not repeat the internal state JSON in your response.
        """

        answer = self.llm.invoke(f"{system_prompt}\n\nUser: {query}").content
        self.chat_history.append((query, answer))
        return answer
class MCPAgent:
    def __init__(self, index_path="faiss_structural_index"):
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Индекс не найден: {index_path}")
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.llm = ChatOllama(
            model="llama3.2",
            temperature=0,
            num_ctx=4096,        # Ограничиваем окно контекста, чтобы не перегружать RAM
            repeat_penalty=1.2,  # Оптимизация против зацикливания (которое мы видели в логах)
            num_predict=512      # Ограничиваем длину ответа для скорости
        )
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