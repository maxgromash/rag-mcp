import time
from rag_agent import AdvancedMCPAgent

def print_divider():
    print("\n" + "="*80 + "\n")

def evaluate_advanced():
    # Инициализируем нашего продвинутого агента
    print("🤖 Загрузка Advanced RAG Agent (это может занять время, качается реранкер)...")
    agent = AdvancedMCPAgent(index_path="faiss_structural_index")

    # Тестовые вопросы: один простой, один сложный, один "провокационный"
    test_queries = [
        "How to implement a tool in MCP?",
        "Compare stdio and HTTP transports in detail",
        "How to bake a cake using MCP protocol?" # Этого нет в базе, проверим фильтрацию
    ]

    for query in test_queries:
        print_divider()
        print(f"🔍 ИСХОДНЫЙ ВОПРОС: {query}")

        # --- РЕЖИМ 1: BASE RAG ---
        print("\n--- [ РЕЖИМ 1: BASE RAG ] ---")
        base_answer = agent.ask_advanced(query, use_rewrite=False, use_rerank=False)
        print(f"Ответ: {base_answer[:300]}...")

        # --- РЕЖИМ 2: ADVANCED RAG ---
        print("\n--- [ РЕЖИМ 2: ADVANCED RAG ] ---")

        # Выведем промежуточные этапы для наглядности
        rewritten = agent.rewrite_query(query)
        print(f"🔄 Переписанный запрос: {rewritten}")

        # Получаем чанки до и после реранкинга (вытащим логику наружу для теста)
        initial_docs = agent.vector_db.similarity_search(rewritten, k=10)

        # Реранкинг
        pairs = [[rewritten, doc.page_content] for doc in initial_docs]
        scores = agent.reranker.predict(pairs)
        scored_docs = sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)

        print("\n📊 Топ-чанки после реранкинга (Score > 0.3):")
        survived_count = 0
        for score, doc in scored_docs:
            status = "✅ ВЫЖИЛ" if score > agent.threshold else "❌ ОТСЕЯН"
            if score > agent.threshold: survived_count += 1
            print(f"- [{status}] Score: {score:.4f} | File: {doc.metadata.get('source_file', 'unknown')}")
            # Выведем кусочек контента выживших
            if score > agent.threshold:
                print(f"  Content: {doc.page_content[:100]}...")

        #advanced_answer = agent.ask_advanced(query, use_rewrite=True, use_rerank=True)
        advanced_answer = agent.ask_advanced(query, use_rewrite=False, use_rerank=True)
        print(f"\n✨ Финальный ответ: {advanced_answer}")

        if survived_count == 0:
            print("⚠️ ВНИМАНИЕ: Реранкер отфильтровал все результаты как нерелевантные!")

if __name__ == "__main__":
    evaluate_advanced()