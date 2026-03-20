from rag_agent import AdvancedMCPAgent

def run_citation_test():
    print("🤖 Инициализация Advanced RAG Agent...")
    # Убедись, что путь совпадает с твоей папкой индекса
    agent = AdvancedMCPAgent(index_path="faiss_structural_index")

    # Твой набор из 10 контрольных вопросов
    test_queries = [
        "What is the core architecture of the Model Context Protocol?",
        "Explain the differences between stdio and HTTP transports.",
        "How can a server define and expose 'Resources' to an LLM?",
        "What is the specific role of 'Tools' in MCP and how to call them?",
        "What are 'Prompt templates' and how do they work?",
        "Explain the 'Sampling' feature: how can a server request a completion?",
        "How does MCP use JSON-RPC 2.0 for communication?",
        "What are the main security considerations for MCP implementations?",
        "How to use the MCP Inspector for debugging servers?",
        "How to bake a sourdough bread using MCP protocol?" # Тест на "не знаю"
    ]

    print(f"🚀 Запуск оценки на {len(test_queries)} вопросах...")
    print("="*80)

    for i, q in enumerate(test_queries, 1):
        print(f"\n❓ [Вопрос {i}/10]: {q}")

        # Замеряем время ответа
        import time
        start = time.time()
        answer = agent.ask_advanced(q)
        duration = time.time() - start

        print(f"💡 ОТВЕТ ({duration:.2f}s):\n{answer}")

        # --- БЛОК АВТО-ВАЛИДАЦИИ ---
        has_quotes = '"' in answer or '«' in answer
        has_sources = "SOURCES" in answer.upper() or "SOURCE:" in answer.upper()

        # Проверка на режим "Не знаю" для 10-го вопроса
        is_oob_question = "bread" in q.lower() or "cake" in q.lower()
        idk_triggered = "базе знаний нет" in answer.lower() or "not enough information" in answer.lower()

        print("\n📝 ЧЕК-ЛИСТ:")
        print(f"  - Прямые цитаты: {'✅ ПРИСУТСТВУЮТ' if has_quotes else '❌ ОТСУТСТВУЮТ'}")
        print(f"  - Ссылки на файлы: {'✅ ПРИСУТСТВУЮТ' if has_sources else '❌ ОТСУТСТВУЮТ'}")

        if is_oob_question:
            print(f"  - Режим 'Не знаю': {'✅ СРАБОТАЛ' if idk_triggered else '❌ ОШИБКА (Галлюцинация!)'}")

        print("-" * 60)

if __name__ == "__main__":
    run_citation_test()