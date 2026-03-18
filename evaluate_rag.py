import os
import time
from rag_agent import MCPAgent # Импортируем твой класс

# 1. Твой набор контрольных вопросов (Ground Truth)
TEST_SUITE = [
    {"question": "Что такое Model Context Protocol (MCP)?", "expected": "Открытый стандарт для связи ИИ с данными и инструментами.", "source": "architecture.mdx"},
    {"question": "Назови три основных примитива в MCP.", "expected": "Resources, Prompts, Tools.", "source": "architecture.mdx"},
    {"question": "Как работает транспорт через stdio?", "expected": "Через стандартные потоки ввода-вывода (stdin/stdout).", "source": "transports.mdx"},
    {"question": "Для чего используется MCP Inspector?", "expected": "Для тестирования и отладки серверов.", "source": "inspector.mdx"},
    {"question": "Что представляют собой 'Ресурсы' в MCP?", "expected": "Данные только для чтения, доступные модели.", "source": "resources.mdx"},
    {"question": "В чем разница между инструментом (Tool) и ресурсом (Resource)?", "expected": "Инструменты активны (выполняются), ресурсы пассивны (читаются).", "source": "concepts"},
    {"question": "Что позволяет делать функция 'Sampling'?", "expected": "Позволяет серверу запрашивать генерацию от LLM.", "source": "sampling.mdx"},
    {"question": "Какой формат сообщений используется под капотом MCP?", "expected": "JSON-RPC 2.0.", "source": "architecture.mdx"},
    {"question": "Что такое шаблоны 'Prompts'?", "expected": "Преднастроенные шаблоны взаимодействия.", "source": "prompts.mdx"},
    {"question": "Как клиент MCP подключается к нескольким серверам?", "expected": "Через клиентский оркестратор соединений.", "source": "architecture.mdx"}
]

def run_evaluation():
    # Инициализируем агента (проверь имя папки индекса!)
    try:
        agent = MCPAgent(index_path="faiss_structural_index")
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        return

    results = []
    print(f"🚀 Начинаем оценку {len(TEST_SUITE)} вопросов...")

    for i, item in enumerate(TEST_SUITE, 1):
        q = item["question"]
        print(f"[{i}/{len(TEST_SUITE)}] Processing: {q[:40]}...")

        # Замеряем время для интереса (RAG обычно чуть медленнее)
        start = time.time()
        answer_no_rag = agent.ask(q, use_rag=False)
        time_no_rag = time.time() - start

        start = time.time()
        answer_with_rag = agent.ask(q, use_rag=True)
        time_with_rag = time.time() - start

        results.append({
            "question": q,
            "expected": item["expected"],
            "no_rag": answer_no_rag,
            "with_rag": answer_with_rag,
            "source": item["source"]
        })

    # 2. Сохраняем результат в красивый Markdown
    save_to_markdown(results)

def save_to_markdown(results):
    with open("RAG_Comparison_Report.md", "w", encoding="utf-8") as f:
        f.write("# RAG Evaluation Report\n\n")
        f.write("| # | Question | Expected (Key Point) | Without RAG | With RAG (MCP Docs) |\n")
        f.write("|---|---|---|---|---|\n")

        for i, res in enumerate(results, 1):
            # Убираем переносы строк для корректности таблицы
            no_rag = res['no_rag'].replace('\n', ' ')
            with_rag = res['with_rag'].replace('\n', ' ')

            f.write(f"| {i} | **{res['question']}** | *{res['expected']}* | {no_rag} | **{with_rag}** |\n")

    print("\n✅ Отчет сформирован: RAG_Comparison_Report.md")

if __name__ == "__main__":
    run_evaluation()