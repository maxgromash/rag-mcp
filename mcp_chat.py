import sys
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner
from rag_agent import ConversationalMCPAgent

console = Console()

def run_mcp_chat():
    try:
        with console.status("[bold green]Загрузка базы знаний и LLM...", spinner="dots"):
            agent = ConversationalMCPAgent()
    except Exception as e:
        console.print(f"[bold red]Ошибка инициализации: {e}")
        return

    console.print(Panel.fit(
        "[bold cyan]MCP RAG CHAT v2.0[/bold cyan]\n"
        "[dim]Локальный ассистент по Model Context Protocol[/dim]",
        border_style="blue"
    ))

    console.print("[italic gray]Команды: 'exit' - выход, 'state' - память задачи, 'clear' - очистка истории[/italic gray]\n")

    while True:
        try:
            user_input = console.input("[bold yellow]Вы: [/bold yellow]").strip()

            if not user_input: continue
            if user_input.lower() in ['exit', 'quit']: break

            if user_input.lower() == 'state':
                console.print(Panel(str(agent.task_state), title="[bold magenta]Current Task State[/bold magenta]", border_style="magenta"))
                continue

            # Индикатор раздумий
            with console.status("[bold blue]Агент думает и ищет в доках...", spinner="bouncingBar"):
                answer = agent.ask_chat(user_input)

            # Рендерим Markdown-ответ
            console.print(f"\n[bold green]AI:[/bold green]")
            console.print(Markdown(answer))
            console.print("[dim]" + "-" * 30 + "[/dim]\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[bold red]Произошла ошибка: {e}")

    console.print("\n[bold blue]До связи! Удачи в разработке MCP-серверов. 👋[/bold blue]")

if __name__ == "__main__":
    run_mcp_chat()