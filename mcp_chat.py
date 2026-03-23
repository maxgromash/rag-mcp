from rag_agent import ConversationalMCPAgent

def start_chat():
    agent = ConversationalMCPAgent()
    print("🚀 MCP Production-Like Chat Started!")
    print("Type 'exit' to quit. Type 'state' to see current task memory.\n")

    while True:
        user_input = input("👤 You: ")
        if user_input.lower() in ['exit', 'quit']: break

        if user_input.lower() == 'state':
            print(f"🧠 TASK STATE: {agent.task_state}")
            continue

        answer = agent.ask_chat(user_input)
        print(f"\n🤖 AI: {answer}\n")
        print("-" * 40)

if __name__ == "__main__":
    start_chat()