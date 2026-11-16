import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from dotenv import load_dotenv
import json

load_dotenv()

SERVERS = {
    "ExpenseTracker": {
        "transport": "stdio",
        "command": "/Users/shreyas/Documents/Learnings/mcp-server-practice/expense-tracker-mcp/.venv/bin/python",
        "args": [
            "/Users/shreyas/Documents/Learnings/mcp-server-practice/expense-tracker-mcp/src/expense_tracker_mcp/main.py"
        ],
    },
}


async def main():
    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()
    named_tools = {}
    for tool in tools:
        named_tools[tool.name] = tool    

    llm = ChatOpenAI(model="gpt-5")
    llm_with_tools = llm.bind_tools(tools)

    prompt = "Add $20 for sandwich today"
    response = await llm_with_tools.ainvoke(prompt)

    if not getattr(response, "tool_calls", None):
        print("\nLLM Reply:", response.content)
        return

    tool_messages = []
    for tc in response.tool_calls:
        selected_tool = tc["name"]
        selected_tool_args = tc.get("args") or {}
        selected_tool_id = tc["id"]
        tool_result = await named_tools[selected_tool].ainvoke(selected_tool_args)
        tool_messages.append(ToolMessage(tool_call_id=selected_tool_id, content=json.dumps(tool_result)))

    final_response = await llm_with_tools.ainvoke([prompt, response, *tool_messages])
    print("Final Response:", final_response.content)

if __name__ == "__main__":
    asyncio.run(main())
