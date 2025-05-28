import asyncio
import os
import shutil
import json
from contextlib import AsyncExitStack
from typing import List

from agents import Agent, Runner, gen_trace_id, trace
from agents.mcp import MCPServer, MCPServerStdio


async def run(mcp_servers: List[MCPServer]):
    agent = Agent(
        name="Assistant",
        instructions="Use the mcp-server tools to answer questions. All questions should be answered using the tools provided by the MCP servers.",
        mcp_servers=mcp_servers,
        # model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )

    # List the files it can read
    message = "Read the files and list them."
    print(f"Running: {message}")
    result = await Runner.run(starting_agent=agent, input=message)
    print(result.final_output)

    # Ask about books
    message = "Locate the file containing a list of my favorite books. What is my #1 favorite book?"
    print(f"\n\nRunning: {message}")
    result = await Runner.run(starting_agent=agent, input=message)
    print(result.final_output)

    # Ask a question that reads then reasons.
    message = "Locate the file containing a list of my favorite songs. Suggest one new song that I might like."
    print(f"\n\nRunning: {message}")
    result = await Runner.run(starting_agent=agent, input=message)
    print(result.final_output)

    # Ask a question that reads then reasons.
    message = "Fetch deeplearning.ai and find an interesting term to search papers around. Summarize your findings and write them to a file called 'suggested_research.txt'"
    print(f"\n\nRunning: {message}")
    result = await Runner.run(starting_agent=agent, input=message)
    print(result.final_output)

    # Ask a question that reads then reasons.
    message = "Summarize the paper titled 'Is Physics Sick? [In Praise of Classical Physics]' using the details available in the directory 'papers/'. Write the summary to a file called 'physics_summary.txt'."
    print(f"\n\nRunning: {message}")
    result = await Runner.run(starting_agent=agent, input=message)
    print(result.final_output)

async def main():
    # Ensure npx is installed
    if not shutil.which("npx"):
        raise RuntimeError("npx is not installed. Please install it with `npm install -g npx`.")

    # Load MCP server configurations
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "server_config.json")
    with open(config_path, "r") as f:
        data = json.load(f)

    # Enter all MCPServerStdio contexts
    async with AsyncExitStack() as stack:
        servers: List[MCPServer] = []
        for name, cfg in data.get("mcpServers", {}).items():
            server = await stack.enter_async_context(
                MCPServerStdio(
                    name=f"{name} server",
                    params=cfg,
                )
            )
            servers.append(server)

        trace_id = gen_trace_id()
        with trace(workflow_name="MCP Multi-Server Example", trace_id=trace_id):
            print(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}\n")
            await run(servers)


if __name__ == "__main__":
    asyncio.run(main())