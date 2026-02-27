from typing import List, Dict

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

class MCPClient:

	@classmethod
	async def get_mcp_tools(cls, tool_map: Dict) -> List[
    	BaseTool]:
		client = MultiServerMCPClient(
			tool_map
		)
		tools = await client.get_tools()
		return tools

	@classmethod
	async def check_mcp(cls, tool_map: Dict) -> bool:
		try:
			client = MultiServerMCPClient(
				tool_map
			)

			await client.get_tools()
			return True
		except Exception as e:
			raise e