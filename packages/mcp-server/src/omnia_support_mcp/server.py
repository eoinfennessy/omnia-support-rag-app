"""MCP server entry point — registers customer support tools."""

import os
from typing import Literal, cast, get_args

from mcp.server.fastmcp import FastMCP

from omnia_support_mcp.tools.inventory import get_product, search_product
from omnia_support_mcp.tools.orders import get_order, list_customer_orders
from omnia_support_mcp.tools.refunds import get_refund, process_refund

mcp = FastMCP(
    "omnia-support-tools",
    host=os.environ.get("MCP_HOST", "127.0.0.1"),
    port=int(os.environ.get("MCP_PORT", "8000")),
)

mcp.tool()(get_order)
mcp.tool()(list_customer_orders)
mcp.tool()(get_product)
mcp.tool()(search_product)
mcp.tool()(process_refund)
mcp.tool()(get_refund)

Transport = Literal["stdio", "sse", "streamable-http"]


def main() -> None:
    """Run the MCP server."""
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport not in get_args(Transport):
        raise SystemExit(f"Invalid MCP_TRANSPORT={transport!r}, expected one of {get_args(Transport)}")
    mcp.run(transport=cast("Transport", transport))


if __name__ == "__main__":
    main()
