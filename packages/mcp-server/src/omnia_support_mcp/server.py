"""MCP server entry point — registers customer support tools."""

from mcp.server.fastmcp import FastMCP

from omnia_support_mcp.tools.inventory import get_product, search_product
from omnia_support_mcp.tools.orders import get_order, list_customer_orders
from omnia_support_mcp.tools.refunds import get_refund, process_refund

mcp = FastMCP("omnia-support-tools")

mcp.tool()(get_order)
mcp.tool()(list_customer_orders)
mcp.tool()(get_product)
mcp.tool()(search_product)
mcp.tool()(process_refund)
mcp.tool()(get_refund)


def main() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
