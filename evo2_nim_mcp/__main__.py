"""Entry point: `python -m evo2_nim_mcp` and the `evo2-nim-mcp` script."""

from evo2_nim_mcp.server import mcp


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
