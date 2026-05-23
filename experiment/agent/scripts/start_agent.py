"""
Start Agent --Entry Point  (Phase 6)
======================================

Usage:
  cd experiment
  python scripts/start_agent.py [--port 8000] [--host 0.0.0.0] [--reload]

Opens the web chat at http://localhost:8000/
API docs at http://localhost:8000/docs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure `experiment/` is on the Python path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Station Agent --FastAPI server")
    p.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    p.add_argument(
        "--reload", action="store_true",
        help="Enable auto-reload on file change (development mode)"
    )
    p.add_argument(
        "--log-level", default="info",
        choices=["debug", "info", "warning", "error"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import uvicorn
    except ImportError:
        print(
            "[ERROR] uvicorn is not installed.\n"
            "Install it with:  pip install fastapi uvicorn[standard]\n"
        )
        sys.exit(1)

    print(f"\n  Station Agent  ·  http://{args.host}:{args.port}/")
    print(f"  API docs       ·  http://{args.host}:{args.port}/docs")
    print(f"  Reload         ·  {'enabled' if args.reload else 'disabled'}\n")

    uvicorn.run(
        "src.tool_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        app_dir=str(ROOT),
    )


if __name__ == "__main__":
    main()
