"""
Startup helper — run from inside the financial_advisor_agent/ directory.
Adds the project root to sys.path before launching uvicorn.
"""
import sys
import os
from pathlib import Path

# Add project root to path so `config`, `agent`, `data_layer`, `app` are importable
root = Path(__file__).parent
sys.path.insert(0, str(root))

import uvicorn
from config import settings

if __name__ == "__main__":
    print("=" * 60)
    print("  Autonomous Financial Advisor Agent")
    print(f"  URL: http://localhost:{settings.port}")
    print(f"  API docs: http://localhost:{settings.port}/docs")
    print("=" * 60)
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        reload_dirs=[str(root)],
    )
