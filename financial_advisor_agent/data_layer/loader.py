"""
Raw JSON data loader.

Loads all 6 mock financial data files once at startup and keeps them in memory.
Zero business logic here — pure I/O only.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# The 6 required data files — discovered dynamically, not hardcoded in tools/agents
REQUIRED_FILES = [
    "market_data.json",
    "portfolios.json",
    "news_data.json",
    "mutual_funds.json",
    "historical_data.json",
    "sector_mapping.json",
]


class DataLoader:
    """
    Loads all 6 financial mock-data JSON files from ``data_dir``.
    Files are read once at startup; all downstream code works from these
    in-memory dicts — no repeated disk I/O.
    """

    def __init__(self, data_dir: str) -> None:
        self._data_dir = Path(data_dir).resolve()
        logger.info("DataLoader: resolving data directory → %s", self._data_dir)
        self._validate_directory()
        self._load_all()
        logger.info("DataLoader: all %d files loaded successfully.", len(REQUIRED_FILES))

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_directory(self) -> None:
        """Raise early if data_dir or any required file is missing."""
        if not self._data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self._data_dir}\n"
                "Set DATA_DIR in your .env to the folder containing the 6 JSON files."
            )
        missing = [f for f in REQUIRED_FILES if not (self._data_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing required data files in {self._data_dir}: {missing}"
            )

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self, filename: str) -> dict:
        path = self._data_dir / filename
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        logger.debug("Loaded %s (%d bytes)", filename, path.stat().st_size)
        return data

    def _load_all(self) -> None:
        self.market_data: dict    = self._load("market_data.json")
        self.portfolios: dict     = self._load("portfolios.json")
        self.news_data: dict      = self._load("news_data.json")
        self.mutual_funds: dict   = self._load("mutual_funds.json")
        self.historical: dict     = self._load("historical_data.json")
        self.sector_mapping: dict = self._load("sector_mapping.json")

    # ── Convenience accessors (raw, no transformation) ─────────────────────────

    @property
    def all_stocks(self) -> dict:
        """Raw stocks dict from market_data.json → stocks."""
        return self.market_data.get("stocks", {})

    @property
    def all_indices(self) -> dict:
        """Raw indices dict from market_data.json → indices."""
        return self.market_data.get("indices", {})

    @property
    def all_sectors(self) -> dict:
        """Raw sector_performance dict from market_data.json."""
        return self.market_data.get("sector_performance", {})

    @property
    def all_portfolios(self) -> dict:
        """Raw portfolios dict from portfolios.json."""
        return self.portfolios.get("portfolios", {})

    @property
    def all_news(self) -> list:
        """Raw news list from news_data.json."""
        return self.news_data.get("news", [])

    @property
    def all_mutual_funds(self) -> dict:
        """Raw mutual_funds dict from mutual_funds.json."""
        return self.mutual_funds.get("mutual_funds", {})

    @property
    def all_sector_definitions(self) -> dict:
        """Raw sectors dict from sector_mapping.json."""
        return self.sector_mapping.get("sectors", {})

    @property
    def macro_correlations(self) -> dict:
        """Macro correlation rules from sector_mapping.json."""
        return self.sector_mapping.get("macro_correlations", {})

    @property
    def market_breadth(self) -> dict:
        """Market breadth from historical_data.json."""
        return self.historical.get("market_breadth", {})

    @property
    def fii_dii_data(self) -> dict:
        """FII/DII flow data from historical_data.json."""
        return self.historical.get("fii_dii_data", {})

    @property
    def index_history(self) -> dict:
        """Index historical data from historical_data.json."""
        return self.historical.get("index_history", {})

    @property
    def stock_history(self) -> dict:
        """Stock historical data from historical_data.json."""
        return self.historical.get("stock_history", {})

    @property
    def sector_weekly_performance(self) -> dict:
        """Sector weekly performance from historical_data.json."""
        return self.historical.get("sector_weekly_performance", {})
