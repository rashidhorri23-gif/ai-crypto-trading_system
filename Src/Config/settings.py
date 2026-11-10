# System Configuration

PROJECT_NAME = "AI Crypto Trading System"
PROJECT_VERSION = "1.0.0"

# Execution Settings
EXECUTION_MODE = "MANUAL"
RUN_MODE = "ON_DEMAND"
TRADING_ENABLED = False

# Market Settings
MARKETS = ["SPOT", "SWAP"]
MAX_ASSETS = 50

# Initial Pilot Asset
ASSETS = ["BTC-USDT"]

# Supported Timeframes
TIMEFRAMES = [
    "5m",
    "15m",
    "1H",
    "4H",
    "1D"
]

# Time Settings
TIMEZONE = "UTC"

# Data Settings
DATA_PROVIDER = "UNSELECTED"
USE_COMPLETED_CANDLES_ONLY = True

# Analysis Settings
ENABLE_TECHNICAL_ANALYSIS = True
ENABLE_SMC_ANALYSIS = True
ENABLE_NEWS_ANALYSIS = False
ENABLE_TRADER_PERFORMANCE_ANALYSIS = False

# Risk Management
DEFAULT_RISK_PERCENT = 1.0
MAX_DAILY_LOSS_PERCENT = 3.0
MAX_OPEN_POSITIONS = 3

# Reporting
TELEGRAM_ENABLED = False

# Safety Settings
ALLOW_LIVE_ORDERS = False
ALLOW_WITHDRAWALS = False
