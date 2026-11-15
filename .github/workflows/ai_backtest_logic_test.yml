
name: AI Backtest Logic Tests

on:
  workflow_dispatch:

jobs:
  test-backtest-logic:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run execution logic tests
        run: |
          python -m unittest Src.Engines.AutonomousTechnicalAnalyst.test_ai_backtest_execution -v
