
name: Five Signal BTC Vision Backtest

on:
  workflow_dispatch:

permissions:
  contents: read

jobs:
  five-signal-backtest:
    runs-on: ubuntu-latest
    timeout-minutes: 30

    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run Five Signal Vision Backtest
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
          OPENROUTER_VISION_MODEL: nvidia/nemotron-3.5-super-vl:free
        run: |
          python -m Src.Engines.AutonomousTechnicalAnalyst.five_signal_vision_backtest

      - name: Upload Five Signal Reports
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: Five-Signal-BTC-Vision-Backtest
          path: Reports/Five_Signal_Vision_Backtest/
          if-no-files-found: warn
          retention-days: 30
