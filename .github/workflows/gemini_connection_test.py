
name: Gemini Connection Test

on:
  workflow_dispatch:

permissions:
  contents: read

jobs:
  test-gemini:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install Gemini SDK
        run: pip install google-genai

      - name: Test Gemini connection
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
        run: |
          python -c "from google import genai; import os; client = genai.Client(api_key=os.environ['GEMINI_API_KEY']); response = client.models.generate_content(model='gemini-flash-latest', contents='Reply only with: GEMINI_CONNECTION_SUCCESSFUL'); print(response.text)"
