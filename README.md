# Chef Bella — AI Recipe Assistant

A smart recipe assistant built with **Streamlit** and **LangChain**, powered by **AWS Bedrock (Amazon Nova 2 Lite)**.

---

## Features

- Chat with an AI chef for recipes, ingredient substitutions, and nutrition info
- Automatic query routing (recipe / substitution / nutrition / general / off-topic)
- Conversation memory with configurable window size
- Structured recipe cards with ingredients and steps
- Powered by Amazon Nova 2 Lite via AWS Bedrock

---

## Prerequisites

- Python 3.10 or 3.11
- An AWS Bedrock Bearer Token (get one from AWS Console → Amazon Bedrock → API Keys)

---

## Setup & Run

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **Mac / Linux:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python -m streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## Usage

1. Open the app in your browser
2. In the **sidebar**, paste your AWS Bedrock Bearer Token in the password field and click **Connect**
3. Start chatting — ask for recipes, substitutions, or nutrition info

> **Note:** AWS Bearer Tokens expire every **12 hours**. Get a new one from:
> AWS Console → Amazon Bedrock → API Keys → Create short-term key

---

## Project Structure

```
Streamlit App/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker container config
├── .dockerignore       # Files excluded from Docker build
└── README.md           # This file
```

---

## Docker (optional)

Build and run with Docker:

```bash
docker build -t chef-bella .
docker run -p 8501:8501 chef-bella
```
