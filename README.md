# Automatic News Text Summarizer

A **bilingual (Arabic + English) abstractive news summarization system** powered by Hugging Face Transformers (mT5, AraBART), FastAPI, and a custom web frontend. Paste any news article in Arabic or English and receive a concise abstractive summary instantly.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Models | Hugging Face Transformers (mT5, AraBART) |
| Backend API | FastAPI + Uvicorn |
| Deep Learning | PyTorch |
| Frontend | HTML / CSS / JavaScript |

---

## Recommended Model Checkpoints

| Model | Description |
|-------|-------------|
| [`csebuetnlp/mT5_multilingual_XLSum`](https://huggingface.co/csebuetnlp/mT5_multilingual_XLSum) | Fine-tuned on XL-Sum dataset across 44 languages including Arabic & English. **Default model.** |
| [`moussaKam/AraBART`](https://huggingface.co/moussaKam/AraBART) | BART-based model with 139M parameters, optimised for Arabic summarization. |
| [`ArabicNLP/mT5-base_ar`](https://huggingface.co/ArabicNLP/mT5-base_ar) | mT5-base adapted specifically for Arabic NLP tasks. |
| [`google/mt5-base`](https://huggingface.co/google/mt5-base) | General-purpose multilingual T5 covering 101 languages (requires fine-tuning for summarization). |

---

## Project Structure

```
Automatic-News-Text-Summarizer/
├── backend/
│   ├── __init__.py
│   ├── config.py        # Configuration constants
│   ├── model.py         # SummarizerModel class & singleton
│   └── main.py          # FastAPI application
├── frontend/
│   ├── index.html       # Main UI page
│   ├── style.css        # Styling with RTL support
│   └── script.js        # API interaction logic
├── tests/
│   ├── __init__.py
│   └── test_api.py      # Pytest tests for the API
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Yxusef777/Automatic-News-Text-Summarizer.git
cd Automatic-News-Text-Summarizer
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the API server

```bash
uvicorn backend.main:app --reload
```

The server will start at `http://localhost:8000`. On first run the model weights (~1.2 GB) will be downloaded automatically from Hugging Face Hub.

### 5. Open the frontend

Open `frontend/index.html` in your browser (or serve it with a simple HTTP server):

```bash
python -m http.server 8080 --directory frontend
```

Then navigate to `http://localhost:8080`.

---

## API Usage

### Health check

```bash
curl http://localhost:8000/health
```

### Summarize an article

```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Paste your news article here...",
    "language": "auto",
    "max_length": 150
  }'
```

**Response:**

```json
{
  "summary": "Generated abstractive summary...",
  "model_used": "csebuetnlp/mT5_multilingual_XLSum",
  "language": "en",
  "original_length": 512
}
```

---

## Running Tests

```bash
pytest tests/
```

---

## License

This project is provided for educational and research purposes. See [LICENSE](LICENSE) for details.