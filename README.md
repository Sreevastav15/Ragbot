# Document Extraction App

A simple and efficient application for uploading PDF documents, extracting structured data (questions and answers), and enabling users to chat with their documents using Retrieval-Augmented Generation (RAG).

---

## 🚀 Features
- **PDF Uploading** – Upload documents through the frontend.
- **Automatic Text Extraction** – Extract text from PDFs using PyPDF.
- **Chunking & Embeddings** – Split text into chunks and generate embeddings with HuggingFace models.
- **Vector Search** – Store embeddings in ChromaDB for retrieval.
- **RAG Chat Support** – Ask questions about the uploaded document.
- **PostgreSQL** – Store documents, extracted questions, and answers.
- **FastAPI Backend** – REST API for uploads, queries, and chat.
- **React Frontend** – User-friendly interface for uploads and chat.

---

## 🛠 Tech Stack
### **Backend**
- FastAPI
- SQLAlchemy
- LangChain
- Groq Api
- Google Embeddings
- ChromaDB
- PyPDFLoader
- Python 3.10+

### **Frontend**
- React
- Tailwind CSS
- Axios
- React Hot Toast

---

## 📥 Installation & Setup

### **1️⃣ Clone the Repository**
```bash
git clone <repo-url>
cd projects/Ragbot
```

### **2️⃣ Backend Setup**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Your backend now runs at:
```
http://localhost:8000
```

### **3️⃣ Frontend Setup**
```bash
cd frontend
npm install
npm start
```

Frontend runs at:
```
http://localhost:3000
```

---
### API Keys
The project use Groq Api. You can get your key at:

https://console.groq.com/keys

The project uses Google API. You can get your key at:

https://aistudio.google.com/api-keys

You also need PostgreSQL url to run this project.

---

## ✨ Author
**Sreevastav Vavilala** – 2025

