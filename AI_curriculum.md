# AI Engineer Program
**Duration:** 60 Hours · **Format:** 70% Hands-on, 30% Theory · **Environment:** VS Code + GitHub

---

## Prerequisites
- Python basics — functions, classes, error handling
- VS Code and terminal comfort
- GitHub — commits, repos, README

No ML or math background required.

---

## Hour Allocation

| Module | Topic | Hours |
|--------|-------|-------|
| 1 | Python for AI & EDA | 10 |
| 2 | Machine Learning | 13 |
| 3 | Deep Learning | 13 |
| 4 | Generative AI | 19 |
| 5 | Agentic AI | 5 |
| **Total** | | **60** |

---

## Module 1 — Python for AI & EDA
**Hours 1–10**

### Goal
Build fluency with the data tools every AI workflow depends on before touching a model.

### Tech Stack
Python · NumPy · Pandas · Matplotlib · Seaborn · `uv` · GitHub

### Hour-by-Hour

| Hour | Topic |
|------|-------|
| 1 | AI landscape overview; `uv` project setup, GitHub workflow |
| 2 | NumPy — arrays, vectorised operations, broadcasting |
| 3 | Pandas — loading, slicing, and exploring tabular data |
| 4 | Data cleaning — missing values, duplicates, type coercion |
| 5 | Descriptive statistics — mean, median, std, correlation |
| 6 | Matplotlib — line, bar, scatter, histogram plots |
| 7 | Seaborn & Plotly — distribution plots, heatmaps, interactive charts |
| 8 | EDA hands-on — full exploration of a real dataset (Netflix / IPL / housing) |
| 9 | EDA report — summarising findings, writing insights |
| 10 | **Project:** End-to-end EDA → clean CSV → GitHub push with README |

### Project — EDA Report
A Jupyter notebook that loads a raw dataset, cleans it, visualises key patterns, and summarises 5 actionable insights.

---

## Module 2 — Machine Learning
**Hours 11–23**

### Goal
Understand how machines learn from data, evaluate predictions, and fail — the foundation for reasoning about any AI system.

### Tech Stack
scikit-learn · Pandas · Matplotlib · joblib

### Hour-by-Hour

| Hour | Topic |
|------|-------|
| 11 | What is ML — supervised vs unsupervised; the training loop |
| 12 | Data splitting — train / val / test; avoiding leakage |
| 13 | Preprocessing — scaling, encoding, imputation with scikit-learn pipelines |
| 14 | Linear regression — intuition, coefficients, residuals |
| 15 | Logistic regression — classification, decision boundary, probabilities |
| 16 | Decision trees — splits, depth, overfitting by hand |
| 17 | Random forests & ensemble methods — why many weak models beat one strong one |
| 18 | Model evaluation — accuracy, precision, recall, F1, ROC-AUC |
| 19 | Hyperparameter tuning — GridSearchCV, cross-validation |
| 20 | Feature importance and selection — what the model actually uses |
| 21 | Unsupervised learning — K-means clustering, dimensionality reduction (PCA) |
| 22 | ML in the real world — credit scoring, churn prediction, fraud detection |
| 23 | **Project:** Train, evaluate, and compare two models on a tabular dataset; push to GitHub |

### Project — ML Classifier
A scikit-learn pipeline that preprocesses data, trains two models, compares them with a metric report, and saves the best model with `joblib`.

---

## Module 3 — Deep Learning
**Hours 24–36**

### Goal
See how neural networks power image, text, and sequential AI — and learn to train, debug, and deploy them.

### Tech Stack
PyTorch · TensorFlow/Keras · TensorBoard · Hugging Face Transformers · Google Colab (GPU)

### Hour-by-Hour

| Hour | Topic |
|------|-------|
| 24 | Neural network intuition — neurons, layers, activations, loss |
| 25 | Backpropagation and gradient descent — simplified math, visual demo |
| 26 | Build your first NN in PyTorch — MNIST digit classifier |
| 27 | CNNs — how AI "sees" images; convolution, pooling, feature maps |
| 28 | CNN hands-on — cat vs dog classifier in Colab |
| 29 | Overfitting and regularisation — dropout, batch norm, early stopping |
| 30 | Visualising training — TensorBoard loss/accuracy curves |
| 31 | Transfer learning — fine-tuning ResNet / MobileNet on custom data |
| 32 | Recurrent networks & sequence intuition — RNNs, LSTMs (conceptual) |
| 33 | Evolution to Transformers — self-attention and why it replaced RNNs |
| 34 | Hugging Face intro — using pre-trained models for text classification and Q&A |
| 35 | Fine-tuning BERT on a small custom dataset |
| 36 | **Project:** Image classifier with transfer learning; accuracy report pushed to GitHub |

### Project — Transfer Learning Classifier
A fine-tuned CNN that classifies custom images, reports precision/recall, and includes training curves and a confusion matrix.

---

## Module 4 — Generative AI
**Hours 37–55**

### Goal
Move from models that predict to models that create — understand LLMs, prompting, embeddings, RAG, and how to build reliable Gen AI applications.

### Tech Stack
OpenAI / Anthropic API · Hugging Face · LangChain · FAISS / ChromaDB · Pydantic · FastAPI · Python `logging`

### Hour-by-Hour

| Hour | Topic |
|------|-------|
| 37 | How LLMs work — tokens, context windows, temperature, sampling |
| 38 | LLMs as APIs — request/response lifecycle, cost and latency control |
| 39 | Prompt engineering — system vs user prompts, few-shot, chain-of-thought |
| 40 | Structured outputs — JSON schema enforcement with Pydantic |
| 41 | Error handling for LLM calls — timeouts, retries, fallback prompts |
| 42 | Logging and observability — audit trails, token tracking |
| 43 | **Mini-project:** LLM utility service — structured input → validated JSON output |
| 44 | Embeddings — what they are, how similarity search works |
| 45 | Vector databases — FAISS and ChromaDB hands-on |
| 46 | What is RAG — why models forget and how retrieval fixes it |
| 47 | Document loaders — ingesting PDFs, CSVs, web pages |
| 48 | Building a RAG pipeline end-to-end |
| 49 | Improving retrieval quality — chunking strategies, metadata filtering |
| 50 | **Mini-project:** PDF Q&A chatbot with source citations |
| 51 | Image generation concepts — diffusion models, DALL-E, Stable Diffusion |
| 52 | Multimodal models — vision + language (GPT-4o, Claude vision) |
| 53 | Responsible AI — bias, hallucination, explainability, safety |
| 54 | Wrapping a Gen AI feature in FastAPI |
| 55 | **Project:** Deployed RAG API — FastAPI endpoint over a document store |

### Project — Deployed RAG API
A FastAPI service that:
- Accepts a natural-language question
- Retrieves relevant chunks from a vector store
- Returns an LLM-generated answer with source references
- Logs every request with latency and token usage

---

## Module 5 — Agentic AI
**Hours 56–60**

### Goal
Understand what makes an AI system "agentic" — perception, planning, tool use, and memory — and build one minimal but complete agent.

### Tech Stack
LangChain AgentExecutor · LangGraph (intro) · Tool schemas

### Hour-by-Hour

| Hour | Topic |
|------|-------|
| 56 | What is an agent — perception, reasoning, action, memory loop; ReAct pattern |
| 57 | Defining tools — schemas, descriptions; building a file and search tool |
| 58 | AgentExecutor — how the reasoning loop works; multi-step task demo |
| 59 | Stateful agents intro — LangGraph nodes and edges; when state matters |
| 60 | **Project:** Task automation agent — given a goal, selects tools, executes steps, logs decisions |

### Project — Task Automation Agent
An agent that:
- Accepts a plain-English goal
- Selects and executes tools across multiple steps
- Logs every reasoning step and tool call
- Handles at least one tool failure gracefully

---

## Final Outcomes

By the end of this program, students will be able to:

- Perform end-to-end EDA on real datasets
- Train, evaluate, and compare ML models
- Build and fine-tune deep learning models for image and text
- Build production-grade Gen AI applications with RAG and FastAPI
- Understand agentic AI architecture and build a minimal working agent
- Maintain a professional GitHub portfolio across all five domains
