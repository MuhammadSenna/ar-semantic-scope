# Arabic QnA Semantic Search Application

A Streamlit web application for semantic search through Arabic Questions and Answers using AI-powered similarity matching.

## Features

- 🔍 **Semantic Search**: Find similar questions using advanced NLP models
- 🌐 **Arabic Language Support**: Optimized for Arabic text processing
- 📊 **Dataset Statistics**: Visualize data distribution and statistics  
- 🎯 **Relevance Scoring**: Shows similarity percentages for search results
- 💻 **User-friendly Interface**: Clean, responsive web interface

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Internet connection (for downloading models and datasets)

### Step 1: Clone or Download
Save the main application code as `app.py` and the requirements as `requirements.txt`

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv arabic_qna_env
source arabic_qna_env/bin/activate  # On Windows: arabic_qna_env\Scripts\activate
```

### Step 3: Install NumPy (Important!)

```bash
pip install "numpy<2.0"
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Install Faiss (See Troubleshooting if issues occur)

```bash
pip install --upgrade --force-reinstall faiss-cpu
```


### Step 6: Run the Application
```bash
streamlit run app.py
```

The application will automatically:
1. Download the Arabic QnA dataset from Hugging Face Hub
2. Load the multilingual sentence transformer model
3. Create embeddings and search index
4. Launch the web interface

## Usage

1. **Search**: Enter your Arabic question in the search box
2. **Examples**: Click on example questions to try the system
3. **Results**: View semantically similar questions with similarity scores
4. **Statistics**: Enable dataset statistics in the sidebar to see data insights

## Example Queries

Try these Arabic questions:
- `ما السبب في صغر الأسنان` (What causes small teeth?)
- `كيف أتعلم البرمجة` (How do I learn programming?)
- `ما هي فوائد الرياضة` (What are the benefits of sports?)
- `كيف أحسن من صحتي` (How can I improve my health?)

## Technical Details

### Models Used
- **Sentence Transformer**: `distiluse-base-multilingual-cased-v2`
- **Search Engine**: FAISS (Facebook AI Similarity Search)
- **Dataset**: `sadeem-ai/arabic-qna` from Hugging Face Hub

### Architecture
1. **Data Loading**: Loads and filters Arabic QnA dataset
2. **Embedding**: Creates vector representations of questions using sentence transformers
3. **Indexing**: Builds FAISS index for fast similarity search
4. **Search**: Performs semantic similarity search with cosine similarity
5. **Display**: Shows results with Arabic text formatting

## Performance Considerations

- **First Run**: Initial setup takes 2-5 minutes to download models and create embeddings
- **Memory Usage**: Requires approximately 2-4 GB RAM for full dataset
- **Search Speed**: Sub-second search after initial setup


## Project Structure
```
arabic-qna-search/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .streamlit/           # Streamlit configuration (optional)
    └── config.toml
```



## Troubleshooting

### Common Issues and Solutions

#### 1. **Datasets IndexError: "Wrong key type"**
**Problem:** `TypeError: Wrong key type: '822' of type '<class 'numpy.int64'>'. Expected one of int, slice, range, str or Iterable.`

**Solution:** Convert dataset to list:
```python
# Instead of:
doc_text = qna_dataset["train"]["text"]

# Use:
doc_text = list(qna_dataset["train"]["text"])
```

#### 2. **NumPy Compatibility Error with Faiss**
**Problem:** 
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.2 as it may crash.
AttributeError: _ARRAY_API not found
```

**Solution:** Downgrade NumPy before installing Faiss:
```bash
pip install "numpy<2.0"
pip install --upgrade --force-reinstall faiss-cpu
```

#### 3. **Faiss Installation Error on Windows**
**Problem:** 
```
Building wheel for faiss-cpu (pyproject.toml) ... error
error: command 'swig.exe' failed: None
```

**Solution:** Use force reinstall to get pre-compiled wheels:
```bash
pip install --upgrade --force-reinstall faiss-cpu
```

### 4. Faiss Normalize Error
**Problem:**
```
IndexError: tuple index out of range when calling faiss.normalize_L2()
 ```
**Solution:** Reshape single embeddings to 2D:
```
# For single query
question_embd = model.encode(question)
question_embd = question_embd.reshape(1, -1) # Add batch dimension
faiss.normalize_L2(question_embd)
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Dataset**: Sadeem AI for the Arabic QnA dataset
- **Models**: Sentence Transformers team for multilingual models
- **Framework**: Streamlit team for the amazing web framework
- **Search**: Facebook AI Research for FAISS