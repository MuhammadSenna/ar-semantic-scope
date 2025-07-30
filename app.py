import streamlit as st
import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import os

# Page configuration
st.set_page_config(
    page_title="Arabic QnA Semantic Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better Arabic text display
st.markdown("""
<style>
.arabic-text {
    direction: rtl;
    text-align: right;
    font-family: 'Arial Unicode MS', 'Tahoma', sans-serif;
    font-size: 16px;
    line-height: 1.6;
    padding: 10px;
    background-color: #4f5157;
    border-radius: 5px;
    margin: 10px 0;
}

.question-box {
    background-color: #41a5f0;
    border-left: 4px solid #1f77b4;
    padding: 15px;
    margin: 10px 0;
    border-radius: 5px;
}

.answer-box {
    background-color: #45cc45;
    border-left: 4px solid #2ca02c;
    padding: 15px;
    margin: 10px 0;
    border-radius: 5px;
}

.similarity-score {
    background-color: #c7a330;
    border: 1px solid #ffeaa7;
    padding: 5px 10px;
    border-radius: 15px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prepare_data():
    """Load and prepare the Arabic QnA dataset"""
    with st.spinner("Loading Arabic QnA dataset..."):
        # Load dataset
        qna_dataset = load_dataset("sadeem-ai/arabic-qna")
        
        # Filter for entries that have answers
        qna_dataset = qna_dataset.filter(lambda x: x["has_answer"] == True)
        
        # Extract text and metadata
        doc_text = list(qna_dataset["train"]["text"])
        questions = list(qna_dataset["train"]["question"])
        answers = list(qna_dataset["train"]["answer"])
        
        metadata = [
            {
                "source": rec["source"],
                "title": rec["title"],
                "question": rec["question"],
                "answer": rec["answer"]
            }
            for rec in qna_dataset["train"]
        ]
        
        return doc_text, metadata, questions, answers, qna_dataset

@st.cache_resource
def load_model():
    """Load the sentence transformer model"""
    model_id = 'sentence-transformers/distiluse-base-multilingual-cased-v2'
    try:
        model = SentenceTransformer(model_id)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def create_embeddings_and_index(_model, doc_text):
    """Create embeddings and FAISS index"""
    with st.spinner("Creating embeddings and search index..."):
        # Create embeddings
        encoded_text = _model.encode(doc_text, show_progress_bar=False)
        
        # Normalize embeddings
        norm_encoded_docs = deepcopy(encoded_text)
        faiss.normalize_L2(norm_encoded_docs)
        
        # Create FAISS index
        dim = encoded_text.shape[1]
        text_ids = np.array([i for i in range(len(doc_text))], dtype=np.int64)
        
        faiss_index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        faiss_index.add_with_ids(norm_encoded_docs, text_ids)
        
        return faiss_index, encoded_text

def search_similar_questions(model, faiss_index, query, metadata, top_k=5):
    """Search for similar questions using semantic similarity"""
    # Encode query
    question_embd = model.encode(query)
    question_embd = question_embd.reshape(1, -1)
    faiss.normalize_L2(question_embd)
    
    # Search
    similarities, indices = faiss_index.search(question_embd, top_k)
    
    results = []
    for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
        if idx != -1:  # Valid result
            results.append({
                'rank': i + 1,
                'similarity': float(similarity),
                'question': metadata[idx]['question'],
                'answer': metadata[idx]['answer'],
                'source': metadata[idx]['source'],
                'title': metadata[idx]['title']
            })
    
    return results

def display_search_results(results):
    """Display search results in a formatted way"""
    for result in results:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"""
            <div class="question-box">
                <h4>السؤال (Question #{result['rank']}):</h4>
                <div class="arabic-text">{result['question']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="answer-box">
                <h4>الإجابة (Answer):</h4>
                <div class="arabic-text">{result['answer']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("المصدر والتفاصيل (Source & Details)"):
                st.write(f"**المصدر (Source):** {result['source']}")
                st.write(f"**العنوان (Title):** {result['title']}")
        
        with col2:
            similarity_percentage = result['similarity'] * 100
            st.markdown(f"""
            <div class="similarity-score">
                التشابه: {similarity_percentage:.1f}%
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()

def show_dataset_statistics(qna_dataset, questions, answers):
    """Display dataset statistics"""
    st.subheader("إحصائيات البيانات (Dataset Statistics)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("إجمالي الأسئلة (Total Questions)", len(questions))
    
    with col2:
        avg_question_length = np.mean([len(q.split()) for q in questions])
        st.metric("متوسط طول السؤال (Avg Question Length)", f"{avg_question_length:.1f} words")
    
    with col3:
        avg_answer_length = np.mean([len(a.split()) for a in answers])
        st.metric("متوسط طول الإجابة (Avg Answer Length)", f"{avg_answer_length:.1f} words")
    
    # Distribution chart
    st.subheader("توزيع مصادر البيانات (Data Sources Distribution)")
    
    sources = [item['source'] for item in qna_dataset['train']]
    source_counts = pd.Series(sources).value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=source_counts.values, y=source_counts.index, ax=ax)
    plt.title('Distribution of Data Sources')
    plt.xlabel('Count')
    plt.ylabel('Source')
    st.pyplot(fig)

def main():
    st.title("🔍 البحث الدلالي في الأسئلة والأجوبة العربية")
    st.title("Arabic QnA Semantic Search")
    
    st.markdown("""
    هذا التطبيق يستخدم تقنيات الذكاء الاصطناعي للبحث في قاعدة بيانات الأسئلة والأجوبة العربية باستخدام التشابه الدلالي.
    
    This application uses AI techniques to search through an Arabic QnA database using semantic similarity.
    """)
    
    # Load data and model
    try:
        doc_text, metadata, questions, answers, qna_dataset = load_and_prepare_data()
        model = load_model()
        
        if model is None:
            st.error("فشل في تحميل النموذج. يرجى المحاولة مرة أخرى.")
            return
        
        faiss_index, embeddings = create_embeddings_and_index(model, doc_text)
        
        # Sidebar
        with st.sidebar:
            st.header("إعدادات البحث (Search Settings)")
            
            top_k = st.slider(
                "عدد النتائج (Number of Results)",
                min_value=1,
                max_value=10,
                value=5
            )
            
            st.divider()
            
            show_stats = st.checkbox("عرض إحصائيات البيانات (Show Dataset Statistics)")
            
            if show_stats:
                show_dataset_statistics(qna_dataset, questions, answers)
        
        # Main search interface
        st.subheader("البحث (Search)")
        
        # Example questions
        st.markdown("**أمثلة على الأسئلة (Example Questions):**")
        example_questions = [
            "ما السبب في صغر الأسنان",
            "كيف أتعلم البرمجة",
            "ما هي فوائد الرياضة",
            "كيف أحسن من صحتي"
        ]
        
        cols = st.columns(len(example_questions))
        for i, example in enumerate(example_questions):
            if cols[i].button(example, key=f"example_{i}"):
                st.session_state.search_query = example
        
        # Search input
        search_query = st.text_input(
            "اكتب سؤالك هنا (Enter your question here):",
            value=st.session_state.get('search_query', ''),
            placeholder="مثال: ما السبب في صغر الأسنان؟"
        )
        
        if st.button("بحث (Search)", type="primary") or search_query:
            if search_query.strip():
                with st.spinner("جاري البحث... (Searching...)"):
                    results = search_similar_questions(
                        model, faiss_index, search_query, metadata, top_k
                    )
                    
                    if results:
                        st.success(f"تم العثور على {len(results)} نتيجة مشابهة")
                        st.markdown("---")
                        display_search_results(results)
                    else:
                        st.warning("لم يتم العثور على نتائج مشابهة")
            else:
                st.warning("يرجى إدخال سؤال للبحث")
                
    except Exception as e:
        st.error(f"حدث خطأ في تحميل البيانات: {str(e)}")
        st.markdown("""
        **حلول مقترحة:**
        1. تأكد من اتصالك بالإنترنت
        2. أعد تشغيل التطبيق
        3. تحقق من توفر مساحة كافية على القرص الصلب
        """)

if __name__ == "__main__":
    main()