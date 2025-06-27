import os
import requests
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import google.generativeai as genai

# Hugging Face 저장소에서 FAISS 파일 다운로드
@st.cache_resource
def download_faiss_from_hf():
    hf_url = "https://huggingface.co/datasets/WildOjisan/test_embedding/resolve/main"
    os.makedirs("faiss_data", exist_ok=True)

    for fname in ["index.faiss", "index.pkl"]:
        file_path = f"faiss_data/{fname}"
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(requests.get(f"{hf_url}/{fname}").content)

# FAISS + 임베딩 모델 로드
@st.cache_resource
def load_vector_db():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vector_db = FAISS.load_local("faiss_data", embedding_model, allow_dangerous_deserialization=True)
    return vector_db.as_retriever()

# Gemini API 설정
@st.cache_resource
def load_gemini_model():
    genai.configure(api_key="YOUR_GEMINI_API_KEY")  # 🔑 본인의 Gemini API 키로 교체!
    return genai.GenerativeModel("gemini-2.5-flash")

# RAG 응답 함수
def gemini_rag_answer(query, retriever, model):
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
당신은 문서 기반 질문에 답하는 AI입니다.
다음 문서를 참고해서 질문에 답하십시오:

문서:
{context}

질문:
{query}

답변:
"""
    response = model.generate_content(prompt)
    return response.text

# Streamlit 인터페이스
st.title("📚 문서 검색 기반 Gemini 챗봇")

download_faiss_from_hf()
retriever = load_vector_db()
model = load_gemini_model()

query = st.text_input("❓ 궁금한 내용을 입력하세요:")

if query:
    with st.spinner("🔍 검색 중..."):
        answer = gemini_rag_answer(query, retriever, model)
    st.markdown("### 🤖 Gemini의 답변:")
    st.write(answer)
