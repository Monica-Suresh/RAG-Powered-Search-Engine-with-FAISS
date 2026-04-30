"""
RAG-Powered Search Engine — Streamlit Deployment
Final Capstone Project

Run with:
    streamlit run app.py
"""

import os
import time
import textwrap
import streamlit as st

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG-Powered Q&A Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar — configuration ───────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")
    use_openai = st.toggle("Use OpenAI GPT-3.5", value=False)

    if use_openai:
        openai_key = st.text_input("OpenAI API Key", type="password",
                                   placeholder="sk-...")
    else:
        openai_key = ""
        st.info("Running in FREE mode using HuggingFace models (no API key needed).")

    st.divider()
    st.subheader("RAG Parameters")
    chunk_size    = st.slider("Chunk Size (chars)",    200, 1000, 500, 50)
    chunk_overlap = st.slider("Chunk Overlap (chars)",  50,  300, 100, 25)
    top_k         = st.slider("Top-K Documents",         1,    8,   4,  1)

    st.divider()
    st.subheader("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files to add to knowledge base:",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to supplement the built-in knowledge base"
    )

    st.divider()
    st.caption("**Model:** sentence-transformers/all-MiniLM-L6-v2")
    st.caption("**Vector DB:** FAISS")
    st.caption("**Framework:** LangChain")

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🔍 RAG-Powered Search Engine")
st.markdown(
    "Ask any question about **AI/ML, NLP, Transformers, or RAG Systems**. "
    "Answers are grounded in a curated knowledge base using Retrieval-Augmented Generation."
)
st.divider()

# ── Knowledge Base definition ─────────────────────────────────────────────────
KNOWLEDGE_BASE = {
    "AI and ML Fundamentals": (
        "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines. "
        "Machine learning (ML) enables systems to learn from experience without being explicitly programmed. "
        "There are three major types of machine learning: supervised learning, unsupervised learning, "
        "and reinforcement learning. Supervised learning uses labelled data for classification and regression. "
        "Unsupervised learning finds hidden patterns in unlabelled data via clustering and dimensionality reduction. "
        "Reinforcement learning trains agents to maximise cumulative reward through environment interaction. "
        "Deep learning uses multi-layer neural networks. CNNs excel at images; Transformers dominate language. "
        "Overfitting occurs when a model performs well on training data but poorly on new data. "
        "Prevention methods include regularisation (L1/L2), dropout, early stopping, and cross-validation. "
        "Evaluation metrics: accuracy, precision, recall, F1-score, AUC-ROC for classification; "
        "MAE, MSE, R-squared for regression."
    ),
    "Natural Language Processing": (
        "NLP focuses on enabling computers to understand and generate human language. "
        "Key tasks: tokenisation, part-of-speech tagging, named entity recognition (NER), "
        "sentiment analysis, machine translation, and question answering. "
        "Word2Vec introduced skip-gram and CBOW architectures for word embeddings. "
        "GloVe factorises a word co-occurrence matrix. FastText represents words as bags of character n-grams, "
        "enabling it to handle out-of-vocabulary words. "
        "BERT uses Masked Language Modelling (bidirectional) pre-training. "
        "GPT uses causal left-to-right language modelling for text generation. "
        "GPT-3 has 175 billion parameters and shows strong few-shot capabilities. "
        "Evaluation: BLEU for translation, ROUGE for summarisation, F1 for reading comprehension."
    ),
    "Transformer Architecture": (
        "The Transformer was introduced in Attention Is All You Need (Vaswani et al., 2017). "
        "It relies entirely on self-attention mechanisms instead of recurrence or convolution. "
        "Multi-Head Self-Attention computes: Attention(Q,K,V) = softmax(Q*K_T / sqrt(d_k)) * V. "
        "The Encoder has 6 layers of multi-head self-attention and feed-forward networks. "
        "The Decoder has 6 layers with masked self-attention, cross-attention, and feed-forward. "
        "Positional Encoding adds position information using sine and cosine functions. "
        "RAG (Retrieval-Augmented Generation) was introduced by Lewis et al. (2020) at Facebook AI. "
        "RAG combines LLM generation with retrieval from an external knowledge base. "
        "It addresses LLM limitations: knowledge cutoff, hallucination, and costly updates. "
        "Embedding models for RAG: all-MiniLM-L6-v2, all-mpnet-base-v2, OpenAI text-embedding-ada-002."
    ),
    "Building Production RAG Systems": (
        "Production RAG requires ingestion, retrieval, and generation components. "
        "Ingestion pipeline: document loading, text extraction, chunking, embedding, and indexing. "
        "Embedding quality is the single most important factor in RAG performance. "
        "Dense retrieval uses embedding-based cosine similarity. "
        "Sparse retrieval uses BM25 or TF-IDF for keyword matching. "
        "Hybrid retrieval combines dense and sparse scores using Reciprocal Rank Fusion. "
        "Prompt engineering: instruct LLM to answer only from context; say I don't know if unsure. "
        "RAGAS evaluates Context Relevance, Faithfulness, and Answer Relevance. "
        "Latency optimisation: cache frequent queries; use FAISS HNSW index; stream LLM tokens. "
        "Security: sanitise queries for prompt injection; implement access control and audit logging."
    ),
}


def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file."""
    from pypdf import PdfReader
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text


@st.cache_resource(show_spinner="Building knowledge base and FAISS index...")
def build_rag_pipeline(use_openai_flag, openai_api_key, c_size, c_overlap, topk, additional_docs=None):
    """Build the full RAG pipeline. Cached so it only runs once per session."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from langchain_core.prompts import PromptTemplate
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings

    # Build documents
    raw_docs = [
        Document(
            page_content=content,
            metadata={"title": title, "source": title}
        )
        for title, content in KNOWLEDGE_BASE.items()
    ]

    # Add documents from uploaded PDFs
    if additional_docs:
        for pdf_file, pdf_text in additional_docs:
            raw_docs.append(
                Document(
                    page_content=pdf_text,
                    metadata={"title": pdf_file.name, "source": f"PDF: {pdf_file.name}"}
                )
            )

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=c_size,
        chunk_overlap=c_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(raw_docs)

    # Embeddings
    if use_openai_flag and openai_api_key:
        from langchain_openai import OpenAIEmbeddings
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=openai_api_key
        )
    else:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    # FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    retriever   = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": topk}
    )

    # Prompt
    prompt_template = PromptTemplate(
        template=(
            "You are a knowledgeable AI assistant.\n"
            "Use ONLY the context provided below to answer the question accurately and concisely.\n"
            "If the answer is not contained in the context, respond with:\n"
            "\"I don't have enough information in the provided documents to answer this question.\"\n"
            "Do NOT make up facts.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Helpful Answer:"
        ),
        input_variables=["context", "question"]
    )

    # LLM
    if use_openai_flag and openai_api_key:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=openai_api_key
        )
    else:
        from langchain_huggingface import HuggingFacePipeline
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline

        model_id  = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        hf_model  = AutoModelForCausalLM.from_pretrained(model_id)
        gen_pipe  = hf_pipeline(
            "text-generation",
            model=hf_model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            do_sample=False
        )
        llm = HuggingFacePipeline(pipeline=gen_pipe)

    # Custom RAG Chain (replaces deprecated RetrievalQA)
    class RAGChain:
        def __init__(self, llm, retriever, prompt):
            self.llm = llm
            self.retriever = retriever
            self.prompt = prompt
        
        def invoke(self, inputs):
            query = inputs.get('query', inputs.get('question', ''))
            
            # Retrieve relevant documents
            retrieved_docs = self.retriever.invoke(query)
            
            # Format context from retrieved documents
            context = '\n\n'.join([doc.page_content for doc in retrieved_docs])
            
            # Format the prompt
            formatted_prompt = self.prompt.format(context=context, question=query)
            
            # Call LLM using invoke method (works for both OpenAI and HuggingFace)
            answer = self.llm.invoke(formatted_prompt)
            
            # Return in same format as RetrievalQA
            return {
                'result': answer,
                'source_documents': retrieved_docs
            }

    # Create the RAG chain
    chain = RAGChain(
        llm=llm,
        retriever=retriever,
        prompt=prompt_template
    )

    return chain, vectorstore, chunks


# ── Build pipeline ─────────────────────────────────────────────────────────────
try:
    # Process uploaded PDFs
    additional_documents = None
    pdf_info = ""
    
    if uploaded_files:
        # Clear cache if files are uploaded to rebuild with new documents
        st.cache_resource.clear()
        additional_documents = []
        for pdf_file in uploaded_files:
            try:
                pdf_text = extract_text_from_pdf(pdf_file)
                additional_documents.append((pdf_file, pdf_text))
                pdf_info += f", {len(uploaded_files)} PDF(s) uploaded"
            except Exception as pdf_error:
                st.warning(f"Error processing {pdf_file.name}: {pdf_error}")
    
    rag_chain, vectorstore, chunks = build_rag_pipeline(
        use_openai, openai_key, chunk_size, chunk_overlap, top_k, 
        additional_docs=additional_documents
    )
    st.success(
        f"Knowledge base ready! "
        f"{len(KNOWLEDGE_BASE) + (len(uploaded_files) if uploaded_files else 0)} documents | {len(chunks)} chunks | "
        f"{'OpenAI' if use_openai else 'HuggingFace'} mode{pdf_info}"
    )
except Exception as e:
    st.error(f"Error building pipeline: {e}")
    st.stop()

# ── Sample questions ───────────────────────────────────────────────────────────
st.subheader("💡 Try these sample questions")
col1, col2, col3, col4 = st.columns(4)

sample_qs = {
    "Types of ML":           "What are the three major types of machine learning?",
    "BERT vs GPT":           "How is BERT different from GPT in training objective?",
    "Transformer formula":   "What is the Scaled Dot-Product Attention formula?",
    "RAG limitations":       "What limitations of LLMs does RAG address?",
}

clicked_q = None
for col, (label, question) in zip([col1, col2, col3, col4], sample_qs.items()):
    if col.button(label, use_container_width=True):
        clicked_q = question

# ── Q&A input ─────────────────────────────────────────────────────────────────
st.subheader("❓ Ask a Question")

# Two-column layout for better interactivity
col_input, col_stats = st.columns([3, 1])

with col_input:
    user_query = st.text_input(
        "Your question:",
        value=clicked_q if clicked_q else "",
        placeholder="e.g. What is overfitting and how can it be prevented?",
    )

with col_stats:
    st.metric("Documents", len(KNOWLEDGE_BASE) + (len(uploaded_files) if uploaded_files else 0))
    st.metric("Chunks", len(chunks))

ask_btn = st.button("🔍  Search & Answer", type="primary", use_container_width=True)

# ── Process query ──────────────────────────────────────────────────────────────
if ask_btn and user_query.strip():
    with st.spinner("Retrieving relevant context and generating answer..."):
        start  = time.time()
        result = rag_chain.invoke({"query": user_query})
        elapsed = time.time() - start

    answer   = result["result"].strip()
    src_docs = result["source_documents"]
    sources  = list(dict.fromkeys(d.metadata["title"] for d in src_docs))

    # Answer box with better styling
    st.divider()
    st.subheader("💬 Answer")
    
    # Use columns for better layout
    col_answer, col_metrics = st.columns([3, 1])
    
    with col_answer:
        st.markdown(f"> {answer}")
    
    with col_metrics:
        st.metric("Response Time", f"{elapsed:.2f}s")
        st.metric("Documents Retrieved", len(src_docs))

    # Interactive sources display
    st.subheader("📚 Retrieved Sources")
    source_cols = st.columns(min(len(sources), 4))
    for col, s in zip(source_cols, sources):
        with col:
            st.info(f"📄 {s}")

    # Retrieved chunks with enhanced interactivity
    st.subheader("📋 Document Chunks")
    
    # Tabs for each retrieved chunk
    if src_docs:
        chunk_tabs = st.tabs([f"Chunk {i+1}" for i in range(len(src_docs))])
        
        for chunk_tab, doc in zip(chunk_tabs, src_docs):
            with chunk_tab:
                st.markdown(f"**Source:** {doc.metadata['title']}")
                st.markdown(f"**Content:**")
                st.text_area(
                    f"Content preview",
                    value=textwrap.fill(doc.page_content, width=90),
                    height=250,
                    disabled=True,
                    label_visibility="collapsed"
                )
                
                # Copy button for convenience
                if st.button("📋 Copy to clipboard", key=f"copy_{id(doc)}"):
                    st.write("✅ Content copied! (Note: Use browser developer tools)")

elif ask_btn:
    st.warning("Please enter a question before clicking Search.")

# ── Knowledge Base Explorer ────────────────────────────────────────────────────
st.divider()
st.subheader("📖 Knowledge Base Explorer")

# Tabs for built-in and uploaded documents
tab1, tab2 = st.tabs(["Built-in Knowledge Base", "Uploaded Documents"])

with tab1:
    with st.expander("Browse the built-in knowledge base documents"):
        for title, content in KNOWLEDGE_BASE.items():
            st.markdown(f"### {title}")
            st.markdown(content)
            st.divider()

with tab2:
    if uploaded_files:
        st.info(f"✅ {len(uploaded_files)} PDF(s) successfully loaded into the knowledge base")
        for pdf_file in uploaded_files:
            with st.expander(f"📄 {pdf_file.name}"):
                try:
                    pdf_text = extract_text_from_pdf(pdf_file)
                    st.text_area(
                        f"Content of {pdf_file.name}",
                        value=pdf_text[:1000] + "..." if len(pdf_text) > 1000 else pdf_text,
                        height=300,
                        disabled=True
                    )
                except Exception as e:
                    st.error(f"Error reading file: {e}")
    else:
        st.info("📁 No PDF files uploaded yet. Upload PDFs in the sidebar to add custom documents to your knowledge base.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "RAG-Powered Search Engine | Final Capstone Project | "
    "LangChain + FAISS + HuggingFace"
)
