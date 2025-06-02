import os
import tempfile
import time
from typing import Dict, List, Any, Optional
import boto3
import streamlit as st
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dataclasses import dataclass


AWS_REGION = "us-east-1"
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
CLAUDE_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
LLAMA_MODEL_ID = "meta.llama3-70b-instruct-v1:0"

# Initialize Bedrock client
@st.cache_resource
def get_bedrock_client():
    return boto3.client(service_name='bedrock-runtime', region_name=AWS_REGION)

@st.cache_resource
def get_bedrock_embeddings():
    return BedrockEmbeddings(client=get_bedrock_client(), model_id=EMBED_MODEL_ID)

@dataclass
class AgentResponse:
    content: str
    metadata: Dict[str, Any]
    processing_time: float

class DocumentProcessor:
    """Handles document upload and processing"""
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def process_uploaded_files(self, uploaded_files) -> Optional[FAISS]:
        """Process uploaded PDF files and create vector store"""
        if not uploaded_files:
            return None
        
        all_documents = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for uploaded_file in uploaded_files:
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    loader = PyPDFLoader(temp_file_path)
                    documents = loader.load()
                    
                    # Add metadata
                    for doc in documents:
                        doc.metadata['source_file'] = uploaded_file.name
                        doc.metadata['file_size'] = len(uploaded_file.getbuffer())
                    
                    all_documents.extend(documents)
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    return None
        
        if not all_documents:
            return None
        
        # Split documents
        split_docs = self.text_splitter.split_documents(all_documents)
        
        # Create vector store
        vectorstore = FAISS.from_documents(split_docs, self.embeddings)
        vectorstore.save_local("faiss_index")
        
        return vectorstore

class RetrievalAgent:
    """Specialized agent for document retrieval"""
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.name = "Retrieval Agent"
    
    def retrieve(self, query: str, k: int = 5) -> AgentResponse:
        start_time = time.time()
        
        try:
            vectorstore = FAISS.load_local(
                "faiss_index", 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            docs = vectorstore.similarity_search_with_score(query, k=k)
            
            retrieved_docs = []
            for doc, score in docs:
                retrieved_docs.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'relevance_score': float(score)
                })
            
            processing_time = time.time() - start_time
            
            return AgentResponse(
                content=retrieved_docs,
                metadata={
                    'documents_found': len(retrieved_docs),
                    'avg_relevance': sum(d['relevance_score'] for d in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            return AgentResponse(
                content=[],
                metadata={'error': str(e)},
                processing_time=time.time() - start_time
            )

class AnalysisAgent:
    """Specialized agent for content analysis"""
    
    def __init__(self, bedrock_client):
        self.client = bedrock_client
        self.llm = ChatBedrock(
            client=bedrock_client,
            model_id=CLAUDE_MODEL_ID,
            model_kwargs={'max_tokens': 1000}
        )
        self.name = "Analysis Agent"
    
    def analyze(self, query: str, retrieved_docs: List[Dict]) -> AgentResponse:
        start_time = time.time()
        
        if not retrieved_docs:
            return AgentResponse(
                content="No documents available for analysis.",
                metadata={'error': 'No documents provided'},
                processing_time=time.time() - start_time
            )
        
        # Prepare context from top documents
        context = "\n\n".join([doc['content'] for doc in retrieved_docs[:3]])
        
        prompt = f"""You are an expert document analyst. Analyze the following documents to answer the user's question comprehensively.

Question: {query}

Documents:
{context}

Provide a detailed, accurate analysis that:
1. Directly answers the question
2. Cites specific information from the documents
3. Highlights key insights
4. Notes any limitations in the available information

Response:"""
        
        try:
            response = self.llm.invoke(prompt)
            processing_time = time.time() - start_time
            
            return AgentResponse(
                content=response.content,
                metadata={
                    'documents_analyzed': len(retrieved_docs[:3]),
                    'context_length': len(context)
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            return AgentResponse(
                content=f"Analysis failed: {str(e)}",
                metadata={'error': str(e)},
                processing_time=time.time() - start_time
            )

class ValidationAgent:
    """Specialized agent for response validation"""
    
    def __init__(self, bedrock_client):
        self.client = bedrock_client
        self.llm = ChatBedrock(
            client=bedrock_client,
            model_id=LLAMA_MODEL_ID,
            model_kwargs={'max_tokens': 500}
        )
        self.name = "Validation Agent"
    
    def validate(self, query: str, analysis: str, source_docs: List[Dict]) -> AgentResponse:
        start_time = time.time()
        
        context = "\n".join([doc['content'][:500] for doc in source_docs[:2]])
        
        prompt = f"""Review the following analysis for accuracy and completeness against the source documents.

Original Question: {query}

Analysis to Validate:
{analysis}

Source Documents:
{context}

Provide validation feedback:
1. Accuracy score (1-10)
2. Any factual inconsistencies
3. Missing important information
4. Overall confidence level

Validation:"""
        
        try:
            response = self.llm.invoke(prompt)
            processing_time = time.time() - start_time
            
            return AgentResponse(
                content=response.content,
                metadata={
                    'validation_completed': True,
                    'sources_checked': len(source_docs[:2])
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            return AgentResponse(
                content=f"Validation failed: {str(e)}",
                metadata={'error': str(e), 'validation_completed': False},
                processing_time=time.time() - start_time
            )

class MultiAgentOrchestrator:
    """Orchestrates the multi-agent workflow"""
    
    def __init__(self):
        self.bedrock_client = get_bedrock_client()
        self.embeddings = get_bedrock_embeddings()
        
        self.retrieval_agent = RetrievalAgent(self.embeddings)
        self.analysis_agent = AnalysisAgent(self.bedrock_client)
        self.validation_agent = ValidationAgent(self.bedrock_client)
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process query through the multi-agent pipeline"""
        
        # Step 1: Retrieve relevant documents
        retrieval_result = self.retrieval_agent.retrieve(query)
        
        if not retrieval_result.content:
            return {
                'success': False,
                'error': 'No documents found or retrieval failed',
                'retrieval_result': retrieval_result
            }
        
        # Step 2: Analyze documents
        analysis_result = self.analysis_agent.analyze(query, retrieval_result.content)
        
        # Step 3: Validate analysis
        validation_result = self.validation_agent.validate(
            query, analysis_result.content, retrieval_result.content
        )
        
        total_processing_time = (
            retrieval_result.processing_time + 
            analysis_result.processing_time + 
            validation_result.processing_time
        )
        
        return {
            'success': True,
            'query': query,
            'retrieval_result': retrieval_result,
            'analysis_result': analysis_result,
            'validation_result': validation_result,
            'total_processing_time': total_processing_time,
            'documents_processed': retrieval_result.metadata.get('documents_found', 0)
        }

def init_session_state():
    """Initialize session state variables"""
    if 'vector_store_ready' not in st.session_state:
        st.session_state.vector_store_ready = False
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

def check_vector_store():
    """Check if vector store exists"""
    try:
        embeddings = get_bedrock_embeddings()
        FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        return True
    except:
        return False

def main():
    st.set_page_config(
        page_title="DocuMind AI",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .status-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    .metric-card {
        background: black;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    .agent-status {
        display: flex;
        align-items: center;
        padding: 0.5rem;
        margin: 0.25rem 0;
        background: black;
        border-radius: 0.25rem;
    }
    
    /* Cool Footer Styles */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: black
        color: white;
        text-align: center;
        padding: 12px 0;
        font-family: 'Arial', sans-serif;
        font-weight: 600;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        z-index: 999;
        animation: pulse 2s infinite;
    }
    
    .footer-content {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 8px;
    }
    
    .heart {
        color: #ff6b6b;
        font-size: 1.2em;
        animation: heartbeat 1.5s ease-in-out infinite;
    }
    
    .footer-text {
        font-size: 14px;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3);
    }
    
    @keyframes heartbeat {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 -2px 10px rgba(0,0,0,0.1); }
        50% { box-shadow: 0 -4px 20px rgba(102, 126, 234, 0.3); }
        100% { box-shadow: 0 -2px 10px rgba(0,0,0,0.1); }
    }
    
    /* Add bottom padding to main content to avoid footer overlap */
    .main .block-container {
        padding-bottom: 80px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🧠 DocuMind AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Intelligent Multi-Agent Document Analysis System</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📄 Document Management")
        
        # File Upload
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Select one or more PDF files to analyze"
        )
        
        if uploaded_files:
            st.info(f"📎 {len(uploaded_files)} file(s) selected")
            
            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                with st.spinner("Processing documents..."):
                    processor = DocumentProcessor(get_bedrock_embeddings())
                    vector_store = processor.process_uploaded_files(uploaded_files)
                    
                    if vector_store:
                        st.session_state.vector_store_ready = True
                        st.session_state.processed_files = [f.name for f in uploaded_files]
                        st.success("✅ Documents processed successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to process documents")
        
        st.markdown("---")
        
        # System Status
        st.markdown("### System Status")
        
        vector_store_status = check_vector_store()
        st.session_state.vector_store_ready = vector_store_status
        
        if vector_store_status:
            st.markdown('<div class="agent-status">✅ Vector Store Ready</div>', unsafe_allow_html=True)
            st.markdown('<div class="agent-status">🤖 Agents Online</div>', unsafe_allow_html=True)
            st.markdown('<div class="agent-status">🔥Upload Document using Browser File or Drag n Drop📍</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="color: #ef4444;">❌ No documents loaded</div>', unsafe_allow_html=True)
        
        if st.session_state.processed_files:
            st.markdown("### 📋 Processed Files")
            for file_name in st.session_state.processed_files:
                st.markdown(f"• {file_name}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 Ask Questions")
        
        # Query input
        query = st.text_input(
            "Enter your question:",
            placeholder="What are the key findings in the document?",
            disabled=not st.session_state.vector_store_ready
        )
        
        # Process query
        if st.button("🔍 Analyze", disabled=not query or not st.session_state.vector_store_ready):
            if query:
                orchestrator = MultiAgentOrchestrator()
                
                with st.spinner("🤖 Multi-agent processing in progress..."):
                    # Progress indicators
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("🔍 Retrieving relevant documents...")
                    progress_bar.progress(25)
                    
                    result = orchestrator.process_query(query)
                    
                    status_text.text("🧠 Analyzing content...")
                    progress_bar.progress(50)
                    time.sleep(0.5)  # Brief pause for UI
                    
                    status_text.text("✅ Validating response...")
                    progress_bar.progress(75)
                    time.sleep(0.5)
                    
                    status_text.text("📝 Finalizing response...")
                    progress_bar.progress(100)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                
                if result['success']:
                    # Display main response
                    st.markdown("### 🎯 Analysis Result")
                    st.markdown(result['analysis_result'].content)
                    
                    # Expandable sections for additional details
                    with st.expander("🔍 Validation & Quality Check"):
                        st.markdown("**Validation Results:**")
                        st.write(result['validation_result'].content)
                    
                    with st.expander("📊 Processing Details"):
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Documents Found", result['documents_processed'])
                        with col_b:
                            st.metric("Processing Time", f"{result['total_processing_time']:.2f}s")
                        with col_c:
                            st.metric("Agents Used", "3")
                    
                    # Add to history
                    st.session_state.query_history.append({
                        'query': query,
                        'response': result['analysis_result'].content,
                        'timestamp': time.time()
                    })
                
                else:
                    st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
        
        # Display helpful tips when no documents are loaded
        if not st.session_state.vector_store_ready:
            st.info("👆 Please upload PDF documents using the sidebar to get started")
    
    with col2:
        st.markdown("### 📈 Quick Stats")
        
        if st.session_state.vector_store_ready:
            st.markdown('<div class="metric-card"><h3>✅</h3><p>System Ready</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card"><h3>⏳</h3><p>Awaiting Documents</p></div>', unsafe_allow_html=True)
        
        st.markdown(f'<div class="metric-card"><h3>{len(st.session_state.query_history)}</h3><p>Queries Processed</p></div>', unsafe_allow_html=True)
        
        # Recent queries
        if st.session_state.query_history:
            st.markdown("### 📝 Recent Queries")
            for query_data in st.session_state.query_history[-3:]:
                with st.expander(f"Q: {query_data['query'][:50]}..."):
                    st.write(query_data['response'][:200] + "...")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <div class="footer-content">
            <span class="footer-text">Made by Jay Zalani with</span>
            <span class="heart">❤️</span>
            <span class="footer-text">Love</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()