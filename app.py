import json  # Added missing import
from langchain_aws import BedrockEmbeddings, ChatBedrock
import boto3
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Configuration
AWS_REGION = "us-east-1"
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
LLM_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

# Initialize Bedrock clients
bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name=AWS_REGION)
bedrock_embeddings = BedrockEmbeddings(
    client=bedrock_runtime,
    model_id=EMBED_MODEL_ID
)

def test_embedding_access():
    """Verify Bedrock access before proceeding"""
    try:
        response = bedrock_runtime.invoke_model(
            modelId=EMBED_MODEL_ID,
            body=json.dumps({"inputText": "test"}))
        return True

    except Exception as e:  # Added missing except clause
        st.error(f"Access Denied: {str(e)}")
        return False
        

def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    return text_splitter.split_documents(documents)

def get_vector_store(docs):
    vectorstore = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

prompt_template = """
Human: Answer the question using only the provided context. 
Provide a comprehensive 300-word response. If unsure, say "I don't know".

<context>
{context}
</context>

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
)

def get_qa_chain():
    vectorstore = FAISS.load_local(
        "faiss_index",
        bedrock_embeddings,
        allow_dangerous_deserialization=True
    )
    
    llm = ChatBedrock(
        client=bedrock_runtime,
        model_id=LLM_MODEL_ID,
        model_kwargs={'max_tokens': 512}
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

def main():
    st.set_page_config("PDF Chat with Titan Embeddings")
    st.header("Document Chat Assistant")
    
    if not test_embedding_access():
        st.stop()
    
    with st.sidebar:
        st.subheader("Document Management")
        if st.button("Update Vector Store"):
            with st.spinner("Processing documents..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Vector store updated!")
    
    query = st.text_input("Ask about your documents:")
    
    if query:
        with st.spinner("Analyzing documents..."):
            try:
                qa_chain = get_qa_chain()
                result = qa_chain.invoke({"query": query})
                st.write("### Answer")
                st.write(result['result'])
                
                with st.expander("See source documents"):
                    for doc in result['source_documents']:
                        st.write(f"Page {doc.metadata['page']}: {doc.page_content[:300]}...")
                        
            except Exception as e:
                st.error(f"Query failed: {str(e)}")

if __name__ == "__main__":
    main()