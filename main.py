from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_huggingface.llms import HuggingFaceEndpoint
import pinecone
from dotenv import load_dotenv
import os

class ChatBot():
    load_dotenv()
    
    # Load and split documents
    loader = TextLoader('./data.txt')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
    docs = text_splitter.split_documents(documents)

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings()

    # Initialize Pinecone
    pinecone.init(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment='gcp-starter'
    )

    index_name = "langchain-demo"

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, metric="cosine", dimension=768)
        docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    else:
        docsearch = Pinecone.from_existing_index(index_name, embeddings)

    # Use HuggingFaceEndpoint instead of HuggingFaceHub
    endpoint_url = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm = HuggingFaceEndpoint(
        endpoint_url=endpoint_url,
        huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY'),
        temperature=0.8,
        top_p=0.8,
        top_k=50
    )

    from langchain import PromptTemplate

    template = """
    You are an imoot advisor. The user needs to complete his tax return for 2024. You want to help him 

    Context: {context}
    Question: {question}
    Answer: 
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    from langchain.schema.runnable import RunnablePassthrough
    from langchain.schema.output_parser import StrOutputParser

    rag_chain = (
        {"context": docsearch.as_retriever(),  "question": RunnablePassthrough()} 
        | prompt 
        | llm
        | StrOutputParser() 
    )
