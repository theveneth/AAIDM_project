from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_huggingface.llms import HuggingFaceEndpoint
import pinecone
from dotenv import load_dotenv
import os

class ChatBot:
    def __init__(self):
        load_dotenv()

        # Load and split documents
        data_dir = './data/'
        documents = []
        for file in os.listdir(data_dir):
            if file.endswith('.txt') and not file.endswith('_ref.txt'):
                with open(os.path.join(data_dir, file), 'r') as doc_file:
                    doc_text = doc_file.read()
                ref_file = file.replace('.txt', '_ref.txt')
                with open(os.path.join(data_dir, ref_file), 'r') as ref_file:
                    reference = ref_file.read().strip()
                documents.append({"page_content": doc_text, "reference": reference})

        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=4)
        docs = []
        for document in documents:
            text_chunks = text_splitter.split_text(document["page_content"])
            for chunk in text_chunks:
                docs.append(Document(page_content=chunk, metadata={"reference": document["reference"]}))
        #print(docs)
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings()

        # Initialize Pinecone
        pinecone.init(
            api_key=os.getenv('PINECONE_API_KEY'),
            environment='gcp-starter'
        )

        index_name = "tax-return-helper-bot"

        if index_name not in pinecone.list_indexes():
            pinecone.create_index(name=index_name, metric="cosine", dimension=768)
            self.docsearch = Pinecone.from_documents(docs, embedding = embeddings, index_name=index_name)
        else:
            self.docsearch = Pinecone.from_existing_index(index_name, embeddings)

        # Use HuggingFaceEndpoint instead of HuggingFaceHub
        endpoint_url = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.llm = HuggingFaceEndpoint(
            endpoint_url=endpoint_url,
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY'),
            temperature=0.1,  
            top_p=0.5,       
            top_k=40,        
            repetition_penalty=1.2  
        )

        from langchain_core.prompts import PromptTemplate
        
        template = """
        Tu es un conseiller fiscal. L'utilisateur doit remplir sa déclaration de revenus française pour 2024. 
        Tu dois répondre aux questions en français. Ne pas faire d'encart et garder le texte sur le même plan.

        Context: {context}
        Question: {question}
        Answer: 
        """

        self.prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        from langchain.schema.runnable import RunnablePassthrough
        from langchain.schema.output_parser import StrOutputParser

        class CustomOutputParser(StrOutputParser):
            def parse(self, output: str) -> str:
                return output.strip()

        # Custom chain to include references
        class CustomChain:
            def __init__(self, retriever, llm, prompt):
                self.retriever = retriever
                self.llm = llm
                self.prompt = prompt
            
            def invoke(self, question):
                context_docs = self.retriever.get_relevant_documents(question)
                if not context_docs:
                    #return "I\'m sorry, I couldn't find any relevant information to answer your question."
                    return "Je suis désolé, je n'ai pas trouvé de réponse à votre question"
                
                context = context_docs[0].page_content
                reference = context_docs[0].metadata.get("reference", "NAucune référence disponible")

                prompt_input = {
                    "context": context,
                    "question": question,
                }

                text_return = self.llm(self.prompt.format(**prompt_input))
                txt = str(text_return) + ' \n Reference: ' + str(reference)
                print(txt)
                return txt
        
        self.rag_chain = CustomChain(self.docsearch.as_retriever(), self.llm, self.prompt)

        # Debugging: Log initialization
        print("ChatBot initialized successfully")

