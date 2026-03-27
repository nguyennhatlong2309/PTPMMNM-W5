from langchain_community.vectorstores import FAISS
# import read_file
from langchain_community.llms import Ollama
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate

import time


class vectorStoreHandle():
    def __init__(self):
        from langchain_huggingface import HuggingFaceEmbeddings

        self.embeddings = HuggingFaceEmbeddings(model_name="./hf_models")

        try:
            self.vector_store = FAISS.load_local(
                "./demo_index",
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        except:
            self.vector_store = None

        self.retriever = None
        if self.vector_store:
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k":3})

        self.llm = Ollama(model="llama3:8b")

        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
                Bạn là trợ lý AI.

                BẮT BUỘC: Luôn trả lời bằng TIẾNG VIỆT, không được dùng tiếng Anh
                Chỉ được trả lời dựa trên tài liệu dưới đây.
                Nếu không có thông tin, hãy nói: "Tôi không biết".
                
                Tài liệu:
                {context}

                Câu hỏi:
                {question}

                Trả lời:
"""
        )

        if self.retriever:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": self.prompt}
            )

    def ask(self, question):
        if not self.vector_store:
            return "Chưa có dữ liệu"
        
        answer = self.qa_chain.invoke({"query": question})["result"]
        return answer

    def create_store(self, texts):
        chunks = self.spliter_chunks(texts)
        
        # tạo vector store
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": self.prompt}
        )
        self.saveVectorStore()

    def spliter_chunks(self, texts):
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        if isinstance(texts, str):
            texts = [texts]

        return RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        ).create_documents(texts)

    def saveVectorStore(self):
        self.vector_store.save_local("./demo_index")
        print("da luu vector store")

    def add_data(self, texts):
        chunks = self.spliter_chunks(texts)

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vector_store.add_documents(chunks)

        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff"
        )
        self.vector_store.save_local("./demo_index")


start = time.time()
VTH = vectorStoreHandle()
e1 = time.time()
docs = VTH.ask("câu chuyện kể vè nội dung gì?")
e2= time.time()
print(docs)

print(f"thoi gian khoi tao {e1-start:.2f} \nthoi gian truy xuat {e2-e1:.2f}")