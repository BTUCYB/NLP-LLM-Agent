import os
import csv
import time
import torch
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.document_loaders import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Deepseek
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

load_dotenv()

class InsiderThreatAnalyzer:
    def __init__(self):
        self.knowledge_base = self._create_knowledge_base()
        self.llm = self._initialize_deepseek()
        self.qa_chain = self._create_analysis_chain()
        self.thinking_animation_running = False

    def _create_knowledge_base(self):
        # Load CSV knowledge base
        loader = CSVLoader(
            file_path="./knowledge_base/insider_threat_data.csv",
            csv_args={
                "delimiter": ",",
                "fieldnames": ["user_id", "behavior_pattern", "risk_level", "timestamp", "description"]
            }
        )
        documents = loader.load()

        # Process CSV data
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=256
        )
        texts = text_splitter.split_documents(documents)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        return FAISS.from_documents(texts, embeddings)

    def _initialize_deepseek(self):
        return Deepseek(
            model="deepseek-ai/deepseek-coder-1.3b-instruct",
            temperature=0.3,
            max_tokens=1024,
            thinking_animation=self._show_thinking_animation
        )

    def _create_analysis_chain(self):
        prompt_template = """[Insider Threat Analysis Protocol]
        
        Context: {context}
        
        Question: {question}
        
        Thinking Process:
        1. Analyze user behavior patterns
        2. Cross-reference with historical incidents
        3. Evaluate data access anomalies
        4. Consider user privileges
        5. Assess compliance violations
        
        Required Output Format:
        THINKING: <step-by-step reasoning>
        ANSWER: <final risk assessment with confidence score>
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.knowledge_base.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

    def _show_thinking_animation(self):
        frames = ["Analyzing patterns", "Cross-referencing logs", "Evaluating risks"]
        for frame in frames:
            print(f"\r{frame}", end="", flush=True)
            time.sleep(1)
        print("\r", end="", flush=True)

    def analyze(self, question):
        try:
            print("\n[Initializing Threat Analysis]")
            result = self.qa_chain({"query": question})
            return self._process_response(result)
        except Exception as e:
            return self._fallback_response(question)

    def _process_response(self, result):
        response_text = result["result"]
        
        # Split thinking and answer
        thinking = ""
        answer = ""
        if "THINKING:" in response_text:
            parts = response_text.split("ANSWER:")
            thinking = parts[0].replace("THINKING:", "").strip()
            answer = parts[1].strip() if len(parts) > 1 else ""
        else:
            answer = response_text
        
        return {
            "thinking": thinking,
            "answer": answer,
            "sources": list(set([doc.metadata["source"] for doc in result["source_documents"]]))
        }

    def _fallback_response(self, question):
        prompt = f"""As a cybersecurity expert, analyze this potential insider threat scenario:
        {question}
        
        Provide detailed reasoning considering:
        - Common insider threat indicators
        - Behavioral analytics
        - Data protection best practices
        
        Format your response with:
        THINKING: <your analysis process>
        ANSWER: <your conclusion>"""
        
        response = self.llm.generate([prompt])
        return self._process_response({"result": response.generations[0][0].text})

if __name__ == "__main__":
    analyzer = InsiderThreatAnalyzer()
    
    questions = [
        "User ID 456 accessed sensitive financial records multiple times after working hours",
        "How would you detect unauthorized database exports by a privileged user?",
        "Developer downloaded entire customer database to personal device"
    ]
    
    for question in questions:
        print(f"\n\033[1mQuery:\033[0m {question}")
        result = analyzer.analyze(question)
        
        print("\n\033[1mThinking Process:\033[0m")
        print(result["thinking"])
        
        print("\n\033[1mRisk Assessment:\033[0m")
        print(result["answer"])
        
        if result["sources"]:
            print("\n\033[1mReference Sources:\033[0m")
            print("\n".join(result["sources"]))
        
        print("\n" + "="*80 + "\n")