import os
from dotenv import load_dotenv
load_dotenv()
from src.llm.run_chain import TextAnalyzerChain
test = TextAnalyzerChain()
text=input("Enter the text to analyze : ")
test.output(text)
