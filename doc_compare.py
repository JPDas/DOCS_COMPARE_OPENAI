import os
import yaml
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.example_selector.semantic_similarity import (
    SemanticSimilarityExampleSelector,
)
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores import Chroma


os.environ["OPENAI_API_KEY"] = 'sk-'
class DocumentsCompare:
    def __init__(self, file1, file2) -> None:
        self.upload_file1 = file1
        self.upload_file2 = file2
        self.chunk_size = 1500
        self.max_tokens = 8191
        self.top_k = 250
        self.top_p = 0.5

        self.model_type = 'gpt3.5-turbo'
        self.llm_model = None
        self.generic_samples = None
        self.example_selector = None
        
    def preprocess(self):
        # using PyPDF PdfReader to read in the first PDF file as text
        document_1 = PdfReader(self.upload_file1)
        # using PyPDF PdfReader to read in the second PDF file as text
        document_2 = PdfReader(self.upload_file2)

        # creating an empty string for us to append all the text extracted from the first PDF
        doc_1_text = ""
        # creating an empty string for us to append all the text extracted from the second PDF
        doc_2_text = ""
        # a simple for loop to iterate through all pages of both PDFs we uploaded
        for (page_1, page_2) in zip(document_1.pages, document_2.pages):
            # as we loop through each page, we extract the text from the page and append it to the "text" string for both
            # documents
            doc_1_text += page_1.extract_text() + "\n"
            doc_2_text += page_2.extract_text() + "\n"

        #Todo:Check chunking of text
        
        #Todo:Cleaning text

        return doc_1_text, doc_2_text   
    def prompt_finder(self, question):

        with open("Examples/sample_prompt_data.yaml", "r") as stream:
            # storing the sample files in the generic samples variable we initialized
            self.generic_samples = yaml.safe_load(stream)

        self.example_selector = SemanticSimilarityExampleSelector.from_examples(
            # The list of examples available to select from.
            self.generic_samples,
            # The embedding class used to produce embeddings which are used to measure semantic similarity.
            OpenAIEmbeddings(),
            # The VectorStore class that is used to store the embeddings and do a similarity search over.
            Chroma,
            # The number of examples to produce.
            k=1,
        )

        # This is formatting the prompts that are retrieved from the sample_prompts/sample_prompt_data.yaml file
        example_prompt = PromptTemplate(input_variables=["prompt", "assistant"],
                                        template="\n\nHuman: {prompt} \n\nAssistant: "
                                                "{assistant}")
        # This is orchestrating the example selector (finding similar prompts to the question), and example_prompt (formatting
        # the retrieved prompts, and the users request with both uploaded documents to do the comparison on
        prompt = FewShotPromptTemplate(
            example_selector=self.example_selector,
            example_prompt=example_prompt,
            suffix="{input}",
            input_variables=["input"]
        )

         # This is calling the prompt method and passing in the users question to create the final multi-shot prompt,
        # with the semantically similar prompts
        question_with_prompt = prompt.format(input=question)
        # TODO: If you want to see the semantically selected prompts, print them into the console:
        return question_with_prompt
         
    def build_prompt(self, doc_1_text, doc_2_text):
        prompt = f"""\n\nHuman: Please thoroughly analyze and compare Document A and Document B, highlighting the 
        location of every single change. Provide a detailed report that includes the textual alterations, deletions, 
        insertions, and any other modifications between the two documents. Ensure that the report not only lists the 
        changes but also pinpoints where each change occurs in both documents.

                Document A: {doc_1_text}

                Document B: {doc_2_text}

                \n\nAssistant:"""
        
        return prompt
    def llm_compare(self, prompt):
        llm = OpenAI()

        response = llm.predict(prompt)
        return response
