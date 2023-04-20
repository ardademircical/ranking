import os
import json
from collections import defaultdict
import numpy as np
import pandas as pd
from transformers import LayoutLMConfig, LayoutLMModel, T5Tokenizer, T5ForConditionalGeneration, RobertaModel, RobertaTokenizer


class DataInitializer:
   """ 
   Preprocessing of user-inputed documents and generating abstracts of documents for summarization purposes. 
   Outputs stored in a database. 
   For this exercise, I treat the database as a dictionary. 
   """
   
   def __init__(self, meta_dir, roberta_controller) -> None:
        self.METADATA_DIR = meta_dir
        self.structured_doc_to_text = {}
        self.all_documents = {}
        self.all_documents_text = {}
        self.all_document_abstracts = {}
        self.all_document_abstract_embeddings = {}
 
        # initialize LayoutLM for documents that require structure detection and translation to text
        configuration = LayoutLMConfig()
        self.layout_processor = LayoutLMModel(configuration)
        self.layout_configuration = self.layout_processor.config

        self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')

        self.roberta_controller = roberta_controller

   def summarize_document(self, document_id: str, document_text: str):
       # summarize documents using T5   
       prompted_text = "summarize: " + document_text
       tokenized_text = self.t5_tokenizer(prompted_text, return_tensors = "pt")
       summary_processed = self.t5_model.generate(tokenized_text)
       output = self.t5_tokenizer.decode(summary_processed[0])
       return output

   def process_layout(self, document_id: str, document: object):
       # structured document translated to textual format    
       output = self.layout_processor(document)
       self.structured_docs[document_id] = document
       self.structured_doc_to_text[document_id] = output

   def embed_document_abstract(self, document_text):
       return self.roberta_controller.embed_text(document_text)

   def process_data(self):     
        # bulk of the preprocessing
        for metadata_file in os.listdir(self.METADATA_DIR):
            with open(os.path.join(self.METADATA_DIR, metadata_file)) as f_meta:
                for line in f_meta:
                    metadata_dict = json.loads(line)
                    document_id = metadata_dict['document_id']
                    document_type = metadata_dict['document_type']
                    document = metadata_dict['document']
                    meta_data_text = metadata_dict['meta_data']
                    
                    if document_type in ["pdf, table, slides"]:
                        self.process_layout(document_id, document)
                        document_text = self.structured_doc_to_text[document_id]
                    else:
                        document_text = metadata_dict["document_text"]
                    
                    # create abstract of document with T5 for summarization
                    document_abstract = meta_data_text + self.summarize_document(document_id, document_text)
                    self.all_document_abstracts[document_id] = document_abstract
    
                    # embed document abstract
                    self.all_document_abstract_embeddings[document_id] = self.embed_document_abstract(document_abstract)

                    self.all_documents[document_id] = document
                    self.all_documents_text[document_id] = document_text

                    