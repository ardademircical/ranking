from transformers import AutoTokenizer, RobertaModel, RobertaForQuestionAnswering, pipeline
import torch

class RobertaController:

    """
    Wrapper for RoBERTa models, 
    We use RoBERTa Base for embedding queries and document texts, RoBERTaForQuestionAnswering for QA finetuned on SQUAD2.
    
    """
    
    def __init__(self, query_mode=False):
        self.roberta_base_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.roberta_base_model = RobertaModel.from_pretrained("roberta-base")
        if self.query_mode:
            self.roberta_qa_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
            self.roberta_qa_model = RobertaForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
            self.qa_pipeline = pipeline('question-answering', model=self.roberta_qa_model, tokenizer=self.roberta_qa_tokenizer)

    def embed_text(self, text):
        inputs = self.roberta_base_tokenizer(text, return_tensors="pt")
        outputs = self.roberta_base_model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states

    def answer(self, query, document_context):
        qa_input = {'question': query, 'context': document_context}
        return self.qa_pipeline(qa_input)






