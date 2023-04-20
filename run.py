import os, sys
import argparse
from data_initializer import DataInitializer
from roberta_controller import RobertaController
from faiss_comparison import FaissComparison
import json


# example: python train_run.py keyword temp_keyword
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Query details and company name.')
    parser.add_argument('--mode', type=str, default='query')
    parser.add_argument('--query', type=str, default= None)
    parser.add_argument('--company', type=str, default=None)
    parser.add_argument('--single_domain', type=str, default=None)
    parser.add_argument('--meta_dir', type=str, default=None, help='where you store your company meta files')
    parser.add_argument('--processed_data_dir', type=str, default=None, help='where we find processed company files')
    parser.add_argument('--output_json', type=str, default=None, help='where you want to store preprocessed company documents')
    parser.add_argument('--answer_json', type=str, default=None, help='where you want to store answers to query')


    args = parser.parse_args()

    if args.mode == "preprocess":
        roberta_controller = RobertaController(query_mode=False)
        output_json = args.output_json
        company_database = {}
        roberta_base_tokenizer = roberta_controller.roberta_base_tokenizer
        roberta_base_model = roberta_controller.roberta_base_model
        data_initializer = DataInitializer(args.meta_dir, roberta_base_tokenizer, roberta_base_model)
        company_database["structured_doc_to_text"] = data_initializer.structured_doc_to_text
        company_database["all_documents"] = data_initializer.all_documents
        company_database["all_documents_text"] = data_initializer.all_documents_text
        company_database["all_document_abstracts"] = data_initializer.all_document_abstract_embeddings
        with open(output_json, "w") as outfile:
            json.dump(company_database, outfile)


    if args.mode == "query":
        query = args.query
        processed_data_dir = args.processed_data_dir
        answer_json = args.answer_json
        roberta_controller = RobertaController(query_mode=True)
        query_embedding = roberta_controller.embed_text(query)


        with open(processed_data_dir) as json_file:
            company_data = json.load(json_file)
        faiss_comparison = FaissComparison(company_data["all_document_abstracts"])
        scores, retrieved_examples = faiss_comparison.get_k(query_embedding, 15)
        

        full_text_embeddings = {}
        all_documents_text = company_data["all_documents_text"] 
        
        for document_id in retrieved_examples:
            full_text = all_documents_text[document_id]
            full_text_embedding = roberta_controller.embed_text(full_text)
            full_text_embeddings[document_id] = full_text_embedding

        faiss_comparison = FaissComparison(full_text_embeddings)
        full_scores, full_retrieved_examples = faiss_comparison.get_k(query_embedding, 3)

        answers = {}

        for document_id in full_retrieved_examples:
            document_context = all_documents_text[document_id]
            answer = roberta_controller.answer(query, document_context) 
            answers.append({'answer': answer, 'source': document_id})

        with open(answer_json, "w") as outfile:
            json.dump(answers, outfile)

        



