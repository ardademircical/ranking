Below, I break down my take on how to structure our textual search engine into multiple steps, explaining how each component functions and possible areas of improvement

Our query architecture would draw similarities to applications like Elicit.ai and Perplexity.ai. While our retrieval techniques later on will be based on LLMs similar to these apps, we also have to deal with efficiently preprocessing user-inputed documents as we do not rely on a traditional search engine or knowledge graph API in the earlier steps. 

STEP 1: Parse User-Provided Documents Depending On Structure

This step probably requires the most flexibility as we will have to deal with multiple forms of documents, varying in structure. Assuming some amount of our documents would be non-OCR scans, we would have to incorporate some level of OCR and layout understanding. While this task can be handled in multiple ways, I believe LayoutLMv3 would be easy to incorporate. We can leverage the model to extract information from non-OCR documents as well as from processed documents with tables, allowing us to have simple textual representation of documents without losing structural context. It is possible to annotate some of these documents to finetune LayoutLM but this design doc will assume that is not possible.

STEP 2: Preprocessing User-Provided Metadata and Docs / Constructing a User-Specific Vector Database

The next step would be to preprocess the textual representations of our documents. This task is handled in multiple steps:

Using either a Longformer model, GPT-3, or T5 for text summarization to create an abstract for each document. Combine the abstract with metadata provided by the user. We want this combination to be short yet still informative. 
Convert these combinations of abstracts and metadata into embedding representations using RoBERTa (I decided to use RoBERTa but we can definitely deploy a more recent model)
Store embeddings inside a user-specific vector-based database. 


STEP 3: Embed User Query and Perform Semantic Search on User-Specific Vector Database

This step is rather more straightforward. We will embed our user’s question using the same version of RoBERTa as we used in the previous step. We will then find the closest embedding matches using FAISS. (Let’s say we pull the top 5 to 15)

It is important to remember that this database contains embeddings generated from the abstract/metadata combo we defined in the previous step.

STEP 4: Rank the Full Versions of the Top Results Using a More Robust Model Finetuned/Optimized for Question Answering

This step will entail us performing a much more detailed semantic search to drop the number of possible candidates to retrieve information from. Now that we have candidates to draw information from, we will now take the entirety of these documents into consideration.


We stick to the same model we used earlier, in this case RoBERTa, and finetune it for Reading Comprehension and Question Answering. Assuming that we would not be creating user-specific QA datasets, we will finetune on a generalized dataset like MS Marco or SQUAD (MS Marco might be more useful in this case)

If we were to have human annotators create user-specific QA datasets from a portion of the documents available to us, we could have made use of adapter layers to have some level of user-specific flexibility. Assuming that we would not be creating user-specific QA datasets, I will not explore this option here. 

The obvious problem with this approach would be RoBERTa’s token limitation, which would necessitate some form of fragmentation of our document. Estimating that most of our documents are of continuous form, splitting our document into multiple segments would be the best course of action. 

We can use GPT-3 or another model to avoid dealing with fragmentation to the same extent, yet, I will stick to MS Marco fine tuned RoBERTa for embedding generation. We then use FAISS again to select top 3 results.

STEP 5: Extracting the Answer From The Top Result

Now that we have the top 3 documents we perform our question answering on these documents using our fine tuned model. We provide the answer from the top document and cite the document. We also propose answers from the other 2 documents as additional suggestions in case our first document fails to produce the desired output.
