from document_manager import DocumentManager
from text_splitter import TextSplitter
from embedding_manager import EmbeddingManager
from search_engine import SearchEngine
import settings as settings
from transformers import AutoTokenizer
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp

## USER QUERY
question = "Does web popup in webcx support segmentation?"


def run_without_rag(llm, question):
    llm(question)

def run_with_rag(llm, db, template):

    # Create a retriever from the embeddings db
    # It retrieves up to 2 relevant documents.
    retriever = db.as_retriever(search_kwargs={"k": 2})
    qa_chain_prompt = PromptTemplate.from_template(template)

    # Create a question-answering instance (qa) using the RetrievalQA from langchain framework.
    # The chain will take a list of documents, inserts them all into a prompt, and passes that prompt to an LLM:
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        verbose=True,
        chain_type_kwargs={"prompt": qa_chain_prompt, "verbose": True},
    )
    # Ask the question and get a response
    return qa({"query": question})


def run_similarity_search(db):
    search_engine = SearchEngine(db)
    search_results = search_engine.search(question)
    print(search_results[0].page_content)


if __name__ == "__main__":
    doc_manager = DocumentManager(settings.DATA_BASE_DIR, settings.DATA_ALLOWED_EXTENSIONS)
    text_splitter = TextSplitter(settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    embedding_manager = EmbeddingManager(settings.EMBEDDING_MODEL_PATH, settings.EMBEDDING_MODEL_DEVICE, False)
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    llm = LlamaCpp(
        model_path=settings.LLM_PATH,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        top_p=settings.LLM_TOP_P,
        callback_manager=callback_manager,
        verbose=True, # Verbose is required to pass to the callback manager
        )
    
    text_data, metadata = doc_manager.get_documents()
    docs = text_splitter.split(text_data, metadata)
    db = embedding_manager.create_vector_store(docs)

    ##### Checking if the search engine work s#####
    # run_similarity_search(db)

    # Load the pre-trainedtokenizer related to the LLM chosen and loaded in line 34
    # This tokenizer will be used convert inputs to the format the LLM expects
    # This tokenizer is used implecitly by the LLM RetrievalQA
    tokenizer = AutoTokenizer.from_pretrained(
        settings.TOKENIZATION_MODEL_PATH,
        padding=True,
        truncation=True,
        max_length=settings.TOKENIZATION_MAX_LENGTH,
        token=settings.HF_TOKEN)
    
    ###### Baseline answer from LLM without retrieval (RAG) ######
    # print(run_without_rag(llm, question))

    # Prepare RAG prompt
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible.
    Context: {context}
    Question: {question}
    Helpful Answer:"""

    ######## Answer from LLM with retrieval (RAG)  #######
    print(run_with_rag(llm, db, template))





