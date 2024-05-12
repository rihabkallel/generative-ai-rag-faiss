# Description
This repository is a base template for genrative AI (LLM) with a retrieval system (RAG) using a Vector DB (specifically FAISS).
This base project is using the open source model LLAMA-2 downloaded from hugging face compatible with a CPU infrastructure.
You can find the different models used and their configurations under: app/settings.py

- Put your dataset for RAG context under the data/ folder
- app scripts are under app/ :
    - document_manager.py: Retrieve the documents to be embedded.
    - embedding_manager.py: Embed and save the documents in FAISS vector db.
    - search_engine.py: Similarity search with vector DB.
    - settings.py: models and app settings.
    - text_splitter.py: split and create embedded documents for vector db.
    - main.py: app run script.

# Requirements
- Python: >= 3.9
- Venv or Conda for your python virtual environment.
- A hugging face account and an authentication token to download the open source models. (Check the hugging face section below).
- Langchain framework.
- Faiss CPU based library for vector DB and similarity search.

# Hugging Face and model download
- Get your hugginface account and CLI: https://huggingface.co/docs/huggingface_hub/en/guides/cli
- Generate your huggingface token for login and installations of open source models: https://huggingface.co/settings/tokens 
- Get access to llama from meta: https://huggingface.co/meta-llama/Llama-2-7b-hf (wait for review first)
- Models are downloaded from here: https://huggingface.co/TheBloke (using the CLI command below)

# Create a virtual environment
you can use conda or venv:

- Conda
```
conda create -n <your-env-name> python=<your-python-version>
conda activate <your-env-name>
```

- Venv
```
python3 -m venv <your-env-name>
source <your-env-name>/bin/activate
```

# Installation
- Install the requirements
```
pip3 install -r requirements.txt
```

- Install the model

```
huggingface-cli download TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf --local-dir models/ --local-dir-use-symlinks False
```
Models should be installed under the folder models/ in the root of this project.

I am using "meta-llama/Llama-2-7b-hf" instead of "TheBloke/Llama-2-7B-GGUF" because the tokenizer model is not included in the GGUF converted LLM. (the LLM and the tokenizer need to be compatible). In this case, both are of type "llama".

# Execution
- Run:
```  python app/main.py ```
- If you want to verify if the embeddings and similarity search work, uncomment line #67 in main.py.
- If you want to execute the LLM generation without RAG, uncomment line #80 in main.py.
- If you want to execute the LLM generation with RAG, use line #89 in main.py.


# More details
- FAISS does not support cloud deployment but there are other options you can choose from:
https://medium.com/the-ai-forum/which-vector-database-should-you-use-choosing-the-best-one-for-your-needs-5108ec7ba133

- You can use Open AI GPT with Langchain or any other model.
