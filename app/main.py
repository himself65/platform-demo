import os

from llama_index.node_parser import SentenceSplitter
from llama_index.readers import SimpleDirectoryReader
from llama_index.ingestion import IngestionPipeline
from llama_index.embeddings import OpenAIEmbedding


def main():
    # local
    # os.environ["PLATFORM_BASE_URL"] = "http://localhost:8000"
    # os.environ["PLATFORM_APP_URL"] = "http://localhost:3000"

    # staging
    # os.environ["PLATFORM_BASE_URL"] = "https://api.staging.llamaindex.ai/"
    # os.environ["PLATFORM_APP_URL"] = "https://staging.llamaindex.ai/"

    reader = SimpleDirectoryReader(input_dir="./data")
    documents = reader.load_data()
    pipeline = IngestionPipeline(
        project_name="himself65_main",
        name="resume",
        documents=documents,
        transformations=[
            SentenceSplitter(),
            OpenAIEmbedding()
        ]
    )

    pipeline.register()
