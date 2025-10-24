import nest_asyncio
nest_asyncio.apply()

# install all the dependicies requried

# !pip install -q -q llama-index
# !pip install -U -q deepeval
# !pip install faiss-cpu
# !pip install llama-index-readers-web
# !pip install llama-index-vector-stores-faiss
# !pip install llama-index-embeddings-openai
# !pip install llama-index-llms-openai
# !pip install llama-index-ragas
# !pip install ragas
# !pip install llama-index
# !pip install llama-index-experimental



import os
import openai
from getpass import getpass

openai.api_key = getpass("Please provide your OpenAI Key: ")
os.environ["OPENAI_API_KEY"] = openai.api_key


from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.core import (
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)


loader = BeautifulSoupWebReader()
documents = loader.load_data(urls=["https://en.wikipedia.org/wiki/2023_in_video_games"])
index = VectorStoreIndex.from_documents(documents)

documents[0].metadata

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)


embed_model = OpenAIEmbedding()
splitter = SemanticSplitterNodeParser(
    buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
)

# also baseline splitter
base_splitter = SentenceSplitter(chunk_size=700,
                                chunk_overlap=50)

nodes = splitter.get_nodes_from_documents(documents)


# get API key and create embeddings
from llama_index.embeddings.openai import OpenAIEmbedding

embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
)


import faiss

# dimensions of text-ada-embedding-002
d = 1536
faiss_index = faiss.IndexFlatL2(d)
from llama_index.core import VectorStoreIndex, download_loader
from llama_index.vector_stores.faiss import FaissVectorStore
from IPython.display import Markdown, display


vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context,embed_model=embed_model
)

# save index to disk
index.storage_context.persist()

# load index from disk
vector_store = FaissVectorStore.from_persist_dir("./storage")
storage_context = StorageContext.from_defaults(
    vector_store=vector_store, persist_dir="./storage"
)
index = load_index_from_storage(storage_context=storage_context)


from llama_index.core.retrievers import QueryFusionRetriever

retriever = QueryFusionRetriever(
    [index.as_retriever()],
    similarity_top_k=1,
    num_queries=1,  # set this to 1 to disable query generation
    use_async=True,
    verbose=True,
    # query_gen_prompt="...",  # we could override the query generation prompt here
)

nodes_with_scores = retriever.retrieve("What are the top racing games?")

for doc in nodes_with_scores:
  print(doc)


from llama_index.core import PromptTemplate

# reset
query_engine = index.as_query_engine(response_mode="tree_summarize")

# shakespeare!
new_summary_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query in the style of a Shakespeare play.\n"
    "Query: {query_str}\n"
    "Answer: "
)
new_summary_tmpl = PromptTemplate(new_summary_tmpl_str)


query_engine.update_prompts(
    {"response_synthesizer:summary_template": new_summary_tmpl}
)

prompts_dict = query_engine.get_prompts()

response = query_engine.query("What are the top FPS games ?")
print(str(response))

# # set Logging to DEBUG for more detailed outputs
# query_engine = index.as_query_engine()
# response = query_engine.query("I need all the games name present in the webpage ?")
# display(Markdown(f"{response}"))


from datasets import Dataset
from ragas.metrics import context_precision
from ragas.metrics import context_relevancy
from ragas.metrics import answer_similarity
from ragas import evaluate

data_samples ={
    'question': [
        'What significant acquisition occurred in the video game industry in 2023?',
        'Which game swept the five major awards in 2023?',
        'What was the top-rated game released in 2023 according to Metacritic?',
        'Which game received the highest score on Metacritic among the releases of April 11, 2023?',
        'How many games released in 2023 achieved a Metacritic score of 90 or higher?',
        'Which company became the third-largest game publisher in the world after an acquisition in 2023?',
        'Which game won the award for Game of the Year in 2023 at the Japan Game Awards?',
        'Which game received the award for Art Direction at the Game Developers Choice Awards in 2024?'
    ],
    'answer': [
        'Microsoft completed its acquisition of Activision Blizzard for $69 billion, becoming the third-largest game publisher globally.',
        'Baldur\'s Gate 3 was the first game to sweep the five major awards: the Golden Joystick Awards, the Game Awards, the D.I.C.E. Awards, the Game Developer Choice Awards, and the BAFTA Games Awards.',
        'The following table lists the top-rated games released in 2023 based on Metacritic, which generally considers expansions as separate entities.',
        'Romano, Sal (March 6, 2023). Sherlock Holmes: The Awakened remake launches April 11. Gematsu. Archived from the original on March 23, 2023. Retrieved March 23, 2023',
        'The number of highly praised video games released in 2023 was considered unusually high compared to most years, with 25 games having a 90 out of 100 or better aggregate score on Metacritic by October 2023; this made it the best year by number of acclaimed games, the largest since 2004.',
        'Microsoft, after having satisfied worldwide regulatory bodies, completed its US$69 billion acquisition of Activision Blizzard, making them the third-largest game publisher in the world',
        'Category / Organization 27th Japan Game Awards September 20, 2023. Game of the Year Monster Hunter Rise: Sunbreak.',
        'Category / Organization 24th Game Developers Choice Awards March 20, 2024. Art Direction Baldur\'s Gate 3 Alan Wake 2.'
    ],
    'contexts': [
        ['In the video game industry, 2023 saw significant changes within larger publishers and developers. Microsoft, after having satisfied worldwide regulatory bodies, completed its US$69 billion acquisition of Activision Blizzard, making them the third-largest game publisher in the world'],
        ['This question focuses on a game that achieved significant recognition across multiple award ceremonies in 2023.'],
        ['This question pertains to the highest-rated game of the year based on Metacritic scores.'],
        ['This question requires identifying the game with the highest Metacritic score among releases on a specific date.'],
        ['This question involves identifying the number of games that received exceptionally high scores on Metacritic in 2023.'],
        ['This question focuses on a company\'s status change in the gaming industry following an acquisition.'],
        ['This question relates to the recipient of the Game of the Year award at a specific awards ceremony in 2023.'],
        ['This question involves identifying the recipient of a specific award at a particular awards ceremony in 2024.']
    ],
    'ground_truth': [
        'Microsoft acquired Activision Blizzard for $69 billion, becoming the third-largest game publisher',
        'Baldur\'s Gate 3 swept the five major awards in 2023.',
        'The Legend of Zelda: Tears of the Kingdom was the highest-rated game of 2023 on Metacritic, scoring 96 out of 100.',
        'Raji: An Ancient Epic received the highest Metacritic score among the games released on April 11, 2023.',
        'There were 11 games released in 2023 that received a Metacritic score of 90 or higher.',
        'Microsoft became the third-largest game publisher in the world after acquiring Activision Blizzard.',
        'Monster Hunter Rise: Sunbreak won the Game of the Year award at the Japan Game Awards in 2023.',
        'Baldur\'s Gate 3 received the award for Art Direction at the Game Developers Choice Awards in 2024.'
    ]
}

dataset = Dataset.from_dict(data_samples)
score = evaluate(dataset,metrics=[
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_relevancy
    ])
score.to_pandas()

