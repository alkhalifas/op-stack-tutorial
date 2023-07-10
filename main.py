import openai
import os
import pinecone
from tqdm.auto import tqdm

class OPStack:
    """
    A class that represents the OP Stack using OpenAI and Pinecone.

    """
    def __init__(self):

        # Set the embedding model from OpenAI
        self.model = "text-embedding-ada-002"

        # Set OpenAI API Key
        openai.api_key = os.environ['OPENAI_API_KEY']

        # Set Pinecone API Key
        pinecone.init(
            api_key=os.environ['PINECONE_API_KEY'],
            environment="asia-southeast1-gcp-free"  # find next to API key in console
        )

        # Set the index for pinecone
        self.index = pinecone.Index('openai')

    def embed_data(self, batch_size, sentences):
        """
        Embeds the data and upserts to Pinecone
        :param batch_size: Batch size
        :param sentences: Text corpus in the form of a list of strings
        """
        for i in tqdm(range(0, len(sentences), batch_size)):

            # Set end of batch for TQDM
            i_end = min(i + batch_size, len(sentences))

            # Get Lines and IDs
            lines_batch = sentences[i: i + batch_size]
            ids_batch = [str(n) for n in range(i, i_end)]

            # Create Embedding using OpenAI
            res = openai.Embedding.create(input=lines_batch, engine=self.model)
            embeds = [record['embedding'] for record in res['data']]

            # Add metadata and Upsert batch
            meta = [{'text': line} for line in lines_batch]
            to_upsert = zip(ids_batch, embeds, meta)
            self.index.upsert(vectors=list(to_upsert))

    def query(self, query, top_k):
        """
        Function that allows a user to query data in Pinecone
        :param top_k: Top k items to return
        :param query: String representing the query
        :return: Prints the top k items
        """
        # Embed the query using OpenAI
        xq = openai.Embedding.create(input=query, engine=self.model)['data'][0]['embedding']

        # Get results from Pinecone index
        res = self.index.query([xq], top_k=top_k, include_metadata=True)

        for match in res['matches']:
            print(f"{match['score']:.2f}: {match['metadata']['text']}")


if __name__ == '__main__':
    # Create fake dataset
    dataset = [
        "Cats and dogs are excellent pets for your home",
        "Biotech companies develop novel therapeutics that aid humanity",
        "Hospitals help patients by treating them with medicines",
        "Macbooks are great devices to develop code and create content",
        "Tea is one of the worlds oldest drinks dating back thousands of years",
        "Boston is a beautiful city full of exciting places to visit"
    ]

    # Instantiate OPStack
    op_stack = OPStack()

    # Embed the data into pinecone
    op_stack.embed_data(32, dataset)

    # Query data
    op_stack.query("What is Boston known for?", 5)