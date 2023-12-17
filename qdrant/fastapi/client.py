import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm.auto import tqdm
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse


class SearchClient:
    def __init__(self, host, port):
        self.client = QdrantClient(host, port=port)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.retriever = SentenceTransformer(
            "multi-qa-MiniLM-L6-cos-v1", device=self.device
        )
        self.size = 384

    def create_collection(self, collection_name, distance):
        if distance == "cosine":
            distance = models.Distance.COSINE
        elif distance == "dot":
            distance = models.Distance.DOT
        elif distance == "euclid":
            distance = models.Distance.EUCLID
        else:
            raise ("Distance must in list [cosine, dot, euclid]")
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=self.size, distance=distance),
            )
        except UnexpectedResponse as e:
            error_message = "Collection `{}` already exists!".format(
                collection_name)
            if error_message in str(e):
                return {"error": "Collection already exists"}
            else:
                # To handle other cases of UnexpectedResponse if needed
                return {"error": "An unexpected error occurred"}

        return {"message": "Collection created successfully"}

    def batch_indexing(self, df, collection_name):
        batch_size = 128
        for index in tqdm(range(0, len(df), batch_size)):
            i_end = min(index + batch_size, len(df))
            batch = df.iloc[index:i_end]  # extract batch
            emb = self.retriever.encode(batch["text"].tolist()).tolist()
            meta = batch.to_dict(orient="records")  # get metadata
            ids = list(range(index, i_end))  # create unique IDs

            # upsert to qdrant
            self.client.upsert(
                collection_name=collection_name,
                points=models.Batch(ids=ids, vectors=emb, payloads=meta),
            )
        return {"message": "Upsert successful"}

    def search(self, collection_name, question, top_k):
        encoded_query = self.retriever.encode(question).tolist()

        result = self.client.search(
            collection_name=collection_name,
            query_vector=encoded_query,
            limit=top_k,
        )
        return result

    def delete_collection(self, collection_name):
        try:
            self.client.delete_collection(collection_name=collection_name)
        except UnexpectedResponse as e:
            error_message = "Collection `{}` is not exists!".format(
                collection_name)
            if error_message in str(e):
                return {"error": "Collection is not exists"}
            else:
                # To handle other cases of UnexpectedResponse if needed
                return {"error": "An unexpected error occurred"}

        return {"message": "Collection deleted successfully"}
        
    
    def count_points(self,collection_name):
        res = self.client.count(
            collection_name=collection_name,
        )
        return res.count
    
    def add_point(self, collection_name, title, content):
        
        id = self.count_points(collection_name)
        meta = {
            'title': title,
            'text': content 
        }
        emb = self.retriever.encode(meta['text']).tolist()

        try:
            self.client.upsert(
                collection_name=collection_name,
                points=[models.PointStruct(id=id, vector=emb, payload=meta)],
            )
        except UnexpectedResponse as e:
            error_message = "Collection `{}` is not exists!".format(
                collection_name)
            if error_message in str(e):
                return {"error": "Collection is not exists"}
            else:
                # To handle other cases of UnexpectedResponse if needed
                return {"error": "An unexpected error occurred"}
        return {"message": "Upsert successful"}
    def edit_point(self, collection_name, point_id, new_title, new_content):
        # Fetch existing point
        try:
            response = self.client.get(
                collection_name=collection_name,
                point_id=point_id
            )
            if not response.result:
                return {"error": "Point not found"}
        except UnexpectedResponse as e:
            return {"error": "An unexpected error occurred while fetching the point"}

        # Update point's content
        updated_meta = {
            'title': new_title,
            'text': new_content
        }

        # Re-encode the updated text
        updated_emb = self.retriever.encode(updated_meta['text']).tolist()

        # Upsert the updated point
        try:
            self.client.upsert(
                collection_name=collection_name,
                points=[models.PointStruct(id=point_id, vector=updated_emb, payload=updated_meta)],
            )
        except UnexpectedResponse as e:
            return {"error": "An unexpected error occurred during upsert"}

        return {"message": "Point updated successfully"}

    
    def delete_point(self,collection_name,id):
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=[id],
                ),
            )
        except UnexpectedResponse as e:
            error_message = "Collection `{}` is not exists!".format(
                collection_name)
            if error_message in str(e):
                return {"error": "Collection is not exists"}
            else:
                # To handle other cases of UnexpectedResponse if needed
                return {"error": "An unexpected error occurred"}

        return {"message": "Collection deleted successfully"}