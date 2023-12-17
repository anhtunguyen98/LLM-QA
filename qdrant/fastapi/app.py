import os

from fastapi import FastAPI, UploadFile, Response
from fastapi.staticfiles import StaticFiles
import pandas as pd
from client import SearchClient
import json
import os
app = FastAPI()
HOST = os.getenv("QDRANT_HOST", "localhost")
PORT = os.getenv("QDRANT_PORT", 6333)

search_client = SearchClient(host=HOST,port=PORT)



@app.post('/api/create_collection')
async def create_collection(collection_name:str, distance: str):
    output = search_client.create_collection(collection_name=collection_name,
                                    distance=distance)
    output = json.dumps({"output": output},  indent=4, default=dict)
    return Response(content=output, media_type="application/json")


@app.post('/api/indexing_data')
async def indexing_data(collection_name:str,file:UploadFile):
    contents = await file.read()
    df = pd.read_csv(contents)
    output = search_client.batch_indexing(collection_name=collection_name,
                                 df=df)
    output = json.dumps({"output": output},  indent=4, default=dict)
    return Response(content=output, media_type="application/json")

@app.post("/api/add_point")
async def add_point(collection_name:str, title:str, content:str):
    output = search_client.add_point(collection_name=collection_name,
                            title=title,
                            content=content)

    output = json.dumps(output, indent=4, default=str)
    return Response(content=output, media_type="application/json")

    
@app.get("/api/search")
async def search(collection_name:str, query: str, top_k:int):
    output = search_client.search(collection_name=collection_name,
                         question=query,
                         top_k=top_k)
    output = json.dumps({"output": output},  indent=4, default=dict)
    return Response(content=output, media_type="application/json")


@app.post("/api/delete_point")
async def delete_point(collection_name:str, id:int):
    output = search_client.delete_point(collection_name=collection_name,
                               id=id)
    output = json.dumps({"output": output},  indent=4, default=dict)
    return Response(content=output, media_type="application/json")

@app.put("/api/edit_point")
async def edit_point(collection_name: str, point_id: int, new_title: str, new_content: str):
    output = search_client.edit_point(collection_name=collection_name, point_id=point_id, new_title=new_title, new_content=new_content)

    output = json.dumps({"output": output}, indent=4, default=dict)
    return Response(content=output, media_type="application/json")

@app.delete("/api/delete_collection")
async def delete_collection(collection_name:str):
    output = search_client.delete_collection(collection_name=collection_name)

    output = json.dumps({"output": output},  indent=4, default=dict)
    return Response(content=output, media_type="application/json")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7861)