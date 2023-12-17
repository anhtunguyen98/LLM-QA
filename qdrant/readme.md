# FastAPI Server for Vector Data Management

This FastAPI server interfaces with a `SearchClient` to perform operations on a Qdrant server, a vector search engine for managing large datasets.

## Setup and Initialization

- FastAPI is initialized for creating API endpoints.
- Environment variables are used to configure the connection to the Qdrant server and set up the server environment.

## Environment Variables

Create an `.env` file in your project root with the following variables:

- `QDRANT_STORAGE`: Path to the storage directory for Qdrant.
- `QDRANT_HOST`: Hostname for the Qdrant server.
- `QDRANT_PORT`: Port for the Qdrant server.
- `QDRANT_HOST_PORT`: External port to access the Qdrant server.
- `API_HOST_PORT`: Port for the FastAPI server.

## Docker Compose

### Building and Running the Server

1. **Docker Compose File**: Ensure you have a `docker-compose.yml` file configured to set up the FastAPI server and Qdrant service.

2. **Building the Docker Image**:
   - Run the following command to build the Docker image:
     ```
     docker-compose build
     ```

3. **Running the Services**:
   - Start the services defined in your Docker Compose file using:
     ```
     docker-compose up
     ```
     
## API Endpoints

### Create Collection

- **Endpoint**: `/api/create_collection`
- **Method**: POST
- **Description**: Creates a new collection in the Qdrant server.
- **Parameters**:
  - `collection_name`: Name of the collection.
  - `distance`: Distance metric for vector comparison (e.g., cosine, euclidean).
- **Returns**: JSON response with the result of the operation.

#### Python Request
```python
import requests
url = "http://localhost:7861/api/create_collection"
payload = {
    "collection_name": "my_collection",
    "distance": "cosine"
}
response = requests.post(url, json=payload)
print(response.json())
```
```bash
curl -X 'POST' \
  'http://localhost:7861/api/create_collection' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "collection_name": "my_collection",
  "distance": "cosine"
}'
```


### Indexing Data

- **Endpoint**: `/api/indexing_data`
- **Method**: POST
- **Description**: Indexes data from a CSV file into a specified collection.
- **Parameters**:
  - `collection_name`: Name of the collection.
  - `file`: Uploaded CSV file.
- **Returns**: JSON response with the result of the indexing operation.

#### Python Request
```python
import requests

url = "http://localhost:7861/api/indexing_data"
files = {'file': open('data.csv', 'rb')}
data = {
    'collection_name': 'my_collection'
}
response = requests.post(url, files=files, data=data)
print(response.json())

```
```bash
curl -X 'POST' \
  'http://localhost:7861/api/indexing_data' \
  -H 'accept: application/json' \
  -F 'collection_name=my_collection' \
  -F 'file=@data.csv;type=text/csv'
```


### Add Point

- **Endpoint**: `/api/add_point`
- **Method**: POST
- **Description**: Adds a single data point to a specified collection.
- **Parameters**:
  - `collection_name`: Name of the collection.
  - `title`: Title of the data point.
  - `content`: Content of the data point.
- **Returns**: JSON response with the result of the operation.
#### Python Request
```python
import requests

url = "http://localhost:7861/api/add_point"
payload = {
    "collection_name": "my_collection",
    "title": "Example Title",
    "content": "Example content"
}
response = requests.post(url, json=payload)
print(response.json())


```
```bash
curl -X 'POST' \
  'http://localhost:7861/api/add_point' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "collection_name": "my_collection",
  "title": "Example Title",
  "content": "Example content"
}'

```

### Search

- **Endpoint**: `/api/search`
- **Method**: GET
- **Description**: Performs a search in a specified collection.
- **Parameters**:
  - `collection_name`: Name of the collection.
  - `query`: Search query.
  - `top_k`: Number of top results to return.
- **Returns**: JSON response with the search results.

#### Python Request
```python
import requests

url = "http://localhost:7861/api/search"
params = {
    "collection_name": "my_collection",
    "query": "example query",
    "top_k": 10
}
response = requests.get(url, params=params)
print(response.json())


```
```bash
curl -X 'GET' \
  'http://localhost:7861/api/search?collection_name=my_collection&query=example%20query&top_k=10' \
  -H 'accept: application/json'
```

### Delete Point

- **Endpoint**: `/api/delete_point`
- **Method**: POST
- **Description**: Deletes a specific point from a collection.
- **Parameters**:
  - `collection_name`: Name of the collection.
  - `id`: ID of the point to delete.
- **Returns**: JSON response with the result of the operation.

#### Python Request
```python
import requests

url = "http://localhost:7861/api/delete_point"
payload = {
    "collection_name": "my_collection",
    "id": 123
}
response = requests.post(url, json=payload)
print(response.json())

```
```bash
curl -X 'POST' \
  'http://localhost:7861/api/delete_point' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "collection_name": "my_collection",
  "id": 123
}'
```

### Edit Point

- **Endpoint**: `/api/edit_point`
- **Method**: PUT
- **Description**: Deletes a specific point from a collection.
- **Parameters**:
  - `collection_name`: Name of the collection.
  - `id`: ID of the point to delete.
- **Returns**: JSON response with the result of the operation.

#### Python Request
```python
import requests

url = "http://localhost:7861/api/edit_point"
payload = {
    "collection_name": "my_collection",
    "point_id": 123,
    "new_title": "Updated Title",
    "new_content": "Updated content"
}
response = requests.put(url, json=payload)
print(response.json())

```
```bash
curl -X 'PUT' \
  'http://localhost:7861/api/edit_point' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "collection_name": "my_collection",
  "point_id": 123,
  "new_title": "Updated Title",
  "new_content": "Updated content"
}'
```

### Edit Point

- **Endpoint**: `/api/edit_point`
- **Method**: PUT
- **Description**: Updates a specific data point within a collection. This is used to modify the title and content of an existing point.
- **Parameters**:
  - `collection_name`: The name of the collection containing the point to be edited.
  - `point_id`: The unique identifier of the point to be edited.
  - `new_title`: The new title for the point.
  - `new_content`: The new content for the point.
- **Returns**: JSON response with the result of the edit operation


#### Python Request
```python
import requests

url = "http://localhost:7861/api/delete_collection"
params = {
    "collection_name": "my_collection"
}
response = requests.delete(url, params=params)
print(response.json())

```
```bash
curl -X 'DELETE' \
  'http://localhost:7861/api/delete_collection?collection_name=my_collection' \
  -H 'accept: application/json'
```

