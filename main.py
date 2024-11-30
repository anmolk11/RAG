import os
from tqdm import tqdm
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
index = None

docs = []
file_names = []


def get_data(folder_path : str = "Knowledge Base/Jobs") -> list[str]:
    for filename in tqdm(os.listdir(folder_path), desc="Reading the files"):
        file_names.append(filename)
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as file:
                content = file.read()
                docs.append(content)


def make_vector_db():
    embeddings = model.encode(docs, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]  
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"Number of vectors in the index: {index.ntotal}")

    faiss.write_index(index, "vector_database.index")
    print("FAISS index saved as 'vector_database.index'")


def make_query(query : str) -> str:
    query_embedding = model.encode([query]).astype('float32')   
    top_k = 2
    distances, indices = index.search(query_embedding, top_k)

    for i, idx in enumerate(indices[0]):
        print(f"Match {i+1}:")
        print(f"File: {file_names[idx]}")
        print(f"Content: {docs[idx][:200]}...")  
        print(f"Distance: {distances[0][i]}")
        print('-------------------------------------------------')



if __name__ == '__main__':
    get_data()
    # make_vector_db()
    index = faiss.read_index("vector_database.index")
    make_query('SQL and python')
    