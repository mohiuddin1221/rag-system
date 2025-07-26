I use pdfplumber to extract text from PDFs because it preserves the layout and structure, and accurately captures both plain text and tables

For chunking the extracted text, I choose RecursiveCharacterTextSplitter as it intelligently splits the content into semantically coherent pieces, ensuring better performance during vector embedding and semantic search.

I used AzureOpenAIEmbeddings, which leverages OpenAI's text-embedding-ada-002 model through Azure to generate high-quality semantic vector representations of text. I chose it because it captures the contextual meaning of words using transformer-based architecture, making it ideal for tasks like semantic search and retrieval.

I compare the query embedding with stored chunk embeddings using cosine similarity to find semantically similar content. I chose this method because cosine similarity effectively measures the angle between high-dimensional vectors, making it ideal for capturing meaning regardless of text length or scale, and I use Pinecone as the storage setup for fast, scalable vector search.

To ensure meaningful comparison, I use high-quality embeddings (via AzureOpenAIEmbeddings) that capture the semantic meaning of both the query and document chunks, and I split documents into coherent chunks to preserve context. If the query is vague or lacks context, the retrieval might return less relevant results, so adding query rephrasing or conversational memory can improve accuracy.


######
Here, I am using AzureChatOpenAI as the language model, but you can try using your own LLM API. For the vector database, I am using Pinecone.




####   Run a local server####

Run a local server pip install --upgrade "langgraph-cli[inmem]" 
pip install -e . l
anggraph dev