import os
import dotenv
import pandas as pd

from litellm import completion
import streamlit as st

from typing import Optional, Any
from pytidb import TiDBClient
from pytidb.schema import TableModel, Field
from pytidb.embeddings import EmbeddingFunction

dotenv.load_dotenv()

# RAG prompt template
RAG_PROMPT_TEMPLATE = """Answer the question based on the following reference information.

Reference Information:
{context}

Question: {question}

Please answer:"""

db = TiDBClient.connect(
    host=os.getenv("SERVERLESS_CLUSTER_HOST"),
    port=int(os.getenv("SERVERLESS_CLUSTER_PORT")),
    username=os.getenv("SERVERLESS_CLUSTER_USERNAME"),
    password=os.getenv("SERVERLESS_CLUSTER_PASSWORD"),
    database=os.getenv("SERVERLESS_CLUSTER_DATABASE_NAME"),
    enable_ssl=True,
)

text_embed = EmbeddingFunction(
    "bedrock/amazon.titan-embed-text-v2:0",
    timeout=60
)
llm_model = "bedrock/us.amazon.nova-lite-v1:0"


# Define the Chunk table
table_name = "chunks"
class Chunk(TableModel, table=True):
    __tablename__ = table_name
    __table_args__ = {"extend_existing": True}

    id: int = Field(primary_key=True)
    text: str = Field()
    text_vec: Optional[Any] = text_embed.VectorField(
        source_field="text",
    )


sample_chunks = [
    "Llamas are camelids known for their soft fur and use as pack animals.",
    "Python's GIL ensures only one thread executes bytecode at a time.",
    "TiDB is a distributed SQL database with HTAP capabilities.",
    "Einstein's theory of relativity revolutionized modern physics.",
    "The Great Wall of China stretches over 13,000 miles.",
    "Ollama enables local deployment of large language models.",
    "HTTP/3 uses QUIC protocol for improved web performance.",
    "Kubernetes orchestrates containerized applications across clusters.",
    "Blockchain technology enables decentralized transaction systems.",
    "GPT-4 demonstrates remarkable few-shot learning capabilities.",
    "Machine learning algorithms improve with more training data.",
    "Quantum computing uses qubits instead of traditional bits.",
    "Neural networks are inspired by the human brain's structure.",
    "Docker containers package applications with their dependencies.",
    "Cloud computing provides on-demand computing resources.",
    "Artificial intelligence aims to mimic human cognitive functions.",
    "Cybersecurity protects systems from digital attacks.",
    "Big data analytics extracts insights from large datasets.",
    "Internet of Things connects everyday objects to the internet.",
    "Augmented reality overlays digital content on the real world.",
]

table = db.open_table(table_name)
if table is None:
    table = db.create_table(schema=Chunk)


st.title("🔍 RAG Demo")

st.subheader("Database Operations")

left, right = st.columns(2)

if left.button("Reset", type="primary"):
    left.write("Resetting...")
    table.truncate()
    st.rerun()

left.markdown("This option <span style='color: red;'>will delete all data</span></h3>", unsafe_allow_html=True)

if right.button("Save Data"):
    right.write("Saving...")
    # insert sample chunks
    if table.rows() == 0:
        chunks = [Chunk(text=text) for text in sample_chunks]
        table.bulk_insert(chunks)
        st.rerun()

with st.expander("📁 All Chunks in the Database", expanded=False):
    chunks = table.query()
    if chunks:
        data = [{'id': chunk.id, 'text': chunk.text, 'text_vec': chunk.text_vec} for chunk in chunks]
        df = pd.DataFrame(data)
        st.dataframe(df)
    else:
        st.info("No data found in database.")

st.write(
    "Enter your question, and the system will retrieve relevant knowledge and generate an answer"
)
mode = st.radio("Select Mode:", ["Retrieval Only", "RAG Q&A"])

query_limit = st.sidebar.slider("Retrieval Limit", min_value=1, max_value=20, value=5)
query = st.text_input("Enter your question:", "")

if st.button("Send") and query:
    with st.spinner("Processing..."):
        # Retrieve relevant chunks
        res = table.search(query).limit(query_limit)

        if res:
            if mode == "Retrieval Only":
                st.write("### Retrieval Results:")
                st.dataframe(res.to_pandas())
            else:
                text = [chunk.text for chunk in res.to_rows()]

                # Build RAG prompt
                context = "\n".join(text)
                prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=query)

                # Call LLM to generate answer
                response = completion(
                    model=llm_model,
                    messages=[{"content": prompt, "role": "user"}],
                )

                st.markdown(f"### 🤖 {llm_model}")
                st.markdown(
                    """
                <style>
                .llm-response {
                    background: rgba(255, 255, 255, 0.05);
                    padding: 25px;
                    border-radius: 15px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    margin: 15px 0;
                    font-size: 1.1em;
                    line-height: 1.6;
                    color: #e1e4e8;
                }
                .llm-response:hover {
                    background: rgba(255, 255, 255, 0.08);
                    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
                    transition: all 0.3s ease;
                }
                </style>
                """,
                    unsafe_allow_html=True,
                )

                # show the response
                st.markdown(
                    f'<div class="llm-response">{response.choices[0].message.content}</div>',
                    unsafe_allow_html=True,
                )

                with st.expander("📚 Retrieved Knowledge", expanded=False):
                    st.dataframe(res.to_pandas())
        else:
            st.info("No relevant information found")