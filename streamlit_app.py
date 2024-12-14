import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer
import time

# Streamlit app title
st.title("Similarity Calculator")

# File upload section
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Load the CSV file
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # Check for the 'Prompt' column
    if 'Prompt' not in data.columns:
        st.error("The uploaded CSV must contain a 'Prompt' column.")
        st.stop()

    # Handle non-string values in the 'Prompt' column
    data['Prompt'] = data['Prompt'].fillna("MISSING PROMPT").astype(str)

    # User choice for similarity method
    use_embeddings = st.radio(
        "Select similarity calculation method:",
        ("Embeddings (more accurate, slower)", "TF-IDF (faster)", "Jaccard Similarity"),
    )

    # Show a brief description of the selected method
    if use_embeddings == "Embeddings (more accurate, slower)":
        st.info("Embeddings use the pre-trained model 'all-MiniLM-L6-v2' to generate dense vector representations of text, capturing semantic meaning. This method is slower but more accurate.")
    elif use_embeddings == "TF-IDF (faster)":
        st.info("TF-IDF computes term frequency-inverse document frequency to represent text and measure similarity. It is faster but less nuanced.")
    else:
        st.info("Jaccard Similarity measures the overlap between sets of tokens in text. It is simple and interpretable but less precise for complex language.")

    # Add a start button
    if st.button("Start Calculation"):
        # Add a progress bar
        progress_bar = st.progress(0)

        # Add a log container (embedded scrolling window)
        log_container = st.empty()  # Placeholder for logs
        MAX_LOG_LINES = 5  # Limit the log messages to the last 5 updates

        # Calculate similarity scores
        def calculate_similarity(data, use_embeddings):
            prompts = data['Prompt']
            log_messages = []  # Store log messages

            if use_embeddings == "Embeddings (more accurate, slower)":
                # Use sentence embeddings
                model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
                log_messages.append("Starting embedding generation...")
                log_container.markdown("### Logs:\n" + "\n".join(log_messages))

                # Generate embeddings for all prompts in a single batch
                for idx, prompt in enumerate(prompts):
                    # Simulate generating embedding (update for each prompt)
                    embedding = model.encode([prompt], convert_to_tensor=True)
                    
                    # Log progress
                    log_message = f"Processed {idx + 1}/{len(prompts)} prompts."
                    log_messages.append(log_message)
                    log_messages = log_messages[-MAX_LOG_LINES:]  # Keep only the latest logs
                    
                    # Update log container
                    log_container.markdown("### Logs:\n" + "\n".join(log_messages))
                    time.sleep(0.05)  # Simulate time delay for display

                # Generate all embeddings as a batch
                embeddings = model.encode(prompts, convert_to_tensor=True)

                # Compute similarity matrix
                similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings).numpy()

            elif use_embeddings == "TF-IDF (faster)":
                # Use TF-IDF
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(prompts)
                similarity_matrix = cosine_similarity(tfidf_matrix)

            else:
                # Use Jaccard Similarity
                mlb = MultiLabelBinarizer()
                tokenized_prompts = [set(prompt.split()) for prompt in prompts]
                binary_matrix = mlb.fit_transform(tokenized_prompts)
                similarity_matrix = [
                    [jaccard_score(binary_matrix[i], binary_matrix[j]) for j in range(len(binary_matrix))]
                    for i in range(len(binary_matrix))
                ]

            # Get similarity scores for each prompt (excluding self-comparison)
            scores = []
            for i, row in enumerate(similarity_matrix):
                row[i] = 0  # Ignore self-comparison
                max_similarity = max(row)
                scores.append(max_similarity * 100)  # Convert to percentage

                # Update progress bar
                progress_bar.progress(int((i + 1) / len(similarity_matrix) * 100))

            return scores

        # Compute the similarity scores
        data['Similarity Score (%)'] = calculate_similarity(data, use_embeddings)

        # Display the results
        st.write("Similarity scores:", data)

        # Allow the user to download the results
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(data)

        st.download_button(
            label="Download Results",
            data=csv,
            file_name="similarity_scores.csv",
            mime="text/csv",
        )
