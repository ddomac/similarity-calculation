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
    use_method = st.radio(
        "Select similarity calculation method:",
        ("Embeddings", "TF-IDF", "Jaccard Similarity", "All Methods"),
    )

    st.info("The 'All Methods' option runs all three similarity calculations and adds separate columns to the output file.")

    # Add a start button
    if st.button("Start Calculation"):
        progress_bar = st.progress(0)
        log_container = st.empty()  # Placeholder for logs
        MAX_LOG_LINES = 5  # Keep only the latest logs

        # Function to calculate similarity
        def calculate_similarity(prompts, method):
            log_messages = []
            if method == "Embeddings":
                log_messages.append("Generating embeddings...")
                model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
                embeddings = model.encode(prompts, convert_to_tensor=True)
                similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings).numpy()
                log_messages.append("Embeddings similarity calculation completed.")
            elif method == "TF-IDF":
                log_messages.append("Calculating TF-IDF similarity...")
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(prompts)
                similarity_matrix = cosine_similarity(tfidf_matrix)
                log_messages.append("TF-IDF similarity calculation completed.")
            elif method == "Jaccard Similarity":
                log_messages.append("Calculating Jaccard similarity...")
                mlb = MultiLabelBinarizer()
                tokenized_prompts = [set(prompt.split()) for prompt in prompts]
                binary_matrix = mlb.fit_transform(tokenized_prompts)
                similarity_matrix = [
                    [jaccard_score(binary_matrix[i], binary_matrix[j]) for j in range(len(binary_matrix))]
                    for i in range(len(binary_matrix))
                ]
                log_messages.append("Jaccard similarity calculation completed.")
            else:
                return []

            # Calculate scores
            scores = []
            for i, row in enumerate(similarity_matrix):
                row[i] = 0  # Exclude self-comparison
                max_similarity = max(row)
                scores.append(max_similarity * 100)
                # Update progress bar and logs
                progress_bar.progress(int((i + 1) / len(prompts) * 100))
                log_message = f"Processed row {i + 1}/{len(prompts)} for {method}."
                log_messages.append(log_message)
                log_messages = log_messages[-MAX_LOG_LINES:]  # Keep recent logs
                log_container.markdown("### Logs:\n" + "\n".join(log_messages))
                time.sleep(0.05)  # Smooth UI refresh
            return scores

        prompts = data['Prompt']

        # Run selected method(s)
        if use_method == "All Methods":
            st.info("Running all methods: Embeddings, TF-IDF, and Jaccard Similarity.")
            data['Embeddings Score (%)'] = calculate_similarity(prompts, "Embeddings")
            data['TF-IDF Score (%)'] = calculate_similarity(prompts, "TF-IDF")
            data['Jaccard Score (%)'] = calculate_similarity(prompts, "Jaccard Similarity")
        else:
            method_name = use_method.split()[0]
            column_name = f"{method_name} Score (%)"
            data[column_name] = calculate_similarity(prompts, use_method)

        # Display the results
        st.write("Similarity scores:", data)

        # Allow user to download results
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
