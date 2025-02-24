from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit import prompt
import pandas as pd

def get_top5_recommendations(ref_idx, rating_threshold=5, top_n=5):
    # Use df for reference lookup so indices match X and book_vectors
    ref_title = df.loc[ref_idx, "Clean Title"]
    ref_vector = X[ref_idx]  # Using the TF-IDF matrix from df
     
    # Find users who rated the reference book with the desired threshold (fallback to threshold-1)
    users = set(df_merge[(df_merge["Clean Title"] == ref_title) & (df_merge["rating"] == rating_threshold)]["user"])
    if not users:
        users = set(df_merge[(df_merge["Clean Title"] == ref_title) & (df_merge["rating"] == rating_threshold - 1)]["user"])
    
    candidates = pd.DataFrame()
    if users:
        # Filter candidate reviews from these users (and exclude the ref book)
        candidates = df_merge[
            (df_merge["user"].isin(users)) &
            (df_merge["rating"] == rating_threshold) &
            (df_merge["Clean Title"] != ref_title)
        ].drop_duplicates(subset=["Clean Title"])
    if len(candidates) == 0:
        # If no such users, default: use all books except the reference, taking only unique titles
        candidates = df_merge[df_merge["Clean Title"] != ref_title].drop_duplicates(subset=["Clean Title"])
        
    # Build a list of candidate indices corresponding to df.
    # (Remember: book_vectors and X were built on df, not df_merge.)
    candidate_df_indices = []  # Indices from df
    candidate_mapping = {}     # Map from candidate index in df to candidate index in df_merge
    for merge_idx in candidates.index:
        candidate_title = df_merge.loc[merge_idx, "Clean Title"]
        if pd.isna(candidate_title):
            continue
        candidate_key = candidate_title.strip().lower()
        if candidate_key in clean_title_to_index:
            df_idx = clean_title_to_index[candidate_key]
            candidate_df_indices.append(df_idx)
            candidate_mapping[df_idx] = merge_idx  # record mapping back to df_merge
    
    if not candidate_df_indices:
        return pd.DataFrame()  # No candidates found
    
    # Compute cosine similarities in one batch:
    candidate_matrix = X[candidate_df_indices]
    sim_values = cosine_similarity(ref_vector, candidate_matrix)[0]
    
    # Create a list of tuples: (merge_idx, similarity)
    candidate_sim_tuples = []
    for df_idx, sim in zip(candidate_df_indices, sim_values):
        merge_idx = candidate_mapping[df_idx]
        candidate_sim_tuples.append((merge_idx, sim))
    
    # Sort by similarity (highest first) and select top_n candidates
    top_candidates = sorted(candidate_sim_tuples, key=lambda x: x[1], reverse=True)[:top_n]
    
    recs = []
    for merge_idx, sim in top_candidates:
        candidate_title = df_merge.loc[merge_idx, "Clean Title"]
        # Get the corresponding df index from the lookup dictionary
        df_candidate_idx = clean_title_to_index[candidate_title.strip().lower()]
        recs.append({
            "candidate_index": df_candidate_idx,
            "Title": df.loc[df_candidate_idx, "Title"],
            "similarity": sim,
            "rating": df_merge.loc[merge_idx, "rating"]
        })
    return pd.DataFrame(recs)

if __name__ == "__main__":
    print("Setting up...")
    print("0/5")
    df = pd.read_csv('../data/books_data.csv')
    print("1/5")
    ratings = pd.read_csv('../data/Books_rating.csv')

    # Pairwise recommendation tuning
    # Filter & sample reference books (rating 5, 5% sample)
    ratings_renamed = ratings.rename(columns={"User_id": "user", "review/score": "rating"})
    df['combined_features'] = df['Title'].fillna('') + ' ' + df['authors'].fillna('') + ' ' + df['description'].fillna('')
    df['combined_features'] = df['combined_features'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
    df["Clean Title"] = df['Title'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
    ratings_renamed["Clean Title"] = ratings_renamed['Title'].str.lower().str.replace(r'[^\w\s]', '', regex=True)

    df_merge = pd.merge(df, ratings_renamed[["Clean Title", "user", "rating"]], on="Clean Title", how="left")

    print("2/5")
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
    print("3/5")
    book_vectors = {i: tfidf_matrix[i] for i in range(tfidf_matrix.shape[0])}

    # Build a dictionary mapping each Clean Title (converted to string, lowercased, and stripped) to its first index in df.
    clean_title_to_index = {}
    for idx, title in df["Clean Title"].astype(str).items():
        cleaned_title = title.strip().lower()
        if cleaned_title not in clean_title_to_index:
            clean_title_to_index[cleaned_title] = idx

    clean_title_to_index['the great gatsby']

    # Construct the TF-IDF matrix (X) from the combined features column
    print("4/5")
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['combined_features'])

    print("5/5")

    # Build a list of clean titles (keys in your dictionary)
    book_titles = list(clean_title_to_index.keys())
    book_completer = WordCompleter(book_titles, ignore_case=True, sentence=True)

    while True:
        # The prompt now shows autofill suggestions as you type.
        ref_title = prompt("Enter a book title: ", completer=book_completer)
        ref_title = ref_title.strip().lower()
        if ref_title in clean_title_to_index:
            ref_idx = clean_title_to_index[ref_title]
            print("Reference book:", df.loc[ref_idx, "Title"])
            recommendations = get_top5_recommendations(ref_idx)
            print(recommendations)
        else:
            print("Book not found.")
        cont = prompt("Continue? (y/n): ")
        if cont.strip().lower() != 'y':
            break