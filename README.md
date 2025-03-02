# Book Recommendation System

This project is a book recommendation system that leverages user ratings and text analysis to suggest similar books. It uses data from two CSV files containing book metadata and user ratings. The system preprocesses text (e.g., title, authors, description) by cleaning and vectorizing it with TF-IDF, and then computes cosine similarity to generate recommendations.

## How to Use

1. **Clone the Repository**

```bash
git clone https://github.com/Adenegar/Book-Recommendations
```

2. **Install Dependencies:**

Ensure you have Python installed (preferably 3.8+), then install the required packages if you haven't done so already:

```bash
pip install pandas scikit-learn prompt_toolkit tqdm
```

3. **Prepare Your Data:**

Place your book data (e.g., books_data.csv) and user ratings CSV (e.g., Books_rating.csv) in the appropriate folder. I used books/[DATAFILE] from the project's root directory.

4. **Start Using:**

From here open and use the Jupyter notebook or run the main script from the terminal 


Use the python script (from the src folder)
``` bash
python ConsoleBookRecs.py
```