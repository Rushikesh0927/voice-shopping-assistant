import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
import re
import os

class ProductRecommender:
    """
    ML-Powered NLP Engine for Voice-Based Personalized Shopping Assistant.
    
    Uses a three-stage Machine Learning pipeline:
      Stage 1 — Query Understanding: Extracts attributes (color, size, brand)
                and product type from natural language queries.
      Stage 2 — Category Classifier (LinearSVC): Predicts the most relevant
                product category from the user's voice query.
      Stage 3 — Filtered TF-IDF Similarity: Ranks products using cosine
                similarity, boosted by category match and attribute match scores.
    """
    
    # Known attribute keywords for query understanding
    COLORS = {
        'red', 'blue', 'green', 'yellow', 'black', 'white', 'pink', 'purple',
        'orange', 'brown', 'grey', 'gray', 'navy', 'maroon', 'beige', 'gold',
        'silver', 'cream', 'khaki', 'olive', 'teal', 'coral', 'violet',
        'magenta', 'turquoise', 'indigo', 'peach', 'burgundy', 'tan', 'ivory',
        'charcoal', 'aqua', 'crimson', 'lavender', 'mint', 'rust', 'wine',
        'multicolor', 'multicolour', 'multi'
    }
    
    SIZES = {
        'small', 'medium', 'large', 'xl', 'xxl', 'xxxl', 'xs', 'free',
        'slim', 'regular', 'plus', 'petite', 'tall', 'mini', 'maxi'
    }
    
    GENDERS = {
        'men', 'mens', 'women', 'womens', 'boys', 'girls', 'kids', 'baby',
        'unisex', 'male', 'female', 'man', 'woman', 'boy', 'girl',
        'ladies', 'gents', 'children', 'infant', 'toddler'
    }
    
    def __init__(self, data_path="data/Amazon-Products.csv", max_rows=50000):
        self.data_path = data_path
        self.max_rows = max_rows
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None
        
        # ML classifier
        self.category_classifier = None
        self.category_vectorizer = None
        self.label_encoder = None
        self.unique_categories = None
        
        self._load_and_preprocess_data()
        self._build_tfidf_matrix()
        self._train_category_classifier()

    def _clean_text(self, text):
        """Cleans text for NLP processing."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _load_and_preprocess_data(self):
        """Loads the dataset and creates text features."""
        print("Loading dataset for NLP Engine...")

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")

        df = pd.read_csv(self.data_path, nrows=self.max_rows * 2) 
        df = df.dropna(subset=['name', 'main_category', 'discount_price', 'image'])
        self.df = df.head(self.max_rows).copy()

        # Build the search text: product name + categories
        self.df['search_text'] = (
            self.df['name'].fillna('') + ' ' + 
            self.df['main_category'].fillna('') + ' ' + 
            self.df['sub_category'].fillna('')
        )
        self.df['cleaned_text'] = self.df['search_text'].apply(self._clean_text)
        self.df['cleaned_name'] = self.df['name'].fillna('').apply(self._clean_text)
        
        print(f"Loaded {len(self.df)} products into the recommendation engine.")

    def _build_tfidf_matrix(self):
        """Builds the TF-IDF matrix for similarity search."""
        print("Building TF-IDF matrix...")
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=50000,
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['cleaned_text'])
        print("TF-IDF matrix built successfully.")

    def _train_category_classifier(self):
        """Trains a supervised ML classifier (LinearSVC) to predict product category."""
        print("Training ML Category Classifier (LinearSVC)...")
        
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(self.df['main_category'])
        self.unique_categories = list(self.label_encoder.classes_)
        
        self.category_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=20000,
            sublinear_tf=True
        )
        X = self.category_vectorizer.fit_transform(self.df['cleaned_name'])
        
        self.category_classifier = LinearSVC(max_iter=5000, C=1.0)
        self.category_classifier.fit(X, y)
        
        train_acc = self.category_classifier.score(X, y)
        print(f"ML Classifier trained! Accuracy: {train_acc:.1%}")
        print(f"Categories learned: {len(self.unique_categories)}")

    def _parse_query(self, query):
        """
        Stage 1: Query Understanding.
        
        Extracts structured attributes from natural language queries.
        Example: "yellow shirt for boys" → 
            colors=['yellow'], gender=['boys'], product_terms=['shirt']
        """
        cleaned = self._clean_text(query)
        words = cleaned.split()
        
        # Remove filler words
        filler = {'show', 'me', 'find', 'search', 'for', 'some', 'a', 'an', 
                  'the', 'i', 'want', 'need', 'buy', 'get', 'looking', 'good',
                  'best', 'nice', 'new', 'please', 'can', 'you', 'give',
                  'suggest', 'recommend', 'to', 'with', 'and', 'in', 'of'}
        
        colors_found = []
        sizes_found = []
        genders_found = []
        product_terms = []
        
        for word in words:
            if word in filler:
                continue
            elif word in self.COLORS:
                colors_found.append(word)
            elif word in self.SIZES:
                sizes_found.append(word)
            elif word in self.GENDERS:
                genders_found.append(word)
            else:
                product_terms.append(word)
        
        return {
            'colors': colors_found,
            'sizes': sizes_found,
            'genders': genders_found,
            'product_terms': product_terms,
            'full_query': cleaned,
            'all_words': words
        }

    def _predict_category(self, query):
        """Uses the trained ML model to predict query category."""
        cleaned = self._clean_text(query)
        query_vec = self.category_vectorizer.transform([cleaned])
        
        decision_scores = self.category_classifier.decision_function(query_vec)[0]
        exp_scores = np.exp(decision_scores - np.max(decision_scores))
        probabilities = exp_scores / exp_scores.sum()
        
        top_indices = probabilities.argsort()[-3:][::-1]
        top_categories = {}
        for idx in top_indices:
            cat_name = self.label_encoder.inverse_transform([idx])[0]
            top_categories[cat_name] = float(probabilities[idx])
        
        predicted_idx = probabilities.argmax()
        predicted_cat = self.label_encoder.inverse_transform([predicted_idx])[0]
        confidence = float(probabilities[predicted_idx])
        
        return predicted_cat, confidence, top_categories

    def recommend_products(self, query, top_k=6):
        """
        Three-stage ML pipeline:
          1. Parse query → extract attributes (color, gender, product type)
          2. ML classifier → predict best category
          3. TF-IDF similarity → rank products with attribute boosting
        """
        if not query:
            return []
            
        print(f"\n{'='*60}")
        print(f"QUERY: '{query}'")
        
        # ── Stage 1: Query Understanding ──
        parsed = self._parse_query(query)
        print(f"  Parsed => colors:{parsed['colors']}, gender:{parsed['genders']}, "
              f"product:{parsed['product_terms']}")
        
        # ── Stage 2: ML Category Prediction ──
        predicted_cat, confidence, top_cats = self._predict_category(query)
        print(f"  ML Category: '{predicted_cat}' ({confidence:.0%})")
        
        # ── Stage 3: TF-IDF Similarity + Attribute Boosting ──
        cleaned_query = parsed['full_query']
        query_vector = self.vectorizer.transform([cleaned_query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get the product name column as a numpy array for fast access
        names_lower = self.df['cleaned_name'].values
        cats_lower = self.df['main_category'].str.lower().values
        subcats_lower = self.df['sub_category'].str.lower().values
        
        for idx in range(len(similarities)):
            if similarities[idx] < 0.01:
                continue
                
            product_name = names_lower[idx]
            product_cat = cats_lower[idx]
            
            # ── Category Boost (ML-driven) ──
            if product_cat == predicted_cat.lower():
                similarities[idx] *= 1.8
            
            # ── Attribute Boost: Color ──
            if parsed['colors']:
                color_match = any(c in product_name for c in parsed['colors'])
                if color_match:
                    similarities[idx] *= 2.5  # Strong boost for color match
                else:
                    similarities[idx] *= 0.3  # Heavy penalty for wrong color
            
            # ── Attribute Boost: Gender ──
            if parsed['genders']:
                gender_text = product_name + ' ' + product_cat + ' ' + subcats_lower[idx]
                gender_match = any(g in gender_text for g in parsed['genders'])
                # Also check for related gender terms
                gender_map = {
                    'boys': ['boy', 'boys', 'kids', 'children', 'baby', 'infant'],
                    'girls': ['girl', 'girls', 'kids', 'children', 'baby', 'infant'],
                    'men': ['men', 'mens', 'man', 'male', 'gents'],
                    'women': ['women', 'womens', 'woman', 'female', 'ladies'],
                    'kids': ['kids', 'children', 'boy', 'girl', 'baby', 'infant', 'toddler']
                }
                expanded_match = False
                for g in parsed['genders']:
                    related = gender_map.get(g, [g])
                    if any(r in gender_text for r in related):
                        expanded_match = True
                        break
                
                if gender_match or expanded_match:
                    similarities[idx] *= 1.5
                else:
                    similarities[idx] *= 0.5
            
            # ── Product Term Match in Name ──
            if parsed['product_terms']:
                name_words = set(product_name.split())
                term_matches = sum(1 for t in parsed['product_terms'] if t in name_words)
                # Partial match in full name string (handles plurals like shirt/shirts)
                partial_matches = sum(1 for t in parsed['product_terms'] 
                                      if t in product_name)
                
                if term_matches > 0:
                    similarities[idx] *= (1.0 + 0.5 * term_matches)
                elif partial_matches > 0:
                    similarities[idx] *= (1.0 + 0.3 * partial_matches)
        
        # Get top results
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        recommendations = []
        for idx in top_indices:
            if similarities[idx] > 0.01:
                row = self.df.iloc[idx]
                score = float(min(similarities[idx], 1.0))
                recommendations.append({
                    'name': row['name'],
                    'category': row['main_category'],
                    'sub_category': row['sub_category'],
                    'price': row['discount_price'],
                    'actual_price': row['actual_price'],
                    'rating': str(row['ratings']) if pd.notna(row['ratings']) else "N/A",
                    'image': row['image'],
                    'link': row['link'],
                    'similarity_score': round(score, 4)
                })
                
        print(f"  Results: {len(recommendations)} products found")
        if recommendations:
            print(f"  Top result: {recommendations[0]['name'][:60]}...")
        print(f"{'='*60}\n")
        return recommendations

if __name__ == "__main__":
    recommender = ProductRecommender(data_path="../data/Amazon-Products.csv", max_rows=10000)
    
    test_queries = ["yellow shirt", "red shoes for men", "black watch", 
                    "blue dress for women", "running shoes", "headphones"]
    for q in test_queries:
        print(f"\nQuery: '{q}'")
        results = recommender.recommend_products(q)
        for i, res in enumerate(results, 1):
            print(f"  {i}. [{res['category']}] {res['name'][:60]}... (Score: {res['similarity_score']})")
