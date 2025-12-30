import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
data = pd.read_csv("./ai_data.csv")

# TODO: Model to determine AI company category based on description
data_categories = data.dropna(
    subset=["Description", "Category"]
)  # drop rows with missing Description or Category

# Split data into features X and target y (y = f(X))D
X_desc = data_categories["Description"]  # feature column
y_category = data_categories["Category"]  # target column

# split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X_desc, y_category, test_size=0.2, random_state=42
)

# ML models learn better with numerical data
# Vectorizer learns from training descriptions only
vectorizer = TfidfVectorizer(
    max_features=1000, stop_words="english"
)  # limit to top 1000 features with common English stop words removed

# Fit training data and transform it (learn + convert)
X_train_vec = vectorizer.fit_transform(X_train)

# Transform testing data (convert only to avoid data leakage)
X_test_vec = vectorizer.transform(X_test)

# Train the model
model_category = RandomForestClassifier(n_estimators=100, random_state=42)
model_category.fit(X_train_vec, y_train)

# # Save the trained model and vectorizer (wb = write binary)
# pickle.dump(model_category, open("./Models/model_category.pkl", "wb"))
# pickle.dump(vectorizer, open("./Models/vectorizer_category.pkl", "wb"))

# TODO: Model 2A: Predicting success based on various features

# Create 'Success' column for training success prediction model
data.dropna(inplace=True)       # drop NaN rows
data['Category_median'] = data.groupby('Category')['Upvotes'].transform('median')
data['Success'] = (data['Upvotes'] >= data['Category_median']).astype(int)

# Preparing data for Success model
category_encoder = LabelEncoder()
price_encoder = LabelEncoder()
data['Category_enc'] = category_encoder.fit_transform(data['Category'])         # convert and learn categories to numerical labels
data['Price_enc'] = price_encoder.fit_transform(data['Price'])                  # convert and learn prices to numerical labels

# Vectorizer learns from all description in the cleaned dataset
description_vectorizer = TfidfVectorizer(
    max_features=2000, stop_words="english"
)
description_vec = description_vectorizer.fit_transform(data['Description'])     # convert and learn descriptions to numerical data
