import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
data = pd.read_csv("./ai_data.csv")

# TODO: FUNCTION TO DETERMINE CATEGORY BASED ON DESCRIPTION
data_categories = data.dropna(subset=["Description", "Category"])       # drop rows with missing Description or Category

# Split data into features X and target y (y = f(X))
X_desc = data_categories['Description']                                 # feature column
y_category = data_categories["Category"]                                # target column

# split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_desc, y_category, test_size=0.2, random_state=42)

# ML models learn better with numerical data
vectorizer = TfidfVectorizer(max_features=100, stop_words='english')                # limit to top 100 features with common English stop words removed

# Fit training data and transform it (learn + convert)
X_train_vec = vectorizer.fit_transform(X_train)

# Transform testing data (convert only)
X_test_vec = vectorizer.transform(X_test)





