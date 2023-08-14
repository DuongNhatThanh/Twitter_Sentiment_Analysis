# Twitter Sentiment Analysis with Support Vector Machines (SVM)

This code performs sentiment analysis on Twitter data using Support Vector Machines (SVM) with Bag-of-Words (BOW) and Term Frequency-Inverse Document Frequency (tfidf) representations. The code is written in Python and utilizes popular NLP libraries.

## Preprocessing

The `preprocess_text()` function is defined to preprocess the input text. It removes HTML tags, non-alphabetic characters, converts the text to lowercase, tokenizes it, removes stop words, and lemmatizes the words using the NLTK library.

## Libraries Used

The code imports necessary libraries, including NLTK, NumPy, scikit-learn, and Gensim.

## Sentiment Analysis

Two vectorizers are used: CountVectorizer for BOW representation and TfidfVectorizer for tfidf representation.

The Twitter data is divided into training and testing sets (`X_train`, `X_test`, `y_train`, `y_test`).

The SVM classifiers (`clf1` and `clf2`) are trained on the training data with BOW and tfidf representations, respectively.

The classifiers predict the sentiment labels for the test data (`y_predict_count`, `y_predict_tfidf`).

The accuracy of the predictions is calculated using scikit-learn's `accuracy_score` function for both Count and tfidf representations.

The prediction accuracies for BOW and tfidf are printed at the end.

## Execution

To run the code, ensure you have the required libraries installed (NLTK, NumPy, scikit-learn, Gensim). The Twitter data (`X_train`, `X_test`, `y_train`, `y_test`) should be properly formatted before executing the code. The result will show the prediction accuracies for sentiment analysis using BOW and tfidf representations with SVM.

## Note

The `preprocess_text()` function can be customized further for specific text cleaning needs. Additionally, the accuracy may vary based on the dataset used.
