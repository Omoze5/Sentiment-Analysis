## Sentiment Analysis App

This project implements a sentiment analysis model to classify movie reviews as either positive or negative. It uses a Flask web application to provide a user-friendly interface where users can input text and receive a sentiment prediction. The app is deployed on Koyeb, ensuring scalability and ease of use.

## Dataset
The Large Movie Review Dataset,contains 25,000 movie reviews split evenly between positive and negative labels for training and 25,000 for testing. It was preprocessed and used to train a classification model for sentiment analysis.

## Model Training
### Preprocessing
* Text cleaning: Tokenization, removal of stop word and punctuation, lemmatization using SpaCy.
* Vectorization: The reviews were converted to numerical data using TfidfVectorizer from scikit-learn.

## Model Selection
The classification model used for this task is the Multinomial Naive Bayes classifier. It was chosen due to its effectiveness in text-based classification problems.

## Flask Application

### Routes and Endpoints
* / (GET): Home page with a form for inputting a movie review.
* /predict (POST): Accepts user input from the form and returns the sentiment prediction.
  
## Input and Output

* Input: Movie review text (string).
* Output: Predicted sentiment: 'Positive' or 'Negative'.

## Deployment to Koyeb
The app is deployed using Koyeb, a serverless deployment platform. This section explains the steps to deploy the app:

* Create a Koyeb account and link the repository.
* Configure the service and deployment settings.
* Deploy the Flask app by selecting Python as the runtime and setting environment variables. Find the link to the web interface [Sentiment_Analysis](http://permanent-philippa-omicsdata-a103dcc0.koyeb.app/)

## Installation
* Python 3.8+
* Flask
* Dependencies listed in requirements.txt

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Citation

When using this dataset, please cite the following ACL 2011 paper:

[Learning Word Vectors for Sentiment Analysis (Maas et al., 2011)](https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.bib)









