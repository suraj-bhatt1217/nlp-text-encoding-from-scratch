from collections import defaultdict
import numpy as np

# Define the corpus and their corresponding classes
corpus = [
    ("i m happy because i am learning NLP", "happy"),
    ("i m happy", "happy"),
    ("i m sad because i am not learning NLP", "sad"),
    ("i m sad", "sad"),
]


# Function to tokenize text
def tokenize(text):
    return text.split()


# Example tokenization
tokenized_corpus = [(tokenize(sentence), label) for sentence, label in corpus]

# Initialize the frequency dictionary
frequency_dict = defaultdict(int)

# Update the frequency dictionary with word counts for each class
for tokens, label in tokenized_corpus:
    for token in tokens:
        frequency_dict[(token, label)] += 1

# Print the frequency dictionary
for key, count in frequency_dict.items():
    print(f"{key}: {count}")


# Define the encoding function
def encode_tweet(tweet, freq_dict, pos_label="happy", neg_label="sad"):
    # Tokenize the tweet
    tokens = tweet.split()

    # Initialize sums for positive and negative frequencies
    sum_positive = 0
    sum_negative = 0

    # Iterate over each token
    for token in tokens:
        # Add frequency of the token in the positive class
        sum_positive += freq_dict.get((token, pos_label), 0)

        # Add frequency of the token in the negative class
        sum_negative += freq_dict.get((token, neg_label), 0)

    # Create the encoding
    encoding = [1, sum_positive, sum_negative]

    return encoding


#
# List of tweets to encode
tweets = [
    "i am happy and learning NLP",
    "i am sad and not learning NLP",
    "i am learning a lot about NLP",
    "sad day, nothing is going well",
]

# Encode each tweet and store in a list
encodings = [encode_tweet(tweet, frequency_dict) for tweet in tweets]

# Convert the list of encodings to a numpy matrix
encoding_matrix = np.array(encodings)  # X matrix

# Print the matrix
print("Encoding Matrix:")
print(encoding_matrix)
