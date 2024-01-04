import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the feedback data
with open("feedback2.json", "r") as file:
    feedback_data = json.load(file)

# Load the keywords
with open("keywords.json", "r") as file:
    keywords = json.load(file)


# Function to find keywords in comments and extract sentiments
def find_keywords(keywords, feedback):
    sentiments = []
    for i in feedback:
        comment = i["comment"].lower()
        sentiment_val = 0
        for j in keywords:
            if j["keyword"] in comment:
                sentiment_val += j["emphasis"] if j["is_negative"] else -j["emphasis"]
        sentiments.append(sentiment_val)
    return sentiments

# Extract sentiments from feedback using keywords
sentiments = find_keywords(keywords, feedback_data)

# Extract comments from feedback
comments = [i["comment"].lower() for i in feedback_data]

# Tokenize comments
tokenizer = Tokenizer()
tokenizer.fit_on_texts(comments)
sequences = tokenizer.texts_to_sequences(comments)
word_index = tokenizer.word_index
max_sequence_length = max(len(seq) for seq in sequences)
data = pad_sequences(sequences, maxlen=max_sequence_length)

# Create a neural network model (similar to the previous example)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(word_index) + 1, 128, input_length=max_sequence_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Convert sentiments to NumPy array
sentiments_array = np.array(sentiments)

# Train the model using both padded sequences and sentiments array
model.fit(data, sentiments_array, epochs=10, batch_size=32)

# Save the model for future use
model.save("sentiment_analysis_model")


# Load the trained model
model = tf.keras.models.load_model("sentiment_analysis_model")

# Load the feedback data
with open("feedback2.json", "r") as file:
    feedback_data = json.load(file)

# Extract comments from feedback
comments = [i["comment"].lower() for i in feedback_data]

# Tokenize and pad the comments
tokenizer = Tokenizer()
tokenizer.fit_on_texts(comments)
sequences = tokenizer.texts_to_sequences(comments)
max_sequence_length = max(len(seq) for seq in sequences)
data = pad_sequences(sequences, maxlen=max_sequence_length)

# Predict sentiments using the trained model
predicted_sentiments = model.predict(data)

# Calculate the total sentiment score
total_sentiment = sum(predicted_sentiments)

print(f"Total Sentiment Score: {total_sentiment}")