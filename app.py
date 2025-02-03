import tkinter as tk
from tkinter import messagebox
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
import os

# Simulated IMDB dataset
data = {
    "review": [
        "Amazing movie with great acting!", "Horrible film, worst ever!", 
        "Loved it, would watch again!", "Terrible experience, regret watching!",
        "A masterpiece, highly recommended!", "Awful script, waste of time!"
    ],
    "sentiment": [1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# Split data
train_data, test_data = train_test_split(df, test_size=0.3, random_state=42)

# Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_data["review"])
X_train = pad_sequences(tokenizer.texts_to_sequences(train_data["review"]), maxlen=10)
y_train = np.array(train_data["sentiment"])
X_test = pad_sequences(tokenizer.texts_to_sequences(test_data["review"]), maxlen=10)
y_test = np.array(test_data["sentiment"])

# Tkinter UI
root = tk.Tk()
root.title("The Rogue Reviewer - A Data Poisoning Adventure")

def poison_data():
    global train_data
    poisoned_reviews = random.sample(range(len(train_data)), k=1)  # Inject one poisoned entry
    train_data.iloc[poisoned_reviews, 1] = 1 - train_data.iloc[poisoned_reviews, 1]  # Flip sentiment
    messagebox.showinfo("Poisoning Complete", "The dataset has been poisoned!")

def train_model():
    model = Sequential([
        Embedding(input_dim=5000, output_dim=32, input_length=10),
        LSTM(32, return_sequences=False),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=2, verbose=1)
    
    predictions = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, predictions)
    model.save("bi_lstm_model.h5")  # Save the model
    messagebox.showinfo("Model Training", f"LSTM Model trained with accuracy: {accuracy:.2f} and saved successfully!")

def open_google_colab():
    os.system("start https://colab.research.google.com/")

# Buttons for user interaction
btn_poison = tk.Button(root, text="Inject Poisoned Data", command=poison_data)
btn_poison.pack(pady=10)

btn_train = tk.Button(root, text="Train Bi-LSTM Model", command=train_model)
btn_train.pack(pady=10)

btn_colab = tk.Button(root, text="Open Google Colab", command=open_google_colab)
btn_colab.pack(pady=10)

root.mainloop()
