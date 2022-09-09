import argparse
import string
import re
import dill
import pickle
import matplotlib.pyplot as plt

from numpy import asarray, zeros
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score

from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences
from keras.layers.core import Dropout
from keras.layers import Embedding
from keras.layers import LSTM, Dense
from keras.layers import BatchNormalization


##############################################################
def parse_args():
    """
    Takes parameter from user for training
    :return: Inputs
    """
    # print("Inside: parse_args()")
    parser = argparse.ArgumentParser(
        description="Train Neural Network for product categorization"
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        help="Learning Rate parameter for the Model",
    )
    parser.add_argument(
        "--model_name",
        default="model.h5",
        help="Name of the saved models. Please enter with its extension(ex.: .h5, .pkl)",
    )
    parser.add_argument(
        "--model_dir",
        default="models",
        help="The output directory of the saved models during training",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epoch",
        default=5,
        help="Step for training",
    )
    parser.add_argument(
        "--train_size",
        default=0.7,
        help="Proportion of data which will be used in training",
    )

    # arg parsing debug
    # print(vars(parser.parse_args()))
    return parser.parse_args()


def preprocess_text(sen):
    # Remove punctuations
    sentence = sen.translate(str.maketrans('', '', string.punctuation))

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    # lower the sentence
    sentence = sentence.lower()

    return sentence


def plot_result(history, item):
    """
    Draws loss and categorical accuracy graph

    :param history: Includes trained model information
    :param item: graph name
    :return: None
    """
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


def main():
    """
    Main function to start training
    :return: None
    """

    args = parse_args()

    # Load data
    data = pd.read_csv("../data/categories.csv")
    del data["Unnamed: 0"]

    data = data.sample(frac=1).reset_index()
    del data["index"]

    # Split X and y
    X = data["product_name"].apply(preprocess_text)
    y = data["categories"]

    # Initial train and test split.
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=float(args.train_size), stratify=y.values,
                                                        shuffle=True, random_state=26)

    print("Shape of the x_train: ", x_train.shape)
    print("Shape of the x_test: ", x_test.shape)
    print("Shape of the y_train: ", y_train.shape)
    print("Shape of the y_test: ", y_test.shape)

    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(","), binary="true")
    y_train = vectorizer.fit_transform(y_train).toarray()
    y_test = vectorizer.transform(y_test).toarray()

    # Save vectorizer
    dill.dump(vectorizer, open('../vectorizer/vectorizer.pickle', 'wb'))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_train)

    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)

    vocab_size = len(tokenizer.word_index) + 1
    print("Vocab size: ", vocab_size)

    maxlen = 100

    x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)
    x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)

    # Save tokenizer
    with open('../tokenizer/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    embeddings_dictionary = dict()

    glove_file = open('../glove.6B.100d.txt', encoding="utf8")

    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    glove_file.close()

    embedding_matrix = zeros((vocab_size, 100))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

    # Build model
    model = keras.Sequential()
    # Configuring the parameters
    model.add(Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False))
    # model.add(Embedding(vocab_size, 100))
    model.add(LSTM(128, return_sequences=True))
    model.add(BatchNormalization())
    # Adding a dropout layer
    model.add(Dropout(0.4))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    # Adding a dense output layer with sigmoid activation
    model.add(Dense(y_train.shape[1], activation='sigmoid'))

    print("Model Summary:")
    print(model.summary())

    metrics = [
        "categorical_accuracy",
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
    ]

    # Define optimizer
    optimizer = keras.optimizers.Adam(learning_rate=float(args.learning_rate))

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)

    history = model.fit(x_train, y_train,
                        epochs=int(args.epoch),
                        verbose=1,
                        validation_split=0.3,
                        batch_size=int(args.batch_size))

    print("Training finished.")
    
    model.save(f"../{args.model_dir}/{args.model_name}")
    print("Model is saved.")

    plot_result(history, "loss")
    plot_result(history, "categorical_accuracy")

    predictions = model.predict([x_test])
    thresholds = [0.01, 0.02, 0.03]

    print("Evaluation for threshold.")
    print("---------------------------")
    for val in thresholds:
        print("For threshold: ", val)
        pred = predictions.copy()

        pred[pred >= val] = 1
        pred[pred < val] = 0

        precision = precision_score(y_test, pred, average='micro')
        recall = recall_score(y_test, pred, average='micro')
        f1 = f1_score(y_test, pred, average='micro')

        print("Choose the threshold value by looking the results above.")

        print("Micro-average quality numbers")
        print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))


if __name__ == "__main__":
    main()
