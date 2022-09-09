from flask import Flask, request, jsonify
import dill
import pickle

from tensorflow import keras
from keras.utils.data_utils import pad_sequences
from statistics import mode
import logging

from src.train import preprocess_text

# # Create logs directory if id doesn't exist
# Path("/logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename='logs/record.log', level=logging.DEBUG)

app = Flask("Product_Categorization")


def predict_sample(text):
    encoded_text = tokenizer.texts_to_sequences(text)

    padded_docs = pad_sequences(encoded_text, maxlen=100, padding='post')

    pred = model.predict(padded_docs).tolist()
    threshold = sum(pred[0]) / len(pred[0])
    for i in range(len(pred[0])):
        if pred[0][i] < threshold:
            pred[0][i] = 0
        else:
            pred[0][i] = 1

    return vectorizer.inverse_transform(pred)[0]


@app.route("/prediction", methods=['GET', 'POST'])
def product_categorize():
    try:
        query = request.get_json()["product_name"]
        prediction = predict_sample(preprocess_text(query))

        if len(prediction) < 3:
            categories = prediction[0] + " > " + prediction[1]

        else:
            # you can return more if you like
            categories = prediction[0] + " > " + prediction[1] + " > " + prediction[2]

        return jsonify({"categories": categories.strip()})
    except Exception as e:
        app.logger.warning("User input is not entered" + "\n" + e)


if __name__ == "__main__":
    # load tokenizer
    with open('tokenizer/tokenizer.pickle', 'rb') as tkn:
        tokenizer = pickle.load(tkn)

    # with open('vectorizer/vectorizer.pickle', 'rb') as vec:
    vectorizer = dill.load(open('vectorizer/vectorizer.pickle', 'rb'))

    model = keras.models.load_model('models/model.h5')

    # Launch the Flask dev server
    app.run(host="localhost", debug=False)
