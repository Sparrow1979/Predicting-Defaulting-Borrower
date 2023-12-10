import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import pickle
import category_encoders as ce

import logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    loaded_pipeline = pickle.load(f)


@app.route('/predict', methods=['POST'])
def loan_default_predict():
    try:
        sample_test = request.json

        sample_test_X = pd.DataFrame([sample_test])

        y_pred = loaded_pipeline.predict(sample_test_X)

        logging.info("test: %s", y_pred)

        if y_pred == 0:
            result = "Normal loan application"
        else:
            result = "Potentially defaulted loan application"

        logging.info("Loan default prediction: %s", result)

        return jsonify({"Loan default prediction": result})

    except Exception as e:
        logging.error("Error: %s", str(e))
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
