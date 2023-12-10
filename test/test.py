import requests
import pandas as pd


app_url = "https://loans-404215.lm.r.appspot.com/predict"
# app_url = "http://127.0.0.1:8080/predict"


joined_data_sample = pd.read_csv('joined_data.csv')
test_data_sample = pd.read_csv('test_data_sample.csv')

joined_data_clean = joined_data_sample.dropna()
test_data_clean = joined_data_sample.dropna()

sample_test = joined_data_clean.sample(n=1)
#sample_test = test_data_clean.sample(n=1)


sample_test_dict = sample_test.to_dict(orient='records')[0]

response = requests.post(app_url, json=sample_test_dict)


if response.status_code == 200:
    data = response.json()
    print("Loan behavior prediction:", data.get("Loan default prediction"))
else:
    print("Error:", response.status_code, response.text)
