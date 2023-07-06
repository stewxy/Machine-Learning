import requests
response = requests.get("https://cat-fact.herokuapp.com/facts")
print(response.status_code)
print(response.json())
