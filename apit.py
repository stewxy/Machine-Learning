import requests
import json

response = requests.get("https://cat-fact.herokuapp.com/facts")
print(response.status_code)
print(response.json())

#convert python object to string and sort
def jprint(obj):
    text=json.dumps(obj, sort_keys=True, indent=4)
    print(text)
jprint(response.json())

#print([item for item in response.json() if item["_id"] == "58e00b5f0aac31001185ed24"])

'''
allText=[]
for i in response.json():
    allText.append(i["text"])
print(allText)


for i in response.json():
    print(i["text"])
'''