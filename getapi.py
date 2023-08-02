import requests

url = 'http://127.0.0.1:3000/updata_version/'
data = {"device_id": "002",  "url": "http://tsingze.com/files/test.txt"}
response = requests.post(url, data=data)

print(response.json())