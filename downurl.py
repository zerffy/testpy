import requests

url = "http://brother-li.top/files/07-18.docx"
response = requests.get(url)

if response.status_code == 200:
    with open("07-18.docx", "wb") as f:
        f.write(response.content)
    print("文件下载成功！")
else:
    print("文件下载失败，HTTP状态码:", response.status_code)
