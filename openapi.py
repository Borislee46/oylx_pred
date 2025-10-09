import requests

url = "http://menshen.test.xdf.cn/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer 0a33aaf283384e9e9ecc120916c6a588",
}

data = {
    "model": "doubao-seed-1.6-flash",
    "messages": [
        {
            "content": [
                {"text": "你好", "type": "text"},
                {
                    "image_url": {
                        "url": "https://ark-project.tos-cn-beijing.ivolces.com/images/view.jpeg"
                    },
                    "type": "image_url",
                },
            ],
            "role": "user",
        }
    ],
    "thinking": {"type": "enabled"},
}

response = requests.post(url, headers=headers, json=data)

print(response.json())
