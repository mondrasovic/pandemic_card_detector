import requests

url = "http://172.17.0.2:5000/predict-image"
image_path = "epidemic.jpg"

with open(image_path, "rb") as file:
    response = requests.post(url, files={"image": file})

if response.ok:
    print("Image uploaded successfully")
    print(response.json())
else:
    print("Error when processing image")
