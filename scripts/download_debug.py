import requests
import json

spec = {
    "base_url": "https://solar.datainsight.vn",
    "username": "oee2024@gmail.com",
    "password": "Oee@2124",
    "device_id": "9b745ee0-377d-11f0-af45-2533bc830589",
}


def get_real_keys():
    # 1. Get Token
    auth = requests.post(
        f"{spec['base_url']}/api/auth/login",
        json={"username": spec["username"], "password": spec["password"]},
    )
    token = auth.json()["token"]
    headers = {"X-Authorization": f"Bearer {token}"}

    # 2. Get Keys directly from API
    url = f"{spec['base_url']}/api/plugins/telemetry/DEVICE/{spec['device_id']}/keys/timeseries"
    response = requests.get(url, headers=headers)

    print("AVAILABLE KEYS FOUND ON DEVICE")
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    get_real_keys()
