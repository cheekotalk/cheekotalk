import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

def get_response(order_number) :
    KEY = os.getenv("CHEEKO_ORDER_API")
    headers = {
        'X-API-Key': KEY,
        'accept': 'application/json',
    }
    response = requests.get(
        'http://cheerupkorea.iptime.org:16741/orders',
        headers=headers,
        params={
            'start': 0,
            'size': 10,
            'filters': json.dumps([{"id": "order_code", "value": order_number}]),
            'filterModes': json.dumps({"order_code": "equal"}),
            'sorting': json.dumps([{"id": "order_date", "desc": True}])
        }
    )
    return response