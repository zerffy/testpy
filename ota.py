import urllib3
import re
import hashlib
import json
import time
import paho.mqtt.client as mqtt
from threading import Timer
# import environ
import requests

# env = environ.Env()
# environ.Env.read_env('.env')
# ts_db_url = env("ts_rest_api") + "/rest/sql"


MQTTHOST = "47.93.1.95"
MQTTPORT = 1883
mqtt_client = mqtt.Client()


# 连接MQTT服务器
def on_mqtt_connect():
    mqtt_client.connect(MQTTHOST, MQTTPORT, 60)
    mqtt_client.loop_start()


def on_version(device_id,url):
    try:
        # Mock response data (replace this with actual response data if available)
        mock_response_data = {
            'data': {
                'data': {
                    # 'sv': sv,
                    # 'hv': hv,
                    'url': url,
                    'size': 1024  # Replace this with the actual file size in bytes
                }
            }
        }

        # Simulate the HTTP request and response
        try:
            # Here, we are not making an actual request. Instead, we are using the mock response data.
            r = json.loads(json.dumps(mock_response_data))
        except Exception as e:
            print("Error occurred during HTTP request:", e)
            return

        if 'data' not in r['data']:
            return

        r = r['data']['data']
        if 'http' not in r['url']:
            return

        v = {}
        # v['version'] = r['sv']
        # v['hardVersion'] = r['hv']
        v['url'] = r['url']
        # v['channel'] = 'cv' + r['hv']
        # m = hashlib.md5(str(r['url'].replace(" ", "") + "sengainupdate").encode("utf-8"))
        # v['key'] = m.hexdigest()
        # v['size'] = str(r['size'])
        push_str = "UPUP" + str(json.dumps(v)).replace("http://", "http").replace(",", ", ").replace(":", ": ").\
            replace("http", "http://").replace("  ", " ")
        print(device_id, push_str, time.time())
        time.sleep(2)
        mqtt_client.publish('zhang/' + device_id, push_str, 0)
        print("发布成功" + "*" * 20 + "\n")

    except Exception as e:
        print(e)


def main():
    on_mqtt_connect()
    on_version("003", "http://tsingze.com/files/test.txt")


if __name__ == '__main__':
    main()
