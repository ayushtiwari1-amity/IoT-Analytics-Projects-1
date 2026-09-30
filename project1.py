import json
import random
import sys
import time
import threading
import numpy as np
import paho.mqtt.client as mqtt
from sklearn.ensemble import IsolationForest

BROKER_URL = "broker.hivemq.com"
BROKER_PORT = 1883
TOPIC_TELEMETRY = "factory/pump_01/telemetry"
TOPIC_ALERTS = "factory/pump_01/alerts"
MQTT_PROTOCOL = mqtt.MQTTv311


class IoTAnalyticsEngine:
    def __init__(self):
        print("⚙️ [Analytics Engine] Initializing Machine Learning Model...")
        self.model = IsolationForest(contamination=0.08, random_state=42)
        self._train_baseline_model()
        
        self.client = mqtt.Client(
            client_id="smart_factory_analytics_engine",
            protocol=MQTT_PROTOCOL
        )
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def _train_baseline_model(self):
        np.random.seed(42)
        healthy_temp = np.random.normal(65.0, 2.0, (600, 1))
        healthy_vib = np.random.normal(0.05, 0.01, (600, 1))
        healthy_press = np.random.normal(40.0, 1.0, (600, 1))

        baseline_dataset = np.hstack([healthy_temp, healthy_vib, healthy_press])
        self.model.fit(baseline_dataset)
        print("✅ [Analytics Engine] Baseline Isolation Forest Model Trained.")

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"✅ [Analytics Engine] Connected to MQTT Broker ({BROKER_URL})")
            client.subscribe(TOPIC_TELEMETRY)
            print(f"📡 [Analytics Engine] Subscribed to Topic: '{TOPIC_TELEMETRY}'\n")

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            temp = payload.get("temperature_c", 0.0)
            vib = payload.get("vibration_g", 0.0)
            press = payload.get("pressure_psi", 0.0)
            device_id = payload.get("device_id", "Unknown")

            features = np.array([[temp, vib, press]])
            prediction = self.model.predict(features)[0]

            if prediction == -1:
                status = "🚨 ANOMALY DETECTED"
                self._dispatch_alert(payload)
            else:
                status = "✅ NORMAL"

            print(f"[STREAM] Device: {device_id} | Status: {status} | "
                  f"Temp: {temp:.2f}°C | Vib: {vib:.4f}g | Press: {press:.2f} PSI")

        except Exception as e:
            print(f"❌ [Analytics Engine Error]: {e}")

    def _dispatch_alert(self, anomaly_data):
        alert_payload = {
            "alert_type": "CRITICAL_EQUIPMENT_FAULT",
            "device_id": anomaly_data.get("device_id"),
            "timestamp": anomaly_data.get("timestamp"),
            "metrics": anomaly_data
        }
        self.client.publish(TOPIC_ALERTS, json.dumps(alert_payload))
        print(f"  └─> ⚠️ ALERT DISPATCHED TO BROKER: {TOPIC_ALERTS}")

    def start(self):
        self.client.connect(BROKER_URL, BROKER_PORT, keepalive=60)
        self.client.loop_forever()


class IoTSensorNode:
    def __init__(self, device_id="pump_01"):
        self.device_id = device_id
        self.client = mqtt.Client(
            client_id=f"sensor_node_{device_id}",
            protocol=MQTT_PROTOCOL
        )

    def start(self):
        self.client.connect(BROKER_URL, BROKER_PORT, keepalive=60)
        print(f"📡 [Sensor Node] Publishing Telemetry Stream for '{self.device_id}'...")

        try:
            while True:
                temperature = random.normalvariate(65.0, 2.0)
                vibration = random.normalvariate(0.05, 0.01)
                pressure = random.normalvariate(40.0, 1.0)

                if random.random() < 0.15:
                    vibration *= random.uniform(3.5, 6.0)
                    temperature += random.uniform(20.0, 35.0)

                payload = {
                    "device_id": self.device_id,
                    "timestamp": round(time.time(), 3),
                    "temperature_c": round(temperature, 2),
                    "vibration_g": round(vibration, 4),
                    "pressure_psi": round(pressure, 2)
                }

                self.client.publish(TOPIC_TELEMETRY, json.dumps(payload))
                time.sleep(1.0)

        except KeyboardInterrupt:
            self.client.disconnect()


if __name__ == "__main__":
    print("==========================================================")
    print("     MODULE V: REAL-TIME IOT ANOMALY DETECTION SYSTEM     ")
    print("==========================================================\n")

    engine = IoTAnalyticsEngine()
    engine_thread = threading.Thread(target=engine.start, daemon=True)
    engine_thread.start()

    time.sleep(2)

    sensor_node = IoTSensorNode(device_id="pump_01")
    try:
        sensor_node.start()
    except KeyboardInterrupt:
        sys.exit(0)