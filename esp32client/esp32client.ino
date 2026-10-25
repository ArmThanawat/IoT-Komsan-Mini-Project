// ESP32 Client

#include <Arduino.h>
#include <esp_arduino_version.h>
#include <WiFi.h>
#include <PubSubClient.h>

#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEScan.h>
#include <BLEClient.h>
#include <BLEAdvertisedDevice.h>
#include <BLERemoteCharacteristic.h>

#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>

// ================================
// WiFi and MQTT
// ================================
const char* WIFI_SSID = "Liztsz";
const char* WIFI_PASSWORD = "4c697a74737a";

const char* MQTT_BROKER = "wss://broker.hivemq.com";
constexpr uint16_t MQTT_PORT = 1883;

const char* MQTT_LED_TOPIC =
  "psu/gesture/demo01/led";

const char* MQTT_STATUS_TOPIC =
  "psu/gesture/demo01/client/status";

WiFiClient wifiClient;
PubSubClient mqtt(wifiClient);

uint32_t lastWifiAttempt = 0;
uint32_t lastMqttAttempt = 0;

constexpr uint32_t WIFI_RETRY_MS = 10000;
constexpr uint32_t MQTT_RETRY_MS = 5000;

// ================================
// BLE
// ================================
#define SERVICE_UUID \
  "7c6a0001-14b7-4a1c-8ca6-a0e527117203"

#define GESTURE_UUID \
  "7c6a0002-14b7-4a1c-8ca6-a0e527117203"

static BLEUUID serviceUUID(SERVICE_UUID);
static BLEUUID gestureUUID(GESTURE_UUID);

BLEClient* bleClient = nullptr;
BLEAdvertisedDevice* foundServer = nullptr;

volatile bool connected = false;

// ================================
// LED PWM
// ================================
constexpr int LED_PIN = 26;
constexpr uint32_t PWM_FREQ = 5000;
constexpr uint8_t PWM_RESOLUTION = 8;
constexpr uint8_t PWM_CHANNEL = 0;

int brightness = 0;
int savedBrightness = 128;
bool armed = false;

// ================================
// Gesture data
// ================================
struct GesturePacket {
  uint32_t sequence;
  int label;
};

QueueHandle_t gestureQueue = nullptr;

uint32_t lastSequence = 0;

const char* gestureNames[] = {
  "TILT_LEFT",
  "TILT_RIGHT",
  "UP",
  "DOWN",
  "IDLE"
};

// ================================
// PWM functions
// ================================
bool setBrightness(int value) {
  value = constrain(value, 0, 255);

#if ESP_ARDUINO_VERSION_MAJOR >= 3
  return ledcWrite(LED_PIN, value);
#else
  ledcWrite(PWM_CHANNEL, value);
  return true;
#endif
}

void initializeLED() {
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

#if ESP_ARDUINO_VERSION_MAJOR >= 3
  if (!ledcAttach(
        LED_PIN,
        PWM_FREQ,
        PWM_RESOLUTION)) {
    Serial.println("ERROR: PWM init failed");
    while (true) delay(1000);
  }
#else
  if (ledcSetup(
        PWM_CHANNEL,
        PWM_FREQ,
        PWM_RESOLUTION)
      <= 0) {
    Serial.println("ERROR: PWM init failed");
    while (true) delay(1000);
  }

  ledcAttachPin(LED_PIN, PWM_CHANNEL);
#endif

  if (!setBrightness(0)) {
    Serial.println("ERROR: PWM write failed");
    while (true) delay(1000);
  }
}

// ================================
// WiFi and MQTT
// ================================
void initializeWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  lastWifiAttempt = millis();

  Serial.println("WiFi connection started");
}

void initializeMQTT() {
  mqtt.setServer(MQTT_BROKER, MQTT_PORT);
  mqtt.setBufferSize(256);
  mqtt.setKeepAlive(30);
  mqtt.setSocketTimeout(2);
}

void maintainNetwork() {
  uint32_t now = millis();

  if (WiFi.status() != WL_CONNECTED) {
    if (now - lastWifiAttempt >= WIFI_RETRY_MS) {
      lastWifiAttempt = now;

      Serial.println("Retrying WiFi...");
      WiFi.disconnect();
      WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    }
    return;
  }

  if (mqtt.connected()) {
    mqtt.loop();
    return;
  }

  if (now - lastMqttAttempt < MQTT_RETRY_MS) {
    return;
  }

  lastMqttAttempt = now;

  char clientId[40];

  snprintf(
    clientId,
    sizeof(clientId),
    "gesture-client-%llX",
    static_cast<unsigned long long>(ESP.getEfuseMac()));

  Serial.println("Connecting MQTT...");

  if (mqtt.connect(
        clientId,
        MQTT_STATUS_TOPIC,
        0,
        true,
        "offline")) {

    mqtt.publish(
      MQTT_STATUS_TOPIC,
      "online",
      true);

    Serial.println("MQTT connected");
  } else {
    Serial.printf(
      "MQTT failed, state=%d\n",
      mqtt.state());
  }
}

// ================================
// Gesture control
// ================================
bool applyGesture(int label) {
  if (label == 4) {
    armed = true;
    return false;
  }

  if (label < 0 || label > 3) {
    return false;
  }

  // ON/OFF require idle first.
  // UP/DOWN can repeat.
  if (!armed && label != 2 && label != 3) {
    return false;
  }

  armed = false;

  switch (label) {
    case 0:
      if (brightness > 0)
        savedBrightness = brightness;

      brightness = 0;
      break;

    case 1:
      brightness = savedBrightness;
      break;

    case 2:
      brightness = min(255, brightness + 32);

      if (brightness > 0)
        savedBrightness = brightness;
      break;

    case 3:
      if (brightness > 0)
        savedBrightness = brightness;

      brightness = max(0, brightness - 32);

      if (brightness > 0)
        savedBrightness = brightness;
      break;
  }

  return true;
}

// ================================
// BLE callbacks
// ================================
class ClientCallbacks : public BLEClientCallbacks {
  void onConnect(BLEClient* client) override {
    connected = true;
    Serial.println("BLE connected");
  }

  void onDisconnect(BLEClient* client) override {
    connected = false;
    Serial.println("BLE disconnected");
  }
};

class ScanCallbacks : public BLEAdvertisedDeviceCallbacks {
  void onResult(BLEAdvertisedDevice device) override {
    if (foundServer != nullptr)
      return;

    if (device.haveServiceUUID() && device.isAdvertisingService(serviceUUID)) {

      foundServer = new BLEAdvertisedDevice(device);
      BLEDevice::getScan()->stop();
    }
  }
};

// ================================
// BLE notification receive
// ================================
void onGestureNotify(
  BLERemoteCharacteristic* characteristic,
  uint8_t* data,
  size_t length,
  bool isNotify) {
  if (length == 0 || length >= 20)
    return;

  char message[20];

  memcpy(message, data, length);
  message[length] = '\0';

  unsigned long sequence = 0;
  int label = -1;
  int used = 0;

  if (sscanf(
        message,
        "%lu,%d%n",
        &sequence,
        &label,
        &used)
      != 2) {
    return;
  }

  if (used != static_cast<int>(length) || label < 0 || label >= 5) {
    return;
  }

  GesturePacket packet;
  packet.sequence = static_cast<uint32_t>(sequence);
  packet.label = label;

  // Process outside BLE callback
  if (gestureQueue != nullptr) {
    xQueueSend(gestureQueue, &packet, 0);
  }
}

// ================================
// Connect BLE Server
// ================================
bool connectToServer() {
  if (foundServer == nullptr)
    return false;

  Serial.println("Connecting BLE Server...");

  if (!bleClient->connect(foundServer)) {
    Serial.println("BLE connection failed");
    return false;
  }

  BLERemoteService* service =
    bleClient->getService(serviceUUID);

  if (service == nullptr) {
    Serial.println("BLE service not found");
    bleClient->disconnect();
    return false;
  }

  BLERemoteCharacteristic* characteristic =
    service->getCharacteristic(gestureUUID);

  if (characteristic == nullptr) {
    Serial.println("Gesture characteristic not found");
    bleClient->disconnect();
    return false;
  }

  if (!characteristic->canNotify()) {
    Serial.println("BLE notify not supported");
    bleClient->disconnect();
    return false;
  }

  // This API returns void in your BLE library.
  characteristic->registerForNotify(onGestureNotify);

  lastSequence = 0;
  armed = false;
  xQueueReset(gestureQueue);

  connected = true;

  Serial.println("BLE notifications registered");
  return true;
}

// ================================
// Scan BLE Server
// ================================
void scanForServer() {
  if (foundServer != nullptr) {
    delete foundServer;
    foundServer = nullptr;
  }

  Serial.println("Scanning BLE Server...");

  BLEScan* scanner = BLEDevice::getScan();

  scanner->start(5, false);
  scanner->clearResults();

  if (foundServer == nullptr) {
    Serial.println("BLE Server not found");
  }
}

// ================================
// Publish LED state to MQTT
// ================================
void publishLEDState(
  const GesturePacket& packet,
  const char* status) {
  char payload[192];

  snprintf(
    payload,
    sizeof(payload),
    "{\"sequence\":%lu,"
    "\"gesture_id\":%d,"
    "\"gesture\":\"%s\","
    "\"status\":\"%s\","
    "\"pwm\":%d,"
    "\"brightness_percent\":%d}",
    static_cast<unsigned long>(packet.sequence),
    packet.label,
    gestureNames[packet.label],
    status,
    brightness,
    (brightness * 100) / 255);

  if (mqtt.connected()) {
    bool sent = mqtt.publish(
      MQTT_LED_TOPIC,
      payload);

    Serial.printf(
      "MQTT LED | %s | %s\n",
      sent ? "QUEUED" : "FAILED",
      payload);
  } else {
    Serial.println(
      "MQTT offline: LED state not published");
  }
}

// ================================
// Process received gesture
// ================================
void processGesture(const GesturePacket& packet) {
  if (packet.sequence <= lastSequence)
    return;

  if (lastSequence != 0 && packet.sequence != lastSequence + 1) {
    Serial.println(
      "WARNING: Missing BLE packets");
  }

  lastSequence = packet.sequence;

  int oldBrightness = brightness;
  bool accepted = applyGesture(packet.label);

  const char* status =
    packet.label == 4 ? "ARMED" : accepted ? "EXECUTED"
                                           : "WAIT_FOR_IDLE";

  if (accepted) {
    if (!setBrightness(brightness)) {
      Serial.println("ERROR: PWM write failed");

      // Attempt to restore prior LED state
      brightness = oldBrightness;
      setBrightness(brightness);
      status = "PWM_ERROR";
    }
  }

  Serial.printf(
    "Seq=%lu | Gesture=%s | Status=%s | PWM=%d->%d\n",
    static_cast<unsigned long>(packet.sequence),
    gestureNames[packet.label],
    status,
    oldBrightness,
    brightness);

  publishLEDState(packet, status);
}

// ================================
// Setup
// ================================
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("ESP32 Gesture Client");
  Serial.println("BLE + LED PWM + MQTT");

  initializeLED();

  gestureQueue = xQueueCreate(
    16,
    sizeof(GesturePacket));

  if (gestureQueue == nullptr) {
    Serial.println("ERROR: Queue creation failed");
    while (true) delay(1000);
  }

  BLEDevice::init("ESP32-Gesture-Client");

  bleClient = BLEDevice::createClient();

  bleClient->setClientCallbacks(
    new ClientCallbacks());

  BLEScan* scanner = BLEDevice::getScan();

  scanner->setAdvertisedDeviceCallbacks(
    new ScanCallbacks());

  scanner->setActiveScan(true);
  scanner->setInterval(100);
  scanner->setWindow(80);

  initializeWiFi();
  initializeMQTT();

  Serial.println("LED GPIO26 ready");
}

// ================================
// Main loop
// ================================
void loop() {
  maintainNetwork();

  if (!connected) {
    armed = false;

    if (bleClient->isConnected()) {
      bleClient->disconnect();
    }

    xQueueReset(gestureQueue);

    scanForServer();

    if (foundServer != nullptr) {
      connectToServer();

      delete foundServer;
      foundServer = nullptr;
    }

    if (!connected) {
      delay(200);
    }

    return;
  }

  GesturePacket packet;

  while (xQueueReceive(
           gestureQueue,
           &packet,
           0)
         == pdTRUE) {
    processGesture(packet);
  }

  delay(10);
}
