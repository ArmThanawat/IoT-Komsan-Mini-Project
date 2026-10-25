// ESP32 Server

#include <Arduino.h>
#include <Wire.h>
#include <math.h>
#include <WiFi.h>
#include <PubSubClient.h>

#include "MPU6050Simple.h"
#include "MotionFeatures.h"
#include "model.h"

#include <tflm_esp32.h>
#include <eloquent_tinyml.h>

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ================================
// WiFi and MQTT
// ================================
const char* WIFI_SSID = "Liztsz";
const char* WIFI_PASSWORD = "4c697a74737a";

const char* MQTT_BROKER = "wss://broker.hivemq.com";
constexpr uint16_t MQTT_PORT = 1883;

const char* MQTT_GESTURE_TOPIC =
  "psu/gesture/demo01/prediction";

const char* MQTT_STATUS_TOPIC =
  "psu/gesture/demo01/server/status";

WiFiClient wifiClient;
PubSubClient mqtt(wifiClient);

uint32_t lastWifiAttempt = 0;
uint32_t lastMqttAttempt = 0;

constexpr uint32_t WIFI_RETRY_MS = 10000;
constexpr uint32_t MQTT_RETRY_MS = 5000;

// ================================
// MPU6050
// ================================
constexpr int SDA_PIN = 4;
constexpr int SCL_PIN = 5;
constexpr uint8_t MPU_ADDRESS = 0x68;

constexpr uint32_t SAMPLE_PERIOD_US = 20000;
constexpr uint32_t MAX_SAMPLE_GAP_US = 30000;

// ================================
// TinyML
// ================================
constexpr uint32_t TENSOR_ARENA = 24 * 1024;

MPU6050Simple mpu;
Eloquent::TF::Sequential<2, TENSOR_ARENA> tf;

float windowData[kWindowSize][kAxes];
float featureData[kFeatureCount];

int sampleCount = 0;
uint32_t lastSampleUs = 0;
uint32_t gestureSequence = 0;

// ================================
// BLE
// ================================
#define SERVICE_UUID \
  "7c6a0001-14b7-4a1c-8ca6-a0e527117203"

#define GESTURE_UUID \
  "7c6a0002-14b7-4a1c-8ca6-a0e527117203"

BLEServer* bleServer = nullptr;
BLECharacteristic* gestureCharacteristic = nullptr;

volatile bool deviceConnected = false;
volatile bool restartAdvertising = false;

const char* gestureNames[] = {
  "TILT_LEFT",
  "TILT_RIGHT",
  "UP",
  "DOWN",
  "IDLE"
};

// ================================
// Error handling
// ================================
void fatalError(const char* message) {
  Serial.print("FATAL: ");
  Serial.println(message);

  while (true) {
    delay(1000);
  }
}

// ================================
// WiFi initialization
// ================================
void initializeWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  lastWifiAttempt = millis();

  Serial.println("WiFi connection started");
}

// ================================
// MQTT initialization
// ================================
void initializeMQTT() {
  mqtt.setServer(MQTT_BROKER, MQTT_PORT);
  mqtt.setBufferSize(256);
  mqtt.setKeepAlive(30);
  mqtt.setSocketTimeout(2);

  Serial.println("MQTT configured");
}

// ================================
// WiFi and MQTT maintenance
// ================================
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
    "gesture-server-%llX",
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
// BLE callbacks
// ================================
class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* server) override {
    deviceConnected = true;
    restartAdvertising = false;
    Serial.println("BLE Client connected");
  }

  void onDisconnect(BLEServer* server) override {
    deviceConnected = false;
    restartAdvertising = true;
    Serial.println("BLE Client disconnected");
  }
};

// ================================
// Initialize BLE
// ================================
void initializeBLE() {
  BLEDevice::init("ESP32-Gesture-Server");

  bleServer = BLEDevice::createServer();

  bleServer->setCallbacks(
    new ServerCallbacks());

  BLEService* service =
    bleServer->createService(SERVICE_UUID);

  gestureCharacteristic =
    service->createCharacteristic(
      GESTURE_UUID,
      BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);

  gestureCharacteristic->addDescriptor(
    new BLE2902());

  gestureCharacteristic->setValue("0,4");

  service->start();

  BLEAdvertising* advertising =
    BLEDevice::getAdvertising();

  advertising->addServiceUUID(SERVICE_UUID);
  advertising->setScanResponse(true);

  BLEDevice::startAdvertising();

  Serial.println("BLE Server advertising");
}

// ================================
// BLE maintenance
// ================================
void maintainBLE() {
  if (restartAdvertising && !deviceConnected) {
    restartAdvertising = false;
    bleServer->startAdvertising();
    Serial.println("BLE advertising restarted");
  }
}

// ================================
// Publish gesture over BLE + MQTT
// ================================
void sendGesture(int label) {
  gestureSequence++;

  // BLE: short packet for default ATT MTU
  char blePacket[20];

  snprintf(
    blePacket,
    sizeof(blePacket),
    "%lu,%d",
    static_cast<unsigned long>(gestureSequence),
    label);

  gestureCharacteristic->setValue(blePacket);

  if (deviceConnected) {
    gestureCharacteristic->notify();
  }

  Serial.printf(
    "BLE TX | Seq=%lu | Gesture=%s\n",
    static_cast<unsigned long>(gestureSequence),
    gestureNames[label]);

  // MQTT: JSON for monitoring
  char mqttPayload[160];

  snprintf(
    mqttPayload,
    sizeof(mqttPayload),
    "{\"sequence\":%lu,\"gesture_id\":%d,"
    "\"gesture\":\"%s\",\"uptime_ms\":%lu}",
    static_cast<unsigned long>(gestureSequence),
    label,
    gestureNames[label],
    static_cast<unsigned long>(millis()));

  if (mqtt.connected()) {
    bool sent = mqtt.publish(
      MQTT_GESTURE_TOPIC,
      mqttPayload);

    Serial.printf(
      "MQTT TX | %s | %s\n",
      sent ? "QUEUED" : "FAILED",
      mqttPayload);
  } else {
    Serial.println(
      "MQTT offline: gesture not published");
  }
}

// ================================
// Validate model
// ================================
void checkModel() {
  if (GESTURE_NUM_INPUTS != 24 || GESTURE_NUM_CLASSES != 5) {
    fatalError("Expected 24-input 5-class model");
  }

  const tflite::Model* model =
    tflite::GetModel(model_data);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    fatalError("TFLite schema mismatch");
  }

  const auto* graphs = model->subgraphs();

  if (!graphs || graphs->size() != 1) {
    fatalError("Expected one model subgraph");
  }

  const auto* graph = graphs->Get(0);

  if (!graph->inputs() || !graph->outputs() || graph->inputs()->size() != 1 || graph->outputs()->size() != 1) {
    fatalError("Invalid model inputs/outputs");
  }

  const auto* input =
    graph->tensors()->Get(graph->inputs()->Get(0));

  const auto* output =
    graph->tensors()->Get(graph->outputs()->Get(0));

  if (input->type() != tflite::TensorType_INT8 || output->type() != tflite::TensorType_INT8 || !input->shape() || input->shape()->size() != 2 || input->shape()->Get(0) != 1 || input->shape()->Get(1) != 24 || !output->shape() || output->shape()->size() != 2 || output->shape()->Get(0) != 1 || output->shape()->Get(1) != 5) {
    fatalError("Expected INT8 [1,24] and [1,5]");
  }

  const auto* quant = input->quantization();

  if (!quant || !quant->scale() || !quant->zero_point() || quant->scale()->size() != 1 || quant->zero_point()->size() != 1 || !isfinite(gesture_input_scale) || gesture_input_scale <= 0 || fabsf(quant->scale()->Get(0) - gesture_input_scale) > 1e-7f || quant->zero_point()->Get(0) != gesture_input_zero_point) {
    fatalError("Model quantization mismatch");
  }

  for (int i = 0; i < kFeatureCount; i++) {
    if (!isfinite(gesture_mean[i]) || !isfinite(gesture_std[i]) || gesture_std[i] <= 0) {
      fatalError("Invalid normalization");
    }
  }
}

// ================================
// TinyML classification
// ================================
void classifyWindow() {
  extractFeatures(windowData, featureData);

  for (int i = 0; i < kFeatureCount; i++) {
    if (!isfinite(featureData[i])) {
      fatalError("Invalid feature");
    }
  }

  int8_t input[kFeatureCount];

  gesture_prepare_input(
    featureData,
    input);

  uint32_t startUs = micros();

  if (!tf.predict(input).isOk()) {
    Serial.println(tf.exception.toString());
    fatalError("TinyML inference failed");
  }

  uint32_t inferenceUs = micros() - startUs;

  int label = static_cast<int>(tf.classification);

  if (label < 0 || label >= 5) {
    fatalError("Invalid predicted class");
  }

  Serial.printf(
    "Gesture=%s | Inference=%lu us\n",
    gestureNames[label],
    static_cast<unsigned long>(inferenceUs));

  sendGesture(label);
}

// ================================
// Setup
// ================================
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("ESP32 Gesture Server");
  Serial.println("TinyML + BLE + MQTT");

  // MPU6050
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);
  Wire.setTimeOut(50);

  if (!mpu.begin(
        Wire,
        MPU_ADDRESS,
        MPU6050Simple::AccelRange::G4,
        MPU6050Simple::GyroRange::DPS500)) {
    fatalError(mpu.errorMessage());
  }

  Serial.println("MPU6050 initialized");

  // TinyML
  checkModel();

  tf.setNumInputs(24);
  tf.setNumOutputs(5);

  if (tf.resolver.AddFullyConnected() != kTfLiteOk || tf.resolver.AddSoftmax() != kTfLiteOk) {
    fatalError("Operator registration failed");
  }

  if (!tf.begin(model_data).isOk()) {
    Serial.println(tf.exception.toString());
    fatalError("Model initialization failed");
  }

  Serial.println("TinyML initialized");

  // BLE and MQTT
  initializeBLE();
  initializeWiFi();
  initializeMQTT();

  lastSampleUs = micros();
}

// ================================
// Main loop
// ================================
void loop() {
  maintainBLE();
  maintainNetwork();

  uint32_t nowUs = micros();
  uint32_t elapsed = nowUs - lastSampleUs;

  if (elapsed < SAMPLE_PERIOD_US) {
    delay(1);
    return;
  }

  if (elapsed > MAX_SAMPLE_GAP_US && sampleCount > 0) {
    sampleCount = 0;
    Serial.println("Sampling gap: window discarded");
  }

  lastSampleUs = nowUs;

  float reading[kAxes];

  if (!mpu.readFeatures(reading)) {
    fatalError(mpu.errorMessage());
  }

  for (int axis = 0; axis < kAxes; axis++) {
    if (!isfinite(reading[axis])) {
      fatalError("Invalid sensor reading");
    }

    windowData[sampleCount][axis] = reading[axis];
  }

  sampleCount++;

  if (sampleCount == kWindowSize) {
    sampleCount = 0;

    classifyWindow();

    lastSampleUs = micros();
  }
}
