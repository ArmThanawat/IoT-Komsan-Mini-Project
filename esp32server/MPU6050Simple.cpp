#include "MPU6050Simple.h"

bool MPU6050Simple::writeRegister(uint8_t reg, uint8_t value) {
  wire_->beginTransmission(address_);
  wire_->write(reg);
  wire_->write(value);
  if (wire_->endTransmission(true) != 0) {
    error_ = Error::I2C;
    return false;
  }
  error_ = Error::None;
  return true;
}

bool MPU6050Simple::readRegisters(uint8_t reg, uint8_t* bytes, uint8_t count) {
  wire_->beginTransmission(address_);
  wire_->write(reg);
  if (wire_->endTransmission(false) != 0) {
    error_ = Error::I2C;
    return false;
  }
  const size_t received = wire_->requestFrom(address_, count, true);
  if (received != count || wire_->available() < count) {
    while (wire_->available()) wire_->read();
    error_ = Error::ShortRead;
    return false;
  }
  for (uint8_t i = 0; i < count; ++i) {
    const int value = wire_->read();
    if (value < 0) {
      error_ = Error::ShortRead;
      return false;
    }
    bytes[i] = static_cast<uint8_t>(value);
  }
  error_ = Error::None;
  return true;
}

int16_t MPU6050Simple::decode(const uint8_t* bytes) {
  const uint16_t value = (static_cast<uint16_t>(bytes[0]) << 8) | bytes[1];
  const int32_t signedValue = (value & 0x8000u) ? static_cast<int32_t>(value) - 65536 : value;
  return static_cast<int16_t>(signedValue);
}

bool MPU6050Simple::begin(TwoWire& wire, uint8_t address,
                          AccelRange accel, GyroRange gyro) {
  ready_ = false;
  if ((address != 0x68 && address != 0x69) || static_cast<uint8_t>(accel) > 3 || static_cast<uint8_t>(gyro) > 3) {
    error_ = Error::InvalidArgument;
    return false;
  }
  wire_ = &wire;
  address_ = address;
  uint8_t identity;
  if (!readRegisters(0x75, &identity, 1)) return false;
  // WHO_AM_I remains 0x68 even when AD0 sets bus address to 0x69.
  if (identity != 0x68) {
    error_ = Error::WrongDevice;
    return false;
  }
  if (!writeRegister(0x6B, 0x01)) return false;  // wake, X gyro PLL
  delay(100);
  if (!writeRegister(0x6C, 0x00) ||  // enable all axes
      !writeRegister(0x6A, 0x00) ||  // disable FIFO/master mode
      !writeRegister(0x23, 0x00) ||  // no FIFO sources
      !writeRegister(0x38, 0x00) ||  // polling, no interrupts
      !writeRegister(0x1A, 0x03) ||  // DLPF: accel44Hz/gyro42Hz
      !writeRegister(0x19, 19) ||    // 1000/(19+1)=50Hz sensor output
      !writeRegister(0x1C, static_cast<uint8_t>(accel) << 3) || !writeRegister(0x1B, static_cast<uint8_t>(gyro) << 3)) return false;
  const float accelScales[] = { 16384.0f, 8192.0f, 4096.0f, 2048.0f };
  const float gyroScales[] = { 131.0f, 65.5f, 32.8f, 16.4f };
  accelScale_ = accelScales[static_cast<uint8_t>(accel)];
  gyroScale_ = gyroScales[static_cast<uint8_t>(gyro)];
  delay(100);  // allow filters to settle
  ready_ = true;
  error_ = Error::None;
  return true;
}

bool MPU6050Simple::readRaw(RawData& output) {
  if (!ready_) {
    error_ = Error::NotInitialized;
    return false;
  }
  uint8_t bytes[14];
  if (!readRegisters(0x3B, bytes, sizeof(bytes))) return false;
  RawData next;
  next.ax = decode(bytes);
  next.ay = decode(bytes + 2);
  next.az = decode(bytes + 4);
  next.temperature = decode(bytes + 6);
  next.gx = decode(bytes + 8);
  next.gy = decode(bytes + 10);
  next.gz = decode(bytes + 12);
  output = next;
  return true;
}

bool MPU6050Simple::read(Data& output) {
  RawData raw;
  if (!readRaw(raw)) return false;
  Data next;
  next.ax_g = raw.ax / accelScale_;
  next.ay_g = raw.ay / accelScale_;
  next.az_g = raw.az / accelScale_;
  next.gx_dps = raw.gx / gyroScale_;
  next.gy_dps = raw.gy / gyroScale_;
  next.gz_dps = raw.gz / gyroScale_;
  next.temperature_c = raw.temperature / 340.0f + 36.53f;
  output = next;
  return true;
}

bool MPU6050Simple::readFeatures(float output[6]) {
  if (!output) {
    error_ = Error::InvalidArgument;
    return false;
  }
  Data value;
  if (!read(value)) return false;
  output[0] = value.ax_g;
  output[1] = value.ay_g;
  output[2] = value.az_g;
  output[3] = value.gx_dps;
  output[4] = value.gy_dps;
  output[5] = value.gz_dps;
  return true;
}

const char* MPU6050Simple::errorMessage() const {
  switch (error_) {
    case Error::None: return "OK";
    case Error::NotInitialized: return "Sensor not initialized";
    case Error::InvalidArgument: return "Invalid argument";
    case Error::I2C: return "I2C transaction failed";
    case Error::ShortRead: return "Incomplete I2C response";
    case Error::WrongDevice: return "WHO_AM_I is not MPU6050 (0x68)";
  }
  return "Unknown error";
}