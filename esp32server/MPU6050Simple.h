#pragma once
#include <Arduino.h>
#include <Wire.h>

// Call Wire.begin(...) yourself before begin(). No dynamic allocation.
class MPU6050Simple {
public:
  enum class AccelRange : uint8_t { G2 = 0,
                                    G4 = 1,
                                    G8 = 2,
                                    G16 = 3 };
  enum class GyroRange : uint8_t { DPS250 = 0,
                                   DPS500 = 1,
                                   DPS1000 = 2,
                                   DPS2000 = 3 };
  enum class Error : uint8_t {
    None,
    NotInitialized,
    InvalidArgument,
    I2C,
    ShortRead,
    WrongDevice
  };
  struct RawData {
    int16_t ax, ay, az, temperature, gx, gy, gz;
  };
  struct Data {
    float ax_g, ay_g, az_g, temperature_c, gx_dps, gy_dps, gz_dps;
  };

  bool begin(TwoWire& wire = Wire, uint8_t address = 0x68,
             AccelRange accel = AccelRange::G2,
             GyroRange gyro = GyroRange::DPS250);
  // On failure, the caller's output is left unchanged.
  bool readRaw(RawData& output);
  bool read(Data& output);
  bool readFeatures(float output[6]);  // ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps
  Error lastError() const {
    return error_;
  }
  const char* errorMessage() const;

private:
  TwoWire* wire_ = nullptr;
  uint8_t address_ = 0x68;
  bool ready_ = false;
  float accelScale_ = 16384.0f;
  float gyroScale_ = 131.0f;
  Error error_ = Error::NotInitialized;
  bool writeRegister(uint8_t reg, uint8_t value);
  bool readRegisters(uint8_t reg, uint8_t* bytes, uint8_t count);
  static int16_t decode(const uint8_t* bytes);
};