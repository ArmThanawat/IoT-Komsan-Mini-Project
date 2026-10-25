#pragma once

class LightControl {
public:
  int brightness = 0;
  int savedBrightness = 128;
  bool armed = false;

  void disarm() {
    armed = false;
  }

  bool accept(int label) {
    if (label == 4) {
      armed = true;
      return false;
    }

    if (label < 0 || label > 3)
      return false;

    // ON/OFF commands require an idle window.
    // UP/DOWN commands are allowed to repeat.
    if (!armed && label != 2 && label != 3)
      return false;

    armed = false;

    switch (label) {
      case 0:  // Tilt left: OFF
        if (brightness > 0)
          savedBrightness = brightness;
        brightness = 0;
        break;

      case 1:  // Tilt right: ON
        brightness = savedBrightness;
        break;

      case 2:  // Move up: brighter
        brightness += 32;
        if (brightness > 255)
          brightness = 255;
        if (brightness > 0)
          savedBrightness = brightness;
        break;

      case 3:  // Move down: dimmer
        brightness -= 32;
        if (brightness < 0)
          brightness = 0;
        if (brightness > 0)
          savedBrightness = brightness;
        break;
    }

    return true;
  }
};