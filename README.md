# Acoustic Hardness Classification With TinyML

![C++](https://img.shields.io/badge/C++-17-00599C.svg)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![TensorFlow 2.17](https://img.shields.io/badge/tensorflow-2.17-orange.svg)
![Arduino](https://img.shields.io/badge/arduino-nano%2033%20ble-00979D.svg)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)


Real-time audio-based surface hardness classification running on the Arduino Nano 33 BLE Sense Rev 2.

By analyzing the acoustic response of impacts, this project classifies material surfaces into three hardness categories: **soft**, **medium**, and **hard**.

---

## Project Overview

This project implements an end-to-end machine learning pipeline on a microcontroller:

1. **Phase 1: Feasibility Validation** ✅
   - Collect pilot audio samples across 3 hardness classes
   - Validate acoustic separation via signal analysis
   - Confirm classification viability

2. **Phase 2: Data Collection**
    (TBD)

3. **Phase 3: ML Pipeline**
    (TBD)
    
4. **Phase 4: Embedded Deployment**
    (TBD)

---

## Key Results & Findings

(TBD)

---

## Hardware


- **Board:** Arduino Nano 33 BLE Sense Rev 2
- **Microcontroller:** Nordic nRF52840 64MHz
- **Sensors:** 
  - MP34DT06JTR PDM Microphone (audio input)
  - BMI270 + BMM150 IMU (auxiliary)
- **Drop System:** Custom payload dropper (7cm fixed height, 1cm iron marble)

---

## Methodology

### Acoustic Setup
- **Sampling Rate:** 16 kHz
- **Recording Duration:** 2.5 seconds per impact
- **Trigger:** Sound amplitude threshold-based capture
- **Pre-trigger Buffer:** 50ms (ambient baseline)
- **Post-trigger Window:** 2.45s

### Classification Task
- **Soft:** Low-energy, quick decay
- **Medium:** Moderate energy, partial damping
- **Hard:** High-energy, sustained ringing

---

## Tech Stack

**Core Framework & Training:**
- TensorFlow — Deep learning framework

**Utilities:**
- NumPy, Pandas — Data manipulation and analysis
- Matplotlib — Visualization

**Development:**
- Black — Code formatting
- Flake8 — Linting
- Pytest — Unit testing

---

## Folder Structure

(TBD)

---

## How to Run

(TBD)

---

## Citations & References

- [Arduino Official Page](https://www.arduino.cc/)
- [Arduino Docs](https://docs.arduino.cc/)
- [Nano 33 BLE Sense Rev2 Docs Page](https://docs.arduino.cc/hardware/nano-33-ble-sense-rev2/)

---

## Contact

For questions reach out via GitHub (Kev-HL).
