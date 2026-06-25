# Data Schema

Each raw sample is stored as a JSON file with the following structure:

## Metadata Field

| Field | Type | Description |
|-------|------|-------------|
| `sample_id` | string (UUID) | Unique identifier, generated at capture time |
| `timestamp` | string (ISO 8601) | Capture timestamp in UTC timezone |
| `material` | string | Name of the test surface (e.g., "kitchen_tile", "yoga_mat") |
| `class` | string | One of: "hard", "medium", "soft" |
| `comment` | string \| null | Optional user comment about the sample |

## Hardware Field

| Field | Type | Description |
|-------|------|-------------|
| `board` | string | Arduino board model |
| `microcontroller` | string | Microcontroller chip |
| `microphone` | string | Microphone model |
| `arduino_ide_version` | string | Arduino IDE version used |
| `mic_pdm_gain` | integer | PDM microphone gain setting |

## Audio Field

| Field | Type | Description |
|-------|------|-------------|
| `sample_rate` | integer | Sample rate in Hz |
| `num_samples` | integer | Total number of audio samples |
| `duration_seconds` | float | Duration of recording in seconds |
| `pre_trigger_samples` | integer | Number of samples before trigger (pre-buffer) |
| `post_trigger_samples` | integer | Number of samples after trigger |
| `trigger_threshold` | integer | Amplitude threshold that triggered capture |
| `overflow` | boolean | Whether buffer overflow occurred |
| `values` | array<int16> | Audio samples as signed 16-bit integers |

## Example

```json
{
  "metadata": {
    "sample_id": "a1b2c3d4-e5f6-47ab-8cd9-ef1234567890",
    "timestamp": "2026-06-24T16:47:00.139464+00:00",
    "material": "floor_tile",
    "class": "hard",
    "comment": null
  },
  "hardware": {
    "board": "Arduino Nano 33 BLE Sense Rev2",
    "microcontroller": "Nordic nRF52840",
    "microphone": "ST MP34DT06JTR",
    "arduino_ide_version": "2.3.10",
    "mic_pdm_gain": 20
  },
  "audio": {
    "sample_rate": 16000,
    "num_samples": 40000,
    "duration_seconds": 2.5,
    "pre_trigger_samples": 800,
    "post_trigger_samples": 39200,
    "trigger_threshold": 1000,
    "overflow": false,
    "values": [-1, 7, -57, 260, 6, 42, -15, 8, ...]
  }
}