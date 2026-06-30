// Copyright 2026 Kev-HL

// Define some metadata to send with the samples
#define ARDUINO_IDE_VERSION "2.3.10"
#define BOARD_MODEL "Arduino Nano 33 BLE Sense Rev2"
#define MICROCONTROLLER "Nordic nRF52840"
#define MICROPHONE_MODEL "ST MP34DT06JTR"

// Include microphone library
#include <PDM.h>

// Audio configuration
static const char CHANNELS = 1;        // Mono, one MEMS mic on the board
static const int SAMPLE_RATE = 16000;  // Valid values are 16000 or 41667 Hz
static const int GAIN = 20;  // Default for Arduino Nano 33 BLE Sense Rev2 is 20
static const int BAUD_RATE =
    1000000;  // 1Mbps max rate for Nano 33 BLE Sense Rev2

// Buffer to read samples into
// Note: default doublebuffer size (nRF52840) is 512 bytes == 256 samples
// For safety he intermediate sampleBuffer is sized at twice that (512 samples)
const int SAMPLE_BUFFER_SIZE = 512;  // 512 samples, 512x16 bits = 1024 bytes
int16_t sampleBuffer[SAMPLE_BUFFER_SIZE];
volatile int samplesRead;

// Timing variables for buffer overflow check
// Max should be less than:
// sampleBuffer size / (sampling rate x 2 bytes/sample)
// e.g. 512 / (16000 x 2) = 0.016 s
uint32_t loopStartTime = 0;
uint32_t maxLoopTime = 0;
bool overflowFlag = 0;

// Circular buffer to store audio history
const int CIRCULAR_BUFFER_SIZE = 40000;  // 40000 == 2.5 seconds at 16kHz
int16_t circularBuffer[CIRCULAR_BUFFER_SIZE];
int bufferWriteIndex = 0;
uint32_t waitStart;

// Trigger state
volatile bool recordingTriggered = false;
int triggerIndex = 0;
int samplesRecorded = 0;
const int SAMPLES_TO_RECORD_AFTER = 39200;  // 39200  == 2.45 seconds at 16kHz
const int PRE_TRIGGER_SAMPLES = 800;  // 800 samples == 0.05 seconds at 16kHz
const int TRIGGER_THRESHOLD = 750;

// Initial setup
void setup() {
  Serial.begin(BAUD_RATE);
  while (!Serial) {
  }  // Wait til serial ready

  // Safety check for circular buffer size
  if (CIRCULAR_BUFFER_SIZE < (PRE_TRIGGER_SAMPLES + SAMPLES_TO_RECORD_AFTER)) {
    Serial.println("ERROR: Circular buffer size too small!");
    while (1) {
    }  // Halt system if there is something wrong
  }

  // Configure PDM, object is instantiated (singleton) in PDM library
  PDM.onReceive(onPDMdata);
  PDM.setGain(GAIN);  // Default gain on Nano 33 BLE Sense Rev2 is 20

  if (!PDM.begin(CHANNELS, SAMPLE_RATE)) {  // PDM.begin returns 1 on success
    Serial.println("Failed to start PDM!");
    while (1) {
    }  // Halt system if there is something wrong
  }

  // Initialize digital pin LED_BUILTIN as an output.
  pinMode(LED_BUILTIN, OUTPUT);

  waitStart = millis();  // Record start time for initial wait on new recording
  while ((millis() - waitStart) < 3000) {
  }  // Wait on initial board connection
  Serial.println("Recording continuously. Drop object to trigger capture...");
  digitalWrite(LED_BUILTIN, HIGH);  // LED ON: ready for trigger
}

// Continuous execution
void loop() {
  // Process samples and check for trigger
  if (samplesRead) {
    loopStartTime = micros();  // Record start time
    for (int i = 0; i < samplesRead; i++) {
      // Add sample to circular buffer and update write index
      circularBuffer[bufferWriteIndex] = sampleBuffer[i];
      bufferWriteIndex = (bufferWriteIndex + 1) % CIRCULAR_BUFFER_SIZE;

      // Safety: Wait time to fill circular buffer (2x preTrigger duration)
      if ((millis() - waitStart) <
          (2 * 1000 * PRE_TRIGGER_SAMPLES / SAMPLE_RATE)) {
        continue;
      }

      // Check for trigger (large amplitude = impact)
      if (!recordingTriggered && abs(sampleBuffer[i]) > TRIGGER_THRESHOLD) {
        // Set event flag to true
        recordingTriggered = true;
        digitalWrite(LED_BUILTIN, LOW);  // LED OFF: busy recording/transmitting

        // Record index where trigger happened
        triggerIndex = ((bufferWriteIndex - 1 + CIRCULAR_BUFFER_SIZE) %
                        CIRCULAR_BUFFER_SIZE);
        // Reset samplesRecorded to start a new capture
        samplesRecorded = 0;
        Serial.println("--- TRIGGER DETECTED ---");
      }

      // If triggered, count samples until we have enough post-trigger data
      if (recordingTriggered) {
        samplesRecorded++;
        if (samplesRecorded >= SAMPLES_TO_RECORD_AFTER) {
          // Toggle flag if overflow has happened (note: 2 bytes per sample)
          if (maxLoopTime >=
              (1000000 * SAMPLE_BUFFER_SIZE / (2 * SAMPLE_RATE))) {
            overflowFlag = 1;
          }
          // Transmit data
          printRecording();
          // Set event flag to false
          recordingTriggered = false;
          // Print max loop time during capture and warn if overflow detected
          Serial.print("Max loop iteration time during data capture: ");
          Serial.print(maxLoopTime);
          Serial.println(" micros.");
          if (overflowFlag) {
            Serial.println("WARNING: Buffer overflow detected!");
          }
          // Reset variables before breaking loop
          maxLoopTime = 0;
          samplesRead = 0;
          overflowFlag = 0;
          waitStart = millis();
          Serial.println("--- READY FOR NEXT DROP ---");
          digitalWrite(LED_BUILTIN, HIGH);  // LED ON: ready for trigger
          return;
        }
      }
    }
    // Reset samplesRead for next onPDMdata() call
    samplesRead = 0;
    uint32_t loopTime = micros() - loopStartTime;
    if (loopTime > maxLoopTime) {
      maxLoopTime = loopTime;
    }
  }
}

// Print the captured recording: pre + post trigger samples
void printRecording() {
  Serial.println("--- RECORDING DATA ---");

  // Calculate start index (before trigger)
  int startIndex =
      ((triggerIndex - PRE_TRIGGER_SAMPLES + CIRCULAR_BUFFER_SIZE) %
       CIRCULAR_BUFFER_SIZE);
  int totalSamples = PRE_TRIGGER_SAMPLES + SAMPLES_TO_RECORD_AFTER;

  // Print metadata + values in JSON format for easier parsing
  Serial.print("{");
  Serial.print("\"arduino_ide_version\":\"" ARDUINO_IDE_VERSION "\",");
  Serial.print("\"board\":\"" BOARD_MODEL "\",");
  Serial.print("\"microcontroller\":\"" MICROCONTROLLER "\",");
  Serial.print("\"microphone\":\"" MICROPHONE_MODEL "\",");
  Serial.print("\"sample_rate\":");
  Serial.print(SAMPLE_RATE);
  Serial.print(",");
  Serial.print("\"duration_seconds\":");
  Serial.print(static_cast<float>(totalSamples) / SAMPLE_RATE, 2);
  Serial.print(",");
  Serial.print("\"num_samples\":");
  Serial.print(totalSamples);
  Serial.print(",");
  Serial.print("\"pre_trigger_samples\":");
  Serial.print(PRE_TRIGGER_SAMPLES);
  Serial.print(",");
  Serial.print("\"post_trigger_samples\":");
  Serial.print(SAMPLES_TO_RECORD_AFTER);
  Serial.print(",");
  Serial.print("\"trigger_threshold\":");
  Serial.print(TRIGGER_THRESHOLD);
  Serial.print(",");
  Serial.print("\"mic_pdm_gain\":");
  Serial.print(GAIN);
  Serial.print(",");
  Serial.print("\"overflow\":");
  Serial.print(overflowFlag);
  Serial.print(",");
  Serial.print("\"values\":[");

  // Print samples
  for (int i = 0; i < totalSamples; i++) {
    int index = (startIndex + i) % CIRCULAR_BUFFER_SIZE;
    Serial.print(circularBuffer[index]);
    if (i < totalSamples - 1) Serial.print(",");
  }

  // Close JSON
  Serial.print("]");
  Serial.println("}");

  Serial.println("--- END RECORDING ---");
}

// PDM callback
void onPDMdata() {
  int bytesAvailable = PDM.available();  // bytes available to read
  PDM.read(sampleBuffer, bytesAvailable);
  samplesRead = bytesAvailable / 2;  // 2 bytes per sample
}
