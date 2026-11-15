# WiFi Radar

Camera-free indoor presence and activity sensing using Wi-Fi Channel State Information (CSI) from an ESP32-S3.

Classes: empty room, person static, movement, fall detection.

Research prototype. Alerts are estimates and not a medical or emergency system.

## Run

    cd wifi_radar
    pip install -r requirements.txt
    python -m wifi_radar

Open http://127.0.0.1:8000, then run room calibration with the room empty.

## Structure

- main, components: ESP32-S3 firmware
- wifi_radar: processing application and trained model
- android-notifier: Android notification app
- docs: firmware web installer

## License

Apache-2.0

## On-board AI (no computer)

Set the device role to "On-board AI (no computer)" in the dashboard at http://<device-name>.local.
The trained model is stored in the `model` flash partition and runs on the ESP32-S3; the dashboard shows the result directly.
