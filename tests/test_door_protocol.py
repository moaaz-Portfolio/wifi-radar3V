import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wifi_radar"))

from wifi_radar.door import PACKET_V1, PACKET_V2, parse_packet


class DoorProtocolTests(unittest.TestCase):
    def test_legacy_dor1_packet_is_still_supported(self):
        packet = PACKET_V1.pack(b"DOR1", 7, 140, 1, 1, 0)
        parsed = parse_packet(packet)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["sequence"], 7)
        self.assertEqual(parsed["distance_cm"], 140)
        self.assertTrue(parsed["present"])
        self.assertFalse(parsed["crossing"])
        self.assertFalse(parsed["sensor_fault"])

    def test_dor2_reports_reference_and_crossing(self):
        packet = PACKET_V2.pack(b"DOR2", 9, 55, 1, 1, 0, 1, 1, 0, 180)
        parsed = parse_packet(packet)
        self.assertIsNotNone(parsed)
        self.assertTrue(parsed["crossing"])
        self.assertTrue(parsed["baseline_valid"])
        self.assertEqual(parsed["baseline_cm"], 180)

    def test_dor2_reports_sensor_fault(self):
        packet = PACKET_V2.pack(b"DOR2", 10, 0, 0, 0, 0, 0, 1, 1, 180)
        parsed = parse_packet(packet)
        self.assertIsNotNone(parsed)
        self.assertTrue(parsed["sensor_fault"])
        self.assertIsNone(parsed["distance_cm"])

    def test_invalid_packet_is_rejected(self):
        self.assertIsNone(parse_packet(b"bad"))
        packet = PACKET_V2.pack(b"NOPE", 1, 20, 0, 1, 0, 0, 0, 0, 0)
        self.assertIsNone(parse_packet(packet))


if __name__ == "__main__":
    unittest.main()
