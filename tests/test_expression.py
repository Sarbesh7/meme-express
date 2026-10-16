"""Offline detector and OBS-routing checks; no webcam or OBS server required."""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import expression as e


def landmarks(points, width=640, height=480):
    return [SimpleNamespace(x=float(x / width), y=float(y / height), z=0.0)
            for x, y in points]


def face_points(left_ear=0.30, right_ear=0.30, roll=0, mirror=False, mar=0.10):
    points = np.tile([320.0, 240.0], (478, 1))
    points[10], points[152] = (320, 130), (320, 350)
    points[1], points[6] = (320, 255), (320, 245)
    points[61], points[291] = (290, 295), (350, 295)
    points[13], points[14] = (320, 295 - 30 * mar), (320, 295 + 30 * mar)
    points[234], points[454] = (200, 240), (440, 240)
    for indices, center, ear in (
        ((33, 160, 158, 133, 153, 144), 260, right_ear),
        ((362, 385, 387, 263, 373, 380), 380, left_ear),
    ):
        points[list(indices)] = [
            (center - 20, 200), (center - 7, 200 - 20 * ear),
            (center + 7, 200 - 20 * ear), (center + 20, 200),
            (center + 7, 200 + 20 * ear), (center - 7, 200 + 20 * ear),
        ]
    radians = math.radians(roll)
    rotation = np.array([[math.cos(radians), -math.sin(radians)],
                         [math.sin(radians), math.cos(radians)]])
    points = (points - [320, 240]) @ rotation.T + [320, 240]
    if mirror:
        points[:, 0] = 640 - points[:, 0]
    return points


def shush_points():
    points = np.tile([320.0, 360.0], (21, 1))
    points[0] = (320, 430)
    points[5:9] = [(320, 350), (320, 315), (320, 285), (320, 255)]
    for mcp, x in ((9, 340), (13, 360), (17, 380)):
        points[mcp:mcp + 4] = [(x, 355), (x, 320), (x + 5, 342), (x, 360)]
    return points


def geometry(**kwargs):
    return e.expression_geometry(landmarks(face_points(**kwargs)), 640, 480)


class FaceGestureTests(unittest.TestCase):
    def test_tilt_both_directions_and_mirrored(self):
        for roll in (-30, 25):
            for mirror in (False, True):
                with self.subTest(roll=roll, mirror=mirror):
                    self.assertTrue(e.detect_confused_tilt(geometry(roll=roll, mirror=mirror)))

    def test_tilt_rejects_neutral_extreme_and_open_mouth(self):
        for args in ({}, {"roll": 60}, {"roll": 25, "mar": 0.6},
                     {"roll": 25, "left_ear": 0.08}):
            with self.subTest(args=args):
                self.assertFalse(e.detect_confused_tilt(geometry(**args)))

    def test_wink_either_eye_and_mirrored(self):
        for left, right in ((0.08, 0.30), (0.30, 0.08)):
            for mirror in (False, True):
                self.assertTrue(e.detect_wink(geometry(left_ear=left, right_ear=right, mirror=mirror)))

    def test_blink_squint_and_neutral_do_not_wink(self):
        for left, right in ((0.08, 0.08), (0.19, 0.20), (0.30, 0.30)):
            self.assertFalse(e.detect_wink(geometry(left_ear=left, right_ear=right)))
        self.assertFalse(e.detect_wink(geometry(left_ear=0.08, roll=30)))

    def test_profile_and_tiny_face_are_rejected(self):
        points = face_points(left_ear=0.08, roll=0)
        points[1, 0] = 380
        g = e.expression_geometry(landmarks(points), 640, 480)
        self.assertFalse(g["frontal"])
        self.assertFalse(e.detect_wink(g))
        points = face_points(roll=25) * 0.1
        self.assertIsNone(e.expression_geometry(landmarks(points), 640, 480))

    def test_geometry_uses_pixels_for_non_square_images(self):
        points = face_points(roll=25)
        for width, height in ((640, 480), (1280, 720), (480, 640)):
            g = e.expression_geometry(landmarks(points, width, height), width, height)
            self.assertAlmostEqual(g["roll"], 25, places=4)
            self.assertAlmostEqual(g["left_ear"], 0.30, places=4)

    def test_absent_face_never_triggers(self):
        self.assertIsNone(e.expression_geometry(None, 640, 480))
        self.assertFalse(e.detect_confused_tilt(None))
        self.assertFalse(e.detect_wink(None))
        self.assertFalse(e.detect_shush([], None, 640, 480)[0])


class ShushTests(unittest.TestCase):
    def detect(self, points, count=1, g=None):
        hand = (landmarks(points), "Left", 0.99)
        return e.detect_shush([hand] * count, geometry() if g is None else g, 640, 480)[0]

    def test_index_at_lips(self):
        self.assertTrue(self.detect(shush_points()))

    def test_mirrored_hand_and_face(self):
        points = shush_points()
        points[:, 0] = 640 - points[:, 0]
        self.assertTrue(self.detect(points, g=geometry(mirror=True)))

    def test_pointing_away_from_face(self):
        points = shush_points()
        points[:, 0] += 100
        self.assertFalse(self.detect(points))

    def test_open_palm_and_peace_sign(self):
        for extended in ((9,), (9, 13, 17)):
            points = shush_points()
            for mcp in extended:
                x = points[mcp, 0]
                points[mcp:mcp + 4] = [(x, 355), (x, 320), (x, 285), (x, 255)]
            self.assertFalse(self.detect(points))

    def test_curled_index_and_two_hands_rejected(self):
        points = shush_points()
        points[8] = (320, 340)
        self.assertFalse(self.detect(points))
        self.assertFalse(self.detect(shush_points(), count=2))
        self.assertFalse(self.detect(shush_points(), count=0))


class TimingAndRoutingTests(unittest.TestCase):
    def test_short_wink_does_not_confirm(self):
        timer = e.TimedConfirmation(e.WINK_CONFIRM_SECONDS)
        self.assertFalse(timer.update(True, 0.0))
        self.assertFalse(timer.update(True, 0.10))
        self.assertFalse(timer.update(False, 0.15))
        self.assertFalse(timer.update(True, 0.20))
        self.assertTrue(timer.update(True, 0.43))

    def test_missing_face_or_frame_gap_resets_confirmation(self):
        timer = e.TimedConfirmation(0.3)
        self.assertFalse(timer.update(True, 1.0))
        self.assertFalse(timer.update(True, 2.0))
        self.assertTrue(timer.update(True, 2.31))
        self.assertFalse(timer.update(False, 2.4))
        self.assertFalse(timer.update(True, 2.5))

    def test_all_six_mappings_and_conflict_priority(self):
        for i, source in enumerate(e.ALL_SOURCES):
            # choose_meme argument order is cover, talking, mouth, tilt, wink, shush.
            signals = [False] * 6
            signals[(2, 0, 1, 3, 4, 5)[i]] = True
            self.assertEqual(e.choose_meme(*signals), source)
        self.assertEqual(e.choose_meme(True, True, True, True, True, True), e.SOURCE_FACE_COVER)
        self.assertEqual(e.choose_meme(False, True, True, True, True, True), e.SOURCE_SHUSH)
        self.assertIsNone(e.choose_meme(False, False, False, False, False, False))

    def test_manual_key_overrides_ongoing_detection_until_timeout(self):
        cap = Mock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        result = SimpleNamespace(face_landmarks=[], hand_landmarks=[], handedness=[], pose_landmarks=[])
        detector = Mock()
        detector.detect_for_video.return_value = result
        with patch.object(e.time, "monotonic", side_effect=[0, 0, 0.01, 0.2, 0.4, 1.6]), \
             patch.object(e.cv2, "imshow"), \
             patch.object(e.cv2, "waitKey", side_effect=[ord("5"), -1, -1, ord("q")]), \
             patch.object(e, "choose_meme", return_value=e.SOURCE_CONFUSED_TILT), \
             patch.object(e, "sync_obs_sources") as sync:
            e.run_detection(cap, detector, detector, detector)
        self.assertEqual([call.args[0] for call in sync.call_args_list], [
            e.SOURCE_CONFUSED_TILT, e.SOURCE_WINK, e.SOURCE_WINK,
            e.SOURCE_WINK, e.SOURCE_CONFUSED_TILT,
        ])


class OBSRoutingTests(unittest.TestCase):
    def setUp(self):
        self.ws = Mock()
        self.ws.call.return_value = SimpleNamespace(status=True)
        for name, value in (
            ("ws", self.ws), ("is_obs_connected", True),
            ("scene_item_ids", {s: [i] for i, s in enumerate(e.ALL_SOURCES)}),
            ("source_visibility", {}), ("last_obs_attempt", float("-inf")),
        ):
            patcher = patch.object(e, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_initialize_switch_and_hide_without_redundant_calls(self):
        e.sync_obs_sources(e.SOURCE_WINK)
        self.assertEqual(self.ws.call.call_count, 6)
        self.assertTrue(e.source_visibility[e.SOURCE_WINK])
        self.assertEqual(sum(e.source_visibility.values()), 1)
        e.sync_obs_sources(e.SOURCE_WINK)
        self.assertEqual(self.ws.call.call_count, 6)
        self.ws.call.reset_mock()
        e.sync_obs_sources(e.SOURCE_CONFUSED_TILT)
        calls = [call.args[0].dataout for call in self.ws.call.call_args_list]
        self.assertEqual([c["sceneItemEnabled"] for c in calls], [False, True])
        e.sync_obs_sources(None)
        self.assertFalse(any(e.source_visibility.values()))

    def test_missing_new_source_does_not_break_existing_sources(self):
        del e.scene_item_ids[e.SOURCE_SHUSH]
        e.sync_obs_sources(e.SOURCE_SHUSH)
        self.assertTrue(e.is_obs_connected)
        e.sync_obs_sources(e.SOURCE_MOUTH_OPEN)
        self.assertTrue(e.source_visibility[e.SOURCE_MOUTH_OPEN])

    def test_reconnect_reapplies_all_source_states(self):
        self.ws.call.return_value = SimpleNamespace(status=False)
        e.sync_obs_sources(e.SOURCE_WINK)
        self.assertFalse(e.is_obs_connected)

        def respond(request):
            if request.name == "GetSceneList":
                return SimpleNamespace(datain={"scenes": [{"sceneName": e.SCENE_NAME}]})
            if request.name == "GetSceneItemList":
                return SimpleNamespace(datain={"sceneItems": [
                    {"sourceName": s, "sceneItemId": i} for i, s in enumerate(e.ALL_SOURCES)
                ]})
            return SimpleNamespace(status=True)

        self.ws.call.side_effect = respond
        e.sync_obs_sources(e.SOURCE_SHUSH)
        self.assertTrue(e.is_obs_connected)
        self.assertEqual(sum(e.source_visibility.values()), 1)
        self.assertTrue(e.source_visibility[e.SOURCE_SHUSH])


if __name__ == "__main__":
    unittest.main()
