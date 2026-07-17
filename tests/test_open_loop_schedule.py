"""Open-loop waypoint schedule: honor times, no guidance."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from mach_cutoff.config import AircraftConfig
from mach_cutoff.flight.aircraft import PointMassAircraft
from mach_cutoff.flight.waypoints import FlightPath, Waypoint


class OpenLoopScheduleTests(unittest.TestCase):
    def test_segment_speeds_follow_waypoint_times(self):
        start = datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc)
        # Short segment then long segment: different schedule speeds.
        mid = start + timedelta(seconds=600.0)  # 10 minutes
        end = mid + timedelta(seconds=3600.0)  # 60 minutes

        # Roughly eastbound constant latitude strip for simple distance scaling.
        path = FlightPath(
            [
                Waypoint(lat_deg=40.0, lon_deg=-100.0, alt_m=12_000.0, time_utc=start),
                Waypoint(lat_deg=40.0, lon_deg=-99.0, alt_m=12_000.0, time_utc=mid),
                Waypoint(lat_deg=40.0, lon_deg=-95.0, alt_m=12_000.0, time_utc=end),
            ]
        )
        cfg = AircraftConfig(mach=1.2, reference_sound_speed_mps=340.0)
        aircraft = PointMassAircraft(path, cfg)

        # Midpoint of first segment.
        t_seg0 = start + timedelta(seconds=300.0)
        state0 = aircraft.state_at(t_seg0)
        # Midpoint of second segment.
        t_seg1 = mid + timedelta(seconds=1800.0)
        state1 = aircraft.state_at(t_seg1)

        # Second segment is ~4x longer and 6x longer in time => slower than first.
        self.assertGreater(state0.speed_mps, state1.speed_mps)
        self.assertAlmostEqual(
            state0.mach,
            state0.speed_mps / cfg.reference_sound_speed_mps,
            places=6,
        )
        self.assertAlmostEqual(
            state1.mach,
            state1.speed_mps / cfg.reference_sound_speed_mps,
            places=6,
        )

        # Config mach is NOT used for kinematics (schedule owns speed).
        self.assertNotAlmostEqual(state0.mach, cfg.mach, places=2)

        # Position is on the first segment (lon between -100 and -99).
        self.assertGreater(state0.lon_deg, -100.0)
        self.assertLess(state0.lon_deg, -99.0)
        # Second segment lon between -99 and -95.
        self.assertGreater(state1.lon_deg, -99.0)
        self.assertLess(state1.lon_deg, -95.0)

    def test_duration_matches_waypoint_schedule(self):
        start = datetime(2025, 6, 1, 0, 0, tzinfo=timezone.utc)
        end = start + timedelta(hours=2)
        path = FlightPath(
            [
                Waypoint(lat_deg=30.0, lon_deg=-90.0, alt_m=10_000.0, time_utc=start),
                Waypoint(lat_deg=31.0, lon_deg=-88.0, alt_m=10_000.0, time_utc=end),
            ]
        )
        aircraft = PointMassAircraft(path, AircraftConfig())
        self.assertEqual(aircraft.start_time, start)
        self.assertEqual(aircraft.end_time, end)
        self.assertAlmostEqual(aircraft.duration_s, 7200.0, places=6)

        # Hold at endpoints outside the schedule window.
        before = aircraft.state_at(start - timedelta(minutes=5))
        after = aircraft.state_at(end + timedelta(minutes=5))
        self.assertAlmostEqual(before.lat_deg, 30.0, places=4)
        self.assertAlmostEqual(after.lat_deg, 31.0, places=4)

    def test_constant_altitude_on_long_segment(self):
        """Equal endpoint altitudes stay constant (no ECEF-chord dip)."""
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = start + timedelta(hours=1)
        alt = 12_000.0
        path = FlightPath(
            [
                Waypoint(lat_deg=40.0, lon_deg=-100.0, alt_m=alt, time_utc=start),
                Waypoint(lat_deg=40.0, lon_deg=-90.0, alt_m=alt, time_utc=end),
            ]
        )
        mid = path.state_at(start + timedelta(minutes=30))
        self.assertAlmostEqual(mid["alt_m"], alt, places=1)
        # Still progresses in longitude.
        self.assertGreater(mid["lon_deg"], -100.0)
        self.assertLess(mid["lon_deg"], -90.0)

    def test_altitude_linear_when_endpoints_differ(self):
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = start + timedelta(hours=1)
        path = FlightPath(
            [
                Waypoint(lat_deg=40.0, lon_deg=-100.0, alt_m=10_000.0, time_utc=start),
                Waypoint(lat_deg=40.0, lon_deg=-99.0, alt_m=14_000.0, time_utc=end),
            ]
        )
        mid = path.state_at(start + timedelta(minutes=30))
        self.assertAlmostEqual(mid["alt_m"], 12_000.0, places=1)


if __name__ == "__main__":
    unittest.main()

