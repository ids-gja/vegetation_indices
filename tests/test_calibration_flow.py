import numpy as np
import pytest
from ids_peak import ids_peak

from calibration.flow import (
    GainControls,
    FrameAverage,
    finish_calibration,
    recommend_gains,
)


def test_frame_average_uses_all_raw_frames():
    average = FrameAverage(2)
    assert average.add(np.array([[10, 20], [30, 40]], dtype=np.uint16)) is None
    np.testing.assert_array_equal(
        average.add(np.array([[14, 24], [34, 44]], dtype=np.uint16)),
        [[12, 22], [32, 42]],
    )


def test_frame_average_rejects_changed_shape():
    average = FrameAverage(2)
    average.add(np.ones((2, 2)))
    with pytest.raises(ValueError, match="shape"):
        average.add(np.ones((4, 4)))


class FakeNode:
    def __init__(self, value=1):
        self.value = value
        self.values = {"Red": 1.0, "Green": 1.0, "Blue": 1.0, "AnalogAll": 1.0}
        self.selector = None
        self.increment_type = ids_peak.NodeIncrementType_FixedIncrement

    def AvailableEntries(self):
        return [FakeEntry(name) for name in self.gain.values]

    def SetCurrentEntry(self, name):
        self.selector = name
        if hasattr(self, "gain"):
            self.gain.selector = name

    def Value(self):
        return self.values[self.selector] if self.selector else self.value

    def SetValue(self, value):
        if self.selector:
            self.values[self.selector] = value
        else:
            self.value = value

    def Minimum(self):
        return 1.0

    def Maximum(self):
        return 8.0

    def Increment(self):
        if self.increment_type != ids_peak.NodeIncrementType_FixedIncrement:
            raise RuntimeError("node does not have an increment")
        return 0.25

    def IncrementType(self):
        return self.increment_type


class FakeMap:
    def __init__(self, entries=None):
        self.selector = FakeNode()
        self.gain = FakeNode()
        if entries is not None:
            self.gain.values = {name: 1.0 for name in entries}
        self.selector.gain = self.gain
        self.black = FakeNode(2.0)
        self.exposure = FakeNode(100.0)

    def FindNode(self, name):
        if name == "GainSelector":
            return self.selector
        if name == "Gain":
            self.gain.selector = self.selector.selector
            return self.gain
        if name == "BlackLevel":
            return self.black
        if name == "ExposureTime":
            return self.exposure
        raise KeyError(name)


class FakeEntry:
    def __init__(self, name):
        self.name = name

    def StringValue(self):
        return self.name


def test_two_stage_calibration_uses_readbacks_and_black_level():
    nodes = FakeMap()
    controls = GainControls(nodes)
    raw = np.array([[12, 22], [42, 12]], dtype=np.float32)
    requested = recommend_gains(raw, controls, black_level=2.0)
    assert requested == {"R": 2.0, "G": 4.0, "NIR": 1.0}
    controls.apply(requested)
    assert controls.read() == requested
    calibrated = finish_calibration(
        np.array([[42, 42], [42, 42]], dtype=np.float32),
        controls,
        black_level=2.0,
    )
    assert calibrated.hardware_gains == requested
    assert calibrated.offsets == {"R": 2.0, "G": 2.0, "NIR": 2.0}
    assert calibrated.gains == {"R": 0.025, "G": 0.025, "NIR": 0.025}


def test_gain_entries_are_discovered_from_the_camera():
    nodes = FakeMap(("DigitalRed", "analogGreen", "NirBlue", "Global"))
    controls = GainControls(nodes)
    assert controls.channel_entries == {
        "R": "DigitalRed",
        "G": "analogGreen",
        "NIR": "NirBlue",
    }
    assert controls.auxiliary_options() == [("Global", 1.0, 8.0, 0.25, 1.0)]
    assert controls.apply_auxiliary("Global", 2.5) == 2.5
    assert controls.read_auxiliary("Global") == 2.5
    controls.apply({"R": 2.0, "G": 3.0, "NIR": 4.0})
    assert controls.read() == {"R": 2.0, "G": 3.0, "NIR": 4.0}


def test_continuous_camera_gain_does_not_query_increment():
    nodes = FakeMap()
    nodes.gain.increment_type = ids_peak.NodeIncrementType_NoIncrement
    controls = GainControls(nodes)
    assert controls.auxiliary_options() == [("AnalogAll", 1.0, 8.0, None, 1.0)]
    assert all(limits.increment is None for limits in controls.limits().values())


def test_unsupported_list_gain_increment_reports_clear_error():
    nodes = FakeMap()
    nodes.gain.increment_type = ids_peak.NodeIncrementType_ListIncrement
    controls = GainControls(nodes)
    with pytest.raises(ValueError, match="ListIncrement"):
        controls.limits()



@pytest.mark.parametrize(
    "entries, error",
    [
        (("Red", "DigitalRed", "Green", "Blue"), "multiple.*red"),
        (("Red", "Green", "AnalogAll"), "missing.*blue"),
    ],
)
def test_ambiguous_or_missing_channels_fail_explicitly(entries, error):
    with pytest.raises(ValueError, match=error):
        GainControls(FakeMap(entries))
