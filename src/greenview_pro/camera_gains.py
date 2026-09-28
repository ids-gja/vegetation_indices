"""Per-channel IDS camera gains for the live preview."""

import numpy as np


CHANNELS = {"R": "red", "G": "green", "NIR": "blue"}


def channel_gain_entries(selector):
    entries = [entry.StringValue() for entry in selector.AvailableEntries()]
    channels = {}
    for channel, keyword in CHANNELS.items():
        matches = [name for name in entries if keyword in name.casefold()]
        if not matches:
            raise ValueError(f"missing {keyword} GainSelector entry: {entries}")
        if len(matches) != 1:
            raise ValueError(f"multiple {keyword} GainSelector entries: {matches}")
        channels[channel] = matches[0]
    return channels


def reset_color_gains(nodemap):
    selector = nodemap.FindNode("GainSelector")
    gain = nodemap.FindNode("Gain")
    for name in channel_gain_entries(selector).values():
        selector.SetCurrentEntry(name)
        gain.SetValue(1.0)
        actual = float(gain.Value())
        if not np.isclose(actual, 1.0, rtol=0, atol=1e-6):
            raise ValueError(f"{name} gain must be 1.0, camera reports {actual}")
