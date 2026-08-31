"""Tests for device management."""

from tokenizer_bn.device import device_info, resolve_device, torch_available


def test_resolve_device_cpu():
    assert resolve_device("cpu") == "cpu"


def test_device_info_structure():
    info = device_info("cpu")
    assert "resolved_device" in info
    assert "torch_available" in info
    assert info["resolved_device"] == "cpu"


def test_resolve_device_auto_without_torch():
    # auto should return cpu when torch missing or no GPU
    device = resolve_device("auto")
    assert device in ("cpu", "cuda", "mps")
    if not torch_available():
        assert device == "cpu"
