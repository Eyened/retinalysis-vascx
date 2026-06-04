from unittest.mock import patch

import torch

from vascx.inference.device import auto_device, resolve_device


def test_resolve_device_uses_explicit_string() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_uses_explicit_torch_device() -> None:
    assert resolve_device(torch.device("cpu")) == torch.device("cpu")


@patch("vascx.inference.device.auto_device", return_value=torch.device("cpu"))
def test_resolve_device_falls_back_to_auto(mock_auto_device) -> None:
    assert resolve_device(None) == torch.device("cpu")
    mock_auto_device.assert_called_once()


@patch("vascx.inference.device.torch.cuda.is_available", return_value=True)
def test_auto_device_prefers_cuda(_mock_cuda) -> None:
    assert auto_device() == torch.device("cuda:0")


@patch("vascx.inference.device.torch.backends.mps.is_available", return_value=True)
@patch("vascx.inference.device.torch.cuda.is_available", return_value=False)
def test_auto_device_uses_mps_when_cuda_unavailable(
    _mock_cuda, _mock_mps
) -> None:
    assert auto_device() == torch.device("mps")


@patch("vascx.inference.device.torch.backends.mps.is_available", return_value=False)
@patch("vascx.inference.device.torch.cuda.is_available", return_value=False)
def test_auto_device_falls_back_to_cpu(_mock_cuda, _mock_mps) -> None:
    assert auto_device() == torch.device("cpu")
