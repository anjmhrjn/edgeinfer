"""INT8 post-training quantization: calibration, layer groups, QDQ export."""


class QuantError(ValueError):
    """A quantization input is wrong. `code` is the finding code (docs/report-codes.md)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
