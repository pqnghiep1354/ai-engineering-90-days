"""
Knowledge module for EIA Generator.
"""

from .regulations_kb import (
    RegulationsKB,
    Regulation,
    Standard,
    get_regulations_kb,
    QCVN_AIR_AMBIENT,
    QCVN_AIR_INDUSTRIAL,
    QCVN_WASTEWATER_INDUSTRIAL,
    QCVN_WASTEWATER_DOMESTIC,
    QCVN_NOISE,
    QCVN_VIBRATION,
)

__all__ = [
    "RegulationsKB",
    "Regulation",
    "Standard",
    "get_regulations_kb",
    "QCVN_AIR_AMBIENT",
    "QCVN_AIR_INDUSTRIAL",
    "QCVN_WASTEWATER_INDUSTRIAL",
    "QCVN_WASTEWATER_DOMESTIC",
    "QCVN_NOISE",
    "QCVN_VIBRATION",
]
