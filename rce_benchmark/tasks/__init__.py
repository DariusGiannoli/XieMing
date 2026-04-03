"""Benchmark stage runners."""

from .stage1_speed import run_stage1_speed
from .stage2_middlebury import run_stage2_middlebury

__all__ = ["run_stage1_speed", "run_stage2_middlebury"]
