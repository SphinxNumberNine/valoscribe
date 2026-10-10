"""
Data types for detection results.

This module contains Pydantic models for structured detection outputs
from various CV detectors (OCR, ability detection, etc.).
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class RoundInfo(BaseModel):
    """Information about the current round."""

    round_number: int = Field(..., ge=1, le=100, description="Round number (1-24 regular, 25+ overtime)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence (0-1)")
    raw_text: Optional[str] = Field(None, description="Raw OCR text before parsing")


class ScoreInfo(BaseModel):
    """Information about team scores."""

    team1_score: int = Field(..., ge=0, le=99, description="Team 1 score (0-13 regular, higher in overtime)")
    team2_score: int = Field(..., ge=0, le=99, description="Team 2 score (0-13 regular, higher in overtime)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence (0-1)")
    team1_raw_text: Optional[str] = Field(None, description="Raw OCR text for team 1")
    team2_raw_text: Optional[str] = Field(None, description="Raw OCR text for team 2")


class TimerInfo(BaseModel):
    """Information about round timer."""

    time_seconds: float = Field(..., ge=0.0, le=200.0, description="Time remaining in seconds")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence (0-1)")
    raw_text: Optional[str] = Field(None, description="Raw detected digits before parsing")


class SpikeInfo(BaseModel):
    """Information about spike detection."""

    spike_planted: bool = Field(..., description="Whether spike is planted")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence (0-1)")


class AbilityInfo(BaseModel):
    """Information about ability charges."""

    charges: int = Field(..., ge=0, le=10, description="Number of charges available (0 = on cooldown)")
    total_blobs_detected: int = Field(..., ge=0, description="Total blobs found before filtering")


class CreditsInfo(BaseModel):
    """Information about credits icon detection."""

    credits_visible: bool = Field(
        ...,
        description="Whether the detector matched its credits icon in the selected HUD region; "
        "the active-round detector checks dead_credits, the preround detector checks credits",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence (0-1)")


class UltimateInfo(BaseModel):
    """Information about ultimate ability charges."""

    charges: int = Field(..., ge=0, le=8, description="Number of ultimate points/segments (0-8)")
    is_full: bool = Field(..., description="Whether ultimate is fully charged (solid ring)")
    total_blobs_detected: int = Field(..., ge=0, description="Total blobs found before filtering")


class HealthInfo(BaseModel):
    """Information about player health."""

    health: int = Field(..., ge=0, le=150, description="Player health (0-150)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence (0-1)")
    raw_text: Optional[str] = Field(None, description="Raw detected digits before parsing")


class ArmorInfo(BaseModel):
    """Information about player armor."""

    armor: int = Field(..., ge=0, le=50, description="Player armor (0-50)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence (0-1)")
    raw_text: Optional[str] = Field(None, description="Raw detected digits before parsing")


class AgentInfo(BaseModel):
    """Information about detected agent."""

    agent_name: str = Field(..., description="Name of the detected agent")
    side: str = Field(..., description="Side of the agent: 'attack' or 'defense'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence (0-1)")


class KillfeedAgentDetection(BaseModel):
    """Information about a killfeed entry with agent detection."""

    killer_agent: str = Field(..., description="Agent name of the killer")
    killer_side: str = Field(..., description="Side of killer: 'attack' or 'defense'")
    victim_agent: str = Field(..., description="Agent name of the victim")
    victim_side: str = Field(..., description="Side of victim: 'attack' or 'defense'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence (0-1)")
    weapon: Optional[str] = Field(None, description="Weapon used (TODO)")


class KillEvent(BaseModel):
    """Information about a kill event from killfeed."""

    killer: str = Field(..., description="Player name who got the kill")
    victim: str = Field(..., description="Player name who was killed")
    weapon: Optional[str] = Field(None, description="Weapon used")
    headshot: bool = Field(False, description="Whether it was a headshot")
    timestamp_ms: float = Field(..., description="Timestamp in video (milliseconds)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence (0-1)")
