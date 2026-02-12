"""Unit tests for sentiment analysis module."""

import pytest

from config.settings import Config
from src.sentiment import SentimentAnalyzer


class TestSentimentFilters:
    def test_extreme_greed_blocks_long(self):
        sa = SentimentAnalyzer(Config())
        assert sa.should_block_long(90) is True
        assert sa.should_block_long(75) is True

    def test_normal_greed_allows_long(self):
        sa = SentimentAnalyzer(Config())
        assert sa.should_block_long(74) is False
        assert sa.should_block_long(50) is False

    def test_extreme_fear_blocks_short(self):
        sa = SentimentAnalyzer(Config())
        assert sa.should_block_short(10) is True
        assert sa.should_block_short(25) is True

    def test_normal_fear_allows_short(self):
        sa = SentimentAnalyzer(Config())
        assert sa.should_block_short(26) is False
        assert sa.should_block_short(50) is False

    def test_disabled_blocks_nothing(self):
        config = Config()
        config.sentiment.fear_greed_enabled = False
        sa = SentimentAnalyzer(config)
        assert sa.should_block_long(100) is False
        assert sa.should_block_short(0) is False

    def test_funding_favors_short(self):
        sa = SentimentAnalyzer(Config())
        assert sa.funding_favors_short(0.015) is True
        assert sa.funding_favors_short(0.005) is False

    def test_funding_favors_long(self):
        sa = SentimentAnalyzer(Config())
        assert sa.funding_favors_long(-0.015) is True
        assert sa.funding_favors_long(-0.005) is False
