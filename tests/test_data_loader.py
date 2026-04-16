"""Tests for data loaders and parsers."""

import os
import tempfile

from src.data.preprocessor import PreprocessorConfig, TextPreprocessor
from src.utils.text_utils import MBTI_TYPES, clean_text_pipeline, count_words, remove_mbti_mentions


class TestTextUtils:
    def test_remove_urls(self):
        text = "Check out http://example.com for details"
        result = clean_text_pipeline(text, remove_urls=True)
        assert "http" not in result

    def test_remove_mbti_mentions(self):
        text = "As an INTJ, I think logically"
        result = remove_mbti_mentions(text)
        assert "INTJ" not in result
        assert "[TYPE]" not in result

    def test_count_words(self):
        assert count_words("hello world foo") == 3
        assert count_words("") == 0

    def test_all_mbti_types(self):
        assert len(MBTI_TYPES) == 16


class TestTextPreprocessor:
    def test_basic_cleaning(self):
        preprocessor = TextPreprocessor()
        text = "Hello INTJ! Check http://example.com"
        result = preprocessor.clean(text)
        assert "http" not in result
        assert "INTJ" not in result

    def test_min_words_filter(self):
        preprocessor = TextPreprocessor(PreprocessorConfig(min_words=10))
        assert not preprocessor.is_valid("Too short")
        assert preprocessor.is_valid(
            "This is a longer text with many words that should pass the filter check"
        )

    def test_clean_and_validate_returns_none_for_short(self):
        preprocessor = TextPreprocessor(PreprocessorConfig(min_words=10))
        result = preprocessor.clean_and_validate("short")
        assert result is None


class TestMBTIParser:
    def test_parse_dimensions(self):
        from src.data.mbti_parser import parse_mbti_dimensions

        dims = parse_mbti_dimensions("INTP")
        assert dims == {"IE": "I", "SN": "N", "TF": "T", "JP": "P"}

    def test_parse_dimensions_invalid(self):
        from src.data.mbti_parser import parse_mbti_dimensions

        dims = parse_mbti_dimensions("INVALID")
        assert dims == {}

    def test_mbti_parser_with_temp_csv(self):
        import csv

        from src.data.mbti_parser import MBTIParser

        # Create a minimal CSV
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["type", "posts"])
            writer.writerow(
                [
                    "INTP",
                    "I love thinking about abstract possibilities and the connections between ideas|||"
                    "Ideas that make me wonder about the nature of things and how theory meets practice",
                ]
            )
            writer.writerow(
                [
                    "ENFJ",
                    "People are so important to me because I help friends build meaningful connections|||"
                    "I love connecting with friends and supporting their growth whenever I can",
                ]
            )
            temp_path = f.name

        try:
            parser = MBTIParser()
            records = parser.parse(temp_path)
            assert len(records) >= 1
            assert any(r["label_mbti"] == "INTP" for r in records)
        finally:
            os.unlink(temp_path)


class TestEssaysParser:
    def test_label_mapping(self):
        from src.data.essays_parser import label_to_binary

        assert label_to_binary("y") == "HIGH"
        assert label_to_binary("n") == "LOW"
        assert label_to_binary("Y") == "HIGH"
