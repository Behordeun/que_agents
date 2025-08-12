from unittest.mock import patch

import pandas as pd
import pytest

from src.que_agents.utils.data_validator import validate_customer_feedback_csv


class TestDataValidator:
    """Test data validation functionality"""

    @patch("src.que_agents.utils.data_validator.pd.read_csv")
    def test_validate_customer_feedback_csv_success(self, mock_read_csv):
        """Test successful CSV validation"""
        mock_df = pd.DataFrame(
            {
                "Customer ID": [1, 2, 3],
                "Feedback Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "Rating": [4, 5, 3],
                "Category": ["Support", "Billing", "Technical"],
                "Resolution Status": ["Resolved", "Pending", "Resolved"],
                "Sentiment": ["Positive", "Neutral", "Negative"],
            }
        )
        mock_read_csv.return_value = mock_df

        result = validate_customer_feedback_csv("test.csv")

        assert result is True
        mock_read_csv.assert_called_once_with("test.csv")

    @patch("src.que_agents.utils.data_validator.pd.read_csv")
    def test_validate_customer_feedback_csv_missing_customer_ids(self, mock_read_csv):
        """Test validation with missing customer IDs"""
        mock_df = pd.DataFrame(
            {
                "Customer ID": [1, None, 3],
                "Feedback Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "Rating": [4, 5, 3],
                "Category": ["Support", "Billing", "Technical"],
                "Resolution Status": ["Resolved", "Pending", "Resolved"],
                "Sentiment": ["Positive", "Neutral", "Negative"],
            }
        )
        mock_read_csv.return_value = mock_df

        result = validate_customer_feedback_csv("test.csv")

        assert result is False

    @patch("src.que_agents.utils.data_validator.pd.read_csv")
    def test_validate_customer_feedback_csv_invalid_dates(self, mock_read_csv):
        """Test validation with invalid dates"""
        mock_df = pd.DataFrame(
            {
                "Customer ID": [1, 2, 3],
                "Feedback Date": ["2023-01-01", "invalid-date", "2023-01-03"],
                "Rating": [4, 5, 3],
                "Category": ["Support", "Billing", "Technical"],
                "Resolution Status": ["Resolved", "Pending", "Resolved"],
                "Sentiment": ["Positive", "Neutral", "Negative"],
            }
        )
        mock_read_csv.return_value = mock_df

        result = validate_customer_feedback_csv("test.csv")

        assert result is False

    @patch("src.que_agents.utils.data_validator.pd.read_csv")
    def test_validate_customer_feedback_csv_missing_ratings(self, mock_read_csv):
        """Test validation with missing ratings"""
        mock_df = pd.DataFrame(
            {
                "Customer ID": [1, 2, 3],
                "Feedback Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "Rating": [4, None, 3],
                "Category": ["Support", "Billing", "Technical"],
                "Resolution Status": ["Resolved", "Pending", "Resolved"],
                "Sentiment": ["Positive", "Neutral", "Negative"],
            }
        )
        mock_read_csv.return_value = mock_df

        result = validate_customer_feedback_csv("test.csv")

        assert result is False

    @patch("src.que_agents.utils.data_validator.pd.read_csv")
    def test_validate_customer_feedback_csv_invalid_rating_range(self, mock_read_csv):
        """Test validation with ratings outside valid range"""
        mock_df = pd.DataFrame(
            {
                "Customer ID": [1, 2, 3],
                "Feedback Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "Rating": [4, 6, 0],  # 6 and 0 are outside valid range
                "Category": ["Support", "Billing", "Technical"],
                "Resolution Status": ["Resolved", "Pending", "Resolved"],
                "Sentiment": ["Positive", "Neutral", "Negative"],
            }
        )
        mock_read_csv.return_value = mock_df

        result = validate_customer_feedback_csv("test.csv")

        assert result is False

    @patch("src.que_agents.utils.data_validator.pd.read_csv")
    def test_validate_customer_feedback_csv_missing_required_fields(
        self, mock_read_csv
    ):
        """Test validation with missing required fields"""
        mock_df = pd.DataFrame(
            {
                "Customer ID": [1, 2, 3],
                "Feedback Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "Rating": [4, 5, 3],
                "Category": ["Support", None, "Technical"],  # Missing category
                "Resolution Status": ["Resolved", "Pending", "Resolved"],
                "Sentiment": ["Positive", "Neutral", "Negative"],
            }
        )
        mock_read_csv.return_value = mock_df

        result = validate_customer_feedback_csv("test.csv")

        assert result is False

    @patch("src.que_agents.utils.data_validator.pd.read_csv")
    def test_validate_customer_feedback_csv_exception(self, mock_read_csv):
        """Test validation with exception during CSV reading"""
        mock_read_csv.side_effect = Exception("File not found")

        result = validate_customer_feedback_csv("nonexistent.csv")

        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
