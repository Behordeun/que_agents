"""
Unit tests for Data Populator to improve code coverage.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.que_agents.utils.data_populator import populate_sample_data


class TestDataPopulator:
    """Test data populator functionality"""

    def test_populate_sample_data_success(self):
        """Test successful sample data population"""
        with patch("src.que_agents.utils.data_populator.get_session") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value = mock_db

            with patch("builtins.print") as mock_print:
                populate_sample_data()

                # Verify session operations
                assert mock_db.add.call_count > 0
                assert mock_db.commit.call_count > 0
                mock_db.close.assert_called_once()

                # Verify success message was printed
                mock_print.assert_called()

    def test_populate_sample_data_error(self):
        """Test sample data population with error"""
        with patch("src.que_agents.utils.data_populator.get_session") as mock_session:
            mock_db = MagicMock()
            mock_db.add.side_effect = Exception("Database error")
            mock_session.return_value = mock_db

            with patch("builtins.print") as mock_print:
                populate_sample_data()

                # Verify rollback was called on error
                mock_db.rollback.assert_called_once()
                mock_db.close.assert_called_once()

                # Verify error message was printed
                mock_print.assert_called()

    def test_populate_sample_data_commit_error(self):
        """Test sample data population with commit error"""
        with patch("src.que_agents.utils.data_populator.get_session") as mock_session:
            mock_db = MagicMock()
            mock_db.commit.side_effect = Exception("Commit error")
            mock_session.return_value = mock_db

            with patch("builtins.print") as mock_print:
                populate_sample_data()

                # Verify rollback was called on commit error
                mock_db.rollback.assert_called_once()
                mock_db.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
