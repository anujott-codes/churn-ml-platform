import zipfile
from pathlib import Path
import pytest

from src.components.data_ingestion import DataIngestion
from src.config.data_source_config import (
    KAGGLE_TRAIN_FILENAME,
    KAGGLE_TEST_FILENAME,
    TRAIN_FILENAME,
    TEST_FILENAME,
)
from src.exception import ChurnPipelineException


def create_fake_kaggle_zip(tmp_path: Path):
    """
    Create a valid fake Kaggle zip file containing expected train/test CSVs.
    """
    zip_path = tmp_path / "fake_dataset.zip"

    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.writestr(KAGGLE_TRAIN_FILENAME, "col1,col2\n1,2\n")
        zipf.writestr(KAGGLE_TEST_FILENAME, "col1,col2\n3,4\n")

    return zip_path

## test for data ingestion component
def test_data_ingestion_success(tmp_path, mocker):
    """
    Test successful download and extraction.
    """

    ingestion = DataIngestion(dataset="fake-dataset", output_dir=tmp_path)

    # Mock KaggleApi
    mock_api = mocker.patch(
        "src.components.data_ingestion.KaggleApi"
    ).return_value

    mock_api.authenticate.return_value = None

    def fake_download(dataset, path, unzip):
        create_fake_kaggle_zip(tmp_path)

    mock_api.dataset_download_files.side_effect = fake_download

    ingestion.download_and_extract_kaggle_dataset()

    # Assertions
    assert (tmp_path / TRAIN_FILENAME).exists()
    assert (tmp_path / TEST_FILENAME).exists()

    # Ensure zip file removed
    assert not any(tmp_path.glob("*.zip"))

## test for missing files in zip
def test_data_ingestion_missing_files(tmp_path, mocker):
    """
    Test failure when expected CSVs are missing inside zip.
    """

    ingestion = DataIngestion(dataset="fake-dataset", output_dir=tmp_path)

    mock_api = mocker.patch(
        "src.components.data_ingestion.KaggleApi"
    ).return_value

    mock_api.authenticate.return_value = None

    def fake_download(dataset, path, unzip):
        zip_path = tmp_path / "invalid.zip"
        with zipfile.ZipFile(zip_path, "w") as zipf:
            zipf.writestr("random.csv", "data")

    mock_api.dataset_download_files.side_effect = fake_download

    with pytest.raises(ChurnPipelineException):
        ingestion.download_and_extract_kaggle_dataset()

## test for no zip file found
def test_no_zip_found(tmp_path, mocker):
    """Critical: Test when download doesn't create zip"""
    ingestion = DataIngestion(output_dir=tmp_path)
    mock_api = mocker.patch("src.components.data_ingestion.KaggleApi").return_value
    mock_api.authenticate.return_value = None
    mock_api.dataset_download_files.return_value = None
    
    with pytest.raises(ChurnPipelineException):
        ingestion.download_and_extract_kaggle_dataset()

## test for API authentication failure
def test_api_failure(tmp_path, mocker):
    """Critical: Test API errors"""
    ingestion = DataIngestion(output_dir=tmp_path)
    mock_api = mocker.patch("src.components.data_ingestion.KaggleApi").return_value
    mock_api.authenticate.side_effect = Exception("Auth failed")
    
    with pytest.raises(ChurnPipelineException):
        ingestion.download_and_extract_kaggle_dataset()

## test for file content verification
def test_file_content_preserved(tmp_path, mocker):
    """Important: Verify data integrity"""
    ingestion = DataIngestion(output_dir=tmp_path)
    mock_api = mocker.patch("src.components.data_ingestion.KaggleApi").return_value
    mock_api.authenticate.return_value = None
    mock_api.dataset_download_files.side_effect = lambda **kw: create_fake_kaggle_zip(tmp_path)
    
    ingestion.download_and_extract_kaggle_dataset()
    
    assert "col1,col2" in (tmp_path / TRAIN_FILENAME).read_text()