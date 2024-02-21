from format.main import download_nltk_resource, synonym_replacement, augment_example, get_synonyms
from unittest.mock import patch 
import random

def test_download_nltk_resource_found():
    with patch('nltk.data.find') as mock_find:
        download_nltk_resource('corpora/wordnet')
        mock_find.assert_called_once_with('corpora/wordnet')

@patch('format.main.synonym_replacement', return_value='augmented text')
def test_augment_example_with_augmentation(mock_replacement):
    random.seed(0)
    example = {'text': 'original text'}
    augmented_example = augment_example(example, augment_rate=1.0)
    assert augmented_example['text'] == 'augmented text'

@patch('format.main.synonym_replacement', return_value='augmented text')
def test_augment_example_without_augmentation(mock_replacement):
    random.seed(0)
    example = {'text': 'original text'}
    augmented_example = augment_example(example, augment_rate=0.0)
    assert augmented_example['text'] == 'original text'
