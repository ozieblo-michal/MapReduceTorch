# from format import main

from format.main import download_nltk_resource, synonym_replacement, augment_example, get_synonyms

from unittest.mock import MagicMock, patch
# import nltk


import random

def test_download_nltk_resource_found():
    with patch('nltk.data.find') as mock_find:
        download_nltk_resource('corpora/wordnet')
        mock_find.assert_called_once_with('corpora/wordnet')

def test_download_nltk_resource_not_found():
    with patch('nltk.data.find', side_effect=LookupError), \
         patch('nltk.download') as mock_download:
        download_nltk_resource('corpora/wordnet')
        mock_download.assert_called_once_with('wordnet')



@patch('nltk.corpus.wordnet.synsets')
def test_get_synonyms(mock_synsets):
    mock_synset = MagicMock()
    mock_synset.lemmas.return_value = [MagicMock(name=lambda: 'test_synonym')]
    mock_synsets.return_value = [mock_synset]
    
    synonyms = get_synonyms('test')
    assert 'test_synonym' in synonyms
    assert 'test' not in synonyms


@patch('format.main.get_synonyms', return_value=['replacement'])
def test_synonym_replacement(mock_get_synonyms):
    original_sentence = "This is a test sentence."
    expected_sentence = "This is a replacement sentence."
    modified_sentence = synonym_replacement(original_sentence, n=1)
    assert modified_sentence == expected_sentence


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
