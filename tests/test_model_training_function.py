import pytest
from unittest.mock import MagicMock, patch
import optuna
# from transformers import DistilBertTokenizer, DistilBertForMaskedLM, Trainer, TrainingArguments
# import transformers
from training import script
from datasets import Dataset

@pytest.fixture
def mock_dependencies():
    with patch("transformers.DistilBertTokenizer.from_pretrained", return_value=MagicMock()) as mock_tokenizer, \
         patch("transformers.DistilBertForMaskedLM.from_pretrained", return_value=MagicMock()) as mock_model, \
         patch("transformers.Trainer", return_value=MagicMock(evaluate=MagicMock(return_value={"eval_loss": 0.5}))) as mock_trainer:
        yield mock_tokenizer, mock_model, mock_trainer

def test_model_training_function_uses_suggested_hyperparameters(mock_dependencies):

#     data = {
#     "text": ["The quick brown fox jumps over the lazy dog", "I love machine learning"]
# }
    
#     augmented_sentences = [script.synonym_replacement(sentence, n=2) for sentence in data["text"]]

#     dataset = Dataset.from_dict(augmented_sentences)

#     tokenized_dataset = dataset.map(script.tokenize_and_mask_function, batched=True)

    # global train_dataset, eval_dataset
    # train_dataset = tokenized_dataset
    # eval_dataset = tokenized_dataset




    with patch('training.script.train_dataset', return_value=MagicMock()), \
        patch('training.script.eval_dataset', return_value=MagicMock()):
        # Your test code here



        mock_tokenizer, mock_model, mock_trainer = mock_dependencies

        trial = optuna.trial.FixedTrial({'num_train_epochs': 3, 'learning_rate': 1e-4})

        
        # train_dataset = MagicMock()
        # eval_dataset = MagicMock()

        eval_loss = script.model_training_function(trial)

        mock_model.assert_called_once() 
        mock_trainer.return_value.train.assert_called_once()
        assert eval_loss == 0.5, "Expected evaluation loss does not match"

        _, kwargs = mock_trainer.call_args
        assert kwargs['args'].num_train_epochs == 3, "Num train epochs does not match suggestion"
        assert kwargs['args'].learning_rate == 1e-4, "Learning rate does not match suggestion"
