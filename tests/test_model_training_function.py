import pytest
from unittest.mock import MagicMock, patch
import optuna
from transformers import DistilBertTokenizer, DistilBertForMaskedLM, Trainer, TrainingArguments
from training import script

@pytest.fixture
def mock_dependencies():
    with patch("script.DistilBertTokenizer.from_pretrained", return_value=MagicMock()) as mock_tokenizer, \
         patch("script.DistilBertForMaskedLM.from_pretrained", return_value=MagicMock()) as mock_model, \
         patch("script.Trainer", return_value=MagicMock(evaluate=MagicMock(return_value={"eval_loss": 0.5}))) as mock_trainer:
        yield mock_tokenizer, mock_model, mock_trainer

def test_model_training_function_uses_suggested_hyperparameters(mock_dependencies):
    mock_tokenizer, mock_model, mock_trainer = mock_dependencies

    trial = optuna.trial.FixedTrial({'num_train_epochs': 3, 'learning_rate': 1e-4})

    global train_dataset, eval_dataset
    train_dataset = MagicMock()
    eval_dataset = MagicMock()

    eval_loss = script.model_training_function(trial)

    mock_model.assert_called_once() 
    mock_trainer.return_value.train.assert_called_once()
    assert eval_loss == 0.5, "Expected evaluation loss does not match"

    _, kwargs = mock_trainer.call_args
    assert kwargs['args'].num_train_epochs == 3, "Num train epochs does not match suggestion"
    assert kwargs['args'].learning_rate == 1e-4, "Learning rate does not match suggestion"
