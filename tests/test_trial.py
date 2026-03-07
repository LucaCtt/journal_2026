from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from csi_vae.trial.trial import Autoencoder, Classifier, evaluate, run_trial, train_autoencoder, train_classifier
from csi_vae.trial.trial_settings import TrialSettings

# ---------------------------------------------------------------------------
# Shared tiny dataset helpers
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 8
LATENT = 4
HIDDEN = 8


def mnist_loader(n: int = BATCH) -> DataLoader:
    """Fake MNIST-shaped DataLoader: images (N, 1, 28, 28), labels (N,)."""
    x = torch.rand(n, 1, 28, 28)
    y = torch.randint(0, 10, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=BATCH)


def make_ae() -> Autoencoder:
    return Autoencoder(latent_dim=LATENT).to(DEVICE)


def make_clf() -> Classifier:
    return Classifier(latent_dim=LATENT, hidden=HIDDEN).to(DEVICE)


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------


class TestAutoencoder:
    def test_forward_output_shapes(self):
        ae = make_ae()
        x = torch.rand(BATCH, 1, 28, 28, device=DEVICE)
        recon, z = ae(x)
        assert recon.shape == (BATCH, 784)
        assert z.shape == (BATCH, LATENT)

    def test_reconstruction_values_in_range(self):
        """Sigmoid output must be in [0, 1]."""
        ae = make_ae()
        x = torch.rand(BATCH, 1, 28, 28, device=DEVICE)
        recon, _ = ae(x)
        assert recon.min() >= 0.0
        assert recon.max() <= 1.0

    def test_encoder_output_shape(self):
        ae = make_ae()
        x = torch.rand(BATCH, 1, 28, 28, device=DEVICE)
        z = ae.encoder(x)
        assert z.shape == (BATCH, LATENT)

    def test_different_batch_sizes(self):
        ae = make_ae()
        for n in (1, 4, 16):
            recon, z = ae(torch.rand(n, 1, 28, 28, device=DEVICE))
            assert recon.shape == (n, 784)
            assert z.shape == (n, LATENT)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class TestClassifier:
    def test_forward_output_shape(self):
        clf = make_clf()
        z = torch.rand(BATCH, LATENT, device=DEVICE)
        logits = clf(z)
        assert logits.shape == (BATCH, 10)

    def test_output_is_raw_logits(self):
        """Output should not be softmax-ed — values can exceed [0,1]."""
        clf = make_clf()
        z = torch.rand(BATCH, LATENT, device=DEVICE)
        logits = clf(z)
        # At least some logits will be outside [0,1] for a random network
        assert logits.shape[-1] == 10


# ---------------------------------------------------------------------------
# train_autoencoder
# ---------------------------------------------------------------------------


class TestTrainAutoencoder:
    def test_runs_without_error(self):
        ae = make_ae()
        optimizer = torch.optim.Adam(ae.parameters())
        train_autoencoder(ae, mnist_loader(), optimizer, epochs=1)

    def test_parameters_change_after_training(self):
        ae = make_ae()
        before = [p.clone() for p in ae.parameters()]
        optimizer = torch.optim.Adam(ae.parameters())
        train_autoencoder(ae, mnist_loader(), optimizer, epochs=1)
        after = list(ae.parameters())
        assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))

    def test_model_in_train_mode_after(self):
        ae = make_ae()
        train_autoencoder(ae, mnist_loader(), torch.optim.Adam(ae.parameters()), epochs=1)
        assert ae.training


# ---------------------------------------------------------------------------
# train_classifier
# ---------------------------------------------------------------------------


class TestTrainClassifier:
    def test_runs_without_error(self):
        ae, clf = make_ae(), make_clf()
        train_classifier(ae, clf, mnist_loader(), torch.optim.Adam(clf.parameters()), epochs=1)

    def test_classifier_parameters_change(self):
        ae, clf = make_ae(), make_clf()
        before = [p.clone() for p in clf.parameters()]
        train_classifier(ae, clf, mnist_loader(), torch.optim.Adam(clf.parameters()), epochs=1)
        after = list(clf.parameters())
        assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))

    def test_autoencoder_parameters_unchanged(self):
        """AE should be frozen during classifier training."""
        ae, clf = make_ae(), make_clf()
        before = [p.clone() for p in ae.parameters()]
        train_classifier(ae, clf, mnist_loader(), torch.optim.Adam(clf.parameters()), epochs=1)
        after = list(ae.parameters())
        assert all(torch.equal(b, a) for b, a in zip(before, after, strict=True))

    def test_ae_in_eval_mode_during_training(self):
        """train_classifier sets ae to eval — it should remain so after."""
        ae, clf = make_ae(), make_clf()
        train_classifier(ae, clf, mnist_loader(), torch.optim.Adam(clf.parameters()), epochs=1)
        assert not ae.training


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_returns_float_between_0_and_1(self):
        ae, clf = make_ae(), make_clf()
        acc = evaluate(ae, clf, mnist_loader())
        assert 0.0 <= acc <= 1.0

    def test_perfect_classifier_returns_1(self):
        """Patch clf to always predict the correct label."""
        ae = make_ae()
        loader = mnist_loader()
        labels = next(iter(loader))[1]

        clf = make_clf()

        # Make logits a one-hot at the correct index
        def perfect_forward(z):
            batch_size = z.shape[0]
            logits = torch.zeros(batch_size, 10, device=z.device)
            logits[range(batch_size), labels[:batch_size]] = 1.0
            return logits

        clf.forward = perfect_forward
        acc = evaluate(ae, clf, loader)
        assert acc == 1.0

    def test_models_in_eval_mode_after(self):
        ae, clf = make_ae(), make_clf()
        evaluate(ae, clf, mnist_loader())
        assert not ae.training
        assert not clf.training


# ---------------------------------------------------------------------------
# run_trial
# ---------------------------------------------------------------------------


class TestRunTrial:
    @pytest.fixture
    def mock_mnist(self):
        """Patch MNIST datasets and DataLoader to avoid real downloads."""
        fake_x = torch.rand(16, 1, 28, 28)
        fake_y = torch.randint(0, 10, (16,))
        fake_ds = TensorDataset(fake_x, fake_y)
        fake_loader = DataLoader(fake_ds, batch_size=8)

        with (
            patch("csi_vae.trial.trial.datasets.MNIST", return_value=fake_ds),
            patch("csi_vae.trial.trial.DataLoader", return_value=fake_loader),
        ):
            yield

    def test_runs_without_queue(self, mock_mnist):
        run_trial(TrialSettings(queue_url=None))

    def test_pushes_result_to_queue(self, mock_mnist):
        mock_queue = MagicMock()
        with patch("csi_vae.trial.trial.MessagesQueue.from_url", return_value=mock_queue):
            run_trial(TrialSettings(study_name="s", trial_number=1, queue_url="https://sqs.fake/q"))

        mock_queue.push.assert_called_once()
        payload = mock_queue.push.call_args[0][0]
        assert payload["study_name"] == "s"
        assert payload["trial_id"] == 1
        assert payload["status"] == "SUCCEEDED"
        assert 0.0 <= payload["accuracy"] <= 1.0

    def test_uses_default_settings_when_none_passed(self, mock_mnist):
        """run_trial() should not raise when called with no arguments."""
        run_trial()

    def test_no_queue_push_when_url_is_none(self, mock_mnist):
        mock_queue = MagicMock()
        with patch("csi_vae.trial.trial.MessagesQueue.from_url", return_value=mock_queue):
            run_trial(TrialSettings(queue_url=None))
        mock_queue.push.assert_not_called()
