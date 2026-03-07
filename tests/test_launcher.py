from unittest.mock import MagicMock, patch

import optuna
import pytest

from csi_vae.launcher.launcher import _collect_results, run_launcher, run_study
from csi_vae.launcher.launcher_settings import LauncherSettings

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings():
    return LauncherSettings(
        study_name="test-study",
        journal_path=None,  # in-memory study, no disk I/O
        n_trials=4,
        trials_batch_size=2,
        poll_interval=0,  # no sleeping in tests
        aws_job_queue="test-queue",
        aws_job_definition="test-def",
        param_lr_min=1e-4,
        param_lr_max=1e-2,
    )


@pytest.fixture
def mock_queue():
    q = MagicMock()
    q.url = "https://sqs.fake/queue"
    q.pop.return_value = []
    return q


@pytest.fixture
def mock_submitter():
    s = MagicMock()
    s.submit.return_value = "batch-job-id"
    return s


# ---------------------------------------------------------------------------
# LauncherSettings
# ---------------------------------------------------------------------------


class TestLauncherSettings:
    def test_n_batches_exact_division(self):
        s = LauncherSettings(n_trials=10, trials_batch_size=2)
        assert s.n_batches == 5

    def test_n_batches_rounds_up(self):
        s = LauncherSettings(n_trials=5, trials_batch_size=2)
        assert s.n_batches == 3

    def test_n_batches_single_trial(self):
        s = LauncherSettings(n_trials=1, trials_batch_size=2)
        assert s.n_batches == 1

    def test_n_batches_batch_larger_than_trials(self):
        s = LauncherSettings(n_trials=2, trials_batch_size=10)
        assert s.n_batches == 1


# ---------------------------------------------------------------------------
# _collect_results
# ---------------------------------------------------------------------------


class TestCollectResults:
    def test_returns_accuracy_for_succeeded_trial(self, mock_queue):
        mock_queue.pop.return_value = [{"trial_id": 1, "status": "SUCCEEDED", "accuracy": 0.95}]
        results = _collect_results(mock_queue, [1], poll_interval=0)
        assert results == [(1, 0.95)]

    def test_returns_none_accuracy_for_failed_trial(self, mock_queue):
        mock_queue.pop.return_value = [{"trial_id": 1, "status": "FAILED", "accuracy": 0.95}]
        results = _collect_results(mock_queue, [1], poll_interval=0)
        assert results == [(1, None)]

    def test_collects_multiple_trials(self, mock_queue):
        mock_queue.pop.side_effect = [
            [{"trial_id": 0, "status": "SUCCEEDED", "accuracy": 0.9}],
            [{"trial_id": 1, "status": "SUCCEEDED", "accuracy": 0.8}],
        ]
        results = _collect_results(mock_queue, [0, 1], poll_interval=0)
        assert sorted(results) == [(0, 0.9), (1, 0.8)]

    def test_ignores_unknown_trial_ids(self, mock_queue):
        mock_queue.pop.side_effect = [
            [{"trial_id": 99, "status": "SUCCEEDED", "accuracy": 0.5}],
            [{"trial_id": 1, "status": "SUCCEEDED", "accuracy": 0.7}],
        ]
        results = _collect_results(mock_queue, [1], poll_interval=0)
        assert results == [(1, 0.7)]

    def test_polls_until_all_complete(self, mock_queue):
        """Queue returns one result per poll — should loop until both done."""
        mock_queue.pop.side_effect = [
            [],
            [{"trial_id": 0, "status": "SUCCEEDED", "accuracy": 0.6}],
            [{"trial_id": 1, "status": "SUCCEEDED", "accuracy": 0.7}],
        ]
        results = _collect_results(mock_queue, [0, 1], poll_interval=0)
        assert len(results) == 2
        assert mock_queue.pop.call_count == 3

    def test_missing_trial_id_in_message_is_ignored(self, mock_queue):
        mock_queue.pop.side_effect = [
            [{"status": "SUCCEEDED", "accuracy": 0.9}],  # no trial_id
            [{"trial_id": 1, "status": "SUCCEEDED", "accuracy": 0.8}],
        ]
        results = _collect_results(mock_queue, [1], poll_interval=0)
        assert results == [(1, 0.8)]


# ---------------------------------------------------------------------------
# run_study
# ---------------------------------------------------------------------------


class TestRunStudy:
    @pytest.fixture
    def in_memory_study(self):
        return optuna.create_study(direction="maximize")

    def _make_succeeded_queue(self, mock_queue, trial_numbers: list[int]):
        """Configure mock_queue.pop to return one result per call per trial."""
        mock_queue.pop.side_effect = [[{"trial_id": t, "status": "SUCCEEDED", "accuracy": 0.8}] for t in trial_numbers]

    def test_submits_correct_number_of_jobs(self, mock_submitter, mock_queue, settings):
        # 4 trials total, batch size 2 → 2 batches of 2
        mock_queue.pop.side_effect = [
            [{"trial_id": t, "status": "SUCCEEDED", "accuracy": 0.8}] for t in range(settings.n_trials)
        ]
        run_study(mock_submitter, mock_queue, settings)

        assert mock_submitter.submit.call_count == settings.n_trials

    def test_tells_study_accuracy_on_success(self, mock_submitter, mock_queue, settings, in_memory_study):
        with patch("csi_vae.launcher.launcher.optuna.create_study", return_value=in_memory_study):
            mock_queue.pop.side_effect = [
                [{"trial_id": t, "status": "SUCCEEDED", "accuracy": 0.75}] for t in range(settings.n_trials)
            ]
            run_study(mock_submitter, mock_queue, settings)

        assert all(t.value == 0.75 for t in in_memory_study.trials)

    def test_tells_study_fail_on_failed_trial(self, mock_submitter, mock_queue, settings, in_memory_study):
        with patch("csi_vae.launcher.launcher.optuna.create_study", return_value=in_memory_study):
            mock_queue.pop.side_effect = [[{"trial_id": t, "status": "FAILED"}] for t in range(settings.n_trials)]
            run_study(mock_submitter, mock_queue, settings)

        assert all(t.state == optuna.trial.TrialState.FAIL for t in in_memory_study.trials)

    def test_trial_settings_passed_to_submitter(self, mock_submitter, mock_queue, settings, in_memory_study):
        with patch("csi_vae.launcher.launcher.optuna.create_study", return_value=in_memory_study):
            mock_queue.pop.side_effect = [
                [{"trial_id": t, "status": "SUCCEEDED", "accuracy": 0.8}] for t in range(settings.n_trials)
            ]
            run_study(mock_submitter, mock_queue, settings)

        for submit_call in mock_submitter.submit.call_args_list:
            trial_settings = submit_call[0][0]
            assert trial_settings.study_name == settings.study_name
            assert trial_settings.queue_url == mock_queue.url
            assert settings.param_lr_min <= trial_settings.param_lr <= settings.param_lr_max


# ---------------------------------------------------------------------------
# run_launcher
# ---------------------------------------------------------------------------


class TestRunLauncher:
    @pytest.fixture
    def mock_deps(self, settings):
        mock_submitter = MagicMock()
        mock_submitter.submit.return_value = "job-id"
        mock_queue = MagicMock()
        mock_queue.url = "https://sqs.fake/q"

        with (
            patch("csi_vae.launcher.launcher.TrialSubmitter", return_value=mock_submitter),
            patch("csi_vae.launcher.launcher.MessagesQueue", return_value=mock_queue),
            patch("csi_vae.launcher.launcher.run_study") as mock_run_study,
        ):
            yield mock_queue, mock_submitter, mock_run_study

    def test_creates_and_destroys_queue(self, mock_deps, settings):
        mock_queue, _, _ = mock_deps
        run_launcher(settings)
        mock_queue.create.assert_called_once_with(settings.study_name)
        mock_queue.destroy.assert_called_once()

    def test_destroys_queue_even_if_run_study_raises(self, mock_deps, settings):
        mock_queue, _, mock_run_study = mock_deps
        mock_run_study.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError, match="boom"):
            run_launcher(settings)
        mock_queue.destroy.assert_called_once()

    def test_uses_default_settings_when_none(self, mock_deps):
        run_launcher()  # should not raise

    def test_passes_correct_aws_config_to_submitter(self, settings):
        mock_queue = MagicMock()
        mock_queue.url = "https://sqs.fake/q"
        with (
            patch("csi_vae.launcher.launcher.MessagesQueue", return_value=mock_queue),
            patch("csi_vae.launcher.launcher.run_study"),
            patch("csi_vae.launcher.launcher.TrialSubmitter") as mock_submitter_cls,
        ):
            run_launcher(settings)
            mock_submitter_cls.assert_called_once_with(settings.aws_job_queue, settings.aws_job_definition)
