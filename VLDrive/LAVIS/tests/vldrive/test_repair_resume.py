import pytest

from resume_morai_repair import acquire_training_lock, remaining_epochs, validate_full_phase


def settings_and_pilot():
    settings = dict(completed_pilot_updates=200, full_stop_update=1480,
                    full_epochs=2, draws_per_epoch=640, seed=607,
                    checkpoint_every=200, checkpoint_slots=2,
                    model=dict(training_variant='morai_repair_v1'))
    pilot = dict(updates=200, gate=dict(passed=True, checks=dict(direction=True, retention=True)))
    return settings, pilot


def test_recovery_rejects_an_existing_worker_lock(tmp_path):
    pytest.importorskip('fcntl')
    guard = acquire_training_lock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match='already running'):
            acquire_training_lock(tmp_path)
    finally:
        guard.close()
    replacement = acquire_training_lock(tmp_path)
    replacement.close()


def test_recovery_rejects_failed_pilot_or_changed_horizon():
    settings, pilot = settings_and_pilot()
    assert validate_full_phase(settings, pilot, updates=600) == (200, 1480)
    pilot['gate']['checks']['direction'] = False
    with pytest.raises(ValueError, match='pilot gate'):
        validate_full_phase(settings, pilot)
    pilot['gate']['checks']['direction'] = True
    settings['full_stop_update'] += 1
    with pytest.raises(ValueError, match='update contract'):
        validate_full_phase(settings, pilot)


@pytest.mark.parametrize('updates', [0, 199, 1480, 1481])
def test_recovery_requires_an_unfinished_full_phase(updates):
    settings, pilot = settings_and_pilot()
    with pytest.raises(ValueError, match='unfinished full'):
        validate_full_phase(settings, pilot, updates=updates)


@pytest.mark.parametrize('resume_update', [200, 600, 839, 840, 1479])
def test_resume_balanced_sampling_cursor_matches_uninterrupted_suffix(resume_update):
    """The bag seed and its permutation cursor must match across an interruption."""
    import random
    import torch
    settings, _ = settings_and_pilot()

    def sequence(start):
        result = []
        for segment in remaining_epochs(settings, start):
            # Reproduce the two-stage seeded sampling: a balanced/random bag
            # is regenerated per full epoch, then train shuffles the bag.
            bag = random.Random(segment['seed']).choices(range(113), k=640)
            order = torch.randperm(640, generator=torch.Generator().manual_seed(settings['seed'])).tolist()
            for step in range(segment['start_update'], segment['updates']):
                local_offset = step - segment['sampling_step_offset']
                result.append((step, bag[order[local_offset]]))
        return result

    expected = [(step, index) for step, index in sequence(200) if step >= resume_update]
    assert sequence(resume_update) == expected
    assert len(expected) == settings['full_stop_update'] - resume_update
