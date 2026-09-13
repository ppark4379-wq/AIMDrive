from train_morai_repair import category, check_gate, summarize


def test_direction_requires_actual_lateral_motion_not_just_sign():
    rows = [dict(category='moving_left', actual_future=[[3., -10.]]*5,
                 prediction=[[.01, -10.]]*5, errors_m=[2.99]*5)]
    assert summarize(rows)['moving_left']['direction_accuracy'] == 0.


def test_topk_budget_cannot_pass_saturated_training_selector():
    before = {'curved': {'ade_m': 3.}}
    after = {'curved': {'ade_m': 1.}, 'moving_left': {'direction_accuracy': 1.},
             'moving_right': {'direction_accuracy': 1.}}
    settings = dict(minimum_updates=160, curved_ade_max_m=1.5, curved_ade_ratio_max=.8,
        direction_accuracy_min=.8, retention_min=.15, retention_max=.5, saturated_keep_fraction_max=.05)
    telemetry = dict(sampled_retention=1., soft_retention=.9999, saturated_keep_fraction=1.)
    assert not check_gate(before, after, telemetry, settings, 640)['passed']
    telemetry = dict(sampled_retention=.36, soft_retention=.35, saturated_keep_fraction=0.)
    assert check_gate(before, after, telemetry, settings, 160)['passed']
    assert not check_gate(before, after, telemetry, settings, 159)['passed']


def test_categories_match_actual_future_head_coordinate_convention():
    assert category([[0., 0.]]*5) == 'stationary'
    assert category([[0., -10.]]*5) == 'moving_straight'
    assert category([[3., -10.]]*5) == 'moving_left'
    assert category([[-3., -10.]]*5) == 'moving_right'
