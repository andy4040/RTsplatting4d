from rt_pipeline.surface_decisions import decide_group, psnr_gain


CFG = dict(decision_observations=3, required_passes=2, min_validation_pixels=128,
           min_validation_views=2, min_gain_db=.5, min_per_view_gain_db=0.,
           min_ablation_gain_db=.1, max_outside_mse_increase=.02)


def history(gains=(.8, .9), outside=0., ablation=.2, pixels=200):
    return [{'iteration':i*1000, 'outside_relative_mse_increase':outside,
             'groups':{'0':[{'pixels':pixels, 'gain_db':g, 'ablation_gain_db':ablation} for g in gains]}}
            for i in range(1, 4)]


def test_repeated_multiview_gain_accepts():
    assert decide_group(history(), 0, CFG)['accepted']
    assert abs(psnr_gain(.01, .005)-3.0102999566) < 1e-8


def test_one_good_view_cannot_hide_bad_view():
    assert not decide_group(history((2., -.1)), 0, CFG)['accepted']


def test_outside_damage_rejects():
    assert not decide_group(history(outside=.03), 0, CFG)['accepted']


def test_background_only_improvement_rejects():
    assert not decide_group(history(ablation=0.), 0, CFG)['accepted']


def test_empty_or_small_region_and_insufficient_history_reject():
    assert not decide_group(history(pixels=0), 0, CFG)['accepted']
    assert not decide_group(history()[:2], 0, CFG)['accepted']


def test_latest_window_and_required_frequency():
    bad = history((0., 0.))
    good = history()
    assert decide_group(bad+good[:2], 0, CFG)['accepted']
    assert not decide_group(good+bad[:2], 0, CFG)['accepted']
