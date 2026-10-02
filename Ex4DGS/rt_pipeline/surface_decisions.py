"""Predeclared validation rules. Test images never enter this module."""
import numpy as np


def psnr_gain(control_mse, treatment_mse):
    return float(10*np.log10(max(control_mse, 1e-12)/max(treatment_mse, 1e-12)))


def decide_group(history, group_id, config):
    recent = history[-config['decision_observations']:]
    evidence = []
    for observation in recent:
        views = observation['groups'][str(group_id)]
        eligible = [v for v in views if v['pixels'] >= config['min_validation_pixels']]
        gains = [v['gain_db'] for v in eligible]
        ablations = [v['ablation_gain_db'] for v in eligible]
        good = (len(eligible) >= config['min_validation_views']
                and np.mean(gains) >= config['min_gain_db']
                and all(g >= config['min_per_view_gain_db'] for g in gains)
                and np.mean(ablations) >= config['min_ablation_gain_db']
                and observation['outside_relative_mse_increase'] <= config['max_outside_mse_increase'])
        evidence.append({'iteration':observation['iteration'], 'passes':bool(good),
                         'eligible_views':len(eligible), 'mean_gain_db':float(np.mean(gains)) if gains else None})
    accepted = len(recent) == config['decision_observations'] and sum(e['passes'] for e in evidence) >= config['required_passes']
    return {'accepted':accepted, 'evidence':evidence,
            'meaning':'useful reflection/transmission hypothesis' if accepted else 'not supported by this trial; not proof of no glass'}
