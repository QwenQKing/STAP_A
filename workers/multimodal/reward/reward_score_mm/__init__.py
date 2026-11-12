def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if extra_info and extra_info.get("reward_mode") == "stap_multi_anchor":
        from . import stap_score

        res = stap_score.compute_score(solution_str, ground_truth, extra_info=extra_info)
        return float(res) if isinstance(res, (int, float, bool)) else float(res[0])

    raise ValueError("This anonymous package only includes the STAP multi-anchor reward path.")

