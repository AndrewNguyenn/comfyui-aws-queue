from affinity import RUN_MAX, pick

A, B = "waiIllustriousSDXL_v170", "2D_Pixel_Sprites"


def test_a_fresh_worker_takes_the_oldest_job():
    assert pick([A, B, A], [30, 10, 20], None, 0) == 1


def test_the_job_for_the_loaded_model_wins_over_an_older_one():
    assert pick([B, A, B], [10, 20, 30], A, 3) == 1


def test_with_nothing_for_its_model_a_worker_switches_to_the_oldest():
    assert pick([B, B], [20, 10], A, 3) == 1


def test_after_run_max_in_a_row_the_waiting_model_gets_its_turn():
    assert pick([A, B], [10, 20], A, RUN_MAX) == 1


def test_run_max_does_not_switch_when_nothing_else_waits():
    assert pick([A, A], [20, 10], A, RUN_MAX + 5) == 1


def test_an_unknown_model_runs_on_any_worker():
    assert pick(["", B], [20, 10], A, 1) == 0
