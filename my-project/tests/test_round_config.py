"""What the server tells every client about the round it is about to run.

The pictures Ultralytics draws are not free and were being drawn six times to be kept
once. These assert the decision, not the drawing.
"""
import pytest

from my_project.server_app import round_config


def test_plots_are_drawn_on_the_round_whose_output_survives():
    """The client passes exist_ok=True, so every round overwrote the previous round's
    pictures in the same directory. Drawing them in rounds 1..n-1 cost GPU time for
    files that were destroyed before anything read them."""
    drew = [round_config(r, 6, 4)["plots"] for r in range(1, 7)]
    assert drew == [False, False, False, False, False, True]


def test_a_single_round_run_still_draws_its_pictures():
    """Off-by-one guard: with num_rounds=1 the first round IS the last one, and a
    dashboard with no pictures at all would look like a broken artifact server."""
    assert round_config(1, 1, 4)["plots"] is True


def test_a_run_that_overshoots_its_round_count_still_draws():
    """`>=`, not `==`. A strategy that runs an extra round must not silently produce
    a run with no diagnostics at all."""
    assert round_config(7, 6, 4)["plots"] is True


def test_plots_every_round_restores_the_old_behaviour():
    assert all(round_config(r, 6, 4, plots_every_round=True)["plots"]
               for r in range(1, 7))


def test_lr0_and_mosaic_default_to_sentinels_that_mean_leave_ultralytics_alone():
    """0.0 is a legitimate mosaic value meaning 'off', so it cannot double as
    'unspecified' -- a default silently flipped to its opposite is this project's
    signature bug. Negative is the sentinel; lr0 uses 0.0 because no run wants a
    learning rate of zero."""
    cfg = round_config(1, 6, 4)
    assert cfg["mosaic"] < 0, "mosaic sentinel must not collide with 'mosaic off'"
    assert cfg["lr0"] == 0.0
    assert cfg["optimizer"] == "auto"


def test_the_round_config_carries_nothing_per_vehicle():
    """FedAvg hands ONE FitIns to every client, so anything per-vehicle put here
    arrives as the last value written -- the B9 bug, where the whole fleet trained a
    single shard. batch_id is injected per client in configure_fit instead, and this
    asserts nobody moves it back."""
    cfg = round_config(3, 6, 4)
    assert "batch_id" not in cfg
    assert "proximal_mu" not in cfg


@pytest.mark.parametrize("rounds", [1, 2, 6, 12])
def test_exactly_one_round_draws_pictures(rounds):
    drew = [round_config(r, rounds, 1)["plots"] for r in range(1, rounds + 1)]
    assert sum(drew) == 1, f"{sum(drew)} of {rounds} rounds draw plots, expected 1"


def test_the_backbone_freeze_is_round_one_only():
    """A freeze that stayed on would federate a frozen backbone for the whole run --
    a different experiment wearing this one's name. Round 1 is the round where the
    part-random head is pulling good features apart; after it, there is nothing to
    protect them from."""
    sent = [round_config(r, 6, 4, freeze_round1=10)["freeze"] for r in range(1, 7)]
    assert sent == [10, 0, 0, 0, 0, 0]


def test_freeze_is_off_unless_asked_for():
    """Default 0 so adding the lever changes no existing number. The client only passes
    `freeze` to Ultralytics when it is > 0, because freeze=0 is a value, not a silence."""
    assert round_config(1, 6, 4)["freeze"] == 0
    assert all(round_config(r, 3, 1)["freeze"] == 0 for r in range(1, 4))


def test_the_image_size_is_a_property_of_the_round():
    """Broadcast rather than read from a module constant, so the holdout and the
    centralised baseline can be scored at the same resolution. The client used to take it
    from DEFAULT_IMAGE_SIZE, which made the federation's size unreachable from the
    pipeline -- and an unfair comparison one edit away."""
    assert round_config(1, 6, 4)["imgsz"] == 640
    assert round_config(3, 6, 4, imgsz=1024)["imgsz"] == 1024


def test_the_evaluate_floor_does_not_pin_the_whole_fleet():
    """`fraction_evaluate` was inert: Flower takes
    `max(int(available * fraction_evaluate), min_evaluate_clients)`, and this project
    passed `min_evaluate_clients = min_clients`, which the pipeline sets to the vehicle
    count. So the floor WAS the fleet and the lever could not reduce anything.

    Measured before the fix: a 2-round run at 0.34 still did 12 self-evaluations -- six
    clients, both rounds, full participation."""
    from my_project.server_app import evaluate_floor

    assert evaluate_floor(1.0, 6) == 6, "unchanged when every vehicle should evaluate"
    assert evaluate_floor(0.34, 6) == 2, "0.34 of six is two, and two is what must be asked"
    assert evaluate_floor(0.5, 6) == 3
    assert evaluate_floor(0.01, 6) == 1, "never zero: metrics.csv needs at least one row"
    assert evaluate_floor(2.0, 6) == 6, "never more than the fleet"
