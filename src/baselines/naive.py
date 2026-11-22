from src.detectors.data import load_window, model_provider
from src.detectors.pass_rate import compare_windows, restrict_to_common_probes
from src.detectors.result import AlertContext, DetectorResult

# Float guard for the boundary: effect is (diff * 100), so a true -5.0 can come out as -4.999999999.
_EPS = 1e-9


def detect_naive(
    model,
    category,
    reference_window,
    current_window,
    drop_points,
    drops_only=False,
    provider="unknown",
    window_runs=(),
):
    if drop_points <= 0:
        raise ValueError(f"drop_points must be positive, got {drop_points}")

    def result(status, reason, effect=None):
        return DetectorResult(
            detector="naive",
            metric_unit="points",
            statistic=effect,
            p_value=None,
            effect_size=effect,
            ci_low=None,
            ci_high=None,
            reason=reason,
            status=status,
            context=AlertContext(
                provider=provider,
                model=model,
                category=category,
                window_runs=list(window_runs),
                threshold=drop_points,
            ),
        )

    if not reference_window or not current_window:
        return result("no_data", "A window has no rows.")

    reference_window, current_window = restrict_to_common_probes(
        reference_window, current_window
    )
    compared = compare_windows(reference_window, current_window)
    if compared is None:
        return result("no_data", "No scored probes appear in both windows.")
    effect = compared[3]

    if drops_only:
        alarm = effect <= -drop_points + _EPS
    else:
        alarm = abs(effect) >= drop_points - _EPS

    fixed = f"the fixed {drop_points:.1f}"
    if effect == 0:
        move = "Pass rate did not change"
    else:
        move = f"Pass rate {'fell' if effect < 0 else 'rose'} {abs(effect):.1f} points"

    if alarm:
        reason = f"{move}, at least {fixed}."
    elif drops_only and effect > 0:
        reason = (
            f"{move}; only drops count here, so no alarm (fixed {drop_points:.1f})."
        )
    else:
        reason = f"{move}, less than {fixed}."
    return result("alarm" if alarm else "ok", reason, effect)


def detect_naive_runs(
    model,
    category,
    reference_runs,
    current_runs,
    drop_points,
    drops_only=False,
    conn=None,
):
    """Load both windows from the database by run ids and run detect_naive on them."""
    return detect_naive(
        model,
        category,
        load_window(model, category, reference_runs, conn),
        load_window(model, category, current_runs, conn),
        drop_points,
        drops_only,
        provider=model_provider(model, conn) or "unknown",
        window_runs=current_runs,
    )
