import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from ue_helper.utils import percentage_to_zscore

KEY_LIME = "#EBF38B"
INDIGO = "#16425B"
INDIGO_50 = "#8AA0AD"
KEPPEL = "#16D5C2"
KEPPEL_50 = "#8AEAE1"
BLACK = "#000000"
GREY_80 = "#333333"

def percentage_to_zscore(confidence_percent: float) -> float:
    """
    Convert a two-tailed confidence percentage to the corresponding z-score.

    The mapping assumes a standard normal distribution and a two-sided interval:
    z = Φ⁻¹(0.5 + p/200), where Φ is the standard normal CDF and p is the percentage.

    Examples
    --------
    68% -> 1.00
    90% -> 1.645
    95% -> 1.96
    99% -> 2.576

    Parameters
    ----------
    confidence_percent : float
        Confidence level in percent (e.g., 95 for a 95% CI).

    Returns
    -------
    float
        The corresponding z-score for a two-tailed interval.
    """
    confidence_fraction = confidence_percent / 100.0
    return float(norm.ppf(0.5 + confidence_fraction / 2.0))

def plot_validation(
    ground_truth_df,
    prediction_df,
    uncertainty_df,
    validation_param=None,
    title=None,
    uncertainty_percentage=95,
):
    """
    Simple validation plot for a given parameter.

    Circles (INDIGO)  = ground truth
    Triangles (KEPPEL) + error bars = prediction ± uncertainty
    """
    sigma_scale = percentage_to_zscore(uncertainty_percentage)
    if validation_param is None:
        validation_param = prediction_df.columns[0]
    # Filter data
    gt = ground_truth_df[validation_param].to_numpy()
    pr = prediction_df[validation_param].to_numpy()
    unc = uncertainty_df[validation_param].to_numpy()

    # Use index as validation point
    x = np.arange(len(gt))

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))

    # Ground truth circles
    ax.scatter(x, gt, color=INDIGO, marker='x', label='Ground truth', zorder=3)

    # Predictions with error bars (KEPPEL triangles)
    ax.errorbar(
        x, pr, yerr=sigma_scale*unc,
        fmt='^', color=KEPPEL, ecolor=KEPPEL,
        elinewidth=1.5, capsize=4, label=f'Prediction ± {uncertainty_percentage}% Uncertainty', zorder=2
    )

    # Cosmetics
    ax.set_xlabel("Validation point")
    ax.set_ylabel(f"{validation_param}")
    ax.set_title(title or f"Validation — {validation_param}")
    ax.grid(alpha=0.2)
    ax.legend(frameon=True)
    ax.set_xlim(-0.5, len(x)-0.5)

    plt.tight_layout()
    return fig, ax