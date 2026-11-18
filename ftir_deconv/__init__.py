"""FTIR Si-H peak deconvolution toolkit."""
from .core import (  # noqa: F401
    LITERATURE_PRESETS,
    ANCHOR_METHODS,
    PRESETS,
    TABLE_COLUMNS,
    SIGMA_NH_CM2,
    SIGMA_SIH_CM2,
    TABLE2_SIH,
    FitResult,
    PeakSpec,
    PipelineConfig,
    PipelineOutput,
    TOUCHUP_MODES,
    airpls_baseline,
    apply_pins,
    apply_touchups,
    suggest_touchups,
    als_baseline,
    autolevel_baseline,
    anchor_baseline,
    bond_concentration,
    fit_gaussians,
    linear_edge_baseline,
    load_spectrum,
    model_selection,
    peaks_from_table,
    run_pipeline,
    polyfit_baseline,
    rubberband_baseline,
    smooth_savgol,
    suggest_anchors,
)

from .survey import (  # noqa: F401
    BAND_COLUMNS,
    SURVEY_BASELINES,
    SurveyResult,
    default_band_table,
    survey_spectrum,
)

__version__ = "0.1.0"
