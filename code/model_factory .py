"""
HDDM model factory for Bräutigam et al. (2024) Experiment 2 Stroop data.

Target environment: HDDM 1.0.1RC / PyMC 2.3.8.
"""

import hddm

VALID_MODELS = ("m_v", "m_a", "m_va")

REQUIRED_COLUMNS = ("subj_idx", "rt", "response", "reward", "congruency")


def _validate_data(data):
    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(
            "Missing required columns: %s. Available columns: %s"
            % (missing, list(data.columns))
        )

    if data[list(REQUIRED_COLUMNS)].isnull().any().any():
        bad = data[list(REQUIRED_COLUMNS)].isnull().sum()
        bad = bad[bad > 0]
        raise ValueError("Missing values in required HDDM columns:\n%s" % bad)

    responses = set(data["response"].unique().tolist())
    if not responses.issubset({0, 1}):
        raise ValueError(
            "response must contain only 0/1 for accuracy coding; found %s"
            % sorted(responses)
        )


def build_model(data, model_name, p_outlier=0.05):
    """
    Build one of the three candidate hierarchical HDDM models.

    m_v:
        v depends on reward x congruency
        a and t are estimated but condition-invariant

    m_a:
        v depends on congruency
        a depends on reward
        t is estimated but condition-invariant

    m_va:
        v depends on reward x congruency
        a depends on reward
        t is estimated but condition-invariant

    z is fixed at HDDM's default 0.5 by not including z.
    """
    _validate_data(data)

    if model_name not in VALID_MODELS:
        raise ValueError(
            "Unknown model %r. Choose one of: %s"
            % (model_name, ", ".join(VALID_MODELS))
        )

    if model_name == "m_v":
        depends_on = {
            "v": ["reward", "congruency"],
        }

    elif model_name == "m_a":
        depends_on = {
            "v": "congruency",
            "a": "reward",
        }

    else:  # m_va
        depends_on = {
            "v": ["reward", "congruency"],
            "a": "reward",
        }

    print("Model:", model_name)
    print("Data columns:", list(data.columns))
    print("depends_on:", depends_on)
    print("include: ['v', 'a', 't']")
    print("z: fixed at HDDM default 0.5")
    print("p_outlier:", p_outlier)

    model = hddm.HDDM(
        data,
        include=["v", "a", "t"],
        depends_on=depends_on,
        p_outlier=p_outlier,
        bias=False,
    )
    return model
