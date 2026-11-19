"""Peer-group scoring: metrics, risk adjustment, outlier rule, edge cases."""

import numpy as np
import pandas as pd
import pytest

from careagent.scoring.peer import (
    CMS_COLUMNS,
    STATUS_INSUFFICIENT_DATA,
    STATUS_INSUFFICIENT_PEERS,
    STATUS_OUTLIER,
    STATUS_TYPICAL,
    _modified_z,
    prepare,
    score_providers,
)
from tests.conftest import (
    MILD_HIGH_NPI,
    MISSING_DATA_NPI,
    OUTLIER_CHARGE_NPI,
    OUTLIER_COST_NPI,
    SMALL_GROUP_NPI,
    TYPICAL_NPI,
    make_cms_frame,
)


@pytest.fixture
def scored(cms_frame):
    providers, groups = score_providers(cms_frame)
    return providers.set_index("npi"), groups.set_index("provider_type")


class TestPrepare:
    def test_metrics_are_ratios_of_cms_totals(self, cms_frame):
        df = prepare(cms_frame).set_index("npi")
        raw = cms_frame.set_index("Rndrng_NPI").loc[TYPICAL_NPI]
        row = df.loc[TYPICAL_NPI]
        assert row.cost_per_beneficiary == pytest.approx(
            float(raw.Tot_Mdcr_Stdzd_Amt) / float(raw.Tot_Benes))
        assert row.services_per_beneficiary == pytest.approx(
            float(raw.Tot_Srvcs) / float(raw.Tot_Benes))
        assert row.charge_to_allowed_ratio == pytest.approx(
            float(raw.Tot_Sbmtd_Chrg) / float(raw.Tot_Mdcr_Alowd_Amt))

    def test_missing_value_stays_missing_never_zero(self, cms_frame):
        df = prepare(cms_frame).set_index("npi")
        row = df.loc[MISSING_DATA_NPI]
        assert np.isnan(row.bene_avg_risk_score)
        assert not row.scorable
        assert row.data_completeness < 1.0

    @pytest.mark.parametrize("bad", ["0", "-5", "abc", ""])
    def test_zero_negative_or_garbage_denominator_gives_missing_metric(self, cms_frame, bad):
        cms_frame.loc[cms_frame.Rndrng_NPI == TYPICAL_NPI, "Tot_Benes"] = bad
        row = prepare(cms_frame).set_index("npi").loc[TYPICAL_NPI]
        assert np.isnan(row.cost_per_beneficiary)
        assert np.isnan(row.services_per_beneficiary)
        assert not row.scorable

    def test_missing_cms_column_is_a_clear_error(self, cms_frame):
        with pytest.raises(ValueError, match="Bene_Avg_Risk_Scre"):
            prepare(cms_frame.drop(columns=["Bene_Avg_Risk_Scre"]))

    def test_npi_whitespace_is_stripped(self, cms_frame):
        cms_frame.loc[0, "Rndrng_NPI"] = f" {cms_frame.loc[0, 'Rndrng_NPI']} "
        assert " " not in prepare(cms_frame).npi.iloc[0]


class TestModifiedZ:
    def test_matches_iglewicz_hoaglin_formula(self):
        values = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
        z, med, mad = _modified_z(values)
        assert med == 3.0 and mad == 1.0
        assert z.iloc[-1] == pytest.approx(0.6745 * 97)

    def test_zero_mad_gives_nan_so_nobody_is_flagged(self):
        z, _, mad = _modified_z(pd.Series([5.0] * 10 + [50.0]))
        assert mad == 0
        assert z.isna().all()


class TestStatuses:
    def test_planted_cost_outlier_is_flagged_with_reason(self, scored):
        providers, _ = scored
        row = providers.loc[OUTLIER_COST_NPI]
        assert row.billing_pattern_status == STATUS_OUTLIER
        assert row.cost_ratio_to_peer_median > 4
        assert "Medicare cost per beneficiary" in row.outlier_reasons
        assert "Cardiology median" in row.outlier_reasons

    def test_planted_charge_outlier_is_flagged_on_charge_only(self, scored):
        providers, _ = scored
        row = providers.loc[OUTLIER_CHARGE_NPI]
        assert row.billing_pattern_status == STATUS_OUTLIER
        assert "charge to allowed" in row.outlier_reasons.lower()
        assert "cost per beneficiary" not in row.outlier_reasons

    def test_ordinary_provider_is_typical(self, scored):
        providers, _ = scored
        assert providers.loc[TYPICAL_NPI].billing_pattern_status == STATUS_TYPICAL
        assert pd.isna(providers.loc[TYPICAL_NPI].outlier_reasons)

    def test_small_specialty_is_not_compared(self, scored):
        providers, groups = scored
        row = providers.loc[SMALL_GROUP_NPI]
        assert row.billing_pattern_status == STATUS_INSUFFICIENT_PEERS
        assert row.peer_group_size == 5
        assert np.isnan(row.cost_z)
        assert "Rare Specialty" not in groups.index

    def test_missing_data_is_its_own_status(self, scored):
        providers, _ = scored
        assert providers.loc[MISSING_DATA_NPI].billing_pattern_status == STATUS_INSUFFICIENT_DATA

    def test_every_provider_gets_exactly_one_status(self, scored, cms_frame):
        providers, _ = scored
        assert len(providers) == len(cms_frame)
        assert providers.billing_pattern_status.notna().all()

    def test_outliers_are_rare_on_ordinary_data(self, scored):
        providers, _ = scored
        compared = providers[providers.billing_pattern_status.isin([STATUS_TYPICAL, STATUS_OUTLIER])]
        assert (compared.billing_pattern_status == STATUS_OUTLIER).sum() == 2  # only the planted ones


class TestRule:
    def test_practical_significance_gate(self, cms_frame):
        """z alone is not enough: the value must also be >= min_ratio x the median."""
        loose = score_providers(cms_frame, z_threshold=2.5, min_ratio=1.0)[0].set_index("npi")
        gated = score_providers(cms_frame, z_threshold=2.5, min_ratio=2.0)[0].set_index("npi")
        row = loose.loc[MILD_HIGH_NPI]
        assert row.cost_z > 2.5 and row.cost_ratio_to_peer_median < 2.0  # precondition
        assert row.billing_pattern_status == STATUS_OUTLIER
        assert gated.loc[MILD_HIGH_NPI].billing_pattern_status == STATUS_TYPICAL

    def test_low_values_are_never_flagged(self, cms_frame):
        mask = cms_frame.Rndrng_NPI == TYPICAL_NPI
        cms_frame.loc[mask, "Tot_Mdcr_Stdzd_Amt"] = str(
            float(cms_frame.loc[mask, "Tot_Mdcr_Stdzd_Amt"].iloc[0]) / 20)
        row = score_providers(cms_frame)[0].set_index("npi").loc[TYPICAL_NPI]
        assert row.cost_z < -3.5
        assert row.billing_pattern_status == STATUS_TYPICAL

    def test_identical_risk_scores_skip_adjustment_without_warnings(self, cms_frame):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _, groups = score_providers(cms_frame, min_peers=5)
        assert groups.set_index("provider_type").loc["Rare Specialty", "cost_risk_slope"] == 0.0

    def test_min_peers_is_configurable(self, cms_frame):
        providers, _ = score_providers(cms_frame, min_peers=5)
        assert providers.set_index("npi").loc[SMALL_GROUP_NPI].billing_pattern_status != \
            STATUS_INSUFFICIENT_PEERS


class TestRiskAdjustment:
    def test_recovers_planted_risk_slope(self, scored):
        _, groups = scored
        # fixture: cost per beneficiary proportional to risk^1
        assert groups.loc["Cardiology", "cost_risk_slope"] == pytest.approx(1.0, abs=0.15)

    def test_sicker_patients_alone_do_not_make_an_outlier(self):
        df = make_cms_frame()
        cardio = df.Rndrng_Prvdr_Type == "Cardiology"
        top = df[cardio].Bene_Avg_Risk_Scre.replace("", np.nan).astype(float).idxmax()
        providers = score_providers(df)[0]
        # Highest-risk provider has high raw cost but typical risk-adjusted cost
        assert providers.loc[top, "cost_per_beneficiary"] > \
            providers.loc[cardio & providers.scorable, "cost_per_beneficiary"].median()
        assert abs(providers.loc[top, "cost_z"]) < 3.5

    def test_peer_groups_are_separate(self, scored):
        _, groups = scored
        # Family Practice is cheaper; it gets its own median, not Cardiology's
        assert groups.loc["Family Practice", "median_cost_per_beneficiary"] < \
            groups.loc["Cardiology", "median_cost_per_beneficiary"] * 0.6


class TestDeterminismAndRanges:
    def test_same_input_same_output(self, cms_frame):
        a = score_providers(cms_frame)[0]
        b = score_providers(cms_frame.copy())[0]
        pd.testing.assert_frame_equal(a, b)

    def test_percentiles_in_range(self, scored):
        providers, _ = scored
        for col in ["cost_percentile", "services_percentile", "charge_percentile",
                    "isolation_forest_percentile"]:
            values = providers[col].dropna()
            assert values.between(0, 100).all()

    def test_isolation_forest_ranks_planted_outliers_high(self, scored):
        providers, _ = scored
        assert providers.loc[OUTLIER_COST_NPI, "isolation_forest_percentile"] >= 95
        assert providers.loc[OUTLIER_CHARGE_NPI, "isolation_forest_percentile"] >= 95

    def test_empty_input(self):
        empty = pd.DataFrame(columns=list(CMS_COLUMNS))
        providers, groups = score_providers(empty)
        assert providers.empty and groups.empty
