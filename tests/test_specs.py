"""Tests for static specification dataclasses."""
import pytest
from cpsych.core.specs import (
    StateVar, Param, ModuleSpec, EnvSpec,
    Temporality, Exposure,
)


class TestStateVar:
    def test_frozen(self):
        sv = StateVar("x", Temporality.CONTINUOUS, 0.0)
        with pytest.raises(AttributeError):
            sv.name = "y"

    def test_defaults(self):
        sv = StateVar("x", Temporality.EPISODIC)
        assert sv.initial == 0.0
        assert sv.doc == ""


class TestParam:
    def test_frozen(self):
        p = Param("alpha", 0.1, 0.0, 1.0)
        with pytest.raises(AttributeError):
            p.default = 0.5

    def test_defaults(self):
        p = Param("alpha", 0.1, 0.0, 1.0)
        assert p.exposure == Exposure.INTERNAL
        assert p.unit == ""

    def test_exposed(self):
        p = Param("alpha", 0.1, 0.0, 1.0, exposure=Exposure.EXPOSED)
        assert p.exposure == Exposure.EXPOSED


class TestModuleSpec:
    @pytest.fixture
    def spec(self):
        return ModuleSpec(
            name="test_module",
            states=(
                StateVar("x", Temporality.CONTINUOUS, 1.0),
                StateVar("y", Temporality.EPISODIC, 0.0),
            ),
            params=(
                Param("alpha", 0.1, 0.0, 1.0),
                Param("beta", 0.5, 0.0, 2.0, exposure=Exposure.EXPOSED),
            ),
            input_names=("gain",),
            output_names=("activity",),
            timescale=0.3,
        )

    def test_frozen(self, spec):
        with pytest.raises(AttributeError):
            spec.name = "other"

    def test_state_index(self, spec):
        assert spec.state_index("x") == 0
        assert spec.state_index("y") == 1

    def test_state_index_missing(self, spec):
        with pytest.raises(KeyError):
            spec.state_index("z")

    def test_param_index(self, spec):
        assert spec.param_index("alpha") == 0
        assert spec.param_index("beta") == 1

    def test_param_index_missing(self, spec):
        with pytest.raises(KeyError):
            spec.param_index("gamma")

    def test_counts(self, spec):
        assert len(spec.states) == 2
        assert len(spec.params) == 2
        assert len(spec.input_names) == 1
        assert len(spec.output_names) == 1


class TestEnvSpec:
    def test_frozen(self):
        es = EnvSpec("bandit", ("drift_diff",), ("r",))
        with pytest.raises(AttributeError):
            es.name = "other"
