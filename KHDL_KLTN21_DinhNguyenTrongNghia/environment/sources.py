from __future__ import annotations

from collections.abc import Iterable, Sequence

from .types import EmissionSource, SourceType


class SourceFactory:
    """Factory helpers for creating standard source categories."""

    DEFAULT_RATES = {
        SourceType.TRAFFIC: 1.0,
        SourceType.INDUSTRIAL: 2.0,
        SourceType.CONSTRUCTION: 1.5,
        SourceType.NOISE: 0.25,
    }

    @classmethod
    def create(
        cls,
        x: float,
        y: float,
        source_type: SourceType | str,
        emission_rate: float | None = None,
        sigma: float | None = None,
        name: str | None = None,
        **metadata,
    ) -> EmissionSource:
        source_type = (
            source_type
            if isinstance(source_type, SourceType)
            else SourceType(str(source_type).lower())
        )

        if emission_rate is None:
            emission_rate = cls.DEFAULT_RATES[source_type]

        return EmissionSource(
            x=x,
            y=y,
            source_type=source_type,
            emission_rate=emission_rate,
            sigma=sigma,
            name=name,
            metadata=metadata,
        )


def create_sources(
    definitions: Sequence[dict] | Iterable[EmissionSource],
) -> list[EmissionSource]:
    """Create/validate a list of emission sources.

    A definition can contain:
        x, y, source_type, emission_rate, sigma, name, plus metadata.
    """
    result: list[EmissionSource] = []

    for item in definitions:
        if isinstance(item, EmissionSource):
            result.append(item)
            continue

        data = dict(item)
        known = {
            "x",
            "y",
            "source_type",
            "emission_rate",
            "sigma",
            "name",
        }
        metadata = data.pop("metadata", {})
        extra = {k: data.pop(k) for k in list(data) if k not in known}

        source = SourceFactory.create(
            metadata=metadata | extra,
            **data,
        )
        result.append(source)

    return result
