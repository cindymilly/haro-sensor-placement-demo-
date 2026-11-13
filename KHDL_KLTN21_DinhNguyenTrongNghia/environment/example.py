"""Quick demonstration of the 01_environment module."""

import matplotlib.pyplot as plt

from .scenario import create_scenario


SOURCES = [
    {
        "x": 12,
        "y": 15,
        "source_type": "traffic",
        "emission_rate": 1.0,
        "name": "traffic_01",
    },
    {
        "x": 35,
        "y": 38,
        "source_type": "industrial",
        "emission_rate": 2.0,
        "name": "industrial_01",
    },
    {
        "x": 25,
        "y": 10,
        "source_type": "construction",
        "emission_rate": 1.5,
        "name": "construction_01",
    },
]


def main():
    result = create_scenario(
        grid_size=50,
        source_definitions=SOURCES,
        sigma=8.0,
        scenario_name="demo_50x50",
        background_noise_std=0.02,
        random_seed=120,
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(
        result.concentration,
        origin="lower",
        extent=(
            result.grid_x.min(),
            result.grid_x.max(),
            result.grid_y.min(),
            result.grid_y.max(),
        ),
        aspect="equal",
    )

    for source in result.sources:
        plt.scatter(source.x, source.y, marker="x", s=80)
        plt.text(source.x + 0.5, source.y + 0.5, source.name or source.source_type.value)

    plt.colorbar(label="Simulated concentration")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(result.metadata["scenario_name"])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
