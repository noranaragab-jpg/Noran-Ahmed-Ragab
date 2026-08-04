from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import csv
import math


class DatabaseError(ValueError):
    """Raised when the database or requested point is outside the valid domain."""


@dataclass(frozen=True)
class NumericalResult:
    lower: float
    average: float
    upper: float


@dataclass(frozen=True)
class ClassicalResult:
    terzaghi: float | None
    terzaghi_denominator: float
    bjerrum: float
    nc: float


def load_csv_database(path: str | Path) -> list[dict[str, float]]:
    """
    Load only the inputs and Optum lower/upper bounds.

    Terzaghi and Bjerrum & Eide values stored in the workbook are intentionally
    ignored. They are recalculated directly from their equations for every input.
    """
    required_columns = (
        "H", "B", "q", "R_int", "gamma", "Su", "sf_lower", "sf_upper"
    )
    records: list[dict[str, float]] = []

    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in required_columns if name not in (reader.fieldnames or [])]
        if missing:
            raise DatabaseError("Missing CSV columns: " + ", ".join(missing))

        for row_number, row in enumerate(reader, start=2):
            try:
                records.append({name: float(row[name]) for name in required_columns})
            except (TypeError, ValueError) as exc:
                raise DatabaseError(
                    f"Invalid numeric value in database row {row_number}."
                ) from exc

    return records


class BasalHeaveEngine:
    """
    Piecewise-linear interpolation of Optum lower and upper bounds only.

    Classical methods are calculated analytically and never interpolated.

    Numerical database domain:
      q = 0, 10, 20 kPa
      Su = 5, 10, 20, 30, 40, 50, 60 kPa
      gamma = 16, 18, 20 kN/m3
      H = 6, 8, 10, 12, 14 m
      B = 4 to 30 m at 2 m intervals

    The only absent q-Su node is q=20 kPa, Su=5 kPa. The valid continuous
    q-Su domain is therefore:
      Su >= 5 kPa for q <= 10 kPa
      Su >= q/2 for 10 < q <= 20 kPa
    """

    def __init__(self, records: Iterable[dict[str, float]]):
        self.records = list(records)
        if not self.records:
            raise DatabaseError("The numerical database is empty.")

        self.axes = {
            name: sorted({float(row[name]) for row in self.records})
            for name in ("q", "Su", "gamma", "H", "B")
        }
        self.r_int_values = sorted({float(row["R_int"]) for row in self.records})

        self._values: dict[
            tuple[float, float, float, float, float], tuple[float, float]
        ] = {}

        for row in self.records:
            key = (
                float(row["q"]),
                float(row["Su"]),
                float(row["gamma"]),
                float(row["H"]),
                float(row["B"]),
            )
            if key in self._values:
                raise DatabaseError(f"Duplicate database point: {key}")

            lower = float(row["sf_lower"])
            upper = float(row["sf_upper"])
            if lower > upper:
                raise DatabaseError(f"Lower bound exceeds upper bound at {key}.")
            self._values[key] = (lower, upper)

        self._validate_database_shape()

    def _validate_database_shape(self) -> None:
        expected_axes = {
            "q": [0.0, 10.0, 20.0],
            "Su": [5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "gamma": [16.0, 18.0, 20.0],
            "H": [6.0, 8.0, 10.0, 12.0, 14.0],
            "B": [
                4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0,
                18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0,
            ],
        }

        for name, expected in expected_axes.items():
            if self.axes[name] != expected:
                raise DatabaseError(
                    f"Unexpected {name} axis. Found {self.axes[name]}, expected {expected}."
                )

        if self.r_int_values != [1.0]:
            raise DatabaseError(
                f"Expected R_int=1.0 only; found {self.r_int_values}."
            )

        if len(self.records) != 4200:
            raise DatabaseError(
                f"Expected 4200 Optum database rows, found {len(self.records)}."
            )

        for q in self.axes["q"]:
            for su in self.axes["Su"]:
                pair_should_exist = not (q == 20.0 and su == 5.0)
                for gamma in self.axes["gamma"]:
                    for H in self.axes["H"]:
                        for B in self.axes["B"]:
                            exists = (q, su, gamma, H, B) in self._values
                            if exists != pair_should_exist:
                                raise DatabaseError(
                                    "Unexpected missing or extra Optum grid point at "
                                    f"q={q}, Su={su}, gamma={gamma}, H={H}, B={B}."
                                )

    @staticmethod
    def _bracket(axis: list[float], value: float) -> tuple[float, float, float]:
        tolerance = 1e-10
        if value < axis[0] - tolerance or value > axis[-1] + tolerance:
            raise DatabaseError(
                f"Value {value:g} is outside [{axis[0]:g}, {axis[-1]:g}]."
            )

        for node in axis:
            if abs(value - node) <= tolerance:
                return node, node, 0.0

        for lower, upper in zip(axis[:-1], axis[1:]):
            if lower < value < upper:
                return lower, upper, (value - lower) / (upper - lower)

        return axis[-1], axis[-1], 0.0

    @staticmethod
    def minimum_su_for_q(q: float) -> float:
        return 5.0 if q <= 10.0 else q / 2.0

    def validate_inputs(
        self,
        *,
        Su: float,
        gamma: float,
        q: float,
        H: float,
        B: float,
    ) -> None:
        for name, value in (
            ("Su", Su), ("gamma", gamma), ("q", q), ("H", H), ("B", B)
        ):
            if not math.isfinite(value):
                raise DatabaseError(f"{name} must be a finite number.")

        self._bracket(self.axes["q"], q)
        self._bracket(self.axes["Su"], Su)
        self._bracket(self.axes["gamma"], gamma)
        self._bracket(self.axes["H"], H)
        self._bracket(self.axes["B"], B)

        minimum_su = self.minimum_su_for_q(q)
        if Su < minimum_su - 1e-10:
            raise DatabaseError(
                f"The Optum database does not cover q={q:g} kPa with Su={Su:g} kPa. "
                f"For q={q:g} kPa, Su must be at least {minimum_su:g} kPa."
            )

    @staticmethod
    def _mix_pair(
        a: tuple[float, float],
        b: tuple[float, float],
        t: float,
    ) -> tuple[float, float]:
        return (
            a[0] * (1.0 - t) + b[0] * t,
            a[1] * (1.0 - t) + b[1] * t,
        )

    def _trilinear(
        self,
        *,
        q_node: float,
        su_node: float,
        gamma: float,
        H: float,
        B: float,
    ) -> tuple[float, float]:
        g0, g1, tg = self._bracket(self.axes["gamma"], gamma)
        h0, h1, th = self._bracket(self.axes["H"], H)
        b0, b1, tb = self._bracket(self.axes["B"], B)

        result = [0.0, 0.0]

        for use_g1 in (0, 1):
            if g0 == g1 and use_g1:
                continue
            gn = g1 if use_g1 else g0
            wg = 1.0 if g0 == g1 else (tg if use_g1 else 1.0 - tg)

            for use_h1 in (0, 1):
                if h0 == h1 and use_h1:
                    continue
                hn = h1 if use_h1 else h0
                wh = 1.0 if h0 == h1 else (th if use_h1 else 1.0 - th)

                for use_b1 in (0, 1):
                    if b0 == b1 and use_b1:
                        continue
                    bn = b1 if use_b1 else b0
                    wb = 1.0 if b0 == b1 else (tb if use_b1 else 1.0 - tb)

                    weight = wg * wh * wb
                    lower, upper = self._values[(q_node, su_node, gn, hn, bn)]
                    result[0] += weight * lower
                    result[1] += weight * upper

        return result[0], result[1]

    def interpolate_optum(
        self,
        *,
        Su: float,
        gamma: float,
        q: float,
        H: float,
        B: float,
    ) -> NumericalResult:
        """Interpolate only the Optum lower and upper bounds."""
        self.validate_inputs(Su=Su, gamma=gamma, q=q, H=H, B=B)

        def node(q_node: float, su_node: float) -> tuple[float, float]:
            return self._trilinear(
                q_node=q_node,
                su_node=su_node,
                gamma=gamma,
                H=H,
                B=B,
            )

        # Complete rectangular q-Su domain for Su >= 10 kPa.
        if Su >= 10.0 - 1e-10:
            q0, q1, tq = self._bracket(self.axes["q"], q)
            su_axis = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
            s0, s1, ts = self._bracket(su_axis, Su)

            at_q0 = node(q0, s0) if s0 == s1 else self._mix_pair(
                node(q0, s0), node(q0, s1), ts
            )

            if q0 == q1:
                pair = at_q0
            else:
                at_q1 = node(q1, s0) if s0 == s1 else self._mix_pair(
                    node(q1, s0), node(q1, s1), ts
                )
                pair = self._mix_pair(at_q0, at_q1, tq)

        # Complete rectangle q=0–10 kPa and Su=5–10 kPa.
        elif q <= 10.0 + 1e-10:
            tq = q / 10.0
            ts = (Su - 5.0) / 5.0
            at_q0 = self._mix_pair(node(0.0, 5.0), node(0.0, 10.0), ts)
            at_q10 = self._mix_pair(node(10.0, 5.0), node(10.0, 10.0), ts)
            pair = self._mix_pair(at_q0, at_q10, tq)

        # Triangular cell bounded by (q,Su)=(10,5),(10,10),(20,10).
        else:
            u = (q - 10.0) / 10.0
            v = (Su - 5.0) / 5.0
            weight_a = 1.0 - v
            weight_b = v - u
            weight_c = u

            if min(weight_a, weight_b, weight_c) < -1e-9:
                raise DatabaseError(
                    "The requested q-Su point lies outside the Optum data-supported triangle."
                )

            a = node(10.0, 5.0)
            b = node(10.0, 10.0)
            c = node(20.0, 10.0)
            pair = (
                weight_a * a[0] + weight_b * b[0] + weight_c * c[0],
                weight_a * a[1] + weight_b * b[1] + weight_c * c[1],
            )

        lower, upper = pair
        return NumericalResult(
            lower=lower,
            average=(lower + upper) / 2.0,
            upper=upper,
        )

    # Backward-compatible alias for earlier code/tests.
    interpolate = interpolate_optum

    @staticmethod
    def calculate_classical(
        *,
        Su: float,
        gamma: float,
        q: float,
        H: float,
        B: float,
    ) -> ClassicalResult:
        """
        Calculate Terzaghi and Bjerrum & Eide directly from equations.

        Assumptions for Terzaghi:
          homogeneous clay: C_ub = C_uh = Su
          deep clay/no nearby hard stratum: B1 = B/sqrt(2)
          surcharge q is included in the driving pressure

        Terzaghi:
          FS = 5.7 Su / [gamma H + q - Su H / B1]

        Bjerrum & Eide:
          FS = Su Nc / (gamma H + q)
          Nc = 7.2, if H/B > 3
          Nc = 5.14 + 1.05(H/B) - 0.18(H/B)^2, if H/B <= 3
        """
        if Su <= 0.0 or gamma <= 0.0 or H <= 0.0 or B <= 0.0 or q < 0.0:
            raise DatabaseError(
                "Su, gamma, H and B must be positive, and q must be non-negative."
            )

        B1 = B / math.sqrt(2.0)
        terzaghi_denominator = gamma * H + q - Su * H / B1
        terzaghi = (
            None
            if terzaghi_denominator <= 0.0
            else 5.7 * Su / terzaghi_denominator
        )

        ratio = H / B
        nc = (
            7.2
            if ratio > 3.0
            else 5.14 + 1.05 * ratio - 0.18 * ratio**2
        )
        bjerrum = Su * nc / (gamma * H + q)

        return ClassicalResult(
            terzaghi=terzaghi,
            terzaghi_denominator=terzaghi_denominator,
            bjerrum=bjerrum,
            nc=nc,
        )

    def calculate(
        self,
        *,
        Su: float,
        gamma: float,
        q: float,
        H: float,
        B: float,
        recommendation_basis: str = "average",
    ) -> dict[str, float | str | None]:
        # ONLY these Optum outputs come from database interpolation.
        numerical = self.interpolate_optum(Su=Su, gamma=gamma, q=q, H=H, B=B)

        # Classical methods are always calculated from equations.
        classical = self.calculate_classical(Su=Su, gamma=gamma, q=q, H=H, B=B)

        basis = recommendation_basis.lower().strip()
        if basis == "lower":
            recommended = numerical.lower
        elif basis == "average":
            recommended = numerical.average
        elif basis == "upper":
            recommended = numerical.upper
        else:
            raise DatabaseError(
                "recommendation_basis must be lower, average, or upper."
            )

        gap = (
            (numerical.upper - numerical.lower) / numerical.average * 100.0
            if numerical.average != 0.0
            else math.nan
        )

        return {
            "Su": Su,
            "gamma": gamma,
            "q": q,
            "H": H,
            "B": B,
            "H_over_B": H / B,
            "B_over_H": B / H,
            "B1": B / math.sqrt(2.0),
            "driving_pressure": gamma * H + q,
            "terzaghi": classical.terzaghi,
            "terzaghi_denominator": classical.terzaghi_denominator,
            "Nc": classical.nc,
            "bjerrum": classical.bjerrum,
            "numerical_lower": numerical.lower,
            "numerical_average": numerical.average,
            "numerical_upper": numerical.upper,
            "recommended": recommended,
            "recommendation_basis": basis,
            "bound_gap_percent": gap,
        }

    def nearest_cases(
        self,
        *,
        Su: float,
        gamma: float,
        q: float,
        H: float,
        B: float,
        count: int = 6,
    ) -> list[dict[str, float]]:
        """Return nearest original Optum models; classical columns are not used."""
        ranges = {
            name: self.axes[name][-1] - self.axes[name][0]
            for name in ("Su", "gamma", "q", "H", "B")
        }
        target = {"Su": Su, "gamma": gamma, "q": q, "H": H, "B": B}

        ranked: list[dict[str, float]] = []
        for row in self.records:
            distance2 = 0.0
            for name, target_value in target.items():
                scale = ranges[name] or 1.0
                distance2 += ((row[name] - target_value) / scale) ** 2

            ranked.append({
                "Su": row["Su"],
                "gamma": row["gamma"],
                "q": row["q"],
                "H": row["H"],
                "B": row["B"],
                "sf_lower": row["sf_lower"],
                "sf_upper": row["sf_upper"],
                "distance": math.sqrt(distance2),
            })

        ranked.sort(key=lambda item: item["distance"])
        return ranked[:count]
