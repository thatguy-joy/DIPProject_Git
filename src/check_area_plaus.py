from pathlib import Path
import csv
import math

DATA_DIR = Path(r"X:\DIP-Project\data")
IN_CSV = DATA_DIR / "lesion_measurements.csv"
OUT_CSV = DATA_DIR / "lesion_measurements_plausibility.csv"


def classify_ratio(
    r: float,
    plausible_min: float = 0.4,
    plausible_max: float = 2.0,
    hard_min: float = 0.05,
    hard_max: float = 10.0,
) -> str:

    if not math.isfinite(r):
        return "invalid"

    if r < hard_min or r > hard_max:
        return "invalid"

    if plausible_min <= r <= plausible_max:
        return "plausible"

    if r < plausible_min:
        return "suspicious_small"

    return "suspicious_large"


def main():
    print(f"Input : {IN_CSV}")
    print(f"Output: {OUT_CSV}")

    if not IN_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {IN_CSV}")

    with IN_CSV.open("r", newline="") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames_in = reader.fieldnames

        if fieldnames_in is None:
            raise RuntimeError("Input CSV has no header / fieldnames")

        required = {"measured_length_cm", "lesion_area_cm2"}
        missing = required - set(fieldnames_in)
        if missing:
            raise RuntimeError(f"Missing required columns in CSV: {missing}")

        extra_fields = ["area_disc_cm2", "ratio_auto_disc", "area_plausibility"]
        fieldnames_out = fieldnames_in + extra_fields

        with OUT_CSV.open("w", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames_out)
            writer.writeheader()

            counts = {
                "plausible": 0,
                "suspicious_small": 0,
                "suspicious_large": 0,
                "invalid": 0,
                "total": 0,
            }

            for row in reader:
                counts["total"] += 1

                try:
                    L = float(row["measured_length_cm"])
                    A_auto = float(row["lesion_area_cm2"])
                except (ValueError, TypeError):
                    L = float("nan")
                    A_auto = float("nan")

                if not math.isfinite(L) or L <= 0:
                    area_disc = float("nan")
                    ratio = float("nan")
                    label = "invalid"
                else:
                    area_disc = math.pi * (L / 2.0) ** 2

                    if not math.isfinite(A_auto) or A_auto <= 0 or area_disc <= 0:
                        ratio = float("nan")
                        label = "invalid"
                    else:
                        ratio = A_auto / area_disc
                        label = classify_ratio(ratio)

                if label in counts:
                    counts[label] += 1
                else:
                    counts["invalid"] += 1

                row["area_disc_cm2"] = (
                    f"{area_disc:.4f}" if math.isfinite(area_disc) else ""
                )
                row["ratio_auto_disc"] = f"{ratio:.4f}" if math.isfinite(ratio) else ""
                row["area_plausibility"] = label

                writer.writerow(row)

    print("\nSummary:")
    for k, v in counts.items():
        print(f"  {k:16s}: {v}")


if __name__ == "__main__":
    main()
