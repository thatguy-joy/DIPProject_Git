#still under construction!!!!⚒️

from pathlib import Path
import csv
import pydicom

RAW_ROOT = Path("data/raw")
OUT_CSV = Path("data/dicom_tags.csv")

def main() -> None:
    if not RAW_ROOT.exists():
        print(f"[ERROR] RAW_ROOT does not exist: {RAW_ROOT}")
        return

    dcm_files = sorted(RAW_ROOT.rglob("*.dcm"))
    print(f"Found {len(dcm_files)} DICOM files under {RAW_ROOT}")

    if not dcm_files:
        return

    #Open CSV once -> write header
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "file_path",        
                "group_hex",
                "element_hex",
                "tag_str",
                "keyword",
                "vr",
                "value",
            ]
        )

        for path in dcm_files:
            print(f"\n[INFO] Processing {path}")
            try:
                #Read metadata
                ds = pydicom.dcmread(path, stop_before_pixels=True)
            except Exception as e:
                print(f"  [WARN] Could not read {path}: {e}")
                continue

            for elem in ds:
                try:
                    group_hex = f"{elem.tag.group:04X}"
                    element_hex = f"{elem.tag.element:04X}"
                    tag_str = f"({group_hex},{element_hex})"
                    keyword = getattr(elem, "keyword", "") or ""
                    vr = elem.VR or ""

                    try:
                        value_str = str(elem.value)
                    except Exception as ve:
                        value_str = f"<unprintable: {ve}>"

                    writer.writerow(
                        [
                            str(path),
                            group_hex,
                            element_hex,
                            tag_str,
                            keyword,
                            vr,
                            value_str,
                        ]
                    )
                except Exception as e:
                    print(f"  [WARN] Failed to export tag {elem} in {path}: {e}")

    print(f"\n DICOM tags written successfully! {OUT_CSV}")


if __name__ == "__main__":
    main()
