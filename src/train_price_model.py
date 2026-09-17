from pathlib import Path

from pricing_model import save_bundle, train_bundle


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "notebooks" / "dataset_prepared.csv"
ARTIFACT = ROOT / "artifacts" / "price_band_model.joblib"


if __name__ == "__main__":
    bundle = train_bundle(DATASET)
    save_bundle(bundle, ARTIFACT)
    print(f"Saved {ARTIFACT}")
    print(f"Features: {len(bundle.raw_feature_cols)}; trained through: {bundle.trained_through}")
