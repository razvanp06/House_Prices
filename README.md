## What I did

- **Target:** sale prices are skewed (most houses are cheap, a few are very expensive), so the model predicts `log(1 + SalePrice)`. This also matches the competition metric, which is RMSE on the log price.
- **Outliers:** removed two very large houses sold for unusually low prices (unfinished "Partial" sales) that broke the size–price trend.
- **Missing values**, handled by what they mean:
  - "Feature doesn't exist" (no pool, no garage, no basement) → `"None"` for categories, 0 for areas and counts
  - Garage year for houses without a garage → year the house was built
  - Truly unknown values → most common value, or median lot frontage per neighborhood, learned from train only
- **Encoding:**
  - Ordered categories (quality and condition ratings) → numbers that keep the order
  - Unordered categories (neighborhood, building type, …) → one-hot encoding
- **Model:** Ridge regression with standardized features, evaluated with 5-fold cross-validation

## Next steps

- Log-transform skewed numeric features (lot area, living area, …) to limit extreme predictions
- Tune Ridge's `alpha` with cross-validation
- Try gradient boosting (XGBoost / LightGBM) and blend it with Ridge
- Move all preprocessing into a scikit-learn Pipeline so every step is refit inside each CV fold

## Running it

Download `train.csv` and `test.csv` from the competition page into a `data/` folder, then run the notebook. Requires pandas, numpy, scikit-learn and matplotlib.
