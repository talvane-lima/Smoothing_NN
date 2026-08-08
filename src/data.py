import torch
import numpy as np
from sklearn.datasets import fetch_openml, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import Tuple


def get_bank_marketing_data(
    test_size: float = 0.20,
    val_size: float = 0.20,
    balance_method: str = "oversample",
    random_state: int = 42
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], int, np.ndarray]:
    """
    Loads and preprocesses the UCI Bank Marketing dataset (~45,211 rows) for Full-Batch Training.
    Applies balancing techniques (Random Oversampling / Class Weighting) to handle the 88/12 class imbalance.
    
    Args:
        test_size (float): Proportion of test split (default: 0.20).
        val_size (float): Proportion of validation split (default: 0.20).
        balance_method (str): 'oversample' (balances train set to 50/50), 'weights', or 'none'.
        random_state (int): Random seed.
        
    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), num_features, class_weights
    """
    print(f"Fetching Bank Marketing dataset (45,211 records)...")
    
    try:
        data = fetch_openml(name="bank-marketing", version=1, as_frame=True, parser="auto")
        df = data.frame
        
        target_col = data.target_names[0] if data.target_names else "Class"
        if target_col not in df.columns:
            target_col = df.columns[-1]
            
        X = df.drop(columns=[target_col])
        raw_y = df[target_col].astype(str).str.lower().str.strip()
        
        # Convert binary target to 0 (No Deposit) and 1 (Deposit Subscribed)
        unique_vals = set(raw_y.unique())
        if unique_vals == {"1", "2"}:
            # In OpenML bank-marketing: '1' is 'no' (0), '2' is 'yes' (1)
            y = raw_y.apply(lambda val: 1 if val == "2" else 0).values.astype(np.int64)
        elif unique_vals == {"0", "1"}:
            y = raw_y.apply(lambda val: 1 if val == "1" else 0).values.astype(np.int64)
        else:
            y = raw_y.apply(lambda val: 1 if val in {"yes", "true", "y", "success", "1"} else 0).values.astype(np.int64)
        
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, num_cols),
                ("cat", cat_pipeline, cat_cols)
            ]
        )
        X_processed = preprocessor.fit_transform(X)
        X_processed = np.nan_to_num(X_processed, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        print(f"Successfully loaded Bank Marketing from OpenML: {X_processed.shape[0]} samples.")

    except Exception as e:
        print(f"Warning: OpenML download unavailable ({e}). Generating realistic Bank Marketing synthetic replica (~45,000 samples)...")
        X_raw, y = make_classification(
            n_samples=45211,
            n_features=25,
            n_informative=15,
            n_redundant=5,
            weights=[0.88, 0.12],  # Realistic 88/12 bank deposit class imbalance
            n_classes=2,
            flip_y=0.03,
            random_state=random_state
        )
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X_raw).astype(np.float32)

    # Train / Val / Test Stratified Split (60% Train, 20% Val, 20% Test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    val_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative, random_state=random_state, stratify=y_train_val
    )

    # Compute class distribution before balancing
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    class_weights = np.array([1.0, neg_count / max(pos_count, 1)], dtype=np.float32)

    # Apply Balancing Technique on the Training Set only
    if balance_method == "oversample":
        idx_0 = np.where(y_train == 0)[0]
        idx_1 = np.where(y_train == 1)[0]
        n_maj = len(idx_0)
        rng = np.random.default_rng(random_state)
        idx_1_resampled = rng.choice(idx_1, size=n_maj, replace=True)
        balanced_idx = np.concatenate([idx_0, idx_1_resampled])
        rng.shuffle(balanced_idx)
        
        X_train = X_train[balanced_idx]
        y_train = y_train[balanced_idx]
        print(f"  [Balancing Strategy: Random Oversampling] -> Train set balanced to 50% No ({len(idx_0):,}) / 50% Yes ({n_maj:,}) [Total Train: {len(y_train):,}]")
    elif balance_method == "weights":
        print(f"  [Balancing Strategy: Class-Weighted Loss] -> Loss weights: Class 0 = 1.00, Class 1 = {class_weights[1]:.2f}")
    else:
        print(f"  [Balancing Strategy: None]")

    train_data = (torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_data = (torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_data = (torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    num_features = X_processed.shape[1]
    print(f"Dataset Loaded: Total={len(y):,} | Class 0 (No)={np.sum(y == 0):,} ({np.mean(y == 0)*100:.1f}%), Class 1 (Yes)={np.sum(y == 1):,} ({np.mean(y == 1)*100:.1f}%)")
    print(f"Splits ready: Train={len(y_train):,}, Val={len(y_val):,}, Test={len(y_test):,} | Features={num_features}")
    return train_data, val_data, test_data, num_features, class_weights
