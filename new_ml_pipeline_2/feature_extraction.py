def source_normalise_features(feature_cols, df_out):
    cols_to_normalise = [c for c in feature_cols if c not in _NORMALISE_EXCLUDE]
    df_out[cols_to_normalise] = df_out[cols_to_normalise].astype(float)
    # Existing for loop over sources will be here
    for source in sources:
        # process data from source
        pass
