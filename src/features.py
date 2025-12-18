def add_features(df):
    df = df.copy()
    df["avg_grade"] = (df["G1"] + df["G2"]) / 2
    df["study_ratio"] = df["studytime"] / (df["absences"] + 1)
    return df
