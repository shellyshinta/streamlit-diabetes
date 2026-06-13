import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve
)

from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline


# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Prediksi Diabetes KNN",
    layout="wide"
)

st.title("Aplikasi Prediksi Diabetes Menggunakan KNN")
st.write("Metode: Z-Score, SMOTE-ENN, KNN, 10-Fold Cross Validation")


# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("diabetes_prediction.csv")

st.subheader("1. Preview Dataset")
st.dataframe(df.head())

st.write("Jumlah data:", df.shape[0])
st.write("Jumlah kolom:", df.shape[1])


# =========================
# CEK TARGET CLASS
# =========================
if "Diagnosis" in df.columns:
    target_col = "Diagnosis"
elif "Outcome" in df.columns:
    target_col = "Outcome"
else:
    st.error("Kolom target tidak ditemukan. Pastikan ada kolom Diagnosis atau Outcome.")
    st.stop()


# =========================
# ENCODING DATA KATEGORIK
# =========================
df_processed = df.copy()

for col in df_processed.columns:
    if df_processed[col].dtype == "object":
        encoder = LabelEncoder()
        df_processed[col] = encoder.fit_transform(df_processed[col])


# =========================
# SPLIT FITUR DAN TARGET
# =========================
X = df_processed.drop(target_col, axis=1)
y = df_processed[target_col]

st.subheader("2. Distribusi Class Awal")
st.write(y.value_counts())

fig_class, ax_class = plt.subplots(figsize=(6, 4))
y.value_counts().plot(kind="bar", ax=ax_class)
ax_class.set_title("Distribusi Class Diabetes")
ax_class.set_xlabel("Class")
ax_class.set_ylabel("Jumlah Data")
st.pyplot(fig_class)


# =========================
# SPLIT TRAIN TEST
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

st.subheader("3. Pembagian Data Train dan Test")
st.write("Jumlah data training:", X_train.shape[0])
st.write("Jumlah data testing:", X_test.shape[0])


# =========================
# DISTRIBUSI SEBELUM DAN SESUDAH SMOTE-ENN
# =========================
st.subheader("4. Distribusi Kelas Sebelum dan Sesudah SMOTE-ENN")

smoteenn_preview = SMOTEENN(random_state=42)
X_resampled, y_resampled = smoteenn_preview.fit_resample(X_train, y_train)

col_a, col_b = st.columns(2)

with col_a:
    st.write("Sebelum SMOTE-ENN")
    st.write(pd.Series(y_train).value_counts())

    fig_before, ax_before = plt.subplots(figsize=(5, 4))
    pd.Series(y_train).value_counts().plot(kind="bar", ax=ax_before)
    ax_before.set_title("Sebelum SMOTE-ENN")
    ax_before.set_xlabel("Class")
    ax_before.set_ylabel("Jumlah Data")
    st.pyplot(fig_before)

with col_b:
    st.write("Sesudah SMOTE-ENN")
    st.write(pd.Series(y_resampled).value_counts())

    fig_after, ax_after = plt.subplots(figsize=(5, 4))
    pd.Series(y_resampled).value_counts().plot(kind="bar", ax=ax_after)
    ax_after.set_title("Sesudah SMOTE-ENN")
    ax_after.set_xlabel("Class")
    ax_after.set_ylabel("Jumlah Data")
    st.pyplot(fig_after)


# =========================
# PEMILIHAN K TERBAIK OTOMATIS
# =========================
st.subheader("5. Hasil 10-Fold Cross Validation")

k_values = [3, 5, 7, 11, 15, 22]

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "auc": "roc_auc"
}

results = []

for k in k_values:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("smoteenn", SMOTEENN(random_state=42)),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])

    cv_result = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring
    )

    results.append({
        "K": k,
        "Accuracy": cv_result["test_accuracy"].mean(),
        "Precision": cv_result["test_precision"].mean(),
        "Recall": cv_result["test_recall"].mean(),
        "F1-Score": cv_result["test_f1"].mean(),
        "AUC": cv_result["test_auc"].mean()
    })

results_df = pd.DataFrame(results)
best_k = int(results_df.loc[results_df["Accuracy"].idxmax(), "K"])

st.dataframe(results_df)
st.success(f"K terbaik otomatis berdasarkan Accuracy tertinggi adalah K = {best_k}")


# =========================
# GRAFIK PERBANDINGAN NILAI K
# =========================
st.subheader("6. Grafik Perbandingan Nilai K")

fig_k, ax_k = plt.subplots(figsize=(9, 5))
ax_k.plot(results_df["K"], results_df["Accuracy"], marker="o", label="Accuracy")
ax_k.plot(results_df["K"], results_df["Precision"], marker="o", label="Precision")
ax_k.plot(results_df["K"], results_df["Recall"], marker="o", label="Recall")
ax_k.plot(results_df["K"], results_df["F1-Score"], marker="o", label="F1-Score")
ax_k.plot(results_df["K"], results_df["AUC"], marker="o", label="AUC")
ax_k.set_xlabel("Nilai K")
ax_k.set_ylabel("Score")
ax_k.set_title("Perbandingan Metrik Berdasarkan Nilai K")
ax_k.legend()
ax_k.grid(True)
st.pyplot(fig_k)


# =========================
# MODEL FINAL
# =========================
final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("smoteenn", SMOTEENN(random_state=42)),
    ("knn", KNeighborsClassifier(n_neighbors=best_k))
])

final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)
y_proba = final_model.predict_proba(X_test)[:, 1]


# =========================
# EVALUASI AKHIR DATA TESTING
# =========================
st.subheader("7. Evaluasi Akhir Data Testing")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Accuracy", f"{accuracy:.3f}")
col2.metric("Precision", f"{precision:.3f}")
col3.metric("Recall", f"{recall:.3f}")
col4.metric("F1-Score", f"{f1:.3f}")
col5.metric("AUC", f"{auc:.3f}")

st.text("Classification Report")
st.text(classification_report(y_test, y_pred))


# =========================
# CONFUSION MATRIX
# =========================
st.subheader("8. Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negatif", "Positif"]
)
disp.plot(ax=ax_cm)
st.pyplot(fig_cm)


# =========================
# ROC CURVE
# =========================
st.subheader("9. ROC Curve dan AUC")

fpr, tpr, threshold = roc_curve(y_test, y_proba)

fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
ax_roc.plot([0, 1], [0, 1], linestyle="--")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curve")
ax_roc.legend()
ax_roc.grid(True)
st.pyplot(fig_roc)


# =========================
# SENSITIVITAS FITUR
# =========================
st.subheader("10. Sensitivitas Fitur")

baseline_acc = accuracy
sensitivity_results = []

for feature in X.columns:
    X_train_drop = X_train.drop(feature, axis=1)
    X_test_drop = X_test.drop(feature, axis=1)

    model_drop = Pipeline([
        ("scaler", StandardScaler()),
        ("smoteenn", SMOTEENN(random_state=42)),
        ("knn", KNeighborsClassifier(n_neighbors=best_k))
    ])

    model_drop.fit(X_train_drop, y_train)

    pred_drop = model_drop.predict(X_test_drop)
    acc_drop = accuracy_score(y_test, pred_drop)

    sensitivity_results.append({
        "Fitur Dihapus": feature,
        "Accuracy Setelah Dihapus": acc_drop,
        "Penurunan Accuracy": baseline_acc - acc_drop
    })

sensitivity_df = pd.DataFrame(sensitivity_results)
sensitivity_df = sensitivity_df.sort_values(
    by="Penurunan Accuracy",
    ascending=False
)

st.dataframe(sensitivity_df)

fig_sens, ax_sens = plt.subplots(figsize=(8, 5))
ax_sens.barh(
    sensitivity_df["Fitur Dihapus"],
    sensitivity_df["Penurunan Accuracy"]
)
ax_sens.set_title("Sensitivitas Fitur terhadap Accuracy")
ax_sens.set_xlabel("Penurunan Accuracy")
ax_sens.set_ylabel("Fitur")
st.pyplot(fig_sens)


# =========================
# HASIL PREDIKSI DATA TESTING
# =========================
st.subheader("11. Hasil Prediksi Data Testing")

df_hasil = X_test.copy()
df_hasil["Class Asli"] = y_test
df_hasil["Prediksi"] = y_pred
df_hasil["Probabilitas Diabetes"] = y_proba
df_hasil["Keterangan"] = df_hasil["Prediksi"].map({
    0: "Negatif Diabetes",
    1: "Positif Diabetes"
})

st.dataframe(df_hasil)