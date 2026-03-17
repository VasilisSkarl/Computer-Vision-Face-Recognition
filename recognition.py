import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Φόρτωση dataset
df = pd.read_csv("dataset.csv")

# Διαχωρισμός χαρακτηριστικών και ετικετών
X = df.drop(columns=['ταυτότητα']).astype(float)
y = LabelEncoder().fit_transform(df['ταυτότητα'])
y_cat = to_categorical(y)

# Κανονικοποίηση χαρακτηριστικών
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Διαχωρισμός σε train/test
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y_cat, test_size=0.25, stratify=y, random_state=42)

# Κατασκευή νευρωνικού δικτύου
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')  # 3 κατηγορίες
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Εκπαίδευση
model.fit(x_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=1)

# Πρόβλεψη και αξιολόγηση
y_pred_probs = model.predict(x_test)
y_pred = y_pred_probs.argmax(axis=1)
y_true = y_test.argmax(axis=1)

print("Ακρίβεια:", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))