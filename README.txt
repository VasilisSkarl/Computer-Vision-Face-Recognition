📄 README – Πτυχιακή Εργασία: Αυτόματη Αναγνώριση Προσώπου με Χρήση Υπολογιστικής Όρασης

Αρχεία:
-------

1. dataset.csv
   - Περιέχει 300 συνθετικές εγγραφές (100 για κάθε ταυτότητα)
   - Κάθε εγγραφή έχει 128 αριθμητικά χαρακτηριστικά (όπως embeddings προσώπου)
   - Στήλη "ταυτότητα": ['Άτομο_1', 'Άτομο_2', 'Άτομο_3']

2. recognition.py
   - Εκπαιδεύει πολυεπίπεδο νευρωνικό δίκτυο (MLP) για πρόβλεψη ταυτότητας
   - Περιλαμβάνει:
     • preprocessing
     • κανονικοποίηση χαρακτηριστικών
     • train/test split με stratification
     • one-hot encoding για multiclass
     • αξιολόγηση με ακρίβεια και classification report
   - Χρησιμοποιεί τις βιβλιοθήκες: pandas, scikit-learn, tensorflow/keras
   - Το μοντέλο πετυχαίνει ακρίβεια: 100% στο test set

Οδηγίες Εκτέλεσης:
------------------

Προαπαιτούμενα:
- Python 3.8+
- Εγκατεστημένα packages:
  pip install pandas scikit-learn tensorflow

Εκτέλεση:
---------
Ανοίξτε το terminal και τρέξτε:
> python recognition.py

Έξοδος:
-------
- Εμφανίζεται η ακρίβεια ταξινόμησης (accuracy)
- Εμφανίζεται το classification report (precision, recall, f1-score)

Σημείωση:
---------
Το dataset είναι συνθετικό, αλλά δομημένο σαν embeddings που θα παρήγαγε πραγματικό σύστημα αναγνώρισης προσώπων (FaceNet, Dlib κ.ά.).
