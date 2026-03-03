#######################################################################
#                                                                     #
#                    Diabetes Dataset Analysis                        #
#                                                                     #
# Authors: Augustin Chavanes & Corentin Salvi                         #
#######################################################################

#region Importation des bibliothèques necessaires
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.preprocessing import StandardScaler
import warnings
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
#endregion

# Chargement des données
diabetes = pd.read_csv('./train.csv', na_values=['?'])

# Création des dossiers de sortie
output_dir = './DataTreatment'
output_dir_after = './DataAfterTreatment'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir_after, exist_ok=True)

# ==================================================
# ÉTAPE 0: NETTOYAGE ET ETAT BRUT
# ==================================================

# Suppression de l'ID (aucune valeur prédictive)
if 'id' in diabetes.columns:
    diabetes = diabetes.drop(columns=['id'])

# Suppression des colonnes non souhaitées
columns_to_drop = ['education_level', 'income_level']
diabetes = diabetes.drop(columns=columns_to_drop, errors='ignore')
diabetes['age_bmi'] = diabetes['age'] * diabetes['bmi']
# --- SAUVEGARDE DES DISTRIBUTIONS BRUTES (DataTreatment) ---
print(f"[OK] Génération des distributions BRUTES dans {output_dir}...")
raw_numeric_cols = diabetes.select_dtypes(include=['int64', 'float64']).columns
for column in raw_numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(diabetes[column], kde=True, color='orange')
    plt.title(f'Distribution BRUTE de {column}')
    plt.savefig(os.path.join(output_dir, f'{column}_brut.png'))
    plt.close()

# ==================================================
# ÉTAPE 1: ENCODAGE DES VARIABLES CATÉGORIELLES
# ==================================================

categorical_columns = ['gender', 'ethnicity', 'smoking_status', 'employment_status']
cols_to_encode = [c for c in categorical_columns if c in diabetes.columns]

diabetes = pd.get_dummies(diabetes, columns=cols_to_encode, drop_first=False)
print(f"[OK] One-Hot Encoding appliqué.")

# ==================================================
# ÉTAPE 2: Z-SCORE (STANDARDIZATION)
# ==================================================

target_col = 'diagnosed_diabetes'
numeric_cols = diabetes.select_dtypes(include=['int64', 'float64']).columns.tolist()

if target_col in numeric_cols:
    numeric_cols.remove(target_col)

scaler = StandardScaler()
diabetes[numeric_cols] = scaler.fit_transform(diabetes[numeric_cols])
print(f"[OK] Z-Score appliqué.")

# ==================================================
# ÉTAPE 3: VISUALISATION APRÈS TRAITEMENT (DataAfterTreatment)
# ==================================================

print(f"[OK] Génération des distributions TRAITÉES dans {output_dir_after}...")
for column in numeric_cols:
    plt.figure(figsize=(8, 4))
    diabetes[column].hist(bins=30, edgecolor='black', color='skyblue')
    plt.title(f'Distribution de {column} (après Z-Score)')
    plt.savefig(os.path.join(output_dir_after, f'{column}_standardise.png'))
    plt.close()


# ==================================================
# ÉTAPE 4: ANALYSE DE CORRÉLATION
# ==================================================

if target_col in diabetes.columns:
    # Calcul de la corrélation
    correlations = diabetes.corr()[target_col].sort_values(ascending=False)
    
    # Affichage console
    print("\n" + "="*50)
    print("TOP 10 DES VARIABLES LIÉES POSITIVEMENT :")
    print(correlations.head(11)) 
    # --- Génération de la Heatmap ---
    plt.figure(figsize=(20, 15))
    
    # Masque pour masquer la partie redondante (triangle supérieur)
    mask = np.triu(np.ones_like(diabetes.corr(), dtype=bool))
    
    sns.heatmap(diabetes.corr(), 
                mask=mask, 
                cmap='coolwarm', 
                center=0, 
                linewidths=0.1,
                annot=False) # Désactivé pour la clarté sur 35 colonnes
    
    plt.title(f'Matrice de Corrélation Finale', fontsize=16)
    
    # Sauvegarde sous le nom demandé
    plt.savefig(os.path.join("./", 'correlation_matrix.png'), dpi=300, bbox_inches='tight')

print(f"\n[OK] Matrice de corrélation enregistrée sous : correlation_matrix.png")

# ==================================================
# ÉTAPE 5: PRÉPARATION POUR LE RÉSEAU DE NEURONES
# ==================================================

# X = Caractéristiques, y = Cible
X = diabetes.drop(columns=[target_col])
y = diabetes[target_col]

# Séparation Entraînement / Validation (pour surveiller l'apprentissage)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==================================================
# ÉTAPE 6: CONSTRUCTION DU RÉSEAU DE NEURONES (MLP)
# ==================================================

model = Sequential([
    # Couche d'entrée (nombre de neurones = nombre de colonnes dans X)
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2), # Évite le surapprentissage (overfitting)
    
    # Couches cachées
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    
    # Couche de sortie (1 neurone car classification binaire 0/1)
    Dense(1, activation='sigmoid') 
])

# Compilation
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement
print("\n[INFO] Début de l'entraînement du réseau de neurones...")
history = model.fit(
    X_train, y_train, 
    epochs=20, 
    batch_size=128, 
    validation_data=(X_val, y_val),
    verbose=1
)

# ==================================================
# ÉTAPE 7: APPLICATION SUR LE FICHIER TEST.CSV
# ==================================================

print("\n[INFO] Chargement et traitement du fichier de test...")
test_df = pd.read_csv('./test.csv', na_values=['?'])
test_df['age_bmi'] = test_df['age'] * test_df['bmi']
# /!\ IMPORTANT : Appliquer EXACTEMENT le même traitement qu'au train.csv /!\
# 1. Suppression ID et colonnes inutiles
test_id = test_df['id'] if 'id' in test_df.columns else None # On garde l'ID pour le résultat final
test_df = test_df.drop(columns=['id', 'education_level', 'income_level'], errors='ignore')

# 2. One-Hot Encoding
test_df = pd.get_dummies(test_df, columns=cols_to_encode, drop_first=False)

# 3. Z-Score (on utilise le scaler déjà entraîné sur le train pour ne pas biaiser)
# On s'assure que les colonnes sont dans le même ordre
test_df = test_df[X.columns] 
test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

# 4. Prédiction
predictions_prob = model.predict(test_df)
predictions_final = (predictions_prob > 0.5).astype(int) # Seuil à 0.5 pour classer 0 ou 1

# Sauvegarde des résultats
results = pd.DataFrame({
    'id': test_id,
    'probability': predictions_prob.flatten(),
    'prediction': predictions_final.flatten()
})
results.to_csv(os.path.join("./", 'predictions_test.csv'), index=False)

print(f"\n[OK] Prédictions terminées ! Résultats sauvegardés dans {output_dir_after}/predictions_test.csv")

# ==================================================
# ÉTAPE 8: VISUALISATION DES PERFORMANCES DU RÉSEAU
# ==================================================

# 1. Graphique de l'apprentissage (Loss et Accuracy)
plt.figure(figsize=(12, 5))

# Courbe de la perte (Loss)
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Val Loss', color='red', linestyle='--')
plt.title('Évolution de la Perte (Loss)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Courbe de la précision (Accuracy)
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='green')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange', linestyle='--')
plt.title('Évolution de la Précision (Accuracy)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join("./", 'neural_network_learning_curves.png'))




# 2. Distribution des probabilités prédites sur le fichier TEST
plt.figure(figsize=(8, 5))
sns.histplot(results['probability'], bins=50, kde=True, color='purple')
plt.axvline(x=0.5, color='red', linestyle='--', label='Seuil de décision (0.5)')
plt.title('Distribution des probabilités de diabète (Fichier Test)')
plt.xlabel('Probabilité prédite (0 = Sain, 1 = Diabétique)')
plt.ylabel('Nombre de patients')
plt.legend()
plt.savefig(os.path.join("./", 'test_predictions_distribution.png'))

# 3. Répartition finale (Sains vs Diabétiques prédits)
plt.figure(figsize=(6, 6))
results['prediction'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'salmon'], labels=['Sain (0)', 'Diabétique (1)'])
plt.title('Proportion des prédictions sur le fichier Test')
plt.ylabel('')
plt.savefig(os.path.join("./", 'test_predictions_pie.png'))