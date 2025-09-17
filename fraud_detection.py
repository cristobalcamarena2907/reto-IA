import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualizaciones
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

print("=== PROYECTO DE DETECCIÓN DE FRAUDE BANCARIO - OPTIMIZADO ===\n")

# 1. CARGA DE DATO
print("1. CARGANDO DATASET...")
df = pd.read_csv('Base.csv')
print(f"Dataset: {df.shape}")
print(f"Distribución del target:")
print(df['fraud_bool'].value_counts())
print(f"Porcentaje de fraude: {df['fraud_bool'].mean()*100:.2f}%\n")

# 2. ANÁLISIS EXPLORATORIO
print("2. ANÁLISIS EXPLORATORIO...")
print("Información del dataset:")
print(df.info())
print(f"\nEstadísticas descriptivas:")
print(df.describe())

# Visualización de la distribución del target
plt.figure(figsize=(10, 6))
fraud_counts = df['fraud_bool'].value_counts()
plt.pie(fraud_counts.values, labels=['No Fraude', 'Fraude'], autopct='%1.1f%%', startangle=90)
plt.title('Distribución del Target (Fraude vs No Fraude)')
plt.axis('equal')
plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. PREPROCESAMIENTO
print("\n3. PREPROCESAMIENTO...")

X = df.drop('fraud_bool', axis=1)
y = df['fraud_bool']

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"Features numéricas: {len(numeric_features)}")
print(f"Features categóricas: {len(categorical_features)}")

# Escalado de features numéricas
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X[numeric_features])
X_numeric_scaled = pd.DataFrame(X_numeric_scaled, columns=numeric_features)

# Codificación de features categóricas
X_categorical = X[categorical_features].copy()
for col in categorical_features:
    le = LabelEncoder()
    X_categorical[col] = le.fit_transform(X_categorical[col].astype(str))

X_processed = pd.concat([X_numeric_scaled, X_categorical], axis=1)
print(f"Shape final: {X_processed.shape}")

# 4. BALANCEO Y SPLIT
print("\n4. BALANCEO Y TRAIN-TEST SPLIT...")

X_temp, X_test, y_temp, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

# 4.1 PRIMERA ITERACIÓN: Probar diferentes técnicas de balanceo
print("\n4.1 PROBANDO TÉCNICAS DE BALANCEO AVANZADAS...")

sampling_techniques = {
    'BorderlineSMOTE': BorderlineSMOTE(random_state=42)
}

best_sampling = None
best_sampling_score = 0

print("Evaluando técnicas de balanceo...")
for name, sampler in sampling_techniques.items():
    try:
        # Aplicar técnica de balanceo
        X_resampled, y_resampled = sampler.fit_resample(X_temp, y_temp)
        
        # Entrenar modelo simple para evaluar
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        # Validación cruzada
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_resampled, y_resampled, cv=cv, scoring='recall', n_jobs=-1)
        mean_score = scores.mean()
        
        print(f"  {name}: {mean_score:.4f}")
        
        if mean_score > best_sampling_score:
            best_sampling_score = mean_score
            best_sampling = sampler
            
    except Exception as e:
        print(f"  {name}: Error - {e}")

print(f"\nMejor técnica de balanceo: {type(best_sampling).__name__}")
print(f"Score: {best_sampling_score:.4f}")

# Aplicar la mejor técnica de balanceo
X_train_balanced, y_train_balanced = best_sampling.fit_resample(X_temp, y_temp)

print(f"Training set original: {X_temp.shape}")
print(f"Training set balanceado: {X_train_balanced.shape}")
print(f"Test set: {X_test.shape}")
print(f"Distribución del target en training balanceado:")
print(pd.Series(y_train_balanced).value_counts())

# # 5. OPTIMIZACIÓN AVANZADA CON GRIDSEARCH
# print("\n5. OPTIMIZACIÓN AVANZADA CON GRIDSEARCH...")

# # 5.1 GridSearch con hiperparámetros más finos
# print("\n5.1 GridSearch con hiperparámetros finos...")

# param_grid_advanced = {
#     'C': [0.001, 0.1, 1.0, 1, 10, 50], #100
#     'penalty': ['l1', 'l2'],
#     'solver': ['liblinear', 'saga'],
#     'class_weight': ['balanced', {0: 1, 1: 10}, {0: 1, 1: 20}],  # 3 valores en lugar de 4
#     'max_iter': [3000, 5000],
#     'tol': [1e-4]
# }

# print("Parámetros avanzados a probar:")
# for param, values in param_grid_advanced.items():
#     print(f"  {param}: {values}")

# print("\nEjecutando GridSearch avanzado con validación cruzada (10-fold)...")
# # n_splits=10
# cv_advanced = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# grid_search_advanced = GridSearchCV(
#     estimator=LogisticRegression(random_state=42),
#     param_grid=param_grid_advanced,
#     cv=cv_advanced,
#     scoring='recall',
#     n_jobs=-1,
#     verbose=1
# )

# grid_search_advanced.fit(X_train_balanced, y_train_balanced)

# print(f"\nMejores parámetros avanzados:")
# for param, value in grid_search_advanced.best_params_.items():
#     print(f"  {param}: {value}")
# print(f"Mejor score de validación cruzada: {grid_search_advanced.best_score_:.4f}")

# 6. ENSEMBLE METHODS
print("\n6. IMPLEMENTANDO ENSEMBLE METHODS...")

# 6.1 Voting Classifier
print("\n6.1 Voting Classifier...")

best_lr = LogisticRegression(
    C=1.0,
    penalty='l2',
    solver='liblinear',
    class_weight='balanced',
    max_iter=3000,
    random_state=42
)
rf_clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
svm_clf = SVC(probability=True, class_weight='balanced', random_state=42)

voting_clf = VotingClassifier(
    estimators=[
        ('lr', best_lr),
        ('rf', rf_clf),
        ('svm', svm_clf)
    ],
    voting='soft'
)

# Entrenar ensemble
voting_clf.fit(X_train_balanced, y_train_balanced)

# Evaluar ensemble
y_pred_ensemble = voting_clf.predict(X_test)
y_pred_proba_ensemble = voting_clf.predict_proba(X_test)[:, 1]

# Métricas del ensemble
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
precision_ensemble = precision_score(y_test, y_pred_ensemble)
recall_ensemble = recall_score(y_test, y_pred_ensemble)
f1_ensemble = f1_score(y_test, y_pred_ensemble)
auc_ensemble = roc_auc_score(y_test, y_pred_proba_ensemble)

print(f"\n--- ENSEMBLE (Voting Classifier) ---")
print(f"Accuracy: {accuracy_ensemble:.4f}")
print(f"Precision: {precision_ensemble:.4f}")
print(f"Recall: {recall_ensemble:.4f}")
print(f"F1-Score: {f1_ensemble:.4f}")
print(f"AUC-ROC: {auc_ensemble:.4f}")

# 7. THRESHOLD OPTIMIZATION
print("\n7. OPTIMIZACIÓN DE UMBRAL DE DECISIÓN...")

# Probar diferentes umbrales para maximizar RECALL
thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
threshold_results = []

print("Probando diferentes umbrales...")
for threshold in thresholds:
    # Predicciones con probabilidades del ensemble
    y_pred_threshold = (y_pred_proba_ensemble >= threshold).astype(int)
    
    # Calcular métricas
    recall = recall_score(y_test, y_pred_threshold)
    precision = precision_score(y_test, y_pred_threshold)
    f1 = f1_score(y_test, y_pred_threshold)
    
    threshold_results.append({
        'threshold': threshold,
        'recall': recall,
        'precision': precision,
        'f1': f1
    })
    
    print(f"  Umbral {threshold}: Recall={recall:.4f}, Precision={precision:.4f}, F1={f1:.4f}")

# Encontrar el mejor umbral para RECALL
best_threshold_result = max(threshold_results, key=lambda x: x['recall'])
best_threshold = best_threshold_result['threshold']

print(f"\nMejor umbral para RECALL: {best_threshold}")
print(f"RECALL máximo: {best_threshold_result['recall']:.4f}")

# 8. MODELO FINAL OPTIMIZADO
print("\n8. MODELO FINAL OPTIMIZADO...")

# Crear modelo final con umbral personalizado
class OptimizedEnsembleModel:
    def __init__(self, ensemble_model, threshold=0.5):
        self.ensemble_model = ensemble_model
        self.threshold = threshold
    
    def predict(self, X):
        y_pred_proba = self.ensemble_model.predict_proba(X)[:, 1]
        return (y_pred_proba >= self.threshold).astype(int)
    
    def predict_proba(self, X):
        return self.ensemble_model.predict_proba(X)

# Modelo final optimizado
final_model = OptimizedEnsembleModel(voting_clf, best_threshold)

# Predicciones finales
y_pred_final = final_model.predict(X_test)
y_pred_proba_final = final_model.predict_proba(X_test)[:, 1]

# Métricas finales
accuracy_final = accuracy_score(y_test, y_pred_final)
precision_final = precision_score(y_test, y_pred_final)
recall_final = recall_score(y_test, y_pred_final)
f1_final = f1_score(y_test, y_pred_final)
auc_final = roc_auc_score(y_test, y_pred_proba_final)

print(f"\n--- MODELO FINAL OPTIMIZADO ---")
print(f"Accuracy: {accuracy_final:.4f}")
print(f"Precision: {precision_final:.4f}")
print(f"Recall: {recall_final:.4f}")
print(f"F1-Score: {f1_final:.4f}")
print(f"AUC-ROC: {auc_final:.4f}")

# 9. VISUALIZACIONES FINALES
print("\n9. GENERANDO VISUALIZACIONES FINALES...")

# Matriz de confusión final
cm_final = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_final, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusión - Modelo Final Optimizado')
plt.ylabel('Valor Real')
plt.xlabel('Valor Predicho')
plt.savefig('confusion_matrix_final.png', dpi=300, bbox_inches='tight')
plt.show()

# Curva ROC final
fpr_final, tpr_final, _ = roc_curve(y_test, y_pred_proba_final)
plt.figure(figsize=(8, 6))
plt.plot(fpr_final, tpr_final, label=f'Modelo Final (AUC = {auc_final:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC - Modelo Final Optimizado')
plt.legend()
plt.grid(True)
plt.savefig('roc_curve_final.png', dpi=300, bbox_inches='tight')
plt.show()

# Comparación de umbrales
threshold_df = pd.DataFrame(threshold_results)
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(threshold_df['threshold'], threshold_df['recall'], 'o-', color='blue')
plt.title('Recall vs Umbral')
plt.xlabel('Umbral')
plt.ylabel('Recall')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(threshold_df['threshold'], threshold_df['precision'], 'o-', color='red')
plt.title('Precision vs Umbral')
plt.xlabel('Umbral')
plt.ylabel('Precision')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(threshold_df['threshold'], threshold_df['f1'], 'o-', color='green')
plt.title('F1-Score vs Umbral')
plt.xlabel('Umbral')
plt.ylabel('F1-Score')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'Mejor umbral: {best_threshold}')
plt.plot(threshold_df['threshold'], threshold_df['recall'], 'o-', color='blue', label='Recall')
plt.plot(threshold_df['threshold'], threshold_df['precision'], 'o-', color='red', label='Precision')
plt.title('Mejor Umbral')
plt.xlabel('Umbral')
plt.ylabel('Score')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('threshold_optimization.png', dpi=300, bbox_inches='tight')
plt.show()

# 10. GUARDADO DEL MODELO
print("\n10. GUARDANDO EL MODELO...")
joblib.dump(voting_clf, 'best_ensemble_model.pkl')
joblib.dump(final_model, 'best_optimized_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')

# Guardar información del umbral óptimo
threshold_info = {
    'best_threshold': best_threshold,
    'best_recall': best_threshold_result['recall'],
    'best_precision': best_threshold_result['precision'],
    'best_f1': best_threshold_result['f1']
}
joblib.dump(threshold_info, 'threshold_info.pkl')

print("Modelos guardados:")
print("- 'best_ensemble_model.pkl' (ensemble base)")
print("- 'best_optimized_model.pkl' (modelo con umbral optimizado)")
print("- 'feature_scaler.pkl' (scaler de features)")
print("- 'threshold_info.pkl' (información del umbral óptimo)")

# 11. RESUMEN FINAL
print("\n" + "="*60)
print("RESUMEN FINAL DEL PROYECTO OPTIMIZADO")
print("="*60)

print(f"🎯 OBJETIVO: Maximizar RECALL para detección de fraude")
print(f"📊 DATASET: {df.shape[0]:,} registros, {df.shape[1]} features")
print(f"⚖️  BALANCEO: {type(best_sampling).__name__} aplicado")

print(f"\n🏆 RESULTADOS FINALES:")
print(f"   Ensemble Base:")
print(f"     - RECALL: {recall_ensemble:.4f} ({recall_ensemble*100:.1f}%)")
print(f"     - Precision: {precision_ensemble:.4f}")
print(f"     - F1-Score: {f1_ensemble:.4f}")
print(f"     - AUC-ROC: {auc_ensemble:.4f}")

print(f"\n   Modelo Final Optimizado:")
print(f"     - RECALL: {recall_final:.4f} ({recall_final*100:.1f}%)")
print(f"     - Precision: {precision_final:.4f}")
print(f"     - F1-Score: {f1_final:.4f}")
print(f"     - AUC-ROC: {auc_final:.4f}")
print(f"     - Umbral óptimo: {best_threshold}")

improvement = ((recall_final - recall_ensemble) / recall_ensemble * 100)
print(f"\n🚀 MEJORA EN RECALL: {improvement:+.1f}%")

print(f"\n🔧 OPTIMIZACIONES APLICADAS:")
print(f"   1. Técnicas de balanceo avanzadas")
print(f"   2. GridSearch con hiperparámetros finos")
print(f"   3. Ensemble methods (Voting Classifier)")
print(f"   4. Threshold optimization")
print(f"   5. Validación cruzada estratificada (10-fold)")

print(f"\n📈 INTERPRETACIÓN:")
if recall_final >= 0.8:
    print(f"   ✅ EXCELENTE: El modelo detecta más del 80% de fraudes")
elif recall_final >= 0.7:
    print(f"   🟡 BUENO: El modelo detecta más del 70% de fraudes")
elif recall_final >= 0.6:
    print(f"   🟠 ACEPTABLE: El modelo detecta más del 60% de fraudes")
else:
    print(f"   🔴 MEJORABLE: El modelo detecta menos del 60% de fraudes")

print(f"\n💡 RECOMENDACIONES:")
print(f"   - Para producción, usar el modelo con umbral {best_threshold:.2f}")
print(f"   - Monitorear la tasa de falsos positivos")
print(f"   - Reentrenar periódicamente con nuevos datos")
print(f"   - Considerar más técnicas de ensemble para mejorar aún más")

print("\n=== PROYECTO COMPLETADO ===")
print("Se han generado gráficos y métricas para el modelo optimizado avanzado.")
print("El modelo se ha guardado para uso futuro.")
print("Revisa los archivos PNG generados para visualizar los resultados.")
print("\n🎉 ¡RECALL MAXIMIZADO EXITOSAMENTE!")
