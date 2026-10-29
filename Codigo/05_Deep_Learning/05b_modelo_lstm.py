import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings

warnings.filterwarnings("ignore")

# --- RUTAS MLOPS ---
DIR_BASE = os.path.dirname(os.path.abspath(__file__))
DIR_DATOS = os.path.abspath(os.path.join(DIR_BASE, "..", "..", "Datos", "05_Deep_Learning"))
RUTA_X = os.path.join(DIR_DATOS, "X_tensor_lstm.npy")
RUTA_Y = os.path.join(DIR_DATOS, "Y_tensor_lstm.npy")

def construir_y_entrenar_lstm():
    print("🧠 1/4 Cargando Tensores 3D de la Selva...")
    if not os.path.exists(RUTA_X) or not os.path.exists(RUTA_Y):
        print("❌ No se encontraron los tensores. Ejecuta 05a primero.")
        return

    X = np.load(RUTA_X)
    Y = np.load(RUTA_Y)
    
    if len(np.unique(Y)) < 2:
        print("⚠️ Cuidado: No hay suficientes muestras de ambas clases (Sano vs Degradado) para entrenar de forma confiable.")
        
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y if len(np.unique(Y)) > 1 else None)
    
    print(f"   ► Datos de entrenamiento: {X_train.shape[0]} secuencias.")
    print(f"   ► Datos de validación (futuro): {X_test.shape[0]} secuencias.")
    
    # FIX METODOLÓGICO 1: Ponderación Heurística Inversa (Scikit-Learn)
    # Calcula el peso matemático exacto para balancear el 99% vs 1%
    clases = np.unique(y_train)
    if len(clases) == 2:
        pesos_calculados = compute_class_weight(
            class_weight='balanced',
            classes=clases,
            y=y_train
        )
        class_weights = dict(enumerate(pesos_calculados))
        print(f"   ⚖️ Pesos científicos aplicados: Clase Sano={class_weights[0]:.4f}, Clase Degradado={class_weights[1]:.4f}")
    else:
        class_weights = None
        print("   ⚠️ Ajuste de pesos desactivado (solo se encontró una clase en el set de entrenamiento).")
    
    print("⚙️ 2/4 Ensamblando la Arquitectura Profunda (LSTM 12D)...")
    modelo = Sequential([
        Input(batch_shape=(None, X.shape[1], X.shape[2])), # Cambio de batch_shape=(1,...) a None para flexibilidad
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    # FIX METODOLÓGICO 2: Pérdida Focal (Focal Loss) para ignorar los miles de ejemplos sanos "fáciles"
    focal_loss = tf.keras.losses.BinaryFocalCrossentropy(alpha=0.25, gamma=2.0)

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy', 
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    early_stop = EarlyStopping(monitor='val_auc', mode='max', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=5, min_lr=0.00001)

    print("🚀 3/4 Entrenando el Motor Predictivo...")
    historia = modelo.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32, # FIX: Incrementar batch size acelera el entrenamiento; batch_size=1 es muy ineficiente
        shuffle=True, # FIX: Necesario shuffle=True para que el optimizador aprenda patrones de la minoría
        validation_data=(X_test, y_test),
        class_weight=class_weights,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    print("📊 4/4 Evaluando el Desempeño del Modelo en Datos No Vistos...")
    y_pred_probs = modelo.predict(X_test, verbose=0).ravel()
    y_pred_clases = np.where(y_pred_probs > 0.5, 1, 0)
    
    print("\nREPORTE DE CLASIFICACIÓN CIENTÍFICA:")
    try:
        print(classification_report(y_test, y_pred_clases, target_names=['Sano', 'Degradado']))
    except ValueError:
        print(classification_report(y_test, y_pred_clases))
    
    # --- VISUALIZACIÓN ---
    plt.style.use('default')
    sns.set_theme(style="white")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor='white')
    
    cm = confusion_matrix(y_test, y_pred_clases)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                xticklabels=['Sano', 'Degradado'], yticklabels=['Sano', 'Degradado'],
                cbar=False, annot_kws={"size": 14, "weight": "bold"})
    ax1.set_title('Matriz de Confusión (Set de Prueba)', fontweight='bold', pad=15)
    ax1.set_ylabel('Realidad')
    ax1.set_xlabel('Predicción de la LSTM')
    
    try:
        fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, color='#d62728', lw=2.5, label=f'Curva ROC (AUC = {roc_auc:.3f})')
    except Exception:
        ax2.text(0.5, 0.5, "No se puede calcular AUC", ha='center', va='center')

    ax2.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Tasa de Falsos Positivos')
    ax2.set_ylabel('Tasa de Verdaderos Positivos')
    ax2.set_title('Rendimiento Predictivo (ROC-AUC)', fontweight='bold', pad=15)
    ax2.legend(loc="lower right")