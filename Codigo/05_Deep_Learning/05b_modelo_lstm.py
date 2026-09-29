import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
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
    X = np.load(RUTA_X)
    Y = np.load(RUTA_Y)
    
    # Separamos en conjunto de Entrenamiento (80%) y Prueba (20%)
    # Usamos stratify para garantizar que la misma proporción de anomalías vaya a ambas partes
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    
    print(f"   ► Datos de entrenamiento: {X_train.shape[0]} secuencias.")
    print(f"   ► Datos de validación (futuro): {X_test.shape[0]} secuencias.")
    
    # Manejo del desbalance de clases (hay mucha más selva sana que deforestada)
    clases, conteos = np.unique(y_train, return_counts=True)
    peso_clase_0 = (1 / conteos[0]) * (len(y_train) / 2.0)  # Sano
    peso_clase_1 = (1 / conteos[1]) * (len(y_train) / 2.0)  # Deforestado
    class_weights = {0: peso_clase_0, 1: peso_clase_1}
    
    print("⚙️ 2/4 Ensamblando la Arquitectura Profunda (LSTM)...")
    modelo = Sequential([
        # Capa LSTM 1: Extrae patrones en la línea de tiempo
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        BatchNormalization(),
        Dropout(0.3), # Apaga el 30% de las neuronas para evitar memorización
        
        # Capa LSTM 2: Condensa la información
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        
        # Red Densa de toma de decisión
        Dense(16, activation='relu'),
        
        # Salida: Neurona binaria (1 = Degradación Inminente, 0 = Sano)
        Dense(1, activation='sigmoid')
    ])
    
    # Compilamos el modelo
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy', # Mide el error en clasificación binaria
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    # Callbacks: "Salvavidas" durante el entrenamiento
    # 1. Detiene el entrenamiento si la red deja de aprender después de 10 intentos
    early_stop = EarlyStopping(monitor='val_auc', mode='max', patience=15, restore_best_weights=True)
    # 2. Reduce la velocidad de aprendizaje si se atasca
    reduce_lr = ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=5, min_lr=0.00001)

    print("🚀 3/4 Entrenando el Motor Predictivo (Esto puede tardar unos minutos)...")
    historia = modelo.fit(
        X_train, y_train,
        epochs=100, # Épocas máximas
        batch_size=32, # Aprende de 32 meses a la vez
        validation_data=(X_test, y_test),
        class_weight=class_weights,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    print("📊 4/4 Evaluando el Desempeño del Modelo en Datos No Vistos...")
    # Predicción sobre el 20% de datos ocultos
    y_pred_probs = modelo.predict(X_test).ravel()
    # Si la probabilidad de deforestación es mayor al 50%, marcamos como 1
    y_pred_clases = np.where(y_pred_probs > 0.5, 1, 0)
    
    print("\nREPORTE DE CLASIFICACIÓN CIENTÍFICA:")
    print(classification_report(y_test, y_pred_clases, target_names=['Sano', 'Degradado']))
    
    # --- VISUALIZACIÓN DE RENDIMIENTO ---
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred_clases)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                xticklabels=['Sano', 'Degradado'], yticklabels=['Sano', 'Degradado'])
    ax1.set_title('Matriz de Confusión (Set de Prueba)', fontweight='bold')
    ax1.set_ylabel('Realidad')
    ax1.set_xlabel('Predicción de la LSTM')
    
    # 2. Curva ROC-AUC (Qué tan bien separa la IA las dos clases)
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    ax2.plot(fpr, tpr, color='cyan', lw=2, label=f'Curva ROC (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Tasa de Falsos Positivos')
    ax2.set_ylabel('Tasa de Verdaderos Positivos')
    ax2.set_title('Rendimiento Predictivo (ROC-AUC)', fontweight='bold')
    ax2.legend(loc="lower right")
    
    plt.tight_layout()
    ruta_img = os.path.join(DIR_DATOS, "01_Evaluacion_LSTM.png")
    plt.savefig(ruta_img, dpi=300)
    
    # Guardamos el cerebro neuronal para usarlo en inferencia futura
    ruta_modelo = os.path.join(DIR_DATOS, "modelo_lstm_deforestacion.h5")
    modelo.save(ruta_modelo)
    
    print(f"✅ Panel de evaluación guardado en: {ruta_img}")
    print(f"✅ Cerebro neuronal exportado a: {ruta_modelo}")

if __name__ == "__main__":
    construir_y_entrenar_lstm()