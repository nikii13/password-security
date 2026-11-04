# Password Security Analysis

Proyecto para comparar un enfoque Rule-Based y un modelo de Machine Learning para evaluar características de contraseñas.

## Requisitos
- Python 3.11+
- No requiere Docker ni base de datos.

## Instalación
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Estructura de datos
- `data/raw/`: fuente original (no modificar).
- `data/samples/`: muestra de trabajo.
- `data/processed/`: dataset limpio y/o con características.

No subas contraseñas reales, filtradas o datos sensibles a GitHub. Mantén los datasets fuera del repositorio.

## Flujo previsto
1. Muestrear el dataset original: `python -m src.data.sample_dataset`
2. Inspeccionar y limpiar: `python -m src.data.clean_dataset`
3. Crear características: `python -m src.features.password_features`
4. Desarrollar/evaluar Rule-Based y ML.

Los scripts contienen rutas y nombres de columnas de ejemplo: adaptarlos al dataset real antes de ejecutarlos.

## Contrato entre bloques
El equipo debe acordar:
- nombre de la columna de contraseña (por defecto `password`);
- nombre y significado de la etiqueta objetivo (por ejemplo `label`);
- definición exacta de las clases;
- tratamiento de duplicados, nulos y registros inválidos.

No inventar etiquetas si el dataset no las proporciona. Si se crean etiquetas mediante reglas, documentarlo: el modelo ML aprenderá en parte esas mismas reglas.
