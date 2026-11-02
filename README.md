# Datos para adaptar un LLM a documentación técnica

Convierte 107 archivos de documentación técnica de 10 proyectos (`docs_raw/`) en tres datasets: texto para pre-entrenamiento continuo, preguntas y respuestas para ajuste supervisado, y un conjunto de evaluación. Cada fila conserva la procedencia exacta: archivo, hash, sección y líneas de origen. El proyecto **no entrena ningún modelo**; prepara los datos y valida que estén bien formados.

**Video explicativo (1920×1080, en español):** [docs/video/explainer.mp4](docs/video/explainer.mp4)

## Requisitos

- Python 3.9 o superior, con `pip` y `venv`.
- Sin GPU, sin conexión a servicios externos y sin claves. Solo `pip` descarga dependencias.

## Instalación

Desde la carpeta del proyecto:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

`requirements-dev.txt` incluye `requirements.txt` (PyYAML) y pytest.

## Uso

### 1. Ejecutar el pipeline completo

```bash
python -m src.pipeline --stage all
```

Lee `docs_raw/`, aplica las seis etapas y escribe los resultados en `outputs/` y `reports/`. Si una validación falla, el proceso termina con error y código distinto de cero.

### 2. Ejecutar las pruebas

```bash
python -m pytest -q
```

Las pruebas construyen proyectos temporales a partir de `configs/` y `docs_raw/`, así que no modifican los resultados del repositorio.

### Etapas

| Etapa | Qué hace |
|---|---|
| `ingestion` | Inventaría cada archivo y calcula su SHA-256. Registra lo que no se pudo leer. |
| `curation` | Normaliza, enmascara datos sensibles, elimina duplicados exactos, filtra páginas sin contenido técnico, extrae hechos y crea fragmentos. |
| `cpt` | Construye texto continuo para pre-entrenamiento continuo. |
| `sft` | Convierte hechos en preguntas y respuestas; conserva solo las que pasan la verificación contra la fuente. |
| `evaluation` | Construye preguntas que no aparecen en SFT, por nivel, incluidas las de abstención. |
| `validation` | Verifica procedencia, datos inventados y fuga entre datasets; escribe los reportes. |

Para ejecutar una etapa sola:

```bash
python -m src.pipeline --stage sft
```

Agregue `--force` para rehacer una etapa aunque no haya cambios. Sin `--force`, una etapa cuyas entradas, código y configuración no cambiaron se omite.

### Configuración

- `configs/pipeline.yaml`: rutas, semilla, parámetros de deduplicación, fragmentación, muestreo y validación.
- `configs/curated_facts.yaml`: hechos curados.
- `configs/quarantine.yaml`: exclusiones de archivos, redacciones y líneas en cuarentena manual.

La generación con LLM está apagada por defecto (`sft.llm.enabled: false`). Para activarla haría falta además `LLM_MODE=api`, `LLM_BASE_URL` y `LLM_MODEL`; no se ha ejecutado en este repositorio.

## Resultados esperados

Al ejecutar el pipeline, los archivos siguientes quedan generados (son los mismos que están versionados en el repositorio):

| Archivo | Filas | Contenido |
|---|---:|---|
| [outputs/cpt.jsonl](outputs/cpt.jsonl) | 203 | Texto continuo para pre-entrenamiento continuo |
| [outputs/sft.jsonl](outputs/sft.jsonl) | 400 | Preguntas y respuestas validadas; 298 por reglas y 102 de hechos curados con ayuda de IA |
| [outputs/eval.jsonl](outputs/eval.jsonl) | 89 | 39 `fact_holdout`, 27 `document_holdout`, 23 `unanswerable` |
| [reports/summary.md](reports/summary.md) | | Resumen de calidad generado por el pipeline |
| [reports/leakage_report.json](reports/leakage_report.json) | | Revisión de fuga entre SFT y evaluación |

Cada fila incluye `provenance` con `source_file`, `source_hash`, `document_id`, `section` y el rango de líneas. Según `reports/summary.md`, la procedencia es válida en el 100 % de las filas de los tres archivos.

Las pruebas (`python -m pytest -q`) pasaron en la última verificación (42 pruebas). Dos ejecuciones del pipeline producen archivos idénticos byte a byte.

## Estructura del proyecto

```
docs_raw/          Documentación original (solo lectura)
src/               Código del pipeline, una carpeta por etapa
configs/           Configuración, hechos curados y cuarentena
outputs/           Datasets finales: cpt.jsonl, sft.jsonl, eval.jsonl
reports/           Reportes de calidad, deduplicación y fuga
tests/             Pruebas unitarias y de integración
docs/video/        Video explicativo (explainer.mp4)
work/, logs/       Archivos intermedios y registros (ignorados por git)
EXPERIMENT.md      Diseño del experimento E0 a E4 (sin ejecutar)
```

## Estado del proyecto

- **No se entrenó ningún modelo.** No hay resultados experimentales.
- `EXPERIMENT.md` describe el diseño de los experimentos E0 a E4 y los criterios de decisión. Ninguno se ha ejecutado.
- Las citas al paper de referencia en `EXPERIMENT.md` aún no están verificadas.

## Limitaciones

- Cobertura desigual: `price-engine-api` tiene 2 filas de CPT y `comparison` tiene 4 de SFT.
- Muestras pequeñas: en el peor caso, el intervalo de Wilson al 95 % es de ±15 a ±19 puntos por nivel.
- Las plantillas se notan: 264 de 400 filas de SFT son de dificultad `easy`.
- La validación comprueba que las respuestas aparezcan en la fuente, pero no prueba que su significado sea correcto.
- El tamaño de 10 millones de documentos es un diseño (`EXPERIMENT.md`, sección 9.6); no se probó.

## Problemas conocidos

- **Datos sensibles en `docs_raw/`.** Algunos archivos contienen nombres de personas y hosts de aspecto interno (por ejemplo, `price-engine-api/latest/guide/contacto.md`). Las salidas generadas los enmascaran o excluyen, pero el original se conserva. Revise antes de publicar el repositorio.
- **Versión canónica de `2p-revenue-optimizer-api`.** La carpeta `latest` apunta a `0.0.3-rc-1`, aunque `0.2.2` parece más reciente. Se usa `latest`; confirmar con quien entregó la documentación.
- **Revisión pendiente de hechos curados.** `configs/curated_facts.yaml`, sobre todo la sección `multi_source`, requiere revisión manual.

## Documentos relacionados

- [EXPERIMENT.md](EXPERIMENT.md): hipótesis, brazos E0 a E4, métricas y criterios de decisión.
- [reports/summary.md](reports/summary.md): resumen de calidad.
