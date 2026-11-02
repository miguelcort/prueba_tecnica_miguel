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
| [outputs/eval.jsonl](outputs/eval.jsonl) | 88 | 43 `fact_holdout`, 22 `document_holdout`, 23 `unanswerable` |
| [reports/summary.md](reports/summary.md) | | Resumen de calidad generado por el pipeline |
| [reports/leakage_report.json](reports/leakage_report.json) | | Revisión de fuga entre SFT y evaluación |
| [reports/constant_baseline.json](reports/constant_baseline.json) | | Lo que aprueba una respuesta fija, sin modelo: el piso de las métricas |
| [reports/neutral_eval.jsonl](reports/neutral_eval.jsonl) | 64 | Datos enseñados por CPT y SFT, con otra redacción, para comparar brazos sin sesgo |
| [reports/training_validation_split.json](reports/training_validation_split.json) | | Filas que quedan fuera del entrenamiento para elegir la época: 32 de SFT y 15 de CPT |

Cada fila incluye `provenance` con `source_file`, `source_hash`, `document_id`, `section` y el rango de líneas. Según `reports/summary.md`, la procedencia es válida en el 100 % de las filas de los tres archivos.

Las pruebas (`python -m pytest -q`) pasaron en la última verificación (54 pruebas). Dos ejecuciones del pipeline producen archivos idénticos byte a byte.

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
- Muestras pequeñas: en el peor caso, el intervalo de Wilson al 95 % es de ±14 a ±19 puntos por nivel.
- Las plantillas se notan: 264 de 400 filas de SFT son de dificultad `easy`.
- La validación comprueba que las respuestas aparezcan en la fuente, pero no prueba que su significado sea correcto.
- El tamaño de 10 millones de documentos es un diseño (`EXPERIMENT.md`, sección 9.6); no se probó.

- `key_facts` no prueba que una respuesta sea correcta: una respuesta fija aprueba 11 de 43 filas de `fact_holdout` (25,6 %). Se reporta como piso y `EXPERIMENT.md` lo descuenta, pero no está corregido.

## Problemas conocidos

- **Datos sensibles en `docs_raw/`.** Algunos archivos contienen nombres de personas y hosts de aspecto interno (por ejemplo, `price-engine-api/latest/guide/contacto.md`). Las salidas generadas los enmascaran o excluyen, pero el original se conserva. Revise antes de publicar el repositorio.
- **Versión canónica de `2p-revenue-optimizer-api`.** La carpeta `latest` apunta a `0.0.3-rc-1`, aunque `0.2.2` parece más reciente. Se usa `latest`; confirmar con quien entregó la documentación.
- **Revisión pendiente de hechos curados.** `configs/curated_facts.yaml`, sobre todo la sección `multi_source`, requiere revisión manual.

## Decisiones y por qué

- **Sin LLM para generar.** 298 filas de SFT salen de reglas y 102 de hechos curados con ayuda de IA. Así el flujo es reproducible y gratis, a cambio de ser más repetitivo.
- **Partición por documento y por dato, no por proyecto.** Cada dato es propio de su servicio, y 5 de 10 proyectos no tienen documentos suficientes para retener uno. Un proyecto retenido solo mediría recuperación.
- **Versión canónica: la que publica `latest`, nunca la inferida por nombre o fecha.** Adivinar la versión es peor que declarar el riesgo, que sigue abierto en `2p-revenue-optimizer-api`.
- **Deduplicar solo lo exacto.** Se eliminaron 21 copias exactas. MinHash marcó 2 pares que no se borran, porque sus diferencias son técnicas.
- **Cuarentena manual de 65 líneas.** Contradicen otras páginas o sirven para evadir límites del WAF; enseñarlas sería peor que perderlas.
- **Fuga medida por proyecto, sin largo mínimo.** Un mínimo de 12 caracteres ocultaba claves como `Gradle 7.3` y reportaba 0. Ahora se compara por palabra completa dentro de cada proyecto, ignorando una lista corta y explícita de claves genéricas. 6 filas pasaron de `document_holdout` a `fact_holdout`, porque CPT sí las enseña.
- **Un nivel neutral y un piso constante, no comparaciones decididas de antemano.** E4 contra E3 en `fact_holdout` ganaba por construcción, así que no probaba nada.
- **SmolLM3-3B como modelo propuesto.** Apache 2.0, 3B de parámetros y soporte declarado de inglés y español, los idiomas de estos datos. Sigue sin entrenarse.

## Siguientes pasos y por qué

1. **Revisar a mano `configs/curated_facts.yaml`.** Son 102 filas de SFT y hay que poder defender cada una.
2. **Confirmar la versión canónica de `2p-revenue-optimizer-api`.** Si `latest` no es la correcta, cambian sus filas de los tres archivos.
3. **Ejecutar E0 y E1.** No entrenan, y son la línea que cualquier entrenamiento debe superar.
4. **Sobreajustar E3 con pocas filas.** Valida el flujo con poco costo antes de gastar en E2 y E4.
5. **Corregir `key_facts` y poblar `must_not_contain`.** Hoy aprueban respuestas fijas o incorrectas, y es el mayor límite de lo que se puede concluir.
6. **Ampliar `fact_holdout`.** Con 32 filas sin piso, solo se detectan diferencias de unos 30 puntos.
7. **Probar el flujo en otra máquina.** El PDF avisa que puede correrse en la entrevista, y solo se probó en macOS con Python 3.9.

## Documentos relacionados

- [EXPERIMENT.md](EXPERIMENT.md): hipótesis, brazos E0 a E4, métricas y criterios de decisión.
- [reports/summary.md](reports/summary.md): resumen de calidad.
