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
| [outputs/eval.jsonl](outputs/eval.jsonl) | 88 | 42 `fact_holdout`, 23 `document_holdout`, 23 `unanswerable` |
| [reports/summary.md](reports/summary.md) | | Resumen de calidad generado por el pipeline |
| [reports/leakage_report.json](reports/leakage_report.json) | | Revisión de fuga entre SFT y evaluación |
| [reports/constant_baseline.json](reports/constant_baseline.json) | | Lo que aprueba una respuesta fija, sin modelo: el piso de las métricas |
| [reports/neutral_eval.jsonl](reports/neutral_eval.jsonl) | 64 | Datos enseñados por CPT y SFT, con otra redacción, para comparar brazos sin sesgo |
| [reports/training_validation_split.json](reports/training_validation_split.json) | | Filas que quedan fuera del entrenamiento para elegir la época: 32 de SFT y 15 de CPT |

Cada fila incluye `provenance` con `source_file`, `source_hash`, `document_id`, `section` y el rango de líneas. Según `reports/summary.md`, la procedencia es válida en el 100 % de las filas de los tres archivos.

Las pruebas (`python -m pytest -q`) pasaron en la última verificación (55 pruebas). Dos ejecuciones del pipeline producen archivos idénticos byte a byte.

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

- `key_facts` no prueba que una respuesta sea correcta: una respuesta fija aprueba 11 de 42 filas de `fact_holdout` (26,2 %). Se reporta como piso y `EXPERIMENT.md` lo descuenta. Se corrigieron las claves que cambiaban una medición, no todas (ver "Problemas de datos").

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
- **Fuga medida por proyecto, sin largo mínimo.** Un mínimo de 12 caracteres ocultaba claves como `Gradle 7.3` y reportaba 0. Ahora se compara por palabra completa dentro de cada proyecto, ignorando una lista corta y explícita de claves genéricas. 5 filas pasaron de `document_holdout` a `fact_holdout`, porque CPT sí las enseña.
- **Un nivel neutral y un piso constante, no comparaciones decididas de antemano.** E4 contra E3 en `fact_holdout` ganaba por construcción, así que no probaba nada.
- **SmolLM3-3B como modelo propuesto.** Apache 2.0, 3B de parámetros y soporte declarado de inglés y español, los idiomas de estos datos. Sigue sin entrenarse.

## Problemas de datos: cuáles se resuelven y por qué

El criterio es si el problema cambia una medición o una decisión de `EXPERIMENT.md`. Los IDs son los de `outputs/eval.jsonl` actual.

| Problema | Decisión | Por qué | Evidencia |
|---|---|---|---|
| La respuesta correcta no pasaba su propia clave por una forma verbal o un plural | Resuelto | Ningún brazo podía llegar al 100 % en esas filas | `eval_000058` y `eval_000063`: claves `request parameters` e `important branches`. `neutral_000050`: `frozen image` |
| `eval_000063` pregunta por el tipo de almacén, y `MySQL` no era clave | Resuelto | Una respuesta sin el dato preguntado podía aprobar | `MySQL` agregado, citando la línea de la fuente |
| `eval_000016` exigía `8 jobs`, que ya está en la pregunta | Resuelto | La clave no probaba nada; repetir la pregunta la cumplía | Ahora exige `4 scopes`, que es parte del motivo |
| `eval_000037` usaba `3PL` e `inbound`, palabras sueltas presentes en CPT por otros motivos | Resuelto | La fila estaba en `fact_holdout` sin que CPT enseñara el dato. Vuelve a `document_holdout` | Claves `3PL stock movements` e `inbound movements` |
| `must_not_contain` prohibía una palabra de la respuesta correcta | Resuelto | La fila era imposible de aprobar | `probe_000033`: `version`. `src/evaluation/build.py` ya no prohíbe lo que dice la respuesta |
| `eval_000006` exige solo `RetailFlow`, y CPT lo menciona como `RetailFlow CLI` | Aceptado | La clave es la respuesta correcta; endurecerla rechazaría una respuesta correcta. CPT no enseña bien el dato, así que pesa 1 fila en contra de E2 | Sigue en `fact_holdout` |
| `eval_000002`: una regla de palabras no comprueba la dirección de una preferencia | Aceptado | Necesita lectura humana, no otra clave | Revisión de filas discrepantes (9.5) |
| `eval_000001`: `100 ms` no coincide con `100ms` | Aceptado | Agregar alias exige cambiar la regla de puntuación | Revisión de filas discrepantes (9.5) |
| Claves largas que rechazan una paráfrasis correcta | Aceptado | Afecta a todos los brazos por igual, y la revisión humana cubre las discrepancias | Revisión de filas discrepantes (9.5) |
| Fuga por paráfrasis: `eval_000063` (CPT dice MySQL) y `eval_000009` (CPT dice JDK 17) | Aceptado | La comprobación es por claves y no detecta sinónimos. Puede penalizar a un brazo que responde bien | Sin cambios |
| Regla para detectar abstención | Abierto, sin cambios | Decisión pendiente: Abstention accuracy no se puede calcular hasta tenerla | [EXPERIMENT.md](EXPERIMENT.md), 9.4 |
| Neutral usa datos que SFT ya enseñó | Abierto, sin cambios | Mide retención con otra pregunta, no generalización, y así está declarado | [EXPERIMENT.md](EXPERIMENT.md), 9.2 |

## Cómo se valida que lo construido es correcto

| Comprobación | Qué demuestra | Cómo reproducirla |
|---|---|---|
| Etapa de validación | Procedencia, grounding, separación y conteos. El flujo falla si no se cumplen | `python -m src.pipeline --stage validation` |
| Prueba contra una fuga plantada | Que la validación detecta una fuga y un original alterado | `python -m pytest -q -k "planted or changed_original"` |
| Cada respuesta de referencia pasa su propia regla | Que evaluación, neutral y sonda se pueden aprobar | `python -m pytest -q -k gold_answer` |
| Piso constante | Cuánto aprueba una respuesta fija, sin modelo | [reports/constant_baseline.json](reports/constant_baseline.json) |
| Reproducibilidad | Dos ejecuciones dan archivos idénticos, y los versionados coinciden con una regeneración | `python -m pytest -q -k "byte_identical or committed_outputs"` |
| Revisión independiente | Agentes que recalculan los conteos y leen muestras de filas, además de buscar defectos | [.claude/agents/llm-qa.md](.claude/agents/llm-qa.md) |

**Sin validar:** si el modelo aprende, porque no hay modelo ni predicciones; el significado de las 102 filas curadas, que falta revisar a mano; y el flujo en otra máquina.

## Qué se probaría primero

E0 y E1, porque no entrenan y fijan la línea que cualquier entrenamiento debe superar. Después, E3 con pocas filas para validar el flujo, y al final E2 y E4. El detalle está en [EXPERIMENT.md](EXPERIMENT.md), 9.2.

## Siguientes pasos y por qué

1. **Revisar a mano `configs/curated_facts.yaml`.** Son 102 filas de SFT y hay que poder defender cada una.
2. **Confirmar la versión canónica de `2p-revenue-optimizer-api`.** Si `latest` no es la correcta, cambian sus filas de los tres archivos.
3. **Ejecutar E0 y E1.** No entrenan, y son la línea que cualquier entrenamiento debe superar.
4. **Sobreajustar E3 con pocas filas.** Valida el flujo con poco costo antes de gastar en E2 y E4.
5. **Terminar de corregir `key_facts` y poblar `must_not_contain`.** Aún aprueban respuestas fijas o incorrectas, y es el mayor límite de lo que se puede concluir.
6. **Ampliar `fact_holdout`.** Con 31 filas sin piso, solo se detectan diferencias de unos 30 puntos.
7. **Probar el flujo en otra máquina.** El PDF avisa que puede correrse en la entrevista, y solo se probó en macOS con Python 3.9.

## Documentos relacionados

- [EXPERIMENT.md](EXPERIMENT.md): hipótesis, brazos E0 a E4, métricas y criterios de decisión.
- [reports/summary.md](reports/summary.md): resumen de calidad.
- [Presentación en Google Drive](https://docs.google.com/presentation/d/1FnoXjosHr8nlPvqhcNjMQBC6CdFYagPf/edit?usp=sharing&ouid=112428884312748451969&rtpof=true&sd=true): material para la entrevista, con el diseño del experimento.
