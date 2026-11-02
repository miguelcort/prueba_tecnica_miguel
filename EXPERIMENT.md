# Diseño del experimento

**Nada de este documento se ejecutó.** No se entrenó ningún modelo y no hay resultados experimentales: solo hipótesis y criterios de decisión. Los conteos sí son reales y salen de `reports/`.

## Referencia metodológica

La inspiración es [DeepSeek LLM](https://arxiv.org/abs/2401.02954) (arXiv 2401.02954). No es una réplica: ese trabajo preentrena modelos de 7B y 67B desde cero sobre 2 billones (10¹²) de tokens, y aquí se adapta un modelo abierto ya preentrenado con muy pocos documentos.

| Principio verificado en el paper | Adaptación aquí |
|---|---|
| Deduplicación agresiva, con mayor alcance: entre 91 volcados elimina cuatro veces más documentos que dentro de uno solo | SHA-256 y MinHash entre todas las versiones, sin borrar pares con diferencias técnicas |
| Filtrado con evaluación lingüística y semántica | Reglas de contenido técnico, cuarentena y grounding |
| Remezcla para aumentar dominios subrepresentados | No se implementó; la cobertura desigual se reporta por proyecto y categoría |
| Separar preentrenamiento y SFT, con vigilancia de sobreajuste | Etapas CPT y SFT separadas, y comparadas entre sí |

"Continued pre-training" no aparece en el paper; lo más cercano es "continual training", dicho del scheduler de aprendizaje. Ese término viene del enunciado de esta prueba. El paper se consultó con una herramienta de resumen sobre su versión HTML (ar5iv), y se revisaron unos 100 000 de 132 000 caracteres: no hubo lectura completa, y las citas deben comprobarse antes de presentarlas. DPO y reinforcement learning quedan fuera de alcance.

## 9.1 Hipótesis

- **CPT:** mejora el vocabulario y el conocimiento de estos servicios, medido en `fact_holdout`.
- **SFT:** mejora la conducta: responder directo, nombrar el servicio y abstenerse sin evidencia.
- **CPT + SFT:** se complementan; CPT aporta el dato y SFT la forma de responderlo.
- **RAG:** gana en documentos no vistos y en información cambiante, porque lee la fuente vigente. No se asume que el fine-tuning sea superior.

## 9.2 Experimentos y niveles que aplican

| Brazo | Definición | `fact_holdout` (39) | `document_holdout` (27) | `unanswerable` (23) |
|---|---|---|---|---|
| E0 Base | Modelo sin adaptar | Línea base | Debe abstenerse | Debe abstenerse |
| E1 RAG | Base con recuperación | Debe responder | Debe responder | Debe abstenerse |
| E2 CPT | Base + `cpt.jsonl` | Debe responder | Debe abstenerse | Debe abstenerse |
| E3 SFT | Base + `sft.jsonl` | Bajo por diseño | Debe abstenerse | Debe abstenerse |
| E4 CPT + SFT | E2 y luego `sft.jsonl` | Debe responder | Debe abstenerse | Debe abstenerse |

Índice de RAG: los 203 chunks de CPT **más los 24 de documentos retenidos**. Excluir esos documentos haría injusta la comparación. Los 28 chunks de versiones no canónicas solo entran con su versión como filtro.

**E3 tendrá puntajes bajos en `fact_holdout` por diseño:** el conjunto excluye todo dato de SFT para evitar fuga, así que E3 nunca vio esas respuestas. Eso no prueba que SFT falle. Para medir si E3 memorizó, usar `reports/sft_recall_probe.jsonl` (33 filas con redacción no vista de datos entrenados), que es **solo diagnóstico** y no forma parte de la evaluación sin fuga.

**Qué probar primero, de menor a mayor costo:** E0 y E1, que no entrenan. Después, sobreajustar E3 con pocas filas para validar el flujo con la sonda. Luego E2 y E4.

Modelos candidatos, **sin verificar** en licencia, tamaño o español: SmolLM3, Qwen de tamaño pequeño, Phi. Elegir uno y fijarlo en todos los brazos.

## 9.3 Comparaciones

- **Se compara:** E1, E2, E3 y E4 contra E0; E4 contra E3 (valor de CPT); E4 contra E1 (entrenar o recuperar).
- **Datos:** el mismo `eval.jsonl` en todos los brazos, nunca usado para ajustar. Para elegir hiperparámetros, separar una validación desde `sft.jsonl` y `cpt.jsonl`.
- **Constantes:** modelo base, tokenizador, prompt de sistema, parámetros de generación, semilla, máximo de tokens y hardware.
- **Mejora demostrada:** diferencia pareada que supere el criterio de decisión y cuyo intervalo excluya cero.
- **Limitaciones:**
  - E0: casi no sabe nada de estos servicios ficticios.
  - E1: depende del recuperador, y el fallo puede estar en la búsqueda.
  - E2: con 203 chunks, puede memorizar sin generalizar.
  - E3: no aprende datos fuera de SFT.
  - E4: suma costo y riesgo de olvidar lo aprendido en CPT.

## 9.4 Métricas

| Métrica | Cómo se calcula | Qué evalúa |
|---|---|---|
| Factual accuracy | Todos los `key_facts` presentes tras normalizar y ningún `must_not_contain` | Dato correcto |
| Faithfulness | Identificadores y números de la respuesta presentes en `evidence` o en el contexto recuperado, con revisión humana de una muestra | Que no invente |
| Answer relevance | Revisión humana o juez LLM auditado, de 1 a 3, sobre una muestra | Que responda lo preguntado |
| Exact Match | Respuesta normalizada igual a `expected_answer`; solo en valores cortos | Precisión estricta |
| F1 | Solapamiento de tokens contra `expected_answer` | Cobertura parcial |
| Abstention accuracy | Aciertos sobre `unanswerable` y sobre `closed_book_expected_behavior`, más la tasa de abstención falsa | Calibración |
| Latencia | p50 y p95 por pregunta, en el mismo hardware | Viabilidad |
| Coste de inferencia | Tokens y tiempo de cómputo por 1000 preguntas | Viabilidad |
| Recall@K (RAG) | Preguntas cuyos `chunk_id` de `evidence` aparecen en el top-K; K=5; en `answer_multi_source` deben aparecer todos | Recuperación |
| Context relevance (RAG) | Chunks recuperados que contienen algún `key_fact` | Ruido del contexto |
| Retrieval precision (RAG) | Chunks de evidencia en el top-K, dividido por K | Precisión de búsqueda |

Límites estadísticos: el peor caso del intervalo de Wilson al 95 % es ±19 puntos con n=23, ±18 con n=27, ±15 con n=39, ±23 con las 15 `answer_multi_source` y ±10 con las 89 totales. Solo se detectan diferencias grandes. Usar pruebas pareadas (McNemar o bootstrap) por pregunta, y ninguna conclusión fina por proyecto.

## 9.5 Decisiones posteriores

Los umbrales son **criterios iniciales**, a revisar antes de ver resultados. Con estos tamaños, una diferencia menor a 15 puntos se trata como no concluyente.

| Resultado | Siguiente decisión |
|---|---|
| E2 no supera a E0 por 15 puntos en `fact_holdout` | Revisar calidad y cantidad del corpus antes de entrenar más |
| E3 acierta más del 80 % en la sonda, pero falla en abstención | Reforzar las filas de abstención |
| E3 mejora la conducta, no la precisión factual | Revisar la generación y validación de respuestas |
| E3 no supera el 50 % en la sonda | El flujo o los datos fallan; detener y depurar |
| E1 supera a E2, E3 y E4 en `document_holdout`, y no pierde en `fact_holdout` | Priorizar RAG; usar fine-tuning solo para la conducta |
| E4 supera a E3 por 15 puntos en `fact_holdout` | Se justifica el costo de CPT |
| E4 no supera a E3 | Eliminar CPT |
| Recall@5 de E1 menor a 80 % | Mejorar el recuperador antes de culpar al modelo |
| Más del 20 % de `unanswerable` respondidas en el mejor brazo | Priorizar el comportamiento de abstención |

## 9.6 Escalabilidad a 10 millones de documentos

**Implementado aquí:** lectura de JSONL en streaming, puntos de control por hash de entradas, firmas MinHash con LSH y paralelismo opcional por documento. Nada se probó a esa escala.

**Diseño, sin implementar:**

- **Incremental:** procesar solo documentos con hash nuevo, sin rehacer lo que no cambió. Hoy los puntos de control funcionan por etapa, no por documento.
- **Lotes:** particionar por proyecto y hash, con reintentos e idempotencia.
- **Checkpoints:** manifiesto por lote en almacenamiento de objetos.
- **Deduplicación distribuida:** agrupar por hash exacto y por bandas LSH. La memoria actual guarda firmas y `lines`, y esa parte debe ir a un almacén clave-valor.
- **Paralelización:** motor distribuido sobre Parquet. La validación cruzada entre SFT y evaluación compara todos contra todos, y debe pasar a índices.
- **Almacenamiento intermedio:** Parquet por etapa, con esquemas versionados.
- **Logs y monitoreo:** los logs JSON pasan a métricas por lote: tasa de rechazo, de grounding y de duplicados, con alertas.
- **Versionado:** versión de dataset ligada al hash de código, configuración y corpus.
- **Generación:** las reglas y los hechos curados no alcanzan a esa escala; haría falta el LLM con grounding y una auditoría humana por muestreo.
