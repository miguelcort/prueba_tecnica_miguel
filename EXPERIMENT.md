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

- **CPT:** mejora el vocabulario y el conocimiento de estos servicios, medido en `fact_holdout` sin las filas del piso constante (9.5).
- **SFT:** mejora la conducta: responder directo, nombrar el servicio y abstenerse sin evidencia.
- **CPT + SFT:** se complementan; CPT aporta el dato y SFT la forma de responderlo.
- **RAG:** gana en documentos no vistos y en información cambiante, porque lee la fuente vigente. No se asume que el fine-tuning sea superior.

## 9.2 Experimentos y niveles que aplican

| Brazo | Definición | `fact_holdout` (42) | `document_holdout` (23) | `unanswerable` (23) | Neutral (64) |
|---|---|---|---|---|---|
| E0 Base | Modelo sin adaptar | Línea base | Debe abstenerse | Debe abstenerse | Línea base |
| E1 RAG | Base con recuperación | Debe responder | Debe responder | Debe abstenerse | Debe responder |
| E2 CPT | Base + `cpt.jsonl` | Debe responder | Debe abstenerse | Debe abstenerse | Debe responder |
| E3 SFT | Base + `sft.jsonl` | Sin regla de decisión; abstenerse no penaliza | Debe abstenerse | Debe abstenerse | Debe responder |
| E4 CPT + SFT | E2 y luego `sft.jsonl` | Debe responder | Debe abstenerse | Debe abstenerse | Debe responder |

En `document_holdout`, 4 filas tienen `closed_book_expected_behavior: not_scored`: parte de su evidencia sí se entrena, y falta al menos una clave. Sirven para medir recuperación, pero en los brazos sin recuperación no se puntúa ni responder ni abstenerse.

**Nivel neutral** ([reports/neutral_eval.jsonl](reports/neutral_eval.jsonl), 64 filas, generado por el pipeline): datos que CPT y SFT enseñan a la vez, con una redacción que SFT no usó. Ningún brazo adaptado parte sin la respuesta, así que la comparación no está decidida de antemano, como sí lo estaba E4 contra E3 en `fact_holdout`. Usa la misma lógica de `key_facts` y `must_not_contain`, y excluye las filas que una respuesta constante ya aprueba. No forma parte de la evaluación sin fuga: E3 y E4 vieron esa respuesta exacta, así que mide retención con otra pregunta, no generalización.

Índice de RAG: los 203 chunks de CPT **más los 24 de documentos retenidos**. Excluir esos documentos haría injusta la comparación. Los 28 chunks de versiones no canónicas solo entran con su versión como filtro.

**E3 tendrá puntajes bajos en `fact_holdout` por diseño:** el conjunto excluye todo dato de SFT para evitar fuga, así que E3 nunca vio esas respuestas. Eso no prueba que SFT falle. Tampoco llegará a cero, porque el piso constante es 26,2 % (9.5). Para medir si E3 memorizó, usar `reports/sft_recall_probe.jsonl` (33 filas con redacción no vista de datos entrenados), que es **solo diagnóstico** y no forma parte de la evaluación sin fuga.

**Qué probar primero, de menor a mayor costo:** E0 y E1, que no entrenan. Después, sobreajustar E3 con pocas filas para validar el flujo con la sonda. Luego E2 y E4.

El modelo y la receta por brazo están en la sección 9.7.

## 9.3 Comparaciones

- **Se compara:** E2 contra E0 en `fact_holdout` sin piso (¿CPT enseña?); E4 contra E3 en neutral (valor de CPT); E4 contra E1 en neutral (entrenar o recuperar). E1 en `document_holdout` se juzga contra un umbral absoluto, no contra brazos que deben abstenerse.
- **Datos:** el mismo `eval.jsonl` y el mismo nivel neutral en todos los brazos, nunca usados para ajustar. Para elegir la época se usa el corte de validación de [reports/training_validation_split.json](reports/training_validation_split.json).
- **Constantes:** modelo base, tokenizador, prompt de sistema, parámetros de generación, semilla, máximo de tokens y hardware.
- **Mejora demostrada:** la zona "gana" de la sección 9.5.
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
| Abstention accuracy | Aciertos sobre `unanswerable` y sobre `closed_book_expected_behavior`, más la tasa de abstención falsa. Se excluyen `not_scored` y la abstención de E3 en `fact_holdout`, cuya respuesta nunca vio | Calibración |
| Latencia | p50 y p95 por pregunta, en el mismo hardware | Viabilidad |
| Coste de inferencia | Tokens y tiempo de cómputo por 1000 preguntas | Viabilidad |
| Recall@K (RAG) | Preguntas cuyos `chunk_id` de `evidence` aparecen en el top-K; K=5; en `answer_multi_source` deben aparecer todos | Recuperación |
| Context relevance (RAG) | Chunks recuperados que contienen algún `key_fact` | Ruido del contexto |
| Retrieval precision (RAG) | Chunks de evidencia en el top-K, dividido por K | Precisión de búsqueda |

Límites estadísticos: el peor caso del intervalo de Wilson al 95 % es ±19 puntos con n=23, ±15 con n=42, ±12 con n=64, ±23 con las 15 `answer_multi_source` y ±10 con las 88 totales. Solo se detectan diferencias grandes (ver potencia en 9.5). Usar pruebas pareadas por pregunta, y ninguna conclusión fina por proyecto.

**Limitación de `key_facts`, sin corregir en esta versión.** Comprobar que unas palabras aparezcan no prueba que la respuesta sea correcta: "application.yml has preference" aprueba una pregunta cuya respuesta correcta es la contraria, y `must_not_contain` está vacío en las filas con respuesta. Las reglas de 9.5 lo compensan con un piso constante y con revisión manual, pero no lo eliminan.

## 9.5 Decisiones posteriores

Los umbrales son **criterios iniciales**, fijados antes de ver resultados. Ningún número de esta sección es un resultado de un modelo: no se entrenó ninguno.

**Piso constante** ([reports/constant_baseline.json](reports/constant_baseline.json), medido sin modelo). Respuesta fija: repetir la pregunta y agregar la lista `evaluation.generic_keys` (tipos de dato, verbos y códigos HTTP).

| Conjunto | Filas | Aprobadas por la respuesta constante | Filas sin piso |
|---|---:|---:|---:|
| `fact_holdout` | 42 | 11 (26,2 %) | 31 |
| `document_holdout` | 23 | 0 (0 %) | 23 |
| Neutral | 64 | 0 (0 %, excluidas por construcción) | 64 |
| `sft_recall_probe` | 33 | 14 (42,4 %) | 19 |

Cómo se usa: las reglas se calculan sin las filas del piso, y el puntaje con todas las filas se reporta junto al piso. Un brazo que no supere el piso no aprendió, aunque el número parezca alto. Antes de declarar "gana", una persona lee las filas donde los dos brazos discrepan y se reporta el resultado con y sin esa revisión.

**Regla por pregunta.** Cada brazo entrenado usa 3 semillas. Una pregunta cuenta como acierto si lo es en al menos 2 de 3; también se reporta la dispersión entre semillas. δ es la diferencia pareada en puntos, con intervalo bootstrap por pregunta al 98,3 %. McNemar exacto, con α = 0,05/3 = 0,0167 por las tres comparaciones principales (Bonferroni).

| Zona | Condición |
|---|---|
| **Gana** | p < 0,0167 y δ ≥ efecto mínimo |
| **Pierde** | El límite superior del intervalo queda por debajo del efecto mínimo |
| **No concluyente** | Cualquier otro caso |

Una ganancia de 1 a 14 puntos no se declara ganadora: queda "no concluyente" o "pierde", según su intervalo.

| Comparación | Conjunto (n) | Efecto mínimo | Gana | No concluyente | Pierde |
|---|---|---:|---|---|---|
| E2 contra E0 | `fact_holdout` sin piso (31) | 30 | CPT enseña datos: seguir con E4 | Ampliar el conjunto antes de decidir | Revisar calidad y cantidad del corpus antes de entrenar más |
| E4 contra E3 | Neutral (64) | 25 | Se justifica el costo de CPT | Quedarse con E3, que es más simple, sin afirmar que CPT no sirve | Eliminar CPT |
| E4 contra E1 | Neutral (64) | 25 | Vale entrenar para responder sin recuperar | Quedarse con E1, que no entrena | Priorizar RAG; fine-tuning solo para la conducta |

Controles de un solo brazo, que no entran en la corrección múltiple:

| Resultado | Siguiente decisión |
|---|---|
| E1 acierta 17 o más de 23 en `document_holdout` (límite inferior de Wilson ≥ 50 %) | RAG responde en documentos no vistos |
| E1 acierta 6 o menos de 23 (límite superior de Wilson < 50 %) | RAG falla: revisar antes de compararlo |
| Recall@5 de E1 menor a 80 % | Mejorar el recuperador antes de culpar al modelo |
| E3 acierta 80 % o más en las 19 filas de la sonda sin piso, pero falla en abstención | Reforzar las filas de abstención |
| E3 no supera el 50 % en esas 19 filas | El flujo o los datos fallan; detener y depurar |
| E3 mejora la conducta (nombra el servicio, se abstiene), pero no supera a E0 en neutral | Revisar la generación y validación de respuestas |
| 5 o más de 23 `unanswerable` respondidas en el mejor brazo (más del 20 %) | Priorizar el comportamiento de abstención |

Se eliminaron dos reglas sesgadas: E4 contra E3 en `fact_holdout`, que solo tiene datos de CPT ausentes de SFT, y E1 contra los demás en `document_holdout`, donde los demás deben abstenerse.

**Potencia** (McNemar exacto, bilateral, calculada, no medida). δ = p10 − p01. Supuesto: discordancia en contra p01, la tasa con la que el mejor brazo pierde una fila que el otro acierta. No hay datos para estimarla, así que se muestran 0, 5 y 10 %. Las filas de una misma plantilla o proyecto no son independientes: la potencia real puede ser menor.

**Tabla sin recalcular.** Se calculó con n=43, 32 y 22. Tras corregir las claves, esos conjuntos tienen 42, 31 y 23 filas, y el repositorio no incluye el script para recalcularla: las filas de esos tres tamaños son aproximaciones. La fila de neutral sigue vigente.

| n (α = 0,0167) | δ = 15 | δ = 20 | δ = 25 | δ = 30 | δ = 40 | δ con 80 % de potencia |
|---:|---|---|---|---|---|---|
| 64 (neutral) | 86/42/28 % | 98/67/50 % | 100/85/70 % | 100/94/85 % | 100/100/97 % | 14/24/29 |
| 43 (`fact_holdout`) | 47/23/16 % | 78/42/29 % | 94/61/44 % | 99/77/60 % | 100/94/85 % | 21/32/38 |
| 32 (`fact_holdout` sin piso) | 19/14/10 % | 46/26/18 % | 72/42/30 % | 89/57/43 % | 99/82/67 % | 27/39/47 |
| 22 (`document_holdout`) | 4/5/5 % | 13/13/10 % | 30/22/16 % | 51/33/24 % | 84/56/44 % | 39/52/61 |

**Qué se puede detectar.** Neutral detecta 25 puntos si la discordancia en contra es de hasta unos 5 %. `fact_holdout` sin piso solo detecta con holgura 30 puntos si E0 casi nunca gana a E2, lo que es plausible porque E0 no conoce estos servicios ficticios; con 5 % en contra, la potencia baja a 57 %. `document_holdout` no sirve para comparar brazos entre sí, y por eso solo tiene un umbral absoluto: con esa regla, E1 pasa con probabilidad 44 % si su acierto real es 70 %, y 84 % si es 80 % (binomial con 17 de 23). Para 15 puntos con 80 % de potencia harían falta entre 59 y 168 filas por comparación, según la discordancia (α = 0,0167); 20 puntos, entre 44 y 107. Son más filas de las que tiene `fact_holdout`, y ampliarlo es el siguiente paso si estas zonas dejan todo como no concluyente.

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

## 9.7 Modelo y receta de entrenamiento

**Nada de esta sección se ejecutó.** Los hiperparámetros son valores iniciales, no resultados probados. Licencia, tamaño y disponibilidad se comprobaron el 2026-11-02 en las fuentes indicadas.

**Modelo: [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)**, el mismo en todos los brazos.

- **Licencia:** Apache 2.0, sin acceso restringido ([API de Hugging Face](https://huggingface.co/api/models/HuggingFaceTB/SmolLM3-3B)).
- **Tamaño:** 3 075 098 624 parámetros; 6,15 GB de pesos en BF16, en dos archivos.
- **Por qué:** es pequeño, abierto, y su ficha declara soporte nativo de inglés y español, los dos idiomas de estos datos (3 de 10 proyectos están en español). Publica cómo se entrenó y usó TRL en su post-entrenamiento.
- **Cuidado:** el modo de razonamiento viene activo. Hay que apagarlo igual al entrenar y al evaluar (`enable_thinking=False` o `/no_think`), o se medirá otra cosa. Requiere `transformers` 4.53.0 o superior.

| Descartado | Licencia y tamaño | Motivo |
|---|---|---|
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | Apache 2.0; 0,6B | Es la alternativa si falta GPU, pero su ficha no detalla el español ("100+ languages") |
| [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) | MIT; 3,8B | Entrenado sobre todo en inglés; su ficha advierte peor rendimiento en otros idiomas |
| [Gemma 3 1B](https://huggingface.co/google/gemma-3-1b-it) | Licencia propia de Gemma; 1B | No es Apache ni MIT, y exige aceptar condiciones para descargar |
| [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) | Apache 2.0; 5,1B | Más grande, y multimodal: capacidad que este problema no usa |
| [Tiny Aya](https://huggingface.co/CohereLabs/tiny-aya-global) | CC-BY-NC; 3,35B | Uso no comercial y acceso restringido |

**Recuperar o ajustar.** La [FAQ de Unsloth](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/faq-+-is-fine-tuning-right-for-me) recomienda combinar ambos: RAG para el dato vigente y fine-tuning para la tarea y el formato. Por eso E1 no se trata como rival a vencer, sino como la línea que el entrenamiento debe justificar. RL y DPO quedan fuera: el [curso de Hugging Face](https://huggingface.co/learn/llm-course/chapter12/2) lo plantea sobre un modelo ya preentrenado y con una señal de recompensa, y aquí aún no hay ningún ajuste que refinar ni una recompensa confiable, porque `key_facts` aprueba respuestas incorrectas.

**Formato:** `cpt.jsonl` es "language modeling" estándar (`text`) y `sft.jsonl` es conversacional (`messages`), ambos aceptados por `SFTTrainer` ([TRL](https://huggingface.co/docs/trl/main/en/dataset_formats)). Antes de entrenar hay que quitar las columnas `metadata` y `provenance`. La distinción entre preentrenar (el texto trae sus propias etiquetas) y ajustar por instrucción (pares de pregunta y respuesta) sigue a [Raschka](https://sebastianraschka.com/faq/docs/pretraining-vs-finetuning-vs-instruction-finetuning.html). La [guía de datasets de Unsloth](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide) pide al menos 100 filas y prefiere más de 1000: SFT queda en 368, poco para generalizar.

**Corte de validación** ([reports/training_validation_split.json](reports/training_validation_split.json), determinista y por proyecto, 10 %). SFT: 32 filas de 16 datos, sin separar las dos preguntas de un dato; quedan 368 para entrenar. CPT: 15 filas, por sección; quedan 188. No incluye nada que necesiten `fact_holdout`, la sonda o el nivel neutral. Solo se usa para elegir la época; `eval.jsonl` nunca.

| Brazo | Datos de entrenamiento | Método | Épocas (máximo) | Tasa de aprendizaje | Lote | Largo máximo |
|---|---|---|---:|---:|---:|---:|
| E0 | Ninguno | Sin entrenar | - | - | - | - |
| E1 | Ninguno | BM25 top-5; índice de 203 + 24 chunks | - | - | - | - |
| E2 | 188 de CPT, formato `text` | LoRA | 3 | 1e-4 | 4 | 512 |
| E3 | 368 de SFT, formato `messages` | LoRA | 3 | 2e-4 | 8 | 512 |
| E4 | Adaptador de E2 de la misma semilla, y luego 368 de SFT | LoRA, continuando el mismo adaptador | 3 | 1e-4 | 8 | 512 |

- **LoRA en E2, E3 y E4:** r=16, alpha=32, dropout 0,05, sobre `q_proj`, `k_proj`, `v_proj` y `o_proj` (confirmar los nombres con `print(model)`). Con tan pocos datos, un ajuste completo solo facilitaría memorizar.
- **Parada:** pérdida de validación en cada época, con paciencia de 1; se conserva el mejor punto. Las 1 a 3 épocas y la vigilancia de la validación siguen al [curso de Hugging Face](https://huggingface.co/learn/llm-course/chapter11/3). Warmup de 5 % y coseno. Si el entrenamiento baja y la validación sube desde la primera época, reducir la tasa a la mitad.
- **Largo máximo:** contar tokens antes de entrenar y confirmar que no se trunca nada; los chunks tienen hasta 1304 caracteres.
- **Semillas:** 13, 42 y 2026 en E2, E3 y E4. E0 y E1 no entrenan y generan de forma determinista, así que basta una corrida.
- **Generación, igual en todos:** sin muestreo, `max_new_tokens=256`, el mismo prompt de sistema y `enable_thinking=False`. E1 agrega los chunks recuperados al prompt. BM25 es el primer recuperador porque no entrena nada y es fácil de auditar; Recall@5 dirá si hace falta otro.
- **Hardware supuesto, sin medir:** una GPU de 16 GB o más para LoRA en BF16. E0 y E1 pueden correr en CPU, más lento.
