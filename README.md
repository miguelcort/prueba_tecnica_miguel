# Datos para adaptar un LLM a documentación técnica

Convierte `docs_raw/` (107 archivos de 10 proyectos) en datos de entrenamiento y evaluación para un modelo abierto pequeño, con procedencia verificable en cada fila. Se inspira en [DeepSeek LLM](https://arxiv.org/abs/2401.02954) para curar datos y separar etapas, pero **no lo replica**: ellos preentrenan desde cero, aquí se adapta un modelo ya preentrenado. **No se entrenó ningún modelo.**

## Ejecución desde cero

Requiere Python 3.9 o superior. El flujo no necesita GPU, red ni claves; solo `pip` descarga dependencias.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m src.pipeline --stage all
python -m pytest -q
```

Etapas sueltas, en orden: `--stage ingestion|curation|cpt|sft|evaluation|validation`. Agregar `--force` para rehacer una etapa sin cambios. Si alguna validación falla, el proceso termina con error.

## Entregables

| Archivo | Filas | Contenido |
|---|---:|---|
| [outputs/cpt.jsonl](outputs/cpt.jsonl) | 203 | Texto continuo para continued pre-training |
| [outputs/sft.jsonl](outputs/sft.jsonl) | 400 | Preguntas y respuestas para supervised fine-tuning |
| [outputs/eval.jsonl](outputs/eval.jsonl) | 89 | Evaluación separada: 39 `fact_holdout`, 27 `document_holdout`, 23 `unanswerable` |
| [EXPERIMENT.md](EXPERIMENT.md) | | Hipótesis, brazos E0 a E4, métricas y decisiones |

Las métricas de calidad están en [reports/summary.md](reports/summary.md) y la revisión de fuga en [reports/leakage_report.json](reports/leakage_report.json).

## Decisiones clave

- **Sin LLM para generar:** 298 filas de SFT salen de reglas y 102 de hechos curados con ayuda de IA, verificados contra citas textuales. Es reproducible y gratis, pero más repetitivo. La interfaz de LLM existe, apagada por defecto, y solo se probó con un proveedor falso.
- **Versión canónica:** la que publica la carpeta `latest`, nunca la inferida por nombre o fecha. En `2p-revenue-optimizer-api`, 0.2.2 parece más nueva que 0.0.3-rc-1, a la que apunta `latest`; es un riesgo abierto.
- **Duplicados:** 21 exactos eliminados. MinHash marcó 2 pares con diferencias técnicas, que no se eliminan.
- **Partición por documento y por dato:** por proyecto no sirve, porque cada dato es propio de su servicio y 5 de 10 proyectos no tienen 3 documentos con 3 datos o más. Ver `project_split_viability` en el reporte de fuga.
- **Evaluación sin datos de SFT:** por eso E3 puntuará bajo por diseño. Para medir memorización está [reports/sft_recall_probe.jsonl](reports/sft_recall_probe.jsonl), solo diagnóstico.
- **Datos sensibles:** salidas con correos, hosts internos y canales enmascarados; `contacto.md` excluido. También se quitaron los criterios de whitelist del WAF, que sirven para evadir límites; siguen las columnas `is_bot`, así que la mitigación es parcial.
- **Contradicciones:** 65 líneas en cuarentena manual ([configs/quarantine.yaml](configs/quarantine.yaml)).

## Validación

Cada corrida recalcula el SHA-256 de cada fuente, comprueba `document_id`, líneas y sección, verifica las respuestas contra su fuente y confirma la ausencia en `unanswerable`. Procedencia válida: 100 % en los tres archivos. Para la fuga busca preguntas y respuestas de evaluación en SFT, y resultó 0 casos. Las respuestas de `fact_holdout` sí están en CPT por diseño, y 10 preguntas comparten un dato parcial con SFT: ambas cosas se reportan. El grounding detecta datos inventados, pero no prueba que el significado sea correcto.

## Limitaciones

- Cobertura desigual: `price-engine-api` tiene 2 filas de CPT y `comparison` 4 de SFT.
- Muestras chicas: de ±15 a ±19 puntos por nivel, en el peor caso.
- Las plantillas siguen notándose, y 264 de 400 filas de SFT son `easy`.
- El tamaño de 10 millones de documentos es diseño, no se probó.

## Lo que falta por hacer

**Para entregar:**

1. [ ] Decidir qué hacer con `docs_raw/`: viene tal como se entregó, con nombres de personas en `price-engine-api/latest/guide/contacto.md` y hosts de aspecto interno, y un repositorio público los expone.
2. [ ] Confirmar la versión canónica de `2p-revenue-optimizer-api`, si es posible, con quien entregó la prueba.
3. [ ] Revisar a mano [configs/curated_facts.yaml](configs/curated_facts.yaml), sobre todo la sección `multi_source`.
4. [ ] Comprobar las citas del paper en `EXPERIMENT.md`; no se leyó completo.
5. [ ] Publicar, desde esta carpeta y sin líneas de atribución en el commit:

   ```bash
   git init && git add . && git commit -m "feat: add LLM data pipeline"
   ```
   ```bash
   gh repo create <nombre-del-repo> --public --source=. --push
   ```
6. [ ] Probar los comandos de arriba en otra máquina.
7. [ ] Preparar la presentación a partir de este README y `EXPERIMENT.md`.

**Si se continúa:**

8. [ ] Ejecutar E0 y E1, y luego el resto de `EXPERIMENT.md`.
9. [ ] Activar generación con LLM para ganar variedad, siempre con grounding y muestreo humano.
10. [ ] Implementar lo pendiente para escalar (`EXPERIMENT.md`, sección 9.6).
