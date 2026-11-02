"""Question and answer wordings per fact kind and language.

Each kind has a pool of question wordings and of answer wordings. A seeded hash picks them per
fact, so the dataset does not repeat one sentence shape, and the last wording of the shuffled pool
stays out of training. {svc} always names the service: a question could fit several services.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from ..common import rank

TEMPLATES = {
    ("endpoint", "en"): (
        ["What does `{method} {path}` do in {svc}?", "In {svc}, which endpoint should I call for this operation: {purpose}?", "Which HTTP method and path of {svc} correspond to the operation described as \"{purpose}\"?", "Explain the purpose of `{method} {path}` in {svc}."],
        ["In {svc}, `{method} {path}` is used for: {purpose}.", "{svc} exposes `{method} {path}` for this operation: {purpose}."],
    ),
    ("endpoint", "es"): (
        ["¿Qué hace `{method} {path}` en {svc}?", "En {svc}, ¿qué endpoint debo llamar para esta operación: {purpose}?", "¿Para qué sirve `{method} {path}` en {svc}?", "Explica la función de `{method} {path}` en {svc}."],
        ["En {svc}, `{method} {path}` sirve para {purpose}", "{svc} expone `{method} {path}` para {purpose}"],
    ),
    ("parameter", "en"): (
        ["What is the `{name}` parameter of {operation} in {svc}?", "Describe the `{name}` parameter accepted by {operation} in {svc}.", "In {svc}, which data type does the `{name}` parameter of {operation} have, and is it mandatory?", "Do I have to send `{name}` when calling {operation} in {svc}, and of what type is it?"],
        ["In {svc}, the `{name}` parameter of {operation} is of type {type} and is {required_text}. {description_text}", "{operation} in {svc} takes `{name}` ({type}, {required_text}). {description_text}"],
    ),
    ("parameter", "es"): (
        ["¿Qué es el parámetro `{name}` de {operation} en {svc}?", "Describe el parámetro `{name}` que acepta {operation} en {svc}.", "En {svc}, ¿qué tipo de dato tiene el parámetro `{name}` de {operation} y es obligatorio?", "¿Debo enviar `{name}` al llamar a {operation} en {svc}, y de qué tipo es?"],
        ["En {svc}, el parámetro `{name}` de {operation} es de tipo {type} y es {required_text}. {description_text}", "{operation} en {svc} recibe `{name}`, de tipo {type} y {required_text}. {description_text}"],
    ),
    ("responses", "en"): (
        ["Which HTTP status codes can `{operation}` return in {svc}?", "List the documented response codes of `{operation}` in {svc}.", "In {svc}, what response status codes should a client of `{operation}` be ready to handle?"],
        ["In {svc}, `{operation}` documents these status codes: {codes_text}.", "The documented responses of `{operation}` in {svc} are {codes_text}."],
    ),
    ("responses", "es"): (
        ["¿Qué códigos HTTP puede devolver `{operation}` en {svc}?", "Enumera los códigos de respuesta documentados de `{operation}` en {svc}.", "En {svc}, ¿qué códigos de estado debe estar listo para manejar un cliente de `{operation}`?"],
        ["En {svc}, `{operation}` documenta estos códigos de estado: {codes_text}.", "Las respuestas documentadas de `{operation}` en {svc} son {codes_text}."],
    ),
    ("column", "en"): (
        ["What is the type of column `{column}` in table `{table}` of {svc}?", "How is the `{table}.{column}` column defined in {svc}?", "In {svc}, which data type does `{table}.{column}` use?", "Can `{table}.{column}` be null in {svc}, and what type is it?"],
        ["In {svc}, `{table}.{column}` is {type} ({constraints}).{description_text}", "{svc} defines `{table}.{column}` as {type} ({constraints}).{description_text}"],
    ),
    ("table_columns", "en"): (
        ["Which columns does table `{table}` have in {svc}?", "List the columns of the `{table}` table in {svc}.", "What fields make up the `{table}` table in {svc}?"],
        ["In {svc}, table `{table}` has these columns: {columns_text}.", "The `{table}` table of {svc} is made of {columns_text}."],
    ),
    ("table_columns", "es"): (
        ["¿Qué columnas tiene la tabla `{table}` en {svc}?", "Enumera las columnas de la tabla `{table}` en {svc}.", "¿Qué campos componen la tabla `{table}` en {svc}?"],
        ["En {svc}, la tabla `{table}` tiene estas columnas: {columns_text}.", "La tabla `{table}` de {svc} se compone de {columns_text}."],
    ),
    ("endpoint_list", "en"): (
        ["Which endpoints does {svc} document?", "List the service paths exposed by {svc}.", "What API paths are available in {svc}?"],
        ["{svc} documents these paths: {paths_text}.", "The paths exposed by {svc} are {paths_text}."],
    ),
    ("error_code", "en"): (
        ["What does the error `{error}` mean in {svc}?", "When does {svc} return the `{error}` error?", "Which HTTP status accompanies the `{error}` error in {svc}, and what does it indicate?"],
        ["In {svc}, `{error}` is returned with HTTP {code}: {description}.", "{svc} answers HTTP {code} with `{error}`: {description}."],
    ),
    ("error_cause", "en"): (
        ["What are the common causes of the `{error}` error in {svc}?", "Why would {svc} return `{error}`?", "I am getting `{error}` from {svc}. What usually causes it?"],
        ["In {svc}, common causes of `{error}` are: {causes}.", "`{error}` in {svc} usually comes from: {causes}."],
    ),
    ("http_action", "en"): (
        ["What should I do when {svc} returns HTTP {code}?", "{svc} answered with status {code}. What is the required action?", "According to the {svc} guide, how do I react to a {code} response?"],
        ["When {svc} returns HTTP {code} ({error_type}), the required action is: {action}.", "For HTTP {code} ({error_type}) from {svc}, the guide says: {action}."],
    ),
    ("troubleshooting", "en"): (
        ["How do I fix the issue \"{issue}\" in {svc}?", "{svc} shows the symptom \"{symptoms}\". What is the solution?", "What is the documented remedy for \"{issue}\" when working with {svc}?"],
        ["For \"{issue}\" in {svc} (symptom: {symptoms}), the solution is: {solution}.", "The {svc} guide resolves \"{issue}\" (symptom: {symptoms}) this way: {solution}."],
    ),
    ("alert", "en"): (
        ["When does the alert \"{alert}\" fire in {svc}?", "What does the \"{alert}\" alert indicate in {svc}?", "What situation triggers \"{alert}\" in {svc}?"],
        ["In {svc}, \"{alert}\" means: {description}.", "The \"{alert}\" alert of {svc} signals: {description}."],
    ),
    ("queue", "en"): (
        ["In the {env} environment of {svc}, which topic and path does the BigQ scope `{scope}` use?", "How is BigQ scope `{scope}` wired in {svc} for {env}?", "For {env} in {svc}, what topic feeds BigQ scope `{scope}` and which path consumes it?"],
        ["In {env}, {svc} BigQ scope `{scope}` reads topic `{topic}` through consumer `{consumer}` and calls path `{path}`.", "BigQ scope `{scope}` of {svc} in {env} uses topic `{topic}`, consumer `{consumer}` and path `{path}`."],
    ),
    ("database", "en"): (
        ["Which schema and cluster does {svc} use in {env}?", "Where is the {env} database of {svc}?", "In the {env} environment, what {type} schema and cluster back {svc}?"],
        ["In {env}, {svc} uses {type} schema `{schema}` on cluster `{cluster}`.", "The {env} database of {svc} is {type} schema `{schema}`, on cluster `{cluster}`."],
    ),
    ("cron", "en"): (
        ["What is the schedule of the `{name}` job in {svc}?", "Which endpoint does the `{name}` job call in {svc}?", "When does the `{name}` cron of {svc} run, and which endpoint does it target?"],
        ["In {svc}, the `{name}` job runs with cron rule `{rule}` and calls `{endpoint}`.", "The `{name}` job of {svc} calls `{endpoint}` under cron rule `{rule}`."],
    ),
    ("stack", "en"): (
        ["Which technology does {svc} use for {layer}?", "What is the {layer} layer of {svc} built with?", "In the {svc} technology stack, what fills the {layer} role?"],
        ["{svc} uses {technology} for {layer} ({purpose}).", "For {layer}, {svc} relies on {technology} ({purpose})."],
    ),
    ("exception", "en"): (
        ["What HTTP status does `{operation}` return in {svc} when `{exception}` is raised?", "How does {svc} surface `{exception}` on `{operation}`?", "If `{operation}` hits `{exception}` in {svc}, what status does the caller receive?"],
        ["In {svc}, when `{operation}` raises `{exception}`, the caller receives HTTP {status}.", "{svc} turns `{exception}` on `{operation}` into HTTP {status}."],
    ),
}

# Some spec summaries stay in English inside Spanish services: quoted, they do not mix grammar.
ENGLISH_PURPOSE = re.compile(r"\b(?:the|if|is|of|to|with|from|check|get|returns?)\b", re.IGNORECASE)
SPANISH_MARK = re.compile(r"\b(?:el|la|los|las|de|del|si|para|una?|que)\b", re.IGNORECASE)
QUOTED_ANSWERS = ["En {svc}, la especificación describe `{method} {path}` así: {purpose}.", "La especificación de {svc} resume `{method} {path}` como {purpose}."]

CURATED_KINDS = ("conceptual", "numeric", "procedural", "multi_source")
# Spec prose repeats the subject; inside "sirve para ..." it would read as an orphan clause.
ORPHAN_LEADS = re.compile(r"^(?:este servicio (?:se encarga de|permite)|este endpoint (?:se encarga de|permite)|servicio para|permite|se encarga de)\s+", re.IGNORECASE)


def service_label(fact: dict) -> str:
    if fact["version_scope"] != "version_scoped":
        return fact["project"]
    word = "versión" if fact["language"] == "es" else "version"
    return f"{fact['project']} ({word} {fact['version']})"


def is_english(text: str) -> bool:
    return bool(ENGLISH_PURPOSE.search(text)) and not SPANISH_MARK.search(text)


def _fields(fact: dict) -> Dict[str, object]:
    fields = dict(fact["extra"])
    fields["svc"] = service_label(fact)
    if "required" in fields:
        if fact["language"] == "es":
            fields["required_text"] = "obligatorio" if fields["required"] else "opcional"
        else:
            fields["required_text"] = "required" if fields["required"] else "optional"
    for source, target in (("codes", "codes_text"), ("columns", "columns_text"), ("paths", "paths_text")):
        if source in fields:
            fields[target] = ", ".join(f"`{item}`" for item in fields[source])
    label = "Descripción" if fact["language"] == "es" else "Described as"
    if fact["kind"] == "parameter":
        description = str(fields.get("description") or "").rstrip(".")
        quoted = fact["language"] == "es" and is_english(description)
        fields["description_text"] = (f'{label}: "{description}".' if quoted else f"{label}: {description}.") if description else ""
    if fact["kind"] == "column":
        fields["constraints"] = "nullable" if fields["constraints"] == "null" else fields["constraints"]
        fields["description_text"] = f" {label}: {fields['description']}." if fields.get("description") else ""
    if fact["kind"] == "endpoint" and fact["language"] == "es" and is_english(str(fields["purpose"])):
        fields["purpose"] = '"' + str(fields["purpose"]).rstrip(".") + '"'
    elif fact["kind"] == "endpoint" and fact["language"] == "es":
        purpose = ORPHAN_LEADS.sub("", str(fields["purpose"])).strip()
        fields["purpose"] = purpose[:1].lower() + purpose[1:]
    return fields


def _tidy(text: str) -> str:
    return re.sub(r"\.+(?=[?.])", "", " ".join(text.split())).strip()


def render(fact: dict, seed: int) -> Tuple[List[str], str, str]:
    """Returns (training questions, one unseen question, answer)."""
    if fact["kind"] in CURATED_KINDS:
        return list(fact["extra"]["train_questions"]), fact["extra"]["eval_question"], fact["value"]
    questions, answers = TEMPLATES[(fact["kind"], fact["language"])]
    fields = _fields(fact)
    if fact["kind"] == "endpoint" and fact["language"] == "es" and str(fields["purpose"]).startswith('"'):
        answers = QUOTED_ANSWERS
    ordered = sorted(questions, key=lambda wording: rank(seed, "question", fact["fact_id"], wording))
    answer = sorted(answers, key=lambda wording: rank(seed, "answer", fact["fact_id"], wording))[0]
    return [_tidy(q.format(**fields)) for q in ordered[:-1]], _tidy(ordered[-1].format(**fields)), _tidy(answer.format(**fields))


def difficulty_of(fact: dict) -> str:
    """hard: needs 2+ passages. medium: 3+ key facts, or curated with 2+ quotes. easy: one lookup."""
    if fact["kind"] == "multi_source":
        return "hard"
    if len(fact["key_facts"]) >= 3 or len(fact["extra"].get("evidence_quotes", [])) >= 2:
        return "medium"
    return "easy"
