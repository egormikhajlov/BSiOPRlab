from flask import Flask, render_template, request, jsonify

from src.domain.models import Alternative, Criterion
from src.services.electre import ElectreDecisionMaker

app = Flask(__name__)


# ---------- ИСХОДНЫЕ ДАННЫЕ (ВАРИАНТ 9) ----------

criteria = [
    Criterion("prod", "Производительность (изд./час)", weight=0.50, is_maximize=True),
    Criterion("cost", "Стоимость (тыс. у.е.)",          weight=0.25, is_maximize=False),
    Criterion("rel",  "Надёжность (шкала Харрингтона)", weight=0.25, is_maximize=True),
]

alternatives = [
    Alternative("CT1", "СТ1", {"prod": 25, "cost": 140, "rel": 0.75}),
    Alternative("CT2", "СТ2", {"prod": 25, "cost": 100, "rel": 0.50}),
    Alternative("CT3", "СТ3", {"prod": 30, "cost": 200, "rel": 0.90}),
    Alternative("CT4", "СТ4", {"prod": 15, "cost": 100, "rel": 0.65}),
    Alternative("CT5", "СТ5", {"prod": 20, "cost": 100, "rel": 0.50}),
    Alternative("CT6", "СТ6", {"prod": 35, "cost": 200, "rel": 0.75}),
]

weights = {c.id: c.weight for c in criteria}

decision_maker = ElectreDecisionMaker(alternatives, criteria)


# ---------- ROUTES ----------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/calculate", methods=["POST"])
def calculate():
    data = request.json or {}

    c_thr = float(data.get("c_threshold", 0.5))
    d_thr = float(data.get("d_threshold", 0.5))
    use_pareto = bool(data.get("use_pareto", True))

    # Шаг 1: Парето
    pareto_set = decision_maker.get_pareto_optimal_set()

    # Шаг 2: Работаем либо по Парето, либо по всем альтернативам
    active = pareto_set if use_pareto else alternatives

    # Шаг 3: ЭЛЕКТРА
    result = decision_maker.apply_electre(
        active_alts=active,
        weights=weights,
        c_threshold=c_thr,
        d_threshold=d_thr,
    )

    return jsonify({
        "pareto_set": [a.name for a in pareto_set],
        "weights": weights,
        "criteria": [
            {"id": c.id, "name": c.name, "is_maximize": c.is_maximize}
            for c in criteria
        ],
        "electre": result,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)