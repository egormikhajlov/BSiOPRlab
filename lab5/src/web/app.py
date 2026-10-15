from flask import Flask, render_template, request, jsonify
from src.domain.models import SimulationConfig
from src.services.monte_carlo import MonteCarloSimulator

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/simulate", methods=["POST"])
def simulate():
    data = request.json or {}

    try:
        config = SimulationConfig(
            n=int(data.get("n", 10000)),
            seed=int(data.get("seed", 0xACE1), 0),  # 0 -> автопрефикс (0x...)
            a_percent=float(data.get("a_percent", 25.0)),
            cost_control=float(data.get("cost_control", 2.0)),
            cost_fix=float(data.get("cost_fix", 65.0)),
            cost_replace=float(data.get("cost_replace", 110.0)),
            step=int(data.get("step", 5))
        )
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Некорректные параметры: {e}"}), 400

    if config.n <= 0:
        return jsonify({"error": "Количество испытаний должно быть положительным."}), 400
    if config.seed == 0:
        return jsonify({"error": "Seed LFSR не может быть нулевым."}), 400
    if not (0 < config.step <= 100):
        return jsonify({"error": "Шаг должен быть в диапазоне 1..100."}), 400

    simulator = MonteCarloSimulator(config)
    result = simulator.run()

    return jsonify({
        "config": {
            "n": config.n,
            "seed": hex(config.seed),
            "a_percent": config.a_percent,
            "cost_control": config.cost_control,
            "cost_fix": config.cost_fix,
            "cost_replace": config.cost_replace,
            "step": config.step
        },
        "points": [
            {
                "control_percent": p.control_percent,
                "total_cost": round(p.total_cost, 2),
                "cost_per_party": round(p.cost_per_party, 4)
            }
            for p in result.points
        ],
        "best_percent": result.best_percent,
        "best_cost": round(result.best_cost, 2)
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)