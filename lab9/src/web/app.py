from flask import Flask, render_template, request, jsonify

from src.domain.models import Pattern
from src.services.hopfield import HopfieldNetwork

app = Flask(__name__)


# ---------- ОБРАЗЫ: БУКВЫ М, Е, В (10x10) ----------

M = [
    [1,1,1,0,0,0,0,0,0,1],
    [1,1,1,1,0,0,0,0,1,1],
    [1,0,1,1,1,0,0,1,0,1],
    [1,0,0,1,1,1,1,0,0,1],
    [1,0,0,0,1,1,0,0,0,1],
    [1,0,0,0,1,1,0,0,0,1],
    [1,0,0,1,1,1,1,0,0,1],
    [1,0,1,1,0,0,1,1,0,1],
    [1,1,1,0,0,0,0,1,1,1],
    [1,1,1,0,0,0,0,0,0,1],
]

E = [
    [1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1],
]

V = [
    [1,1,1,1,1,1,1,1,1,0],
    [1,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,0],
    [1,1,1,1,1,1,1,1,1,0],
    [1,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,0],
    [1,1,1,1,1,1,1,1,1,0],
    [1,0,0,0,0,0,0,0,0,1],
]

patterns = [
    Pattern("М", M),
    Pattern("Е", E),
    Pattern("В", V),
]

network = HopfieldNetwork(patterns)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/patterns")
def get_patterns():
    """Возвращает эталонные образы."""
    return jsonify({
        "patterns": [
            {"name": p.name, "matrix": p.matrix} for p in patterns
        ],
        "size": [len(patterns[0].matrix), len(patterns[0].matrix[0])]
    })


@app.route("/api/noisy", methods=["POST"])
def get_noisy():
    """Генерирует зашумлённый образ."""
    data = request.json or {}
    pattern_name = data.get("pattern", "М")
    noise = int(data.get("noise", 20))
    seed = int(data.get("seed", 42))

    pattern = next((p for p in patterns if p.name == pattern_name), None)
    if not pattern:
        return jsonify({"error": "Образ не найден"}), 404

    import random
    rng = random.Random(seed)
    noisy = network.add_noise(pattern.matrix, noise, rng)
    restored, _ = network.recall_matrix(noisy)

    return jsonify({
        "noisy": noisy,
        "restored": restored,
        "original": pattern.matrix,
    })


@app.route("/api/test", methods=["POST"])
def run_tests():
    """Прогоняет серию тестов по всем буквам и уровням зашумления."""
    data = request.json or {}
    seed = int(data.get("seed", 42))
    trials = int(data.get("trials", 10))

    noise_levels = [10, 20, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    results = network.run_tests(noise_levels, trials_per_level=trials, seed=seed)

    # Агрегируем по проценту зашумления (средняя точность по всем буквам)
    aggregated = {}
    for r in results:
        aggregated.setdefault(r.noise_percent, []).append(r.accuracy)

    summary = [
        {
            "noise": noise,
            "avg_accuracy": sum(accs) / len(accs),
            "per_pattern": {
                r.pattern_name: r.accuracy
                for r in results if r.noise_percent == noise
            }
        }
        for noise, accs in sorted(aggregated.items())
    ]

    return jsonify({
        "noise_levels": noise_levels,
        "trials_per_level": trials,
        "results": [
            {
                "noise": r.noise_percent,
                "pattern": r.pattern_name,
                "success": r.success,
                "trials": r.trials,
                "accuracy": r.accuracy,
            }
            for r in results
        ],
        "summary": summary
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)