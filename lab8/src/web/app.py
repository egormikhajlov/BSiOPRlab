from flask import Flask, render_template, request, jsonify

from src.domain.models import DataObject
from src.services.clustering import ClusteringSystem

app = Flask(__name__)


# ---------- ДАННЫЕ: 12 ГОРОДОВ ----------
# Признаки: [Население (тыс.), Зарплата (тыс. ₽), Площадь (км²)]

cities = [
    DataObject("1",  [1625, 55, 502],  "Новосибирск"),
    DataObject("2",  [1495, 58, 1112], "Екатеринбург"),
    DataObject("3",  [1257, 52, 615],  "Казань"),
    DataObject("4",  [1244, 50, 410],  "Нижний Новгород"),
    DataObject("5",  [1187, 48, 530],  "Челябинск"),
    DataObject("6",  [1156, 47, 541],  "Самара"),
    DataObject("7",  [1172, 45, 573],  "Омск"),
    DataObject("8",  [1137, 46, 348],  "Ростов-на-Дону"),
    DataObject("9",  [1128, 49, 707],  "Уфа"),
    DataObject("10", [1093, 54, 348],  "Красноярск"),
    DataObject("11", [1055, 47, 800],  "Пермь"),
    DataObject("12", [1004, 44, 859],  "Волгоград"),
]

clustering_system = ClusteringSystem(cities)


def _prepare_response(clusters, prototypes):
    """Форматирует данные для отправки в веб-интерфейс."""
    result = []
    for c_id, objects in clusters.items():
        result.append({
            "cluster_id": c_id + 1,
            "prototype": [round(p, 2) for p in prototypes[c_id]],
            "objects": [
                {"name": obj.name, "features": obj.features}
                for obj in objects
            ]
        })
    return result


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def data():
    """Возвращает исходные данные."""
    return jsonify({
        "feature_names": [
            "Население (тыс. чел.)",
            "Средняя зарплата (тыс. ₽)",
            "Площадь (км²)"
        ],
        "objects": [
            {"name": c.name, "features": c.features} for c in cities
        ]
    })


@app.route("/api/cluster", methods=["POST"])
def cluster():
    data = request.json or {}
    method = data.get("method", "kmeans")

    if method == "kmeans":
        try:
            k = int(data.get("k", 3))
        except (ValueError, TypeError):
            return jsonify({"error": "Некорректное K"}), 400
        if k < 2 or k > len(cities):
            return jsonify({"error": f"K должно быть от 2 до {len(cities)}"}), 400

        clusters, prototypes, history = clustering_system.kmeans(k)
        return jsonify({
            "method": "kmeans",
            "k": k,
            "clusters": _prepare_response(clusters, prototypes),
            "history": history,
        })

    elif method == "maximin":
        clusters, prototypes, history = clustering_system.maximin()
        return jsonify({
            "method": "maximin",
            "k": len(prototypes),
            "clusters": _prepare_response(clusters, prototypes),
            "history": history,
        })

    else:
        return jsonify({"error": "Unknown method"}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)