from flask import Flask, render_template, request, jsonify

from src.domain.models import DataObject
from src.services.recognition import RecognitionSystem

app = Flask(__name__)


# ---------- ОБУЧАЮЩАЯ ВЫБОРКА (10 ВИН) ----------
# Признаки: [Крепость (%), Сахар (г/л), Кислотность (г/л), Танинность (0-10), Темп. подачи (°C)]

training_data = [
    # Класс 1: Красные вина
    DataObject("1",  [13.5, 2.5, 6.0, 8.5, 18], 1, "Каберне Совиньон"),
    DataObject("2",  [13.0, 3.0, 5.5, 7.0, 18], 1, "Мерло"),
    DataObject("3",  [12.5, 2.0, 5.8, 6.5, 16], 1, "Пино Нуар"),
    DataObject("4",  [14.0, 2.8, 6.2, 8.0, 18], 1, "Сира"),
    DataObject("5",  [13.8, 3.2, 5.9, 7.5, 17], 1, "Мальбек"),
    # Класс 2: Белые вина
    DataObject("6",  [12.5, 4.5, 6.5, 1.5, 10], 2, "Шардоне"),
    DataObject("7",  [12.0, 3.0, 7.0, 1.0,  9], 2, "Совиньон Блан"),
    DataObject("8",  [11.5, 8.0, 7.5, 0.8,  8], 2, "Рислинг"),
    DataObject("9",  [12.2, 4.0, 6.8, 1.2, 10], 2, "Пино Гриджио"),
    DataObject("10", [12.8, 6.5, 6.6, 1.0,  9], 2, "Гевюрцтраминер"),
]

recognition_system = RecognitionSystem(training_data)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/info")
def info():
    """Возвращает информацию об обучающей выборке и обученной модели."""
    return jsonify({
        "feature_names": [
            "Крепость (%)",
            "Сахар (г/л)",
            "Кислотность (г/л)",
            "Танинность (0-10)",
            "Темп. подачи (°C)"
        ],
        "training_data": [
            {
                "id": obj.id,
                "name": obj.name,
                "features": obj.features,
                "class_label": obj.class_label
            }
            for obj in training_data
        ],
        "prototypes": recognition_system.get_prototypes_info(),
        "training": recognition_system.get_training_info(),
    })


@app.route("/api/recognize", methods=["POST"])
def recognize():
    data = request.json or {}
    try:
        features = [
            float(data.get("f1", 0)),
            float(data.get("f2", 0)),
            float(data.get("f3", 0)),
            float(data.get("f4", 0)),
            float(data.get("f5", 0)),
        ]
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Некорректные данные: {e}"}), 400

    features_norm = recognition_system.normalize_single(features)

    res_min = recognition_system.predict_min_distance(features_norm)
    res_per = recognition_system.predict_perceptron(features_norm)

    return jsonify({
        "features_norm": [round(f, 4) for f in features_norm],
        "min_distance": res_min,
        "perceptron": res_per,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)