from flask import Flask, render_template, request

app = Flask(__name__)


MAPS = {
    "density": {
        "low": 0.2,
        "medium": 0.5,
        "high": 0.9
    },
    "speed": {
        "calm": 0.2,
        "rushed": 0.6,
        "fast": 0.9
    },
    "direction": {
        "organized": 0.1,
        "mixed": 0.5,
        "chaotic": 0.9
    },
    "threat": {
        "none": 0.0,
        "suspected": 0.5,
        "confirmed": 1.0
    },
    "exit": {
        "clear": 0.0,
        "partial": 0.5,
        "blocked": 1.0
    },
    "noise": {
        "normal": 0.2,
        "shouting": 0.6,
        "screaming": 1.0
    }
}


def manual_score(data):
    score = (
        0.25 * data["density"] +
        0.20 * data["speed"] +
        0.20 * data["direction"] +
        0.15 * data["threat"] +
        0.10 * data["exit"] +
        0.10 * data["noise"]
    ) * 100

    return round(score, 2)


def risk_level(score):
    if score <= 30:
        return "Safe"

    if score <= 60:
        return "Alert"

    if score <= 80:
        return "High Risk"

    return "Panic"


def explain_manual(data):
    reasons = []

    if data["density"] > 0.7:
        reasons.append("High crowd density is increasing the risk.")

    if data["speed"] > 0.7:
        reasons.append("Fast movement suggests possible rushing behavior.")

    if data["direction"] > 0.7:
        reasons.append("Chaotic direction indicates unstable crowd flow.")

    if data["threat"] > 0.7:
        reasons.append("Confirmed threat is a major panic trigger.")

    if data["exit"] > 0.7:
        reasons.append("Blocked exit can create dangerous bottlenecks.")

    if data["noise"] > 0.7:
        reasons.append("Screaming or high noise may indicate fear or distress.")

    if not reasons:
        reasons.append("Crowd conditions appear mostly stable.")

    return reasons


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        values = {
            "density": MAPS["density"][request.form["density"]],
            "speed": MAPS["speed"][request.form["speed"]],
            "direction": MAPS["direction"][request.form["direction"]],
            "threat": MAPS["threat"][request.form["threat"]],
            "exit": MAPS["exit"][request.form["exit"]],
            "noise": MAPS["noise"][request.form["noise"]]
        }

        score = manual_score(values)

        result = {
            "score": score,
            "level": risk_level(score),
            "reasons": explain_manual(values)
        }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)