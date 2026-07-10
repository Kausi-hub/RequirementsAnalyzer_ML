from flask import Flask, render_template, request, jsonify, session, send_file
import os
from datetime import datetime
import markdown2
import logging

from requirement_engine import analyze_requirements

app = Flask(__name__)
app.secret_key = os.urandom(24)

logging.basicConfig(level=logging.INFO)

os.makedirs("logs", exist_ok=True)

@app.route("/")
def home():
    session["conversation_id"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    return render_template("index_req.html")


@app.route("/reason", methods=["POST"])
def reason():
    try:
        data = request.json

        requirements = data.get("requirements", [])
        threshold = float(data.get("threshold", 0.80))

        if not requirements:
            return jsonify({"error": "No requirements supplied"}), 400

        result = analyze_requirements(requirements)

        conversation_id = session.get(
            "conversation_id",
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )

        log_file = f"logs/conversation_{conversation_id}.md"

        with open(log_file, "w", encoding="utf-8") as f:
            f.write("# Requirement Analysis\n\n")

            for r in requirements:
                f.write(f"- {r}\n")

            f.write(f"\n## Satisfaction Score\n{result['satisfaction']}\n\n")

            for category, items in result["issues"].items():
                f.write(f"### {category}\n")
                for item in items:
                    f.write(f"- {item}\n")

        with open(log_file, "r", encoding="utf-8") as f:
            html_content = markdown2.markdown(f.read())

        return jsonify({
            "issues": result["issues"],
            "satisfaction_level": result["satisfaction"],
            "satisfactory": result["satisfaction"] >= threshold,
            "log_content": html_content,
            "log_file": log_file
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download-log")
def download_log():
    file = request.args.get("file")

    if not file or not os.path.exists(file):
        return "File not found", 404

    return send_file(file, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, port=5050)