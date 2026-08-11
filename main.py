"""Brain HQ application entrypoint."""

from bess.webapp import app


if __name__ == "__main__":
    app.run(debug=True)
