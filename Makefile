run:      ; uvicorn app.main:app --reload
test:     ; python -m unittest discover -s tests -v
up:       ; docker compose up --build
simulate: ; python scripts/simulate.py
