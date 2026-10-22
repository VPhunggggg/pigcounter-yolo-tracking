#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip >/dev/null
pip install -r requirements.txt

mkdir -p uploads outputs

export PYTHONUNBUFFERED=1
python app.py &
SERVER_PID=$!

echo "Đang chờ server khởi động..."
for i in $(seq 1 120); do
    if curl -s http://127.0.0.1:5000/api/health >/dev/null 2>&1; then
        echo "Server sẵn sàng tại http://127.0.0.1:5000"
        break
    fi
    sleep 1
done

if command -v xdg-open >/dev/null; then xdg-open http://127.0.0.1:5000; fi
if command -v open >/dev/null; then open http://127.0.0.1:5000; fi

wait $SERVER_PID
