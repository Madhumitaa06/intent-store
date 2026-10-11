#!/usr/bin/env bash
set -u

cd "$(dirname "$0")"

echo "Checking IntentStore..."

if curl -fsS http://127.0.0.1:5000/ >/dev/null 2>&1; then
    echo "IntentStore is already running."
    echo "Open port 5000 in the Codespaces Ports tab."
    exit 0
fi

echo "Starting IntentStore in the background..."
nohup python -u frontend/app.py > /tmp/intentstore.log 2>&1 &
PID=$!
echo "$PID" > /tmp/intentstore.pid

for i in $(seq 1 120); do
    if curl -fsS http://127.0.0.1:5000/ >/dev/null 2>&1; then
        echo
        echo "SUCCESS: IntentStore is running."
        echo "Open port 5000 in the Codespaces Ports tab."
        echo "Server logs: /tmp/intentstore.log"
        exit 0
    fi

    if ! kill -0 "$PID" 2>/dev/null; then
        echo "Startup failed. Recent server logs:"
        tail -80 /tmp/intentstore.log
        exit 1
    fi

    sleep 2
done

echo "Startup timed out. Recent server logs:"
tail -80 /tmp/intentstore.log
exit 1
