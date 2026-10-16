#!/usr/bin/env bash
# Stop the GPU instance, close SSH and print how long the session ran.
# Stopping ends compute billing; the root volume is still billed until the instance is
# terminated (D-023).
set -euo pipefail
source "$(dirname "$0")/gpu-config.sh"

id=$(require_instance)
state=$(instance_field "$id" State.Name)
if [ "$state" = stopped ]; then
    echo "instance $id is already stopped"
else
    aws ec2 stop-instances --instance-ids "$id" >/dev/null
    echo "stopping $id ..."
    aws ec2 wait instance-stopped --instance-ids "$id"
    log_session down "$id"
fi
revoke_all_ssh
echo "instance $id: $(instance_field "$id" State.Name)"

# Duration of the last up -> down pair in the session log.
if [ -f "$SESSION_LOG" ]; then
    python3 - "$SESSION_LOG" <<'EOF'
import sys
from datetime import datetime
events = [line.split() for line in open(sys.argv[1]) if line.strip()]
if len(events) >= 2 and events[-1][1] == "down" and events[-2][1] == "up":
    t = [datetime.strptime(e[0], "%Y-%m-%dT%H:%M:%SZ") for e in events[-2:]]
    print(f"session: {events[-2][0]} -> {events[-1][0]} ({(t[1] - t[0]).total_seconds() / 3600:.2f} h)")
EOF
fi
echo "log the session in private/gpu-log.md"
