#!/usr/bin/env bash
# Start the GPU instance, allow SSH from this machine only, and wait until SSH works.
# Billing starts here: end every session with scripts/gpu-down.sh.
set -euo pipefail
source "$(dirname "$0")/gpu-config.sh"

id=$(require_instance)
state=$(instance_field "$id" State.Name)
case "$state" in
    running) echo "instance $id is already running" ;;
    stopped)
        aws ec2 start-instances --instance-ids "$id" >/dev/null
        echo "starting $id ..."
        aws ec2 wait instance-running --instance-ids "$id"
        log_session up "$id"
        ;;
    *) die "instance $id is $state; wait for it to settle and retry" ;;
esac

allow_ssh_from_here
host=$(ssh_host)
echo "waiting for ssh on $host ..."
for _ in $(seq 1 30); do
    if ssh "${SSH_OPTS[@]}" -o BatchMode=yes "$host" true 2>/dev/null; then
        echo "ready: ssh ${SSH_OPTS[*]} $host"
        echo "REMINDER: the instance is billing. Stop it with scripts/gpu-down.sh."
        exit 0
    fi
    sleep 10
done
die "ssh to $host did not come up within 5 minutes (instance is still running; scripts/gpu-down.sh stops it)"
