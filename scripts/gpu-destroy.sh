#!/usr/bin/env bash
# Terminate the GPU instance between work phases (D-023): deletes the instance, its root
# volume (engines, container image, data copy) and its idle alarm. Keeps the security group
# and key pair, which cost nothing, for the next gpu-create.sh. Reports are already on the
# Mac (remote.sh pulls them after every run).
# Only prints what it would do unless run with --yes.
set -euo pipefail
source "$(dirname "$0")/gpu-config.sh"

id=$(require_instance)
echo "will terminate $id ($(instance_field "$id" InstanceType), $(instance_field "$id" State.Name)) and delete alarm $GPU_ALARM_NAME"
if [ "${1:-}" != "--yes" ]; then
    echo "dry run: nothing deleted. Re-run with --yes to terminate."
    exit 0
fi
"$(dirname "$0")/remote.sh" pull-reports 2>/dev/null || echo "note: instance not reachable; reports not re-pulled"
aws ec2 terminate-instances --instance-ids "$id" >/dev/null
echo "terminating $id ..."
aws ec2 wait instance-terminated --instance-ids "$id"
log_session down "$id"
aws cloudwatch delete-alarms --alarm-names "$GPU_ALARM_NAME"
revoke_all_ssh
echo "terminated $id; deleted alarm $GPU_ALARM_NAME"
