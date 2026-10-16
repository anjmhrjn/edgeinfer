#!/usr/bin/env bash
# Show the GPU instance, its SSH rule and its idle auto-stop alarm. Read-only.
set -euo pipefail
source "$(dirname "$0")/gpu-config.sh"

id=$(instance_id)
if [ -z "$id" ]; then
    echo "instance: none tagged Name=$GPU_NAME in $AWS_REGION"
    exit 0
fi
aws ec2 describe-instances --instance-ids "$id" --query 'Reservations[0].Instances[0].{
    id: InstanceId, type: InstanceType, state: State.Name, launched: LaunchTime,
    ami: ImageId, public_dns: PublicDnsName}' --output table

sg=$(security_group_id)
if [ -n "$sg" ]; then
    rules=$(aws ec2 describe-security-groups --group-ids "$sg" \
        --query 'SecurityGroups[0].IpPermissions[?FromPort==`22`].IpRanges[].CidrIp' --output text)
    echo "ssh allowed from: ${rules:-nobody}  (this machine: $(my_ip)/32)"
fi

alarm=$(aws cloudwatch describe-alarms --alarm-names "$GPU_ALARM_NAME" \
    --query 'MetricAlarms[0].[StateValue,Threshold,EvaluationPeriods,Period]' --output text)
if [ -z "$alarm" ] || [ "$alarm" = None ]; then
    echo "idle auto-stop alarm: MISSING ($GPU_ALARM_NAME); don't leave jobs unattended"
else
    read -r state threshold periods period <<<"$alarm"
    echo "idle auto-stop alarm: $state (stop after $((periods * period / 60)) min below $threshold% CPU)"
fi

if [ -f "$SESSION_LOG" ]; then
    echo "last session events:"
    tail -n 4 "$SESSION_LOG" | sed 's/^/  /'
fi
