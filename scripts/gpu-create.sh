#!/usr/bin/env bash
# One-time setup of the GPU instance (D-023). Creates billable resources, so it only
# prints the plan unless run with --yes. Needs credentials that may create EC2 resources
# and CloudWatch alarms (run `aws login` first).
#
# Creates: an ed25519 SSH key (~/.ssh, imported as an EC2 key pair), a security group
# with no inbound rules (gpu-up.sh opens SSH to the current IP only), the instance from
# the latest Deep Learning Base GPU AMI, and the idle auto-stop alarm.
set -euo pipefail
source "$(dirname "$0")/gpu-config.sh"

apply=0
[ "${1:-}" = "--yes" ] && apply=1

[ -z "$(instance_id)" ] || die "an instance tagged Name=$GPU_NAME already exists ($(instance_id))"
quota=$(aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA \
    --query 'Quota.Value' --output text)
python3 -c "import sys; sys.exit(float('$quota') < 4)" \
    || die "G and VT on-demand vCPU quota is $quota; g4dn.xlarge needs 4"
ami=$(aws ssm get-parameter --name "$GPU_AMI_PARAM" --query Parameter.Value --output text)
ami_name=$(aws ec2 describe-images --image-ids "$ami" --query 'Images[0].Name' --output text)
vpc=$(aws ec2 describe-vpcs --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)
[ "$vpc" != None ] || die "no default VPC in $AWS_REGION"

cat <<EOF
Plan ($AWS_REGION):
  ssh key        $GPU_SSH_KEY $([ -f "$GPU_SSH_KEY" ] && echo "(exists)" || echo "(new, ed25519)") -> EC2 key pair $GPU_KEY_NAME
  security group $GPU_SG_NAME in $vpc, no inbound rules until gpu-up.sh
  instance       $GPU_TYPE, $ami ($ami_name)
                 root gp3 ${GPU_ROOT_GB} GB (deleted with the instance), IMDSv2 only,
                 tags Name=$GPU_NAME Project=edgeinfer
  alarm          $GPU_ALARM_NAME: stop when CPU < $GPU_IDLE_CPU_PERCENT% for $((GPU_IDLE_PERIODS * 5)) min
Costs: the instance bills per second while running; the root volume bills while the
instance exists, running or stopped.
EOF
if [ "$apply" -ne 1 ]; then
    echo "dry run: nothing created. Re-run with --yes to create."
    exit 0
fi

if [ ! -f "$GPU_SSH_KEY" ]; then
    ssh-keygen -t ed25519 -N "" -C "$GPU_NAME" -f "$GPU_SSH_KEY" >/dev/null
    echo "created $GPU_SSH_KEY"
fi
if ! aws ec2 describe-key-pairs --key-names "$GPU_KEY_NAME" >/dev/null 2>&1; then
    aws ec2 import-key-pair --key-name "$GPU_KEY_NAME" \
        --public-key-material "fileb://$GPU_SSH_KEY.pub" \
        --tag-specifications "ResourceType=key-pair,Tags=[{Key=Project,Value=edgeinfer}]" >/dev/null
    echo "imported key pair $GPU_KEY_NAME"
fi

sg=$(security_group_id)
if [ -z "$sg" ]; then
    sg=$(aws ec2 create-security-group --group-name "$GPU_SG_NAME" --vpc-id "$vpc" \
        --description "edgeinfer GPU: SSH from the current IP only (gpu-up.sh)" \
        --tag-specifications "ResourceType=security-group,Tags=[{Key=Project,Value=edgeinfer}]" \
        --query GroupId --output text)
    echo "created security group $sg"
fi

user_data=$(printf '#!/bin/bash\nmkdir -p %s\nchown -R %s:%s /data\n' \
    "$GPU_DATA_DIR" "$GPU_SSH_USER" "$GPU_SSH_USER")
id=$(aws ec2 run-instances --image-id "$ami" --instance-type "$GPU_TYPE" \
    --key-name "$GPU_KEY_NAME" --security-group-ids "$sg" \
    --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$GPU_ROOT_GB,VolumeType=gp3,DeleteOnTermination=true}" \
    --metadata-options HttpTokens=required,HttpEndpoint=enabled \
    --instance-initiated-shutdown-behavior stop \
    --user-data "$user_data" \
    --tag-specifications \
    "ResourceType=instance,Tags=[{Key=Name,Value=$GPU_NAME},{Key=Project,Value=edgeinfer}]" \
    "ResourceType=volume,Tags=[{Key=Name,Value=$GPU_NAME},{Key=Project,Value=edgeinfer}]" \
    --query 'Instances[0].InstanceId' --output text)
echo "launched $id; waiting until running ..."
aws ec2 wait instance-running --instance-ids "$id"
log_session up "$id"

aws cloudwatch put-metric-alarm --alarm-name "$GPU_ALARM_NAME" \
    --alarm-description "Stop $GPU_NAME when idle (edgeinfer D-023)" \
    --namespace AWS/EC2 --metric-name CPUUtilization --dimensions "Name=InstanceId,Value=$id" \
    --statistic Average --period 300 --evaluation-periods "$GPU_IDLE_PERIODS" \
    --threshold "$GPU_IDLE_CPU_PERCENT" --comparison-operator LessThanThreshold \
    --treat-missing-data notBreaching \
    --alarm-actions "arn:aws:automate:$AWS_REGION:ec2:stop"
echo "created alarm $GPU_ALARM_NAME"

exec "$(dirname "$0")/gpu-up.sh"
