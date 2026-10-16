# Shared settings and helpers for the gpu-*.sh and remote.sh scripts. Source, don't run.
# Every setting can be overridden from the environment. See D-023.

: "${AWS_REGION:=us-east-1}"
: "${GPU_NAME:=edgeinfer-gpu}"            # Name tag; the scripts find the instance by it
: "${GPU_TYPE:=g4dn.xlarge}"
: "${GPU_ROOT_GB:=100}"                   # gp3 root volume; billed while the instance exists
: "${GPU_SG_NAME:=edgeinfer-gpu-ssh}"
: "${GPU_KEY_NAME:=edgeinfer-gpu}"
: "${GPU_SSH_KEY:=$HOME/.ssh/edgeinfer-gpu}"
: "${GPU_SSH_USER:=ubuntu}"
: "${GPU_REMOTE_DIR:=edgeinfer}"          # relative to the remote home directory
: "${GPU_DATA_DIR:=/data/cityscapes}"
# Idle auto-stop: stop the instance when average CPU stays below the threshold for
# GPU_IDLE_PERIODS consecutive 5-minute periods.
: "${GPU_IDLE_CPU_PERCENT:=5}"
: "${GPU_IDLE_PERIODS:=6}"
: "${GPU_AMI_PARAM:=/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id}"
export AWS_REGION

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_LOG="$REPO_ROOT/private/gpu-sessions.log"
GPU_ALARM_NAME="$GPU_NAME-idle-stop"

die() {
    echo "error: $*" >&2
    exit 1
}

# The one non-terminated instance tagged Name=$GPU_NAME; empty if there is none.
instance_id() {
    local ids
    ids=$(aws ec2 describe-instances \
        --filters "Name=tag:Name,Values=$GPU_NAME" \
        "Name=instance-state-name,Values=pending,running,stopping,stopped" \
        --query 'Reservations[].Instances[].InstanceId' --output text)
    if [ "$(wc -w <<<"$ids")" -gt 1 ]; then
        die "more than one instance tagged Name=$GPU_NAME: $ids"
    fi
    echo "$ids"
}

require_instance() {
    local id
    id=$(instance_id)
    [ -n "$id" ] || die "no instance tagged Name=$GPU_NAME in $AWS_REGION (scripts/gpu-create.sh makes one)"
    echo "$id"
}

instance_field() {  # instance_field <id> <JMESPath under the instance>
    aws ec2 describe-instances --instance-ids "$1" \
        --query "Reservations[0].Instances[0].$2" --output text
}

security_group_id() {
    aws ec2 describe-security-groups --filters "Name=group-name,Values=$GPU_SG_NAME" \
        --query 'SecurityGroups[0].GroupId' --output text | sed 's/^None$//'
}

my_ip() {
    curl -sf https://checkip.amazonaws.com | tr -d '[:space:]'
}

# Make the security group allow SSH from this machine's current IP only.
allow_ssh_from_here() {
    local sg ip old
    sg=$(security_group_id)
    [ -n "$sg" ] || die "security group $GPU_SG_NAME not found"
    ip=$(my_ip)
    [ -n "$ip" ] || die "could not determine this machine's public IP"
    old=$(aws ec2 describe-security-groups --group-ids "$sg" \
        --query 'SecurityGroups[0].IpPermissions[?FromPort==`22`].IpRanges[].CidrIp' --output text)
    for cidr in $old; do
        if [ "$cidr" != "$ip/32" ]; then
            aws ec2 revoke-security-group-ingress --group-id "$sg" --protocol tcp --port 22 \
                --cidr "$cidr" >/dev/null
            echo "ssh: revoked $cidr"
        fi
    done
    if ! grep -qw "$ip/32" <<<"$old"; then
        aws ec2 authorize-security-group-ingress --group-id "$sg" --protocol tcp --port 22 \
            --cidr "$ip/32" >/dev/null
        echo "ssh: allowed $ip/32"
    fi
}

revoke_all_ssh() {
    local sg old
    sg=$(security_group_id)
    [ -n "$sg" ] || return 0
    old=$(aws ec2 describe-security-groups --group-ids "$sg" \
        --query 'SecurityGroups[0].IpPermissions[?FromPort==`22`].IpRanges[].CidrIp' --output text)
    for cidr in $old; do
        aws ec2 revoke-security-group-ingress --group-id "$sg" --protocol tcp --port 22 \
            --cidr "$cidr" >/dev/null
        echo "ssh: revoked $cidr"
    done
}

ssh_host() {
    local id dns
    id=$(require_instance)
    [ "$(instance_field "$id" State.Name)" = running ] || die "instance $id is not running (scripts/gpu-up.sh)"
    dns=$(instance_field "$id" PublicDnsName)
    [ -n "$dns" ] && [ "$dns" != None ] || die "instance $id has no public DNS name"
    echo "$GPU_SSH_USER@$dns"
}

# The public DNS name changes on every start, so the host key is stored under a fixed alias:
# the first connection records it, later ones are checked against it.
SSH_OPTS=(-i "$GPU_SSH_KEY" -o IdentitiesOnly=yes -o ConnectTimeout=10
    -o HostKeyAlias="$GPU_NAME" -o StrictHostKeyChecking=accept-new
    -o UserKnownHostsFile="$HOME/.ssh/known_hosts_edgeinfer")

log_session() {  # log_session <up|down> <instance id>
    mkdir -p "$(dirname "$SESSION_LOG")"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $1 $2" >>"$SESSION_LOG"
}
