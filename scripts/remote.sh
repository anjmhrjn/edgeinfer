#!/usr/bin/env bash
# Run work on the GPU instance (D-023). The instance must be running (scripts/gpu-up.sh).
#
#   remote.sh run "<cmd>"            sync the repo, build the image if its inputs changed,
#                                    run <cmd> in the TensorRT container, pull reports back
#   remote.sh host "<cmd>"           run <cmd> on the instance itself (nvidia-smi, df -h, ...)
#   remote.sh push-data              copy the Cityscapes val split to $GPU_DATA_DIR
#   remote.sh push-artifacts <path>.. copy files or directories under artifacts/ (ONNX models
#                                    with their sidecars, golden outputs)
#   remote.sh pull-reports           copy artifacts/reports/ back to the Mac
#   remote.sh sha256 <path>          print the sha256 of a file under artifacts/ on the
#                                    instance (exit 1 if it isn't there); registry checks
#
# Cityscapes data goes only to this private instance, never anywhere public (license).
set -euo pipefail
source "$(dirname "$0")/gpu-config.sh"

usage() { sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
[ $# -ge 1 ] || usage
action=$1
shift

host=$(ssh_host)
remote="$host:$GPU_REMOTE_DIR"
RSYNC=(rsync -rlptz -e "ssh ${SSH_OPTS[*]}")

remote_sh() {  # run a command string on the instance
    ssh "${SSH_OPTS[@]}" "$host" "$1"
}

sync_code() {
    # Committed reports travel with the code, so git on the instance sees the same tree as
    # the Mac (otherwise every report there would be marked git_dirty). Reports written on
    # the instance are pulled first, so --delete can't remove one that isn't on the Mac yet.
    remote_sh "mkdir -p $GPU_REMOTE_DIR/artifacts/reports"
    pull_reports
    "${RSYNC[@]}" --delete \
        --include '/artifacts/' --include '/artifacts/reports/***' --exclude '/artifacts/*' \
        --exclude-from "$REPO_ROOT/scripts/rsync-excludes.txt" \
        "$REPO_ROOT/" "$remote/"
}

pull_reports() {
    # --update: a report the Mac rewrote after the last sync (a deploy run's deploy.json)
    # is newer than the instance's copy and must not be replaced by it.
    "${RSYNC[@]}" --update "$remote/artifacts/reports/" "$REPO_ROOT/artifacts/reports/"
}

case "$action" in
    run)
        [ $# -eq 1 ] || usage
        sync_code
        # The image tag is a hash of everything the image is built from, so a changed
        # uv.lock or Dockerfile triggers a rebuild and an unchanged one reuses the image.
        tty=""
        [ -t 1 ] && tty="-t"
        # A failed check (exit 1) still writes its report, so reports are pulled whatever
        # the exit status, and that status is passed on afterwards.
        status=0
        remote_sh "set -euo pipefail
cd ~/$GPU_REMOTE_DIR
mkdir -p artifacts/reports
tag=\$(cat docker/Dockerfile.trt docker/entrypoint.sh pyproject.toml uv.lock .python-version | sha256sum | cut -c1-12)
if ! sudo docker image inspect edgeinfer-trt:\$tag >/dev/null 2>&1; then
    echo \"building edgeinfer-trt:\$tag ...\"
    sudo docker build -f docker/Dockerfile.trt -t edgeinfer-trt:\$tag .
fi
sudo docker run --rm $tty --gpus all --hostname $GPU_NAME \\
    --user \$(id -u):\$(id -g) -e HOME=/tmp \\
    -v \$HOME/$GPU_REMOTE_DIR:/workspace/edgeinfer -v /data:/data:ro \\
    -e CITYSCAPES_ROOT=$GPU_DATA_DIR \\
    edgeinfer-trt:\$tag bash -c $(printf '%q' "$1")" || status=$?
        pull_reports
        exit "$status"
        ;;
    host)
        [ $# -eq 1 ] || usage
        remote_sh "$1"
        ;;
    push-data)
        root=${CITYSCAPES_ROOT:?set CITYSCAPES_ROOT to the local Cityscapes directory}
        remote_sh "mkdir -p $GPU_DATA_DIR/leftImg8bit/val $GPU_DATA_DIR/gtFine/val"
        # val images and their labelIds only: everything the eval and golden manifests read.
        "${RSYNC[@]}" --include '*/' --include '*_leftImg8bit.png' --exclude '*' \
            "$root/leftImg8bit/val/" "$host:$GPU_DATA_DIR/leftImg8bit/val/"
        "${RSYNC[@]}" --include '*/' --include '*_gtFine_labelIds.png' --exclude '*' \
            "$root/gtFine/val/" "$host:$GPU_DATA_DIR/gtFine/val/"
        ;;
    push-artifacts)
        [ $# -ge 1 ] || usage
        for path in "$@"; do
            path=${path%/}
            case "$path" in
                artifacts/reports* | artifacts/engines*) die "$path is written on the instance, not pushed" ;;
                artifacts/?*) ;;
                *) die "push-artifacts takes paths under artifacts/, got $path" ;;
            esac
            if [ -d "$REPO_ROOT/$path" ]; then
                remote_sh "mkdir -p $GPU_REMOTE_DIR/$path"
                "${RSYNC[@]}" "$REPO_ROOT/$path/" "$remote/$path/"
            elif [ -f "$REPO_ROOT/$path" ]; then
                remote_sh "mkdir -p $GPU_REMOTE_DIR/$(dirname "$path")"
                "${RSYNC[@]}" "$REPO_ROOT/$path" "$remote/$path"
            else
                die "no such file or directory: $path"
            fi
        done
        ;;
    pull-reports)
        pull_reports
        ;;
    sha256)
        [ $# -eq 1 ] || usage
        case "$1" in
            artifacts/?*) ;;
            *) die "sha256 takes a path under artifacts/, got $1" ;;
        esac
        remote_sh "set -o pipefail; cd $GPU_REMOTE_DIR && sha256sum $(printf '%q' "$1") | cut -d' ' -f1"
        ;;
    *) usage ;;
esac
