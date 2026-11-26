# Source in a Slurm job (after jureca/env.sh) to run on NDEV chips as a 1xNDEV mesh. NDEV<=1: no-op.
# Picks NDEV Ethernet-linked chips among those Slurm assigned (jureca/pick_line.py; CHIPS=a,b,... pins
# them), narrows TT_VISIBLE_DEVICES to them and writes a mesh graph descriptor to $RUN_DIR
# (TT_MESH_GRAPH_DESC_PATH). Request spare chips (--gres=gpu:12..16 for 8): not every set has a line.
# Taken from train_vit.sbatch on the vit-ddp branch.
if [ "${NDEV:-1}" -gt 1 ]; then
    if [ -z "$CHIPS" ]; then
        PICK=$(timeout 300 python jureca/pick_line.py "$NDEV") \
            || { echo "no $NDEV-chip Ethernet line among chips $TT_VISIBLE_DEVICES; request more (--gres)"; exit 1; }
        CHIPS=$(echo "$PICK" | sed -n 's/^CHIPS=//p')   # UMD logs to stdout too -> tagged line
    fi
    for c in ${CHIPS//,/ }; do
        [[ ",$TT_VISIBLE_DEVICES," == *",$c,"* ]] || { echo "chip $c is not in the Slurm allocation"; exit 1; }
    done
    [ "$(echo "$CHIPS" | tr ',' '\n' | grep -c .)" -eq "$NDEV" ] || { echo "CHIPS=$CHIPS is not $NDEV chips"; exit 1; }
    export TT_VISIBLE_DEVICES=$CHIPS
    export TT_MESH_GRAPH_DESC_PATH=$RUN_DIR/mgd_1x$NDEV.textproto
    cat > "$TT_MESH_GRAPH_DESC_PATH" <<EOF
mesh_descriptors {
  name: "M0"
  arch: BLACKHOLE
  device_topology { dims: [ 1, $NDEV ], dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
EOF
    echo "DDP over chips $TT_VISIBLE_DEVICES, MGD $TT_MESH_GRAPH_DESC_PATH"
fi
