"""Print N PCI chip ids (comma-separated, tagged CHIPS=...) from TT_VISIBLE_DEVICES that form an
Ethernet line (each consecutive pair linked by >= 2 eth channels). Read-only. Exit 1 if none found.
Usage (on a compute node): python jureca/pick_line.py N   -- used by train_vit.sbatch for DDP."""

import sys, collections, tt_umd

n = int(sys.argv[1])
cd = tt_umd.TopologyDiscovery.create_cluster_descriptor()
pci = cd.get_chips_with_mmio()  # logical id -> PCI id
links = collections.Counter()
for c, chans in cd.get_ethernet_connections().items():
    for peer, _ in chans.values():
        links[frozenset((pci[c], pci[peer]))] += 1
# UMD also discovers chips reachable over Ethernet that are NOT in TT_VISIBLE_DEVICES; never pick those
import os

allowed = (
    {int(x) for x in os.environ["TT_VISIBLE_DEVICES"].split(",")} if os.environ.get("TT_VISIBLE_DEVICES") else None
)
adj = collections.defaultdict(set)
for pair, k in links.items():
    if k >= 2 and len(pair) == 2 and (allowed is None or pair <= allowed):
        a, b = pair
        adj[a].add(b)
        adj[b].add(a)
print("eth graph:", {k: sorted(v) for k, v in sorted(adj.items())}, file=sys.stderr)


def dfs(path):
    if len(path) == n:
        return path
    for nb in sorted(adj[path[-1]]):
        if nb not in path and (r := dfs(path + [nb])):
            return r


for start in sorted(pci.values()):
    if (p := dfs([start])) if n > 1 else [start]:
        print("CHIPS=" + ",".join(map(str, p)))
        sys.exit(0)  # tagged: UMD logs to stdout too
sys.exit(1)
