# Private OCI registry

The john1 registry listens only on the Tailscale address at port 5000 and stores
data under the orchestration root. It is pinned by multi-platform manifest digest:

`registry@sha256:85347ed2ecde64161c7a4788a4d7d3dcc9d6f86f7be95834022e3c6a423a945a`

Workers configure this private endpoint as an insecure registry because all traffic
is confined to the trusted Tailnet. Research images are still submitted only by
immutable digest.
