"""Role-scoped D0 command plans and fail-closed runtime verification."""

from __future__ import annotations

import base64
import contextlib
import gzip
import hashlib
import io
import ipaddress
import json
import os
import plistlib
import re
import selectors
import shlex
import signal
import ssl
import stat
import subprocess
import tarfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .artifacts import (
    homebrew_bottle_path,
    homebrew_metadata_path,
    probe_context,
    validate_homebrew_formula_projection,
    verify_scanner_oci_archive,
    verify_smoke_oci_archive,
)
from .canonical import (
    CAMPAIGN_ID,
    COLIMA_CONFIG,
    DOCKER_CONFIG_JOHN2,
    DOCKER_CONFIG_WORKER,
    FROZEN_HOMEBREW,
    FROZEN_RUNTIME,
    HOST_REPORT_SCHEMA,
    PROBE_ARCHIVE_SHA256,
    PROBE_ARCHIVE_SIZE,
    PROBE_DOCKERFILE,
    PROBE_PAYLOAD,
    PROBE_PAYLOAD_SHA256,
    SCANNER_IMAGE,
    SMOKE_IMAGE,
    D0Error,
    canonical_json,
    document_sha256,
    primary_operation,
    sha256_bytes,
    validate_host_report,
    validate_work_packet,
)
from .inventory import (
    InventoryPolicy,
    compare_homebrew_install,
    compare_homebrew_ledger,
    compare_inventories,
    inventory_managed_homebrew_link,
    inventory_roots,
    podman_negative_control,
    runtime_activity,
    secure_owner_directory,
    selected_runtime_paths,
)

BREW = "/opt/homebrew/bin/brew"
COLIMA = "/opt/homebrew/bin/colima"
DOCKER = "/opt/homebrew/bin/docker"
LIMACTL = "/opt/homebrew/bin/limactl"
MAX_CAPTURE_BYTES = 128 * 1024 * 1024
RUNTIME_MAX_BYTES = 20 * 1024**3
FREE_FRACTION_PPM = 250_000
PROFILE = "cascadia-r2"
EXPECTED_ENGINE_VERSION = "29.5.2"
EXPECTED_ENGINE_STORAGE_DRIVER = "overlayfs"
EXPECTED_ENGINE_CGROUP_DRIVER = "cgroupfs"
EXPECTED_DAEMON_CONFIG = {
    "exec-opts": [f"native.cgroupdriver={EXPECTED_ENGINE_CGROUP_DRIVER}"],
    "features": {"buildkit": True, "containerd-snapshotter": True},
}
SCANNER_LOCAL_REFERENCE = "localhost/cascadia-r2-buildkit-syft-scanner:stable-1"
SCANNER_REGISTRY_HOST = "127.0.0.1"
SCANNER_REGISTRY_RESOLVER_HOST = "localhost"
SCANNER_REGISTRY_PORT = 5047
SCANNER_REGISTRY_REPOSITORY = "cascadia/buildkit-syft-scanner"
SCANNER_REGISTRY_ROOT = "/run/cascadia-r2-d0-scanner-registry"
SCANNER_REGISTRY_PROCESS_MARKER = "cascadia-r2-d0-scanner-registry-server-v1"
SCANNER_SOCKET_SAMPLER_PROCESS_MARKER = "cascadia-r2-d0-socket-sampler-v1"
SCANNER_EXPORT_SHA256 = "e92b612bee19f5bcdb2195599cead4c40fb684f0d4b3ac9a86e4f92c238c6841"
SCANNER_EXPORT_SIZE = 43_167_232
SCANNER_REGISTRY_CA_PATH = "/usr/local/share/ca-certificates/cascadia-r2-d0-loopback-registry.crt"
SCANNER_REGISTRY_CA_CERT = b"""-----BEGIN CERTIFICATE-----
MIIDQTCCAimgAwIBAgIIQ1JDMkQwQ0EwDQYJKoZIhvcNAQELBQAwLjEsMCoGA1UE
AwwjQ2FzY2FkaWEgUjIgRDAgTG9vcGJhY2sgUmVnaXN0cnkgQ0EwHhcNMjYwNjE5
MDk1MjAwWhcNMzYwNjE2MDk1MjAwWjAuMSwwKgYDVQQDDCNDYXNjYWRpYSBSMiBE
MCBMb29wYmFjayBSZWdpc3RyeSBDQTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCC
AQoCggEBAObNZSq4VFNwVFFEejLw1tNraj8kstEIitMT4HlfAVqXqwpvplnbU/De
/2RWgp6hTr/KRne5x2x1ZoIO5L+nrvJPzxD7Z1z4UOufm/35ctxFBLPq9PGvMrWG
ntlNd3UyMgD7XQym/swsNGePq6P8VA55eggbwjlrcNT1CgKPUIpDO1y0klN/3AVQ
GWGTv4bop7nU0rsStqp4C2JZx3Lt9JNVM4LxfjrCIDomjme5wE1eEmwgVoLWHvlq
xYf1+IbXAJp7/JQ+u7Q7lVLbNRwem89mAwq+nLK2C27tJJIgQecDMGuJ7mrkP0h9
sK3K2pGOnCwT8Fak7O7f2iTVaSFRenMCAwEAAaNjMGEwHQYDVR0OBBYEFHqTilrs
agzc9GCgp2H2lnNWWMA6MB8GA1UdIwQYMBaAFHqTilrsagzc9GCgp2H2lnNWWMA6
MA8GA1UdEwEB/wQFMAMBAf8wDgYDVR0PAQH/BAQDAgEGMA0GCSqGSIb3DQEBCwUA
A4IBAQDYTFGoBJ2coU/Mtf+czYSpctFP9ZNHV6bpXnzk3dLXxaGsDk3GPfwTD0is
M2jUH/k2SIfJZHoAw35IARWp20oGLjTg9dkgnWKwp8rYAExnsEjwaVAdZvMAg5Zu
QxTq9+xwSnX2JT+cbeNs312CLzKcJwI4ncZE7o2ThhIk+zyUWUGhFDdDkSDkfiBp
0/eHBBdq8A7uW7LqbCwuTgdoUr5zhWzorIsEobXg0p/zp+FsQQaGcLC/zcwlLnmd
ldBH2ZTWqznNpsUEXQUg4nQpgii+aSjB+M0Mby6E/vduf3JqpDCVFggFU/Iub1rx
ExJ/XB6tAAXjxoT2xxQ6I4m6tNVb
-----END CERTIFICATE-----
"""
SCANNER_REGISTRY_SERVER_CERT = b"""-----BEGIN CERTIFICATE-----
MIIDVzCCAj+gAwIBAgIIQ1JDMkQwU1IwDQYJKoZIhvcNAQELBQAwLjEsMCoGA1UE
AwwjQ2FzY2FkaWEgUjIgRDAgTG9vcGJhY2sgUmVnaXN0cnkgQ0EwHhcNMjYwNjE5
MDk1MjAwWhcNMzYwNjE2MDk1MjAwWjAUMRIwEAYDVQQDDAlsb2NhbGhvc3QwggEi
MA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQDtSj8CpoPGgHBnW8auFJaAnh/i
UYz9dv5B8qgv3uQ8LhfMQHM08yo1RT7k7Q/utEkuHWGPHQ8XCbUO9yqm2eR4eig8
dkUGxlKIwU3DFYyt3DWpVWDARIy7TpceLa/VoZIMljEwTZ8+alnOmoexvYPztYCH
Mcz0bAMqXagPrZcFE/2qgRos//ln9YY+NFFssdg0Oqj7fLou3GBbpyjAyaQjdu2C
ZT0g8++G4K1E8Emx0ZsEe6vs2LbSzgBxtfUa97//Com5G5hsvoyWOBPYK3Ub5gA3
+DrQTq0//OK43gHTH5GR/Juh6XxzoBJaJZfE7OMlX8EkjfEG42wWrVJSCVnLAgMB
AAGjgZIwgY8wDAYDVR0TAQH/BAIwADAOBgNVHQ8BAf8EBAMCBaAwEwYDVR0lBAww
CgYIKwYBBQUHAwEwGgYDVR0RBBMwEYIJbG9jYWxob3N0hwR/AAABMB0GA1UdDgQW
BBRNsP6f3OwhNK9aJma9244RwLc0ZzAfBgNVHSMEGDAWgBR6k4pa7GoM3PRgoKdh
9pZzVljAOjANBgkqhkiG9w0BAQsFAAOCAQEAGGxjIPSa/cIa/TXTE6X/OiYf9aAW
iW8fVbnxIyGBSLryYJe45WW6EFJHf48rWkYed/7ShczshUteUWVa/UzSxjFHrizn
MpEtKIEZ9cbaSt44rbgwiYXOqn/c6ZrdLMdQ0E0EGGihDvnnyAD4WwzvRo4RgBQG
tP0wfhWx2XHlpK4Rwx04Ywokgmwzu8uBWmpo1DQGAqOhSxoO8xNgYcRuBr/LETc8
/b4P3yUY26nnskWAMc8KvwnkgLwhzlEG1o2323ZN8dYXbzc/M+T8z+C9lN6bDXKk
RbXKbe2B2M480cMuzJLb/dvC0BiJq0Zls6T31+P3P6K5/0oJA7azzrf52Q==
-----END CERTIFICATE-----
"""
SCANNER_REGISTRY_SERVER_KEY = b"""-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDtSj8CpoPGgHBn
W8auFJaAnh/iUYz9dv5B8qgv3uQ8LhfMQHM08yo1RT7k7Q/utEkuHWGPHQ8XCbUO
9yqm2eR4eig8dkUGxlKIwU3DFYyt3DWpVWDARIy7TpceLa/VoZIMljEwTZ8+alnO
moexvYPztYCHMcz0bAMqXagPrZcFE/2qgRos//ln9YY+NFFssdg0Oqj7fLou3GBb
pyjAyaQjdu2CZT0g8++G4K1E8Emx0ZsEe6vs2LbSzgBxtfUa97//Com5G5hsvoyW
OBPYK3Ub5gA3+DrQTq0//OK43gHTH5GR/Juh6XxzoBJaJZfE7OMlX8EkjfEG42wW
rVJSCVnLAgMBAAECggEADGESNnYva7ypoIrGK7DBRUZ6jVkJzPNXQmOBaLXtHEKf
VTWLjRiefBb3uAOyEBbxtmkr/MZcUixceMJmRDwb0jk1PipBYaAZEC9dei72nt9d
IU9l88HfwanQ8m8ZvEBomSWAvW9fPmdX9hJTv+8OIjm20dbERYAZDi3Cpca9mTMf
EESvu7J+nlsMHxhZq2AOf0wfhzVp6Dpm0a+85RCSpf/Qts6xcJ10/x8F78EkLfSt
SVSlglaGrHlz9eDjvn5QZ6j+D5VhQh6wWXJuOBzqfj+70TP0881SfiURamFlhAbj
AKcSqV9xGxxQiL5IsWU52HIhPh4kqlCsEiTnQdZOYQKBgQD8tg0dBgDh2KlN+Eyb
mwGy8jbAmrOcNfhnmwm69077qEKQTJciIrDhg4T0YgSaEf67kqx00yB+CTraZjal
qgkKXsSTZuaXIi5gcmckn8hX0Vn8CWdwFUVVornCCSMJxH/hS0/7tZMczSmw6g9k
HTzGb6rLkDmCNaKkEb7hiYxz2wKBgQDwYNEiVjnYho/yVBifJv3zcGRJKuEkvORX
sCzsCs+DhlOIqpSoQrS+wFfTCOB1qcr+JJ4b2My8pLn5/kFql6OKuWbaf8X56oBp
08pe0bjzkrXKwcnCvKnHXmZG/T2ihHKvov2KszMSQyVRHFB/TbGESZaVRU6b27Xb
poPcxfmM0QKBgQCMAvlk+SyH9Jho4IbhN5JLaLM5Jv0YMTa9gEJ12gtilqi6dhTO
DtZdO5bwJ1ZRXmL53Zu65jZ8XfTDiBoC0yBLJJJY8IwVdBSpzviia/x92zm10CgF
C2PsvEma3aESClKnqihYVxN4w5qzsBpy51gCwV+phPC32auQp1xQbPrqPQKBgEEf
SalOyO8jTX4uUFlVu/km2tSDvGkyj34+KX1tVFjinGDrLckEAWmoPGLdBcp6zJbb
nsYWjykQS54xxtE08caUggvyD9WsNUv2Z94WXVAH0B51L88FQ83Sgkz7MKaF0XhJ
5Pydndl1vXdi/1/t0YjwUs5v72MEPBmc3B6EuB3xAoGBAJqqJOJWIy/leSocnbdb
uTgjeGndLAMiefrfEOQcXrOcUMj7Q/bKxGyGULUTzFBFyoza4AShBLsYv9nRt7M6
ecEGSptdFy8ES4HxziVlR1uSH5uPCn/6gl/p2XdjPyYlKwu7foHYRitG+hwWSiA1
J1yIEZa0eA9DkpnWnt91tTD0
-----END PRIVATE KEY-----
"""
SCANNER_REGISTRY_CA_SHA256 = sha256_bytes(SCANNER_REGISTRY_CA_CERT)
SCANNER_REGISTRY_SERVER_CERT_SHA256 = sha256_bytes(SCANNER_REGISTRY_SERVER_CERT)
SCANNER_REGISTRY_SERVER_KEY_SHA256 = sha256_bytes(SCANNER_REGISTRY_SERVER_KEY)
SCANNER_REGISTRY_SERVER_CERT_DER_SHA256 = sha256_bytes(
    ssl.PEM_cert_to_DER_cert(SCANNER_REGISTRY_SERVER_CERT.decode("ascii"))
)
GLOBAL_RUNTIME_ENTRYPOINTS = {
    "colima": Path("/opt/homebrew/bin/colima"),
    "lima": Path("/opt/homebrew/bin/limactl"),
    "docker": Path("/opt/homebrew/bin/docker"),
    "docker-buildx": Path("/opt/homebrew/lib/docker/cli-plugins/docker-buildx"),
}
HOST_PLATFORM = {
    "john1": {
        "macos_version": "26.5.1",
        "build_version": "25F80",
        "darwin_release": "25.5.0",
    },
    "john2": {
        "macos_version": "26.5.1",
        "build_version": "25F80",
        "darwin_release": "25.5.0",
    },
    "john3": {
        "macos_version": "26.5.1",
        "build_version": "25F80",
        "darwin_release": "25.5.0",
    },
}
MIN_MEMORY_FREE_PERCENT = 25
HOST_HOME = {
    "john1": "/Users/johnherrick",
    "john2": "/Users/john2",
    "john3": "/Users/john3",
}
FORMULA_ORDER = ("lima", "colima", "docker", "docker-buildx")
VERSION_PATTERNS = {
    "colima": re.compile(r"(?:colima version )?v?0\.10\.3(?:\s|\Z)"),
    "lima": re.compile(r"(?:limactl version )?v?2\.1\.2(?:\s|\Z)"),
    "docker": re.compile(r"Docker version 29\.5\.3,"),
    "docker-buildx": re.compile(r"github\.com/docker/buildx v0\.35\.0(?:\s|\Z)"),
}
_GUEST_AUDIT_SCRIPT = r"""import hashlib,json,os,platform,stat,subprocess
query=subprocess.run(["/usr/bin/dpkg-query","-W","-f=${binary:Package}\t${Version}\t${Architecture}\n"],check=True,capture_output=True,text=True)
packages=[]
licenses=[]
for line in query.stdout.splitlines():
 fields=line.split("\t")
 if len(fields)!=3: raise SystemExit(21)
 name,version,architecture=fields
 packages.append({"name":name,"version":version,"architecture":architecture})
 base=name.split(":",1)[0]
 doc_dir="/usr/share/doc/"+base
 requested=doc_dir+"/copyright"
 resolved=os.path.realpath(requested)
 doc_dir_exists=os.path.lexists(doc_dir)
 doc_dir_is_symlink=os.path.islink(doc_dir)
 requested_exists=os.path.lexists(requested)
 requested_is_symlink=os.path.islink(requested)
 exists=os.path.exists(resolved)
 item={
  "package":name,"requested":requested,"resolved":resolved,
  "doc_dir":doc_dir,"doc_dir_exists":doc_dir_exists,
  "doc_dir_is_symlink":doc_dir_is_symlink,
  "doc_dir_symlink_target":os.readlink(doc_dir) if doc_dir_is_symlink else None,
  "requested_exists":requested_exists,"requested_is_symlink":requested_is_symlink,
  "requested_symlink_target":os.readlink(requested) if requested_is_symlink else None,
  "exists":exists,"present":False,"size":0,"sha256":None,
 }
 if resolved.startswith("/usr/share/doc/") and exists:
  st=os.stat(resolved,follow_symlinks=False)
  if stat.S_ISREG(st.st_mode):
   h=hashlib.sha256(); total=0
   with open(resolved,"rb") as stream:
    while True:
     chunk=stream.read(1048576)
     if not chunk: break
     h.update(chunk); total+=len(chunk)
   item.update({"present":True,"size":total,"sha256":h.hexdigest()})
 licenses.append(item)
def command(argv):
 value=subprocess.run(argv,check=False,capture_output=True,text=True)
 return {"argv":argv,"returncode":value.returncode,"stdout":value.stdout,"stderr":value.stderr}
result={
 "schema_id":"cascadia.r2-map.d0-guest-audit.v1",
 "schema_version":1,
 "packages":packages,
 "licenses":licenses,
 "machine":platform.machine(),
 "release":platform.release(),
 "versions":{
  "docker":command(["/usr/bin/docker","version","--format","{{json .}}"]),
 "containerd":command(["/usr/bin/containerd","--version"]),
 "daemon_config":command(["/usr/bin/cat","/etc/docker/daemon.json"]),
  "runc":command(["/usr/bin/runc","--version"]),
  "containerd_owner":command(["/usr/bin/dpkg-query","-S","/usr/bin/containerd"]),
  "runc_owner":command(["/usr/bin/dpkg-query","-S","/usr/bin/runc"]),
  "os_release":command(["/usr/bin/cat","/etc/os-release"]),
  "listeners":command(["/usr/bin/ss","-H","-lntp"]),
  "network_addresses":command(["/usr/sbin/ip","-j","address","show"]),
  "cpu_count":command(["/usr/bin/nproc"]),
  "meminfo":command(["/usr/bin/cat","/proc/meminfo"]),
  "virtualization":command(["/usr/bin/systemd-detect-virt"]),
  "mounts":command(["/usr/bin/findmnt","-J","-o","TARGET,SOURCE,FSTYPE,OPTIONS"]),
  "block_devices":command(["/usr/bin/lsblk","-b","-J","-o","NAME,KNAME,PATH,PKNAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS,RO"]),
  "root_mount":command(["/usr/bin/findmnt","-J","-b","-T","/","-o","TARGET,SOURCE,FSTYPE,OPTIONS"]),
  "docker_data_mount":command(["/usr/bin/findmnt","-J","-b","-T","/var/lib/docker","-o","TARGET,SOURCE,FSTYPE,OPTIONS"]),
  "binfmt_handlers":command([
   "/usr/bin/python3","-I","-S","-B","-c",
   "import json,os; p='/proc/sys/fs/binfmt_misc'; "
   "names=sorted(os.listdir(p)) if os.path.isdir(p) else []; "
   "print(json.dumps({'control':[n for n in names if n in {'register','status'}],"
   "'handlers':{n:open(p+'/'+n).read(4096) for n in names "
   "if n not in {'register','status'}}},sort_keys=True))",
  ]),
  "nested_virtualization":command([
   "/usr/bin/python3","-I","-S","-B","-c",
   "import json,glob,os; "
   "print(json.dumps({'dev_kvm':os.path.exists('/dev/kvm'),"
   "'modules':sorted(glob.glob('/sys/module/kvm*'))},sort_keys=True))",
  ]),
  "ssh_agent":command([
   "/usr/bin/python3","-I","-S","-B","-c",
   "import os; print(os.environ.get('SSH_AUTH_SOCK',''))",
  ]),
  "processes":command(["/usr/bin/ps","-e","-o","comm="]),
  "kubernetes_state":command([
   "/usr/bin/python3","-I","-S","-B","-c",
   "import glob,json,os; "
   "paths=['/etc/kubernetes','/var/lib/kubelet','/var/lib/rancher/k3s',"
   "'/etc/rancher/k3s']; "
   "print(json.dumps({'paths':sorted(p for p in paths if os.path.lexists(p)),"
   "'units':sorted(glob.glob('/etc/systemd/system/*kube*')+"
   "glob.glob('/etc/systemd/system/*k3s*'))},sort_keys=True))",
  ]),
 },
}
print(json.dumps(result,sort_keys=True,separators=(",",":")))
"""
GUEST_AUDIT_SCRIPT_SHA256 = sha256_bytes(_GUEST_AUDIT_SCRIPT.encode("utf-8"))
_GUEST_BUILDKIT_STATE_SCRIPT = r"""import hashlib,json,os,socket,stat
root="/var/lib/docker/buildkit"
entries=[]
content_blobs=[]
payload_paths=[]
if os.path.exists(root):
 for base,dirs,files in os.walk(root,topdown=True,followlinks=False):
  dirs.sort(); files.sort()
  for name in dirs+files:
   path=os.path.join(base,name); rel=os.path.relpath(path,root); value=os.lstat(path)
   if stat.S_ISLNK(value.st_mode): raise SystemExit(41)
   kind="directory" if stat.S_ISDIR(value.st_mode) else "file"
   if not (stat.S_ISDIR(value.st_mode) or stat.S_ISREG(value.st_mode)): kind="other"
   item={
    "path":rel,
    "type":kind,
    "size":value.st_size,
    "mode":format(stat.S_IMODE(value.st_mode),"04o"),
   }
   if item["type"]=="other": raise SystemExit(42)
   if item["type"]=="file":
    if value.st_size>268435456: raise SystemExit(43)
    h=hashlib.sha256()
    with open(path,"rb") as stream:
     while True:
      chunk=stream.read(1048576)
      if not chunk: break
      h.update(chunk)
    item["sha256"]=h.hexdigest()
    if rel.startswith("content/blobs/"): content_blobs.append(rel)
    if (
     rel.startswith("content/ingest/") or rel.startswith("executor/")
     or rel.startswith("cachemounts/") or rel.startswith("snapshots/snapshots/")
    ): payload_paths.append(rel)
   entries.append(item)
def docker_json(path):
 stream=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
 stream.settimeout(5); stream.connect("/var/run/docker.sock")
 try:
  request="GET "+path+" HTTP/1.1\r\nHost: docker\r\nConnection: close\r\n\r\n"
  stream.sendall(request.encode("ascii")); chunks=[]; total=0
  while True:
   chunk=stream.recv(1048576)
   if not chunk: break
   total+=len(chunk)
   if total>16777216: raise SystemExit(44)
   chunks.append(chunk)
 finally: stream.close()
 response=b"".join(chunks); head,separator,body=response.partition(b"\r\n\r\n")
 if separator!=b"\r\n\r\n" or not head.startswith(b"HTTP/1.1 200 "): raise SystemExit(45)
 headers={}
 for line in head.split(b"\r\n")[1:]:
  key,colon,value=line.partition(b":")
  if colon!=b":" or key.lower() in headers: raise SystemExit(46)
  headers[key.lower()]=value.strip()
 length=headers.get(b"content-length",str(len(body)).encode("ascii"))
 if b"transfer-encoding" in headers or int(length)!=len(body): raise SystemExit(47)
 return json.loads(body)
disk=docker_json("/v1.53/system/df")
cache=disk.get("BuildCache") or []
if not isinstance(cache,list): raise SystemExit(48)
build_cache_records=[]
for item in cache:
 if not isinstance(item,dict): raise SystemExit(49)
 build_cache_records.append({
  "id":item.get("ID"),"type":item.get("Type"),"size":item.get("Size"),
  "in_use":item.get("InUse"),"shared":item.get("Shared"),
  "parents":sorted(item.get("Parents") or []),
 })
entries.sort(key=lambda item:item["path"]); content_blobs.sort(); payload_paths.sort()
build_cache_records.sort(
 key=lambda item:json.dumps(item,sort_keys=True,separators=(",",":"))
)
result={
 "schema_id":"cascadia.r2-map.d0-buildkit-state.v2","root":root,
 "present":os.path.isdir(root),"entries":entries,"content_blobs":content_blobs,
 "payload_paths":payload_paths,"build_cache_records":build_cache_records,
}
print(json.dumps(result,sort_keys=True,separators=(",",":")))
"""
GUEST_BUILDKIT_STATE_SCRIPT_SHA256 = sha256_bytes(_GUEST_BUILDKIT_STATE_SCRIPT.encode("utf-8"))
EGRESS_CONTROL_RECEIVE_TIMEOUT_SECONDS = 5
EGRESS_CONTROL_DOCUMENT_TIMEOUT_SECONDS = 15
EGRESS_CONTROL_SERVER_MARKER = "cascadia-r2-d0-egress-server-v2"
EGRESS_CONTROL_CLIENT_MARKER = "cascadia-r2-d0-egress-client-v2"
_EGRESS_SERVER_SCRIPT = r"""import json,os,socket,struct,sys,time
marker,state,host,port,receive_timeout=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]),int(sys.argv[5])
if marker!='cascadia-r2-d0-egress-server-v2': raise SystemExit(41)
pid=os.fork()
if pid: raise SystemExit(0)
os.setsid()
null=os.open('/dev/null',os.O_RDWR)
for fd in (0,1,2): os.dup2(null,fd)
def write(name,value):
 path=os.path.join(state,name); temp=path+'.partial-'+str(os.getpid())
 with open(temp,'x',encoding='ascii') as stream:
  stream.write(json.dumps(value,sort_keys=True,separators=(',',':')))
  stream.flush(); os.fsync(stream.fileno())
 os.chmod(temp,0o600)
 os.replace(temp,path)
write('server.pid',{'marker':marker,'pid':os.getpid()})
server=socket.socket(); server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
server.bind((host,port)); server.listen(1); server.settimeout(60)
write('server-ready.json',{'ready':True})
received=False; detail='no-connection'; abortive_close=False
try:
 connection,_=server.accept(); connection.settimeout(5)
 if connection.recv(1)!=b'A': detail='initial-byte-differed'
 else:
  connection.settimeout(receive_timeout)
  try: received=connection.recv(1)==b'B'; detail='received' if received else 'closed'
  except OSError as error: detail=type(error).__name__
  connection.setsockopt(socket.SOL_SOCKET,socket.SO_LINGER,struct.pack('ii',1,0))
  connection.close(); abortive_close=True
except OSError as error: detail=type(error).__name__
server.close(); write('outcome.json',{
 'abortive_close':abortive_close,'received_after_guard':received,'detail':detail,
})
"""
_EGRESS_CLIENT_SCRIPT = r"""import json,os,socket,struct,sys,time
marker,state,host,port=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4])
if marker!='cascadia-r2-d0-egress-client-v2': raise SystemExit(52)
pid=os.fork()
if pid: raise SystemExit(0)
os.setsid()
null=os.open('/dev/null',os.O_RDWR)
for fd in (0,1,2): os.dup2(null,fd)
def write(name,value):
 path=os.path.join(state,name); temp=path+'.partial-'+str(os.getpid())
 with open(temp,'x',encoding='ascii') as stream:
  stream.write(json.dumps(value,sort_keys=True,separators=(',',':')))
  stream.flush(); os.fsync(stream.fileno())
 os.chmod(temp,0o600)
 os.replace(temp,path)
write('client.pid',{'marker':marker,'pid':os.getpid()})
connection=socket.socket(); connection.settimeout(10)
connection.connect((host,port)); connection.sendall(b'A')
write('established.json',{'established':True})
deadline=time.monotonic()+60
while not os.path.exists(os.path.join(state,'trigger')):
 if time.monotonic()>=deadline:
  write('client-outcome.json',{'triggered':False}); raise SystemExit(51)
 time.sleep(.02)
try:
 connection.sendall(b'B')
 outcome={'triggered':True,'send_returned':True}
except OSError as error:
 outcome={'triggered':True,'send_returned':False,'error':type(error).__name__}
connection.setsockopt(socket.SOL_SOCKET,socket.SO_LINGER,struct.pack('ii',1,0))
connection.close(); outcome['abortive_close']=True; write('client-outcome.json',outcome)
"""
SCANNER_REGISTRY_PREPARE_SCRIPT = r"""import hashlib,json,os,stat,sys,tarfile
root=sys.argv[1]
archive=os.path.join(root,'scanner.oci.tar')
spec={
 'manifest':(sys.argv[2],int(sys.argv[3])),
 'config':(sys.argv[4],int(sys.argv[5])),
 'layer':(sys.argv[6],int(sys.argv[7])),
}
st=os.lstat(root)
if not stat.S_ISDIR(st.st_mode) or st.st_uid!=0 or stat.S_IMODE(st.st_mode)!=0o700:
 raise SystemExit(121)
st=os.lstat(archive)
if not stat.S_ISREG(st.st_mode) or st.st_uid!=0 or st.st_size>268435456:
 raise SystemExit(122)
os.chmod(archive,0o600)
archive_hash=hashlib.sha256()
with open(archive,'rb') as stream:
 while True:
  chunk=stream.read(1048576)
  if not chunk: break
  archive_hash.update(chunk)
if st.st_size!=43167232: raise SystemExit(123)
if archive_hash.hexdigest()!='e92b612bee19f5bcdb2195599cead4c40fb684f0d4b3ac9a86e4f92c238c6841':
 raise SystemExit(124)
manifest_digest,manifest_size=spec['manifest']
config_digest,config_size=spec['config']; layer_digest,layer_size=spec['layer']
expected={
 'blobs':('directory',0,0o755,None),
 'blobs/sha256':('directory',0,0o755,None),
 'blobs/sha256/'+config_digest.split(':',1)[1]:('file',config_size,0o444,config_digest),
 'blobs/sha256/'+layer_digest.split(':',1)[1]:('file',layer_size,0o444,layer_digest),
 'blobs/sha256/'+manifest_digest.split(':',1)[1]:('file',manifest_size,0o444,manifest_digest),
 'index.json':('file',385,0o644,'sha256:0e70c3479ddb4af70d6ca18dc5a199055e0ca1a8578e117745fd2cd7cdc27a10'),
 'manifest.json':('file',251,0o644,'sha256:41905e3b3b02f8cb4f0a9c7950b3a24516c05c9d1651c6b2c43154044cbd088d'),
 'oci-layout':('file',30,0o444,'sha256:18f0797eab35a4597c1e9624aa4f15fd91f6254e5538c1e0d193b2a95dd4acc6'),
}
seen=set(); receipts={}; payloads={}
with tarfile.open(archive,'r:') as source:
 for member in source:
  name=member.name.removeprefix('./')
  if not name or name.startswith('/') or '..' in name.split('/') or name in seen:
   raise SystemExit(125)
  seen.add(name)
  if name not in expected: raise SystemExit(126)
  kind,size,mode,digest=expected[name]
  actual_kind='directory' if member.isdir() else 'file' if member.isfile() else 'other'
  if actual_kind!=kind or member.size!=size or member.mode!=mode: raise SystemExit(127)
  if member.uid!=0 or member.gid!=0: raise SystemExit(128)
  if kind=='directory': continue
  stream=source.extractfile(member)
  if stream is None: raise SystemExit(129)
  payload=stream.read(size+1)
  if len(payload)!=size or 'sha256:'+hashlib.sha256(payload).hexdigest()!=digest:
   raise SystemExit(130)
  payloads[name]=payload
if seen!=set(expected): raise SystemExit(131)
for kind,(digest,size) in spec.items():
 name='blobs/sha256/'+digest.split(':',1)[1]; payload=payloads[name]
 destination=os.path.join(root,kind+'.blob')
 fd=os.open(destination,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 view=memoryview(payload)
 while view:
  written=os.write(fd,view)
  if written<=0: raise SystemExit(132)
  view=view[written:]
 os.fsync(fd); os.close(fd)
 receipts[kind]={'digest':digest,'path':destination,'size':size}
payloads_by_kind={}
for kind,(digest,_) in spec.items():
 payloads_by_kind[kind]=payloads['blobs/sha256/'+digest.split(':',1)[1]]
layout=json.loads(payloads['oci-layout']); index=json.loads(payloads['index.json'])
docker_manifest=json.loads(payloads['manifest.json'])
if layout!={'imageLayoutVersion':'1.0.0'}: raise SystemExit(133)
descriptors=index.get('manifests') if isinstance(index,dict) else None
if not isinstance(descriptors,list) or len(descriptors)!=1: raise SystemExit(134)
descriptor=descriptors[0]
annotations=descriptor.get('annotations',{}) if isinstance(descriptor,dict) else {}
if descriptor.get('digest')!=manifest_digest or descriptor.get('size')!=manifest_size:
 raise SystemExit(135)
local_reference='localhost/cascadia-r2-buildkit-syft-scanner:stable-1'
if annotations.get('io.containerd.image.name')!=local_reference:
 raise SystemExit(136)
if annotations.get('org.opencontainers.image.ref.name')!='stable-1': raise SystemExit(137)
if docker_manifest!=[{
 'Config':'blobs/sha256/'+config_digest.split(':',1)[1],
 'RepoTags':['localhost/cascadia-r2-buildkit-syft-scanner:stable-1'],
 'Layers':['blobs/sha256/'+layer_digest.split(':',1)[1]],
 }]: raise SystemExit(138)
manifest=json.loads(payloads_by_kind['manifest'])
if manifest.get('config',{}).get('digest')!=spec['config'][0]: raise SystemExit(129)
layers=manifest.get('layers')
if not isinstance(layers,list) or [item.get('digest') for item in layers]!=[spec['layer'][0]]:
 raise SystemExit(139)
requests=os.path.join(root,'requests'); os.mkdir(requests,0o700)
result={
 'archive_sha256':archive_hash.hexdigest(),'archive_size':st.st_size,
 'blobs':receipts,'requests_directory':requests,'status':'prepared',
}
sys.stdout.write(json.dumps(result,sort_keys=True,separators=(',',':')))
"""

SCANNER_REGISTRY_TRUST_INSTALL_SCRIPT = r"""import hashlib,json,os,stat,subprocess,sys
root,ca_path=sys.argv[1:3]
expected=sys.argv[3:6]
payload=json.load(sys.stdin)
if set(payload)!={'ca_cert','server_cert','server_key'}: raise SystemExit(151)
values=[payload['ca_cert'].encode(),payload['server_cert'].encode(),payload['server_key'].encode()]
if [hashlib.sha256(value).hexdigest() for value in values]!=expected: raise SystemExit(152)
if os.path.lexists(ca_path): raise SystemExit(153)
update='/usr/sbin/update-ca-certificates'
st=os.lstat(update)
if not stat.S_ISREG(st.st_mode) or not os.access(update,os.X_OK): raise SystemExit(154)
def identity(path):
 st=os.lstat(path); mode=stat.S_IMODE(st.st_mode)
 result={'gid':st.st_gid,'mode':mode,'path':path,'uid':st.st_uid}
 if stat.S_ISREG(st.st_mode):
  result['size']=st.st_size
  digest=hashlib.sha256()
  with open(path,'rb') as stream:
   while True:
    chunk=stream.read(1048576)
    if not chunk: break
    digest.update(chunk)
  result.update({'kind':'file','sha256':digest.hexdigest()})
 elif stat.S_ISLNK(st.st_mode):
  result.update({'kind':'symlink','size':st.st_size,'target':os.readlink(path)})
 elif stat.S_ISDIR(st.st_mode): result.update({'kind':'directory'})
 else: raise SystemExit(155)
 return result
def inventory():
 roots=['/etc/ssl/certs','/usr/local/share/ca-certificates']; rows=[]
 for selected in roots:
  if not os.path.lexists(selected): raise SystemExit(156)
  rows.append(identity(selected))
  for current,dirs,files in os.walk(selected,topdown=True,followlinks=False):
   dirs.sort(); files.sort()
   for name in dirs+files: rows.append(identity(os.path.join(current,name)))
 return rows
baseline=inventory()
baseline_bytes=json.dumps(baseline,sort_keys=True,separators=(',',':')).encode()
baseline_path=os.path.join(root,'trust-baseline.json')
fd=os.open(baseline_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
os.write(fd,baseline_bytes); os.fsync(fd); os.close(fd)
for name,value,mode in (
 ('registry-ca.crt',values[0],0o600),('registry-server.crt',values[1],0o600),
 ('registry-server.key',values[2],0o600),(ca_path,values[0],0o644),
):
 path=name if name.startswith('/') else os.path.join(root,name)
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,mode)
 view=memoryview(value)
 while view:
  written=os.write(fd,view)
  if written<=0: raise SystemExit(157)
  view=view[written:]
 os.fsync(fd); os.close(fd)
completed=subprocess.run([update,'--fresh'],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
 stderr=subprocess.PIPE,env={'PATH':'/usr/sbin:/usr/bin:/sbin:/bin','LANG':'C','LC_ALL':'C'},timeout=60)
if completed.returncode!=0 or len(completed.stdout)>1048576 or len(completed.stderr)>1048576:
 raise SystemExit(158)
bundle='/etc/ssl/certs/ca-certificates.crt'; bundle_bytes=open(bundle,'rb').read(16777217)
if len(bundle_bytes)>16777216 or values[0] not in bundle_bytes: raise SystemExit(159)
installed=inventory()
installed_bytes=json.dumps(installed,sort_keys=True,separators=(',',':')).encode()
result={
 'baseline_sha256':hashlib.sha256(baseline_bytes).hexdigest(),
 'ca_path':ca_path,'ca_sha256':expected[0],
 'installed_sha256':hashlib.sha256(installed_bytes).hexdigest(),
 'server_cert_sha256':expected[1],'server_key_sha256':expected[2],
 'update_stderr_sha256':hashlib.sha256(completed.stderr).hexdigest(),
 'update_stdout_sha256':hashlib.sha256(completed.stdout).hexdigest(),'status':'installed',
}
sys.stdout.write(json.dumps(result,sort_keys=True,separators=(',',':')))
"""

SCANNER_REGISTRY_TRUST_CLEANUP_SCRIPT = r"""import hashlib,json,os,stat,subprocess,sys
root,ca_path=sys.argv[1:3]
baseline_path=os.path.join(root,'trust-baseline.json')
if not os.path.lexists(baseline_path):
 if os.path.lexists(ca_path): raise SystemExit(160)
 empty=hashlib.sha256(b'[]').hexdigest()
 result={
  'baseline_sha256':empty,'ca_path':ca_path,'ca_path_absent':True,
  'restored_sha256':empty,'update_stderr_sha256':hashlib.sha256(b'').hexdigest(),
  'update_stdout_sha256':hashlib.sha256(b'').hexdigest(),'status':'absent',
 }
 sys.stdout.write(json.dumps(result,sort_keys=True,separators=(',',':')))
 raise SystemExit(0)
st=os.lstat(baseline_path)
if not stat.S_ISREG(st.st_mode) or st.st_uid!=0 or stat.S_IMODE(st.st_mode)!=0o600:
 raise SystemExit(161)
baseline_bytes=open(baseline_path,'rb').read(16777217)
if len(baseline_bytes)>16777216: raise SystemExit(162)
baseline=json.loads(baseline_bytes)
if os.path.lexists(ca_path):
 st=os.lstat(ca_path)
 if not stat.S_ISREG(st.st_mode) or st.st_uid!=0 or stat.S_IMODE(st.st_mode)!=0o644:
  raise SystemExit(163)
 os.unlink(ca_path)
update='/usr/sbin/update-ca-certificates'
completed=subprocess.run([update,'--fresh'],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
 stderr=subprocess.PIPE,env={'PATH':'/usr/sbin:/usr/bin:/sbin:/bin','LANG':'C','LC_ALL':'C'},timeout=60)
if completed.returncode!=0 or len(completed.stdout)>1048576 or len(completed.stderr)>1048576:
 raise SystemExit(164)
def identity(path):
 st=os.lstat(path); mode=stat.S_IMODE(st.st_mode)
 result={'gid':st.st_gid,'mode':mode,'path':path,'uid':st.st_uid}
 if stat.S_ISREG(st.st_mode):
  result['size']=st.st_size
  digest=hashlib.sha256()
  with open(path,'rb') as stream:
   while True:
    chunk=stream.read(1048576)
    if not chunk: break
    digest.update(chunk)
  result.update({'kind':'file','sha256':digest.hexdigest()})
 elif stat.S_ISLNK(st.st_mode):
  result.update({'kind':'symlink','size':st.st_size,'target':os.readlink(path)})
 elif stat.S_ISDIR(st.st_mode): result.update({'kind':'directory'})
 else: raise SystemExit(165)
 return result
rows=[]
for selected in ['/etc/ssl/certs','/usr/local/share/ca-certificates']:
 if not os.path.lexists(selected): raise SystemExit(166)
 rows.append(identity(selected))
 for current,dirs,files in os.walk(selected,topdown=True,followlinks=False):
  dirs.sort(); files.sort()
  for name in dirs+files: rows.append(identity(os.path.join(current,name)))
restored_bytes=json.dumps(rows,sort_keys=True,separators=(',',':')).encode()
if rows!=baseline or os.path.lexists(ca_path): raise SystemExit(167)
result={
 'baseline_sha256':hashlib.sha256(baseline_bytes).hexdigest(),'ca_path':ca_path,
 'ca_path_absent':True,'restored_sha256':hashlib.sha256(restored_bytes).hexdigest(),
 'update_stderr_sha256':hashlib.sha256(completed.stderr).hexdigest(),
 'update_stdout_sha256':hashlib.sha256(completed.stdout).hexdigest(),'status':'restored',
}
sys.stdout.write(json.dumps(result,sort_keys=True,separators=(',',':')))
"""

SCANNER_REGISTRY_TLS_CLIENT_SCRIPT = r"""import hashlib,json,socket,ssl,sys
host,port,expected=sys.argv[1],int(sys.argv[2]),sys.argv[3]
context=ssl.create_default_context()
with socket.create_connection((host,port),timeout=10) as raw:
 with context.wrap_socket(raw,server_hostname=host) as connection:
  certificate=connection.getpeercert(binary_form=True)
  protocol=connection.version(); cipher=connection.cipher()
  connection.sendall(b'GET /v2/ HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n')
  response=b''
  while len(response)<=65536:
   chunk=connection.recv(65536)
   if not chunk: break
   response+=chunk
if len(response)>65536 or not response.startswith(b'HTTP/1.1 200 '): raise SystemExit(171)
observed=hashlib.sha256(certificate).hexdigest()
if observed!=expected or protocol not in {'TLSv1.2','TLSv1.3'} or not cipher:
 raise SystemExit(172)
result={
 'cipher':cipher[0],'peer_certificate_der_sha256':observed,'protocol':protocol,
 'response_sha256':hashlib.sha256(response).hexdigest(),'status':'pass',
}
sys.stdout.write(json.dumps(result,sort_keys=True,separators=(',',':')))
"""

SCANNER_SOCKET_SAMPLER_LAUNCH_SCRIPT = r"""import json,os,subprocess,sys,time
marker,root=sys.argv[1:3]
if marker!='cascadia-r2-d0-socket-sampler-v1': raise SystemExit(181)
names=('socket-sampler.pid','socket-sampler.stop','socket-sampler.json')
paths={name:os.path.join(root,name) for name in names}
if any(os.path.lexists(path) for path in paths.values()): raise SystemExit(182)
pid=os.fork()
if pid:
 sys.stdout.write(json.dumps({'pid':pid,'status':'launched'},sort_keys=True,separators=(',',':')))
 raise SystemExit(0)
os.setsid(); null=os.open('/dev/null',os.O_RDWR)
for fd in (0,1,2): os.dup2(null,fd)
def write(path,value):
 payload=json.dumps(value,sort_keys=True,separators=(',',':')).encode()
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 os.write(fd,payload); os.fsync(fd); os.close(fd)
write(paths['socket-sampler.pid'],os.getpid())
deadline=time.monotonic()+30; records=[]; samples=0
while time.monotonic()<deadline and not os.path.exists(paths['socket-sampler.stop']):
 completed=subprocess.run(['/usr/bin/ss','-Hntoape'],stdin=subprocess.DEVNULL,
  stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=1)
 if completed.returncode!=0 or completed.stderr or len(completed.stdout)>1048576: break
 observed_unix_ns=time.time_ns(); samples+=1
 for raw in completed.stdout.splitlines():
  if len(raw)>4096: continue
  text=raw.decode('utf-8','backslashreplace')
  if text and len(records)<1024:
   records.append({'line':text,'observed_unix_ns':observed_unix_ns})
 time.sleep(.005)
write(paths['socket-sampler.json'],{'records':records,'sample_count':samples,'status':'complete'})
"""

SCANNER_SOCKET_SAMPLER_STOP_SCRIPT = r"""import json,os,signal,stat,sys,time
marker,root=sys.argv[1:3]
if marker!='cascadia-r2-d0-socket-sampler-v1': raise SystemExit(191)
pid_path=os.path.join(root,'socket-sampler.pid'); stop_path=os.path.join(root,'socket-sampler.stop')
result_path=os.path.join(root,'socket-sampler.json')
st=os.lstat(pid_path)
if not stat.S_ISREG(st.st_mode) or st.st_uid!=0 or stat.S_IMODE(st.st_mode)!=0o600:
 raise SystemExit(192)
with open(pid_path,encoding='ascii') as stream: pid=json.load(stream)
if not isinstance(pid,int) or isinstance(pid,bool) or pid<=1: raise SystemExit(193)
proc=f'/proc/{pid}'
if os.path.isdir(proc):
 with open(proc+'/cmdline','rb') as stream: parts=stream.read(262145).split(b'\0')
 if marker.encode() not in parts or root.encode() not in parts: raise SystemExit(194)
fd=os.open(stop_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
os.write(fd,b'stop\n'); os.fsync(fd); os.close(fd)
for _ in range(500):
 if os.path.exists(result_path): break
 time.sleep(.01)
else:
 try: os.kill(pid,signal.SIGTERM)
 except ProcessLookupError: pass
 raise SystemExit(195)
st=os.lstat(result_path)
valid=stat.S_ISREG(st.st_mode) and st.st_uid==0
valid=valid and stat.S_IMODE(st.st_mode)==0o600 and st.st_size<=8388608
if not valid:
 raise SystemExit(196)
with open(result_path,encoding='ascii') as stream: result=json.load(stream)
if set(result)!={'records','sample_count','status'} or result.get('status')!='complete':
 raise SystemExit(197)
for path in (result_path,stop_path,pid_path): os.unlink(path)
result['pid']=pid; result['status']='stopped'
sys.stdout.write(json.dumps(result,sort_keys=True,separators=(',',':')))
"""


EGRESS_CONTROL_SOCKET_QUIESCENCE_SCRIPT = r"""import hashlib,json,re,subprocess,sys,time
host,peer,port,seconds=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4])
if not host or not peer or not port.isdigit() or not 1<=int(port)<=65535: raise SystemExit(211)
deadline=time.monotonic()+seconds; samples=0; last=b''; last_matches=[]; consecutive_absent=0
while time.monotonic()<deadline:
 completed=subprocess.run(['/usr/bin/ss','-Hntoa'],stdin=subprocess.DEVNULL,
  stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=1)
 if completed.returncode!=0 or completed.stderr or len(completed.stdout)>1048576:
  raise SystemExit(212)
 samples+=1; last=completed.stdout
 matches=[]
 for raw in last.splitlines():
  if len(raw)>4096: raise SystemExit(213)
  text=raw.decode('utf-8','backslashreplace')
  if host in text and peer in text and ':'+port in text:
   fields=text.split(); timer=re.search(r'timer:\(([^)]*)\)',text)
   timer_fields=timer.group(1).split(',') if timer else None
   state=fields[0] if fields else ''
   safe=state=='TIME-WAIT' and (timer_fields is None or timer_fields[0]=='timewait')
   matches.append({'line':text,'packet_capable':not safe,'state':state,'timer':timer_fields})
 last_matches=matches; packet_capable=[item for item in matches if item['packet_capable']]
 consecutive_absent=consecutive_absent+1 if not matches else 0
 if consecutive_absent>=3:
  result={'last_output_sha256':hashlib.sha256(last).hexdigest(),
   'last_matches':matches,'packet_capable_matches':packet_capable,
   'samples':samples,'status':'absent'}
  sys.stdout.write(json.dumps(result,sort_keys=True,separators=(',',':')))
  raise SystemExit(0)
 time.sleep(.02)
result={'last_output_sha256':hashlib.sha256(last).hexdigest(),
 'last_matches':last_matches,
 'packet_capable_matches':[item for item in last_matches if item['packet_capable']],
 'samples':samples,'status':'timed-out'}
sys.stdout.write(json.dumps(result,sort_keys=True,separators=(',',':')))
"""

EGRESS_CONTROL_PROCESS_CLEANUP_SCRIPT = r"""import json,os,signal,stat,sys,time
root=sys.argv[1]
allowed={'server.pid','client.pid','server-ready.json','established.json','trigger','outcome.json','client-outcome.json'}
markers={'server.pid':'cascadia-r2-d0-egress-server-v2','client.pid':'cascadia-r2-d0-egress-client-v2'}
try: value=os.lstat(root)
except FileNotFoundError:
 print(json.dumps({'processes':[],'state_removed':False,'status':'absent'},sort_keys=True,separators=(',',':')))
 raise SystemExit(0)
valid=stat.S_ISDIR(value.st_mode) and not stat.S_ISLNK(value.st_mode) and value.st_uid==0
if not valid: raise SystemExit(71)
names=os.listdir(root)
for name in names:
 path=os.path.join(root,name); item=os.lstat(path)
 valid=(name in allowed or '.partial-' in name) and stat.S_ISREG(item.st_mode) and item.st_uid==0
 if not valid: raise SystemExit(72)
processes=[]
for name in ('server.pid','client.pid'):
 path=os.path.join(root,name)
 try:
  with open(path,encoding='ascii') as stream: identity=json.load(stream)
  marker=markers[name]
  if set(identity)!={'marker','pid'} or identity.get('marker')!=marker: raise SystemExit(73)
  pid=identity.get('pid')
  if not isinstance(pid,int) or isinstance(pid,bool) or pid<=1: raise SystemExit(74)
  proc='/proc/'+str(pid); present=os.path.isdir(proc)
  row={'initially_present':present,'marker':marker,'pid':pid,'sigkill_sent':False,'sigterm_sent':False}
  if present:
   with open(proc+'/cmdline','rb') as stream: parts=stream.read(262145).split(b'\0')
   if marker.encode() not in parts or root.encode() not in parts: raise SystemExit(75)
   try: os.kill(pid,signal.SIGTERM); row['sigterm_sent']=True
   except ProcessLookupError: pass
   deadline=time.monotonic()+2
   while os.path.isdir(proc) and time.monotonic()<deadline: time.sleep(.01)
   if os.path.isdir(proc):
    try: os.kill(pid,signal.SIGKILL); row['sigkill_sent']=True
    except ProcessLookupError: pass
    deadline=time.monotonic()+2
    while os.path.isdir(proc) and time.monotonic()<deadline: time.sleep(.01)
   if os.path.isdir(proc): raise SystemExit(76)
  row['absent']=True; processes.append(row)
 except FileNotFoundError: pass
for name in os.listdir(root): os.unlink(os.path.join(root,name))
os.rmdir(root)
print(json.dumps({'processes':processes,'state_removed':True,'status':'clean'},sort_keys=True,separators=(',',':')))
"""


SCANNER_ATTESTATION_CLEANUP_SCRIPT = r"""import hashlib,json,subprocess,sys,time
reference=sys.argv[1]; expected=set(sys.argv[2:]); commands=[]; mutation_started=False
base=['/usr/bin/ctr','--namespace','moby']
class Failure(Exception): pass
def fail(stage,message): raise Failure(stage+'|'+message)
def run(stage,arguments):
 value=subprocess.run(base+arguments,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
  stderr=subprocess.PIPE,timeout=30)
 record={'argv':base+arguments,'returncode':value.returncode,
  'stderr_sha256':hashlib.sha256(value.stderr).hexdigest(),
  'stderr_size':len(value.stderr),'stdout_sha256':hashlib.sha256(value.stdout).hexdigest(),
  'stdout_size':len(value.stdout),'stage':stage}
 commands.append(record)
 if value.returncode!=0 or value.stderr or len(value.stdout)>1048576:
  fail(stage,'command-failed')
 return value.stdout
def first_column(stage,kind):
 lines=run(stage,[kind,'list']).decode('ascii').splitlines()
 if not lines: fail(stage,'missing-header')
 return set(line.split()[0] for line in lines[1:] if line.split())
try:
 if not reference.startswith('moby-dangling@sha256:') or len(expected)!=4:
  fail('validate-arguments','reference-or-cardinality')
 if any(not item.startswith('sha256:') or len(item)!=71 for item in expected):
  fail('validate-arguments','digest')
 refs=first_column('pre-images','images'); content=first_column('pre-content','content')
 containers=first_column('pre-containers','containers')
 snapshots=first_column('pre-snapshots','snapshots')
 leases=first_column('pre-leases','leases')
 if refs!={reference} or content!=expected or containers or snapshots or leases:
  fail('validate-precondition','inventory')
 mutation_started=True
 removed_image=run('remove-image',['images','remove','--sync',reference])
 remaining=first_column('post-image-content','content')
 if not remaining<=expected: fail('post-image-content','unexpected-content')
 removed_content=b''
 if remaining:
  removed_content=run('remove-content',['content','remove',*sorted(remaining)])
 for index in range(100):
  refs_after=first_column('poll-images-'+str(index),'images')
  content_after=first_column('poll-content-'+str(index),'content')
  if not refs_after and not content_after: break
  time.sleep(.02)
 else: fail('poll-empty','timed-out')
 if first_column('post-containers','containers'):
  fail('post-containers','not-empty')
 if first_column('post-snapshots','snapshots'): fail('post-snapshots','not-empty')
 if first_column('post-leases','leases'): fail('post-leases','not-empty')
 result={'commands':commands,'content_after':[],'content_before':sorted(content),
  'content_remove_stdout_sha256':hashlib.sha256(removed_content).hexdigest(),
  'image_remove_stdout_sha256':hashlib.sha256(removed_image).hexdigest(),
  'images_after':[],'images_before':sorted(refs),'mutation_started':mutation_started,
  'stage':'complete','status':'clean'}
except (Failure,subprocess.TimeoutExpired,UnicodeDecodeError) as error:
 result={'commands':commands,'error':type(error).__name__+':'+str(error),
  'mutation_started':mutation_started,'stage':str(error).split('|',1)[0],
  'status':'failed'}
sys.stdout.write(json.dumps(result,sort_keys=True,separators=(',',':')))
"""


SCANNER_REGISTRY_SERVER_SCRIPT = r"""import hashlib,http.server,json,os,socketserver,ssl,sys
import threading
import urllib.parse
marker,root,host,port,repository=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]),sys.argv[5]
manifest_digest,config_digest,layer_digest=sys.argv[6:9]
if marker!='cascadia-r2-d0-scanner-registry-server-v1': raise SystemExit(141)
files={
 f'/v2/{repository}/manifests/{manifest_digest}':(
  os.path.join(root,'manifest.blob'),'application/vnd.oci.image.manifest.v1+json',
  manifest_digest,
 ),
 f'/v2/{repository}/blobs/{config_digest}':(
  os.path.join(root,'config.blob'),'application/octet-stream',config_digest,
 ),
 f'/v2/{repository}/blobs/{layer_digest}':(
  os.path.join(root,'layer.blob'),'application/octet-stream',layer_digest,
 ),
}
request_root=os.path.join(root,'requests'); lock=threading.Lock(); sequence=0
def write(path,value):
 payload=json.dumps(value,sort_keys=True,separators=(',',':')).encode()
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 os.write(fd,payload); os.fsync(fd); os.close(fd)
def record(handler,method,raw_path,path,status,digest):
 global sequence
 protocol=handler.connection.version(); cipher=handler.connection.cipher()
 with lock:
  sequence+=1
  name=os.path.join(request_root,f'{sequence:06d}.json')
  write(name,{
   'body_digest':digest,'method':method,'path':path,'raw_path':raw_path,
   'sequence':sequence,'status':status,'tls_cipher':cipher[0] if cipher else None,
   'tls_protocol':protocol,
  })
class Server(socketserver.ThreadingMixIn,http.server.HTTPServer):
 daemon_threads=True
 allow_reuse_address=True
class Handler(http.server.BaseHTTPRequestHandler):
 protocol_version='HTTP/1.1'
 def log_message(self,*args): pass
 def dispatch(self,send_body):
  parsed=urllib.parse.urlsplit(self.path)
  path=urllib.parse.unquote(parsed.path)
  entry=files.get(path); status=200; digest=None; payload=b'{}'; kind='application/json'
  if parsed.query or parsed.fragment: status=404; payload=b''
  elif path=='/v2/': pass
  elif entry is None: status=404; payload=b''
  else:
   source,kind,digest=entry
   with open(source,'rb') as stream: payload=stream.read(268435457)
   if len(payload)>268435456 or 'sha256:'+hashlib.sha256(payload).hexdigest()!=digest:
    status=500; payload=b''
  self.send_response(status)
  self.send_header('Content-Type',kind)
  self.send_header('Content-Length',str(len(payload)))
  self.send_header('Docker-Distribution-Api-Version','registry/2.0')
  if digest is not None: self.send_header('Docker-Content-Digest',digest)
  self.end_headers()
  if send_body and payload:
   try: self.wfile.write(payload)
   except (BrokenPipeError,ConnectionResetError): pass
  record(self,self.command,self.path,path,status,digest)
 def reject(self):
  parsed=urllib.parse.urlsplit(self.path); path=urllib.parse.unquote(parsed.path)
  self.send_response(405); self.send_header('Content-Length','0'); self.end_headers()
  record(self,self.command,self.path,path,405,None)
 def do_GET(self): self.dispatch(True)
 def do_HEAD(self): self.dispatch(False)
 def do_POST(self): self.reject()
 def do_PUT(self): self.reject()
 def do_DELETE(self): self.reject()
pid=os.fork()
if pid:
 sys.stdout.write(json.dumps({'pid':pid,'status':'launched'},sort_keys=True,separators=(',',':')))
 raise SystemExit(0)
os.setsid(); null=os.open('/dev/null',os.O_RDWR)
for fd in (0,1,2): os.dup2(null,fd)
server=Server((host,port),Handler)
context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.minimum_version=ssl.TLSVersion.TLSv1_2
context.load_cert_chain(os.path.join(root,'registry-server.crt'),os.path.join(root,'registry-server.key'))
server.socket=context.wrap_socket(server.socket,server_side=True)
write(os.path.join(root,'server.pid'),os.getpid())
write(os.path.join(root,'server-ready.json'),{
 'host':host,'pid':os.getpid(),'port':port,'repository':repository,'status':'ready',
})
server.serve_forever(poll_interval=.1)
"""


SCANNER_REGISTRY_CLEANUP_SCRIPT = r"""import glob,hashlib,json,os,signal,socket,stat,sys,time
root,host,port,repository=sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4]
manifest_digest,config_digest,layer_digest=sys.argv[5:8]
marker=b'cascadia-r2-d0-scanner-registry-server-v1'
sampler_marker=b'cascadia-r2-d0-socket-sampler-v1'
allowed_paths={
 '/v2/',f'/v2/{repository}/manifests/{manifest_digest}',
 f'/v2/{repository}/blobs/{config_digest}',f'/v2/{repository}/blobs/{layer_digest}',
}
required_requests={f'/v2/{repository}/manifests/{manifest_digest}'}
if not os.path.lexists(root):
 probe=socket.socket(); probe.settimeout(.2)
 listener_absent=probe.connect_ex((host,port))!=0; probe.close()
 orphan_pids=[]
 ancestors={os.getpid()}; parent=os.getppid()
 while parent>1 and parent not in ancestors:
  ancestors.add(parent)
  try:
   with open(f'/proc/{parent}/status',encoding='ascii') as stream: lines=stream.readlines()
   parent=int(next(line for line in lines if line.startswith('PPid:')).split()[1])
  except (FileNotFoundError,PermissionError,StopIteration,ValueError): break
 for proc in glob.glob('/proc/[0-9]*/cmdline'):
  try:
   candidate=int(proc.split('/')[2])
   if candidate not in ancestors:
    with open(proc,'rb') as stream: command=stream.read(262145)
    parts=command.split(b'\0')
    valid=len(command)<=262144 and (marker in parts or sampler_marker in parts)
    if valid and root.encode() in parts:
     orphan_pids.append(candidate)
  except (FileNotFoundError,PermissionError,ValueError): pass
 if not listener_absent or orphan_pids: raise SystemExit(139)
 result={
  'host':host,'listener_absent':True,'orphan_pids':[],'pid':None,'port':port,
  'record_count':0,'record_sha256':hashlib.sha256(b'[]').hexdigest(),
  'requests_valid':False,'root_absent':True,'served_paths':[],'status':'clean',
  'unexpected_requests':[],
 }
 sys.stdout.write(json.dumps(result,sort_keys=True,separators=(',',':')))
 raise SystemExit(0)
st=os.lstat(root)
if not stat.S_ISDIR(st.st_mode) or st.st_uid!=0 or stat.S_IMODE(st.st_mode)!=0o700:
 raise SystemExit(131)
pid_path=os.path.join(root,'server.pid'); pid=None
if os.path.lexists(pid_path):
 st=os.lstat(pid_path)
 if not stat.S_ISREG(st.st_mode) or st.st_uid!=0 or stat.S_IMODE(st.st_mode)!=0o600:
  raise SystemExit(132)
 with open(pid_path,encoding='ascii') as stream: pid=json.load(stream)
 if not isinstance(pid,int) or pid<=1: raise SystemExit(133)
 proc=f'/proc/{pid}'
 if os.path.isdir(proc):
  with open(proc+'/cmdline','rb') as stream: cmdline=stream.read(262145)
  parts=cmdline.split(b'\0')
  valid=len(cmdline)<=262144 and marker in parts and root.encode() in parts
  valid=valid and str(port).encode() in parts and repository.encode() in parts
  if not valid: raise SystemExit(134)
  try: os.kill(pid,signal.SIGTERM)
  except ProcessLookupError: pass
  for _ in range(100):
   if not os.path.isdir(proc): break
   with open(proc+'/stat',encoding='ascii') as stream: fields=stream.read(4096).split()
   if len(fields)>=3 and fields[2]=='Z': break
   time.sleep(.01)
  else: os.kill(pid,signal.SIGKILL)
probe=socket.socket(); probe.settimeout(.2)
listener_absent=probe.connect_ex((host,port))!=0; probe.close()
if not listener_absent: raise SystemExit(137)
ancestors={os.getpid()}; parent=os.getppid()
while parent>1 and parent not in ancestors:
 ancestors.add(parent)
 try:
  with open(f'/proc/{parent}/status',encoding='ascii') as stream: lines=stream.readlines()
  parent=int(next(line for line in lines if line.startswith('PPid:')).split()[1])
 except (FileNotFoundError,PermissionError,StopIteration,ValueError): break
orphan_pids=[]
for proc in glob.glob('/proc/[0-9]*/cmdline'):
 try:
  candidate=int(proc.split('/')[2])
  if candidate not in ancestors:
   with open(proc,'rb') as stream: command=stream.read(262145)
   parts=command.split(b'\0')
   valid=len(command)<=262144 and (marker in parts or sampler_marker in parts)
   if valid and root.encode() in parts:
    orphan_pids.append(candidate)
 except (FileNotFoundError,PermissionError,ValueError): pass
if orphan_pids: raise SystemExit(138)
request_root=os.path.join(root,'requests'); records=[]; unexpected=[]
if os.path.isdir(request_root):
 for name in sorted(os.listdir(request_root)):
  path=os.path.join(request_root,name); st=os.lstat(path)
  valid=stat.S_ISREG(st.st_mode) and st.st_uid==0 and stat.S_IMODE(st.st_mode)==0o600
  if not valid or st.st_size>65536: raise SystemExit(135)
  with open(path,encoding='ascii') as stream: value=json.load(stream)
  records.append(value)
  valid=set(value)=={
   'body_digest','method','path','raw_path','sequence','status','tls_cipher','tls_protocol',
  }
  valid=valid and value.get('method') in {'GET','HEAD'} and value.get('status')==200
  valid=valid and value.get('path') in allowed_paths
  valid=valid and value.get('tls_protocol') in {'TLSv1.2','TLSv1.3'}
  valid=valid and isinstance(value.get('tls_cipher'),str) and bool(value.get('tls_cipher'))
  if not valid: unexpected.append(value)
served_paths={item.get('path') for item in records}
requests_valid=not unexpected and required_requests<=served_paths
record_sha=hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest()
known={
 'scanner.oci.tar','manifest.blob','config.blob','layer.blob','server.pid',
 'server-ready.json','requests','trust-baseline.json','registry-ca.crt',
 'registry-server.crt','registry-server.key',
 'socket-sampler.pid','socket-sampler.stop','socket-sampler.json',
}
entries=set(os.listdir(root)); unknown=entries-known
if unknown: raise SystemExit(136)
if os.path.isdir(request_root):
 for name in os.listdir(request_root): os.unlink(os.path.join(request_root,name))
 os.rmdir(request_root)
for name in sorted(known-{'requests'}):
 path=os.path.join(root,name)
 if os.path.lexists(path): os.unlink(path)
os.rmdir(root)
directory=os.open(os.path.dirname(root),os.O_RDONLY|os.O_DIRECTORY)
os.fsync(directory); os.close(directory)
result={
 'host':host,'listener_absent':listener_absent,'orphan_pids':orphan_pids,
 'pid':pid,'port':port,'record_count':len(records),
 'record_sha256':record_sha,'requests_valid':requests_valid,'root_absent':True,
 'served_paths':sorted(served_paths),'status':'clean','unexpected_requests':unexpected,
}
sys.stdout.write(json.dumps(result,sort_keys=True,separators=(',',':')))
"""
EGRESS_SERVER_SCRIPT_SHA256 = sha256_bytes(_EGRESS_SERVER_SCRIPT.encode("utf-8"))
EGRESS_CLIENT_SCRIPT_SHA256 = sha256_bytes(_EGRESS_CLIENT_SCRIPT.encode("utf-8"))
REQUIRED_OPERATION = {
    "preflight": "preflight-audit",
    "install": "install-runtime",
    "start": "start-runtime",
    "verify": "verify-runtime",
    "rollback": "rollback-runtime",
    "postflight": "postflight-audit",
}


@dataclass(frozen=True)
class Command:
    argv: tuple[str, ...]
    purpose: str
    stdin_identity: str | None = None
    stdout_kind: str = "text"

    def as_dict(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "purpose": self.purpose,
            "stdin_identity": self.stdin_identity,
            "stdout_kind": self.stdout_kind,
        }


@dataclass(frozen=True)
class Completed:
    argv: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes


def _bounded_reap(process: subprocess.Popen[bytes], *, deadline: float, label: str) -> None:
    """Terminate and reap a subprocess without ever outliving the signed phase budget."""

    if process.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    remaining = min(5.0, max(0.0, deadline - time.monotonic()))
    if remaining > 0:
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=remaining)
    if process.poll() is None:
        raise D0Error(f"command could not be reaped within its signed limit: {label}")


class CommandRunner:
    """Bounded argv-only subprocess runner; no shell and no inherited environment."""

    def __init__(
        self,
        environment: Mapping[str, str],
        *,
        timeout_seconds: int = 300,
        output_max_bytes: int = MAX_CAPTURE_BYTES,
        cleanup_reserve_seconds: int = 0,
        _deadline: float | None = None,
        _cleanup_mode: bool = False,
        _capture_state: dict[str, int] | None = None,
    ):
        self.environment = dict(environment)
        if (
            timeout_seconds <= 0
            or output_max_bytes <= 0
            or cleanup_reserve_seconds < 0
            or cleanup_reserve_seconds >= timeout_seconds
        ):
            raise D0Error("command runner limits are invalid")
        self.timeout_seconds = timeout_seconds
        self.output_max_bytes = output_max_bytes
        self.cleanup_reserve_seconds = cleanup_reserve_seconds
        self._deadline = (
            time.monotonic() + timeout_seconds if _deadline is None else float(_deadline)
        )
        self._cleanup_mode = _cleanup_mode
        self._capture_state = {"bytes": 0} if _capture_state is None else _capture_state

    def cleanup_runner(self) -> CommandRunner:
        """Return a view that may consume the phase time reserved for cleanup."""

        if self._cleanup_mode:
            return self
        return CommandRunner(
            self.environment,
            timeout_seconds=self.timeout_seconds,
            output_max_bytes=self.output_max_bytes,
            cleanup_reserve_seconds=0,
            _deadline=self._deadline,
            _cleanup_mode=True,
            _capture_state=self._capture_state,
        )

    def run(
        self,
        argv: Sequence[str],
        *,
        stdin: bytes = b"",
        timeout: int | None = None,
        maximum: int | None = None,
        check: bool = True,
    ) -> Completed:
        if not argv or any(not isinstance(item, str) or "\0" in item for item in argv):
            raise D0Error("command argv is invalid")
        selected_timeout = self.timeout_seconds if timeout is None else timeout
        selected_maximum = self.output_max_bytes if maximum is None else maximum
        if (
            selected_timeout <= 0
            or selected_timeout > self.timeout_seconds
            or selected_maximum <= 0
            or selected_maximum > self.output_max_bytes
        ):
            raise D0Error("command exceeds its signed time or output limit")
        remaining = self._deadline - time.monotonic()
        if not self._cleanup_mode:
            remaining -= self.cleanup_reserve_seconds
        if remaining <= 0:
            label = "primary-work budget" if not self._cleanup_mode else "signed wall-clock limit"
            raise D0Error(f"phase exceeded its {label}")
        effective_timeout = min(float(selected_timeout), remaining)
        try:
            process = subprocess.Popen(
                list(argv),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.environment,
                start_new_session=True,
            )
        except OSError as error:
            raise D0Error(f"command could not execute: {argv[0]}") from error
        stdout = bytearray()
        stderr = bytearray()
        failure: D0Error | None = None
        selector = selectors.DefaultSelector()
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        try:
            for stream, target in ((process.stdout, stdout), (process.stderr, stderr)):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, target)
            if stdin:
                os.set_blocking(process.stdin.fileno(), False)
                selector.register(process.stdin, selectors.EVENT_WRITE, memoryview(stdin))
            else:
                process.stdin.close()
            command_deadline = time.monotonic() + effective_timeout
            while selector.get_map():
                remaining = command_deadline - time.monotonic()
                if remaining <= 0:
                    failure = D0Error(f"command timed out: {argv[0]}")
                    break
                events = selector.select(min(remaining, 0.25))
                for key, _mask in events:
                    stream = key.fileobj
                    if key.events == selectors.EVENT_WRITE:
                        pending = key.data
                        try:
                            written = os.write(stream.fileno(), pending[:65536])
                        except BrokenPipeError:
                            written = len(pending)
                        pending = pending[written:]
                        if not pending:
                            selector.unregister(stream)
                            stream.close()
                        else:
                            selector.modify(stream, selectors.EVENT_WRITE, pending)
                        continue
                    try:
                        chunk = os.read(stream.fileno(), 65536)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(stream)
                        stream.close()
                        continue
                    target = key.data
                    if len(stdout) + len(stderr) + len(chunk) > selected_maximum:
                        failure = D0Error("command output exceeds its cumulative byte limit")
                        break
                    if self._capture_state["bytes"] + len(chunk) > self.output_max_bytes:
                        failure = D0Error("phase output exceeds its signed cumulative byte limit")
                        break
                    self._capture_state["bytes"] += len(chunk)
                    target.extend(chunk)
                if failure is not None:
                    break
            if failure is None:
                remaining = command_deadline - time.monotonic()
                if remaining <= 0:
                    failure = D0Error(f"command timed out: {argv[0]}")
                else:
                    try:
                        process.wait(timeout=remaining)
                    except subprocess.TimeoutExpired:
                        failure = D0Error(f"command timed out: {argv[0]}")
        except OSError as error:
            failure = D0Error(f"command I/O failed: {argv[0]}")
            failure.__cause__ = error
        finally:
            selector.close()
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()
            try:
                _bounded_reap(process, deadline=self._deadline, label=argv[0])
            except D0Error as reap_error:
                if failure is not None:
                    reap_error.__cause__ = failure
                failure = reap_error
        if failure is not None:
            raise failure
        completed = Completed(tuple(argv), process.returncode, bytes(stdout), bytes(stderr))
        if check and completed.returncode != 0:
            stderr_sha256 = sha256_bytes(completed.stderr)
            stderr_preview = completed.stderr[:512].decode("utf-8", errors="backslashreplace")
            raise D0Error(
                "command failed: "
                f"argv={json.dumps(list(argv), separators=(',', ':'))} "
                f"returncode={completed.returncode} stderr_sha256={stderr_sha256} "
                f"stderr_preview={stderr_preview!r}"
            )
        return completed


def _runtime_environment_contract(packet: Mapping[str, Any]) -> dict[str, str]:
    validate_work_packet(packet)
    paths = packet["paths"]
    return {
        "HOME": HOST_HOME[packet["host"]],
        "PATH": "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "LC_ALL": "C",
        "LANG": "C",
        "TZ": "UTC",
        "TMPDIR": paths["homebrew_temp"],
        "HOMEBREW_NO_AUTO_UPDATE": "1",
        "HOMEBREW_NO_ANALYTICS": "1",
        "HOMEBREW_NO_INSTALL_CLEANUP": "1",
        "HOMEBREW_NO_ENV_HINTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "HOMEBREW_CACHE": paths["homebrew_cache"],
        "HOMEBREW_LOGS": paths["homebrew_logs"],
        "HOMEBREW_TEMP": paths["homebrew_temp"],
        "COLIMA_HOME": paths["colima_home"],
        "COLIMA_CACHE_HOME": paths["colima_cache_home"],
        "COLIMA_PROFILE": PROFILE,
        "COLIMA_SAVE_CONFIG": "0",
        "DOCKER_CONFIG": paths["docker_config"],
        "DOCKER_HOST": f"unix://{paths['colima_home']}/{PROFILE}/docker.sock",
    }


def validate_explicit_runtime_environment(
    packet: Mapping[str, Any], environment: Mapping[str, str]
) -> dict[str, str]:
    """Require the complete, exact isolated runtime environment.

    Recovery and live-diagnostic commands must never inherit Colima or Docker
    defaults.  A missing ``COLIMA_HOME`` previously allowed Colima to create a
    second profile beneath ``~/.colima``.  Exact equality makes that class of
    failure impossible: missing, extra, or redirected variables all fail
    before any subprocess can run.
    """

    expected = _runtime_environment_contract(packet)
    observed = dict(environment)
    if observed != expected:
        raise D0Error("explicit isolated runtime environment differs")
    colima_home = expected["COLIMA_HOME"]
    docker_config = expected["DOCKER_CONFIG"]
    docker_host = expected["DOCKER_HOST"]
    if not colima_home or not docker_config or not docker_host:
        raise D0Error("explicit isolated runtime environment is incomplete")
    expected_socket = f"unix://{colima_home}/{PROFILE}/docker.sock"
    if docker_host != expected_socket:
        raise D0Error("explicit Docker endpoint differs from isolated Colima home")
    return expected


def runtime_environment(packet: Mapping[str, Any]) -> dict[str, str]:
    return validate_explicit_runtime_environment(packet, _runtime_environment_contract(packet))


def global_dependency_environment(host: str) -> dict[str, str]:
    """Return a probe environment that cannot create campaign runtime state."""

    if host not in HOST_HOME:
        raise D0Error("unknown D0 host")
    return {
        "HOME": HOST_HOME[host],
        "PATH": "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "LC_ALL": "C",
        "LANG": "C",
        "TZ": "UTC",
        "HOMEBREW_NO_AUTO_UPDATE": "1",
        "HOMEBREW_NO_ANALYTICS": "1",
        "HOMEBREW_NO_INSTALL_CLEANUP": "1",
        "HOMEBREW_NO_ENV_HINTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
    }


def prepare_runtime_environment_paths(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Create or verify isolated temp/cache/log roots before runtime subprocesses."""

    validate_work_packet(packet)
    paths = [
        Path(packet["paths"]["homebrew_cache"]),
        Path(packet["paths"]["homebrew_logs"]),
        Path(packet["paths"]["homebrew_temp"]),
    ]
    dispositions: list[dict[str, str]] = []
    may_create = packet["phase"] in {"install", "rollback"}
    for path in paths:
        existed = path.exists() or path.is_symlink()
        if not existed and not may_create:
            raise D0Error(f"isolated runtime environment path is absent: {path}")
        secure_owner_directory(path)
        dispositions.append({"path": str(path), "disposition": "present" if existed else "created"})
    return {"paths": dispositions, "status": "pass"}


def formulas_for_host(host: str) -> tuple[str, ...]:
    if host == "john2":
        return FORMULA_ORDER
    if host in {"john1", "john3"}:
        return FORMULA_ORDER[:-1]
    raise D0Error("unknown D0 host")


def homebrew_ledger_paths(home: Path, host: str, packet: Mapping[str, Any]) -> list[Path]:
    paths = packet["paths"]
    roots = [
        home / "Library/Caches/Homebrew",
        home / "Library/Logs/Homebrew",
        Path("/opt/homebrew/Cellar"),
        Path("/opt/homebrew/opt"),
        Path("/opt/homebrew/bin"),
        Path("/opt/homebrew/sbin"),
        Path("/opt/homebrew/lib"),
        Path("/opt/homebrew/include"),
        Path("/opt/homebrew/share"),
        Path("/opt/homebrew/etc"),
        Path("/opt/homebrew/Frameworks"),
        Path("/opt/homebrew/var/homebrew/linked"),
        Path("/opt/homebrew/var/homebrew/locks"),
        Path(paths["homebrew_cache"]),
        Path(paths["homebrew_logs"]),
        Path(paths["homebrew_temp"]),
    ]
    return list(dict.fromkeys(roots))


def preflight_audit(
    packet: Mapping[str, Any],
    runner: CommandRunner,
    *,
    home: Path,
) -> dict[str, Any]:
    """Freeze the no-follow selected-runtime, Homebrew, and Podman baseline."""

    validate_work_packet(packet)
    required = REQUIRED_OPERATION["preflight"]
    if packet["phase"] != "preflight" or required not in packet["allowed_operations"]:
        raise D0Error("work packet does not authorize preflight")
    selected = inventory_roots(
        selected_runtime_paths(home),
        label=f"{packet['run_id']}-{packet['host']}-campaign-runtime-pre",
    )
    if selected["totals"]["present_roots"] != 0:
        raise D0Error("campaign-owned runtime state is already present")
    activity = _inactive_runtime_activity(packet)
    if not activity["inactive"]:
        raise D0Error("a container or VM runtime process/listener/mount is active")
    resources = host_resource_snapshot(runner)
    if resources["swap_used_bytes"] != 0:
        raise D0Error("preflight requires zero host swap use")
    paths = packet["paths"]
    isolated_paths = [
        Path(paths["colima_home"]),
        Path(paths["colima_cache_home"]),
        Path(paths["docker_config"]),
        Path(paths["homebrew_cache"]),
        Path(paths["homebrew_logs"]),
        Path(paths["homebrew_temp"]),
    ]
    if packet["host"] in {"john1", "john3"}:
        bootstrap_roots = {
            Path(paths["core_image"]).parent,
            Path(paths["smoke_oci"]).parent,
            Path(paths["homebrew_closure"]).parent,
            Path(paths["runtime_supply"]).parent,
        }
        if len(bootstrap_roots) != 1:
            raise D0Error("worker bootstrap artifacts do not share one atomic root")
        isolated_paths.append(bootstrap_roots.pop())
        isolated_paths.append(Path(paths["runtime_supply_inbox"]).parent)
    isolated_baseline = inventory_roots(
        isolated_paths,
        label=f"{packet['run_id']}-{packet['host']}-isolated-runtime-pre",
    )
    if isolated_baseline["totals"]["present_roots"] != 0:
        raise D0Error("D0 isolated runtime or staging root pre-exists")
    result: dict[str, Any] = {
        "selected_runtime": selected,
        "global_runtime_dependencies": frozen_global_runtime_dependencies(
            packet,
            CommandRunner(global_dependency_environment(packet["host"])),
        ),
        "runtime_activity": activity,
        "homebrew": inventory_roots(
            homebrew_ledger_paths(home, packet["host"], packet),
            label=f"{packet['run_id']}-{packet['host']}-homebrew-pre",
            policy=InventoryPolicy(full_hash_limit=64 * 1024 * 1024, max_entries=500_000),
        ),
        "resources": resources,
        "platform": host_platform_snapshot(packet, runner),
        "isolated_runtime_baseline": isolated_baseline,
    }
    free_bytes = resources["data_volume_kib_available"] * 1024
    effective = min(RUNTIME_MAX_BYTES, (free_bytes * FREE_FRACTION_PPM) // 1_000_000)
    if effective < RUNTIME_MAX_BYTES:
        raise D0Error("host lacks enough audited free space for the frozen runtime ceiling")
    result["runtime_budget_preflight"] = {
        "free_bytes": free_bytes,
        "absolute_limit_bytes": RUNTIME_MAX_BYTES,
        "free_fraction_limit_bytes": (free_bytes * FREE_FRACTION_PPM) // 1_000_000,
        "effective_limit_bytes": effective,
        "status": "pass",
    }
    if packet["host"] == "john1":
        result["podman_negative_control"] = podman_negative_control(home)
    return result


def verify_formula_metadata(value: bytes, formula: str) -> dict[str, Any]:
    metadata = validate_homebrew_formula_projection(value, formula)
    bottle = metadata["bottle"]
    return {
        "formula": formula,
        "version": metadata["version"],
        "license": metadata["license"],
        "bottle_sha256": bottle["sha256"],
        "bottle_url": bottle["url"],
        "projection_size": len(value),
        "projection_sha256": sha256_bytes(value),
        "revision": metadata["revision"],
        "dependencies": metadata["dependencies"],
        "formula_path": metadata["formula_path"],
        "ruby_source_sha256": metadata["ruby_source_sha256"],
        "reviewed_tap_git_head": metadata["reviewed_tap_git_head"],
    }


def verify_homebrew_installer(runner: CommandRunner) -> dict[str, Any]:
    identity = _regular_file_identity(Path(BREW))
    if (
        identity["size"] != FROZEN_HOMEBREW["executable_size"]
        or identity["sha256"] != FROZEN_HOMEBREW["executable_sha256"]
        or stat.S_IMODE(Path(BREW).lstat().st_mode) != 0o755
    ):
        raise D0Error("Homebrew installer bytes or metadata drifted")
    version = runner.run([BREW, "--version"], maximum=64 * 1024).stdout.decode().strip()
    repository = runner.run([BREW, "--repository"], maximum=64 * 1024).stdout.decode().strip()
    if version != FROZEN_HOMEBREW["version_line"] or repository != FROZEN_HOMEBREW["repository"]:
        raise D0Error("Homebrew installer version or repository drifted")
    head = (
        runner.run(["/usr/bin/git", "-C", repository, "rev-parse", "HEAD"], maximum=64 * 1024)
        .stdout.decode()
        .strip()
    )
    origin = (
        runner.run(
            ["/usr/bin/git", "-C", repository, "config", "--get", "remote.origin.url"],
            maximum=64 * 1024,
        )
        .stdout.decode()
        .strip()
    )
    if (
        head != FROZEN_HOMEBREW["repository_git_head"]
        or origin != FROZEN_HOMEBREW["repository_origin"]
    ):
        raise D0Error("Homebrew source identity drifted")
    cleanliness_commands = (
        ["/usr/bin/git", "-C", repository, "diff", "--no-ext-diff", "--quiet"],
        [
            "/usr/bin/git",
            "-C",
            repository,
            "diff",
            "--no-ext-diff",
            "--cached",
            "--quiet",
        ],
    )
    for command in cleanliness_commands:
        runner.run(command, maximum=64 * 1024)
    status_output = runner.run(
        [
            "/usr/bin/git",
            "-C",
            repository,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        maximum=1024 * 1024,
    ).stdout
    if status_output:
        raise D0Error("Homebrew source tree is not clean")
    return {
        **identity,
        "version": version,
        "repository": repository,
        "git_head": head,
        "origin": origin,
        "tracked_diff_empty": True,
        "cached_diff_empty": True,
        "untracked_status_empty": True,
    }


def homebrew_plan(packet: Mapping[str, Any]) -> list[Command]:
    validate_work_packet(packet)
    commands: list[Command] = [
        Command((BREW, "--version"), "verify-homebrew-version"),
        Command((BREW, "--repository"), "verify-homebrew-repository"),
    ]
    for formula in formulas_for_host(packet["host"]):
        commands.append(
            Command(
                (BREW, "list", "--versions", formula),
                f"verify-pre-existing-frozen-{formula}",
            )
        )
    return commands


def _artifact_bottles(packet: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result = {item["name"]: item for item in packet["artifacts"]["bottles"]}
    required = set(formulas_for_host(packet["host"]))
    if set(result) != required:
        raise D0Error("work packet bottle set differs from the host role")
    return result


def _resolved_bottle_path(value: bytes, cache_root: Path) -> Path:
    try:
        text = value.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise D0Error("Homebrew bottle path is not text") from error
    if not text or "\n" in text:
        raise D0Error("Homebrew returned more than one bottle path")
    path = Path(text)
    if not path.is_absolute():
        raise D0Error("Homebrew bottle path is not absolute")
    try:
        path.relative_to(cache_root)
    except ValueError as error:
        raise D0Error("Homebrew bottle escaped the isolated cache") from error
    return path


def _regular_file_identity(path: Path) -> dict[str, Any]:
    try:
        observed = path.lstat()
    except OSError as error:
        raise D0Error(f"cannot inspect verified artifact: {path}") from error
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_uid != os.getuid()
        or observed.st_nlink != 1
    ):
        raise D0Error(f"verified artifact metadata is unsafe: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    size = 0
    try:
        reopened = os.fstat(descriptor)
        if (reopened.st_dev, reopened.st_ino, reopened.st_size) != (
            observed.st_dev,
            observed.st_ino,
            observed.st_size,
        ):
            raise D0Error("artifact changed while opening")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    finally:
        os.close(descriptor)
    return {"path": str(path), "size": size, "sha256": digest.hexdigest()}


def frozen_global_runtime_dependencies(
    packet: Mapping[str, Any], runner: CommandRunner
) -> dict[str, Any]:
    """Positively identify shared, immutable Homebrew runtime dependencies."""

    validate_work_packet(packet)
    formulae: list[dict[str, Any]] = []
    for formula in formulas_for_host(packet["host"]):
        version = FROZEN_RUNTIME[formula]["version"]
        listed = runner.run([BREW, "list", "--versions", formula]).stdout.decode().strip()
        if listed != f"{formula} {version}":
            raise D0Error(f"pre-existing Homebrew version differs for {formula}")
        cellar = Path("/opt/homebrew/Cellar") / formula / version
        try:
            cellar_metadata = cellar.lstat()
        except OSError as error:
            raise D0Error(f"pre-existing Homebrew Cellar root is absent for {formula}") from error
        if stat.S_ISLNK(cellar_metadata.st_mode) or not stat.S_ISDIR(cellar_metadata.st_mode):
            raise D0Error(f"pre-existing Homebrew Cellar root is unsafe for {formula}")
        entrypoint = GLOBAL_RUNTIME_ENTRYPOINTS[formula]
        try:
            resolved = Path(os.path.realpath(entrypoint))
            resolved.relative_to(cellar)
        except (OSError, ValueError) as error:
            raise D0Error(
                f"global runtime entrypoint escapes the frozen Cellar for {formula}"
            ) from error
        identity = _regular_file_identity(resolved)
        formulae.append(
            {
                "formula": formula,
                "version": version,
                "version_line": listed,
                "cellar": str(cellar),
                "entrypoint": str(entrypoint),
                "resolved_entrypoint": str(resolved),
                "entrypoint_identity": identity,
                "disposition": "pre-existing-immutable-dependency",
            }
        )
    result: dict[str, Any] = {
        "formulae": formulae,
        "formulae_mutated": False,
        "status": "pass",
    }
    result["dependencies_sha256"] = sha256_bytes(canonical_json(result))
    return result


def install_homebrew(packet: Mapping[str, Any], runner: CommandRunner) -> dict[str, Any]:
    """Verify the packet closure and reuse exact immutable global dependencies."""

    validate_work_packet(packet)
    if packet["host"] not in {"john1", "john2", "john3"}:
        raise D0Error("Homebrew installation is not authorized on this host")
    if REQUIRED_OPERATION["install"] not in packet["allowed_operations"]:
        raise D0Error("work packet does not authorize runtime installation")
    expected = _artifact_bottles(packet)
    cache_root = Path(packet["paths"]["homebrew_cache"])
    installer = verify_homebrew_installer(runner)
    dependencies_before = frozen_global_runtime_dependencies(packet, runner)
    verified: list[dict[str, Any]] = []
    for formula in formulas_for_host(packet["host"]):
        metadata_path = homebrew_metadata_path(cache_root, formula)
        metadata_output = _read_bounded_regular(metadata_path, 1024 * 1024)
        metadata = verify_formula_metadata(metadata_output, formula)
        bottle_path = homebrew_bottle_path(cache_root, formula)
        identity = _regular_file_identity(bottle_path)
        frozen = expected[formula]
        if identity["size"] != frozen["size"] or identity["sha256"] != frozen["sha256"]:
            raise D0Error(f"fetched Homebrew bottle identity differs for {formula}")
        source = frozen["source"]
        if source != metadata["bottle_url"]:
            raise D0Error(f"Homebrew bottle URL differs for {formula}")
        version = runner.run([BREW, "list", "--versions", formula]).stdout.decode().strip()
        if version != f"{formula} {FROZEN_RUNTIME[formula]['version']}":
            raise D0Error(f"pre-existing Homebrew version differs for {formula}")
        verified.append(
            {
                **metadata,
                **identity,
                "installed_version_line": version,
                "disposition": "verified-closure-not-installed",
            }
        )
    installer_after = verify_homebrew_installer(runner)
    dependencies_after = frozen_global_runtime_dependencies(packet, runner)
    if dependencies_after != dependencies_before:
        raise D0Error("global runtime dependencies changed during closure verification")
    return {
        "installer_before": installer,
        "installer_after": installer_after,
        "dependencies_before": dependencies_before,
        "dependencies_after": dependencies_after,
        "formulae": verified,
        "formulae_mutated": False,
        "status": "pass",
    }


def start_plan(packet: Mapping[str, Any]) -> list[Command]:
    validate_work_packet(packet)
    if packet["host"] not in {"john1", "john2", "john3"}:
        raise D0Error("Colima start is not authorized on this host")
    return [
        Command(
            (
                COLIMA,
                "start",
                "--profile",
                PROFILE,
                "--save-config=false",
                "--disk-image",
                packet["paths"]["core_image"],
            ),
            "start-colima",
        )
    ]


def start_runtime(packet: Mapping[str, Any], runner: CommandRunner) -> dict[str, Any]:
    validate_work_packet(packet)
    if packet["host"] not in {"john1", "john2", "john3"}:
        raise D0Error("Colima start is not authorized on this host")
    if REQUIRED_OPERATION["start"] not in packet["allowed_operations"]:
        raise D0Error("work packet does not authorize Colima start")
    verify_configs(packet)
    identity = _regular_file_identity(Path(packet["paths"]["core_image"]))
    expected = packet["artifacts"]["core_image"]
    if identity["size"] != expected["size"] or identity["sha256"] != expected["sha256"]:
        raise D0Error("staged Colima core image identity differs")
    command = start_plan(packet)[0]
    completed = runner.run(command.argv, timeout=1800)
    config_receipt = verify_configs(packet)
    return {
        "core_image": identity,
        "command": command.as_dict(),
        "stdout_sha256": sha256_bytes(completed.stdout),
        "stderr_sha256": sha256_bytes(completed.stderr),
        "post_start_configs": config_receipt,
        "status": "pass",
    }


def verify_plan(packet: Mapping[str, Any]) -> list[Command]:
    validate_work_packet(packet)
    if packet["host"] not in {"john1", "john2", "john3"}:
        raise D0Error("runtime verification is not authorized on this host")
    commands = [
        Command((COLIMA, "version"), "verify-colima-version"),
        Command((LIMACTL, "--version"), "verify-lima-version"),
        Command((DOCKER, "--version"), "verify-docker-cli-version"),
        Command((COLIMA, "status", "--profile", PROFILE, "--json"), "verify-colima-status"),
        Command((DOCKER, "version", "--format", "{{json .}}"), "verify-engine-version"),
        Command((DOCKER, "info", "--format", "{{json .}}"), "verify-engine-info"),
    ]
    if packet["host"] == "john2":
        commands.extend(
            [
                Command((DOCKER, "buildx", "version"), "verify-buildx-version"),
                Command(
                    (DOCKER, "buildx", "inspect", "--builder", "default"), "verify-buildkit-driver"
                ),
            ]
        )
    return commands


def rollback_plan(packet: Mapping[str, Any]) -> list[Command]:
    validate_work_packet(packet)
    if packet["host"] not in {"john1", "john2", "john3"}:
        raise D0Error("runtime rollback is not authorized on this host")
    commands = [
        Command((COLIMA, "stop", "--profile", PROFILE), "stop-colima"),
        Command(
            (COLIMA, "delete", "--profile", PROFILE, "--data", "--force"), "delete-colima-data"
        ),
    ]
    return commands


def _remove_owned_tree(path: Path) -> None:
    try:
        observed = path.lstat()
    except FileNotFoundError:
        return
    if observed.st_uid != os.getuid() or stat.S_ISLNK(observed.st_mode):
        raise D0Error(f"rollback root is unsafe: {path}")
    if not stat.S_ISDIR(observed.st_mode):
        if not stat.S_ISREG(observed.st_mode):
            raise D0Error(f"rollback root has an unsupported type: {path}")
        path.unlink()
        return
    pending = [path]
    ordered: list[Path] = []
    while pending:
        directory = pending.pop()
        ordered.append(directory)
        for child in os.scandir(directory):
            target = Path(child.path)
            value = target.lstat()
            if value.st_uid != os.getuid():
                raise D0Error(f"rollback entry has the wrong owner: {target}")
            if stat.S_ISDIR(value.st_mode) and not stat.S_ISLNK(value.st_mode):
                pending.append(target)
            elif stat.S_ISREG(value.st_mode) or stat.S_ISLNK(value.st_mode):
                target.unlink()
            else:
                raise D0Error(f"rollback entry has an unsupported type: {target}")
    for directory in reversed(ordered):
        directory.rmdir()


def rollback_runtime(
    packet: Mapping[str, Any],
    runner: CommandRunner,
    *,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove only D0-owned runtime objects; never invokes broad cleanup."""

    validate_work_packet(packet)
    host = packet["host"]
    if (
        host not in {"john1", "john2", "john3"}
        or REQUIRED_OPERATION["rollback"] not in packet["allowed_operations"]
    ):
        raise D0Error("work packet does not authorize runtime rollback")
    baseline = preflight.get("isolated_runtime_baseline")
    if not isinstance(baseline, Mapping) or baseline.get("totals", {}).get("present_roots") != 0:
        raise D0Error("rollback lacks an absent isolated-root preflight ledger")
    commands: list[dict[str, Any]] = []
    colima_path = Path(COLIMA)
    if colima_path.exists():
        resolved_colima = Path(os.path.realpath(colima_path))
        expected_cellar = Path("/opt/homebrew/Cellar/colima") / FROZEN_RUNTIME["colima"]["version"]
        observed = resolved_colima.lstat()
        if (
            resolved_colima.parent.parent != expected_cellar
            or not stat.S_ISREG(observed.st_mode)
            or not observed.st_mode & stat.S_IXUSR
        ):
            raise D0Error("rollback Colima executable metadata differs")
        for command in rollback_plan(packet)[:2]:
            completed = runner.run(command.argv, timeout=900, check=False)
            commands.append(
                {
                    **command.as_dict(),
                    "returncode": completed.returncode,
                    "stdout_sha256": sha256_bytes(completed.stdout),
                    "stderr_sha256": sha256_bytes(completed.stderr),
                }
            )
            if completed.returncode != 0:
                raise D0Error(f"exact Colima rollback command failed: {command.purpose}")
        runtime_disposition = "stopped-and-deleted"
    else:
        absent_activity = _inactive_runtime_activity(packet)
        if not absent_activity["inactive"]:
            raise D0Error("Colima is absent while runtime activity remains")
        commands.append(
            {
                "argv": [],
                "purpose": "colima-absent-inactive-fallback",
                "runtime_activity": absent_activity,
            }
        )
        runtime_disposition = "already-absent-and-inactive"
    activity_after_delete = _inactive_runtime_activity(packet)
    if not activity_after_delete["inactive"]:
        raise D0Error("runtime activity survived exact profile deletion")
    dependencies_before = preflight.get("global_runtime_dependencies")
    dependencies_after = frozen_global_runtime_dependencies(packet, runner)
    if not isinstance(dependencies_before, Mapping) or dependencies_after != dict(
        dependencies_before
    ):
        raise D0Error("global runtime dependencies changed during rollback")
    paths = packet["paths"]
    isolated = [
        Path(paths["colima_home"]),
        Path(paths["colima_cache_home"]),
        Path(paths["docker_config"]),
        Path(paths["homebrew_cache"]),
        Path(paths["homebrew_logs"]),
        Path(paths["homebrew_temp"]),
    ]
    if host in {"john1", "john3"}:
        bootstrap_roots = {
            Path(paths["core_image"]).parent,
            Path(paths["smoke_oci"]).parent,
            Path(paths["homebrew_closure"]).parent,
            Path(paths["runtime_supply"]).parent,
        }
        if len(bootstrap_roots) != 1:
            raise D0Error("worker bootstrap artifacts do not share one atomic root")
        isolated.append(bootstrap_roots.pop())
        isolated.append(Path(paths["runtime_supply_inbox"]).parent)
    baseline_roots = {
        item.get("root")
        for item in baseline.get("roots", [])
        if isinstance(item, Mapping) and item.get("present") is False
    }
    if any(str(path) not in baseline_roots for path in isolated):
        raise D0Error("rollback target was not absent in the authenticated preflight ledger")
    for path in isolated:
        _remove_owned_tree(path)
    return {
        "commands": commands,
        "runtime_disposition": runtime_disposition,
        "activity_after_delete": activity_after_delete,
        "global_runtime_dependencies_after": dependencies_after,
        "global_runtime_dependencies_unchanged": True,
        "homebrew_formulae_removed": False,
        "removed_roots": [str(path) for path in isolated],
        "status": "pass",
    }


def cleanup_bootstrap_staging(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Remove the verified non-authoritative cache and John3 staging copies."""

    validate_work_packet(packet)
    targets = [Path(packet["paths"]["colima_cache_home"])]
    if packet["host"] in {"john1", "john3"}:
        targets.append(Path(packet["paths"]["runtime_supply"]).parent)
    before = inventory_roots(
        targets,
        label=f"{packet['run_id']}-{packet['host']}-bootstrap-staging-before-cleanup",
        policy=InventoryPolicy(full_hash_limit=512 * 1024 * 1024),
    )
    for path in targets:
        _remove_owned_tree(path)
        if path.exists() or path.is_symlink():
            raise D0Error(f"bootstrap staging cleanup failed: {path}")
    after = inventory_roots(
        targets,
        label=f"{packet['run_id']}-{packet['host']}-bootstrap-staging-after-cleanup",
    )
    if after["totals"]["present_roots"] != 0:
        raise D0Error("bootstrap staging cleanup inventory is not empty")
    return {
        "removed": [str(path) for path in targets],
        "before": before,
        "after": after,
        "all_absent": True,
    }


def postflight_audit(
    packet: Mapping[str, Any],
    *,
    home: Path,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove exact baseline restoration after the rollback exercise."""

    validate_work_packet(packet)
    required = REQUIRED_OPERATION["postflight"]
    if packet["phase"] != "postflight" or required not in packet["allowed_operations"]:
        raise D0Error("work packet does not authorize postflight")
    selected_after = inventory_roots(
        selected_runtime_paths(home),
        label=f"{packet['run_id']}-{packet['host']}-campaign-runtime-post",
    )
    homebrew_after = inventory_roots(
        homebrew_ledger_paths(home, packet["host"], packet),
        label=f"{packet['run_id']}-{packet['host']}-homebrew-post",
        policy=InventoryPolicy(full_hash_limit=64 * 1024 * 1024, max_entries=500_000),
    )
    selected_comparison = compare_inventories(
        dict(preflight["selected_runtime"]),
        selected_after,
        label=f"{packet['run_id']}-{packet['host']}-selected-runtime-stability",
    )
    homebrew_comparison = compare_homebrew_ledger(
        dict(preflight["homebrew"]),
        homebrew_after,
        allowed_new_roots=(),
        label=f"{packet['run_id']}-{packet['host']}-homebrew-stability",
    )
    dependency_runner = CommandRunner(global_dependency_environment(packet["host"]))
    global_dependencies_after = frozen_global_runtime_dependencies(packet, dependency_runner)
    global_dependencies_before = preflight.get("global_runtime_dependencies")
    global_dependencies_stable = isinstance(
        global_dependencies_before, Mapping
    ) and global_dependencies_after == dict(global_dependencies_before)
    activity = _inactive_runtime_activity(packet)
    if (
        selected_comparison["status"] != "pass"
        or homebrew_comparison["status"] != "pass"
        or not global_dependencies_stable
        or not activity["inactive"]
    ):
        raise D0Error("postflight baseline restoration failed")
    result: dict[str, Any] = {
        "selected_runtime_after": selected_after,
        "selected_runtime_comparison": selected_comparison,
        "homebrew_after": homebrew_after,
        "homebrew_comparison": homebrew_comparison,
        "global_runtime_dependencies_after": global_dependencies_after,
        "global_runtime_dependencies_stable": global_dependencies_stable,
        "runtime_activity": activity,
    }
    if packet["host"] == "john1":
        podman_before = preflight.get("podman_negative_control")
        podman_after = podman_negative_control(home)
        if (
            not isinstance(podman_before, Mapping)
            or podman_before.get("status") != "pass"
            or podman_before.get("semantic_sha256") != podman_after["semantic_sha256"]
        ):
            raise D0Error("John1 no-Podman-machine/storage/activity semantics changed")
        result["podman_after"] = podman_after
        result["podman_semantics_stable"] = True
    result["status"] = "pass"
    return result


def complete_plan(packet: Mapping[str, Any]) -> dict[str, Any]:
    validate_work_packet(packet)
    phase = packet["phase"]
    operation = primary_operation(packet["host"], phase, packet["allowed_operations"])
    commands: list[Command] = []
    if operation == "install-runtime":
        commands = homebrew_plan(packet)
    elif operation == "start-runtime":
        commands = start_plan(packet)
    elif operation == "verify-runtime":
        commands = verify_plan(packet)
    elif operation == "rollback-runtime":
        commands = rollback_plan(packet)
    return {
        "schema_id": "cascadia.r2-map.d0-command-plan.v1",
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "packet_sha256": packet["packet_sha256"],
        "host": packet["host"],
        "phase": phase,
        "operation": operation,
        "commands": [command.as_dict() for command in commands],
        "internal_handler": len(commands) == 0,
        "execute_by_default": False,
    }


def _safe_directory(path: Path, mode: int = 0o700) -> None:
    secure_owner_directory(path, mode=mode)


def _atomic_private_file(path: Path, payload: bytes, mode: int) -> None:
    _safe_directory(path.parent)
    if path.exists() or path.is_symlink():
        raise D0Error(f"infrastructure file already exists: {path}")
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        position = 0
        while position < len(payload):
            position += os.write(descriptor, payload[position:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def install_configs(packet: Mapping[str, Any]) -> dict[str, Any]:
    validate_work_packet(packet)
    host = packet["host"]
    if host not in {"john1", "john2", "john3"}:
        raise D0Error("runtime config installation is not authorized on this host")
    paths = packet["paths"]
    colima = Path(paths["colima_home"]) / PROFILE / "colima.yaml"
    docker = Path(paths["docker_config"]) / "config.json"
    _atomic_private_file(colima, COLIMA_CONFIG, 0o600)
    docker_payload = DOCKER_CONFIG_JOHN2 if host == "john2" else DOCKER_CONFIG_WORKER
    _atomic_private_file(docker, docker_payload, 0o600)
    return {
        "colima_path": str(colima),
        "colima_sha256": sha256_bytes(COLIMA_CONFIG),
        "docker_path": str(docker),
        "docker_sha256": sha256_bytes(docker_payload),
    }


def _verify_exact_file(path: Path, expected: bytes, mode: int) -> None:
    observed = path.lstat()
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_uid != os.getuid()
        or observed.st_nlink != 1
        or stat.S_IMODE(observed.st_mode) != mode
    ):
        raise D0Error(f"runtime config metadata differs: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        payload = os.read(descriptor, len(expected) + 1)
        if payload != expected or os.read(descriptor, 1):
            raise D0Error(f"runtime config bytes differ: {path}")
    finally:
        os.close(descriptor)


def verify_configs(packet: Mapping[str, Any]) -> dict[str, Any]:
    validate_work_packet(packet)
    paths = packet["paths"]
    _verify_exact_file(Path(paths["colima_home"]) / PROFILE / "colima.yaml", COLIMA_CONFIG, 0o600)
    docker_payload = DOCKER_CONFIG_JOHN2 if packet["host"] == "john2" else DOCKER_CONFIG_WORKER
    _verify_exact_file(Path(paths["docker_config"]) / "config.json", docker_payload, 0o600)
    return {"status": "pass"}


def verify_socket(packet: Mapping[str, Any]) -> dict[str, Any]:
    validate_work_packet(packet)
    path = Path(packet["paths"]["colima_home"]) / PROFILE / "docker.sock"
    observed = path.lstat()
    if not stat.S_ISSOCK(observed.st_mode) or observed.st_uid != os.getuid():
        raise D0Error("Docker socket metadata differs")
    if stat.S_IMODE(observed.st_mode) & 0o077:
        raise D0Error("Docker socket is accessible outside its owner")
    return {
        "path": str(path),
        "uid": observed.st_uid,
        "gid": observed.st_gid,
        "mode": f"{stat.S_IMODE(observed.st_mode):04o}",
    }


def _swap_used_bytes(value: str) -> int:
    match = re.search(r"used\s*=\s*([0-9.]+)([MG])", value)
    if match is None:
        raise D0Error("vm.swapusage output differs")
    factor = 1024**2 if match.group(2) == "M" else 1024**3
    return int(float(match.group(1)) * factor)


def host_resource_snapshot(runner: CommandRunner) -> dict[str, Any]:
    swap = runner.run(["/usr/sbin/sysctl", "-n", "vm.swapusage"]).stdout.decode("ascii")
    disk = runner.run(["/bin/df", "-k", "/System/Volumes/Data"]).stdout.decode("ascii")
    rows = [line.split() for line in disk.splitlines() if line.strip()]
    if len(rows) != 2 or len(rows[1]) < 6:
        raise D0Error("Data-volume df output differs")
    return {
        "swap_used_bytes": _swap_used_bytes(swap),
        "data_volume_kib_total": int(rows[1][1]),
        "data_volume_kib_used": int(rows[1][2]),
        "data_volume_kib_available": int(rows[1][3]),
        "data_volume_capacity": rows[1][4],
        "collected_unix_ms": time.time_ns() // 1_000_000,
    }


class ContinuousSwapMonitor:
    """Sample host swap throughout an action, retaining a tamper-evident summary."""

    def __init__(
        self,
        *,
        interval_seconds: float = 0.1,
        sample_reader: Callable[[], tuple[int, str]] | None = None,
    ) -> None:
        if not 0.05 <= interval_seconds <= 1.0:
            raise D0Error("swap monitor interval differs")
        self.interval_seconds = interval_seconds
        self._sample_reader = sample_reader or self._live_sample
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._error: str | None = None
        self._finalizer: Callable[[dict[str, Any]], Any] | None = None
        self._finalizer_result: Any = None
        self._finalizer_error: BaseException | None = None
        self._sample_count = 0
        self._nonzero_samples = 0
        self._max_used_bytes = 0
        self._first_unix_ms: int | None = None
        self._last_unix_ms: int | None = None
        self._digest = hashlib.sha256()

    @staticmethod
    def _live_sample() -> tuple[int, str]:
        try:
            completed = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "vm.swapusage"],
                check=False,
                capture_output=True,
                timeout=2,
                env={"LC_ALL": "C", "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise D0Error("continuous swap sample command failed") from error
        if completed.returncode != 0 or completed.stderr or len(completed.stdout) > 4096:
            raise D0Error("continuous swap sample output differs")
        try:
            output = completed.stdout.decode("ascii")
        except UnicodeDecodeError as error:
            raise D0Error("continuous swap sample is not ASCII") from error
        return _swap_used_bytes(output), output

    def _sample(self) -> None:
        used_bytes, output = self._sample_reader()
        if not isinstance(used_bytes, int) or isinstance(used_bytes, bool) or used_bytes < 0:
            raise D0Error("continuous swap sample value differs")
        now = time.time_ns() // 1_000_000
        sample = canonical_json(
            {
                "collected_unix_ms": now,
                "output_sha256": sha256_bytes(output.encode("ascii")),
                "used_bytes": used_bytes,
            }
        )
        with self._lock:
            self._sample_count += 1
            self._nonzero_samples += int(used_bytes != 0)
            self._max_used_bytes = max(self._max_used_bytes, used_bytes)
            self._first_unix_ms = now if self._first_unix_ms is None else self._first_unix_ms
            self._last_unix_ms = now
            self._digest.update(len(sample).to_bytes(8, "big"))
            self._digest.update(sample)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self._sample()
            except BaseException as error:
                with self._lock:
                    self._error = f"{type(error).__name__}: {error}"
                self._stop.set()
                return
        with self._lock:
            finalizer = self._finalizer
        if finalizer is None:
            return
        try:
            # The monitor thread takes the last sample, writes the caller's
            # journal/commit marker itself, and only then exits.  This avoids a
            # caller-owned mutation gap between monitoring and finalization.
            self._sample()
            evidence = self._evidence()
            result = finalizer(evidence)
        except BaseException as error:
            with self._lock:
                self._finalizer_error = error
            return
        with self._lock:
            self._finalizer_result = result

    def start(self) -> None:
        if self._thread is not None:
            raise D0Error("swap monitor was already started")
        self._sample()
        self._thread = threading.Thread(target=self._run, name="r2-d0-swap-monitor", daemon=True)
        self._thread.start()

    def _evidence(self) -> dict[str, Any]:
        with self._lock:
            if self._error is not None:
                raise D0Error(f"continuous swap monitoring failed: {self._error}")
            if self._sample_count < 1 or self._first_unix_ms is None or self._last_unix_ms is None:
                raise D0Error("continuous swap monitoring produced no samples")
            evidence = {
                "interval_milliseconds": int(self.interval_seconds * 1000),
                "sample_count": self._sample_count,
                "nonzero_samples": self._nonzero_samples,
                "max_used_bytes": self._max_used_bytes,
                "first_unix_ms": self._first_unix_ms,
                "last_unix_ms": self._last_unix_ms,
                "sample_stream_sha256": self._digest.hexdigest(),
                "status": "pass",
            }
        if evidence["nonzero_samples"] or evidence["max_used_bytes"]:
            raise D0Error("continuous swap monitoring observed nonzero host swap use")
        return evidence

    def stop(self) -> dict[str, Any]:
        thread = self._thread
        if thread is None:
            raise D0Error("swap monitor was not started")
        self._stop.set()
        thread.join(timeout=3)
        if thread.is_alive():
            raise D0Error("swap monitor did not stop within its bound")
        with self._lock:
            if self._finalizer is not None:
                raise D0Error("swap monitor was configured for owned finalization")
        return self._evidence()

    def stop_and_finalize(
        self,
        finalizer: Callable[[dict[str, Any]], Any],
    ) -> tuple[dict[str, Any], Any]:
        """Run ``finalizer`` on the monitor thread before that thread exits."""

        thread = self._thread
        if thread is None:
            raise D0Error("swap monitor was not started")
        if not callable(finalizer):
            raise D0Error("swap monitor finalizer differs")
        with self._lock:
            if self._finalizer is not None:
                raise D0Error("swap monitor finalizer was already configured")
            self._finalizer = finalizer
        self._stop.set()
        thread.join(timeout=3)
        if thread.is_alive():
            raise D0Error("swap monitor did not finalize within its bound")
        with self._lock:
            finalizer_error = self._finalizer_error
            result = self._finalizer_result
        if finalizer_error is not None:
            raise finalizer_error
        return self._evidence(), result


def host_platform_snapshot(packet: Mapping[str, Any], runner: CommandRunner) -> dict[str, Any]:
    """Qualify the exact per-host M4/macOS/internal-root substrate."""

    expected = HOST_PLATFORM[packet["host"]]
    commands = {
        "macos_version": ["/usr/bin/sw_vers", "-productVersion"],
        "build_version": ["/usr/bin/sw_vers", "-buildVersion"],
        "architecture": ["/usr/bin/uname", "-m"],
        "darwin_release": ["/usr/bin/uname", "-r"],
        "cpu_brand": ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
        "logical_cpu": ["/usr/sbin/sysctl", "-n", "hw.logicalcpu"],
        "physical_cpu": ["/usr/sbin/sysctl", "-n", "hw.physicalcpu"],
        "memory_bytes": ["/usr/sbin/sysctl", "-n", "hw.memsize"],
        "model": ["/usr/sbin/sysctl", "-n", "hw.model"],
    }
    observed = {
        key: runner.run(argv, maximum=64 * 1024).stdout.decode("utf-8").strip()
        for key, argv in commands.items()
    }
    required = {
        **expected,
        "architecture": "arm64",
        "cpu_brand": "Apple M4",
        "logical_cpu": "10",
        "physical_cpu": "10",
        "memory_bytes": str(16 * 1024**3),
        "model": "Mac16,10",
    }
    if observed != required:
        raise D0Error("host platform differs from its explicit per-host qualification")
    pressure = runner.run(["/usr/bin/memory_pressure", "-Q"], maximum=64 * 1024).stdout.decode(
        "utf-8"
    )
    match = re.search(r"System-wide memory free percentage:\s*([0-9]+)%", pressure)
    if match is None or int(match.group(1)) < MIN_MEMORY_FREE_PERCENT:
        raise D0Error("host is under memory pressure")
    sip = runner.run(["/usr/bin/csrutil", "status"], maximum=64 * 1024).stdout.decode().strip()
    authenticated = (
        runner.run(["/usr/bin/csrutil", "authenticated-root", "status"], maximum=64 * 1024)
        .stdout.decode()
        .strip()
    )
    if sip != "System Integrity Protection status: enabled." or authenticated != (
        "Authenticated Root status: enabled"
    ):
        raise D0Error("host system-integrity policy differs")
    disk = runner.run(["/usr/sbin/diskutil", "info", "-plist", "/"], maximum=1024 * 1024).stdout
    try:
        disk_info = plistlib.loads(disk)
    except (plistlib.InvalidFileException, ValueError) as error:
        raise D0Error("root-volume identity is invalid") from error
    disk_receipt = {
        "filesystem": disk_info.get("FilesystemType"),
        "protocol": disk_info.get("BusProtocol"),
        "internal": disk_info.get("Internal"),
        "removable": disk_info.get("Removable"),
        "solid_state": disk_info.get("SolidState"),
        "sealed": disk_info.get("Sealed"),
    }
    if disk_receipt != {
        "filesystem": "apfs",
        "protocol": "Apple Fabric",
        "internal": True,
        "removable": False,
        "solid_state": True,
        "sealed": "Yes",
    }:
        raise D0Error("root-volume integrity is not sealed internal Apple Fabric APFS")
    return {
        **observed,
        "memory_free_percent": int(match.group(1)),
        "minimum_memory_free_percent": MIN_MEMORY_FREE_PERCENT,
        "sip": sip,
        "authenticated_root": authenticated,
        "root_volume": disk_receipt,
        "cross_host_os_asymmetry_explicit": True,
        "status": "pass",
    }


def runtime_footprint(packet: Mapping[str, Any]) -> dict[str, Any]:
    validate_work_packet(packet)
    paths = packet["paths"]
    roots = [
        Path(paths["colima_home"]),
        Path(paths["colima_cache_home"]),
        Path(paths["docker_config"]),
        Path(paths["homebrew_cache"]),
        Path(paths["homebrew_logs"]),
        Path(paths["homebrew_temp"]),
    ]
    for formula in formulas_for_host(packet["host"]):
        roots.extend(
            [
                Path("/opt/homebrew/Cellar") / formula,
                Path("/opt/homebrew/opt") / formula,
                Path("/opt/homebrew/bin") / ("limactl" if formula == "lima" else formula),
            ]
        )
    if packet["host"] == "john2":
        roots.extend(
            [
                Path(paths["scanner_oci"]),
                Path(paths["scanner_license"]),
                Path(paths["scanner_source_archive"]),
            ]
        )
    roots = list(dict.fromkeys(roots))
    report = inventory_roots(
        roots,
        label=f"{packet['run_id']}-{packet['host']}-runtime-footprint",
        policy=InventoryPolicy(full_hash_limit=32 * 1024 * 1024),
    )
    if packet["host"] == "john2":
        buildx = FROZEN_RUNTIME["docker-buildx"]
        report["managed_homebrew_links"] = [
            inventory_managed_homebrew_link(
                Path("/opt/homebrew/lib/docker/cli-plugins/docker-buildx"),
                managed_link=Path("/opt/homebrew/lib/docker"),
                cellar_root=Path("/opt/homebrew/Cellar"),
                formula="docker-buildx",
                version=buildx["version"],
                managed_target_relative=Path("lib/docker"),
                requested_suffix=Path("cli-plugins/docker-buildx"),
                installed_file_relative=Path("bin/docker-buildx"),
                managed_link_target=buildx["managed_link_target"],
                requested_link_target=buildx["plugin_link_target"],
                install_receipt_sha256=buildx["install_receipt_sha256"],
                installed_file_sha256=buildx["installed_entrypoint_sha256"],
                label=f"{packet['run_id']}-john2-docker-buildx-link",
                policy=InventoryPolicy(full_hash_limit=128 * 1024 * 1024),
            )
        ]
        report["inventory_sha256"] = document_sha256(report, "inventory_sha256")
    totals = report["totals"]
    if (
        totals["apparent_bytes"] > RUNTIME_MAX_BYTES
        or totals["allocated_bytes"] > RUNTIME_MAX_BYTES
    ):
        raise D0Error("runtime footprint exceeds 20 GiB")
    return report


def enforce_footprint_budget(
    footprint: Mapping[str, Any],
    *,
    preflight_available_kib: int,
) -> dict[str, Any]:
    totals = footprint.get("totals")
    if not isinstance(totals, Mapping):
        raise D0Error("runtime footprint totals are absent")
    apparent = totals.get("apparent_bytes")
    allocated = totals.get("allocated_bytes")
    if not isinstance(apparent, int) or not isinstance(allocated, int):
        raise D0Error("runtime footprint totals differ")
    free_bytes = preflight_available_kib * 1024
    fraction_limit = (free_bytes * FREE_FRACTION_PPM) // 1_000_000
    effective_limit = min(RUNTIME_MAX_BYTES, fraction_limit)
    if effective_limit <= 0 or apparent > effective_limit or allocated > effective_limit:
        raise D0Error("runtime footprint exceeds the frozen absolute or free-space budget")
    return {
        "apparent_bytes": apparent,
        "allocated_bytes": allocated,
        "preflight_free_bytes": free_bytes,
        "absolute_limit_bytes": RUNTIME_MAX_BYTES,
        "free_fraction_limit_bytes": fraction_limit,
        "effective_limit_bytes": effective_limit,
        "status": "pass",
    }


def verify_version_output(component: str, value: bytes) -> None:
    pattern = VERSION_PATTERNS.get(component)
    if pattern is None:
        raise D0Error("unknown version component")
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise D0Error(f"{component} version output is not text") from error
    if pattern.search(text) is None:
        raise D0Error(f"{component} version differs")


def verify_engine_version(
    value: bytes,
    *,
    expected_client_version: str | None = None,
) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D0Error("Docker version output is invalid") from error
    if not isinstance(decoded, dict):
        raise D0Error("Docker version output is not an object")
    client = decoded.get("Client")
    server = decoded.get("Server")
    if not isinstance(client, dict) or not isinstance(server, dict):
        raise D0Error("Docker client/server identities are absent")
    if (
        client.get("Version") != (expected_client_version or FROZEN_RUNTIME["docker"]["version"])
        or server.get("Version") != EXPECTED_ENGINE_VERSION
        or server.get("Os") != "linux"
        or server.get("Arch") not in {"arm64", "aarch64"}
        or not isinstance(server.get("ApiVersion"), str)
        or not isinstance(server.get("GitCommit"), str)
    ):
        raise D0Error("Docker client/server identity differs")
    return {
        "client_version": client["Version"],
        "client_api_version": client.get("ApiVersion"),
        "client_git_commit": client.get("GitCommit"),
        "server_version": server["Version"],
        "server_api_version": server["ApiVersion"],
        "server_git_commit": server["GitCommit"],
        "server_os": server["Os"],
        "server_arch": server["Arch"],
    }


def _normalize_registry_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "InsecureRegistryCIDRs",
        "IndexConfigs",
        "Mirrors",
    }:
        raise D0Error("Docker registry configuration differs")
    cidrs = value["InsecureRegistryCIDRs"]
    if (
        not isinstance(cidrs, list)
        or any(not isinstance(item, str) for item in cidrs)
        or len(cidrs) != len(set(cidrs))
        or set(cidrs) != {"127.0.0.0/8", "::1/128"}
        or value["Mirrors"] != []
    ):
        raise D0Error("Docker registry configuration differs")
    expected_index = {
        "docker.io": {
            "Name": "docker.io",
            "Mirrors": [],
            "Secure": True,
            "Official": True,
        }
    }
    if value["IndexConfigs"] != expected_index:
        raise D0Error("Docker registry configuration differs")
    return {
        "InsecureRegistryCIDRs": sorted(cidrs),
        "IndexConfigs": expected_index,
        "Mirrors": [],
    }


def verify_engine_info(value: bytes) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D0Error("Docker info output is invalid") from error
    if not isinstance(decoded, dict):
        raise D0Error("Docker info output is not an object")
    if (
        decoded.get("OSType") != "linux"
        or decoded.get("Architecture") not in {"aarch64", "arm64"}
        or decoded.get("ServerVersion") != EXPECTED_ENGINE_VERSION
        or decoded.get("Swarm", {}).get("LocalNodeState") not in {"inactive", "locked"}
        or decoded.get("ID") in {None, ""}
        or decoded.get("Name") in {None, ""}
        or decoded.get("OperatingSystem") in {None, ""}
        or decoded.get("KernelVersion") in {None, ""}
        or decoded.get("DockerRootDir") != "/var/lib/docker"
        # The sealed daemon configuration enables the containerd snapshotter
        # and explicitly selects cgroupfs.  Docker reports that combination as
        # ``overlayfs``/``cgroupfs`` (not the legacy graphdriver spelling
        # ``overlay2`` or Docker's otherwise preferred ``systemd`` driver).
        or decoded.get("Driver") != EXPECTED_ENGINE_STORAGE_DRIVER
        or decoded.get("CgroupDriver") != EXPECTED_ENGINE_CGROUP_DRIVER
        or decoded.get("CgroupVersion") not in {2, "2"}
        or bool(decoded.get("HttpProxy"))
        or bool(decoded.get("HttpsProxy"))
        or bool(decoded.get("NoProxy"))
    ):
        raise D0Error("Docker Engine configuration differs")
    registry_config = _normalize_registry_config(decoded.get("RegistryConfig"))
    security_options = decoded.get("SecurityOptions")
    if (
        not isinstance(security_options, list)
        or not all(isinstance(item, str) for item in security_options)
        or not any(item.startswith("name=seccomp") for item in security_options)
        or "name=cgroupns" not in security_options
    ):
        raise D0Error("Docker Engine security options differ")
    return {
        "daemon_id": decoded["ID"],
        "name": decoded["Name"],
        "server_version": decoded["ServerVersion"],
        "os_type": decoded["OSType"],
        "architecture": decoded["Architecture"],
        "operating_system": decoded["OperatingSystem"],
        "kernel_version": decoded["KernelVersion"],
        "docker_root_dir": decoded["DockerRootDir"],
        "driver": decoded["Driver"],
        "cgroup_driver": decoded["CgroupDriver"],
        "cgroup_version": int(decoded["CgroupVersion"]),
        "security_options": sorted(security_options),
        "registry_config_sha256": sha256_bytes(canonical_json(registry_config)),
        "http_proxy_configured": False,
        "https_proxy_configured": False,
        "no_proxy_configured": False,
        "stock_build_api_present": True,
    }


def verify_daemon_config(value: str, engine_info: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the guest daemon configuration to Docker's reported effective state."""

    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise D0Error("guest Docker daemon configuration is invalid") from error
    if decoded != EXPECTED_DAEMON_CONFIG:
        raise D0Error("guest Docker daemon configuration differs")
    if (
        engine_info.get("driver") != EXPECTED_ENGINE_STORAGE_DRIVER
        or engine_info.get("cgroup_driver") != EXPECTED_ENGINE_CGROUP_DRIVER
    ):
        raise D0Error("guest Docker daemon configuration and Engine state disagree")
    return {
        "config": decoded,
        "config_sha256": sha256_bytes(canonical_json(decoded)),
        "engine_storage_driver": engine_info["driver"],
        "engine_cgroup_driver": engine_info["cgroup_driver"],
        "status": "pass",
    }


def _stable_engine_identity(runner: CommandRunner) -> dict[str, Any]:
    version = runner.run(
        [DOCKER, "version", "--format", "{{json .}}"],
        timeout=300,
        maximum=1024 * 1024,
    ).stdout
    info = runner.run(
        [DOCKER, "info", "--format", "{{json .}}"],
        timeout=300,
        maximum=4 * 1024 * 1024,
    ).stdout
    return {
        "version": verify_engine_version(version),
        "info": verify_engine_info(info),
    }


def verify_buildx_inspect(value: bytes) -> dict[str, Any]:
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise D0Error("buildx inspect output is not text") from error
    drivers = re.findall(r"(?m)^Driver:\s*(\S+)\s*$", text)
    buildkits = re.findall(r"(?m)^BuildKit version:\s*(\S+)\s*$", text)
    platforms = re.findall(r"(?m)^Platforms:\s*(\S+)\s*$", text)
    if (
        drivers != ["docker"]
        or buildkits != ["v0.30.0"]
        or platforms != ["linux/arm64"]
        or any(value in text.lower() for value in ("docker-container", "remote", "cloud"))
    ):
        raise D0Error("buildx integrated-driver identity differs")
    return {
        "driver": "docker",
        "buildkit_version": "v0.30.0",
        "platforms": "linux/arm64",
        "output_sha256": sha256_bytes(value),
    }


def _guest_network_projection(value: str) -> dict[str, Any]:
    try:
        document = json.loads(value)
    except json.JSONDecodeError as error:
        raise D0Error("guest network-address inventory is invalid") from error
    if not isinstance(document, list) or any(not isinstance(item, dict) for item in document):
        raise D0Error("guest network-address inventory shape differs")
    by_name = {item.get("ifname"): item for item in document}
    if set(by_name) != {"lo", "eth0", "docker0"}:
        raise D0Error("guest network interface inventory differs")

    def addresses(interface: str) -> list[dict[str, Any]]:
        items = by_name[interface].get("addr_info")
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise D0Error("guest interface address inventory differs")
        return items

    lo = addresses("lo")
    eth0 = addresses("eth0")
    docker0 = addresses("docker0")
    lo_projection = {(item.get("family"), item.get("local"), item.get("prefixlen")) for item in lo}
    docker_ipv4 = [
        item
        for item in docker0
        if item.get("family") == "inet"
        and item.get("local") == "172.17.0.1"
        and item.get("prefixlen") == 16
    ]
    docker_ipv6 = [item for item in docker0 if item.get("family") == "inet6"]
    docker_ipv6_valid = not docker_ipv6
    if len(docker_ipv6) == 1 and isinstance(docker_ipv6[0].get("local"), str):
        try:
            docker_ipv6_valid = (
                docker_ipv6[0].get("prefixlen") == 64
                and docker_ipv6[0].get("scope") == "link"
                and ipaddress.IPv6Address(docker_ipv6[0]["local"]).is_link_local
            )
        except ipaddress.AddressValueError:
            docker_ipv6_valid = False
    eth0_v4 = [
        item
        for item in eth0
        if item.get("family") == "inet"
        and item.get("local") == "192.168.5.1"
        and item.get("prefixlen") == 24
        and item.get("scope") == "global"
    ]
    eth0_v6 = [item for item in eth0 if item.get("family") == "inet6"]
    if (
        lo_projection != {("inet", "127.0.0.1", 8), ("inet6", "::1", 128)}
        or len(docker_ipv4) != 1
        or len(docker_ipv6) > 1
        or not docker_ipv6_valid
        or len(eth0) != 2
        or len(eth0_v4) != 1
        or len(eth0_v6) != 1
        or eth0_v6[0].get("prefixlen") != 64
        or eth0_v6[0].get("scope") != "link"
        or not isinstance(eth0_v6[0].get("local"), str)
        or not eth0_v6[0]["local"].lower().startswith("fe80:")
        or re.fullmatch(r"[0-9A-Fa-f:]+", eth0_v6[0]["local"]) is None
    ):
        raise D0Error("guest effective interface addresses differ")
    link_local = eth0_v6[0]["local"]
    return {
        "interfaces": ["docker0", "eth0", "lo"],
        "loopback_ipv4": "127.0.0.1",
        "loopback_ipv6": "::1",
        "eth0_ipv4": "192.168.5.1",
        "eth0_ipv6_link_local": link_local,
        "docker0_ipv4": "172.17.0.1",
        "docker0_ipv6_link_local": docker_ipv6[0]["local"] if docker_ipv6 else None,
        "docker_network_lifecycle": "warm" if docker_ipv6 else "cold",
        "dns_listener_endpoints": sorted(
            {
                "127.0.0.1:53",
                "127.0.0.53%lo:53",
                "127.0.0.54:53",
                "192.168.5.1:53",
                "[::1]:53",
                f"[{link_local}]%eth0:53",
            }
        ),
        "status": "pass",
    }


def _guest_listener_allowlist(
    value: str,
    network: Mapping[str, Any],
) -> list[dict[str, str]]:
    allowed_ssh = {
        "0.0.0.0:22",
        "[::]:22",
        "*:22",
    }
    dns_endpoints = network.get("dns_listener_endpoints")
    if not isinstance(dns_endpoints, list) or any(
        not isinstance(item, str) for item in dns_endpoints
    ):
        raise D0Error("guest network-derived DNS listener inventory is absent")
    allowed_dns = set(dns_endpoints)
    listeners: list[dict[str, str]] = []
    for line in value.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 5 or fields[0] != "LISTEN":
            raise D0Error("guest TCP listener output shape differs")
        endpoint = fields[3]
        process = fields[5] if len(fields) == 6 else ""
        if endpoint not in allowed_ssh | allowed_dns:
            raise D0Error(f"guest TCP listener is not allowlisted: {endpoint}")
        if endpoint.endswith(":22") and process and "sshd" not in process:
            raise D0Error("guest SSH listener process identity differs")
        if endpoint.endswith(":53") and process and "systemd-resolve" not in process:
            raise D0Error("guest DNS listener process identity differs")
        listeners.append({"endpoint": endpoint, "process": process})
    return listeners


def _guest_mount_projection(value: str) -> dict[str, Any]:
    try:
        document = json.loads(value)
    except json.JSONDecodeError as error:
        raise D0Error("guest mount inventory is invalid") from error
    filesystems = document.get("filesystems") if isinstance(document, dict) else None
    if not isinstance(filesystems, list) or not filesystems:
        raise D0Error("guest mount inventory is empty")
    flattened: list[dict[str, str]] = []

    def visit(items: list[Any]) -> None:
        for item in items:
            if not isinstance(item, dict):
                raise D0Error("guest mount inventory entry differs")
            projected: dict[str, str] = {}
            for key in ("target", "source", "fstype", "options"):
                field = item.get(key)
                if not isinstance(field, str):
                    raise D0Error("guest mount inventory field differs")
                projected[key] = field
            lowered = " ".join(projected.values()).lower()
            if any(
                marker in lowered
                for marker in (
                    "/users/",
                    "/volumes/",
                    "virtiofs",
                    "9p",
                    "mount0",
                    "sshfs",
                    "rosetta",
                )
            ):
                raise D0Error("guest exposes an unauthorized host/shared mount")
            flattened.append(projected)
            children = item.get("children", [])
            if not isinstance(children, list):
                raise D0Error("guest mount child inventory differs")
            visit(children)

    visit(filesystems)
    if not any(item["target"] == "/" for item in flattened):
        raise D0Error("guest root mount is absent")
    return {
        "mount_count": len(flattened),
        "mounts_sha256": sha256_bytes(canonical_json(flattened)),
        "host_shared_mounts": [],
    }


def _validate_guest_binfmt_inventory(value: str) -> dict[str, Any]:
    try:
        document = json.loads(value)
    except json.JSONDecodeError as error:
        raise D0Error("guest binfmt handler evidence is invalid") from error
    if not isinstance(document, dict) or set(document) != {"control", "handlers"}:
        raise D0Error("guest binfmt handler inventory differs")
    control = document["control"]
    handlers = document["handlers"]
    if control != ["register", "status"] or not isinstance(handlers, dict):
        raise D0Error("guest binfmt handler inventory differs")
    if set(handlers) != {"python3.12"} or not isinstance(handlers["python3.12"], str):
        raise D0Error("guest exposes an unauthorized binfmt handler")
    lines = handlers["python3.12"].splitlines()
    if lines != [
        "enabled",
        "interpreter /usr/bin/python3.12",
        "flags: ",
        "offset 0",
        "magic cb0d0d0a",
    ]:
        raise D0Error("guest native Python binfmt handler identity differs")
    return {
        "control_entries": control,
        "handlers": {
            "python3.12": {
                "interpreter": "/usr/bin/python3.12",
                "magic": "cb0d0d0a",
                "enabled": True,
                "foreign_architecture": False,
            }
        },
        "status": "pass",
    }


def _validate_guest_nested_virtualization(value: str) -> dict[str, Any]:
    try:
        document = json.loads(value)
    except json.JSONDecodeError as error:
        raise D0Error("guest nested-virtualization evidence is invalid") from error
    if document != {"dev_kvm": False, "modules": ["/sys/module/kvm"]}:
        raise D0Error("guest nested virtualization capability differs")
    return {
        "usable_kvm_device_present": False,
        "kernel_kvm_module_present": True,
        "architecture_specific_kvm_modules": [],
        "nested_virtualization_enabled": False,
        "status": "pass",
    }


def _single_findmnt(value: str, *, expected_target: str) -> dict[str, str]:
    try:
        document = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise D0Error(f"guest {expected_target} mount evidence is invalid") from error
    filesystems = document.get("filesystems") if isinstance(document, dict) else None
    if not isinstance(filesystems, list) or len(filesystems) != 1:
        raise D0Error(f"guest {expected_target} mount evidence cardinality differs")
    item = filesystems[0]
    if not isinstance(item, dict):
        raise D0Error(f"guest {expected_target} mount evidence differs")
    projected: dict[str, str] = {}
    for key in ("target", "source", "fstype", "options"):
        field = item.get(key)
        if not isinstance(field, str) or not field:
            raise D0Error(f"guest {expected_target} mount field differs")
        projected[key] = field
    if projected["target"] != expected_target:
        raise D0Error(f"guest {expected_target} is not its own mount target")
    options = set(projected["options"].split(","))
    if "rw" not in options or "ro" in options or projected["fstype"] != "ext4":
        raise D0Error(f"guest {expected_target} mount is not writable ext4")
    return projected


def _guest_effective_config(
    versions: Mapping[str, Any], engine_info: Mapping[str, Any]
) -> dict[str, Any]:
    for key in (
        "cpu_count",
        "daemon_config",
        "meminfo",
        "virtualization",
        "mounts",
        "block_devices",
        "root_mount",
        "docker_data_mount",
        "binfmt_handlers",
        "nested_virtualization",
        "ssh_agent",
        "processes",
        "kubernetes_state",
    ):
        result = versions.get(key)
        if not isinstance(result, dict) or result.get("returncode") != 0:
            raise D0Error(f"guest effective {key} command failed")
    try:
        cpu_count = int(versions["cpu_count"]["stdout"].strip())
    except (KeyError, TypeError, ValueError) as error:
        raise D0Error("guest effective CPU count differs") from error
    memory = re.search(r"(?m)^MemTotal:\s+(\d+)\s+kB$", versions["meminfo"]["stdout"])
    if cpu_count != 10 or memory is None:
        raise D0Error("guest effective CPU or memory configuration differs")
    memory_bytes = int(memory.group(1)) * 1024
    if not 13 * 1024**3 <= memory_bytes <= 14 * 1024**3:
        raise D0Error("guest effective memory is outside the frozen 14-GiB envelope")
    virtualization = versions["virtualization"]["stdout"].strip()
    if virtualization != "apple":
        raise D0Error("guest is not running under Apple Virtualization.framework")
    try:
        block_devices = json.loads(versions["block_devices"]["stdout"])
    except (TypeError, json.JSONDecodeError) as error:
        raise D0Error("guest effective block-device inventory is invalid") from error
    devices = block_devices.get("blockdevices") if isinstance(block_devices, dict) else None
    if not isinstance(devices, list) or not devices:
        raise D0Error("guest effective block-device inventory is empty")
    inventory: list[dict[str, Any]] = []
    pending = list(devices)
    while pending:
        item = pending.pop()
        if not isinstance(item, dict):
            raise D0Error("guest effective block-device entry differs")
        children = item.get("children", [])
        if not isinstance(children, list):
            raise D0Error("guest effective block-device children differ")
        pending.extend(children)
        expected_keys = {
            "name",
            "kname",
            "path",
            "pkname",
            "size",
            "type",
            "fstype",
            "mountpoints",
            "ro",
            "children",
        }
        if "children" not in item:
            expected_keys.remove("children")
        name = item.get("name")
        kname = item.get("kname")
        path = item.get("path")
        parent = item.get("pkname")
        size = item.get("size")
        kind = item.get("type")
        fstype = item.get("fstype")
        mountpoints = item.get("mountpoints")
        readonly = item.get("ro")
        if (
            set(item) != expected_keys
            or not isinstance(name, str)
            or not isinstance(kname, str)
            or not isinstance(path, str)
            or not path.startswith("/dev/")
            or (parent is not None and not isinstance(parent, str))
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            or not isinstance(kind, str)
            or (fstype is not None and not isinstance(fstype, str))
            or not isinstance(mountpoints, list)
            or any(point is not None and not isinstance(point, str) for point in mountpoints)
            or not isinstance(readonly, bool)
        ):
            raise D0Error("guest effective block-device identity differs")
        inventory.append(
            {
                "name": name,
                "kname": kname,
                "path": path,
                "pkname": parent,
                "size": size,
                "type": kind,
                "fstype": fstype,
                "mountpoints": mountpoints,
                "read_only": readonly,
            }
        )
    by_identifier: dict[str, dict[str, Any]] = {}
    for item in inventory:
        for identifier in (item["name"], item["kname"], item["path"]):
            if identifier in by_identifier and by_identifier[identifier] is not item:
                raise D0Error("guest block-device identifiers are duplicated")
            by_identifier[identifier] = item

    def backing_disk(source: str) -> tuple[dict[str, Any], dict[str, Any]]:
        device_path = source.split("[", 1)[0]
        leaf = by_identifier.get(device_path) or by_identifier.get(Path(device_path).name)
        if leaf is None:
            raise D0Error("guest mount source is absent from the block-device inventory")
        observed: set[str] = set()
        current = leaf
        while current["type"] != "disk":
            if current["name"] in observed or current["pkname"] is None:
                raise D0Error("guest mount backing-disk lineage differs")
            observed.add(current["name"])
            parent_device = by_identifier.get(current["pkname"])
            if parent_device is None:
                raise D0Error("guest mount parent device is absent")
            current = parent_device
        if leaf["read_only"] or current["read_only"]:
            raise D0Error("guest mount uses a read-only block device")
        return leaf, current

    root_mount = _single_findmnt(versions["root_mount"]["stdout"], expected_target="/")
    data_mount = _single_findmnt(
        versions["docker_data_mount"]["stdout"],
        expected_target="/var/lib/docker",
    )
    root_leaf, root_disk = backing_disk(root_mount["source"])
    data_leaf, data_disk = backing_disk(data_mount["source"])
    if (
        root_disk["name"] == data_disk["name"]
        or root_disk["size"] != 5 * 1024**3
        or data_disk["size"] != 13 * 1024**3
    ):
        raise D0Error("guest effective root/data backing disks differ")
    extra_writable_disks = [
        item["name"]
        for item in inventory
        if item["type"] == "disk"
        and not item["read_only"]
        and item["name"] not in {root_disk["name"], data_disk["name"]}
    ]
    if extra_writable_disks:
        raise D0Error("guest exposes an unauthorized extra writable disk")
    binfmt = _validate_guest_binfmt_inventory(versions["binfmt_handlers"]["stdout"])
    nested = _validate_guest_nested_virtualization(versions["nested_virtualization"]["stdout"])
    if versions["ssh_agent"]["stdout"] != "\n":
        raise D0Error("guest SSH agent forwarding is enabled")
    processes = {
        Path(line.strip()).name
        for line in versions["processes"]["stdout"].splitlines()
        if line.strip()
    }
    kubernetes_processes = {
        "kube-apiserver",
        "kube-controller-manager",
        "kube-scheduler",
        "kubelet",
        "k3s",
        "k3s-server",
        "containerd-shim-runc-v2-k8s.io",
    }
    rosetta_processes = processes & {"rosetta", "rosetta-linux"}
    if processes & kubernetes_processes:
        raise D0Error("guest Kubernetes runtime is unexpectedly active")
    if rosetta_processes:
        raise D0Error("guest Rosetta runtime is unexpectedly active")
    if "dockerd" not in processes:
        raise D0Error("guest Docker runtime process is absent")
    try:
        kubernetes = json.loads(versions["kubernetes_state"]["stdout"])
    except (TypeError, json.JSONDecodeError) as error:
        raise D0Error("guest Kubernetes state evidence is invalid") from error
    if kubernetes != {"paths": [], "units": []}:
        raise D0Error("guest Kubernetes configuration or service state is present")
    mount_projection = _guest_mount_projection(versions["mounts"]["stdout"])
    daemon_config = verify_daemon_config(versions["daemon_config"]["stdout"], engine_info)
    return {
        "cpu_count": cpu_count,
        "memory_bytes": memory_bytes,
        "virtualization": virtualization,
        "mounts": mount_projection,
        "block_devices": sorted(inventory, key=lambda item: item["path"]),
        "root_mount": root_mount,
        "docker_data_mount": data_mount,
        "root_leaf_device": root_leaf,
        "data_leaf_device": data_leaf,
        "root_backing_disk": root_disk,
        "data_backing_disk": data_disk,
        "root_disk_bytes": root_disk["size"],
        "data_disk_bytes": data_disk["size"],
        "extra_writable_disks": [],
        "runtime": "docker" if "dockerd" in processes else None,
        "rosetta_processes": sorted(rosetta_processes),
        "rosetta_enabled": bool(rosetta_processes),
        "nested_virtualization": nested,
        "nested_virtualization_enabled": nested["nested_virtualization_enabled"],
        "binfmt_handlers": binfmt,
        "ssh_agent_forwarding": False,
        "kubernetes_state": kubernetes,
        "kubernetes_active": bool(kubernetes["paths"] or kubernetes["units"]),
        "docker_daemon": daemon_config,
        "disabled_host_controls_bound_to_exact_config": {
            "auto_activate": False,
            "mount_inotify": False,
            "port_forwarder": "none",
            "ssh_config": False,
        },
        "status": "pass",
    }


def _validate_guest_package_license_inventory(
    packages: Any,
    licenses: Any,
) -> tuple[set[str], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Validate a complete package-to-copyright-path inventory.

    Ubuntu and Docker's upstream Debian packages do not guarantee that every
    installed package ships a copyright file.  Absence is therefore explicit
    evidence, not a reason to falsify the inventory or discard the otherwise
    exact package/SBOM attestation.
    """

    if not isinstance(packages, list) or not isinstance(licenses, list):
        raise D0Error("guest package or license inventory is absent")
    if len(packages) != len(licenses) or len(packages) == 0:
        raise D0Error("guest package/license inventory cardinality differs")
    names: set[str] = set()
    for item in packages:
        if not isinstance(item, dict) or set(item) != {"name", "version", "architecture"}:
            raise D0Error("guest package identity differs")
        name = item["name"]
        if not all(isinstance(item[key], str) and item[key] for key in item):
            raise D0Error("guest package fields differ")
        if name in names:
            raise D0Error("guest package names are duplicated")
        names.add(name)

    license_by_package: dict[str, dict[str, Any]] = {}
    for item in licenses:
        if not isinstance(item, dict) or set(item) != {
            "package",
            "requested",
            "resolved",
            "doc_dir",
            "doc_dir_exists",
            "doc_dir_is_symlink",
            "doc_dir_symlink_target",
            "requested_exists",
            "requested_is_symlink",
            "requested_symlink_target",
            "exists",
            "present",
            "size",
            "sha256",
        }:
            raise D0Error("guest license identity differs")
        package = item["package"]
        if not isinstance(package, str) or package not in names or package in license_by_package:
            raise D0Error("guest package names or licenses differ")
        requested = item["requested"]
        resolved = item["resolved"]
        doc_dir = item["doc_dir"]
        doc_dir_exists = item["doc_dir_exists"]
        doc_dir_is_symlink = item["doc_dir_is_symlink"]
        doc_dir_symlink_target = item["doc_dir_symlink_target"]
        requested_exists = item["requested_exists"]
        requested_is_symlink = item["requested_is_symlink"]
        requested_symlink_target = item["requested_symlink_target"]
        exists = item["exists"]
        present = item["present"]
        size = item["size"]
        digest = item["sha256"]
        base = package.split(":", 1)[0]
        if (
            requested != f"/usr/share/doc/{base}/copyright"
            or doc_dir != f"/usr/share/doc/{base}"
            or not isinstance(resolved, str)
            or re.fullmatch(r"/usr/share/doc/[^/]+/copyright", resolved) is None
            or not isinstance(doc_dir_exists, bool)
            or not isinstance(doc_dir_is_symlink, bool)
            or not isinstance(requested_exists, bool)
            or not isinstance(requested_is_symlink, bool)
            or not isinstance(exists, bool)
            or not isinstance(present, bool)
            or not isinstance(size, int)
            or isinstance(size, bool)
        ):
            raise D0Error("guest license path or metadata differs")
        if doc_dir_is_symlink:
            if (
                doc_dir_exists is not True
                or not isinstance(doc_dir_symlink_target, str)
                or not doc_dir_symlink_target
            ):
                raise D0Error("guest license directory symlink identity differs")
        elif doc_dir_symlink_target is not None:
            raise D0Error("guest license directory aliasing differs")
        if requested_is_symlink:
            if (
                requested_exists is not True
                or not isinstance(requested_symlink_target, str)
                or not requested_symlink_target
            ):
                raise D0Error("guest license symlink identity differs")
        elif requested_symlink_target is not None:
            raise D0Error("guest license path aliasing differs")
        if resolved != requested and not (doc_dir_is_symlink or requested_is_symlink):
            raise D0Error("guest license path aliasing differs")
        if present:
            if (
                exists is not True
                or requested_exists is not True
                or size <= 0
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                raise D0Error("guest present copyright identity differs")
        elif exists is not False or size != 0 or digest is not None:
            raise D0Error("guest absent copyright identity differs")
        license_by_package[package] = item
    if set(license_by_package) != names:
        raise D0Error("guest package names or licenses differ")

    lower_names = {name.lower() for name in names}
    required_groups = {
        "docker": any("docker" in name or "moby" in name for name in lower_names),
        "containerd": any("containerd" in name for name in lower_names),
        # Docker's upstream Ubuntu package owns runc through ``containerd.io``
        # rather than a separate ``runc`` package.  The caller additionally
        # verifies dpkg's exact owner for both runtime binaries.
        "runc": any(name.startswith("runc") or name == "containerd.io" for name in lower_names),
    }
    if not all(required_groups.values()):
        raise D0Error("guest Docker/containerd/runc package identity is incomplete")
    relevant_licenses = [
        value
        for name, value in license_by_package.items()
        if "docker" in name.lower()
        or "moby" in name.lower()
        or "containerd" in name.lower()
        or name.lower().startswith("runc")
    ]
    if not relevant_licenses:
        raise D0Error("guest runtime license inventory is absent")
    return names, license_by_package, relevant_licenses


def guest_package_audit(
    packet: Mapping[str, Any],
    runner: CommandRunner,
    *,
    engine_info: Mapping[str, Any],
) -> dict[str, Any]:
    completed = runner.run(
        [
            COLIMA,
            "ssh",
            "--profile",
            PROFILE,
            "--",
            "/usr/bin/python3",
            "-I",
            "-S",
            "-B",
            "-c",
            _GUEST_AUDIT_SCRIPT,
        ],
        timeout=900,
        maximum=64 * 1024 * 1024,
    )
    try:
        audit = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D0Error("guest package audit output is invalid") from error
    if (
        not isinstance(audit, dict)
        or audit.get("schema_id") != "cascadia.r2-map.d0-guest-audit.v1"
        or audit.get("schema_version") != 1
        or audit.get("machine") not in {"aarch64", "arm64"}
    ):
        raise D0Error("guest package audit identity differs")
    packages = audit.get("packages")
    licenses = audit.get("licenses")
    versions = audit.get("versions")
    if not isinstance(versions, dict):
        raise D0Error("guest package or license inventory is absent")
    _names, _license_by_package, relevant_licenses = _validate_guest_package_license_inventory(
        packages,
        licenses,
    )
    assert isinstance(packages, list)
    assert isinstance(licenses, list)
    spdx_packages: list[dict[str, Any]] = []
    for item in packages:
        name = item["name"]
        identifier = re.sub(r"[^A-Za-z0-9.-]", "-", name)
        identifier += "-" + hashlib.sha256(name.encode()).hexdigest()[:12]
        spdx_packages.append(
            {
                "SPDXID": f"SPDXRef-Package-{identifier}",
                "name": name,
                "versionInfo": item["version"],
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": "NOASSERTION",
                "copyrightText": "NOASSERTION",
                "externalRefs": [
                    {
                        "referenceCategory": "PACKAGE-MANAGER",
                        "referenceType": "purl",
                        "referenceLocator": f"pkg:deb/ubuntu/{name}@{item['version']}",
                    }
                ],
            }
        )
    docker_version = versions.get("docker")
    if not isinstance(docker_version, dict) or docker_version.get("returncode") != 0:
        raise D0Error("guest Docker version command failed")
    engine = verify_engine_version(
        docker_version.get("stdout", "").encode(),
        expected_client_version=EXPECTED_ENGINE_VERSION,
    )
    for binary in (
        "containerd",
        "containerd_owner",
        "daemon_config",
        "runc",
        "runc_owner",
        "os_release",
        "listeners",
        "network_addresses",
        "cpu_count",
        "meminfo",
        "virtualization",
        "mounts",
        "block_devices",
        "root_mount",
        "docker_data_mount",
        "binfmt_handlers",
        "nested_virtualization",
        "ssh_agent",
        "processes",
        "kubernetes_state",
    ):
        result = versions.get(binary)
        if not isinstance(result, dict) or result.get("returncode") != 0:
            raise D0Error(f"guest {binary} identity command failed")
    runtime_binary_owners = {
        "containerd": versions["containerd_owner"]["stdout"].strip(),
        "runc": versions["runc_owner"]["stdout"].strip(),
    }
    if runtime_binary_owners != {
        "containerd": "containerd.io: /usr/bin/containerd",
        "runc": "containerd.io: /usr/bin/runc",
    }:
        raise D0Error("guest containerd/runc package ownership differs")
    listener_output = versions["listeners"]["stdout"]
    network = _guest_network_projection(versions["network_addresses"]["stdout"])
    listeners = _guest_listener_allowlist(listener_output, network)
    effective_config = _guest_effective_config(versions, engine_info)
    spdx = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"cascadia-r2-d0-{packet['run_id']}-{packet['host']}-guest",
        "documentNamespace": (
            f"https://cascadia.invalid/r2-map/d0/{packet['run_id']}/{packet['host']}/guest"
        ),
        "creationInfo": {
            "creators": ["Tool: cascadia-r2-d0-helper"],
            "created": datetime.fromtimestamp(
                packet["issued_unix_ms"] / 1000,
                tz=timezone.utc,  # noqa: UP017 -- Apple system Python 3.9 has no datetime.UTC.
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "packages": sorted(spdx_packages, key=lambda item: item["name"]),
        "documentDescribes": sorted(item["SPDXID"] for item in spdx_packages),
    }
    spdx_bytes = canonical_json(spdx)
    return {
        "audit_sha256": sha256_bytes(canonical_json(audit)),
        "audit_script_sha256": GUEST_AUDIT_SCRIPT_SHA256,
        "package_count": len(packages),
        "license_count": len(licenses),
        "missing_license_count": sum(1 for item in licenses if item.get("present") is not True),
        "copyright_inventory_complete": True,
        "copyright_documents_all_present": all(item.get("present") is True for item in licenses),
        "package_inventory": packages,
        "license_inventory": licenses,
        "runtime_licenses": relevant_licenses,
        "engine": engine,
        "containerd_version": versions["containerd"]["stdout"].strip(),
        "runc_version": versions["runc"]["stdout"].strip(),
        "runtime_binary_package_owners": runtime_binary_owners,
        "os_release": versions["os_release"]["stdout"],
        "tcp_listeners": listener_output,
        "tcp_listener_allowlist": listeners,
        "network_addresses": network,
        "effective_config": effective_config,
        "spdx": spdx,
        "spdx_sha256": sha256_bytes(spdx_bytes),
    }


def _colima_status(value: bytes, packet: Mapping[str, Any]) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D0Error("Colima status output is invalid") from error
    if not isinstance(decoded, dict):
        raise D0Error("Colima status output is not an object")
    colima_home = Path(packet["paths"]["colima_home"])
    expected = {
        "display_name": f"colima [profile={PROFILE}]",
        "driver": "macOS Virtualization.Framework",
        "arch": "aarch64",
        "runtime": "docker",
        "mount_type": "virtiofs",
        "docker_socket": f"unix://{colima_home / PROFILE / 'docker.sock'}",
        "containerd_socket": f"unix://{colima_home / PROFILE / 'containerd.sock'}",
        "kubernetes": False,
        "cpu": 10,
        "memory": 14 * 1024**3,
        "disk": 13 * 1024**3,
    }
    modern_keys = set(expected)
    legacy_keys = modern_keys | {"status"}
    if set(decoded) not in {frozenset(modern_keys), frozenset(legacy_keys)}:
        raise D0Error("Colima status effective field set differs")
    effective = {key: decoded.get(key) for key in expected}
    if effective != expected:
        raise D0Error("Colima status effective configuration differs")
    observed_status = decoded.get("status")
    if observed_status is not None and (
        not isinstance(observed_status, str)
        or observed_status.lower() not in {"running", "started"}
    ):
        raise D0Error("Colima profile is not running")
    return {
        "status": "running",
        "schema": "legacy-explicit-status" if observed_status is not None else "modern-statusless",
        "observed_status": observed_status.lower() if isinstance(observed_status, str) else None,
        "effective": effective,
        "raw_sha256": sha256_bytes(value),
    }


def _read_bounded_regular(path: Path, maximum: int) -> bytes:
    identity = _regular_file_identity(path)
    if identity["size"] > maximum:
        raise D0Error("runtime artifact exceeds its byte bound")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    value = b"".join(chunks)
    if len(value) != identity["size"] or sha256_bytes(value) != identity["sha256"]:
        raise D0Error("runtime artifact changed while reading")
    return value


def _expected_colima_context(packet: Mapping[str, Any]) -> dict[str, Any]:
    name = f"colima-{PROFILE}"
    endpoint = runtime_environment(packet)["DOCKER_HOST"]
    directory = hashlib.sha256(name.encode("utf-8")).hexdigest()
    metadata = {
        "Name": name,
        "Metadata": {"Description": f"colima [profile={PROFILE}]"},
        "Endpoints": {"docker": {"Host": endpoint, "SkipTLSVerify": False}},
    }
    metadata_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    return {
        "name": name,
        "endpoint": endpoint,
        "directory": directory,
        "metadata": metadata,
        "metadata_bytes": metadata_bytes,
    }


def _verify_colima_context_storage(packet: Mapping[str, Any]) -> dict[str, Any]:
    expected = _expected_colima_context(packet)
    config_root = Path(packet["paths"]["docker_config"])
    contexts = config_root / "contexts"
    meta = contexts / "meta"
    named = meta / expected["directory"]
    for directory in (contexts, meta, named):
        observed = directory.lstat()
        if (
            not stat.S_ISDIR(observed.st_mode)
            or observed.st_uid != os.getuid()
            or stat.S_IMODE(observed.st_mode) != 0o755
        ):
            raise D0Error("isolated Docker context directory metadata differs")
    expected_children = {
        contexts: {"meta"},
        meta: {expected["directory"]},
        named: {"meta.json"},
    }
    for directory, names in expected_children.items():
        if {item.name for item in directory.iterdir()} != names:
            raise D0Error("isolated Docker context storage contains unexpected state")
    metadata_path = named / "meta.json"
    _verify_exact_file(metadata_path, expected["metadata_bytes"], 0o644)
    return {
        "name": expected["name"],
        "directory": expected["directory"],
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_bytes(expected["metadata_bytes"]),
        "tls_storage_absent": True,
        "credentials_absent": True,
    }


def _docker_context_snapshot(packet: Mapping[str, Any], runner: CommandRunner) -> dict[str, Any]:
    stored = _verify_colima_context_storage(packet)
    expected = _expected_colima_context(packet)
    current = runner.run([DOCKER, "context", "show"], maximum=64 * 1024).stdout
    if current != b"default\n":
        raise D0Error("Docker current context is not the unchanged built-in default")
    listed = runner.run(
        [DOCKER, "context", "ls", "--format", "{{json .}}"],
        maximum=1024 * 1024,
    ).stdout
    rows: list[dict[str, Any]] = []
    for line in listed.splitlines():
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise D0Error("Docker context-list output is invalid") from error
        if not isinstance(row, dict):
            raise D0Error("Docker context-list row is not an object")
        rows.append(row)
    expected_rows = {
        "default": {
            "Current": True,
            "Description": "Current DOCKER_HOST based configuration",
            "DockerEndpoint": expected["endpoint"],
            "Error": "",
            "Name": "default",
        },
        expected["name"]: {
            "Current": False,
            "Description": f"colima [profile={PROFILE}]",
            "DockerEndpoint": expected["endpoint"],
            "Error": "",
            "Name": expected["name"],
        },
    }
    if len(rows) != 2 or {row.get("Name"): row for row in rows} != expected_rows:
        raise D0Error("Docker context inventory differs")
    inspected_bytes = runner.run(
        [DOCKER, "context", "inspect", "default", expected["name"]],
        maximum=1024 * 1024,
    ).stdout
    try:
        inspected = json.loads(inspected_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D0Error("Docker default-context inspect output is invalid") from error
    if not isinstance(inspected, list) or len(inspected) != 2:
        raise D0Error("Docker context inspect cardinality differs")
    by_name = {item.get("Name"): item for item in inspected if isinstance(item, dict)}
    if set(by_name) != {"default", expected["name"]}:
        raise D0Error("Docker context inspect inventory differs")
    default = by_name["default"]
    named = by_name[expected["name"]]
    default_endpoints = default.get("Endpoints")
    named_endpoints = named.get("Endpoints")
    if not isinstance(default_endpoints, dict) or not isinstance(named_endpoints, dict):
        raise D0Error("Docker context endpoint inventory differs")
    default_endpoint = default_endpoints.get("docker")
    named_endpoint = named_endpoints.get("docker")
    expected_named_storage = {
        "MetadataPath": stored["metadata_path"].rsplit("/meta.json", 1)[0],
        "TLSPath": str(
            Path(packet["paths"]["docker_config"]) / "contexts" / "tls" / expected["directory"]
        ),
    }
    if (
        default.get("Metadata") != {}
        or default_endpoint != {"Host": expected["endpoint"], "SkipTLSVerify": False}
        or default.get("TLSMaterial") != {}
        or default.get("Storage") != {"MetadataPath": "<IN MEMORY>", "TLSPath": "<IN MEMORY>"}
        or named.get("Metadata") != expected["metadata"]["Metadata"]
        or named_endpoint != expected["metadata"]["Endpoints"]["docker"]
        or named.get("TLSMaterial") != {}
        or named.get("Storage") != expected_named_storage
    ):
        raise D0Error("Docker context identity differs")
    return {
        "current": "default",
        "context_count": 2,
        "context_names": ["default", expected["name"]],
        "list_sha256": sha256_bytes(listed),
        "inspect_sha256": sha256_bytes(inspected_bytes),
        "named_context": stored,
        "automatic_activation_disabled": True,
        "effective_docker_host": expected["endpoint"],
        "status": "pass",
    }


def _validate_positive_runtime_activity(
    packet: Mapping[str, Any], activity: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the exact owner-local Lima/Colima host activity envelope."""

    if activity.get("launchd") or activity.get("mounts"):
        raise D0Error("runtime startup-item or host-mount state differs")
    processes = activity.get("processes")
    observer_ancestors = activity.get("observer_ancestors")
    sockets = activity.get("active_unix_sockets")
    listeners = activity.get("active_tcp_listeners")
    if not all(
        isinstance(items, list) for items in (processes, observer_ancestors, sockets, listeners)
    ):
        raise D0Error("runtime activity inventory is absent")
    owner = Path(HOST_HOME[packet["host"]]).name
    colima_home = Path(packet["paths"]["colima_home"])
    instance = colima_home / "_lima" / f"colima-{PROFILE}"
    network = colima_home / "_lima" / "_networks" / "user-v2"

    parsed_processes: list[tuple[int, int, str, str]] = []
    for line in processes:
        match = re.fullmatch(r"\s*(\d+)\s+(\d+)\s+(\S+)\s+(.+)", line)
        if match is None:
            raise D0Error("runtime process identity differs")
        parsed_processes.append(
            (int(match.group(1)), int(match.group(2)), match.group(3), match.group(4))
        )
    parsed_observer_ancestors: list[tuple[int, int, str, str]] = []
    for line in observer_ancestors:
        match = re.fullmatch(r"\s*(\d+)\s+(\d+)\s+(\S+)\s+(.+)", line)
        if match is None:
            raise D0Error("runtime observer ancestor identity differs")
        parsed_observer_ancestors.append(
            (int(match.group(1)), int(match.group(2)), match.group(3), match.group(4))
        )
    expected_usernet = (
        f"{LIMACTL} usernet -p {network}/usernet_user-v2.pid "
        f"-e {network}/user-v2_ep.sock --listen-qemu {network}/user-v2_qemu.sock "
        f"--listen {network}/user-v2_fd.sock --subnet 192.168.5.0/24"
    )
    expected_hostagent = (
        f"{LIMACTL} hostagent --pidfile {instance}/ha.pid --socket {instance}/ha.sock "
        f"--guestagent /opt/homebrew/share/lima/lima-guestagent.Linux-aarch64.gz "
        f"colima-{PROFILE}"
    )
    usernet_pids_by_argv = {
        pid
        for pid, parent, user, command in parsed_processes
        if parent == 1 and user == owner and command == expected_usernet
    }
    hostagent_pids_by_argv = {
        pid
        for pid, parent, user, command in parsed_processes
        if parent == 1 and user == owner and command == expected_hostagent
    }
    ssh_rows = [
        (pid, parent, user, command)
        for pid, parent, user, command in parsed_processes
        if command == f"ssh: {instance}/ssh.sock [mux]"
    ]
    daemon_pids = (
        usernet_pids_by_argv
        | hostagent_pids_by_argv
        | {row[0] for row in ssh_rows if row[1] == 1 and row[2] == owner}
    )

    # John1 continuously publishes the required dashboard heartbeat. Its
    # read-only observation is implemented by ``colima status``, which briefly
    # forks ``limactl list``. A process-table snapshot may catch the child
    # before macOS has replaced its provisional ``(colima)`` or ``(limactl)``
    # argv. Admit only that exact observer parent/child chain; every init-owned
    # process and every other runtime client remains fail-closed.
    watcher_pids, observer_pids = _dashboard_observer_processes(
        packet,
        parsed_processes,
        parsed_observer_ancestors,
    )
    if (
        {row[0] for row in parsed_processes} != daemon_pids | observer_pids
        or len(daemon_pids) != 3
        or len(usernet_pids_by_argv) != 1
        or len(hostagent_pids_by_argv) != 1
        or len(ssh_rows) != 1
        or ssh_rows[0][1] != 1
        or ssh_rows[0][2] != owner
    ):
        raise D0Error("runtime process set differs")
    ssh_pid = ssh_rows[0][0]
    usernet_pid = next(iter(usernet_pids_by_argv))
    hostagent_pid = next(iter(hostagent_pids_by_argv))
    lima_pids = {usernet_pid, hostagent_pid}

    socket_owners: dict[str, int] = {}
    allowed_arrow_rows = 0
    allowed_ssh_patterns = (
        re.compile(rf"{re.escape(str(instance / 'ssh.sock'))}\.[A-Za-z0-9]+\Z"),
        re.compile(r"/tmp/lima-psl-127\.0\.0\.1-53-[0-9]+/sock\Z"),
    )
    required_paths = {
        str(network / "user-v2_ep.sock"),
        str(network / "user-v2_qemu.sock"),
        str(network / "user-v2_fd.sock"),
        str(instance / "ha.sock"),
        str(colima_home / PROFILE / "docker.sock"),
        str(colima_home / PROFILE / "containerd.sock"),
    }
    for line in sockets:
        match = re.match(r"^(\S+)\s+(\d+)\s+(\S+)\s+", line)
        if match is None or match.group(3) != owner:
            raise D0Error("runtime Unix-socket owner differs")
        command, pid = match.group(1), int(match.group(2))
        if (command == "limactl" and pid not in lima_pids) or (command == "ssh" and pid != ssh_pid):
            raise D0Error("runtime Unix-socket process differs")
        if command not in {"limactl", "ssh"}:
            raise D0Error("runtime Unix-socket command differs")
        endpoint = line.rsplit(None, 1)[-1]
        if endpoint.startswith("->0x"):
            if command != "limactl":
                raise D0Error("runtime anonymous Unix-socket peer differs")
            allowed_arrow_rows += 1
            continue
        if endpoint in required_paths:
            socket_owners[endpoint] = pid
            continue
        if command == "ssh" and any(
            pattern.fullmatch(endpoint) for pattern in allowed_ssh_patterns
        ):
            continue
        raise D0Error("runtime Unix-socket path differs")
    if set(socket_owners) != required_paths or allowed_arrow_rows < 1:
        raise D0Error("runtime required Unix-socket set differs")
    usernet_pids = {
        socket_owners[str(network / name)]
        for name in ("user-v2_ep.sock", "user-v2_qemu.sock", "user-v2_fd.sock")
    }
    socket_hostagent_pid = socket_owners[str(instance / "ha.sock")]
    if (
        usernet_pids != {usernet_pid}
        or socket_hostagent_pid != hostagent_pid
        or hostagent_pid in usernet_pids
    ):
        raise D0Error("runtime Lima process roles differ")
    if (
        socket_owners[str(colima_home / PROFILE / "docker.sock")] != ssh_pid
        or socket_owners[str(colima_home / PROFILE / "containerd.sock")] != ssh_pid
    ):
        raise D0Error("runtime forwarded socket ownership differs")

    listener_roles: set[str] = set()
    for line in listeners:
        match = re.match(
            r"^limactl\s+(\d+)\s+(\S+)\s+.*\s(IPv[46])\s+.*\sTCP\s+"
            r"(\S+)\s+\(LISTEN\)$",
            line,
        )
        if match is None or match.group(2) != owner:
            raise D0Error("runtime TCP listener identity differs")
        pid, family, endpoint = int(match.group(1)), match.group(3), match.group(4)
        if pid == usernet_pid and family == "IPv4" and endpoint.startswith("127.0.0.1:"):
            try:
                port = int(endpoint.rsplit(":", 1)[1])
            except ValueError as error:
                raise D0Error("runtime usernet listener port differs") from error
            if not 49152 <= port <= 65535:
                raise D0Error("runtime usernet listener port differs")
            listener_roles.add("usernet-loopback")
        elif pid == hostagent_pid and family == "IPv6" and endpoint == "*:53":
            listener_roles.add("hostagent-dns")
        else:
            raise D0Error("runtime TCP listener is outside the exact Lima allowlist")
    if len(listeners) != 2 or listener_roles != {"usernet-loopback", "hostagent-dns"}:
        raise D0Error("runtime TCP listener set differs")
    return {
        **activity,
        "launchd_startup_items_absent": True,
        "host_runtime_mounts_absent": True,
        "host_runtime_tcp_listener_roles": sorted(listener_roles),
        "host_runtime_tcp_listeners_exact": True,
        "runtime_observer_ancestor_count": len(watcher_pids),
        "runtime_observer_ancestry_exact": True,
        "runtime_observer_process_count": len(observer_pids),
        "runtime_observer_processes_exact": True,
        "runtime_process_count": 3,
        "required_unix_socket_count": len(required_paths),
        "status": "pass",
    }


def _dashboard_observer_processes(
    packet: Mapping[str, Any],
    parsed_processes: Sequence[tuple[int, int, str, str]],
    parsed_observer_ancestors: Sequence[tuple[int, int, str, str]],
) -> tuple[set[int], set[int]]:
    """Authenticate the one permitted read-only dashboard process chain."""

    owner = Path(HOST_HOME[packet["host"]]).name
    expected_watcher = (
        f"{HOST_HOME['john1']}/cascadia/.venv/bin/python "
        "tools/r2_map_d0_dashboard_watch.py --watch --interval-seconds 5"
    )
    watcher_rows = [
        (pid, parent, user, command)
        for pid, parent, user, command in parsed_observer_ancestors
        if parent == 1 and user == owner and command == expected_watcher
    ]
    expected_watcher_count = 1 if packet["host"] == "john1" else 0
    if (
        len(parsed_observer_ancestors) != expected_watcher_count
        or len(watcher_rows) != expected_watcher_count
    ):
        raise D0Error("runtime dashboard observer ancestry differs")
    watcher_pids = {row[0] for row in watcher_rows}
    expected_status = f"{COLIMA} status --profile {PROFILE}"
    expected_list = f"{LIMACTL} list colima-{PROFILE} --json"
    status_observer_pids = {
        pid
        for pid, parent, user, command in parsed_processes
        if parent in watcher_pids and user == owner and command == expected_status
    }
    observer_pids = set(status_observer_pids)
    for pid, parent, user, command in parsed_processes:
        if (
            parent in status_observer_pids
            and user == owner
            and command in {expected_list, "(colima)", "(limactl)"}
        ):
            observer_pids.add(pid)
    return watcher_pids, observer_pids


def _validate_inactive_runtime_activity(
    packet: Mapping[str, Any], activity: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove runtime inactivity while admitting only the exact dashboard observer."""

    processes = activity.get("processes")
    observer_ancestors = activity.get("observer_ancestors")
    sockets = activity.get("active_unix_sockets")
    listeners = activity.get("active_tcp_listeners")
    launchd = activity.get("launchd")
    mounts = activity.get("mounts")
    if not all(
        isinstance(items, list)
        for items in (processes, observer_ancestors, sockets, listeners, launchd, mounts)
    ):
        raise D0Error("runtime activity inventory is absent")
    if sockets or listeners or launchd or mounts:
        raise D0Error("runtime activity remains while the runtime is required inactive")

    def parse_rows(rows: Sequence[Any], *, label: str) -> list[tuple[int, int, str, str]]:
        parsed: list[tuple[int, int, str, str]] = []
        for line in rows:
            match = re.fullmatch(r"\s*(\d+)\s+(\d+)\s+(\S+)\s+(.+)", str(line))
            if match is None:
                raise D0Error(f"runtime {label} identity differs")
            parsed.append(
                (int(match.group(1)), int(match.group(2)), match.group(3), match.group(4))
            )
        return parsed

    parsed_processes = parse_rows(processes, label="process")
    parsed_observer_ancestors = parse_rows(
        observer_ancestors,
        label="observer ancestor",
    )
    watcher_pids, observer_pids = _dashboard_observer_processes(
        packet,
        parsed_processes,
        parsed_observer_ancestors,
    )
    if {row[0] for row in parsed_processes} != observer_pids:
        raise D0Error("runtime process remains while the runtime is required inactive")
    return {
        **activity,
        "inactive": True,
        "runtime_daemons_inactive": True,
        "runtime_observer_ancestor_count": len(watcher_pids),
        "runtime_observer_process_count": len(observer_pids),
        "runtime_observer_processes_exact": True,
        "status": "pass",
    }


def _inactive_runtime_activity(packet: Mapping[str, Any]) -> dict[str, Any]:
    return _validate_inactive_runtime_activity(packet, runtime_activity())


def _positive_runtime_activity(packet: Mapping[str, Any]) -> dict[str, Any]:
    return _validate_positive_runtime_activity(packet, runtime_activity())


RUNTIME_ACTIVITY_CONVERGENCE_DEADLINE_SECONDS = 5.0
RUNTIME_ACTIVITY_CONVERGENCE_INTERVAL_SECONDS = 0.2
RUNTIME_ACTIVITY_CONVERGENCE_MAX_SAMPLES = 20


def _bounded_activity_rows(activity: Mapping[str, Any]) -> dict[str, Any]:
    """Return bounded, normalized row evidence suitable for a failure report."""

    fields = (
        "processes",
        "observer_ancestors",
        "active_unix_sockets",
        "active_tcp_listeners",
        "launchd",
        "mounts",
    )
    result: dict[str, Any] = {}
    for field in fields:
        value = activity.get(field)
        rows = sorted(str(row)[:1024] for row in value) if isinstance(value, list) else []
        result[field] = {
            "count": len(rows),
            "rows": rows[:16],
            "rows_sha256": sha256_bytes(canonical_json(rows)),
            "truncated": len(rows) > 16,
        }
    return result


def _stable_activity_projection(validated: Mapping[str, Any]) -> dict[str, Any]:
    """Exclude only authenticated transient dashboard children from stability."""

    daemon_rows: list[str] = []
    for line in validated["processes"]:
        match = re.fullmatch(r"\s*(\d+)\s+(\d+)\s+(\S+)\s+(.+)", line)
        if match is not None and int(match.group(2)) == 1:
            daemon_rows.append(line)
    return {
        "daemon_processes": sorted(daemon_rows),
        "observer_ancestors": sorted(validated["observer_ancestors"]),
        "active_unix_sockets": sorted(validated["active_unix_sockets"]),
        "active_tcp_listeners": sorted(validated["active_tcp_listeners"]),
        "launchd": sorted(validated["launchd"]),
        "mounts": sorted(validated["mounts"]),
    }


def _converged_positive_runtime_activity(
    packet: Mapping[str, Any],
    *,
    sampler: Callable[[], Mapping[str, Any]] = runtime_activity,
    deadline_seconds: float = RUNTIME_ACTIVITY_CONVERGENCE_DEADLINE_SECONDS,
    interval_seconds: float = RUNTIME_ACTIVITY_CONVERGENCE_INTERVAL_SECONDS,
    max_samples: int = RUNTIME_ACTIVITY_CONVERGENCE_MAX_SAMPLES,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Require two consecutive identical valid post-restart activity samples."""

    deadline = clock() + deadline_seconds
    sequence: list[dict[str, Any]] = []
    previous_digest: str | None = None
    for index in range(max_samples):
        activity = dict(sampler())
        diagnostic = _bounded_activity_rows(activity)
        try:
            validated = _validate_positive_runtime_activity(packet, activity)
        except D0Error as error:
            previous_digest = None
            sequence.append(
                {
                    "index": index + 1,
                    "status": "invalid",
                    "error": str(error),
                    "activity": diagnostic,
                }
            )
        else:
            projection = _stable_activity_projection(validated)
            digest = sha256_bytes(canonical_json(projection))
            sequence.append(
                {
                    "index": index + 1,
                    "status": "valid",
                    "stable_projection_sha256": digest,
                    "observer_process_count": validated["runtime_observer_process_count"],
                    "activity": diagnostic,
                }
            )
            if digest == previous_digest:
                return {
                    **validated,
                    "stability_convergence": {
                        "attempt_count": index + 1,
                        "consecutive_identical_valid_samples": 2,
                        "deadline_seconds": deadline_seconds,
                        "interval_seconds": interval_seconds,
                        "stable_projection_sha256": digest,
                        "sample_sequence": sequence,
                        "status": "pass",
                    },
                }
            previous_digest = digest
        if index + 1 >= max_samples or clock() >= deadline:
            break
        sleeper(interval_seconds)
    evidence = {
        "deadline_seconds": deadline_seconds,
        "interval_seconds": interval_seconds,
        "sample_sequence": sequence,
        "status": "fail",
    }
    raise D0Error(
        "runtime activity failed bounded convergence: " + canonical_json(evidence).decode("ascii")
    )


def stop_start_recovery(
    packet: Mapping[str, Any],
    runner: CommandRunner,
    *,
    smoke_archive: bytes,
    first_smoke: Mapping[str, Any],
    context_before: Mapping[str, Any],
    engine_identity_before: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove an exact stop/start cycle without context or smoke drift."""

    observed_engine_before = _stable_engine_identity(runner)
    if observed_engine_before != engine_identity_before:
        raise D0Error("Docker Engine identity drifted before stop/start recovery")
    stopped = runner.run([COLIMA, "stop", "--profile", PROFILE], timeout=900)
    socket_path = Path(packet["paths"]["colima_home"]) / PROFILE / "docker.sock"
    if socket_path.exists() or socket_path.is_symlink():
        raise D0Error("Docker socket survived the exact Colima stop")
    stopped_activity = _inactive_runtime_activity(packet)
    if not stopped_activity["inactive"]:
        raise D0Error("runtime activity survived the exact Colima stop")
    command = start_plan(packet)[0]
    restarted = runner.run(command.argv, timeout=1800)
    verify_configs(packet)
    status = _colima_status(
        runner.run([COLIMA, "status", "--profile", PROFILE, "--json"]).stdout,
        packet,
    )
    socket = verify_socket(packet)
    observed_engine_after = _stable_engine_identity(runner)
    if observed_engine_after != observed_engine_before:
        raise D0Error("Docker Engine identity drifted across stop/start recovery")
    context_after = _docker_context_snapshot(packet, runner)
    if context_after != context_before:
        raise D0Error("Docker default context drifted across stop/start recovery")
    recovered_smoke = smoke_and_volume_roundtrip(
        runner,
        run_id=packet["run_id"],
        smoke_archive=smoke_archive,
    )
    if recovered_smoke != first_smoke:
        raise D0Error("smoke result drifted across stop/start recovery")
    activity = _converged_positive_runtime_activity(packet)
    return {
        "stop_stdout_sha256": sha256_bytes(stopped.stdout),
        "stop_stderr_sha256": sha256_bytes(stopped.stderr),
        "stopped_activity": stopped_activity,
        "restart_command": command.as_dict(),
        "restart_stdout_sha256": sha256_bytes(restarted.stdout),
        "restart_stderr_sha256": sha256_bytes(restarted.stderr),
        "status_after_restart": status,
        "socket_after_restart": socket,
        "engine_before_stop": observed_engine_before,
        "engine_after_restart": observed_engine_after,
        "engine_identity_unchanged": True,
        "context_after_restart": context_after,
        "smoke_after_restart": recovered_smoke,
        "runtime_activity_after_restart": activity,
        "identical_smoke": True,
        "status": "pass",
    }


def verify_positive_runtime(
    packet: Mapping[str, Any],
    runner: CommandRunner,
    *,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the complete positive D0 runtime, storage, security, and feature gate."""

    validate_work_packet(packet)
    host = packet["host"]
    if host not in {"john1", "john2", "john3"}:
        raise D0Error("positive runtime verification is not authorized on this host")
    if REQUIRED_OPERATION["verify"] not in packet["allowed_operations"]:
        raise D0Error("work packet does not authorize runtime verification")
    preflight_resources = preflight.get("resources")
    preflight_homebrew = preflight.get("homebrew")
    if not isinstance(preflight_resources, Mapping) or not isinstance(preflight_homebrew, Mapping):
        raise D0Error("positive verification lacks the original preflight ledger")
    preflight_available_kib = preflight_resources.get("data_volume_kib_available")
    if (
        not isinstance(preflight_available_kib, int)
        or preflight_resources.get("swap_used_bytes") != 0
    ):
        raise D0Error("positive verification preflight resources differ")
    verify_configs(packet)
    before_resources = host_resource_snapshot(runner)
    version_outputs = {
        "colima": runner.run([COLIMA, "version"]).stdout,
        "lima": runner.run([LIMACTL, "--version"]).stdout,
        "docker": runner.run([DOCKER, "--version"]).stdout,
    }
    for component, output in version_outputs.items():
        verify_version_output(component, output)
    status_output = runner.run([COLIMA, "status", "--profile", PROFILE, "--json"]).stdout
    version_output = runner.run([DOCKER, "version", "--format", "{{json .}}"]).stdout
    info_output = runner.run([DOCKER, "info", "--format", "{{json .}}"]).stdout
    engine = verify_engine_version(version_output)
    engine_info = verify_engine_info(info_output)
    docker_context = _docker_context_snapshot(packet, runner)
    runtime_activity_before = _positive_runtime_activity(packet)
    buildkit: dict[str, Any]
    if host == "john2":
        if "buildkit-probe" not in packet["allowed_operations"]:
            raise D0Error("John2 verification packet omits the explicit BuildKit probe")
        buildx_version = runner.run([DOCKER, "buildx", "version"]).stdout
        verify_version_output("docker-buildx", buildx_version)
        inspect = runner.run(
            [DOCKER, "buildx", "inspect", "--builder", "default", "--bootstrap"],
            timeout=300,
        ).stdout
        buildkit = {
            "identity": verify_buildx_inspect(inspect),
            "feature_probe": full_policy_buildkit_probe(packet, runner),
        }
    else:
        forbidden = [
            Path("/opt/homebrew/lib/docker/cli-plugins/docker-buildx"),
            Path(packet["paths"]["docker_config"]) / "buildx",
            Path(HOST_HOME[host]) / ".local/share/buildkit",
        ]
        present = [str(path) for path in forbidden if path.exists() or path.is_symlink()]
        if present:
            raise D0Error(f"{host} buildx/builder state is present: {present!r}")
        buildx = runner.run([DOCKER, "buildx", "version"], check=False)
        if buildx.returncode == 0:
            raise D0Error(f"{host} exposes the unauthorized buildx command")
        daemon_cache = _guest_buildkit_state(runner)
        _require_empty_buildkit_state(daemon_cache)
        buildkit = {
            "buildx_plugin_absent": True,
            "builder_profiles_absent": True,
            "stock_daemon_build_api_present": True,
            "daemon_cache_before": daemon_cache,
        }
    smoke_path = Path(packet["paths"]["smoke_oci"])
    smoke_identity = _regular_file_identity(smoke_path)
    expected_smoke = packet["artifacts"]["smoke_oci"]
    if (
        smoke_identity["size"] != expected_smoke["size"]
        or smoke_identity["sha256"] != expected_smoke["sha256"]
    ):
        raise D0Error("staged Alpine smoke OCI identity differs")
    smoke_archive = _read_bounded_regular(smoke_path, 64 * 1024 * 1024)
    smoke = smoke_and_volume_roundtrip(
        runner,
        run_id=packet["run_id"],
        smoke_archive=smoke_archive,
    )
    recovery = stop_start_recovery(
        packet,
        runner,
        smoke_archive=smoke_archive,
        first_smoke=smoke,
        context_before=docker_context,
        engine_identity_before={"version": engine, "info": engine_info},
    )
    if recovery["engine_after_restart"] != {"version": engine, "info": engine_info}:
        raise D0Error("Docker Engine identity differs after stop/start recovery")
    if host in {"john1", "john3"}:
        daemon_cache_after = _guest_buildkit_state(runner)
        _require_empty_buildkit_state(daemon_cache_after)
        if daemon_cache_after != buildkit["daemon_cache_before"]:
            raise D0Error(f"{host} stock daemon BuildKit state drifted during verification")
        buildkit["daemon_cache_after"] = daemon_cache_after
        buildkit["daemon_cache_unchanged"] = True
    engine_objects_after_recovery = _engine_object_inventory(runner)
    guest = guest_package_audit(packet, runner, engine_info=engine_info)
    homebrew_after = inventory_roots(
        homebrew_ledger_paths(Path(HOST_HOME[host]), host, packet),
        label=f"{packet['run_id']}-{host}-homebrew-positive",
        policy=InventoryPolicy(full_hash_limit=64 * 1024 * 1024, max_entries=500_000),
    )
    homebrew_comparison = compare_homebrew_install(
        dict(preflight_homebrew),
        homebrew_after,
        allowed_formulae=formulas_for_host(host),
        allowed_isolated_roots=(
            packet["paths"]["homebrew_cache"],
            packet["paths"]["homebrew_logs"],
            packet["paths"]["homebrew_temp"],
        ),
        label=f"{packet['run_id']}-{host}-homebrew-positive-delta",
    )
    if homebrew_comparison["status"] != "pass":
        raise D0Error("Homebrew positive-state delta escaped the frozen closure")
    podman_runtime: dict[str, Any] | None = None
    if host == "john1":
        podman_before = preflight.get("podman_negative_control")
        podman_runtime = podman_negative_control(Path(HOST_HOME[host]))
        if (
            not isinstance(podman_before, Mapping)
            or podman_before.get("status") != "pass"
            or podman_before.get("semantic_sha256") != podman_runtime["semantic_sha256"]
        ):
            raise D0Error("John1 Podman semantics changed during Colima verification")
    staging_cleanup = cleanup_bootstrap_staging(packet)
    footprint = runtime_footprint(packet)
    budget = enforce_footprint_budget(
        footprint,
        preflight_available_kib=preflight_available_kib,
    )
    after_resources = host_resource_snapshot(runner)
    if before_resources["swap_used_bytes"] != 0 or after_resources["swap_used_bytes"] != 0:
        raise D0Error("runtime verification observed nonzero host swap use")
    return {
        "versions": {key: value.decode("utf-8").strip() for key, value in version_outputs.items()},
        "colima": _colima_status(status_output, packet),
        "socket": verify_socket(packet),
        "engine": engine,
        "engine_info": engine_info,
        "docker_context": docker_context,
        "runtime_activity_before": runtime_activity_before,
        "stop_start_recovery": recovery,
        "engine_objects_after_recovery": engine_objects_after_recovery,
        "buildkit": buildkit,
        "guest": guest,
        "homebrew_after": homebrew_after,
        "homebrew_comparison": homebrew_comparison,
        "podman_runtime": podman_runtime,
        "podman_semantics_stable": podman_runtime is not None if host == "john1" else None,
        "smoke_image": {
            "index_digest": SMOKE_IMAGE["index_digest"],
            "manifest_digest": SMOKE_IMAGE["manifest_digest"],
            "archive": smoke_identity,
            "roundtrip": smoke,
        },
        "bootstrap_staging_cleanup": staging_cleanup,
        "runtime_footprint": footprint,
        "budget": budget,
        "resources_before": before_resources,
        "resources_after": after_resources,
        "status": "pass",
    }


def hardened_flags(name: str, *, user: str = "65532:65532") -> list[str]:
    flags = [
        "--name",
        name,
        "--platform",
        "linux/arm64",
        "--pull",
        "never",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges=true",
        "--ipc",
        "none",
        "--pids-limit",
        "64",
        "--memory",
        "64m",
        "--memory-swap",
        "64m",
        "--cpus",
        "1",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=16m",
    ]
    if user:
        flags.extend(["--user", user])
    return flags


def _validate_default_private_pid_mode(host_config: Mapping[str, Any]) -> dict[str, Any]:
    """Prove Docker's empty/default mode is a private PID namespace.

    Docker accepts explicit sharing modes such as ``host`` and
    ``container:<id>``. It does not accept the literal mode ``private``; the
    private namespace is represented by an empty ``HostConfig.PidMode``.
    """

    if host_config.get("PidMode") != "":
        raise D0Error("smoke container PID namespace mode differs")
    return {
        "effective_pid_mode": "default-private",
        "host_pid_namespace_shared": False,
        "container_pid_namespace_shared": False,
        "status": "pass",
    }


def _safe_tar(value: bytes, maximum: int = 64 * 1024 * 1024) -> dict[str, bytes]:
    if len(value) > maximum:
        raise D0Error("tar output exceeds its byte bound")
    result: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(value), mode="r:") as archive:
            for member in archive:
                if member.isdir():
                    if member.name.startswith("/") or ".." in member.name.split("/"):
                        raise D0Error("tar output contains an unsafe directory")
                    continue
                if (
                    not member.isfile()
                    or member.name.startswith("/")
                    or ".." in member.name.split("/")
                    or member.name in result
                    or member.size > maximum
                ):
                    raise D0Error("tar output contains an unsafe member")
                stream = archive.extractfile(member)
                if stream is None:
                    raise D0Error("tar output member is unreadable")
                payload = stream.read(maximum + 1)
                if len(payload) != member.size:
                    raise D0Error("tar output member size differs")
                result[member.name] = payload
    except (tarfile.TarError, OSError) as error:
        raise D0Error("tar output is invalid") from error
    return result


OCI_INDEX_MEDIA_TYPES = {
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
}
OCI_MANIFEST_MEDIA_TYPES = {
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
}


def _probe_oci_graph(value: bytes) -> dict[str, Any]:
    members = _safe_tar(value)
    index_bytes = members.get("index.json")
    layout_bytes = members.get("oci-layout")
    if index_bytes is None or layout_bytes is None:
        raise D0Error("probe OCI graph lacks index or layout")
    index = _json_command(index_bytes, "probe OCI graph index")
    layout = _json_command(layout_bytes, "probe OCI graph layout")
    if layout != {"imageLayoutVersion": "1.0.0"}:
        raise D0Error("probe OCI graph layout differs")
    if (
        not isinstance(index, dict)
        or index.get("schemaVersion") != 2
        or index.get("mediaType") not in OCI_INDEX_MEDIA_TYPES
    ):
        raise D0Error("probe OCI graph root index identity differs")
    root_descriptors = index.get("manifests") if isinstance(index, dict) else None
    if not isinstance(root_descriptors, list):
        raise D0Error("probe OCI graph root descriptors differ")
    referenced = {"index.json", "oci-layout"}
    visiting: set[str] = set()
    visited: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    leaves: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    def descriptor_blob(descriptor: Mapping[str, Any]) -> tuple[str, bytes]:
        digest = descriptor.get("digest")
        size = descriptor.get("size")
        media_type = descriptor.get("mediaType")
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(media_type, str)
        ):
            raise D0Error("probe OCI graph descriptor identity differs")
        name = f"blobs/sha256/{digest.split(':', 1)[1]}"
        payload = members.get(name)
        if payload is None:
            raise D0Error("probe OCI graph descriptor is dangling")
        if len(payload) != size or f"sha256:{sha256_bytes(payload)}" != digest:
            raise D0Error("probe OCI graph descriptor digest or size differs")
        referenced.add(name)
        return name, payload

    def walk(descriptor: Any, *, parent_digest: str | None) -> None:
        if not isinstance(descriptor, dict):
            raise D0Error("probe OCI graph descriptor is not an object")
        digest = descriptor.get("digest")
        edges.append(
            {
                "child_digest": digest,
                "descriptor": descriptor,
                "parent_digest": parent_digest,
            }
        )
        if digest in visiting:
            raise D0Error("probe OCI graph contains a descriptor cycle")
        if isinstance(digest, str) and digest in visited:
            if visited[digest]["descriptor"] != descriptor:
                raise D0Error("probe OCI graph descriptor alias differs")
            return
        member_name, payload = descriptor_blob(descriptor)
        assert isinstance(digest, str)
        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise D0Error("probe OCI graph descriptor payload is invalid JSON") from error
        if not isinstance(document, dict):
            raise D0Error("probe OCI graph descriptor payload is not an object")
        media_type = descriptor["mediaType"]
        visiting.add(digest)
        order.append(digest)
        if media_type in OCI_INDEX_MEDIA_TYPES:
            if document.get("schemaVersion") != 2 or document.get("mediaType") != media_type:
                raise D0Error("probe OCI nested index identity differs")
            children = document.get("manifests")
            if not isinstance(children, list):
                raise D0Error("probe OCI nested index descriptors differ")
            node = {
                "descriptor": descriptor,
                "document": document,
                "kind": "index",
                "member": member_name,
            }
            visited[digest] = node
            for child in children:
                walk(child, parent_digest=digest)
        elif media_type in OCI_MANIFEST_MEDIA_TYPES:
            if document.get("schemaVersion") != 2 or document.get("mediaType") != media_type:
                raise D0Error("probe OCI leaf manifest identity differs")
            config = document.get("config")
            layers = document.get("layers")
            if not isinstance(config, dict) or not isinstance(layers, list):
                raise D0Error("probe OCI leaf manifest structure differs")
            references: list[dict[str, Any]] = []
            for kind, reference in [
                ("config", config),
                *(("layer", layer) for layer in layers),
            ]:
                if not isinstance(reference, dict):
                    raise D0Error("probe OCI leaf blob descriptor differs")
                reference_name, reference_payload = descriptor_blob(reference)
                references.append(
                    {
                        "descriptor": reference,
                        "kind": kind,
                        "member": reference_name,
                        "sha256": sha256_bytes(reference_payload),
                        "size": len(reference_payload),
                    }
                )
            node = {
                "descriptor": descriptor,
                "document": document,
                "kind": "manifest",
                "member": member_name,
                "references": references,
            }
            visited[digest] = node
            leaves.append(descriptor)
        else:
            raise D0Error("probe OCI graph descriptor media type differs")
        visiting.remove(digest)

    for descriptor in root_descriptors:
        walk(descriptor, parent_digest=None)
    unreferenced = sorted(set(members) - referenced)
    if unreferenced:
        raise D0Error("probe OCI graph has unreferenced members")
    return {
        "attestation_descriptor_count": sum(
            1
            for descriptor in leaves
            if isinstance(descriptor.get("annotations"), dict)
            and descriptor["annotations"].get("vnd.docker.reference.type") == "attestation-manifest"
        ),
        "edges": edges,
        "index": index,
        "layout": layout,
        "leaf_descriptors": leaves,
        "nodes": [visited[digest] for digest in order],
        "referenced_members": sorted(referenced),
        "root_descriptors": root_descriptors,
        "status": "pass",
        "unreferenced_members": [],
    }


def _validate_probe_oci_attachment_contract(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Validate Docker's index-level attachment when in-toto subjects are empty."""

    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise D0Error("probe OCI attachment graph is incomplete")
    node_by_digest = {
        node["descriptor"]["digest"]: node
        for node in nodes
        if isinstance(node, dict)
        and isinstance(node.get("descriptor"), dict)
        and isinstance(node["descriptor"].get("digest"), str)
    }
    manifest_nodes = [
        node for node in nodes if isinstance(node, dict) and node.get("kind") == "manifest"
    ]
    attestations = []
    for node in manifest_nodes:
        descriptor = node["descriptor"]
        annotations = descriptor.get("annotations")
        if (
            isinstance(annotations, dict)
            and annotations.get("vnd.docker.reference.type") == "attestation-manifest"
        ):
            attestations.append(node)
    if len(attestations) != 1:
        raise D0Error("probe OCI attachment manifest count differs")
    attestation = attestations[0]
    attestation_descriptor = attestation["descriptor"]
    attestation_digest = attestation_descriptor["digest"]
    annotations = attestation_descriptor.get("annotations")
    if (
        not isinstance(annotations, dict)
        or set(annotations) != {"vnd.docker.reference.digest", "vnd.docker.reference.type"}
        or annotations.get("vnd.docker.reference.type") != "attestation-manifest"
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(annotations.get("vnd.docker.reference.digest", "")),
        )
        is None
        or attestation_descriptor.get("platform") != {"architecture": "unknown", "os": "unknown"}
    ):
        raise D0Error("probe OCI Docker attachment annotations or platform differ")
    parent_edges = [
        edge
        for edge in edges
        if isinstance(edge, dict) and edge.get("child_digest") == attestation_digest
    ]
    if len(parent_edges) != 1 or not isinstance(parent_edges[0].get("parent_digest"), str):
        raise D0Error("probe OCI attachment parent is ambiguous")
    parent_digest = parent_edges[0]["parent_digest"]
    parent = node_by_digest.get(parent_digest)
    if parent is None or parent.get("kind") != "index":
        raise D0Error("probe OCI attachment parent is not an index")
    sibling_edges = [
        edge
        for edge in edges
        if isinstance(edge, dict) and edge.get("parent_digest") == parent_digest
    ]
    sibling_digests = [edge.get("child_digest") for edge in sibling_edges]
    reference_digest = annotations["vnd.docker.reference.digest"]
    image_nodes = [
        node
        for node in manifest_nodes
        if node["descriptor"].get("digest") == reference_digest
        and node["descriptor"].get("annotations") is None
        and node["descriptor"].get("platform") == {"architecture": "arm64", "os": "linux"}
    ]
    if (
        len(sibling_edges) != 2
        or set(sibling_digests) != {attestation_digest, reference_digest}
        or len(image_nodes) != 1
    ):
        raise D0Error("probe OCI attachment image target is cross-index or ambiguous")
    image_parent_edges = [
        edge
        for edge in edges
        if isinstance(edge, dict) and edge.get("child_digest") == reference_digest
    ]
    if len(image_parent_edges) != 1 or image_parent_edges[0].get("parent_digest") != parent_digest:
        raise D0Error("probe OCI attachment image target has multiple parents")
    layer_owners: dict[str, list[str]] = {}
    for node in manifest_nodes:
        owner = node["descriptor"]["digest"]
        layers = node["document"].get("layers")
        if not isinstance(layers, list):
            raise D0Error("probe OCI attachment layer ownership differs")
        for layer in layers:
            if not isinstance(layer, dict) or not isinstance(layer.get("digest"), str):
                raise D0Error("probe OCI attachment layer descriptor differs")
            layer_owners.setdefault(layer["digest"], []).append(owner)
    statement_digests: list[str] = []
    predicate_types: list[str] = []
    for layer in attestation["document"]["layers"]:
        layer_annotations = layer.get("annotations")
        predicate_type = (
            layer_annotations.get("in-toto.io/predicate-type")
            if isinstance(layer_annotations, dict)
            else None
        )
        if (
            layer.get("mediaType") != "application/vnd.in-toto+json"
            or not isinstance(layer_annotations, dict)
            or set(layer_annotations) != {"in-toto.io/predicate-type"}
            or not isinstance(predicate_type, str)
            or layer_owners.get(layer["digest"]) != [attestation_digest]
        ):
            raise D0Error("probe OCI statement layer is not exclusively attached")
        statement_digests.append(layer["digest"])
        predicate_types.append(predicate_type)
    if len(statement_digests) != len(set(statement_digests)):
        raise D0Error("probe OCI attachment statement layers are duplicated")
    return {
        "attestation_manifest_digest": attestation_digest,
        "binding": "docker-index-descriptor",
        "image_manifest_digest": reference_digest,
        "parent_index_digest": parent_digest,
        "predicate_types": predicate_types,
        "statement_layer_digests": statement_digests,
        "status": "pass",
    }


def _flatten_probe_oci(value: bytes) -> tuple[bytes, dict[str, Any]]:
    members = _safe_tar(value)
    graph = _probe_oci_graph(value)
    flat_index = canonical_json(
        {
            "manifests": graph["leaf_descriptors"],
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "schemaVersion": 2,
        }
    )
    files: dict[str, bytes] = {
        "index.json": flat_index,
        "oci-layout": members["oci-layout"],
    }
    for node in graph["nodes"]:
        if node["kind"] != "manifest":
            continue
        files[node["member"]] = members[node["member"]]
        for reference in node["references"]:
            files[reference["member"]] = members[reference["member"]]
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
        for name in sorted(files):
            payload = files[name]
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o444
            info.uid = 0
            info.gid = 0
            info.mtime = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, io.BytesIO(payload))
    return output.getvalue(), graph


def _validate_probe_spdx_predicate(value: Any) -> dict[str, Any]:
    """Validate the exact SPDX document emitted by the pinned Syft scanner."""

    if not isinstance(value, dict):
        raise D0Error("probe SPDX-2.3 predicate is not an object")
    creation = value.get("creationInfo")
    creators = creation.get("creators") if isinstance(creation, dict) else None
    created = creation.get("created") if isinstance(creation, dict) else None
    namespace = value.get("documentNamespace")
    expected_namespace = re.compile(
        r"https://anchore\.com/syft/dir/sbom-"
        r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
    )
    expected_package = {
        "SPDXID": "SPDXRef-DocumentRoot-Directory-sbom",
        "copyrightText": "NOASSERTION",
        "downloadLocation": "NOASSERTION",
        "filesAnalyzed": False,
        "licenseConcluded": "NOASSERTION",
        "licenseDeclared": "NOASSERTION",
        "name": "sbom",
        "primaryPackagePurpose": "FILE",
        "supplier": "NOASSERTION",
    }
    expected_relationship = {
        "relatedSpdxElement": "SPDXRef-DocumentRoot-Directory-sbom",
        "relationshipType": "DESCRIBES",
        "spdxElementId": "SPDXRef-DOCUMENT",
    }
    if (
        set(value)
        != {
            "SPDXID",
            "creationInfo",
            "dataLicense",
            "documentNamespace",
            "name",
            "packages",
            "relationships",
            "spdxVersion",
        }
        or value.get("spdxVersion") != "SPDX-2.3"
        or value.get("dataLicense") != "CC0-1.0"
        or value.get("SPDXID") != "SPDXRef-DOCUMENT"
        or value.get("name") != "sbom"
        or not isinstance(namespace, str)
        or expected_namespace.fullmatch(namespace) is None
        or not isinstance(creation, dict)
        or set(creation) != {"created", "creators", "licenseListVersion"}
        or creators
        != [
            "Organization: Anchore, Inc",
            "Tool: syft-v1.42.3",
            "Tool: buildkit-v0.30.0",
        ]
        or creation.get("licenseListVersion") != "3.28"
        or not isinstance(created, str)
        or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", created) is None
        or value.get("packages") != [expected_package]
        or value.get("relationships") != [expected_relationship]
    ):
        raise D0Error("probe SPDX-2.3 document identity or required structure differs")
    return {
        "content_binding": "spdx-document-root-package",
        "document_name": value["name"],
        "document_namespace": namespace,
        "file_count": 0,
        "files_field_present": False,
        "package_count": 1,
        "probe_checksum_algorithms": [],
        "probe_file_count": 0,
        "relationship_count": 1,
        "spdx_version": value["spdxVersion"],
        "status": "pass",
    }


def _validate_probe_slsa_v1_predicate(
    value: Any,
    *,
    build_finished_unix_ns: int,
    build_started_unix_ns: int,
    image_layer_digest: str,
) -> dict[str, Any]:
    """Validate the complete maximal SLSA v1 shape emitted by pinned BuildKit."""

    if not isinstance(value, dict) or set(value) != {"buildDefinition", "runDetails"}:
        raise D0Error("probe SLSA v1 maximal provenance structure differs")
    definition = value["buildDefinition"]
    details = value["runDetails"]
    if (
        not isinstance(definition, dict)
        or set(definition)
        != {
            "buildType",
            "externalParameters",
            "internalParameters",
            "resolvedDependencies",
        }
        or not isinstance(details, dict)
        or set(details) != {"builder", "metadata"}
        or definition.get("buildType")
        != "https://github.com/moby/buildkit/blob/master/docs/attestations/slsa-definitions.md"
        or details.get("builder") != {"id": ""}
    ):
        raise D0Error("probe SLSA v1 maximal provenance identity differs")
    external = definition["externalParameters"]
    internal = definition["internalParameters"]
    dependencies = definition["resolvedDependencies"]
    if (
        not isinstance(external, dict)
        or set(external) != {"configSource", "request"}
        or not isinstance(internal, dict)
        or set(internal) != {"buildConfig", "builderPlatform", "dockerfileVersion"}
        or internal.get("builderPlatform") != "linux/arm64"
        or internal.get("dockerfileVersion") != "1.24.0"
        or not isinstance(dependencies, list)
        or len(dependencies) != 2
    ):
        raise D0Error("probe SLSA v1 maximal provenance parameters differ")
    config_source = external["configSource"]
    request = external["request"]
    session_uri = config_source.get("uri") if isinstance(config_source, dict) else None
    request_args = {"force-network-mode": "none", "no-cache": ""}
    if (
        not isinstance(config_source, dict)
        or set(config_source) != {"digest", "path", "uri"}
        or config_source.get("digest") != {"sha256": PROBE_ARCHIVE_SHA256}
        or config_source.get("path") != "Dockerfile"
        or not isinstance(session_uri, str)
        or re.fullmatch(r"http://buildkit-session/[a-z0-9]+", session_uri) is None
        or not isinstance(request, dict)
        or set(request) != {"args", "compatibilityVersion", "frontend", "root"}
        or request.get("args") != request_args
        or request.get("compatibilityVersion") != 20
        or request.get("frontend") != "dockerfile.v0"
        or request.get("root") != {"configSource": config_source, "request": {"args": request_args}}
    ):
        raise D0Error("probe SLSA v1 external parameters differ")
    build_config = internal["buildConfig"]
    if (
        not isinstance(build_config, dict)
        or set(build_config) != {"digestMapping", "llbDefinition"}
        or not isinstance(build_config.get("digestMapping"), dict)
        or not build_config["digestMapping"]
        or not isinstance(build_config.get("llbDefinition"), list)
        or not build_config["llbDefinition"]
    ):
        raise D0Error("probe SLSA v1 internal build configuration differs")
    scanner_hex = SCANNER_IMAGE["manifest_digest"].split(":", 1)[1]
    context_dependency = {
        "digest": {"sha256": PROBE_ARCHIVE_SHA256},
        "uri": session_uri,
    }
    scanner_dependencies = [
        item
        for item in dependencies
        if isinstance(item, dict)
        and item.get("digest") == {"sha256": scanner_hex}
        and isinstance(item.get("uri"), str)
        and re.fullmatch(
            r"pkg:docker/localhost%3A[0-9]+/cascadia/buildkit-syft-scanner"
            rf"\?digest=sha256:{scanner_hex}",
            item["uri"],
        )
        is not None
    ]
    if context_dependency not in dependencies or len(scanner_dependencies) != 1:
        raise D0Error("probe SLSA v1 resolved dependencies differ")
    metadata = details["metadata"]
    if (
        not isinstance(metadata, dict)
        or set(metadata)
        != {
            "buildkit_completeness",
            "buildkit_hermetic",
            "buildkit_metadata",
            "finishedOn",
            "invocationId",
            "startedOn",
        }
        or metadata.get("buildkit_completeness") != {"request": True, "resolvedDependencies": True}
        or metadata.get("buildkit_hermetic") is not True
        or not isinstance(metadata.get("invocationId"), str)
        or re.fullmatch(r"[a-z0-9]+", metadata["invocationId"]) is None
    ):
        raise D0Error("probe SLSA v1 run metadata differs")
    slsa_started_unix_ns = _parse_rfc3339_unix_ns(metadata.get("startedOn"))
    slsa_finished_unix_ns = _parse_rfc3339_unix_ns(metadata.get("finishedOn"))
    if (
        not isinstance(build_started_unix_ns, int)
        or isinstance(build_started_unix_ns, bool)
        or not isinstance(build_finished_unix_ns, int)
        or isinstance(build_finished_unix_ns, bool)
        or build_started_unix_ns <= 0
        or build_finished_unix_ns < build_started_unix_ns
        or build_finished_unix_ns - build_started_unix_ns > 900 * 1_000_000_000
        or slsa_started_unix_ns < build_started_unix_ns
        or slsa_finished_unix_ns < slsa_started_unix_ns
        or slsa_finished_unix_ns > build_finished_unix_ns
        or slsa_finished_unix_ns - slsa_started_unix_ns > 900 * 1_000_000_000
    ):
        raise D0Error("probe SLSA v1 run timestamps escape the build envelope")
    buildkit_metadata = metadata["buildkit_metadata"]
    layers = buildkit_metadata.get("layers") if isinstance(buildkit_metadata, dict) else None
    source = buildkit_metadata.get("source") if isinstance(buildkit_metadata, dict) else None
    layer_descriptors = []
    if isinstance(layers, dict):
        for groups in layers.values():
            if isinstance(groups, list):
                for group in groups:
                    if isinstance(group, list):
                        layer_descriptors.extend(item for item in group if isinstance(item, dict))
    infos = source.get("infos") if isinstance(source, dict) else None
    matching_infos = []
    if isinstance(infos, list):
        for info in infos:
            if not isinstance(info, dict):
                continue
            try:
                data = base64.b64decode(info.get("data", ""), validate=True)
            except (ValueError, TypeError):
                continue
            if (
                info.get("filename") == "Dockerfile"
                and info.get("language") == "Dockerfile"
                and data == PROBE_DOCKERFILE
                and isinstance(info.get("digestMapping"), dict)
                and isinstance(info.get("llbDefinition"), list)
            ):
                matching_infos.append(info)
    if (
        not isinstance(buildkit_metadata, dict)
        or set(buildkit_metadata) != {"layers", "source"}
        or not any(item.get("digest") == image_layer_digest for item in layer_descriptors)
        or not isinstance(source, dict)
        or set(source) != {"infos", "locations"}
        or not isinstance(source.get("locations"), dict)
        or len(matching_infos) != 1
    ):
        raise D0Error("probe SLSA v1 BuildKit metadata differs")
    return {
        "build_type": definition["buildType"],
        "builder_id": "",
        "builder_platform": internal["builderPlatform"],
        "context_sha256": PROBE_ARCHIVE_SHA256,
        "dockerfile_version": internal["dockerfileVersion"],
        "hermetic": True,
        "resolved_dependency_count": len(dependencies),
        "run_duration_unix_ns": slsa_finished_unix_ns - slsa_started_unix_ns,
        "run_finished_unix_ns": slsa_finished_unix_ns,
        "run_started_unix_ns": slsa_started_unix_ns,
        "scanner_manifest_digest": SCANNER_IMAGE["manifest_digest"],
        "status": "pass",
        "timestamp_tolerance_unix_ns": 0,
    }


def _parse_rfc3339_unix_ns(value: Any) -> int:
    """Parse RFC3339 with up to nanosecond precision on every supported Python."""

    if not isinstance(value, str):
        raise D0Error("probe SLSA v1 run timestamp is not text")
    matched = re.fullmatch(
        r"([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2})"
        r"(?:\.([0-9]{1,9}))?(Z|[+-][0-9]{2}:[0-9]{2})",
        value,
    )
    if matched is None:
        raise D0Error("probe SLSA v1 run timestamp is not RFC3339")
    base, fraction, offset = matched.groups()
    try:
        parsed = datetime.strptime(base, "%Y-%m-%dT%H:%M:%S")
    except ValueError as error:
        raise D0Error("probe SLSA v1 run timestamp calendar value differs") from error
    if offset == "Z":
        zone = timezone.utc  # noqa: UP017 - John2 system Python predates datetime.UTC.
    else:
        sign = 1 if offset[0] == "+" else -1
        hours = int(offset[1:3])
        minutes = int(offset[4:6])
        if hours > 23 or minutes > 59:
            raise D0Error("probe SLSA v1 run timestamp offset differs")
        zone = timezone(sign * timedelta(hours=hours, minutes=minutes))
    whole_seconds = int(parsed.replace(tzinfo=zone).timestamp())
    fractional_ns = int((fraction or "").ljust(9, "0"))
    return whole_seconds * 1_000_000_000 + fractional_ns


def _verify_flat_probe_oci(
    value: bytes,
    *,
    attachment_contract: Mapping[str, Any] | None = None,
    build_finished_unix_ns: int,
    build_started_unix_ns: int,
) -> dict[str, Any]:
    members = _safe_tar(value)
    if "index.json" not in members or "oci-layout" not in members:
        raise D0Error("probe OCI layout is incomplete")
    try:
        index = json.loads(members["index.json"])
        layout = json.loads(members["oci-layout"])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D0Error("probe OCI index is invalid") from error
    if layout != {"imageLayoutVersion": "1.0.0"}:
        raise D0Error("probe OCI layout version differs")
    manifests = index.get("manifests") if isinstance(index, dict) else None
    if not isinstance(manifests, list) or len(manifests) < 2:
        raise D0Error("probe OCI attestations are absent")
    predicate_types: set[str] = set()
    predicate_receipts: list[dict[str, Any]] = []
    deferred_slsa_v1: list[tuple[int, dict[str, Any]]] = []
    image_manifests = 0
    image_digest: str | None = None
    image_layer_receipt: dict[str, Any] | None = None
    attestation_references: list[str | None] = []
    attestation_subjects: list[dict[str, Any]] = []
    referenced_members = {"index.json", "oci-layout"}
    for descriptor in manifests:
        if not isinstance(descriptor, dict):
            raise D0Error("probe OCI descriptor differs")
        digest = descriptor.get("digest")
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise D0Error("probe OCI descriptor digest differs")
        blob = members.get(f"blobs/sha256/{digest.split(':', 1)[1]}")
        if blob is None or f"sha256:{sha256_bytes(blob)}" != digest:
            raise D0Error("probe OCI manifest blob differs")
        referenced_members.add(f"blobs/sha256/{digest.split(':', 1)[1]}")
        try:
            manifest = json.loads(blob)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise D0Error("probe OCI manifest is invalid") from error
        if descriptor.get("size") != len(blob):
            raise D0Error("probe OCI manifest descriptor size differs")
        config_descriptor = manifest.get("config") if isinstance(manifest, dict) else None
        layers = manifest.get("layers") if isinstance(manifest, dict) else None
        if not isinstance(config_descriptor, dict) or not isinstance(layers, list):
            raise D0Error("probe OCI manifest structure differs")
        referenced = [config_descriptor, *layers]
        for reference in referenced:
            if not isinstance(reference, dict):
                raise D0Error("probe OCI blob descriptor differs")
            reference_digest = reference.get("digest")
            reference_size = reference.get("size")
            if not isinstance(reference_digest, str) or not reference_digest.startswith("sha256:"):
                raise D0Error("probe OCI blob digest differs")
            referenced_blob = members.get(f"blobs/sha256/{reference_digest.split(':', 1)[1]}")
            if (
                referenced_blob is None
                or reference_size != len(referenced_blob)
                or f"sha256:{sha256_bytes(referenced_blob)}" != reference_digest
            ):
                raise D0Error("probe OCI referenced blob differs")
            referenced_members.add(f"blobs/sha256/{reference_digest.split(':', 1)[1]}")
        annotations = descriptor.get("annotations", {})
        if annotations.get("vnd.docker.reference.type") != "attestation-manifest":
            image_manifests += 1
            image_digest = digest
            config_blob = members[f"blobs/sha256/{config_descriptor['digest'].split(':', 1)[1]}"]
            try:
                image_config = json.loads(config_blob)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise D0Error("probe image config is invalid") from error
            labels = image_config.get("config", {}).get("Labels", {})
            if (
                image_config.get("architecture") != "arm64"
                or image_config.get("os") != "linux"
                or labels.get("org.opencontainers.image.title") != "cascadia-r2-d0-buildkit-probe"
                or len(layers) != 1
            ):
                raise D0Error("probe image config or layer count differs")
            layer_descriptor = layers[0]
            layer_payload = members[f"blobs/sha256/{layer_descriptor['digest'].split(':', 1)[1]}"]
            media_type = layer_descriptor.get("mediaType")
            try:
                uncompressed = (
                    gzip.decompress(layer_payload)
                    if media_type == "application/vnd.oci.image.layer.v1.tar+gzip"
                    else layer_payload
                    if media_type == "application/vnd.oci.image.layer.v1.tar"
                    else None
                )
            except (OSError, EOFError) as error:
                raise D0Error("probe image layer cannot be decompressed") from error
            if uncompressed is None:
                raise D0Error("probe image layer media type differs")
            layer_members = _safe_tar(uncompressed)
            if layer_members != {"probe.txt": PROBE_PAYLOAD}:
                raise D0Error("probe image layer path or payload differs")
            diff_id = f"sha256:{sha256_bytes(uncompressed)}"
            if image_config.get("rootfs") != {"type": "layers", "diff_ids": [diff_id]}:
                raise D0Error("probe image config diff ID differs")
            image_layer_receipt = {
                "descriptor_digest": layer_descriptor["digest"],
                "descriptor_size": layer_descriptor["size"],
                "media_type": media_type,
                "diff_id": diff_id,
                "probe_path": "/probe.txt",
                "probe_sha256": PROBE_PAYLOAD_SHA256,
            }
            continue
        attestation_references.append(annotations.get("vnd.docker.reference.digest"))
        for layer in layers:
            predicate = layer.get("annotations", {}).get("in-toto.io/predicate-type")
            if not isinstance(predicate, str):
                raise D0Error("probe attestation predicate type is absent")
            layer_blob = members[f"blobs/sha256/{layer['digest'].split(':', 1)[1]}"]
            try:
                statement = json.loads(layer_blob)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise D0Error("probe attestation statement is invalid") from error
            if (
                not isinstance(statement, dict)
                or statement.get("_type") != "https://in-toto.io/Statement/v0.1"
                or statement.get("predicateType") != predicate
                or not isinstance(statement.get("subject"), list)
                or not isinstance(statement.get("predicate"), dict)
            ):
                raise D0Error("probe attestation statement contract differs")
            predicate_value = statement["predicate"]
            attestation_subjects.append(
                {
                    "manifest_digest": digest,
                    "predicate_type": predicate,
                    "subjects": statement["subject"],
                }
            )
            if predicate == "https://spdx.dev/Document":
                predicate_validation = _validate_probe_spdx_predicate(predicate_value)
            elif predicate == "https://slsa.dev/provenance/v0.2":
                builder = predicate_value.get("builder")
                invocation = predicate_value.get("invocation")
                metadata = predicate_value.get("metadata")
                completeness = metadata.get("completeness") if isinstance(metadata, dict) else None
                if (
                    not isinstance(builder, dict)
                    or not builder.get("id")
                    or not predicate_value.get("buildType")
                    or not isinstance(invocation, dict)
                    or not isinstance(invocation.get("parameters"), dict)
                    or not invocation["parameters"]
                    or not isinstance(invocation.get("environment"), dict)
                    or not invocation["environment"]
                    or not isinstance(metadata, dict)
                    or not isinstance(completeness, dict)
                    or completeness.get("parameters") is not True
                    or completeness.get("environment") is not True
                    or completeness.get("materials") is not True
                    or not isinstance(metadata.get("reproducible"), bool)
                    or not isinstance(predicate_value.get("materials"), list)
                ):
                    raise D0Error("probe SLSA v0.2 maximal provenance is incomplete")
            elif predicate == "https://slsa.dev/provenance/v1":
                predicate_validation = {"status": "deferred"}
            else:
                raise D0Error("probe attestation predicate URI is not frozen")
            predicate_types.add(predicate)
            predicate_receipts.append(
                {
                    "predicate_type": predicate,
                    "predicate_validation": (
                        predicate_validation
                        if predicate
                        in {"https://spdx.dev/Document", "https://slsa.dev/provenance/v1"}
                        else {"status": "pass"}
                    ),
                    "statement_sha256": sha256_bytes(layer_blob),
                    "subject_count": len(statement["subject"]),
                }
            )
            if predicate == "https://slsa.dev/provenance/v1":
                deferred_slsa_v1.append((len(predicate_receipts) - 1, predicate_value))
    if image_manifests != 1 or image_layer_receipt is None:
        raise D0Error("probe image manifest count differs")
    for receipt_position, predicate_value in deferred_slsa_v1:
        predicate_receipts[receipt_position]["predicate_validation"] = (
            _validate_probe_slsa_v1_predicate(
                predicate_value,
                build_finished_unix_ns=build_finished_unix_ns,
                build_started_unix_ns=build_started_unix_ns,
                image_layer_digest=image_layer_receipt["descriptor_digest"],
            )
        )
    if image_digest is None or any(
        reference != image_digest for reference in attestation_references
    ):
        raise D0Error("probe attestation reference digest differs")
    image_hex = image_digest.split(":", 1)[1]
    for statement_binding in attestation_subjects:
        subjects = statement_binding["subjects"]
        if subjects and not any(
            isinstance(subject, dict)
            and isinstance(subject.get("digest"), dict)
            and subject["digest"].get("sha256") == image_hex
            for subject in subjects
        ):
            raise D0Error("probe attestation subject does not bind the image manifest")
        if not subjects and (
            attachment_contract is None
            or attachment_contract.get("status") != "pass"
            or attachment_contract.get("binding") != "docker-index-descriptor"
            or attachment_contract.get("image_manifest_digest") != image_digest
            or attachment_contract.get("attestation_manifest_digest")
            != statement_binding["manifest_digest"]
        ):
            raise D0Error("probe empty attestation subject lacks exact Docker attachment binding")
    if set(members) != referenced_members:
        raise D0Error("probe OCI archive has unreferenced members")
    if "https://spdx.dev/Document" not in predicate_types:
        raise D0Error("probe SPDX SBOM attestation is absent")
    if not predicate_types.intersection(
        {"https://slsa.dev/provenance/v0.2", "https://slsa.dev/provenance/v1"}
    ):
        raise D0Error("probe provenance attestation is absent")
    return {
        "archive_bytes": len(value),
        "archive_sha256": sha256_bytes(value),
        "image_manifest_digest": image_digest,
        "predicate_types": sorted(predicate_types),
        "predicates": sorted(predicate_receipts, key=lambda item: item["predicate_type"]),
        "image_manifest_count": image_manifests,
        "image_layer": image_layer_receipt,
        "attestation_binding": (
            dict(attachment_contract) if attachment_contract is not None else None
        ),
    }


def verify_probe_oci(
    value: bytes,
    *,
    build_finished_unix_ns: int,
    build_started_unix_ns: int,
) -> dict[str, Any]:
    """Verify a complete recursive OCI graph and its policy attestations."""

    flat, graph = _flatten_probe_oci(value)
    attachment_contract = _validate_probe_oci_attachment_contract(graph)
    verified = _verify_flat_probe_oci(
        flat,
        attachment_contract=attachment_contract,
        build_finished_unix_ns=build_finished_unix_ns,
        build_started_unix_ns=build_started_unix_ns,
    )
    graph_nodes = [
        {
            "descriptor": node["descriptor"],
            "kind": node["kind"],
            "member": node["member"],
        }
        for node in graph["nodes"]
    ]
    return {
        **verified,
        "archive_bytes": len(value),
        "archive_sha256": sha256_bytes(value),
        "recursive_attestation_descriptor_count": graph["attestation_descriptor_count"],
        "recursive_graph": {
            "leaf_descriptors": graph["leaf_descriptors"],
            "edges": graph["edges"],
            "nodes": graph_nodes,
            "referenced_members": graph["referenced_members"],
            "root_descriptors": graph["root_descriptors"],
            "status": graph["status"],
            "unreferenced_members": graph["unreferenced_members"],
        },
    }


def _probe_oci_inventory(value: bytes) -> dict[str, Any]:
    """Preserve exact OCI structure without requiring policy attestations."""

    members = _safe_tar(value)
    index_bytes = members.get("index.json")
    layout_bytes = members.get("oci-layout")
    if index_bytes is None or layout_bytes is None:
        raise D0Error("probe OCI inventory lacks index or layout")
    index = _json_command(index_bytes, "probe OCI inventory index")
    layout = _json_command(layout_bytes, "probe OCI inventory layout")
    descriptors = index.get("manifests") if isinstance(index, dict) else None
    if not isinstance(descriptors, list):
        raise D0Error("probe OCI inventory manifest descriptors differ")
    referenced = {"index.json", "oci-layout"}
    manifest_rows: list[dict[str, Any]] = []
    for position, descriptor in enumerate(descriptors):
        if not isinstance(descriptor, dict):
            raise D0Error("probe OCI inventory descriptor is not an object")
        digest = descriptor.get("digest")
        member_name = (
            f"blobs/sha256/{digest.split(':', 1)[1]}"
            if isinstance(digest, str) and digest.startswith("sha256:")
            else None
        )
        manifest_bytes = members.get(member_name) if member_name is not None else None
        manifest: Any = None
        manifest_error: str | None = None
        if manifest_bytes is not None:
            referenced.add(member_name)
            try:
                manifest = json.loads(manifest_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                manifest_error = type(error).__name__
        references: list[dict[str, Any]] = []
        if isinstance(manifest, dict):
            config = manifest.get("config")
            layers = manifest.get("layers")
            layer_descriptors = layers if isinstance(layers, list) else []
            for kind, item in [
                ("config", config),
                *(("layer", layer) for layer in layer_descriptors),
            ]:
                if not isinstance(item, dict):
                    continue
                item_digest = item.get("digest")
                item_name = (
                    f"blobs/sha256/{item_digest.split(':', 1)[1]}"
                    if isinstance(item_digest, str) and item_digest.startswith("sha256:")
                    else None
                )
                payload = members.get(item_name) if item_name is not None else None
                if payload is not None:
                    referenced.add(item_name)
                references.append(
                    {
                        "annotations": item.get("annotations"),
                        "descriptor": item,
                        "digest_matches": (
                            payload is not None and f"sha256:{sha256_bytes(payload)}" == item_digest
                        ),
                        "kind": kind,
                        "member": item_name,
                        "member_sha256": sha256_bytes(payload) if payload is not None else None,
                        "member_size": len(payload) if payload is not None else None,
                        "size_matches": payload is not None and item.get("size") == len(payload),
                    }
                )
        manifest_rows.append(
            {
                "annotations": descriptor.get("annotations"),
                "descriptor": descriptor,
                "digest_matches": (
                    manifest_bytes is not None
                    and f"sha256:{sha256_bytes(manifest_bytes)}" == digest
                ),
                "manifest": manifest,
                "manifest_error": manifest_error,
                "manifest_layers_shape": (
                    "absent"
                    if isinstance(manifest, dict) and "layers" not in manifest
                    else "null"
                    if isinstance(manifest, dict) and manifest.get("layers") is None
                    else "list"
                    if isinstance(manifest, dict) and isinstance(manifest.get("layers"), list)
                    else "other"
                    if isinstance(manifest, dict)
                    else None
                ),
                "member": member_name,
                "member_sha256": (
                    sha256_bytes(manifest_bytes) if manifest_bytes is not None else None
                ),
                "member_size": len(manifest_bytes) if manifest_bytes is not None else None,
                "position": position,
                "references": references,
                "size_matches": (
                    manifest_bytes is not None and descriptor.get("size") == len(manifest_bytes)
                ),
            }
        )
    member_rows = [
        {"path": name, "sha256": sha256_bytes(payload), "size": len(payload)}
        for name, payload in sorted(members.items())
    ]
    inventory = {
        "archive_sha256": sha256_bytes(value),
        "archive_size": len(value),
        "attestation_descriptor_count": sum(
            1
            for item in descriptors
            if isinstance(item, dict)
            and isinstance(item.get("annotations"), dict)
            and item["annotations"].get("vnd.docker.reference.type") == "attestation-manifest"
        ),
        "index": index,
        "index_sha256": sha256_bytes(index_bytes),
        "index_size": len(index_bytes),
        "layout": layout,
        "layout_sha256": sha256_bytes(layout_bytes),
        "manifest_descriptors": manifest_rows,
        "members": member_rows,
        "unreferenced_members": sorted(set(members) - referenced),
    }
    try:
        graph = _probe_oci_graph(value)
    except D0Error as error:
        inventory["recursive_graph"] = {
            "error": str(error),
            "status": "invalid",
        }
    else:
        inventory.update(
            {
                "attestation_descriptor_count": graph["attestation_descriptor_count"],
                "recursive_graph": {
                    "leaf_descriptors": graph["leaf_descriptors"],
                    "nodes": [
                        {
                            "descriptor": node["descriptor"],
                            "kind": node["kind"],
                            "member": node["member"],
                        }
                        for node in graph["nodes"]
                    ],
                    "referenced_members": graph["referenced_members"],
                    "root_descriptors": graph["root_descriptors"],
                    "status": graph["status"],
                    "unreferenced_members": graph["unreferenced_members"],
                },
                "unreferenced_members": graph["unreferenced_members"],
            }
        )
    return inventory


def _probe_oci_attestation_inventory(value: bytes) -> dict[str, Any]:
    """Commit the exact recursive attestation statements before policy parsing."""

    members = _safe_tar(value)
    graph = _probe_oci_graph(value)
    statements: list[dict[str, Any]] = []
    attestation_manifests: list[dict[str, Any]] = []
    statement_total = 0
    for node in graph["nodes"]:
        if node["kind"] != "manifest":
            continue
        descriptor = node["descriptor"]
        annotations = descriptor.get("annotations")
        if (
            not isinstance(annotations, dict)
            or annotations.get("vnd.docker.reference.type") != "attestation-manifest"
        ):
            continue
        manifest_payload = members[node["member"]]
        attestation_manifests.append(
            {
                "descriptor": descriptor,
                "document": node["document"],
                "manifest_raw_base64": base64.b64encode(manifest_payload).decode("ascii"),
                "manifest_sha256": sha256_bytes(manifest_payload),
                "manifest_size": len(manifest_payload),
            }
        )
        for position, layer in enumerate(node["document"]["layers"]):
            digest = layer.get("digest") if isinstance(layer, dict) else None
            member = (
                f"blobs/sha256/{digest.split(':', 1)[1]}"
                if isinstance(digest, str) and digest.startswith("sha256:")
                else None
            )
            payload = members.get(member) if member is not None else None
            if payload is None:
                raise D0Error("probe attestation inventory layer is dangling")
            if len(payload) > 8 * 1024 * 1024:
                raise D0Error("probe attestation inventory statement exceeds bound")
            statement_total += len(payload)
            if statement_total > 16 * 1024 * 1024:
                raise D0Error("probe attestation inventory total exceeds bound")
            decoded: Any = None
            decode_error: str | None = None
            try:
                decoded = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                decode_error = type(error).__name__
            predicate = decoded.get("predicate") if isinstance(decoded, dict) else None
            predicate_bytes = canonical_json(predicate) if isinstance(predicate, dict) else None
            statements.append(
                {
                    "decode_error": decode_error,
                    "descriptor": layer,
                    "manifest_digest": descriptor["digest"],
                    "position": position,
                    "predicate": predicate,
                    "predicate_canonical_sha256": (
                        sha256_bytes(predicate_bytes) if predicate_bytes is not None else None
                    ),
                    "predicate_canonical_size": (
                        len(predicate_bytes) if predicate_bytes is not None else None
                    ),
                    "predicate_type_annotation": (
                        layer.get("annotations", {}).get("in-toto.io/predicate-type")
                        if isinstance(layer, dict) and isinstance(layer.get("annotations"), dict)
                        else None
                    ),
                    "predicate_type_statement": (
                        decoded.get("predicateType") if isinstance(decoded, dict) else None
                    ),
                    "statement": decoded,
                    "statement_raw_base64": base64.b64encode(payload).decode("ascii"),
                    "statement_sha256": sha256_bytes(payload),
                    "statement_size": len(payload),
                    "subject": (decoded.get("subject") if isinstance(decoded, dict) else None),
                }
            )
    if not statements:
        raise D0Error("probe attestation inventory statements are absent")
    return {
        "archive_sha256": sha256_bytes(value),
        "archive_size": len(value),
        "attestation_manifest_count": len(attestation_manifests),
        "attestation_manifests": attestation_manifests,
        "graph": {
            "attestation_descriptor_count": graph["attestation_descriptor_count"],
            "leaf_descriptors": graph["leaf_descriptors"],
            "nodes": [
                {
                    "descriptor": node["descriptor"],
                    "document": node["document"],
                    "kind": node["kind"],
                    "member": node["member"],
                }
                for node in graph["nodes"]
            ],
            "referenced_members": graph["referenced_members"],
            "root_descriptors": graph["root_descriptors"],
            "status": graph["status"],
            "unreferenced_members": graph["unreferenced_members"],
        },
        "statement_count": len(statements),
        "statement_total_size": statement_total,
        "statements": statements,
        "status": "diagnostic-pass",
    }


def _guest(
    runner: CommandRunner,
    argv: Sequence[str],
    *,
    stdin: bytes = b"",
    check: bool = True,
    maximum: int = 64 * 1024 * 1024,
) -> Completed:
    return runner.run(
        [COLIMA, "ssh", "--profile", PROFILE, "--", *argv],
        stdin=stdin,
        check=check,
        maximum=maximum,
        timeout=900,
    )


def _json_command(value: bytes, label: str) -> Any:
    try:
        return json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D0Error(f"{label} is not JSON") from error


def _normalize_network_state(value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_network_state(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _normalize_network_state(item)
            for key, item in sorted(value.items())
            if key not in {"bytes", "handle", "packets"}
        }
    return value


def _normalize_address_identity(value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_address_identity(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _normalize_address_identity(item)
            for key, item in sorted(value.items())
            if key not in {"preferred_life_time", "valid_life_time"}
        }
    return value


def _address_lease_timers(addresses: Any) -> list[dict[str, Any]]:
    if not isinstance(addresses, list):
        raise D0Error("guest address inventory differs")
    timers: list[dict[str, Any]] = []
    for interface in addresses:
        if not isinstance(interface, dict):
            raise D0Error("guest address interface differs")
        ifindex = interface.get("ifindex")
        ifname = interface.get("ifname")
        addr_info = interface.get("addr_info")
        if (
            not isinstance(ifindex, int)
            or isinstance(ifindex, bool)
            or ifindex <= 0
            or not isinstance(ifname, str)
            or not ifname
            or not isinstance(addr_info, list)
        ):
            raise D0Error("guest address interface identity differs")
        for address in addr_info:
            if not isinstance(address, dict):
                raise D0Error("guest address entry differs")
            valid = address.get("valid_life_time")
            preferred = address.get("preferred_life_time")
            if (
                not isinstance(valid, int)
                or isinstance(valid, bool)
                or not isinstance(preferred, int)
                or isinstance(preferred, bool)
                or valid < 0
                or preferred < 0
                or valid > 0xFFFFFFFF
                or preferred > valid
            ):
                raise D0Error("guest address lease timer differs")
            identity = {
                "family": address.get("family"),
                "ifindex": ifindex,
                "ifname": ifname,
                "local": address.get("local"),
                "prefixlen": address.get("prefixlen"),
            }
            if (
                identity["family"] not in {"inet", "inet6"}
                or not isinstance(identity["local"], str)
                or not identity["local"]
                or not isinstance(identity["prefixlen"], int)
                or isinstance(identity["prefixlen"], bool)
            ):
                raise D0Error("guest address lease identity differs")
            timers.append(
                {
                    **identity,
                    "preferred_life_time": preferred,
                    "valid_life_time": valid,
                }
            )
    return sorted(
        timers,
        key=lambda item: (
            item["ifindex"],
            item["family"],
            item["local"],
            item["prefixlen"],
        ),
    )


def _validate_network_lease_transition(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    before_timers = before.get("lease_timers")
    after_timers = after.get("lease_timers")
    before_ns = before.get("captured_monotonic_ns")
    after_ns = after.get("captured_monotonic_ns")
    if (
        not isinstance(before_timers, list)
        or not isinstance(after_timers, list)
        or not isinstance(before_ns, int)
        or isinstance(before_ns, bool)
        or not isinstance(after_ns, int)
        or isinstance(after_ns, bool)
        or after_ns < before_ns
    ):
        raise D0Error("guest network lease transition fields differ")
    identity_fields = ("ifindex", "ifname", "family", "local", "prefixlen")
    before_identities = [tuple(item.get(key) for key in identity_fields) for item in before_timers]
    after_identities = [tuple(item.get(key) for key in identity_fields) for item in after_timers]
    if before_identities != after_identities:
        raise D0Error("guest network lease identity or shape drifted")
    elapsed_seconds = (after_ns - before_ns) / 1_000_000_000
    maximum_decrement = int(elapsed_seconds) + 5
    finite_count = 0
    maximum_observed_decrement = 0
    for index, prior in enumerate(before_timers):
        current = after_timers[index]
        for field in ("preferred_life_time", "valid_life_time"):
            prior_value = prior.get(field)
            current_value = current.get(field)
            if (
                not isinstance(prior_value, int)
                or isinstance(prior_value, bool)
                or not isinstance(current_value, int)
                or isinstance(current_value, bool)
            ):
                raise D0Error("guest network lease timer type drifted")
            if prior_value == 0xFFFFFFFF:
                if current_value != prior_value:
                    raise D0Error("guest infinite network lease timer drifted")
                continue
            finite_count += 1
            decrement = prior_value - current_value
            if current_value <= 0 or decrement < 0:
                raise D0Error("guest finite network lease reset or extension detected")
            if decrement > maximum_decrement:
                raise D0Error("guest finite network lease decrement exceeded elapsed time")
            maximum_observed_decrement = max(maximum_observed_decrement, decrement)
    return {
        "elapsed_milliseconds": (after_ns - before_ns) // 1_000_000,
        "finite_timer_count": finite_count,
        "identity_count": len(before_timers),
        "maximum_allowed_decrement": maximum_decrement,
        "maximum_observed_decrement": maximum_observed_decrement,
        "status": "pass",
    }


def _guest_network_snapshot(runner: CommandRunner) -> dict[str, Any]:
    captured_monotonic_ns = time.monotonic_ns()
    routes = _json_command(
        _guest(runner, ["/usr/sbin/ip", "-j", "route", "show", "table", "all"]).stdout,
        "guest routes",
    )
    addresses = _json_command(
        _guest(runner, ["/usr/sbin/ip", "-j", "address", "show"]).stdout,
        "guest addresses",
    )
    ruleset = _json_command(
        _guest(
            runner,
            ["/usr/bin/sudo", "-n", "/usr/sbin/nft", "-j", "list", "ruleset"],
        ).stdout,
        "guest nftables ruleset",
    )
    normalized = {
        "routes": _normalize_network_state(routes),
        "addresses": _normalize_address_identity(addresses),
        "ruleset": _normalize_network_state(ruleset),
    }
    return {
        "captured_monotonic_ns": captured_monotonic_ns,
        "lease_timers": _address_lease_timers(addresses),
        "state": normalized,
        "state_sha256": sha256_bytes(canonical_json(normalized)),
    }


DOCKER_DEFAULT_BRIDGE_OPTIONS = {
    "com.docker.network.bridge.default_bridge": "true",
    "com.docker.network.bridge.enable_icc": "true",
    "com.docker.network.bridge.enable_ip_masquerade": "true",
    "com.docker.network.bridge.host_binding_ipv4": "0.0.0.0",
    "com.docker.network.bridge.name": "docker0",
    "com.docker.network.driver.mtu": "1500",
}


def _docker_default_bridge_inventory(runner: CommandRunner) -> dict[str, Any]:
    completed = runner.run(
        [DOCKER, "network", "inspect", "bridge"],
        maximum=4 * 1024 * 1024,
        timeout=300,
    )
    value = _json_command(completed.stdout, "Docker default bridge network")
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise D0Error("Docker default bridge network shape differs")
    bridge = value[0]
    ipam = bridge.get("IPAM")
    if (
        bridge.get("Name") != "bridge"
        or bridge.get("Driver") != "bridge"
        or bridge.get("Scope") != "local"
        or bridge.get("Internal") is not False
        or bridge.get("Attachable") is not False
        or bridge.get("Ingress") is not False
        or bridge.get("ConfigOnly") is not False
        or bridge.get("EnableIPv4") is not True
        or bridge.get("EnableIPv6") is not False
        or bridge.get("Containers") != {}
        or bridge.get("Options") != DOCKER_DEFAULT_BRIDGE_OPTIONS
        or not isinstance(bridge.get("Id"), str)
        or re.fullmatch(r"[0-9a-f]{64}", bridge["Id"]) is None
        or not isinstance(ipam, dict)
        or ipam.get("Driver") != "default"
        or ipam.get("Options") is not None
        or ipam.get("Config") != [{"Subnet": "172.17.0.0/16", "Gateway": "172.17.0.1"}]
    ):
        raise D0Error("Docker default bridge ownership or schema differs")
    projection = {
        "attachable": False,
        "containers": {},
        "driver": "bridge",
        "enable_ipv4": True,
        "enable_ipv6": False,
        "id": bridge["Id"],
        "ingress": False,
        "internal": False,
        "ipam": ipam,
        "name": "bridge",
        "options": DOCKER_DEFAULT_BRIDGE_OPTIONS,
        "scope": "local",
    }
    return {
        **projection,
        "inspect_sha256": sha256_bytes(completed.stdout),
        "projection_sha256": sha256_bytes(canonical_json(projection)),
        "status": "pass",
    }


def _eui64_link_local(mac: str) -> str:
    if re.fullmatch(r"[0-9a-f]{2}(?::[0-9a-f]{2}){5}", mac) is None:
        raise D0Error("Docker default bridge MAC differs")
    octets = [int(item, 16) for item in mac.split(":")]
    octets[0] ^= 0x02
    interface = bytes([*octets[:3], 0xFF, 0xFE, *octets[3:]])
    value = int(ipaddress.IPv6Address("fe80::")) | int.from_bytes(interface, "big")
    return str(ipaddress.IPv6Address(value))


def _docker_lazy_network_components(state: Mapping[str, Any]) -> dict[str, Any]:
    addresses = state.get("addresses")
    routes = state.get("routes")
    ruleset = state.get("ruleset")
    if (
        not isinstance(addresses, list)
        or not isinstance(routes, list)
        or not isinstance(ruleset, dict)
        or not isinstance(ruleset.get("nftables"), list)
    ):
        raise D0Error("guest network lifecycle state shape differs")
    docker_interfaces = [
        item for item in addresses if isinstance(item, dict) and item.get("ifname") == "docker0"
    ]
    if len(docker_interfaces) != 1 or not isinstance(docker_interfaces[0].get("addr_info"), list):
        raise D0Error("Docker default bridge interface identity differs")
    mac = docker_interfaces[0].get("address")
    if not isinstance(mac, str):
        raise D0Error("Docker default bridge MAC is absent")
    expected_link_local = _eui64_link_local(mac)
    docker_ipv4 = [
        item
        for item in docker_interfaces[0]["addr_info"]
        if isinstance(item, dict) and item.get("family") == "inet"
    ]
    if docker_ipv4 != [
        {
            "broadcast": "172.17.255.255",
            "family": "inet",
            "label": "docker0",
            "local": "172.17.0.1",
            "prefixlen": 16,
            "scope": "global",
        }
    ]:
        raise D0Error("Docker default bridge IPv4 identity differs")
    docker_link_local = [
        item
        for item in docker_interfaces[0]["addr_info"]
        if isinstance(item, dict) and item.get("family") == "inet6"
    ]
    raw_objects = [
        item
        for item in ruleset["nftables"]
        if isinstance(item, dict)
        and any(
            isinstance(details, dict)
            and details.get("family") == "ip"
            and details.get("table", details.get("name")) == "raw"
            for details in item.values()
        )
    ]
    docker_ipv6_routes = [
        item
        for item in routes
        if isinstance(item, dict)
        and item.get("dev") == "docker0"
        and (
            item.get("dst") in {"fe80::/64", "ff00::/8"}
            or (
                item.get("type") == "local"
                and isinstance(item.get("dst"), str)
                and item["dst"].lower().startswith("fe80:")
            )
        )
    ]
    present = bool(docker_link_local or raw_objects or docker_ipv6_routes)
    if not present:
        return {
            "address": None,
            "mode": "cold",
            "raw_objects": [],
            "routes": [],
        }
    if len(docker_link_local) != 1:
        raise D0Error("Docker lazy IPv6 address cardinality differs")
    address = docker_link_local[0]
    try:
        parsed_address = ipaddress.IPv6Address(address.get("local"))
    except (ipaddress.AddressValueError, ValueError):
        parsed_address = None
    if (
        parsed_address is None
        or not parsed_address.is_link_local
        or str(parsed_address) != expected_link_local
        or address.get("prefixlen") != 64
        or address.get("scope") != "link"
        or set(address) != {"family", "local", "prefixlen", "scope"}
    ):
        raise D0Error("Docker lazy IPv6 address schema differs")
    local = str(parsed_address)
    expected_routes = [
        {
            "dev": "docker0",
            "dst": "fe80::/64",
            "flags": ["linkdown"],
            "metric": 256,
            "pref": "medium",
            "protocol": "kernel",
        },
        {
            "dev": "docker0",
            "dst": local,
            "flags": [],
            "metric": 0,
            "pref": "medium",
            "protocol": "kernel",
            "table": "local",
            "type": "local",
        },
        {
            "dev": "docker0",
            "dst": "ff00::/8",
            "flags": ["linkdown"],
            "metric": 256,
            "pref": "medium",
            "protocol": "kernel",
            "table": "local",
            "type": "multicast",
        },
    ]
    if sorted(docker_ipv6_routes, key=canonical_json) != sorted(
        expected_routes, key=canonical_json
    ):
        raise D0Error("Docker lazy IPv6 route schema differs")
    expected_raw = [
        {"table": {"family": "ip", "name": "raw"}},
        {
            "chain": {
                "family": "ip",
                "hook": "prerouting",
                "name": "PREROUTING",
                "policy": "accept",
                "prio": -300,
                "table": "raw",
                "type": "filter",
            }
        },
    ]
    if sorted(raw_objects, key=canonical_json) != sorted(expected_raw, key=canonical_json):
        raise D0Error("Docker lazy firewall schema differs")
    return {
        "address": address,
        "mode": "warm",
        "raw_objects": raw_objects,
        "routes": docker_ipv6_routes,
    }


def _without_docker_lazy_network(
    snapshot: Mapping[str, Any],
    components: Mapping[str, Any],
) -> dict[str, Any]:
    copied = json.loads(canonical_json(snapshot))
    state = copied["state"]
    if components.get("mode") != "warm":
        return copied
    address = components["address"]
    for interface in state["addresses"]:
        if interface.get("ifname") == "docker0":
            interface["addr_info"].remove(address)
    for route in components["routes"]:
        state["routes"].remove(route)
    for item in components["raw_objects"]:
        state["ruleset"]["nftables"].remove(item)
    local = address["local"]
    copied["lease_timers"] = [
        timer
        for timer in copied["lease_timers"]
        if not (
            timer.get("ifname") == "docker0"
            and timer.get("family") == "inet6"
            and timer.get("local") == local
        )
    ]
    copied["state_sha256"] = sha256_bytes(canonical_json(state))
    return copied


def _validate_docker_network_lifecycle(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    bridge_before: Mapping[str, Any],
    bridge_after: Mapping[str, Any],
    allow_cold_transition: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if bridge_before.get("projection_sha256") != bridge_after.get("projection_sha256"):
        raise D0Error("Docker default bridge identity drifted")
    before_components = _docker_lazy_network_components(before["state"])
    after_components = _docker_lazy_network_components(after["state"])
    if before["state"] == after["state"]:
        if before_components["mode"] == "cold" and allow_cold_transition:
            raise D0Error("cold Docker network did not reach its stable lifecycle")
        lease = _validate_network_lease_transition(before, after)
        return (
            {
                "bridge_projection_sha256": bridge_before["projection_sha256"],
                "initial_mode": before_components["mode"],
                "status": "exact-restoration",
                "transition": "none",
            },
            lease,
        )
    if (
        not allow_cold_transition
        or before_components["mode"] != "cold"
        or after_components["mode"] != "warm"
    ):
        raise D0Error("unauthorized Docker network lifecycle transition")
    new_address = after_components["address"]
    matching_timers = [
        timer
        for timer in after.get("lease_timers", [])
        if timer.get("ifname") == "docker0"
        and timer.get("family") == "inet6"
        and timer.get("local") == new_address["local"]
        and timer.get("prefixlen") == 64
    ]
    if (
        len(matching_timers) != 1
        or matching_timers[0].get("preferred_life_time") != 0xFFFFFFFF
        or matching_timers[0].get("valid_life_time") != 0xFFFFFFFF
    ):
        raise D0Error("Docker lazy IPv6 lease lifetime differs")
    stripped_after = _without_docker_lazy_network(after, after_components)
    if stripped_after["state"] != before["state"]:
        raise D0Error("Docker lazy lifecycle included unauthorized network changes")
    lease = _validate_network_lease_transition(before, stripped_after)
    return (
        {
            "bridge_projection_sha256": bridge_before["projection_sha256"],
            "docker0_link_local": after_components["address"],
            "initial_mode": "cold",
            "raw_prerouting": after_components["raw_objects"],
            "status": "validated-lazy-initialization",
            "transition": "cold-to-warm",
        },
        lease,
    )


def _network_boundary_snapshot(
    runner: CommandRunner,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    before = _guest_network_snapshot(runner)
    bridge = _docker_default_bridge_inventory(runner)
    after = _guest_network_snapshot(runner)
    if before["state"] != after["state"]:
        raise D0Error("Docker bridge inspection mutated guest network state")
    lease = _validate_network_lease_transition(before, after)
    return after, bridge, lease


def _bounded_state_differences(
    before: Any,
    after: Any,
    *,
    path: str = "$",
    limit: int = 256,
) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []

    def visit(left: Any, right: Any, selected: str) -> None:
        if len(differences) >= limit or left == right:
            return
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                child = f"{selected}.{key}"
                if key not in left:
                    differences.append({"after": right[key], "path": child, "status": "added"})
                elif key not in right:
                    differences.append({"before": left[key], "path": child, "status": "removed"})
                else:
                    visit(left[key], right[key], child)
                if len(differences) >= limit:
                    return
            return
        if isinstance(left, list) and isinstance(right, list):
            for index in range(max(len(left), len(right))):
                child = f"{selected}[{index}]"
                if index >= len(left):
                    differences.append({"after": right[index], "path": child, "status": "added"})
                elif index >= len(right):
                    differences.append({"before": left[index], "path": child, "status": "removed"})
                else:
                    visit(left[index], right[index], child)
                if len(differences) >= limit:
                    return
            return
        differences.append({"after": right, "before": left, "path": selected, "status": "changed"})

    visit(before, after, path)
    return differences


def guest_network_stability_probe(
    runner: CommandRunner,
    *,
    interval_seconds: float = 2.0,
) -> dict[str, Any]:
    if interval_seconds <= 0 or interval_seconds > 5:
        raise D0Error("guest network stability interval differs")
    before = _guest_network_snapshot(runner)
    time.sleep(interval_seconds)
    after = _guest_network_snapshot(runner)
    differences = _bounded_state_differences(before["state"], after["state"])
    lease_transition = _validate_network_lease_transition(before, after)
    return {
        "after": after,
        "before": before,
        "difference_count": len(differences),
        "differences": differences,
        "interval_milliseconds": int(interval_seconds * 1000),
        "lease_transition": lease_transition,
        "status": "stable" if not differences else "changed",
    }


def _validate_guest_buildkit_state(value: Any) -> dict[str, Any]:
    required = {
        "schema_id",
        "root",
        "present",
        "entries",
        "content_blobs",
        "payload_paths",
        "build_cache_records",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise D0Error("guest BuildKit state fields differ")
    if (
        value["schema_id"] != "cascadia.r2-map.d0-buildkit-state.v2"
        or value["root"] != "/var/lib/docker/buildkit"
        or not isinstance(value["present"], bool)
    ):
        raise D0Error("guest BuildKit state identity differs")
    entries = value["entries"]
    if not isinstance(entries, list):
        raise D0Error("guest BuildKit entry inventory is absent")
    paths: list[str] = []
    for item in entries:
        if not isinstance(item, dict) or item.get("type") not in {"directory", "file"}:
            raise D0Error("guest BuildKit entry identity differs")
        expected_keys = {"path", "type", "size", "mode"}
        if item["type"] == "file":
            expected_keys.add("sha256")
        if set(item) != expected_keys:
            raise D0Error("guest BuildKit entry fields differ")
        path = item["path"]
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or ".." in PurePosixPath(path).parts
            or not isinstance(item["size"], int)
            or isinstance(item["size"], bool)
            or item["size"] < 0
            or not isinstance(item["mode"], str)
            or re.fullmatch(r"[0-7]{4}", item["mode"]) is None
            or (
                item["type"] == "file"
                and (
                    not isinstance(item["sha256"], str)
                    or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
                )
            )
        ):
            raise D0Error("guest BuildKit entry metadata differs")
        paths.append(path)
    if paths != sorted(set(paths)) or (not value["present"] and paths):
        raise D0Error("guest BuildKit entry ordering/presence differs")
    path_set = set(paths)
    for key in ("content_blobs", "payload_paths"):
        selected = value[key]
        if (
            not isinstance(selected, list)
            or selected != sorted(set(selected))
            or any(not isinstance(path, str) or path not in path_set for path in selected)
        ):
            raise D0Error(f"guest BuildKit {key} inventory differs")
    records = value["build_cache_records"]
    if not isinstance(records, list):
        raise D0Error("guest BuildKit semantic cache inventory is absent")
    encoded_records: list[bytes] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "id",
            "type",
            "size",
            "in_use",
            "shared",
            "parents",
        }:
            raise D0Error("guest BuildKit semantic cache record fields differ")
        if (
            not isinstance(record["id"], str)
            or not record["id"]
            or not isinstance(record["type"], str)
            or not record["type"]
            or not isinstance(record["size"], int)
            or isinstance(record["size"], bool)
            or record["size"] < 0
            or not isinstance(record["in_use"], bool)
            or not isinstance(record["shared"], bool)
            or not isinstance(record["parents"], list)
            or record["parents"] != sorted(set(record["parents"]))
            or any(not isinstance(parent, str) or not parent for parent in record["parents"])
        ):
            raise D0Error("guest BuildKit semantic cache record differs")
        encoded_records.append(canonical_json(record))
    if encoded_records != sorted(set(encoded_records)):
        raise D0Error("guest BuildKit semantic cache records are not sorted and unique")
    value["script_sha256"] = GUEST_BUILDKIT_STATE_SCRIPT_SHA256
    value["state_sha256"] = sha256_bytes(canonical_json(value))
    return value


def _guest_buildkit_state(runner: CommandRunner) -> dict[str, Any]:
    output = _guest(
        runner,
        ["/usr/bin/python3", "-I", "-S", "-B", "-c", _GUEST_BUILDKIT_STATE_SCRIPT],
        maximum=256 * 1024 * 1024,
    ).stdout
    return _validate_guest_buildkit_state(_json_command(output, "guest BuildKit state"))


def _require_empty_buildkit_state(
    state: Mapping[str, Any],
    *,
    buildx_disk_usage: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    if (
        state.get("build_cache_records") != []
        or state.get("content_blobs") != []
        or state.get("payload_paths") != []
        or (buildx_disk_usage is not None and list(buildx_disk_usage))
    ):
        raise D0Error("BuildKit semantic cache or payload state is not empty")


def _engine_object_inventory(runner: CommandRunner) -> dict[str, Any]:
    images = (
        runner.run(
            [DOCKER, "image", "ls", "--all", "--no-trunc", "--quiet"], maximum=8 * 1024 * 1024
        )
        .stdout.decode()
        .splitlines()
    )
    containers = (
        runner.run(
            [DOCKER, "container", "ls", "--all", "--no-trunc", "--quiet"],
            maximum=8 * 1024 * 1024,
        )
        .stdout.decode()
        .splitlines()
    )
    if images or containers:
        raise D0Error("BuildKit probe requires an empty Engine image/container ledger")
    return {"images": [], "containers": [], "status": "pass"}


def _daemon_accounting_inventory(runner: CommandRunner) -> dict[str, Any]:
    info = _json_command(
        runner.run(
            [DOCKER, "info", "--format", "{{json .}}"],
            maximum=8 * 1024 * 1024,
        ).stdout,
        "Docker daemon accounting info",
    )
    if not isinstance(info, dict):
        raise D0Error("Docker daemon accounting info differs")
    output = runner.run(
        [DOCKER, "system", "df", "--format", "{{json .}}"],
        maximum=8 * 1024 * 1024,
    ).stdout
    rows: dict[str, dict[str, Any]] = {}
    for line in output.splitlines():
        row = _json_command(line, "Docker daemon accounting row")
        if not isinstance(row, dict) or not isinstance(row.get("Type"), str):
            raise D0Error("Docker daemon accounting row differs")
        rows[row["Type"]] = row
    expected_types = {"Images", "Containers", "Local Volumes", "Build Cache"}
    if set(rows) != expected_types:
        raise D0Error("Docker daemon accounting categories differ")

    def count(name: str) -> int:
        raw = rows[name].get("TotalCount")
        if isinstance(raw, int) and not isinstance(raw, bool):
            value = raw
        elif isinstance(raw, str) and raw.isdigit():
            value = int(raw)
        else:
            raise D0Error("Docker daemon accounting count differs")
        if value < 0:
            raise D0Error("Docker daemon accounting count differs")
        return value

    result = {
        "build_cache": count("Build Cache"),
        "containers": count("Containers"),
        "images": count("Images"),
        "info_containers": info.get("Containers"),
        "info_images": info.get("Images"),
        "volumes": count("Local Volumes"),
    }
    if any(
        not isinstance(result[key], int) or isinstance(result[key], bool) or result[key] < 0
        for key in result
    ):
        raise D0Error("Docker daemon accounting summary differs")
    return result


def _require_empty_daemon_accounting(value: Mapping[str, Any]) -> None:
    expected = {
        "build_cache": 0,
        "containers": 0,
        "images": 0,
        "info_containers": 0,
        "info_images": 0,
        "volumes": 0,
    }
    if dict(value) != expected:
        raise D0Error("Docker daemon accounting baseline is not empty")


def _image_store_rows(runner: CommandRunner) -> list[dict[str, Any]]:
    output = runner.run(
        [DOCKER, "image", "ls", "--all", "--no-trunc", "--format", "{{json .}}"],
        maximum=8 * 1024 * 1024,
    ).stdout
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        value = _json_command(line, "Docker image-store row")
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("Repository"), str)
            or not isinstance(value.get("Tag"), str)
            or not isinstance(value.get("ID"), str)
        ):
            raise D0Error("Docker image-store row identity differs")
        rows.append(value)
    return rows


def _require_single_logical_image(
    rows: Sequence[Mapping[str, Any]],
    *,
    repository: str,
    tag: str,
    image_id: str,
) -> dict[str, Any]:
    logical = {(row.get("Repository"), row.get("Tag"), row.get("ID")) for row in rows}
    if not rows or logical != {(repository, tag, image_id)}:
        raise D0Error("Docker logical image-store identity differs")
    return {
        "logical_image_count": 1,
        "cli_row_count": len(rows),
        "duplicate_cli_rows_normalized": len(rows) - 1,
        "repository": repository,
        "tag": tag,
        "image_id": image_id,
    }


def _require_exact_logical_image_references(
    rows: Sequence[Mapping[str, Any]],
    *,
    references: Sequence[str],
    image_id: str,
) -> dict[str, Any]:
    expected: set[tuple[str, str, str]] = set()
    for reference in references:
        if reference.count(":") != 1:
            raise D0Error("Docker expected image reference differs")
        repository, tag = reference.rsplit(":", 1)
        if not repository or not tag:
            raise D0Error("Docker expected image reference differs")
        expected.add((repository, tag, image_id))
    logical = {(row.get("Repository"), row.get("Tag"), row.get("ID")) for row in rows}
    if not rows or logical != expected:
        raise D0Error("Docker logical image-store reference set differs")
    return {
        "logical_image_count": 1,
        "logical_reference_count": len(expected),
        "cli_row_count": len(rows),
        "duplicate_cli_rows_normalized": len(rows) - len(expected),
        "references": sorted(references),
        "image_id": image_id,
    }


def _scanner_local_generator_reference(verified: Mapping[str, Any]) -> str:
    digest = verified.get("manifest_digest")
    if not isinstance(digest, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        raise D0Error("BuildKit scanner manifest digest differs")
    return f"{SCANNER_LOCAL_REFERENCE}@{digest}"


def _scanner_registry_descriptor(verified: Mapping[str, Any]) -> dict[str, Any]:
    digest = verified.get("manifest_digest")
    if not isinstance(digest, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        raise D0Error("BuildKit scanner manifest digest differs")
    for prefix in ("manifest", "config", "layer"):
        candidate_digest = verified.get(f"{prefix}_digest")
        candidate_size = verified.get(f"{prefix}_size")
        if (
            not isinstance(candidate_digest, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", candidate_digest) is None
            or not isinstance(candidate_size, int)
            or isinstance(candidate_size, bool)
            or candidate_size <= 0
        ):
            raise D0Error("BuildKit scanner registry input schema differs")
    # Bind explicitly to IPv4 loopback while resolving the pinned TLS endpoint
    # through the certificate's reserved localhost DNS identity.
    endpoint = f"{SCANNER_REGISTRY_RESOLVER_HOST}:{SCANNER_REGISTRY_PORT}"
    return {
        "host": SCANNER_REGISTRY_HOST,
        "resolver_host": SCANNER_REGISTRY_RESOLVER_HOST,
        "port": SCANNER_REGISTRY_PORT,
        "repository": SCANNER_REGISTRY_REPOSITORY,
        "root": SCANNER_REGISTRY_ROOT,
        "generator_reference": f"{endpoint}/{SCANNER_REGISTRY_REPOSITORY}@{digest}",
        "manifest_digest": digest,
        "config_digest": verified["config_digest"],
        "layer_digest": verified["layer_digest"],
    }


def _validate_scanner_registry_cleanup(
    value: Any,
    descriptor: Mapping[str, Any],
    *,
    require_requests: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("root_absent") is not True:
        raise D0Error("BuildKit scanner registry cleanup differs")
    if value.get("status") == "absent" and set(value) == {"root_absent", "status"}:
        if require_requests:
            raise D0Error("BuildKit scanner registry request proof is absent")
        return value
    expected_keys = {
        "host",
        "listener_absent",
        "orphan_pids",
        "pid",
        "port",
        "record_count",
        "record_sha256",
        "requests_valid",
        "root_absent",
        "served_paths",
        "status",
        "unexpected_requests",
    }
    if set(value) != expected_keys:
        raise D0Error("BuildKit scanner registry cleanup fields differ")
    pid = value["pid"]
    pid_valid = (isinstance(pid, int) and not isinstance(pid, bool) and pid > 1) or (
        not require_requests and pid is None
    )
    if (
        value["host"] != descriptor["host"]
        or value["port"] != descriptor["port"]
        or value["status"] != "clean"
        or value["listener_absent"] is not True
        or value["orphan_pids"] != []
        or not pid_valid
        or not isinstance(value["record_count"], int)
        or isinstance(value["record_count"], bool)
        or value["record_count"] < 0
        or not isinstance(value["served_paths"], list)
        or not isinstance(value["unexpected_requests"], list)
        or re.fullmatch(r"[0-9a-f]{64}", value["record_sha256"] or "") is None
    ):
        raise D0Error("BuildKit scanner registry cleanup identity differs")
    if pid is None and (
        value["record_count"] != 0
        or value["requests_valid"] is not False
        or value["served_paths"] != []
        or value["unexpected_requests"] != []
    ):
        raise D0Error("BuildKit scanner pre-launch cleanup identity differs")
    if require_requests and (
        value["requests_valid"] is not True
        or value["unexpected_requests"] != []
        or value["record_count"] < 1
    ):
        raise D0Error("BuildKit scanner registry request proof differs")
    return value


def _cleanup_scanner_registry(
    runner: CommandRunner,
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    completed = _guest(
        runner.cleanup_runner(),
        [
            "/usr/bin/sudo",
            "-n",
            "/usr/bin/python3",
            "-I",
            "-S",
            "-B",
            "-c",
            SCANNER_REGISTRY_CLEANUP_SCRIPT,
            str(descriptor["root"]),
            str(descriptor["host"]),
            str(descriptor["port"]),
            str(descriptor["repository"]),
            str(descriptor["manifest_digest"]),
            str(descriptor["config_digest"]),
            str(descriptor["layer_digest"]),
        ],
        maximum=4 * 1024 * 1024,
    )
    value = _json_command(completed.stdout, "BuildKit scanner registry cleanup")
    return _validate_scanner_registry_cleanup(
        value,
        descriptor,
        require_requests=False,
    )


def _cleanup_scanner_registry_trust(
    runner: CommandRunner,
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    completed = _guest(
        runner.cleanup_runner(),
        [
            "/usr/bin/sudo",
            "-n",
            "/usr/bin/python3",
            "-I",
            "-S",
            "-B",
            "-c",
            SCANNER_REGISTRY_TRUST_CLEANUP_SCRIPT,
            str(descriptor["root"]),
            SCANNER_REGISTRY_CA_PATH,
        ],
        maximum=4 * 1024 * 1024,
    )
    value = _json_command(completed.stdout, "BuildKit scanner trust cleanup")
    expected = {
        "baseline_sha256",
        "ca_path",
        "ca_path_absent",
        "restored_sha256",
        "update_stderr_sha256",
        "update_stdout_sha256",
        "status",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or value.get("status") not in {"absent", "restored"}
        or value.get("ca_path") != SCANNER_REGISTRY_CA_PATH
        or value.get("ca_path_absent") is not True
        or value.get("baseline_sha256") != value.get("restored_sha256")
    ):
        raise D0Error("BuildKit scanner trust cleanup differs")
    for field in (
        "baseline_sha256",
        "restored_sha256",
        "update_stderr_sha256",
        "update_stdout_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", value.get(field) or "") is None:
            raise D0Error("BuildKit scanner trust cleanup identity differs")
    return value


def _start_socket_sampler(
    runner: CommandRunner,
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    completed = _guest(
        runner,
        [
            "/usr/bin/sudo",
            "-n",
            "/usr/bin/python3",
            "-I",
            "-S",
            "-B",
            "-c",
            SCANNER_SOCKET_SAMPLER_LAUNCH_SCRIPT,
            SCANNER_SOCKET_SAMPLER_PROCESS_MARKER,
            str(descriptor["root"]),
        ],
        maximum=128 * 1024,
    )
    value = _json_command(completed.stdout, "BuildKit socket sampler launch")
    if (
        not isinstance(value, dict)
        or set(value) != {"pid", "status"}
        or value.get("status") != "launched"
        or not isinstance(value.get("pid"), int)
        or isinstance(value.get("pid"), bool)
        or value.get("pid", 0) <= 1
    ):
        raise D0Error("BuildKit socket sampler launch differs")
    return value


def _stop_socket_sampler(
    runner: CommandRunner,
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    completed = _guest(
        runner.cleanup_runner(),
        [
            "/usr/bin/sudo",
            "-n",
            "/usr/bin/python3",
            "-I",
            "-S",
            "-B",
            "-c",
            SCANNER_SOCKET_SAMPLER_STOP_SCRIPT,
            SCANNER_SOCKET_SAMPLER_PROCESS_MARKER,
            str(descriptor["root"]),
        ],
        maximum=8 * 1024 * 1024,
    )
    value = _json_command(completed.stdout, "BuildKit socket sampler stop")
    if (
        not isinstance(value, dict)
        or set(value) != {"pid", "records", "sample_count", "status"}
        or value.get("status") != "stopped"
        or not isinstance(value.get("pid"), int)
        or isinstance(value.get("pid"), bool)
        or value.get("pid", 0) <= 1
        or not isinstance(value.get("sample_count"), int)
        or isinstance(value.get("sample_count"), bool)
        or value.get("sample_count", 0) <= 0
        or not isinstance(value.get("records"), list)
        or len(value["records"]) > 1024
    ):
        raise D0Error("BuildKit socket sampler result differs")
    for record in value["records"]:
        if (
            not isinstance(record, dict)
            or set(record) != {"line", "observed_unix_ns"}
            or not isinstance(record.get("line"), str)
            or len(record.get("line", "")) > 4096
            or not isinstance(record.get("observed_unix_ns"), int)
            or isinstance(record.get("observed_unix_ns"), bool)
            or record.get("observed_unix_ns", 0) <= 0
        ):
            raise D0Error("BuildKit socket sampler record differs")
    return value


def _prepare_scanner_registry(
    runner: CommandRunner,
    verified: Mapping[str, Any],
) -> dict[str, Any]:
    descriptor = _scanner_registry_descriptor(verified)
    root = str(descriptor["root"])
    created = False
    trust_installed = False
    try:
        _guest(
            runner,
            [
                "/usr/bin/sudo",
                "-n",
                "/usr/bin/python3",
                "-I",
                "-S",
                "-B",
                "-c",
                (
                    "import os,stat,sys; p=sys.argv[1]; "
                    "os.mkdir(p,0o700); s=os.lstat(p); "
                    "assert stat.S_ISDIR(s.st_mode) and s.st_uid==0 "
                    "and stat.S_IMODE(s.st_mode)==0o700"
                ),
                root,
            ],
        )
        created = True
        export = _guest(
            runner,
            [
                "/usr/bin/sudo",
                "-n",
                "/usr/bin/ctr",
                "--namespace",
                "moby",
                "images",
                "export",
                "--platform",
                "linux/arm64",
                f"{root}/scanner.oci.tar",
                SCANNER_LOCAL_REFERENCE,
            ],
            maximum=8 * 1024 * 1024,
        )
        prepared = _guest(
            runner,
            [
                "/usr/bin/sudo",
                "-n",
                "/usr/bin/python3",
                "-I",
                "-S",
                "-B",
                "-c",
                SCANNER_REGISTRY_PREPARE_SCRIPT,
                root,
                str(verified["manifest_digest"]),
                str(verified["manifest_size"]),
                str(verified["config_digest"]),
                str(verified["config_size"]),
                str(verified["layer_digest"]),
                str(verified["layer_size"]),
            ],
            maximum=4 * 1024 * 1024,
        )
        preparation = _json_command(prepared.stdout, "BuildKit scanner registry preparation")
        if not isinstance(preparation, dict) or preparation.get("status") != "prepared":
            raise D0Error("BuildKit scanner registry preparation differs")
        trust_payload = canonical_json(
            {
                "ca_cert": SCANNER_REGISTRY_CA_CERT.decode("ascii"),
                "server_cert": SCANNER_REGISTRY_SERVER_CERT.decode("ascii"),
                "server_key": SCANNER_REGISTRY_SERVER_KEY.decode("ascii"),
            }
        )
        trust_installed = True
        installed = _guest(
            runner,
            [
                "/usr/bin/sudo",
                "-n",
                "/usr/bin/python3",
                "-I",
                "-S",
                "-B",
                "-c",
                SCANNER_REGISTRY_TRUST_INSTALL_SCRIPT,
                root,
                SCANNER_REGISTRY_CA_PATH,
                SCANNER_REGISTRY_CA_SHA256,
                SCANNER_REGISTRY_SERVER_CERT_SHA256,
                SCANNER_REGISTRY_SERVER_KEY_SHA256,
            ],
            stdin=trust_payload,
            maximum=4 * 1024 * 1024,
        )
        trust = _json_command(installed.stdout, "BuildKit scanner trust installation")
        if (
            not isinstance(trust, dict)
            or trust.get("status") != "installed"
            or trust.get("ca_path") != SCANNER_REGISTRY_CA_PATH
            or trust.get("ca_sha256") != SCANNER_REGISTRY_CA_SHA256
            or trust.get("server_cert_sha256") != SCANNER_REGISTRY_SERVER_CERT_SHA256
            or trust.get("server_key_sha256") != SCANNER_REGISTRY_SERVER_KEY_SHA256
        ):
            raise D0Error("BuildKit scanner trust installation differs")
        launched = _guest(
            runner,
            [
                "/usr/bin/sudo",
                "-n",
                "/usr/bin/python3",
                "-I",
                "-S",
                "-B",
                "-c",
                SCANNER_REGISTRY_SERVER_SCRIPT,
                SCANNER_REGISTRY_PROCESS_MARKER,
                root,
                str(descriptor["host"]),
                str(descriptor["port"]),
                str(descriptor["repository"]),
                str(descriptor["manifest_digest"]),
                str(descriptor["config_digest"]),
                str(descriptor["layer_digest"]),
            ],
            maximum=128 * 1024,
        )
        launch = _json_command(launched.stdout, "BuildKit scanner registry launch")
        if (
            not isinstance(launch, dict)
            or launch.get("status") != "launched"
            or not isinstance(launch.get("pid"), int)
        ):
            raise D0Error("BuildKit scanner registry launch differs")
        ready = _wait_guest_control_document(
            runner,
            f"{root}/server-ready.json",
            timeout_seconds=10,
        )
        if ready != {
            "host": descriptor["host"],
            "pid": launch["pid"],
            "port": descriptor["port"],
            "repository": descriptor["repository"],
            "status": "ready",
        }:
            raise D0Error("BuildKit scanner registry readiness differs")
        handshake_command = _guest(
            runner,
            [
                "/usr/bin/python3",
                "-I",
                "-S",
                "-B",
                "-c",
                SCANNER_REGISTRY_TLS_CLIENT_SCRIPT,
                str(descriptor["resolver_host"]),
                str(descriptor["port"]),
                SCANNER_REGISTRY_SERVER_CERT_DER_SHA256,
            ],
            maximum=128 * 1024,
        )
        handshake = _json_command(
            handshake_command.stdout,
            "BuildKit scanner registry TLS handshake",
        )
        if not isinstance(handshake, dict) or handshake.get("status") != "pass":
            raise D0Error("BuildKit scanner registry TLS handshake differs")
        return {
            **descriptor,
            "ctr_export_stdout_sha256": sha256_bytes(export.stdout),
            "ctr_export_stderr_sha256": sha256_bytes(export.stderr),
            "provenance": {
                "source_reference": verified["reference"],
                "export_reference": SCANNER_LOCAL_REFERENCE,
                "served_manifest_digest": verified["manifest_digest"],
                "generator_reference": descriptor["generator_reference"],
            },
            "preparation": preparation,
            "trust": trust,
            "launch": launch,
            "ready": ready,
            "tls_handshake": handshake,
            "status": "ready",
        }
    except BaseException as error:
        if created:
            trust_cleanup: dict[str, Any] | None = None
            trust_error: BaseException | None = None
            if trust_installed:
                try:
                    trust_cleanup = _cleanup_scanner_registry_trust(runner, descriptor)
                except BaseException as cleanup_error:
                    trust_error = cleanup_error
            try:
                cleanup = _cleanup_scanner_registry(runner, descriptor)
            except BaseException as cleanup_error:
                cleanup_error.__cause__ = error
                raise cleanup_error
            if trust_error is not None:
                raise trust_error from error
            trust_cleanup_projection = (
                canonical_json(trust_cleanup).decode("ascii") if trust_cleanup else "null"
            )
            raise D0Error(
                "BuildKit scanner registry preparation failed: "
                f"primary={str(error)!r} "
                f"trust_cleanup={trust_cleanup_projection} "
                f"cleanup={canonical_json(cleanup).decode('ascii')}"
            ) from error
        raise


def _load_scanner_image(runner: CommandRunner, archive: bytes) -> dict[str, Any]:
    verified = verify_scanner_oci_archive(archive)
    reference = verified["reference"]
    loaded = runner.run(
        [DOCKER, "image", "load"],
        stdin=archive,
        timeout=600,
        maximum=64 * 1024 * 1024,
    )
    if reference.encode() not in loaded.stdout and reference.encode() not in loaded.stderr:
        raise D0Error("BuildKit scanner load did not bind the frozen reference")
    tagged = runner.run(
        [
            DOCKER,
            "image",
            "tag",
            verified["manifest_digest"],
            SCANNER_LOCAL_REFERENCE,
        ]
    )
    if tagged.stdout or tagged.stderr:
        raise D0Error("BuildKit scanner local tag command emitted output")
    logical_store = _require_exact_logical_image_references(
        _image_store_rows(runner),
        references=(reference, SCANNER_LOCAL_REFERENCE),
        image_id=verified["manifest_digest"],
    )
    try:
        inspected = json.loads(
            runner.run([DOCKER, "image", "inspect", verified["manifest_digest"]]).stdout
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D0Error("BuildKit scanner image inspect is invalid") from error
    if not isinstance(inspected, list) or len(inspected) != 1:
        raise D0Error("BuildKit scanner image inspect cardinality differs")
    image = inspected[0]
    if (
        image.get("Id") != verified["manifest_digest"]
        or image.get("Architecture") != "arm64"
        or image.get("Os") != "linux"
        or sorted(image.get("RepoTags") or []) != sorted([reference, SCANNER_LOCAL_REFERENCE])
        or image.get("Config", {}).get("Entrypoint") != ["/bin/syft-scanner"]
        or image.get("RootFS") != {"Type": "layers", "Layers": [verified["diff_id"]]}
    ):
        raise D0Error("BuildKit scanner loaded image identity differs")
    return {
        "archive": verified,
        "image_id": image["Id"],
        "reference": reference,
        "local_reference": SCANNER_LOCAL_REFERENCE,
        "generator_reference": _scanner_local_generator_reference(verified),
        "load_stdout_sha256": sha256_bytes(loaded.stdout),
        "load_stderr_sha256": sha256_bytes(loaded.stderr),
        "tag_stdout_sha256": sha256_bytes(tagged.stdout),
        "tag_stderr_sha256": sha256_bytes(tagged.stderr),
        "image_store": logical_store,
        "status": "pass",
    }


def _scanner_attestation_residue_identity(
    verified: Mapping[str, Any],
) -> tuple[str, tuple[str, ...]]:
    reference = f"moby-dangling@{verified['attestation_manifest_digest']}"
    digests = tuple(
        sorted(
            str(verified[key])
            for key in (
                "attestation_config_digest",
                "attestation_manifest_digest",
                "provenance_digest",
                "spdx_digest",
            )
        )
    )
    if re.fullmatch(r"moby-dangling@sha256:[0-9a-f]{64}", reference) is None or any(
        re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None for digest in digests
    ):
        raise D0Error("scanner attestation residue identity differs")
    return reference, digests


def _cleanup_scanner_attestation_residue(
    runner: CommandRunner,
    verified: Mapping[str, Any],
) -> dict[str, Any]:
    reference, digests = _scanner_attestation_residue_identity(verified)
    completed = _guest(
        runner,
        [
            "/usr/bin/sudo",
            "-n",
            "/usr/bin/python3",
            "-I",
            "-S",
            "-B",
            "-c",
            SCANNER_ATTESTATION_CLEANUP_SCRIPT,
            reference,
            *digests,
        ],
        maximum=128 * 1024,
    )
    value = _json_command(
        completed.stdout,
        "scanner attestation residue cleanup",
    )
    if isinstance(value, dict) and value.get("status") == "failed":
        raise D0Error(
            "scanner attestation residue cleanup failed with stage evidence: "
            f"{canonical_json(value).decode('ascii')}"
        )
    expected = {
        "commands": value.get("commands") if isinstance(value, dict) else None,
        "content_after": [],
        "content_before": list(digests),
        "content_remove_stdout_sha256": value.get("content_remove_stdout_sha256")
        if isinstance(value, dict)
        else None,
        "image_remove_stdout_sha256": value.get("image_remove_stdout_sha256")
        if isinstance(value, dict)
        else None,
        "images_after": [],
        "images_before": [reference],
        "mutation_started": True,
        "stage": "complete",
        "status": "clean",
    }
    commands = value.get("commands") if isinstance(value, dict) else None
    commands_valid = isinstance(commands, list) and bool(commands)
    if commands_valid:
        for command in commands:
            if (
                not isinstance(command, dict)
                or set(command)
                != {
                    "argv",
                    "returncode",
                    "stage",
                    "stderr_sha256",
                    "stderr_size",
                    "stdout_sha256",
                    "stdout_size",
                }
                or not isinstance(command.get("argv"), list)
                or command["argv"][:3] != ["/usr/bin/ctr", "--namespace", "moby"]
                or command.get("returncode") != 0
                or not isinstance(command.get("stage"), str)
                or not command["stage"]
                or command.get("stderr_size") != 0
                or not isinstance(command.get("stdout_size"), int)
                or isinstance(command.get("stdout_size"), bool)
                or not 0 <= command["stdout_size"] <= 1024 * 1024
                or any(
                    re.fullmatch(r"[0-9a-f]{64}", command.get(field) or "") is None
                    for field in ("stderr_sha256", "stdout_sha256")
                )
            ):
                commands_valid = False
                break
    if (
        value != expected
        or not commands_valid
        or any(
            re.fullmatch(r"[0-9a-f]{64}", value.get(field) or "") is None
            for field in (
                "content_remove_stdout_sha256",
                "image_remove_stdout_sha256",
            )
        )
    ):
        raise D0Error("scanner attestation residue cleanup receipt differs")
    return value


def _buildx_disk_usage(runner: CommandRunner) -> list[dict[str, Any]]:
    output = runner.run(
        [DOCKER, "buildx", "du", "--builder", "default", "--format", "{{json .}}"],
        maximum=64 * 1024 * 1024,
    ).stdout
    records: list[dict[str, Any]] = []
    for line in output.splitlines():
        value = _json_command(line, "BuildKit disk-usage record")
        if not isinstance(value, dict):
            raise D0Error("BuildKit disk-usage record is not an object")
        records.append(value)
    return records


def _buildkit_egress_program(
    table: str,
    flow: Mapping[str, Any],
    *,
    trace: bool = False,
) -> bytes:
    if re.fullmatch(r"[a-z][a-z0-9_]{0,31}", table) is None:
        raise D0Error("D0 nftables table name differs")
    family = flow.get("family")
    client_address = flow.get("client_address")
    client_port = flow.get("client_port")
    server_address = flow.get("server_address")
    if (
        family not in {"ip", "ip6"}
        or not isinstance(client_address, str)
        or not isinstance(server_address, str)
        or not isinstance(client_port, int)
        or isinstance(client_port, bool)
        or not 1 <= client_port <= 65535
    ):
        raise D0Error("D0 SSH management flow differs")
    lines = [
        f"add table inet {table}",
    ]
    if trace:
        for name, nft_type in (
            ("tcp4", "ifname . ipv4_addr . inet_service . ipv4_addr . inet_service"),
            ("udp4", "ifname . ipv4_addr . inet_service . ipv4_addr . inet_service"),
            ("other4", "ifname . ipv4_addr . ipv4_addr . inet_proto"),
            ("tcp6", "ifname . ipv6_addr . inet_service . ipv6_addr . inet_service"),
            ("udp6", "ifname . ipv6_addr . inet_service . ipv6_addr . inet_service"),
            ("other6", "ifname . ipv6_addr . ipv6_addr . inet_proto"),
        ):
            lines.append(
                f"add set inet {table} {name} {{ type {nft_type}; flags dynamic; counter; }}"
            )
    lines.extend(
        [
            f"add chain inet {table} output {{ type filter hook output priority 0; policy drop; }}",
            f'add rule inet {table} output oifname "lo" accept comment "cascadia-d0-loopback"',
            f"add rule inet {table} output tcp sport 22 ct state established accept "
            'comment "cascadia-d0-ssh-reply"',
        ]
    )
    if trace:
        lines.extend(
            [
                f"add rule inet {table} output meta nfproto ipv4 meta l4proto tcp "
                "update @tcp4 { meta oifname . ip saddr . tcp sport . "
                "ip daddr . tcp dport counter } "
                'comment "cascadia-d0-trace-tcp4"',
                f"add rule inet {table} output meta nfproto ipv4 meta l4proto udp "
                "update @udp4 { meta oifname . ip saddr . udp sport . "
                "ip daddr . udp dport counter } "
                'comment "cascadia-d0-trace-udp4"',
                f"add rule inet {table} output meta nfproto ipv4 meta l4proto != tcp "
                "meta l4proto != udp update @other4 "
                "{ meta oifname . ip saddr . ip daddr . "
                "meta l4proto counter } "
                'comment "cascadia-d0-trace-other4"',
                f"add rule inet {table} output meta nfproto ipv6 meta l4proto tcp "
                "update @tcp6 { meta oifname . ip6 saddr . tcp sport . "
                "ip6 daddr . tcp dport counter } "
                'comment "cascadia-d0-trace-tcp6"',
                f"add rule inet {table} output meta nfproto ipv6 meta l4proto udp "
                "update @udp6 { meta oifname . ip6 saddr . udp sport . "
                "ip6 daddr . udp dport counter } "
                'comment "cascadia-d0-trace-udp6"',
                f"add rule inet {table} output meta nfproto ipv6 meta l4proto != tcp "
                "meta l4proto != udp update @other6 "
                "{ meta oifname . ip6 saddr . ip6 daddr . "
                "meta l4proto counter } "
                'comment "cascadia-d0-trace-other6"',
            ]
        )
    lines.append(f'add rule inet {table} output counter reject comment "cascadia-d0-reject"')
    return ("\n".join(lines) + "\n").encode("ascii")


NFT_GUARD_INSTALL_SCRIPT = r"""import hashlib,ipaddress,json,os,subprocess,sys
table,state,launcher,trace_value=sys.argv[1:5]
if trace_value not in {'0','1'}: raise SystemExit(80)
trace=trace_value=='1'
parts=os.environ.get('SSH_CONNECTION','').split()
if len(parts)!=4: raise SystemExit(81)
client,client_port,server,server_port=parts
try:
 client_ip=ipaddress.ip_address(client); server_ip=ipaddress.ip_address(server)
 client_port=int(client_port); server_port=int(server_port)
except ValueError: raise SystemExit(82)
if client_ip.version!=server_ip.version or server_port!=22 or not 1<=client_port<=65535:
 raise SystemExit(83)
rows=subprocess.run(['/usr/bin/sudo','-n','/usr/bin/ss','-Hnt','state','established','sport','=',':22'],capture_output=True,text=True,check=True).stdout.splitlines()
expected=f'{server}:22 {client}:{client_port}'
normalized=[' '.join(row.split()[-2:]) for row in rows if row.strip()]
if normalized!=[expected]: raise SystemExit(84)
family='ip' if client_ip.version==4 else 'ip6'
lines=[f'add table inet {table}']
if trace:
 for name,nft_type in (
  ('tcp4','ifname . ipv4_addr . inet_service . ipv4_addr . inet_service'),
  ('udp4','ifname . ipv4_addr . inet_service . ipv4_addr . inet_service'),
  ('other4','ifname . ipv4_addr . ipv4_addr . inet_proto'),
  ('tcp6','ifname . ipv6_addr . inet_service . ipv6_addr . inet_service'),
  ('udp6','ifname . ipv6_addr . inet_service . ipv6_addr . inet_service'),
  ('other6','ifname . ipv6_addr . ipv6_addr . inet_proto'),
 ):
  lines.append(f'add set inet {table} {name} {{ type {nft_type}; flags dynamic; counter; }}')
lines.extend([
 f'add chain inet {table} output {{ type filter hook output priority 0; policy drop; }}',
 f'add rule inet {table} output oifname "lo" accept comment "cascadia-d0-loopback"',
 (f'add rule inet {table} output tcp sport 22 ct state established accept '
  'comment "cascadia-d0-ssh-reply"'),
])
if trace:
 lines.extend([
  (f'add rule inet {table} output meta nfproto ipv4 meta l4proto tcp '
   'update @tcp4 { meta oifname . ip saddr . tcp sport . '
   'ip daddr . tcp dport counter } '
   'comment "cascadia-d0-trace-tcp4"'),
  (f'add rule inet {table} output meta nfproto ipv4 meta l4proto udp '
   'update @udp4 { meta oifname . ip saddr . udp sport . '
   'ip daddr . udp dport counter } '
   'comment "cascadia-d0-trace-udp4"'),
  (f'add rule inet {table} output meta nfproto ipv4 meta l4proto != tcp '
   'meta l4proto != udp update @other4 '
   '{ meta oifname . ip saddr . ip daddr . '
   'meta l4proto counter } '
   'comment "cascadia-d0-trace-other4"'),
  (f'add rule inet {table} output meta nfproto ipv6 meta l4proto tcp '
   'update @tcp6 { meta oifname . ip6 saddr . tcp sport . '
   'ip6 daddr . tcp dport counter } '
   'comment "cascadia-d0-trace-tcp6"'),
  (f'add rule inet {table} output meta nfproto ipv6 meta l4proto udp '
   'update @udp6 { meta oifname . ip6 saddr . udp sport . '
   'ip6 daddr . udp dport counter } '
   'comment "cascadia-d0-trace-udp6"'),
  (f'add rule inet {table} output meta nfproto ipv6 meta l4proto != tcp '
   'meta l4proto != udp update @other6 '
   '{ meta oifname . ip6 saddr . ip6 daddr . '
   'meta l4proto counter } '
   'comment "cascadia-d0-trace-other6"'),
 ])
lines.append(f'add rule inet {table} output counter reject comment "cascadia-d0-reject"')
program='\n'.join(lines)+'\n'
launched=subprocess.run(['/usr/bin/sudo','-n','/usr/bin/python3','-I','-S','-B','-c',launcher,state,table,'30'],check=True,capture_output=True,text=True)
if launched.stderr or len(launched.stdout)>4096: raise SystemExit(85)
try: failsafe=json.loads(launched.stdout)
except json.JSONDecodeError: raise SystemExit(86)
fields={'deadline_unix_ms','pid','state_path','status','table'}
valid=isinstance(failsafe,dict) and set(failsafe)==fields
valid=valid and failsafe.get('table')==table and failsafe.get('state_path')==state
valid=valid and failsafe.get('status')=='armed' and isinstance(failsafe.get('pid'),int)
valid=valid and not isinstance(failsafe.get('pid'),bool) and failsafe.get('pid',0)>1
valid=valid and isinstance(failsafe.get('deadline_unix_ms'),int)
valid=valid and not isinstance(failsafe.get('deadline_unix_ms'),bool)
if not valid: raise SystemExit(87)
subprocess.run(['/usr/bin/sudo','-n','/usr/sbin/nft','-f','-'],input=program.encode('ascii'),check=True)
result={'client_address':client,'client_port':client_port,'family':family,'failsafe':failsafe,'program_sha256':hashlib.sha256(program.encode('ascii')).hexdigest(),'server_address':server,'server_port':22,'status':'installed'}
sys.stdout.write(json.dumps(result,sort_keys=True,separators=(',',':')))
"""


NFT_FAILSAFE_LAUNCH_SCRIPT = r"""import json,os,stat,subprocess,sys,time
state,table,seconds=sys.argv[1:4]
try: os.lstat(state); raise SystemExit(91)
except FileNotFoundError: pass
deleter=('import os,subprocess,sys,time; p,t,s=sys.argv[1:4]; time.sleep(int(s)); '
 'subprocess.run(["/usr/sbin/nft","delete","table","inet",t],check=False); '
 'm=p+".fired"; f=os.open(m,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600); '
 'os.write(f,b"fired\\n"); os.fsync(f); os.close(f); '
 'os.unlink(p) if os.path.exists(p) else None')
child=subprocess.Popen(['/usr/bin/python3','-I','-S','-B','-c',deleter,state,table,seconds],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
value={'deadline_unix_ms':int((time.time()+int(seconds))*1000),'pid':child.pid,'state_path':state,'status':'armed','table':table}
fd=os.open(state,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
os.write(fd,json.dumps(value,sort_keys=True,separators=(',',':')).encode())
os.fsync(fd); os.close(fd)
sys.stdout.write(json.dumps(value,sort_keys=True,separators=(',',':')))
"""


NFT_FAILSAFE_CANCEL_SCRIPT = r"""import json,os,signal,stat,sys,time
state,table=sys.argv[1:3]
st=os.lstat(state)
valid=stat.S_ISREG(st.st_mode) and st.st_uid==0 and stat.S_IMODE(st.st_mode)==0o600
if not valid: raise SystemExit(101)
with open(state,encoding='ascii') as stream: value=json.load(stream)
valid=value.get('table')==table and value.get('status')=='armed'
valid=valid and isinstance(value.get('pid'),int)
if not valid: raise SystemExit(102)
pid=value['pid']
try: os.kill(pid,signal.SIGTERM)
except ProcessLookupError: raise SystemExit(103)
for _ in range(100):
 try: os.kill(pid,0)
 except ProcessLookupError: break
 time.sleep(.01)
else: os.kill(pid,signal.SIGKILL)
os.unlink(state)
if os.path.exists(state+'.fired'): raise SystemExit(104)
sys.stdout.write(json.dumps({'failsafe_pid':pid,'state_absent':True,'status':'cancelled','table':table},sort_keys=True,separators=(',',':')))
"""


NFT_FAILSAFE_CLEANUP_SCRIPT = r"""import hashlib,json,os,signal,stat,sys,time
state,table=sys.argv[1:3]
marker=state+'.fired'; state_removed=False; marker_removed=False; pid=None
if os.path.lexists(state):
 st=os.lstat(state)
 valid=stat.S_ISREG(st.st_mode) and st.st_uid==0
 valid=valid and stat.S_IMODE(st.st_mode)==0o600 and st.st_size<=4096
 if not valid: raise SystemExit(111)
 fd=os.open(state,os.O_RDONLY|os.O_NOFOLLOW)
 try: raw=os.read(fd,4097)
 finally: os.close(fd)
 if len(raw)>4096: raise SystemExit(112)
 try: value=json.loads(raw)
 except json.JSONDecodeError: raise SystemExit(113)
 fields={'deadline_unix_ms','pid','state_path','status','table'}
 valid=isinstance(value,dict) and set(value)==fields
 valid=valid and value.get('table')==table and value.get('state_path')==state
 valid=valid and value.get('status')=='armed' and isinstance(value.get('pid'),int)
 valid=valid and not isinstance(value.get('pid'),bool) and value.get('pid',0)>1
 if not valid: raise SystemExit(114)
 pid=value['pid']; proc=f'/proc/{pid}'
 if os.path.isdir(proc):
  with open(proc+'/cmdline','rb') as stream: cmdline=stream.read(16385)
  valid=len(cmdline)<=16384 and state.encode() in cmdline and table.encode() in cmdline
  if not valid: raise SystemExit(115)
  try: os.kill(pid,signal.SIGTERM)
  except ProcessLookupError: pass
  for _ in range(100):
   if not os.path.isdir(proc): break
   with open(proc+'/stat',encoding='ascii') as stream: fields=stream.read(4096).split()
   if len(fields)>=3 and fields[2]=='Z': break
   time.sleep(.01)
  else:
   os.kill(pid,signal.SIGKILL)
 os.unlink(state); state_removed=True
if os.path.lexists(marker):
 st=os.lstat(marker)
 valid=stat.S_ISREG(st.st_mode) and st.st_uid==0
 valid=valid and stat.S_IMODE(st.st_mode)==0o600 and st.st_size==6
 if not valid: raise SystemExit(116)
 fd=os.open(marker,os.O_RDONLY|os.O_NOFOLLOW)
 try: raw=os.read(fd,7)
 finally: os.close(fd)
 expected=bytes.fromhex('66697265640a')
 expected_sha='c46699865b6d7c32e964b9feaaf93ff06d3ad8f4455432137837fa9bd894c85a'
 if raw!=expected or hashlib.sha256(raw).hexdigest()!=expected_sha: raise SystemExit(117)
 os.unlink(marker); marker_removed=True
directory=os.open(os.path.dirname(state),os.O_RDONLY|os.O_DIRECTORY)
os.fsync(directory); os.close(directory)
if os.path.lexists(state) or os.path.lexists(marker): raise SystemExit(118)
sys.stdout.write(json.dumps({'failsafe_pid':pid,'marker_absent':True,'marker_removed':marker_removed,'state_absent':True,'state_removed':state_removed,'status':'clean','table':table},sort_keys=True,separators=(',',':')))
"""


def _nft_install_argv(program: bytes) -> list[str]:
    """Keep nft input guest-local instead of relying on Colima stdin forwarding."""

    try:
        text = program.decode("ascii")
    except UnicodeDecodeError as error:
        raise D0Error("D0 nftables program is not ASCII") from error
    if not text.endswith("\n") or "\0" in text or len(program) > 4096:
        raise D0Error("D0 nftables program transport differs")
    return [
        "/bin/sh",
        "-c",
        'printf "%s" "$1" | /usr/bin/sudo -n /usr/sbin/nft -f -',
        "cascadia-r2-d0-nft",
        text,
    ]


def _validate_failsafe_launch_receipt(value: Any, *, table: str, state_path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "deadline_unix_ms",
        "pid",
        "state_path",
        "status",
        "table",
    }:
        raise D0Error("D0 egress-guard failsafe receipt fields differ")
    if (
        value["table"] != table
        or value["state_path"] != state_path
        or value["status"] != "armed"
        or not isinstance(value["pid"], int)
        or isinstance(value["pid"], bool)
        or value["pid"] <= 1
        or not isinstance(value["deadline_unix_ms"], int)
        or isinstance(value["deadline_unix_ms"], bool)
        or value["deadline_unix_ms"] <= 0
    ):
        raise D0Error("D0 egress-guard failsafe receipt differs")
    return value


def _install_buildkit_egress_guard(
    runner: CommandRunner,
    *,
    table: str,
    state_path: str,
    trace: bool = False,
) -> dict[str, Any]:
    completed = _guest(
        runner,
        [
            "/usr/bin/python3",
            "-I",
            "-S",
            "-B",
            "-c",
            NFT_GUARD_INSTALL_SCRIPT,
            table,
            state_path,
            NFT_FAILSAFE_LAUNCH_SCRIPT,
            "1" if trace else "0",
        ],
        maximum=128 * 1024,
    )
    value = _json_command(completed.stdout, "D0 egress-guard installer receipt")
    if not isinstance(value, dict) or value.get("status") != "installed":
        raise D0Error("D0 egress-guard installer receipt differs")
    expected = _buildkit_egress_program(table, value, trace=trace)
    if value.get("program_sha256") != sha256_bytes(expected):
        raise D0Error("D0 egress-guard installer program differs")
    failsafe = _validate_failsafe_launch_receipt(
        value.get("failsafe"), table=table, state_path=state_path
    )
    if value["failsafe"] != failsafe:
        raise D0Error("D0 egress-guard failsafe receipt differs")
    return value


def _cancel_buildkit_egress_failsafe(
    runner: CommandRunner,
    *,
    table: str,
    state_path: str,
) -> dict[str, Any]:
    completed = _guest(
        runner,
        [
            "/usr/bin/sudo",
            "-n",
            "/usr/bin/python3",
            "-I",
            "-S",
            "-B",
            "-c",
            NFT_FAILSAFE_CANCEL_SCRIPT,
            state_path,
            table,
        ],
        maximum=128 * 1024,
    )
    value = _json_command(completed.stdout, "D0 egress-guard failsafe cancellation")
    if value != {
        "failsafe_pid": value.get("failsafe_pid") if isinstance(value, dict) else None,
        "state_absent": True,
        "status": "cancelled",
        "table": table,
    } or not isinstance(value.get("failsafe_pid"), int):
        raise D0Error("D0 egress-guard failsafe cancellation differs")
    return value


def _cleanup_buildkit_egress_failsafe(
    runner: CommandRunner,
    *,
    table: str,
    state_path: str,
) -> dict[str, Any]:
    """Cancel an armed failsafe or remove its exact root-owned fired marker."""

    completed = _guest(
        runner,
        [
            "/usr/bin/sudo",
            "-n",
            "/usr/bin/python3",
            "-I",
            "-S",
            "-B",
            "-c",
            NFT_FAILSAFE_CLEANUP_SCRIPT,
            state_path,
            table,
        ],
        maximum=128 * 1024,
    )
    value = _json_command(completed.stdout, "D0 egress-guard failsafe cleanup")
    expected = {
        "failsafe_pid": value.get("failsafe_pid") if isinstance(value, dict) else None,
        "marker_absent": True,
        "marker_removed": value.get("marker_removed") if isinstance(value, dict) else None,
        "state_absent": True,
        "state_removed": value.get("state_removed") if isinstance(value, dict) else None,
        "status": "clean",
        "table": table,
    }
    if (
        value != expected
        or not isinstance(value.get("marker_removed"), bool)
        or not isinstance(value.get("state_removed"), bool)
        or (
            value.get("failsafe_pid") is not None
            and (
                not isinstance(value.get("failsafe_pid"), int)
                or isinstance(value.get("failsafe_pid"), bool)
            )
        )
    ):
        raise D0Error("D0 egress-guard failsafe cleanup differs")
    return value


def _egress_control_descriptor(run_id: str) -> dict[str, Any]:
    token = sha256_bytes(run_id.encode("ascii"))[:8]
    octet = 1 + int(token[:2], 16) % 253
    return {
        "namespace": f"cascadia-r2-d0-{token}",
        "host_interface": f"c2d0h{token}",
        "peer_interface": f"c2d0p{token}",
        "host_address": f"198.18.{octet}.1",
        "peer_address": f"198.18.{octet}.2",
        "prefix": 30,
        "port": 43022,
        "state_directory": f"/run/cascadia-r2-d0-egress-{token}",
    }


def _wait_guest_control_document(
    runner: CommandRunner,
    path: str,
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    script = (
        "import json,os,stat,sys,time; p=sys.argv[1]; deadline=time.monotonic()+int(sys.argv[2]); "
        "value=None; "
        "\nwhile time.monotonic()<deadline:"
        "\n try:"
        "\n  fd=os.open(p,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)); st=os.fstat(fd)"
        "\n  valid=stat.S_ISREG(st.st_mode) and st.st_uid==0 "
        "and stat.S_IMODE(st.st_mode)==0o600"
        "\n  if not valid: raise SystemExit(61)"
        "\n  value=os.read(fd,65537); os.close(fd)"
        "\n  if len(value)>65536: raise SystemExit(62)"
        "\n  break"
        "\n except FileNotFoundError: time.sleep(.02)"
        "\nif value is None: raise SystemExit(63)"
        "\nsys.stdout.buffer.write(value)"
    )
    completed = _guest(
        runner,
        [
            "/usr/bin/sudo",
            "-n",
            "/usr/bin/python3",
            "-I",
            "-S",
            "-B",
            "-c",
            script,
            path,
            str(timeout_seconds),
        ],
        maximum=128 * 1024,
    )
    value = _json_command(completed.stdout, "pre-existing egress control document")
    if not isinstance(value, dict):
        raise D0Error("pre-existing egress control document is not an object")
    return value


def _cleanup_preexisting_egress_control(
    runner: CommandRunner,
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    cleanup = runner.cleanup_runner()
    state = descriptor["state_directory"]
    script = EGRESS_CONTROL_PROCESS_CLEANUP_SCRIPT
    python_cleanup = _guest(
        cleanup,
        [
            "/usr/bin/sudo",
            "-n",
            "/usr/bin/python3",
            "-I",
            "-S",
            "-B",
            "-c",
            script,
            state,
        ],
        check=False,
    )
    link_cleanup = _guest(
        cleanup,
        [
            "/usr/bin/sudo",
            "-n",
            "/usr/sbin/ip",
            "link",
            "delete",
            descriptor["host_interface"],
        ],
        check=False,
    )
    namespace_cleanup = _guest(
        cleanup,
        [
            "/usr/bin/sudo",
            "-n",
            "/usr/sbin/ip",
            "netns",
            "delete",
            descriptor["namespace"],
        ],
        check=False,
    )
    namespace_list = _guest(cleanup, ["/usr/bin/sudo", "-n", "/usr/sbin/ip", "netns", "list"])
    link_check = _guest(
        cleanup,
        ["/usr/bin/sudo", "-n", "/usr/sbin/ip", "link", "show", descriptor["host_interface"]],
        check=False,
    )
    state_check = _guest(
        cleanup,
        ["/usr/bin/sudo", "-n", "/usr/bin/test", "!", "-e", state],
        check=False,
    )
    namespaces = {line.split()[0] for line in namespace_list.stdout.decode().splitlines() if line}
    if (
        python_cleanup.returncode != 0
        or descriptor["namespace"] in namespaces
        or link_check.returncode == 0
        or state_check.returncode != 0
    ):
        raise D0Error("pre-existing egress control cleanup was incomplete")
    process_cleanup = _json_command(
        python_cleanup.stdout,
        "pre-existing egress control process cleanup",
    )
    if (
        not isinstance(process_cleanup, dict)
        or process_cleanup.get("status") not in {"absent", "clean"}
        or not isinstance(process_cleanup.get("processes"), list)
        or not isinstance(process_cleanup.get("state_removed"), bool)
    ):
        raise D0Error("pre-existing egress control process cleanup differs")
    return {
        "python_cleanup_returncode": python_cleanup.returncode,
        "link_cleanup_returncode": link_cleanup.returncode,
        "namespace_cleanup_returncode": namespace_cleanup.returncode,
        "namespace_absent": True,
        "process_cleanup": process_cleanup,
        "host_interface_absent": True,
        "state_directory_absent": True,
        "status": "pass",
    }


def _wait_preexisting_egress_socket_absent(
    runner: CommandRunner,
    control: Mapping[str, Any],
) -> dict[str, Any]:
    completed = _guest(
        runner,
        [
            "/usr/bin/python3",
            "-I",
            "-S",
            "-B",
            "-c",
            EGRESS_CONTROL_SOCKET_QUIESCENCE_SCRIPT,
            str(control["host_address"]),
            str(control["peer_address"]),
            str(control["port"]),
            "10",
        ],
        maximum=128 * 1024,
    )
    value = _json_command(
        completed.stdout,
        "pre-existing egress socket quiescence",
    )
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "last_matches",
            "last_output_sha256",
            "packet_capable_matches",
            "samples",
            "status",
        }
        or value.get("status") != "absent"
        or value.get("last_matches") != []
        or value.get("packet_capable_matches") != []
        or not isinstance(value.get("last_matches"), list)
        or re.fullmatch(r"[0-9a-f]{64}", value.get("last_output_sha256") or "") is None
        or not isinstance(value.get("samples"), int)
        or isinstance(value.get("samples"), bool)
        or value.get("samples", 0) <= 0
    ):
        raise D0Error("pre-existing egress socket did not quiesce")
    return value


def _prepare_preexisting_egress_control(
    runner: CommandRunner,
    *,
    run_id: str,
) -> dict[str, Any]:
    descriptor = _egress_control_descriptor(run_id)
    namespace_list = _guest(runner, ["/usr/bin/sudo", "-n", "/usr/sbin/ip", "netns", "list"])
    namespaces = {line.split()[0] for line in namespace_list.stdout.decode().splitlines() if line}
    link = _guest(
        runner,
        ["/usr/bin/sudo", "-n", "/usr/sbin/ip", "link", "show", descriptor["host_interface"]],
        check=False,
    )
    state = _guest(
        runner,
        [
            "/usr/bin/sudo",
            "-n",
            "/usr/bin/test",
            "!",
            "-e",
            descriptor["state_directory"],
        ],
        check=False,
    )
    if descriptor["namespace"] in namespaces or link.returncode == 0 or state.returncode != 0:
        raise D0Error("pre-existing egress control namespace is not initially absent")
    commands = [
        ["/usr/bin/sudo", "-n", "/usr/sbin/ip", "netns", "add", descriptor["namespace"]],
        [
            "/usr/bin/sudo",
            "-n",
            "/usr/sbin/ip",
            "link",
            "add",
            descriptor["host_interface"],
            "type",
            "veth",
            "peer",
            "name",
            descriptor["peer_interface"],
        ],
        [
            "/usr/bin/sudo",
            "-n",
            "/usr/sbin/ip",
            "link",
            "set",
            descriptor["peer_interface"],
            "netns",
            descriptor["namespace"],
        ],
        [
            "/usr/bin/sudo",
            "-n",
            "/usr/sbin/ip",
            "address",
            "add",
            f"{descriptor['host_address']}/{descriptor['prefix']}",
            "dev",
            descriptor["host_interface"],
        ],
        [
            "/usr/bin/sudo",
            "-n",
            "/usr/sbin/ip",
            "link",
            "set",
            descriptor["host_interface"],
            "up",
        ],
        [
            "/usr/bin/sudo",
            "-n",
            "/usr/sbin/ip",
            "netns",
            "exec",
            descriptor["namespace"],
            "/usr/sbin/ip",
            "link",
            "set",
            "lo",
            "up",
        ],
        [
            "/usr/bin/sudo",
            "-n",
            "/usr/sbin/ip",
            "netns",
            "exec",
            descriptor["namespace"],
            "/usr/sbin/ip",
            "address",
            "add",
            f"{descriptor['peer_address']}/{descriptor['prefix']}",
            "dev",
            descriptor["peer_interface"],
        ],
        [
            "/usr/bin/sudo",
            "-n",
            "/usr/sbin/ip",
            "netns",
            "exec",
            descriptor["namespace"],
            "/usr/sbin/ip",
            "link",
            "set",
            descriptor["peer_interface"],
            "up",
        ],
        [
            "/usr/bin/sudo",
            "-n",
            "/usr/bin/python3",
            "-I",
            "-S",
            "-B",
            "-c",
            (
                "import os,sys; p=sys.argv[1]; os.mkdir(p,0o700); "
                "s=os.lstat(p); "
                "assert s.st_uid==0 and (s.st_mode&0o777)==0o700"
            ),
            descriptor["state_directory"],
        ],
    ]
    try:
        for command in commands:
            _guest(runner, command)
        _guest(
            runner,
            [
                "/usr/bin/sudo",
                "-n",
                "/usr/sbin/ip",
                "netns",
                "exec",
                descriptor["namespace"],
                "/usr/bin/python3",
                "-I",
                "-S",
                "-B",
                "-c",
                _EGRESS_SERVER_SCRIPT,
                EGRESS_CONTROL_SERVER_MARKER,
                descriptor["state_directory"],
                descriptor["peer_address"],
                str(descriptor["port"]),
                str(EGRESS_CONTROL_RECEIVE_TIMEOUT_SECONDS),
            ],
        )
        ready = _wait_guest_control_document(
            runner,
            f"{descriptor['state_directory']}/server-ready.json",
            timeout_seconds=10,
        )
        if ready != {"ready": True}:
            raise D0Error("pre-existing egress server did not become ready")
        _guest(
            runner,
            [
                "/usr/bin/sudo",
                "-n",
                "/usr/bin/python3",
                "-I",
                "-S",
                "-B",
                "-c",
                _EGRESS_CLIENT_SCRIPT,
                EGRESS_CONTROL_CLIENT_MARKER,
                descriptor["state_directory"],
                descriptor["peer_address"],
                str(descriptor["port"]),
            ],
        )
        established = _wait_guest_control_document(
            runner,
            f"{descriptor['state_directory']}/established.json",
            timeout_seconds=10,
        )
        if established != {"established": True}:
            raise D0Error("pre-existing egress flow was not established")
    except BaseException:
        _cleanup_preexisting_egress_control(runner, descriptor)
        raise
    return {
        **descriptor,
        "server_script_sha256": EGRESS_SERVER_SCRIPT_SHA256,
        "client_script_sha256": EGRESS_CLIENT_SCRIPT_SHA256,
        "established_before_guard": True,
        "status": "prepared",
    }


def _trigger_preexisting_egress_control(
    runner: CommandRunner,
    control: Mapping[str, Any],
) -> dict[str, Any]:
    trigger = f"{control['state_directory']}/trigger"
    _guest(
        runner,
        [
            "/usr/bin/sudo",
            "-n",
            "/usr/bin/python3",
            "-I",
            "-S",
            "-B",
            "-c",
            (
                "import os,sys; p=sys.argv[1]; "
                "fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600); "
                "os.write(fd,b'trigger\\n'); os.fsync(fd); os.close(fd)"
            ),
            trigger,
        ],
    )
    outcome = _wait_guest_control_document(
        runner,
        f"{control['state_directory']}/outcome.json",
        timeout_seconds=EGRESS_CONTROL_DOCUMENT_TIMEOUT_SECONDS,
    )
    client = _wait_guest_control_document(
        runner,
        f"{control['state_directory']}/client-outcome.json",
        timeout_seconds=EGRESS_CONTROL_DOCUMENT_TIMEOUT_SECONDS,
    )
    if (
        outcome.get("received_after_guard") is not False
        or outcome.get("detail") not in {"TimeoutError", "closed"}
        or outcome.get("abortive_close") is not True
        or set(outcome) != {"abortive_close", "received_after_guard", "detail"}
        or client.get("triggered") is not True
        or client.get("abortive_close") is not True
    ):
        raise D0Error("pre-established original-direction egress bypassed the guard")
    return {
        "server_outcome": outcome,
        "client_outcome": client,
        "original_direction_rejected": True,
        "status": "pass",
    }


def _nft_match_projection(expressions: Sequence[Any]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for expression in expressions:
        if not isinstance(expression, dict) or "match" not in expression:
            continue
        match = expression["match"]
        if not isinstance(match, dict) or set(match) != {"op", "left", "right"}:
            raise D0Error("D0 nftables match expression differs")
        left = match["left"]
        if not isinstance(left, dict) or len(left) != 1:
            raise D0Error("D0 nftables match left operand differs")
        kind, details = next(iter(left.items()))
        key = details.get("key", details.get("field")) if isinstance(details, dict) else None
        if not isinstance(details, dict) or not isinstance(key, str):
            raise D0Error("D0 nftables match key differs")
        matches.append(
            {
                "kind": kind,
                "key": key,
                "dir": details.get("dir", details.get("protocol")),
                "op": match["op"],
                "right": match["right"],
            }
        )
    return matches


def _validate_buildkit_egress_table(
    value: Any,
    *,
    table: str,
    management_flow: Mapping[str, Any],
    trace: bool = False,
) -> dict[str, int]:
    if not isinstance(value, dict) or not isinstance(value.get("nftables"), list):
        raise D0Error("D0 nftables table output differs")
    tables: list[Mapping[str, Any]] = []
    chains: list[Mapping[str, Any]] = []
    rules: list[Mapping[str, Any]] = []
    sets: list[Mapping[str, Any]] = []
    for entry in value["nftables"]:
        if not isinstance(entry, dict):
            raise D0Error("D0 nftables entry differs")
        if "metainfo" in entry:
            continue
        recognized = [key for key in ("table", "set", "chain", "rule") if key in entry]
        if len(recognized) != 1 or len(entry) != 1:
            raise D0Error("D0 nftables object type differs")
        target = entry[recognized[0]]
        if not isinstance(target, Mapping):
            raise D0Error("D0 nftables object differs")
        {"table": tables, "set": sets, "chain": chains, "rule": rules}[recognized[0]].append(target)
    if len(tables) != 1 or any(
        tables[0].get(key) != expected for key, expected in (("family", "inet"), ("name", table))
    ):
        raise D0Error("D0 nftables table identity differs")
    if len(chains) != 1:
        raise D0Error("D0 nftables chain count differs")
    chain = chains[0]
    if any(
        chain.get(key) != expected
        for key, expected in (
            ("family", "inet"),
            ("table", table),
            ("name", "output"),
            ("type", "filter"),
            ("hook", "output"),
            ("prio", 0),
            ("policy", "drop"),
        )
    ):
        raise D0Error("D0 nftables chain identity differs")
    expected_comments: tuple[str, ...] = (
        "cascadia-d0-loopback",
        "cascadia-d0-ssh-reply",
    )
    if trace:
        expected_comments += tuple(
            f"cascadia-d0-trace-{name}"
            for name in ("tcp4", "udp4", "other4", "tcp6", "udp6", "other6")
        )
    expected_comments += ("cascadia-d0-reject",)
    if (
        len(rules) != len(expected_comments)
        or tuple(rule.get("comment") for rule in rules) != expected_comments
    ):
        raise D0Error("D0 nftables rule ordering or identity differs")
    for rule in rules:
        if any(
            rule.get(key) != expected
            for key, expected in (
                ("family", "inet"),
                ("table", table),
                ("chain", "output"),
            )
        ) or not isinstance(rule.get("expr"), list):
            raise D0Error("D0 nftables rule binding differs")
    expected_set_types = {
        "tcp4": [
            "ifname",
            "ipv4_addr",
            "inet_service",
            "ipv4_addr",
            "inet_service",
        ],
        "udp4": [
            "ifname",
            "ipv4_addr",
            "inet_service",
            "ipv4_addr",
            "inet_service",
        ],
        "other4": ["ifname", "ipv4_addr", "ipv4_addr", "inet_proto"],
        "tcp6": [
            "ifname",
            "ipv6_addr",
            "inet_service",
            "ipv6_addr",
            "inet_service",
        ],
        "udp6": [
            "ifname",
            "ipv6_addr",
            "inet_service",
            "ipv6_addr",
            "inet_service",
        ],
        "other6": ["ifname", "ipv6_addr", "ipv6_addr", "inet_proto"],
    }
    if (not trace and sets) or (trace and len(sets) != len(expected_set_types)):
        raise D0Error("D0 nftables trace-set count differs")
    for selected in sets:
        name = selected.get("name")
        if (
            selected.get("family") != "inet"
            or selected.get("table") != table
            or name not in expected_set_types
            or selected.get("type") != expected_set_types[name]
            or selected.get("flags") != ["dynamic"]
            or selected.get("stmt") != [{"counter": None}]
            or not isinstance(selected.get("size"), int)
            or selected.get("size", 0) <= 0
        ):
            raise D0Error("D0 nftables trace-set schema differs")
    loopback, ssh_reply = rules[:2]
    reject = rules[-1]
    loop_expr = loopback["expr"]
    loop_matches = _nft_match_projection(loop_expr)
    if (
        loop_matches != [{"kind": "meta", "key": "oifname", "dir": None, "op": "==", "right": "lo"}]
        or sum(isinstance(item, dict) and "accept" in item for item in loop_expr) != 1
        or any(
            isinstance(item, dict) and ("reject" in item or "counter" in item) for item in loop_expr
        )
    ):
        raise D0Error("D0 nftables loopback rule differs")
    ssh_expr = ssh_reply["expr"]
    ssh_matches = _nft_match_projection(ssh_expr)
    normalized_ssh = {
        (item["kind"], item["key"], item["dir"], item["op"], json.dumps(item["right"]))
        for item in ssh_matches
    }
    expected_ssh = {
        ("payload", "sport", "tcp", "==", json.dumps(22)),
        ("ct", "state", None, "in", json.dumps(["established"])),
    }
    state_alternative = ("ct", "state", None, "==", json.dumps("established"))
    scalar_in_alternative = ("ct", "state", None, "in", json.dumps("established"))
    alternatives = normalized_ssh & {state_alternative, scalar_in_alternative}
    if len(alternatives) == 1:
        normalized_ssh.remove(next(iter(alternatives)))
        normalized_ssh.add(("ct", "state", None, "in", json.dumps(["established"])))
    if (
        normalized_ssh != expected_ssh
        or sum(isinstance(item, dict) and "accept" in item for item in ssh_expr) != 1
        or any(
            isinstance(item, dict) and ("reject" in item or "counter" in item) for item in ssh_expr
        )
    ):
        raise D0Error("D0 nftables SSH-reply rule differs")
    if trace:
        trace_names = ("tcp4", "udp4", "other4", "tcp6", "udp6", "other6")
        expected_operands = {
            "tcp4": [
                {"meta": {"key": "oifname"}},
                {"payload": {"protocol": "ip", "field": "saddr"}},
                {"payload": {"protocol": "tcp", "field": "sport"}},
                {"payload": {"protocol": "ip", "field": "daddr"}},
                {"payload": {"protocol": "tcp", "field": "dport"}},
            ],
            "udp4": [
                {"meta": {"key": "oifname"}},
                {"payload": {"protocol": "ip", "field": "saddr"}},
                {"payload": {"protocol": "udp", "field": "sport"}},
                {"payload": {"protocol": "ip", "field": "daddr"}},
                {"payload": {"protocol": "udp", "field": "dport"}},
            ],
            "other4": [
                {"meta": {"key": "oifname"}},
                {"payload": {"protocol": "ip", "field": "saddr"}},
                {"payload": {"protocol": "ip", "field": "daddr"}},
                {"meta": {"key": "l4proto"}},
            ],
            "tcp6": [
                {"meta": {"key": "oifname"}},
                {"payload": {"protocol": "ip6", "field": "saddr"}},
                {"payload": {"protocol": "tcp", "field": "sport"}},
                {"payload": {"protocol": "ip6", "field": "daddr"}},
                {"payload": {"protocol": "tcp", "field": "dport"}},
            ],
            "udp6": [
                {"meta": {"key": "oifname"}},
                {"payload": {"protocol": "ip6", "field": "saddr"}},
                {"payload": {"protocol": "udp", "field": "sport"}},
                {"payload": {"protocol": "ip6", "field": "daddr"}},
                {"payload": {"protocol": "udp", "field": "dport"}},
            ],
            "other6": [
                {"meta": {"key": "oifname"}},
                {"payload": {"protocol": "ip6", "field": "saddr"}},
                {"payload": {"protocol": "ip6", "field": "daddr"}},
                {"meta": {"key": "l4proto"}},
            ],
        }
        for index, rule in enumerate(rules[2:-1]):
            name = trace_names[index]
            expressions = rule["expr"]
            set_expressions = [
                item["set"]
                for item in expressions
                if isinstance(item, dict) and set(item) == {"set"}
            ]
            if (
                len(set_expressions) != 1
                or set_expressions[0].get("op") != "update"
                or set_expressions[0].get("set") != f"@{name}"
                or set_expressions[0].get("elem") != {"concat": expected_operands[name]}
                or any(
                    isinstance(item, dict) and ({"accept", "drop", "reject", "counter"} & set(item))
                    for item in expressions
                )
            ):
                raise D0Error("D0 nftables trace rule differs")
    reject_expr = reject["expr"]
    if (
        len(reject_expr) != 2
        or not isinstance(reject_expr[0], dict)
        or set(reject_expr[0]) != {"counter"}
        or not isinstance(reject_expr[0]["counter"], dict)
        or set(reject_expr[0]["counter"]) != {"packets", "bytes"}
        or not all(
            isinstance(reject_expr[0]["counter"][key], int)
            and not isinstance(reject_expr[0]["counter"][key], bool)
            and reject_expr[0]["counter"][key] >= 0
            for key in ("packets", "bytes")
        )
        or not isinstance(reject_expr[1], dict)
        or set(reject_expr[1]) != {"reject"}
    ):
        raise D0Error("D0 nftables reject rule differs")
    return {
        "reject_rules": 1,
        "rejected_packets": reject_expr[0]["counter"]["packets"],
        "rejected_bytes": reject_expr[0]["counter"]["bytes"],
    }


def _egress_trace_projection(value: Any, *, table: str) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or not isinstance(value.get("nftables"), list):
        raise D0Error("D0 nftables trace output differs")
    expected_lengths = {
        "tcp4": 5,
        "udp4": 5,
        "other4": 4,
        "tcp6": 5,
        "udp6": 5,
        "other6": 4,
    }
    records: list[dict[str, Any]] = []
    observed_sets: set[str] = set()
    for entry in value["nftables"]:
        if not isinstance(entry, dict) or "set" not in entry:
            continue
        selected = entry["set"]
        if not isinstance(selected, dict) or selected.get("table") != table:
            raise D0Error("D0 nftables trace-set binding differs")
        name = selected.get("name")
        if name not in expected_lengths or name in observed_sets:
            raise D0Error("D0 nftables trace-set identity differs")
        observed_sets.add(name)
        elements = selected.get("elem", [])
        if not isinstance(elements, list):
            raise D0Error("D0 nftables trace elements differ")
        for wrapper in elements:
            if not isinstance(wrapper, dict) or set(wrapper) != {"elem"}:
                raise D0Error("D0 nftables trace element wrapper differs")
            element = wrapper["elem"]
            if not isinstance(element, dict) or set(element) != {"counter", "val"}:
                raise D0Error("D0 nftables trace element differs")
            value_projection = element["val"]
            counter = element["counter"]
            exact_tuple = (
                value_projection.get("concat") if isinstance(value_projection, dict) else None
            )
            if (
                not isinstance(value_projection, dict)
                or set(value_projection) != {"concat"}
                or not isinstance(exact_tuple, list)
                or len(exact_tuple) != expected_lengths[name]
                or not isinstance(counter, dict)
                or set(counter) != {"bytes", "packets"}
                or any(
                    not isinstance(counter[key], int)
                    or isinstance(counter[key], bool)
                    or counter[key] < 0
                    for key in ("bytes", "packets")
                )
            ):
                raise D0Error("D0 nftables trace tuple or counter differs")
            if name.startswith(("tcp", "udp")) and (
                not isinstance(exact_tuple[0], str)
                or not isinstance(exact_tuple[1], str)
                or not isinstance(exact_tuple[2], int)
                or isinstance(exact_tuple[2], bool)
                or not 0 <= exact_tuple[2] <= 65535
                or not isinstance(exact_tuple[3], str)
                or not isinstance(exact_tuple[4], int)
                or isinstance(exact_tuple[4], bool)
                or not 0 <= exact_tuple[4] <= 65535
            ):
                raise D0Error("D0 nftables typed trace tuple differs")
            record = {
                "bytes": counter["bytes"],
                "packets": counter["packets"],
                "set": name,
                "socket_uid": None,
                "socket_uid_available": False,
                "tuple": exact_tuple,
            }
            _egress_trace_tuple_fields(record)
            records.append(record)
    if observed_sets != set(expected_lengths):
        raise D0Error("D0 nftables trace-set inventory differs")
    return sorted(records, key=lambda item: (item["set"], canonical_json(item["tuple"])))


def _egress_trace_delta(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def keyed(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, bytes], Mapping[str, Any]]:
        result: dict[tuple[str, bytes], Mapping[str, Any]] = {}
        for row in rows:
            key = (str(row.get("set")), canonical_json(row.get("tuple")))
            if key in result:
                raise D0Error("D0 nftables trace tuple is duplicated")
            result[key] = row
        return result

    prior = keyed(before)
    current = keyed(after)
    if not set(prior) <= set(current):
        raise D0Error("D0 nftables trace tuple disappeared")
    changed: list[dict[str, Any]] = []
    for key in sorted(current):
        old = prior.get(key, {"bytes": 0, "packets": 0})
        new = current[key]
        packet_delta = new["packets"] - old["packets"]
        byte_delta = new["bytes"] - old["bytes"]
        if packet_delta < 0 or byte_delta < 0:
            raise D0Error("D0 nftables trace counter regressed")
        if packet_delta or byte_delta:
            changed.append(
                {
                    "bytes": byte_delta,
                    "packets": packet_delta,
                    "set": new["set"],
                    "socket_uid": None,
                    "socket_uid_available": False,
                    "tuple": new["tuple"],
                }
            )
    return {
        "bytes": sum(item["bytes"] for item in changed),
        "packets": sum(item["packets"] for item in changed),
        "tuples": changed,
    }


def _egress_trace_tuple_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    schemas = {
        "tcp4": (
            "output_interface",
            "source_address",
            "source_port",
            "destination_address",
            "destination_port",
        ),
        "udp4": (
            "output_interface",
            "source_address",
            "source_port",
            "destination_address",
            "destination_port",
        ),
        "other4": (
            "output_interface",
            "source_address",
            "destination_address",
            "protocol",
        ),
        "tcp6": (
            "output_interface",
            "source_address",
            "source_port",
            "destination_address",
            "destination_port",
        ),
        "udp6": (
            "output_interface",
            "source_address",
            "source_port",
            "destination_address",
            "destination_port",
        ),
        "other6": (
            "output_interface",
            "source_address",
            "destination_address",
            "protocol",
        ),
    }
    name = row.get("set")
    values = row.get("tuple")
    schema = schemas.get(name)
    if schema is None or not isinstance(values, list) or len(values) != len(schema):
        raise D0Error("D0 nftables trace tuple schema differs")
    fields = {field: values[index] for index, field in enumerate(schema)}
    try:
        source = ipaddress.ip_address(fields["source_address"])
        destination = ipaddress.ip_address(fields["destination_address"])
    except (TypeError, ValueError) as error:
        raise D0Error("D0 nftables trace address differs") from error
    expected_version = 4 if str(name).endswith("4") else 6
    if source.version != expected_version or destination.version != expected_version:
        raise D0Error("D0 nftables trace address family differs")
    if name in {"other4", "other6"} and (
        not isinstance(fields["protocol"], str) or not fields["protocol"]
    ):
        raise D0Error("D0 nftables trace protocol differs")
    return fields


def _classify_egress_trace_delta(delta: Mapping[str, Any]) -> dict[str, Any]:
    """Separate denied link-local control multicast from external egress."""

    if set(delta) != {"bytes", "packets", "tuples"} or not isinstance(delta.get("tuples"), list):
        raise D0Error("D0 nftables trace delta shape differs")
    groups: dict[str, list[dict[str, Any]]] = {
        "external_or_unclassified_denied": [],
        "local_control_denied": [],
    }
    for row in delta["tuples"]:
        if not isinstance(row, Mapping):
            raise D0Error("D0 nftables trace delta row differs")
        if any(
            not isinstance(row.get(key), int)
            or isinstance(row.get(key), bool)
            or row.get(key, -1) < 0
            for key in ("bytes", "packets")
        ):
            raise D0Error("D0 nftables trace delta counter differs")
        fields = _egress_trace_tuple_fields(row)
        destination = ipaddress.ip_address(fields["destination_address"])
        output_interface = fields["output_interface"]
        if not isinstance(output_interface, str):
            raise D0Error("D0 nftables trace output interface differs")
        protocol = fields.get("protocol")
        multicast_scope = (
            destination.packed[1] & 0x0F
            if isinstance(destination, ipaddress.IPv6Address) and destination.is_multicast
            else None
        )
        local_interface = (
            output_interface == "docker0"
            or re.fullmatch(r"veth[0-9a-f]+", output_interface) is not None
        )
        local_control = (
            isinstance(destination, ipaddress.IPv6Address)
            and destination.is_multicast
            and multicast_scope == 2
            and protocol == "ipv6-icmp"
            and local_interface
        )
        selected = "local_control_denied" if local_control else "external_or_unclassified_denied"
        groups[selected].append(
            {
                "bytes": row.get("bytes"),
                "destination_address": str(destination),
                "destination_family": destination.version,
                "destination_multicast_scope": multicast_scope,
                "output_interface": output_interface,
                "packets": row.get("packets"),
                "protocol": protocol,
                "set": row.get("set"),
                "tuple": row.get("tuple"),
            }
        )
    summary: dict[str, Any] = {}
    for name, rows in groups.items():
        summary[name] = {
            "bytes": sum(item["bytes"] for item in rows),
            "packets": sum(item["packets"] for item in rows),
            "tuples": rows,
        }
    if (
        summary["local_control_denied"]["bytes"]
        + summary["external_or_unclassified_denied"]["bytes"]
        != delta["bytes"]
        or summary["local_control_denied"]["packets"]
        + summary["external_or_unclassified_denied"]["packets"]
        != delta["packets"]
    ):
        raise D0Error("D0 nftables scoped trace accounting differs")
    return {
        **summary,
        "external_reject_zero": (
            summary["external_or_unclassified_denied"]["bytes"] == 0
            and summary["external_or_unclassified_denied"]["packets"] == 0
        ),
        "local_control_policy": "denied-and-accounted-not-whitelisted",
        "status": "pass",
    }


def _reject_counter_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, int]:
    expected_keys = {"reject_rules", "rejected_packets", "rejected_bytes"}
    if set(before) != expected_keys or set(after) != expected_keys:
        raise D0Error("D0 nftables counter fields differ")
    if before["reject_rules"] != 1 or after["reject_rules"] != 1:
        raise D0Error("D0 nftables reject-rule cardinality differs")
    delta: dict[str, int] = {}
    for key in ("rejected_packets", "rejected_bytes"):
        prior = before[key]
        current = after[key]
        if (
            not isinstance(prior, int)
            or isinstance(prior, bool)
            or not isinstance(current, int)
            or isinstance(current, bool)
            or prior < 0
            or current < prior
        ):
            raise D0Error("D0 nftables counters regressed or differ")
        delta[key] = current - prior
    return delta


def _require_reject_counter_transition(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    expect_positive: bool,
) -> dict[str, int]:
    delta = _reject_counter_delta(before, after)
    changed = delta["rejected_packets"] > 0 and delta["rejected_bytes"] > 0
    unchanged = delta == {"rejected_packets": 0, "rejected_bytes": 0}
    if expect_positive and not changed:
        raise D0Error("D0 nftables egress-denial control did not increment counters")
    if not expect_positive and not unchanged:
        raise D0Error("BuildKit attempted guest egress during the offline probe")
    return delta


def _trace_counter_comparison(
    trace_delta: Mapping[str, Any],
    reject_delta: Mapping[str, Any],
) -> dict[str, Any]:
    trace_packets = trace_delta.get("packets")
    trace_bytes = trace_delta.get("bytes")
    reject_packets = reject_delta.get("rejected_packets")
    reject_bytes = reject_delta.get("rejected_bytes")
    values = (trace_packets, trace_bytes, reject_packets, reject_bytes)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
        raise D0Error("D0 trace counter comparison fields differ")
    return {
        "byte_difference": trace_bytes - reject_bytes,
        "bytes_equal": trace_bytes == reject_bytes,
        "equal": trace_packets == reject_packets and trace_bytes == reject_bytes,
        "packet_difference": trace_packets - reject_packets,
        "packets_equal": trace_packets == reject_packets,
    }


NFT_SCHEMA_INVENTORY_MAX_BYTES = 512 * 1024


def _bounded_raw_nft_inventory(value: bytes) -> dict[str, Any]:
    """Bind raw nft JSON without interpreting its version-specific element schema."""

    if not value or len(value) > NFT_SCHEMA_INVENTORY_MAX_BYTES:
        raise D0Error("D0 raw nftables inventory size differs")
    try:
        text = value.decode("ascii")
        envelope = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D0Error("D0 raw nftables inventory envelope differs") from error
    if (
        not isinstance(envelope, dict)
        or set(envelope) != {"nftables"}
        or not isinstance(envelope["nftables"], list)
        or not 1 <= len(envelope["nftables"]) <= 128
        or any(not isinstance(item, dict) for item in envelope["nftables"])
    ):
        raise D0Error("D0 raw nftables inventory envelope differs")
    return {
        "entry_count": len(envelope["nftables"]),
        "raw_json": text,
        "raw_sha256": sha256_bytes(value),
        "raw_size": len(value),
        "status": "bounded-raw-inventory",
    }


def nft_schema_inventory_probe(
    packet: Mapping[str, Any],
    runner: CommandRunner,
) -> dict[str, Any]:
    """Capture John2's exact nft JSON after one deterministic rejected tuple.

    This diagnostic deliberately validates only the signed command environment,
    a bounded JSON envelope, and exact post-cleanup non-mutation.  The raw set
    element and rule-expression objects are evidence for a later typed parser;
    they are not interpreted here.
    """

    if packet["host"] != "john2":
        raise D0Error("nftables inventory probe is restricted to John2")

    engine_before = _engine_object_inventory(runner)
    du_before = _buildx_disk_usage(runner)
    buildkit_before = _guest_buildkit_state(runner)
    network_before = _guest_network_snapshot(runner)
    table = "cascadia_r2_d0_inventory"
    absent = _guest(
        runner,
        ["/usr/bin/sudo", "-n", "/usr/sbin/nft", "list", "table", "inet", table],
        check=False,
    )
    if absent.returncode == 0:
        raise D0Error("D0 raw nftables inventory table already exists")

    failsafe_state = f"/run/cascadia-r2-d0-{table}-failsafe.json"
    cleanup_runner = runner.cleanup_runner()
    control: dict[str, Any] | None = None
    management_flow: dict[str, Any] | None = None
    outcome: dict[str, Any] | None = None
    inventory: dict[str, Any] | None = None
    failsafe_cleanup: dict[str, Any] | None = None
    control_cleanup: dict[str, Any] | None = None
    network_after: dict[str, Any] | None = None
    engine_after: dict[str, Any] | None = None
    du_after: list[dict[str, Any]] | None = None
    buildkit_after: dict[str, Any] | None = None
    table_cleanup_required = False
    primary_failure: BaseException | None = None
    cleanup_errors: list[str] = []

    try:
        control = _prepare_preexisting_egress_control(
            runner,
            run_id=f"{packet['run_id']}-nft-schema-inventory-v1",
        )
        table_cleanup_required = True
        management_flow = _install_buildkit_egress_guard(
            runner,
            table=table,
            state_path=failsafe_state,
            trace=True,
        )
        outcome = _trigger_preexisting_egress_control(runner, control)
        listed = _guest(
            runner,
            [
                "/usr/bin/sudo",
                "-n",
                "/usr/sbin/nft",
                "-j",
                "list",
                "table",
                "inet",
                table,
            ],
            maximum=NFT_SCHEMA_INVENTORY_MAX_BYTES,
        )
        inventory = _bounded_raw_nft_inventory(listed.stdout)
    except BaseException as error:
        primary_failure = error
    finally:
        if table_cleanup_required:
            try:
                failsafe_cleanup = _cleanup_buildkit_egress_failsafe(
                    cleanup_runner,
                    table=table,
                    state_path=failsafe_state,
                )
            except BaseException:
                cleanup_errors.append("nft-failsafe-state")
            try:
                _guest(
                    cleanup_runner,
                    [
                        "/usr/bin/sudo",
                        "-n",
                        "/usr/sbin/nft",
                        "delete",
                        "table",
                        "inet",
                        table,
                    ],
                    check=False,
                )
                survived = _guest(
                    cleanup_runner,
                    [
                        "/usr/bin/sudo",
                        "-n",
                        "/usr/sbin/nft",
                        "list",
                        "table",
                        "inet",
                        table,
                    ],
                    check=False,
                )
                if survived.returncode == 0:
                    cleanup_errors.append("nft-table-still-present")
            except BaseException:
                cleanup_errors.append("nft-table-command")
        if control is not None:
            try:
                control_cleanup = _cleanup_preexisting_egress_control(
                    cleanup_runner,
                    control,
                )
            except BaseException:
                cleanup_errors.append("preexisting-egress-control")
        try:
            network_after = _guest_network_snapshot(cleanup_runner)
            engine_after = _engine_object_inventory(cleanup_runner)
            du_after = _buildx_disk_usage(cleanup_runner)
            buildkit_after = _guest_buildkit_state(cleanup_runner)
        except BaseException:
            cleanup_errors.append("post-cleanup-audit")

    if cleanup_errors:
        raise D0Error(
            f"D0 raw nftables inventory cleanup was incomplete: {cleanup_errors!r}"
        ) from primary_failure
    if primary_failure is not None:
        raise primary_failure
    if any(
        value is None
        for value in (
            control,
            management_flow,
            outcome,
            inventory,
            failsafe_cleanup,
            control_cleanup,
            network_after,
            engine_after,
            du_after,
            buildkit_after,
        )
    ):
        raise D0Error("D0 raw nftables inventory evidence is incomplete")
    assert network_after is not None
    if network_after["state"] != network_before["state"]:
        differences = _bounded_state_differences(
            network_before["state"],
            network_after["state"],
        )
        raise D0Error(
            "D0 raw nftables inventory did not restore network state: "
            f"{canonical_json(differences).decode('ascii')}"
        )
    if engine_after != engine_before or du_after != du_before or buildkit_after != buildkit_before:
        raise D0Error("D0 raw nftables inventory mutated engine or BuildKit state")
    lease_transition = _validate_network_lease_transition(
        network_before,
        network_after,
    )
    return {
        "buildkit_after": buildkit_after,
        "buildkit_before": buildkit_before,
        "disk_usage_after": du_after,
        "disk_usage_before": du_before,
        "egress_control": control,
        "egress_control_cleanup": control_cleanup,
        "egress_outcome": outcome,
        "engine_after": engine_after,
        "engine_before": engine_before,
        "failsafe_cleanup": failsafe_cleanup,
        "inventory": inventory,
        "management_flow": management_flow,
        "network_after_sha256": network_after["state_sha256"],
        "network_before_sha256": network_before["state_sha256"],
        "network_lease_transition": lease_transition,
        "project_code_executed": False,
        "protected_seed_values_opened": False,
        "scanner_executed": False,
        "status": "diagnostic-pass",
        "table": table,
    }


DOCKER_ACCOUNTING_COMMAND_MAX_BYTES = 8 * 1024 * 1024


def _bounded_completed_evidence(value: Completed) -> dict[str, Any]:
    if (
        len(value.stdout) > DOCKER_ACCOUNTING_COMMAND_MAX_BYTES
        or len(value.stderr) > DOCKER_ACCOUNTING_COMMAND_MAX_BYTES
    ):
        raise D0Error("Docker accounting command output exceeds its bound")
    return {
        "argv": list(value.argv),
        "returncode": value.returncode,
        "stderr": value.stderr.decode("utf-8", "backslashreplace"),
        "stderr_sha256": sha256_bytes(value.stderr),
        "stderr_size": len(value.stderr),
        "stdout": value.stdout.decode("utf-8", "backslashreplace"),
        "stdout_sha256": sha256_bytes(value.stdout),
        "stdout_size": len(value.stdout),
    }


def docker_accounting_inventory_probe(
    packet: Mapping[str, Any],
    runner: CommandRunner,
) -> dict[str, Any]:
    """Read-only inventory for Docker/BuildKit/containerd accounting divergence."""

    if packet["host"] != "john2":
        raise D0Error("Docker accounting inventory is restricted to John2")
    network_before = _guest_network_snapshot(runner)
    engine_before = _engine_object_inventory(runner)
    buildkit_before = _guest_buildkit_state(runner)
    docker_commands = {
        "buildx_disk_usage": [
            DOCKER,
            "buildx",
            "du",
            "--builder",
            "default",
            "--format",
            "{{json .}}",
        ],
        "container_list": [
            DOCKER,
            "container",
            "ls",
            "--all",
            "--no-trunc",
            "--format",
            "{{json .}}",
        ],
        "image_list": [
            DOCKER,
            "image",
            "ls",
            "--all",
            "--digests",
            "--no-trunc",
            "--format",
            "{{json .}}",
        ],
        "info": [DOCKER, "info", "--format", "{{json .}}"],
        "system_df_json": [DOCKER, "system", "df", "--format", "{{json .}}"],
        "system_df_verbose": [DOCKER, "system", "df", "--verbose"],
        "volume_list": [
            DOCKER,
            "volume",
            "ls",
            "--format",
            "{{json .}}",
        ],
    }
    docker_evidence = {
        name: _bounded_completed_evidence(
            runner.run(
                argv,
                maximum=DOCKER_ACCOUNTING_COMMAND_MAX_BYTES,
                check=False,
            )
        )
        for name, argv in docker_commands.items()
    }
    guest_commands = {
        f"ctr_{namespace}_{kind}": [
            "/usr/bin/sudo",
            "-n",
            "/usr/bin/ctr",
            "--namespace",
            namespace,
            *arguments,
        ]
        for namespace in ("moby", "buildkit")
        for kind, arguments in (
            ("containers", ("containers", "list")),
            ("content", ("content", "list")),
            ("images", ("images", "list")),
            ("leases", ("leases", "list")),
            ("snapshots", ("snapshots", "list")),
        )
    }
    guest_commands["docker_data_usage"] = [
        "/usr/bin/sudo",
        "-n",
        "/usr/bin/du",
        "--all",
        "--block-size=1",
        "/var/lib/docker",
    ]
    guest_evidence = {
        name: _bounded_completed_evidence(
            _guest(
                runner,
                argv,
                maximum=DOCKER_ACCOUNTING_COMMAND_MAX_BYTES,
                check=False,
            )
        )
        for name, argv in guest_commands.items()
    }
    engine_after = _engine_object_inventory(runner)
    buildkit_after = _guest_buildkit_state(runner)
    network_after = _guest_network_snapshot(runner)
    if network_after["state"] != network_before["state"]:
        raise D0Error("Docker accounting inventory mutated guest network state")
    if engine_after != engine_before or buildkit_after != buildkit_before:
        raise D0Error("Docker accounting inventory mutated Engine or BuildKit state")
    return {
        "buildkit_after": buildkit_after,
        "buildkit_before": buildkit_before,
        "docker": docker_evidence,
        "engine_after": engine_after,
        "engine_before": engine_before,
        "guest": guest_evidence,
        "network_after_sha256": network_after["state_sha256"],
        "network_before_sha256": network_before["state_sha256"],
        "network_lease_transition": _validate_network_lease_transition(
            network_before,
            network_after,
        ),
        "project_code_executed": False,
        "protected_seed_values_opened": False,
        "read_only": True,
        "scanner_executed": False,
        "status": "diagnostic-pass",
    }


def _egress_socket_state_is_packet_capable(
    state: str,
    timer: list[str] | None,
) -> bool:
    """Only a conventional TIME-WAIT timer is a safe listed terminal state."""

    return not (state == "TIME-WAIT" and (timer is None or timer[0] == "timewait"))


def _egress_socket_observations(
    output: bytes,
    descriptor: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Project exact all-state ss rows for the synthetic negative-control tuple."""

    host = str(descriptor["host_address"])
    peer = str(descriptor["peer_address"])
    port = str(descriptor["port"])
    rows: list[dict[str, Any]] = []
    for raw in output.splitlines():
        if len(raw) > 4096:
            raise D0Error("egress socket inventory row exceeds its bound")
        text = raw.decode("utf-8", "backslashreplace")
        if host not in text or peer not in text or f":{port}" not in text:
            continue
        fields = text.split()
        if not fields:
            raise D0Error("egress socket inventory row is empty")
        timer_match = re.search(r"timer:\(([^)]*)\)", text)
        inode_match = re.search(r"(?:^|\s)ino:(\d+)(?:\s|$)", text)
        uid_match = re.search(r"(?:^|\s)uid:(\d+)(?:\s|$)", text)
        cgroup_match = re.search(r"(?:^|\s)cgroup:(\S+)", text)
        socket_match = re.search(r"(?:^|\s)sk:(\S+)", text)
        timer = timer_match.group(1).split(",") if timer_match else None
        rows.append(
            {
                "cgroup": cgroup_match.group(1) if cgroup_match else None,
                "inode": int(inode_match.group(1)) if inode_match else None,
                "line": text,
                "process_users_available": "users:((" in text,
                "socket": socket_match.group(1) if socket_match else None,
                "state": fields[0],
                "timer": timer,
                "uid": int(uid_match.group(1)) if uid_match else None,
                "packet_capable": _egress_socket_state_is_packet_capable(fields[0], timer),
            }
        )
    return rows


def egress_socket_inventory_probe(
    packet: Mapping[str, Any],
    runner: CommandRunner,
) -> dict[str, Any]:
    """Read-only post-failure inventory for the synthetic egress-control socket."""

    if packet["host"] != "john2":
        raise D0Error("egress socket inventory is restricted to John2")
    descriptor = _egress_control_descriptor(packet["run_id"])
    network_before = _guest_network_snapshot(runner)
    engine_before = _engine_object_inventory(runner)
    buildkit_before = _guest_buildkit_state(runner)
    daemon_before = _daemon_accounting_inventory(runner)
    sockets = _guest(
        runner,
        ["/usr/bin/ss", "-Hntoape"],
        maximum=4 * 1024 * 1024,
    )
    namespace_list = _guest(
        runner,
        ["/usr/bin/sudo", "-n", "/usr/sbin/ip", "netns", "list"],
        maximum=1024 * 1024,
    )
    link = _guest(
        runner,
        [
            "/usr/bin/sudo",
            "-n",
            "/usr/sbin/ip",
            "link",
            "show",
            descriptor["host_interface"],
        ],
        check=False,
        maximum=1024 * 1024,
    )
    state = _guest(
        runner,
        [
            "/usr/bin/sudo",
            "-n",
            "/usr/bin/test",
            "!",
            "-e",
            descriptor["state_directory"],
        ],
        check=False,
    )
    nft = _guest(
        runner,
        [
            "/usr/bin/sudo",
            "-n",
            "/usr/sbin/nft",
            "list",
            "table",
            "inet",
            "cascadia_r2_d0",
        ],
        check=False,
        maximum=1024 * 1024,
    )
    daemon_after = _daemon_accounting_inventory(runner)
    buildkit_after = _guest_buildkit_state(runner)
    engine_after = _engine_object_inventory(runner)
    network_after = _guest_network_snapshot(runner)
    if network_after["state"] != network_before["state"]:
        raise D0Error("egress socket inventory mutated guest network state")
    if (
        daemon_after != daemon_before
        or buildkit_after != buildkit_before
        or engine_after != engine_before
    ):
        raise D0Error("egress socket inventory mutated runtime accounting state")
    namespaces = {
        line.split()[0]
        for line in namespace_list.stdout.decode("utf-8", "strict").splitlines()
        if line
    }
    return {
        "buildkit_after": buildkit_after,
        "buildkit_before": buildkit_before,
        "daemon_after": daemon_after,
        "daemon_before": daemon_before,
        "descriptor": descriptor,
        "engine_after": engine_after,
        "engine_before": engine_before,
        "negative_control_cleanup": {
            "host_interface_absent": link.returncode != 0,
            "namespace_absent": descriptor["namespace"] not in namespaces,
            "nft_table_absent": nft.returncode != 0,
            "state_directory_absent": state.returncode == 0,
        },
        "network_after_sha256": network_after["state_sha256"],
        "network_before_sha256": network_before["state_sha256"],
        "network_lease_transition": _validate_network_lease_transition(
            network_before,
            network_after,
        ),
        "project_code_executed": False,
        "protected_seed_values_opened": False,
        "read_only": True,
        "scanner_executed": False,
        "socket_inventory": {
            "command": ["/usr/bin/ss", "-Hntoape"],
            "matches": _egress_socket_observations(sockets.stdout, descriptor),
            "raw_sha256": sha256_bytes(sockets.stdout),
            "raw_size": len(sockets.stdout),
        },
        "status": "diagnostic-pass",
    }


def docker_accounting_cleanup_probe(
    packet: Mapping[str, Any],
    runner: CommandRunner,
) -> dict[str, Any]:
    """Remove only the exact scanner-attestation residue proven by V21."""

    if packet["host"] != "john2":
        raise D0Error("Docker accounting cleanup is restricted to John2")
    scanner_expected = packet["artifacts"]["scanner_oci"]
    if not isinstance(scanner_expected, Mapping):
        raise D0Error("Docker accounting cleanup packet omits scanner identity")
    scanner_path = Path(packet["paths"]["scanner_oci"])
    scanner_identity = _regular_file_identity(scanner_path)
    if (
        scanner_identity["size"] != scanner_expected["size"]
        or scanner_identity["sha256"] != scanner_expected["sha256"]
    ):
        raise D0Error("Docker accounting cleanup scanner archive differs")
    verified = verify_scanner_oci_archive(_read_bounded_regular(scanner_path, 128 * 1024 * 1024))
    network_before = _guest_network_snapshot(runner)
    engine_before = _engine_object_inventory(runner)
    buildkit_before = _guest_buildkit_state(runner)
    daemon_before = _daemon_accounting_inventory(runner)
    if daemon_before != {
        "build_cache": 0,
        "containers": 0,
        "images": 1,
        "info_containers": 0,
        "info_images": 1,
        "volumes": 0,
    }:
        raise D0Error("Docker accounting cleanup precondition differs")
    cleanup = _cleanup_scanner_attestation_residue(runner, verified)
    engine_after = _engine_object_inventory(runner)
    buildkit_after = _guest_buildkit_state(runner)
    daemon_after = _daemon_accounting_inventory(runner)
    network_after = _guest_network_snapshot(runner)
    _require_empty_daemon_accounting(daemon_after)
    if network_after["state"] != network_before["state"]:
        raise D0Error("Docker accounting cleanup mutated guest network state")
    if engine_after != engine_before or buildkit_after != buildkit_before:
        raise D0Error("Docker accounting cleanup mutated Engine or BuildKit state")
    return {
        "buildkit_after": buildkit_after,
        "buildkit_before": buildkit_before,
        "cleanup": cleanup,
        "daemon_after": daemon_after,
        "daemon_before": daemon_before,
        "engine_after": engine_after,
        "engine_before": engine_before,
        "network_after_sha256": network_after["state_sha256"],
        "network_before_sha256": network_before["state_sha256"],
        "network_lease_transition": _validate_network_lease_transition(
            network_before,
            network_after,
        ),
        "project_code_executed": False,
        "protected_seed_values_opened": False,
        "scanner_executed": False,
        "status": "cleanup-pass",
    }


def _scanner_resolver_context(reference: str) -> tuple[bytes, dict[str, Any]]:
    dockerfile = f"FROM {reference}\n".encode("ascii")
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
        info = tarfile.TarInfo("Dockerfile")
        info.size = len(dockerfile)
        info.mode = 0o444
        info.uid = 0
        info.gid = 0
        info.mtime = 0
        info.uname = ""
        info.gname = ""
        archive.addfile(info, io.BytesIO(dockerfile))
    context = output.getvalue()
    return context, {
        "archive_bytes": len(context),
        "archive_sha256": sha256_bytes(context),
        "dockerfile_sha256": sha256_bytes(dockerfile),
        "reference": reference,
    }


def buildkit_probe(
    packet: Mapping[str, Any],
    runner: CommandRunner,
    *,
    resolver_tls_only: bool = False,
    egress_trace: bool = False,
    output_inventory_only: bool = False,
    attestation_inventory_only: bool = False,
    allow_docker_lazy_init: bool = False,
) -> dict[str, Any]:
    """Prove an integrated offline BuildKit build with a pinned local SBOM scanner."""

    if packet["host"] != "john2":
        raise D0Error("BuildKit feature probe is restricted to John2")
    diagnostic_inventory = output_inventory_only or attestation_inventory_only
    if diagnostic_inventory and resolver_tls_only:
        raise D0Error("BuildKit output inventory cannot use resolver-only mode")
    if output_inventory_only and attestation_inventory_only:
        raise D0Error("BuildKit inventory modes are mutually exclusive")
    if allow_docker_lazy_init and (not egress_trace or resolver_tls_only or diagnostic_inventory):
        raise D0Error("Docker lazy initialization requires the canonical full policy")
    if resolver_tls_only:
        context = b""
        context_receipt: dict[str, Any] = {}
    else:
        context, context_receipt = probe_context()
        expected_context = packet["artifacts"]["probe_context"]
        if (
            context_receipt["archive_bytes"] != PROBE_ARCHIVE_SIZE
            or context_receipt["archive_sha256"] != PROBE_ARCHIVE_SHA256
            or expected_context["size"] != PROBE_ARCHIVE_SIZE
            or expected_context["sha256"] != PROBE_ARCHIVE_SHA256
        ):
            raise D0Error("BuildKit probe context differs from the signed deterministic identity")

    scanner_expected = packet["artifacts"]["scanner_oci"]
    if not isinstance(scanner_expected, Mapping):
        raise D0Error("BuildKit verification packet omits the derived scanner OCI identity")
    scanner_path = Path(packet["paths"]["scanner_oci"])
    scanner_identity = _regular_file_identity(scanner_path)
    if (
        scanner_identity["size"] != scanner_expected["size"]
        or scanner_identity["sha256"] != scanner_expected["sha256"]
    ):
        raise D0Error("BuildKit scanner archive differs from the signed packet")
    scanner_archive = _read_bounded_regular(scanner_path, 128 * 1024 * 1024)
    scanner_verified = verify_scanner_oci_archive(scanner_archive)
    supply_files: dict[str, dict[str, Any]] = {}
    for artifact_key, path_key in (
        ("scanner_license", "scanner_license"),
        ("scanner_source_archive", "scanner_source_archive"),
    ):
        identity = _regular_file_identity(Path(packet["paths"][path_key]))
        expected = packet["artifacts"][artifact_key]
        if identity["size"] != expected["size"] or identity["sha256"] != expected["sha256"]:
            raise D0Error(f"BuildKit {artifact_key} bytes differ from the signed packet")
        supply_files[artifact_key] = identity

    engine_before = _engine_object_inventory(runner)
    daemon_before = _daemon_accounting_inventory(runner)
    _require_empty_daemon_accounting(daemon_before)
    du_before = _buildx_disk_usage(runner)
    buildkit_before = _guest_buildkit_state(runner)
    _require_empty_buildkit_state(buildkit_before, buildx_disk_usage=du_before)
    network_before, bridge_before, boundary_before_lease = _network_boundary_snapshot(runner)

    table = "cascadia_r2_d0"
    absent = _guest(
        runner,
        ["/usr/bin/sudo", "-n", "/usr/sbin/nft", "list", "table", "inet", table],
        check=False,
    )
    if absent.returncode == 0:
        raise D0Error("D0 guest egress-guard table already exists")
    failsafe_state = f"/run/cascadia-r2-d0-{table}-failsafe.json"
    nft_program: bytes | None = None
    management_flow: dict[str, Any] | None = None
    failsafe_cancellation: dict[str, Any] | None = None
    failsafe_cleanup: dict[str, Any] | None = None

    cleanup_runner = runner.cleanup_runner()
    registry_descriptor = _scanner_registry_descriptor(scanner_verified)
    scanner_reference = registry_descriptor["generator_reference"]
    if resolver_tls_only:
        context, context_receipt = _scanner_resolver_context(scanner_reference)
    scanner_image_id = scanner_verified["manifest_digest"]
    scanner_load: dict[str, Any] | None = None
    scanner_registry: dict[str, Any] | None = None
    scanner_trust_cleanup: dict[str, Any] | None = None
    scanner_registry_cleanup: dict[str, Any] | None = None
    scanner_attestation_cleanup: dict[str, Any] | None = None
    verified: dict[str, Any] | None = None
    egress_control: dict[str, Any] | None = None
    control_counters: dict[str, int] | None = None
    control_table_inventory: dict[str, Any] | None = None
    control_table_state: Any = None
    post_build_table_inventory: dict[str, Any] | None = None
    post_build_table_state: Any = None
    trace_before_build: list[dict[str, Any]] | None = None
    trace_after_build: list[dict[str, Any]] | None = None
    egress_trace_evidence: dict[str, Any] | None = None
    build_counter_delta: dict[str, int] | None = None
    build_started_unix_ns: int | None = None
    build_finished_unix_ns: int | None = None
    socket_sampler_launch: dict[str, Any] | None = None
    socket_attribution: dict[str, Any] | None = None
    socket_sampler_active = False
    egress_outcome: dict[str, Any] | None = None
    active_control_route: dict[str, Any] | None = None
    egress_cleanup: dict[str, Any] | None = None
    egress_quiescence: dict[str, Any] | None = None
    egress_control_active = False
    table_state: Any = None
    installed_table_state: Any = None
    network_after: dict[str, Any] | None = None
    bridge_after: dict[str, Any] | None = None
    boundary_after_lease: dict[str, Any] | None = None
    du_after: list[dict[str, Any]] | None = None
    buildkit_after: dict[str, Any] | None = None
    engine_after: dict[str, Any] | None = None
    daemon_after: dict[str, Any] | None = None
    build_capabilities: dict[str, Any] | None = None
    metadata_path: Path | None = None
    metadata_removed = False
    primary_failure: BaseException | None = None
    cleanup_errors: list[str] = []
    table_cleanup_required = False
    registry_cleanup_required = False
    try:
        egress_control = _prepare_preexisting_egress_control(
            runner,
            run_id=packet["run_id"],
        )
        egress_control_active = True
        scanner_load = _load_scanner_image(runner, scanner_archive)
        registry_cleanup_required = True
        scanner_registry = _prepare_scanner_registry(runner, scanner_verified)
        table_cleanup_required = True
        management_flow = _install_buildkit_egress_guard(
            runner,
            table=table,
            state_path=failsafe_state,
            trace=egress_trace,
        )
        nft_program = _buildkit_egress_program(
            table,
            management_flow,
            trace=egress_trace,
        )
        installed_table = _guest(
            runner,
            [
                "/usr/bin/sudo",
                "-n",
                "/usr/sbin/nft",
                "-j",
                "list",
                "table",
                "inet",
                table,
            ],
        )
        installed_table_state = _json_command(
            installed_table.stdout,
            "installed D0 nftables table",
        )
        installed_counters = _validate_buildkit_egress_table(
            installed_table_state,
            table=table,
            management_flow=management_flow,
            trace=egress_trace,
        )
        failsafe_cancellation = _cancel_buildkit_egress_failsafe(
            runner,
            table=table,
            state_path=failsafe_state,
        )
        egress_outcome = _trigger_preexisting_egress_control(runner, egress_control)
        route = _json_command(
            _guest(
                runner,
                [
                    "/usr/sbin/ip",
                    "-j",
                    "route",
                    "get",
                    egress_control["peer_address"],
                ],
                maximum=128 * 1024,
            ).stdout,
            "D0 active negative-control route",
        )
        if not isinstance(route, list) or len(route) != 1 or not isinstance(route[0], dict):
            raise D0Error("D0 active negative-control route evidence differs")
        active_control_route = route[0]
        egress_cleanup = _cleanup_preexisting_egress_control(
            runner,
            egress_control,
        )
        egress_control_active = False
        egress_quiescence = _wait_preexisting_egress_socket_absent(
            runner,
            egress_control,
        )
        control_table = _guest(
            runner,
            [
                "/usr/bin/sudo",
                "-n",
                "/usr/sbin/nft",
                "-j",
                "list",
                "table",
                "inet",
                table,
            ],
        )
        control_table_inventory = _bounded_raw_nft_inventory(control_table.stdout)
        control_table_state = _json_command(
            control_table.stdout,
            "D0 nftables negative-control table",
        )
        control_counters = _validate_buildkit_egress_table(
            control_table_state,
            table=table,
            management_flow=management_flow,
            trace=egress_trace,
        )
        negative_control_delta = _require_reject_counter_transition(
            installed_counters,
            control_counters,
            expect_positive=True,
        )
        if egress_trace:
            trace_before_build = _egress_trace_projection(
                control_table_state,
                table=table,
            )
            socket_sampler_launch = _start_socket_sampler(
                runner,
                registry_descriptor,
            )
            socket_sampler_active = True
        build_started_unix_ns = time.time_ns()
        if resolver_tls_only:
            result = runner.run(
                [
                    DOCKER,
                    "buildx",
                    "build",
                    "--builder",
                    "default",
                    "--platform",
                    "linux/arm64",
                    "--network",
                    "none",
                    "--no-cache",
                    "--pull",
                    "--output",
                    "type=cacheonly",
                    "-",
                ],
                stdin=context,
                timeout=900,
                maximum=64 * 1024 * 1024,
            )
            verified = {
                "mode": "resolver-tls-only",
                "stderr_sha256": sha256_bytes(result.stderr),
                "stdout_sha256": sha256_bytes(result.stdout),
                "status": "pass",
            }
        else:
            build_argv = [
                DOCKER,
                "buildx",
                "build",
                "--builder",
                "default",
                "--platform",
                "linux/arm64",
                "--network",
                "none",
                "--no-cache",
                "--pull=false",
                "--provenance=mode=max",
                f"--sbom=generator={scanner_reference}",
                "--output",
                "type=oci,dest=-",
                "-",
            ]
            if diagnostic_inventory:
                metadata_path = Path(packet["paths"]["homebrew_temp"]) / (
                    f".cascadia-r2-d0-policy-metadata-{os.getpid()}.json"
                )
                if os.path.lexists(metadata_path):
                    raise D0Error("BuildKit output-inventory metadata path already exists")
                build_argv[-1:-1] = ["--metadata-file", str(metadata_path)]
                capability_commands = {
                    "buildx_inspect": [DOCKER, "buildx", "inspect", "default"],
                    "buildx_list": [DOCKER, "buildx", "ls", "--format", "{{json .}}"],
                    "buildx_version": [DOCKER, "buildx", "version"],
                    "docker_info": [DOCKER, "info", "--format", "{{json .}}"],
                }
                build_capabilities = {
                    name: _bounded_completed_evidence(
                        runner.run(
                            argv,
                            maximum=DOCKER_ACCOUNTING_COMMAND_MAX_BYTES,
                            check=False,
                        )
                    )
                    for name, argv in capability_commands.items()
                }
            result = runner.run(
                build_argv,
                stdin=context,
                timeout=900,
                maximum=min(
                    packet["limits"]["output_max_bytes"],
                    512 * 1024 * 1024,
                ),
            )
            build_finished_unix_ns = time.time_ns()
            if diagnostic_inventory:
                if metadata_path is None:
                    raise D0Error("BuildKit output-inventory metadata path is absent")
                metadata_bytes = _read_bounded_regular(metadata_path, 8 * 1024 * 1024)
                metadata_value = _json_command(
                    metadata_bytes,
                    "BuildKit output-inventory metadata",
                )
                metadata_path.unlink()
                if os.path.lexists(metadata_path):
                    raise D0Error("BuildKit output-inventory metadata survived deletion")
                metadata_removed = True
                verified = {
                    "build_capabilities": build_capabilities,
                    "build_command": build_argv,
                    "build_stderr_base64": base64.b64encode(result.stderr).decode("ascii"),
                    "build_stderr_sha256": sha256_bytes(result.stderr),
                    "build_stderr_size": len(result.stderr),
                    "build_stdout_sha256": sha256_bytes(result.stdout),
                    "build_stdout_size": len(result.stdout),
                    "metadata": metadata_value,
                    "metadata_removed": True,
                    "metadata_sha256": sha256_bytes(metadata_bytes),
                    "metadata_size": len(metadata_bytes),
                    "mode": (
                        "raw-attestation-inventory"
                        if attestation_inventory_only
                        else "raw-output-inventory"
                    ),
                    "oci": (
                        _probe_oci_attestation_inventory(result.stdout)
                        if attestation_inventory_only
                        else _probe_oci_inventory(result.stdout)
                    ),
                    "status": "diagnostic-pass",
                }
            else:
                if build_started_unix_ns is None:
                    raise D0Error("BuildKit policy build start timestamp is absent")
                verified = verify_probe_oci(
                    result.stdout,
                    build_finished_unix_ns=build_finished_unix_ns,
                    build_started_unix_ns=build_started_unix_ns,
                )
        if egress_trace:
            post_build_table = _guest(
                runner,
                [
                    "/usr/bin/sudo",
                    "-n",
                    "/usr/sbin/nft",
                    "-j",
                    "list",
                    "table",
                    "inet",
                    table,
                ],
                maximum=NFT_SCHEMA_INVENTORY_MAX_BYTES,
            )
            post_build_table_inventory = _bounded_raw_nft_inventory(post_build_table.stdout)
            post_build_table_state = _json_command(
                post_build_table.stdout,
                "D0 nftables trace immediately after build",
            )
        if socket_sampler_active:
            socket_attribution = _stop_socket_sampler(runner, registry_descriptor)
            socket_sampler_active = False
        if egress_trace:
            if (
                control_table_inventory is None
                or post_build_table_inventory is None
                or post_build_table_state is None
                or trace_before_build is None
                or socket_sampler_launch is None
                or socket_attribution is None
                or control_counters is None
                or active_control_route is None
            ):
                raise D0Error("D0 raw trace diagnostic evidence is incomplete")
            post_build_counters = _validate_buildkit_egress_table(
                post_build_table_state,
                table=table,
                management_flow=management_flow,
                trace=True,
            )
            trace_after_build = _egress_trace_projection(
                post_build_table_state,
                table=table,
            )
            build_counter_delta = _reject_counter_delta(
                control_counters,
                post_build_counters,
            )
            trace_delta = _egress_trace_delta(
                trace_before_build,
                trace_after_build,
            )
            scoped_trace = _classify_egress_trace_delta(trace_delta)
            counter_equality = _trace_counter_comparison(
                trace_delta,
                build_counter_delta,
            )
            destinations: set[str] = set()
            raw_match_tokens: set[str] = set()
            output_interface_values: set[str] = set()
            for item in trace_delta["tuples"]:
                values = item["tuple"]
                fields = _egress_trace_tuple_fields(item)
                destinations.add(str(fields["destination_address"]))
                output_interface_values.add(str(values[0]))
                raw_match_tokens.update(
                    str(fields[key])
                    for key in (
                        "source_address",
                        "source_port",
                        "destination_address",
                        "destination_port",
                    )
                    if key in fields and str(fields[key])
                )
            active_raw_tuple_routes: dict[str, Any] = {}
            active_raw_tuple_route_errors: dict[str, Any] = {}
            for destination in sorted(destinations):
                route_command = _guest(
                    runner,
                    ["/usr/sbin/ip", "-j", "route", "get", destination],
                    maximum=128 * 1024,
                    check=False,
                )
                if route_command.returncode != 0:
                    if not diagnostic_inventory:
                        raise D0Error("D0 active rejected-egress route command failed")
                    active_raw_tuple_route_errors[destination] = _bounded_completed_evidence(
                        route_command
                    )
                    continue
                route = _json_command(
                    route_command.stdout,
                    "D0 active rejected-egress route",
                )
                if not isinstance(route, list) or len(route) != 1 or not isinstance(route[0], dict):
                    if not diagnostic_inventory:
                        raise D0Error("D0 active rejected-egress route evidence differs")
                    active_raw_tuple_route_errors[destination] = {
                        "argv": list(route_command.argv),
                        "returncode": route_command.returncode,
                        "stderr_sha256": sha256_bytes(route_command.stderr),
                        "stderr_size": len(route_command.stderr),
                        "stdout_sha256": sha256_bytes(route_command.stdout),
                        "stdout_size": len(route_command.stdout),
                        "status": "invalid-json-shape",
                    }
                    continue
                active_raw_tuple_routes[destination] = route[0]
            raw_tuple_socket_matches = [
                record
                for record in socket_attribution["records"]
                if any(token in record["line"] for token in raw_match_tokens)
            ]
            control_socket_matches = [
                record
                for record in socket_attribution["records"]
                if str(egress_control["port"]) in record["line"]
                and (
                    str(egress_control["host_address"]) in record["line"]
                    or str(egress_control["peer_address"]) in record["line"]
                )
            ]
            egress_trace_evidence = {
                "active_control_route": active_control_route,
                "active_raw_tuple_routes": active_raw_tuple_routes,
                "active_raw_tuple_route_errors": active_raw_tuple_route_errors,
                "after_build": trace_after_build,
                "before_build": trace_before_build,
                "build_finished_unix_ns": build_finished_unix_ns,
                "build_started_unix_ns": build_started_unix_ns,
                "control_socket_matches": control_socket_matches,
                "counter_equality": counter_equality,
                "delta": trace_delta,
                "nft_output_interface_values": sorted(output_interface_values),
                "nft_socket_uid_available": False,
                "raw_post_build_table": post_build_table_inventory,
                "raw_pre_build_table": control_table_inventory,
                "raw_reject_counters": {
                    "post_build": post_build_counters,
                    "pre_build": control_counters,
                    "delta": build_counter_delta,
                },
                "raw_tuple_socket_matches": raw_tuple_socket_matches,
                "scoped_accounting": scoped_trace,
                "socket_records": socket_attribution["records"],
                "socket_sampler": {
                    "command": ["/usr/bin/ss", "-Hntoape"],
                    "launch": socket_sampler_launch,
                    "pid": socket_attribution["pid"],
                    "sample_count": socket_attribution["sample_count"],
                },
                "status": (
                    "diagnostic-counter-match"
                    if counter_equality["equal"]
                    else "diagnostic-counter-mismatch"
                ),
                "primary_inventory_complete_before_route_enrichment": True,
                "route_enrichment_status": ("partial" if active_raw_tuple_route_errors else "pass"),
            }
    except BaseException as error:
        primary_failure = error
    finally:
        if metadata_path is not None and not metadata_removed and os.path.lexists(metadata_path):
            try:
                metadata_identity = _regular_file_identity(metadata_path)
                if metadata_identity["size"] > 8 * 1024 * 1024:
                    raise D0Error("BuildKit output-inventory metadata exceeds its bound")
                metadata_path.unlink()
                metadata_removed = True
            except BaseException:
                cleanup_errors.append("build-metadata")
        if socket_sampler_active:
            try:
                socket_attribution = _stop_socket_sampler(
                    cleanup_runner,
                    registry_descriptor,
                )
                socket_sampler_active = False
            except BaseException:
                cleanup_errors.append("socket-attribution-sampler")
        if registry_cleanup_required:
            if scanner_registry is not None:
                try:
                    scanner_trust_cleanup = _cleanup_scanner_registry_trust(
                        cleanup_runner,
                        registry_descriptor,
                    )
                except BaseException:
                    cleanup_errors.append("scanner-loopback-trust")
            try:
                scanner_registry_cleanup = _cleanup_scanner_registry(
                    cleanup_runner,
                    registry_descriptor,
                )
            except BaseException:
                cleanup_errors.append("scanner-loopback-registry")
        if table_cleanup_required:
            try:
                failsafe_cleanup = _cleanup_buildkit_egress_failsafe(
                    cleanup_runner,
                    table=table,
                    state_path=failsafe_state,
                )
            except BaseException:
                cleanup_errors.append("nft-failsafe-state")
            try:
                listed = _guest(
                    cleanup_runner,
                    [
                        "/usr/bin/sudo",
                        "-n",
                        "/usr/sbin/nft",
                        "-j",
                        "list",
                        "table",
                        "inet",
                        table,
                    ],
                    check=False,
                )
                if listed.returncode == 0:
                    table_state = _json_command(listed.stdout, "D0 nftables table")
            except BaseException:
                cleanup_errors.append("nft-table-evidence")
            try:
                _guest(
                    cleanup_runner,
                    [
                        "/usr/bin/sudo",
                        "-n",
                        "/usr/sbin/nft",
                        "delete",
                        "table",
                        "inet",
                        table,
                    ],
                    check=False,
                )
                survived = _guest(
                    cleanup_runner,
                    [
                        "/usr/bin/sudo",
                        "-n",
                        "/usr/sbin/nft",
                        "list",
                        "table",
                        "inet",
                        table,
                    ],
                    check=False,
                )
                if survived.returncode == 0:
                    cleanup_errors.append("nft-table-still-present")
            except BaseException:
                cleanup_errors.append("nft-table-command")
        if egress_control is not None and egress_control_active:
            try:
                egress_cleanup = _cleanup_preexisting_egress_control(
                    cleanup_runner,
                    egress_control,
                )
                egress_control_active = False
            except BaseException:
                cleanup_errors.append("preexisting-egress-control")
        try:
            pruned = cleanup_runner.run(
                [DOCKER, "buildx", "prune", "--builder", "default", "--all", "--force"],
                timeout=900,
                maximum=64 * 1024 * 1024,
                check=False,
            )
            if pruned.returncode != 0:
                cleanup_errors.append("buildkit-cache")
        except BaseException:
            cleanup_errors.append("buildkit-cache-command")
        try:
            inspected = cleanup_runner.run(
                [DOCKER, "image", "inspect", scanner_image_id],
                timeout=300,
                check=False,
            )
            if inspected.returncode == 0:
                removed = cleanup_runner.run(
                    [DOCKER, "image", "rm", "--force", scanner_image_id],
                    timeout=300,
                    check=False,
                )
                if removed.returncode != 0:
                    cleanup_errors.append(f"scanner-image:{scanner_image_id}")
        except BaseException:
            cleanup_errors.append(f"scanner-image-command:{scanner_image_id}")
        try:
            if (
                cleanup_runner.run(
                    [DOCKER, "image", "inspect", scanner_image_id],
                    timeout=300,
                    check=False,
                ).returncode
                == 0
            ):
                cleanup_errors.append(f"scanner-image-still-present:{scanner_image_id}")
            if scanner_load is not None:
                scanner_attestation_cleanup = _cleanup_scanner_attestation_residue(
                    cleanup_runner,
                    scanner_verified,
                )
            du_after = _buildx_disk_usage(cleanup_runner)
            buildkit_after = _guest_buildkit_state(cleanup_runner)
            engine_after = _engine_object_inventory(cleanup_runner)
            daemon_after = _daemon_accounting_inventory(cleanup_runner)
            _require_empty_daemon_accounting(daemon_after)
            network_after, bridge_after, boundary_after_lease = _network_boundary_snapshot(
                cleanup_runner
            )
        except BaseException:
            cleanup_errors.append("post-cleanup-audit")

    if cleanup_errors:
        raise D0Error(
            f"D0 BuildKit cleanup was incomplete: {cleanup_errors!r}"
        ) from primary_failure
    if primary_failure is not None:
        raise primary_failure
    if (
        scanner_load is None
        or scanner_registry is None
        or scanner_trust_cleanup is None
        or scanner_registry_cleanup is None
        or scanner_attestation_cleanup is None
        or verified is None
        or control_counters is None
        or table_state is None
        or installed_table_state is None
        or egress_control is None
        or egress_outcome is None
        or active_control_route is None
        or egress_cleanup is None
        or egress_quiescence is None
        or network_after is None
        or bridge_after is None
        or boundary_after_lease is None
        or du_after is None
        or buildkit_after is None
        or engine_after is None
        or daemon_after is None
        or management_flow is None
        or failsafe_cancellation is None
        or failsafe_cleanup is None
        or nft_program is None
        or build_started_unix_ns is None
        or build_finished_unix_ns is None
    ):
        raise D0Error("BuildKit probe evidence is incomplete")
    _validate_scanner_registry_cleanup(
        scanner_registry_cleanup,
        registry_descriptor,
        require_requests=True,
    )
    counters = _validate_buildkit_egress_table(
        table_state,
        table=table,
        management_flow=management_flow,
        trace=egress_trace,
    )
    if egress_trace:
        if egress_trace_evidence is None or build_counter_delta is None:
            raise D0Error("D0 raw trace diagnostic evidence is absent")
        if (
            not diagnostic_inventory
            and not resolver_tls_only
            and (
                egress_trace_evidence["counter_equality"].get("equal") is not True
                or egress_trace_evidence["scoped_accounting"].get("external_reject_zero")
                is not True
            )
        ):
            raise D0Error("D0 full policy attempted external or unaccounted egress")
    else:
        build_counter_delta = _require_reject_counter_transition(
            control_counters,
            counters,
            expect_positive=False,
        )
        egress_trace_evidence = None
    if engine_after != engine_before or daemon_after != daemon_before:
        raise D0Error("BuildKit probe did not restore Engine or daemon accounting")
    _require_empty_buildkit_state(buildkit_after, buildx_disk_usage=du_after)
    network_lifecycle, lease_transition = _validate_docker_network_lifecycle(
        network_before,
        network_after,
        bridge_before=bridge_before,
        bridge_after=bridge_after,
        allow_cold_transition=allow_docker_lazy_init,
    )
    return {
        "mode": (
            "resolver-egress-trace"
            if egress_trace and resolver_tls_only
            else "full-output-inventory-egress-trace"
            if egress_trace and output_inventory_only
            else "full-attestation-inventory-egress-trace"
            if egress_trace and attestation_inventory_only
            else "full-attestation-egress-trace"
            if egress_trace
            else "resolver-tls-only"
            if resolver_tls_only
            else "full-attestation"
        ),
        "context": context_receipt,
        "output": verified,
        "scanner": {
            "supply_files": supply_files,
            "archive": scanner_identity,
            "verification": scanner_verified,
            "load": scanner_load,
            "loopback_registry": scanner_registry,
            "loopback_trust_cleanup": scanner_trust_cleanup,
            "loopback_registry_cleanup": scanner_registry_cleanup,
            "attestation_residue_cleanup": scanner_attestation_cleanup,
            "explicit_generator": scanner_reference,
            "removed": True,
        },
        "egress_guard": {
            "table": table,
            "program_sha256": sha256_bytes(nft_program),
            "management_flow": management_flow,
            "failsafe_cancellation": failsafe_cancellation,
            "failsafe_cleanup": failsafe_cleanup,
            "installed_table_sha256": sha256_bytes(canonical_json(installed_table_state)),
            "negative_control_destination": (
                f"{egress_control['peer_address']}:{egress_control['port']}"
            ),
            "preexisting_flow": egress_control,
            "preexisting_flow_outcome": egress_outcome,
            "preexisting_flow_cleanup": egress_cleanup,
            "preexisting_flow_quiescence": egress_quiescence,
            "installed_counters": installed_counters,
            "negative_control_counters": control_counters,
            "negative_control_delta": negative_control_delta,
            "counters": counters,
            "build_counter_delta": build_counter_delta,
            "diagnostic_trace": egress_trace_evidence,
            "network_before_sha256": network_before["state_sha256"],
            "network_after_sha256": network_after["state_sha256"],
            "network_lease_transition": lease_transition,
            "restored": network_lifecycle["status"] == "exact-restoration",
        },
        "network_lifecycle": {
            **network_lifecycle,
            "boundary_after_lease": boundary_after_lease,
            "boundary_before_lease": boundary_before_lease,
        },
        "engine_before": engine_before,
        "engine_after": engine_after,
        "daemon_before": daemon_before,
        "daemon_after": daemon_after,
        "buildkit_before": buildkit_before,
        "buildkit_after": buildkit_after,
        "disk_usage_before": du_before,
        "disk_usage_after": du_after,
        "status": "pass",
    }


def full_policy_buildkit_probe(
    packet: Mapping[str, Any],
    runner: CommandRunner,
) -> dict[str, Any]:
    """Run the one canonical full BuildKit policy used by qualification and diagnostics.

    Full-policy acceptance is based on scoped trace attribution, not the raw
    reject-counter total. The guard intentionally denies internal link-local
    multicast traffic, so a non-zero total can coexist with zero external or
    unclassified egress. Keeping this entry point shared prevents qualification
    from drifting back to the older total-counter-only policy.
    """

    first = buildkit_probe(
        packet,
        runner,
        egress_trace=True,
        allow_docker_lazy_init=True,
    )
    lifecycle = first.get("network_lifecycle")
    if not isinstance(lifecycle, dict):
        raise D0Error("canonical full policy omitted network lifecycle evidence")
    if lifecycle.get("initial_mode") == "warm":
        if lifecycle.get("status") != "exact-restoration":
            raise D0Error("warm Docker network did not restore exactly")
        lifecycle["full_policy_probe_count"] = 1
        return first
    if (
        lifecycle.get("initial_mode") != "cold"
        or lifecycle.get("status") != "validated-lazy-initialization"
        or lifecycle.get("transition") != "cold-to-warm"
    ):
        raise D0Error("cold Docker network lifecycle did not stabilize")
    second = buildkit_probe(packet, runner, egress_trace=True)
    second_lifecycle = second.get("network_lifecycle")
    if (
        not isinstance(second_lifecycle, dict)
        or second_lifecycle.get("initial_mode") != "warm"
        or second_lifecycle.get("status") != "exact-restoration"
        or first["egress_guard"].get("network_after_sha256")
        != second["egress_guard"].get("network_before_sha256")
        or lifecycle.get("bridge_projection_sha256")
        != second_lifecycle.get("bridge_projection_sha256")
    ):
        raise D0Error("stabilized Docker network continuity differs")
    second["cold_initialization_probe"] = first
    second["network_lifecycle"] = {
        **second_lifecycle,
        "cold_initialization_network_before_sha256": first["egress_guard"]["network_before_sha256"],
        "cold_initialization_network_after_sha256": first["egress_guard"]["network_after_sha256"],
        "cold_initialization_status": lifecycle["status"],
        "full_policy_probe_count": 2,
        "status": "cold-initialized-then-exact-restoration",
    }
    return second


def _validate_smoke_volume_inspects(
    volumes: Sequence[Mapping[str, Any]],
    *,
    expected_names: Sequence[str],
    run_id: str,
) -> list[dict[str, Any]]:
    if len(volumes) != 2 or sorted(expected_names) != sorted(
        volume.get("Name") for volume in volumes
    ):
        raise D0Error("smoke inspected-volume name set differs")
    result: list[dict[str, Any]] = []
    for volume in volumes:
        mountpoint = volume.get("Mountpoint")
        if (
            volume.get("Driver") != "local"
            or volume.get("Labels") != {"cascadia.r2-map.d0.run": run_id}
            or volume.get("Options") not in (None, {})
            or not isinstance(mountpoint, str)
            or not mountpoint.startswith("/var/lib/docker/volumes/")
            or not mountpoint.endswith("/_data")
        ):
            raise D0Error("smoke volume driver, label, or guest mountpoint differs")
        result.append(dict(volume))
    return sorted(result, key=lambda item: item["Name"])


def smoke_and_volume_roundtrip(
    runner: CommandRunner,
    *,
    run_id: str,
    smoke_archive: bytes,
) -> dict[str, Any]:
    archive_identity = verify_smoke_oci_archive(smoke_archive)
    image = "alpine:3.22.1-d0"
    # With the sealed containerd snapshotter enabled Docker exposes the OCI
    # manifest digest as the Engine image ID.  The verified manifest already
    # binds the config digest and layer/diff ID.
    image_id = archive_identity["manifest_digest"]
    input_volume = f"cascadia-r2-d0-{run_id}-input"
    output_volume = f"cascadia-r2-d0-{run_id}-output"
    container = f"cascadia-r2-d0-{run_id}-roundtrip"
    label = f"cascadia.r2-map.d0.run={run_id}"
    names = (
        container,
        f"{container}-import",
        f"{container}-prepare",
        f"{container}-export",
    )
    namespace_owned = False
    result: dict[str, Any] | None = None
    failure: BaseException | None = None
    image_store_baseline: list[dict[str, Any]] = []
    loaded_store: dict[str, Any] | None = None
    try:
        image_store_baseline = _image_store_rows(runner)
        if image_store_baseline:
            raise D0Error("smoke requires an empty Engine image store")
        for volume in (input_volume, output_volume):
            if runner.run([DOCKER, "volume", "inspect", volume], check=False).returncode == 0:
                raise D0Error("smoke volume name already exists")
        for name in names:
            if runner.run([DOCKER, "container", "inspect", name], check=False).returncode == 0:
                raise D0Error("smoke container name already exists")
        labelled_containers = runner.run(
            [DOCKER, "container", "ls", "--all", "--quiet", "--filter", f"label={label}"],
        )
        labelled_volumes = runner.run(
            [DOCKER, "volume", "ls", "--quiet", "--filter", f"label={label}"],
        )
        if labelled_containers.stdout.strip() or labelled_volumes.stdout.strip():
            raise D0Error("smoke run label already exists")
        namespace_owned = True
        load = runner.run([DOCKER, "image", "load"], stdin=smoke_archive, timeout=300)
        if image.encode() not in load.stdout and image.encode() not in load.stderr:
            raise D0Error("smoke image load did not bind the frozen reference")
        loaded_store = _require_single_logical_image(
            _image_store_rows(runner),
            repository="alpine",
            tag="3.22.1-d0",
            image_id=image_id,
        )
        image_inspect_bytes = runner.run([DOCKER, "image", "inspect", image_id]).stdout
        try:
            image_inspect = json.loads(image_inspect_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise D0Error("loaded smoke image inspect is invalid") from error
        if not isinstance(image_inspect, list) or len(image_inspect) != 1:
            raise D0Error("loaded smoke image inspect cardinality differs")
        image_value = image_inspect[0]
        if (
            image_value.get("Id") != image_id
            or image_value.get("Architecture") != "arm64"
            or image_value.get("Os") != "linux"
            or image_value.get("RepoTags") != [image]
            or image_value.get("RootFS", {}).get("Type") != "layers"
            or image_value.get("RootFS", {}).get("Layers") != [archive_identity["diff_id"]]
        ):
            raise D0Error("loaded smoke image config, tag, or layer identity differs")
        for volume in (input_volume, output_volume):
            runner.run([DOCKER, "volume", "create", "--label", label, volume])
        inspected_volumes: list[dict[str, Any]] = []
        for volume_name in (input_volume, output_volume):
            volume_inspect_bytes = runner.run([DOCKER, "volume", "inspect", volume_name]).stdout
            try:
                one_volume = json.loads(volume_inspect_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise D0Error("smoke volume inspect is invalid") from error
            if (
                not isinstance(one_volume, list)
                or len(one_volume) != 1
                or not isinstance(one_volume[0], dict)
                or one_volume[0].get("Name") != volume_name
            ):
                raise D0Error("smoke single-volume inspect identity differs")
            inspected_volumes.append(one_volume[0])
        inspected_volumes = _validate_smoke_volume_inspects(
            inspected_volumes,
            expected_names=(input_volume, output_volume),
            run_id=run_id,
        )
        input_payload = b"cascadia-r2-d0-volume-roundtrip-v1\n"
        input_tar = io.BytesIO()
        with tarfile.open(fileobj=input_tar, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
            info = tarfile.TarInfo("input.txt")
            info.size = len(input_payload)
            info.mode = 0o444
            info.uid = info.gid = info.mtime = 0
            archive.addfile(info, io.BytesIO(input_payload))
        runner.run(
            [
                DOCKER,
                "run",
                "--rm",
                "-i",
                *hardened_flags(f"{container}-import", user="0:0"),
                "--mount",
                f"type=volume,src={input_volume},dst=/input",
                image_id,
                "/bin/sh",
                "-eu",
                "-c",
                "tar -x -C /input && chmod 0444 /input/input.txt && sync",
            ],
            stdin=input_tar.getvalue(),
        )
        runner.run(
            [
                DOCKER,
                "run",
                "--rm",
                *hardened_flags(f"{container}-prepare", user="0:0"),
                "--mount",
                f"type=volume,src={output_volume},dst=/output",
                image_id,
                "/bin/chmod",
                "0733",
                "/output",
            ]
        )
        runner.run(
            [
                DOCKER,
                "create",
                *hardened_flags(container),
                "--label",
                label,
                "--mount",
                f"type=volume,src={input_volume},dst=/input,readonly",
                "--mount",
                f"type=volume,src={output_volume},dst=/output",
                image_id,
                "/bin/sh",
                "-eu",
                "-c",
                "cat /input/input.txt > /output/output.txt && chmod 0444 /output/output.txt",
            ]
        )
        runner.run([DOCKER, "start", "--attach", container])
        inspect = runner.run([DOCKER, "inspect", container]).stdout
        try:
            inspected = json.loads(inspect)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise D0Error("smoke container inspect output differs") from error
        if not isinstance(inspected, list) or len(inspected) != 1:
            raise D0Error("smoke container inspect cardinality differs")
        host_config = inspected[0].get("HostConfig", {})
        config = inspected[0].get("Config", {})
        mounts = inspected[0].get("Mounts", [])
        pid_namespace = _validate_default_private_pid_mode(host_config)
        tmpfs = host_config.get("Tmpfs") or {}
        mount_contract = sorted(
            (
                item.get("Type"),
                item.get("Destination"),
                item.get("RW"),
            )
            for item in mounts
            if isinstance(item, dict)
        )
        if (
            host_config.get("NetworkMode") != "none"
            or host_config.get("IpcMode") != "none"
            or host_config.get("ReadonlyRootfs") is not True
            or host_config.get("CapDrop") != ["ALL"]
            or host_config.get("CapAdd") not in (None, [])
            or host_config.get("SecurityOpt") != ["no-new-privileges=true"]
            or host_config.get("Privileged") is not False
            or host_config.get("AutoRemove") is not False
            or host_config.get("Memory") != 64 * 1024**2
            or host_config.get("MemorySwap") != 64 * 1024**2
            or host_config.get("NanoCpus") != 1_000_000_000
            or host_config.get("PidsLimit") != 64
            or not isinstance(tmpfs, dict)
            or tmpfs.get("/tmp") != "rw,noexec,nosuid,nodev,size=16m"
            or host_config.get("Binds") not in (None, [])
            or host_config.get("Devices") not in (None, [])
            or host_config.get("PortBindings") not in (None, {})
            or config.get("ExposedPorts") not in (None, {})
            or config.get("User") != "65532:65532"
            or config.get("Image") not in {image, image_id}
            or mount_contract
            != [
                ("volume", "/input", False),
                ("volume", "/output", True),
            ]
        ):
            raise D0Error("smoke container hardening differs")
        exported = runner.run(
            [
                DOCKER,
                "run",
                "--rm",
                *hardened_flags(f"{container}-export"),
                "--mount",
                f"type=volume,src={output_volume},dst=/output,readonly",
                image_id,
                "/bin/tar",
                "-c",
                "-C",
                "/output",
                "output.txt",
            ]
        ).stdout
        members = _safe_tar(exported)
        if members != {"output.txt": input_payload}:
            raise D0Error("volume roundtrip output differs")
        result = {
            "input_sha256": sha256_bytes(input_payload),
            "output_sha256": sha256_bytes(members["output.txt"]),
            "image": {
                "id": image_id,
                "manifest_digest": archive_identity["manifest_digest"],
                "layer_digest": archive_identity["layer_digest"],
                "diff_id": archive_identity["diff_id"],
            },
            "container_contract": {
                "network": "none",
                "pid": pid_namespace,
                "ipc": "none",
                "read_only": True,
                "cap_drop": ["ALL"],
                "security_opt": ["no-new-privileges=true"],
                "pids_limit": 64,
                "memory_bytes": 64 * 1024**2,
                "cpu_nanos": 1_000_000_000,
                "host_binds": [],
                "mounts": [
                    [kind, destination, writable] for kind, destination, writable in mount_contract
                ],
            },
            "volumes": sorted(volume["Name"] for volume in inspected_volumes),
        }
    except BaseException as error:
        failure = error
    cleanup_errors: list[str] = []
    cleanup_runner = runner.cleanup_runner()

    def cleanup_command(argv: Sequence[str]) -> Completed | None:
        try:
            return cleanup_runner.run(argv, check=False, timeout=300)
        except BaseException:
            cleanup_errors.append(f"command:{argv[1] if len(argv) > 1 else argv[0]}")
            return None

    if namespace_owned:
        for name in names:
            inspected = cleanup_command([DOCKER, "container", "inspect", name])
            if inspected is not None and inspected.returncode == 0:
                removed = cleanup_command([DOCKER, "rm", "--force", name])
                if removed is None or removed.returncode != 0:
                    cleanup_errors.append(f"container:{name}")
        for volume in (output_volume, input_volume):
            inspected = cleanup_command([DOCKER, "volume", "inspect", volume])
            if inspected is not None and inspected.returncode == 0:
                removed = cleanup_command([DOCKER, "volume", "rm", "--force", volume])
                if removed is None or removed.returncode != 0:
                    cleanup_errors.append(f"volume:{volume}")
        inspected = cleanup_command([DOCKER, "image", "inspect", image_id])
        if inspected is not None and inspected.returncode == 0:
            removed_image = cleanup_command([DOCKER, "image", "rm", "--force", image_id])
            if removed_image is None or removed_image.returncode != 0:
                cleanup_errors.append(f"image:{image_id}")
    if namespace_owned:
        for name in names:
            inspected = cleanup_command([DOCKER, "container", "inspect", name])
            if inspected is None or inspected.returncode == 0:
                cleanup_errors.append(f"container-still-present:{name}")
        for volume in (input_volume, output_volume):
            inspected = cleanup_command([DOCKER, "volume", "inspect", volume])
            if inspected is None or inspected.returncode == 0:
                cleanup_errors.append(f"volume-still-present:{volume}")
        try:
            if _image_store_rows(cleanup_runner) != image_store_baseline:
                cleanup_errors.append("image-store-baseline-drift")
        except BaseException:
            cleanup_errors.append("image-store-audit")
    if cleanup_errors:
        raise D0Error(f"smoke cleanup was incomplete: {cleanup_errors!r}") from failure
    if failure is not None:
        raise failure
    if result is None:
        raise D0Error("smoke result is absent")
    if loaded_store is None:
        raise D0Error("smoke logical image-store evidence is absent")
    result["image_store"] = loaded_store
    result["cleanup"] = "complete"
    return result


def host_report(
    packet: Mapping[str, Any],
    *,
    status: str,
    evidence: Mapping[str, Any],
    started_unix_ms: int,
) -> dict[str, Any]:
    validate_work_packet(packet)
    if status not in {"pass", "fail", "rolled-back"}:
        raise D0Error("host report status differs")
    report: dict[str, Any] = {
        "schema_id": HOST_REPORT_SCHEMA,
        "schema_version": 4,
        "campaign_id": CAMPAIGN_ID,
        "run_id": packet["run_id"],
        "cycle_id": packet["cycle_id"],
        "host": packet["host"],
        "role": packet["role"],
        "phase": packet["phase"],
        "operation": primary_operation(
            packet["host"], packet["phase"], packet["allowed_operations"]
        ),
        "packet_sha256": packet["packet_sha256"],
        "started_unix_ms": started_unix_ms,
        "finished_unix_ms": time.time_ns() // 1_000_000,
        "status": status,
        "evidence": dict(evidence),
        "protected_seed_values_opened": False,
        "project_code_executed": False,
    }
    report["report_sha256"] = document_sha256(report, "report_sha256")
    validate_host_report(report, packet=packet)
    return report


def format_plan(plan: Mapping[str, Any]) -> str:
    lines = []
    for command in plan.get("commands", []):
        argv = command.get("argv", [])
        lines.append(" ".join(shlex.quote(item) for item in argv))
    return "\n".join(lines) + ("\n" if lines else "")
