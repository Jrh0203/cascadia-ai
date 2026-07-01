.PHONY: probe

probe:
	@/usr/bin/find /Users /opt/homebrew /usr/local -type f -name cargo -perm -0100 2>/dev/null || true
	@/bin/ls -la /Users/john2/cascadia-bench/r2-map-v1/toolchains/bin 2>/dev/null || true
