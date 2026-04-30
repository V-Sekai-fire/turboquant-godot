#!/bin/sh
# If arg list is long enough to risk ARG_MAX, write objects to a response file
# and pass @file to the real ar, which GNU ar supports.
if [ "$#" -gt 200 ]; then
    flags="$1"; tgt="$2"; shift 2
    rsp=$(mktemp /tmp/ar-rsp.XXXXXX)
    printf '%s\n' "$@" > "$rsp"
    /usr/bin/ar "$flags" "$tgt" "@$rsp"
    status=$?
    rm -f "$rsp"
    exit $status
fi
exec /usr/bin/ar "$@"
