#!/usr/bin/env python3
'''
Parallel range downloader.

Hugging Face throttles per TCP connection here (measured: 2 MB/s on one stream, but
~2.6 MB/s each across six), so the fix is many connections, not a faster client.
Writes every chunk into one preallocated file with pwrite, so there is nothing to
assemble afterwards and a restart can skip the ranges already on disk.
'''
import os, sys, time, threading, urllib.request

URL, DEST, NCONN = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 16
CHUNK = 64 * 1024 * 1024

def head():
    r = urllib.request.Request(URL, method='HEAD')
    r.add_header('User-Agent', 'curl/8.5.0')
    with urllib.request.urlopen(r, timeout=60) as resp:
        return int(resp.headers['Content-Length']), resp.url

total, real = head()
print('size %.2f GB' % (total/1e9), flush=True)

if not os.path.exists(DEST) or os.path.getsize(DEST) != total:
    with open(DEST, 'wb') as f:
        f.truncate(total)

ranges = [(s, min(s + CHUNK, total) - 1) for s in range(0, total, CHUNK)]
lock = threading.Lock()
done_bytes = [0]
failed = []
idx = [0]
fd = os.open(DEST, os.O_WRONLY)

def worker():
    while True:
        with lock:
            if idx[0] >= len(ranges): return
            i = idx[0]; idx[0] += 1
        s, e = ranges[i]
        for attempt in range(6):
            try:
                r = urllib.request.Request(real)
                r.add_header('Range', 'bytes=%d-%d' % (s, e))
                r.add_header('User-Agent', 'curl/8.5.0')
                with urllib.request.urlopen(r, timeout=120) as resp:
                    buf = resp.read()
                if len(buf) != e - s + 1:
                    raise IOError('short read %d' % len(buf))
                os.pwrite(fd, buf, s)
                with lock:
                    done_bytes[0] += len(buf)
                break
            except Exception as ex:
                if attempt == 5:
                    print('chunk %d FAILED: %s' % (i, ex), flush=True)
                    with lock:
                        failed.append(i)
                    break
                time.sleep(2 * (attempt + 1))

def report():
    t0 = time.time(); last = 0
    while idx[0] < len(ranges) or done_bytes[0] < total:
        time.sleep(10)
        d = done_bytes[0]
        mb = (d - last) / 10 / 1e6
        print('%.1f/%.1f GB  %.0f MB/s  eta %.0f min' %
              (d/1e9, total/1e9, mb, ((total-d)/1e6/max(mb,0.1))/60), flush=True)
        last = d
        if d >= total: return

threading.Thread(target=report, daemon=True).start()
ts = [threading.Thread(target=worker) for _ in range(NCONN)]
[t.start() for t in ts]; [t.join() for t in ts]
os.close(fd)
got = os.path.getsize(DEST)
print('DONE %s %.2f GB (expected %.2f)' % (DEST, got/1e9, total/1e9), flush=True)
print('failed chunks: %r' % failed, flush=True)
print('PDL_OK' if (done_bytes[0] >= total and not failed) else 'PDL_INCOMPLETE', flush=True)
