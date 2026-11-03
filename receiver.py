"""Append validated ESP32 CSV packets over UDP; Ctrl+C stops safely."""
import argparse
import csv
import io
from pathlib import Path
import socket
import pandas as pd
from features import FEATURES, validate_frame

def parse_packet(packet):
    parts = next(csv.reader(io.StringIO(packet.decode('utf-8').strip())))
    if len(parts) != 13:
        raise ValueError('Expected 13 columns.')
    row = dict(zip(FEATURES, map(float, parts[:12])))
    row['label'] = parts[12]
    validate_frame(pd.DataFrame([row]))
    return row

def receive(out='data/measured_valve_dataset.csv', port=5005, host='127.0.0.1', limit=0):
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = FEATURES + ['label']
    existing = path.exists() and path.stat().st_size > 0
    if existing:
        with path.open(newline='', encoding='utf-8') as f:
            if next(csv.reader(f)) != columns:
                raise ValueError('Existing CSV header does not match.')
    count = 0
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock, path.open('a', newline='', encoding='utf-8') as f:
        sock.bind((host, port))
        sock.settimeout(1)
        writer = csv.DictWriter(f, fieldnames=columns)
        if not existing:
            writer.writeheader()
            f.flush()
        print(f'Listening at {host}:{port}; Ctrl+C to stop.', flush=True)
        try:
            while not limit or count < limit:
                try:
                    packet, _ = sock.recvfrom(4096)
                except socket.timeout:
                    continue
                try:
                    row = parse_packet(packet)
                except (ValueError, UnicodeError, StopIteration, csv.Error):
                    print('Skipped invalid packet.', flush=True)
                    continue
                writer.writerow(row)
                f.flush()
                count += 1
                print(f'{count}: {row["label"]}, {row["rms_current"]:.3f} A', flush=True)
        except KeyboardInterrupt:
            pass
    print(f'Saved {count} rows to {path}')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='data/measured_valve_dataset.csv')
    p.add_argument('--port', type=int, default=5005)
    p.add_argument('--host', default='127.0.0.1', help='Use 0.0.0.0 for ESP32 on a trusted LAN.')
    p.add_argument('--limit', type=int, default=0)
    receive(**vars(p.parse_args()))
