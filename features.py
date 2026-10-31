"""Shared helpers: data loading and feature construction."""
import numpy as np
import pandas as pd

CLASSES = ['Availability', 'Benign', 'Reconnaissance', 'Spoofing']
SERVICES = ['-', 'dns', 'mqtt', 'ssl', 'http', 'ntp', 'irc', 'dhcp', 'radius']
PROTOS = {'tcp': 0, 'udp': 1, 'icmp': 2}


def load_data(data_dir):
    train = pd.read_csv(f'{data_dir}/train.csv')
    test = pd.read_csv(f'{data_dir}/test.csv')
    y = train['Target'].map({c: i for i, c in enumerate(CLASSES)}).values
    return train, test, y


def build_features(df, port_extras=False, keep_ports=False):
    """Full-feature table (both traffic directions).

    port_extras: add coarse destination-port range flags and a within-file port-rarity score.
    keep_ports : keep the raw port numbers (only used by the validation script to show they hurt).
    """
    X = df.drop(columns=[c for c in ['id', 'Target'] if c in df.columns]).copy()
    X['proto'] = X['proto'].map(PROTOS).astype(int)
    X['service'] = X['service'].map({s: i for i, s in enumerate(SERVICES)}).fillna(-1).astype(int)
    eps = 1e-6
    n = X.fwd_pkts_tot + X.bwd_pkts_tot
    X['fb_pkt_ratio'] = (X.fwd_pkts_tot + 1) / (X.bwd_pkts_tot + 1)
    X['fb_pay_ratio'] = (X['fwd_pkts_payload.tot'] + 1) / (X['bwd_pkts_payload.tot'] + 1)
    X['pay_per_pkt'] = X['flow_pkts_payload.tot'] / (n + eps)
    X['hdr_per_pkt'] = (X.fwd_header_size_tot + X.bwd_header_size_tot) / (n + eps)
    for flag in ['SYN', 'RST', 'ACK', 'FIN']:
        X[f'{flag.lower()}_per_pkt'] = X[f'flow_{flag}_flag_count'] / (n + eps)
    X['no_reply'] = (X.bwd_pkts_tot == 0).astype(int)
    X['tot_pkts'] = n
    X['dur_log'] = np.log1p(X.flow_duration)
    X['orig_ephemeral'] = (X['id.orig_p'] >= 32768).astype(int)
    if port_extras:
        p = df['id.resp_p']
        X['port_wk'] = (p < 1024).astype(int)
        X['port_reg'] = ((p >= 1024) & (p < 49152)).astype(int)
        X['port_dyn'] = (p >= 49152).astype(int)
        # how common this destination port is inside its own file (scanners touch many rare ports)
        X['port_rarity'] = np.log1p(p.map(p.value_counts())) / np.log1p(len(df))
    drop = ['id.orig_p'] if keep_ports else ['id.orig_p', 'id.resp_p']
    return X.drop(columns=drop).replace([np.inf, -np.inf], 0)


def sender_features(df):
    """Sender-side (forward direction) features only.

    These do not depend on whether the target replied, which matters because the test set contains the
    same SYN-flood signature as train Availability but mostly WITHOUT replies.
    """
    cols = [c for c in df.columns
            if c.startswith('fwd_') and c not in ('fwd_pkts_per_sec', 'fwd_bulk_rate')]
    X = df[cols + ['flow_SYN_flag_count']].copy()
    X['proto'] = df.proto.map(PROTOS).astype(int)
    X['service'] = df.service.map({s: i for i, s in enumerate(SERVICES)}).fillna(-1).astype(int)
    X['hdr_per_pkt'] = df.fwd_header_size_tot / (df.fwd_pkts_tot + 1e-6)
    X['pay_per_pkt'] = df['fwd_pkts_payload.tot'] / (df.fwd_pkts_tot + 1e-6)
    return X
