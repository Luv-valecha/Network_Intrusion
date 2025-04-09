from flask import Flask, request, jsonify
from flask_cors import CORS
from scapy.all import sniff, IP, TCP, UDP
from collections import defaultdict
import pandas as pd

app = Flask(__name__)
CORS(app)

# Service and flag mappings
PORT_SERVICE_MAP = {
    21: 'ftp', 22: 'ssh', 23: 'telnet', 25: 'smtp', 53: 'domain',
    67: 'bootps', 68: 'bootpc', 80: 'http', 110: 'pop_3', 111: 'sunrpc',
    113: 'auth', 119: 'nntp', 123: 'ntp_u', 135: 'epmap', 139: 'netbios_ssn',
    143: 'imap4', 161: 'snmp', 443: 'http_443', 512: 'exec', 513: 'login',
    514: 'shell', 515: 'printer', 993: 'imap4', 995: 'pop_3', 1080: 'socks',
    1433: 'sql_net', 3306: 'mysql', 3389: 'remote_job', 5900: 'vmnet',
    8001: 'http_8001'
}

TCP_FLAG_MAP = {
    'S': 'S0', 'SA': 'SF', 'R': 'REJ', 'RA': 'RSTR',
    'FA': 'S2', 'F': 'S1', 'A': 'S3'
}

def capture_packet_features(duration=10):
    packets = sniff(filter="ip", timeout=duration, store=True)

    connection_stats = defaultdict(lambda: {'count': 0})
    host_stats = defaultdict(lambda: {'srv_count': 0, 'srv_serror_count': 0})
    traffic_stats = defaultdict(lambda: {'src_bytes': 0})

    data = []

    for pkt in packets:
        if IP not in pkt:
            continue

        ip = pkt[IP]
        src_ip, dst_ip = ip.src, ip.dst
        proto = pkt.proto
        src_bytes = len(ip.payload)
        service = None
        flag = None
        serror = False

        if TCP in pkt:
            tcp = pkt[TCP]
            service = tcp.dport
            flag = str(tcp.flags)
            if 'R' in flag:
                serror = True
        elif UDP in pkt:
            udp = pkt[UDP]
            service = udp.dport
            flag = "U"
        else:
            continue

        conn_key = (src_ip, dst_ip)
        reverse_key = (dst_ip, src_ip)
        traffic_stats[conn_key]['src_bytes'] += src_bytes
        dst_bytes = traffic_stats[reverse_key]['src_bytes']

        conn_service_key = (src_ip, dst_ip, service)
        connection_stats[conn_service_key]['count'] += 1
        host_stats[dst_ip]['srv_count'] += 1
        if serror:
            host_stats[dst_ip]['srv_serror_count'] += 1

        same_srv = sum(1 for k in connection_stats if k[0] == src_ip and k[1] == dst_ip and k[2] == service)
        diff_srv = sum(1 for k in connection_stats if k[0] == src_ip and k[1] == dst_ip and k[2] != service)

        same_srv_rate = same_srv / max(1, connection_stats[conn_service_key]['count'])
        diff_srv_rate = diff_srv / max(1, connection_stats[conn_service_key]['count'])

        dst_host_srv_count = host_stats[dst_ip]['srv_count']
        dst_host_srv_serror_rate = host_stats[dst_ip]['srv_serror_count'] / max(1, dst_host_srv_count)

        mapped_service = PORT_SERVICE_MAP.get(service, 'other')
        mapped_flag = TCP_FLAG_MAP.get(flag, 'OTH')

        features = {
            'service': mapped_service,
            'flag': mapped_flag,
            'src_bytes': src_bytes,
            'dst_bytes': dst_bytes,
            'same_srv_rate': same_srv_rate,
            'diff_srv_rate': diff_srv_rate,
            'dst_host_srv_count': dst_host_srv_count,
            'dst_host_same_srv_rate': same_srv_rate,
            'dst_host_diff_srv_rate': diff_srv_rate,
            'dst_host_srv_serror_rate': dst_host_srv_serror_rate,
        }

        data.append(features)

    return pd.DataFrame(data)

@app.route('/capture', methods=['GET'])
def capture():
    try:
        duration = int(request.args.get('duration', 10))
        df = capture_packet_features(duration)
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
